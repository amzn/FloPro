"""Non-blocking persistence for state store writes.

Provides two components:
- PersistenceWriter: a driver-side background thread that consumes write
  events from a queue and persists them via a PersistenceBackend.
- PersistingStoreWrapper: wraps a StateStore to enqueue write events for
  persistence while delegating all operations to the inner store.

The queue can be a threading.Queue (local) or ray.util.queue.Queue (Ray).
"""

import logging
from queue import Queue
from threading import Thread
from typing import Any, List, Optional

from flo_pro_sdk.core.state import AgentPlan, State
from flo_pro_sdk.core.state_store import DirectRef, StateStore
from flo_pro_sdk.core.persistence_backend import PersistenceBackend

logger = logging.getLogger(__name__)


class PersistenceWriter:
    """Non-blocking persistence writer running on the driver.

    Consumes write events from a queue and persists them via PersistenceBackend.
    Works with both threading.Queue (local) and ray.util.queue.Queue (Ray).

    Queue items may contain direct values or references (e.g. Ray ObjectRefs).
    The ``resolver`` callable is used to dereference values before writing.
    For local execution the default identity resolver is used; for Ray,
    pass ``ray.get`` so that ObjectRefs are resolved on the driver.

    Args:
        backend: The persistence backend to write to.
        queue: Optional queue to consume from. If None, creates a
            threading.Queue internally (suitable for local execution).
        state_store: Optional state store to flush before finalizing.
            For Ray stores, this ensures all pending writes are enqueued
            before draining the queue.
        resolver: Callable to dereference queue values. Defaults to
            identity (``lambda x: x``). Pass ``ray.get`` for Ray.
    """

    def __init__(
        self,
        backend: PersistenceBackend,
        state_store: StateStore,
        queue: Any = None,
    ) -> None:
        self._backend = backend
        self._queue: Any = queue if queue is not None else Queue()
        self._state_store = state_store
        self._thread = Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    @property
    def queue(self) -> Any:
        """The queue that write events should be sent to."""
        return self._queue

    def _writer_loop(self) -> None:
        """Background thread that consumes from queue and writes to backend.

        Handles both threading.Queue and ray.util.queue.Queue.
        For Ray queues, .get() should block properly, but we add explicit
        error handling in case of compatibility issues.
        """
        logger.info("PersistenceWriter thread started")
        while True:
            try:
                # Both threading.Queue and ray.util.queue.Queue support .get(block=True)
                item = self._queue.get(block=True)
                logger.debug("Got item from queue: %s", item[0] if item else "None")
            except Exception as e:
                # If we get an exception reading from the queue, log it and retry
                # This shouldn't happen in normal operation
                logger.exception("Error reading from persistence queue: %s", e)
                import time

                time.sleep(0.1)  # Brief sleep to avoid tight loop on persistent errors
                continue

            if item is None:
                # None is the sentinel value to stop the writer
                logger.info("PersistenceWriter received stop signal")
                break

            try:
                if item[0] == "state":
                    _, iteration, state_ref, timestamp = item
                    self._backend.write_state(
                        iteration, state_ref.resolve(), timestamp=timestamp
                    )
                    logger.debug("Persisted state for iteration %d", iteration)
                elif item[0] == "plan":
                    _, iteration, agent_id, plan_ref = item
                    self._backend.write_agent_plan(
                        iteration, agent_id, plan_ref.resolve()
                    )
                    logger.debug(
                        "Persisted plan for agent %s iteration %d", agent_id, iteration
                    )
            except Exception:
                logger.exception(
                    "Error persisting %s for iteration %s", item[0], item[1]
                )
        logger.info("PersistenceWriter thread exiting")

    def finalize(self, timeout: float = 30.0) -> None:
        """Drain the queue, stop the writer thread, and close the backend.

        For Ray stores, flushes the store first to ensure all pending
        writes are enqueued before draining the queue.

        Args:
            timeout: Maximum time in seconds to wait for the writer thread
                to finish. Defaults to 30 seconds.

        Raises:
            RuntimeError: If the writer thread does not finish within the timeout.
        """
        self._state_store.flush()

        # Send stop signal and wait for writer thread to finish
        self._queue.put(None)
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.error(
                "PersistenceWriter thread did not finish within %s seconds", timeout
            )
            raise RuntimeError(
                f"PersistenceWriter thread did not finish within {timeout} seconds"
            )

        self._backend.close()


class PersistingStoreWrapper(StateStore):
    """Wraps a StateStore to enqueue write events for persistence.

    All read operations delegate directly to the inner store.
    Write operations delegate to the inner store AND enqueue the
    write event for the persistence writer.

    Args:
        inner: The underlying StateStore to delegate to.
        persistence_queue: Queue to send write events to.
    """

    def __init__(self, inner: StateStore, persistence_queue: Any) -> None:
        self._inner = inner
        self._queue = persistence_queue

    def store_state(
        self,
        iteration: int,
        state: State,
        timestamp: float | None = None,
        *,
        blocking: bool = False,
    ) -> None:
        self._inner.store_state(
            iteration, state, timestamp=timestamp, blocking=blocking
        )
        self._queue.put(("state", iteration, DirectRef(state), timestamp))

    def store_agent_plan(self, iteration: int, agent_id: str, plan: AgentPlan) -> None:
        self._inner.store_agent_plan(iteration, agent_id, plan)
        self._queue.put(("plan", iteration, agent_id, DirectRef(plan)))

    def store_agent_plans(self, iteration: int, plans: dict[str, AgentPlan]) -> None:
        self._inner.store_agent_plans(iteration, plans)
        for agent_id, plan in plans.items():
            self._queue.put(("plan", iteration, agent_id, DirectRef(plan)))

    def get_state(self, iteration: int) -> Optional[State]:
        return self._inner.get_state(iteration)

    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional[AgentPlan]:
        return self._inner.get_agent_plan(iteration, agent_id)

    def get_recent_states(self, count: int) -> List[State]:
        return self._inner.get_recent_states(count)

    def flush(self) -> None:
        self._inner.flush()
