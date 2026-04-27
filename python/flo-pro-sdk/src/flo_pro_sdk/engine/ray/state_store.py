"""Ray-distributed state store implementations.

Provides two StateStore implementations backed by Ray actors:

- RayStateStore: Simple direct-serialization approach. State objects are
  serialized through the actor mailbox. Suitable for small states.

- RayRefStateStore: Object-store-backed approach. State objects are placed
  in Ray's object store via ray.put() and only lightweight RayRef wrappers
  pass through the actor. Avoids serializing large states through the actor
  mailbox. Suitable for large states (100+ agents, 100k+ variables).

Both implementations support fire-and-forget writes, ray.get() reads,
and optional persistence queue integration.
"""

import logging
from typing import Any, Dict, List, Optional

import ray

from flo_pro_sdk.core.in_memory_state_store import InMemoryStateStore
from flo_pro_sdk.core.state import AgentPlan, State
from flo_pro_sdk.core.state_store import Ref, StateStore

logger = logging.getLogger(__name__)

T = Any  # ray.ObjectRef is untyped


class RayRef(Ref[T]):
    """A Ref backed by a Ray ObjectRef.

    Wrapping in a Ref subclass prevents Ray from auto-resolving the
    ObjectRef when passed as an argument to .remote().
    """

    def __init__(self, ref: ray.ObjectRef) -> None:
        self._ref = ref

    def resolve(self) -> T:
        return ray.get(self._ref)

    @property
    def ref(self) -> ray.ObjectRef:
        return self._ref


# ── Direct-serialization implementation ─────────────────────────────


@ray.remote
class _DirectActor:
    """Actor that stores full state objects directly (simple approach)."""

    def __init__(self, cache_size: int = 5, persistence_queue: Any = None) -> None:
        self._store = InMemoryStateStore(cache_size)
        self._persistence_queue = persistence_queue

    def store_state(
        self, iteration: int, state: State, timestamp: float | None = None
    ) -> None:
        self._store.store_state(iteration, state, timestamp=timestamp)
        if self._persistence_queue is not None:
            self._persistence_queue.put(
                ("state", iteration, RayRef(ray.put(state)), timestamp)
            )

    def store_agent_plan(self, iteration: int, agent_id: str, plan: AgentPlan) -> None:
        self._store.store_agent_plan(iteration, agent_id, plan)
        if self._persistence_queue is not None:
            self._persistence_queue.put(
                ("plan", iteration, agent_id, RayRef(ray.put(plan)))
            )

    def store_agent_plans(self, iteration: int, plans: Dict[str, AgentPlan]) -> None:
        self._store.store_agent_plans(iteration, plans)
        if self._persistence_queue is not None:
            for agent_id, plan in plans.items():
                self._persistence_queue.put(
                    ("plan", iteration, agent_id, RayRef(ray.put(plan)))
                )

    def get_state(self, iteration: int) -> Optional[State]:
        return self._store.get_state(iteration)

    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional[AgentPlan]:
        return self._store.get_agent_plan(iteration, agent_id)

    def get_recent_states(self, count: int) -> List[State]:
        return self._store.get_recent_states(count)

    def flush(self) -> None:
        pass


class RayStateStore(StateStore):
    """StateStore that serializes full state objects through the actor.

    Simple approach suitable for small states. State objects are serialized
    into the actor on write and deserialized on read.

    Args:
        cache_size: Maximum number of iterations to retain.
        persistence_queue: Optional ray.util.queue.Queue for persistence.
        scheduling_strategy: Optional Ray scheduling strategy (e.g.,
            PlacementGroupSchedulingStrategy) to co-locate the actor.
    """

    def __init__(
        self,
        cache_size: int = 5,
        persistence_queue: Any = None,
        scheduling_strategy: Any = None,
    ) -> None:
        if scheduling_strategy is not None:
            self._actor = _DirectActor.options(  # type: ignore[attr-defined]
                scheduling_strategy=scheduling_strategy
            ).remote(cache_size, persistence_queue)
        else:
            self._actor = _DirectActor.remote(cache_size, persistence_queue)  # type: ignore[attr-defined]

    def store_state(
        self,
        iteration: int,
        state: State,
        timestamp: float | None = None,
        *,
        blocking: bool = False,
    ) -> None:
        ref = self._actor.store_state.remote(iteration, state, timestamp=timestamp)
        if blocking:
            ray.get(ref)

    def store_agent_plan(self, iteration: int, agent_id: str, plan: AgentPlan) -> None:
        self._actor.store_agent_plan.remote(iteration, agent_id, plan)

    def store_agent_plans(self, iteration: int, plans: Dict[str, AgentPlan]) -> None:
        self._actor.store_agent_plans.remote(iteration, plans)

    def get_state(self, iteration: int) -> Optional[State]:
        return ray.get(self._actor.get_state.remote(iteration))

    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional[AgentPlan]:
        return ray.get(self._actor.get_agent_plan.remote(iteration, agent_id))

    def get_recent_states(self, count: int) -> List[State]:
        return ray.get(self._actor.get_recent_states.remote(count))

    def flush(self) -> None:
        ray.get(self._actor.flush.remote())


# ── Object-store-backed implementation ──────────────────────────────


@ray.remote
class _RefActor:
    """Actor that stores lightweight RayRef wrappers (scalable approach).

    Both state and agent plans are stored as RayRefs so that only
    lightweight wrappers pass through the actor mailbox.

    Eviction is driven by ``store_state`` calls: when the cache exceeds
    ``cache_size``, the oldest iteration's state and plans are evicted
    together. Plans stored for iterations that never receive a
    ``store_state`` call are not evicted automatically.
    """

    def __init__(self, cache_size: int = 5, persistence_queue: Any = None) -> None:
        self._cache_size = cache_size
        self._persistence_queue = persistence_queue
        self._state_refs: Dict[int, RayRef] = {}
        self._plan_refs: Dict[int, Dict[str, RayRef]] = {}
        self._iteration_order: List[int] = []
        if cache_size > 10:
            logger.warning(
                "Large cache_size=%d may increase object store memory pressure",
                cache_size,
            )

    def store_state(
        self, iteration: int, state_ref: RayRef, timestamp: float | None = None
    ) -> None:
        if iteration not in self._state_refs:
            self._iteration_order.append(iteration)
        self._state_refs[iteration] = state_ref
        self._evict_if_needed()
        if self._persistence_queue is not None:
            self._persistence_queue.put(("state", iteration, state_ref, timestamp))

    def get_state(self, iteration: int) -> Optional[RayRef]:
        return self._state_refs.get(iteration)

    def get_recent_states(self, count: int) -> List[RayRef]:
        recent_iters = self._iteration_order[-count:]
        return [
            self._state_refs[i] for i in reversed(recent_iters) if i in self._state_refs
        ]

    def store_agent_plan(self, iteration: int, agent_id: str, plan_ref: RayRef) -> None:
        if iteration not in self._plan_refs:
            self._plan_refs[iteration] = {}
        self._plan_refs[iteration][agent_id] = plan_ref
        if self._persistence_queue is not None:
            self._persistence_queue.put(("plan", iteration, agent_id, plan_ref))

    def store_agent_plans(self, iteration: int, plan_refs: Dict[str, RayRef]) -> None:
        if iteration not in self._plan_refs:
            self._plan_refs[iteration] = {}
        self._plan_refs[iteration].update(plan_refs)
        if self._persistence_queue is not None:
            for agent_id, plan_ref in plan_refs.items():
                self._persistence_queue.put(("plan", iteration, agent_id, plan_ref))

    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional[RayRef]:
        return self._plan_refs.get(iteration, {}).get(agent_id)

    def flush(self) -> None:
        pass

    def _evict_if_needed(self) -> None:
        while len(self._iteration_order) > self._cache_size:
            oldest = self._iteration_order.pop(0)
            self._state_refs.pop(oldest, None)
            self._plan_refs.pop(oldest, None)


class RayRefStateStore(StateStore):
    """StateStore that uses Ray's object store for large state objects.

    State objects are placed in the object store via ray.put() and wrapped
    in RayRef. The actor stores only RayRef wrappers (~bytes each). On read,
    refs are resolved from the object store (zero-copy on same node).

    Use this for large states (100+ agents, 100k+ variables) where
    serialization through the actor mailbox is a bottleneck.

    Args:
        cache_size: Maximum number of iterations to retain.
        persistence_queue: Optional ray.util.queue.Queue for persistence.
        scheduling_strategy: Optional Ray scheduling strategy (e.g.,
            PlacementGroupSchedulingStrategy) to co-locate the actor.
    """

    def __init__(
        self,
        cache_size: int = 5,
        persistence_queue: Any = None,
        scheduling_strategy: Any = None,
    ) -> None:
        if scheduling_strategy is not None:
            self._actor = _RefActor.options(  # type: ignore[attr-defined]
                scheduling_strategy=scheduling_strategy
            ).remote(cache_size, persistence_queue)
        else:
            self._actor = _RefActor.remote(cache_size, persistence_queue)  # type: ignore[attr-defined]

    def store_state(
        self,
        iteration: int,
        state: State,
        timestamp: float | None = None,
        *,
        blocking: bool = False,
    ) -> None:
        ref = self._actor.store_state.remote(
            iteration, RayRef(ray.put(state)), timestamp=timestamp
        )
        if blocking:
            ray.get(ref)

    def store_agent_plan(self, iteration: int, agent_id: str, plan: AgentPlan) -> None:
        self._actor.store_agent_plan.remote(iteration, agent_id, RayRef(ray.put(plan)))

    def store_agent_plans(self, iteration: int, plans: Dict[str, AgentPlan]) -> None:
        plan_refs = {aid: RayRef(ray.put(p)) for aid, p in plans.items()}
        self._actor.store_agent_plans.remote(iteration, plan_refs)

    def get_state(self, iteration: int) -> Optional[State]:
        wrapper = ray.get(self._actor.get_state.remote(iteration))
        if wrapper is None:
            return None
        return wrapper.resolve()

    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional[AgentPlan]:
        wrapper = ray.get(self._actor.get_agent_plan.remote(iteration, agent_id))
        if wrapper is None:
            return None
        return wrapper.resolve()

    def get_recent_states(self, count: int) -> List[State]:
        wrappers = ray.get(self._actor.get_recent_states.remote(count))
        return [w.resolve() for w in wrappers]

    def flush(self) -> None:
        ray.get(self._actor.flush.remote())
