"""Local execution engine implementation."""

from typing import Dict, List, Optional
from pathlib import Path

from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.agent.agent_runtime import AgentRuntime
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.coordinator.coordinator_runtime import CoordinatorRuntime
from flo_pro_sdk.core.engine import (
    ExecutionEngine,
    QueryExecutor,
    RegistrationExecutor,
    FinalizationExecutor,
)
from flo_pro_sdk.core.handlers import CoordinatorHandler
from flo_pro_sdk.core.in_memory_state_store import InMemoryStateStore
from flo_pro_sdk.core.persistence import PersistenceWriter, PersistingStoreWrapper
from flo_pro_sdk.core.query import QueryStrategy
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.state_store import StoreConfig, StateStore
from flo_pro_sdk.engine.local.handles import LocalAgentHandles
from flo_pro_sdk.engine.local.executors import (
    LocalQueryExecutor,
    LocalRegistrationExecutor,
    LocalFinalizationExecutor,
)


class LocalExecutionEngine(ExecutionEngine):
    """Local execution engine that runs everything in-process.

    Supports optional state tracking via StoreConfig. When configured, the
    engine creates an InMemoryStateStore for fast access to recent iterations
    and optionally persists state via a non-blocking background writer.

    Example:
        # Without tracking (backward compatible)
        engine = LocalExecutionEngine()

        # With in-memory tracking only
        config = StoreConfig(cache_size=10)
        engine = LocalExecutionEngine(store_config=config)

        # With filesystem persistence
        backend = FileSystemBackend(base_dir="/tmp/runs")
        config = StoreConfig(persistence_backend=backend, cache_size=5)
        engine = LocalExecutionEngine(store_config=config)
    """

    def __init__(self, store_config: Optional[StoreConfig] = None) -> None:
        self._agent_handles: Dict[str, LocalAgentHandles] = {}
        self._store_config = store_config or StoreConfig()
        self._state_store, self._persistence = self._build_store(self._store_config)

    def allocate_agents(self, agent_specs: List[AgentSpec]) -> None:
        """Allocate agents in the local process."""
        for agent_spec in agent_specs:
            agent_definition = agent_spec.agent_class.create(
                agent_spec.agent_params or {}
            )
            agent_runtime = AgentRuntime(
                agent_definition=agent_definition,
                agent_spec=agent_spec,
            )
            handles = LocalAgentHandles(
                query=agent_runtime.query,
                register=agent_runtime.register,
                finalize=agent_runtime.finalize,
            )
            self._agent_handles[agent_spec.agent_id] = handles

    def allocate_coordinator(
        self,
        coordinator_spec: CoordinatorSpec,
        query_strategy: QueryStrategy,
        query_executor: QueryExecutor,
        registry: AgentRegistry,
    ) -> CoordinatorHandler:
        """Allocate a coordinator in the local process."""
        coordinator_definition = coordinator_spec.instantiate(registry)
        coordinator_runtime = CoordinatorRuntime(
            coordinator_definition=coordinator_definition,
            coordinator_spec=coordinator_spec,
            query_strategy=query_strategy,
            query_executor=query_executor,
            registry=registry,
            state_store=self._state_store,
        )
        return CoordinatorHandler(
            run_iteration=coordinator_runtime.run_iteration,
            finalize=coordinator_runtime.finalize,
        )

    def get_query_executor(self) -> QueryExecutor:
        return LocalQueryExecutor(self._agent_handles)

    def get_registration_executor(self) -> RegistrationExecutor:
        return LocalRegistrationExecutor(self._agent_handles)

    def get_finalization_executor(self) -> FinalizationExecutor:
        return LocalFinalizationExecutor(self._agent_handles)

    def shutdown(self) -> None:
        if self._persistence is not None:
            self._persistence.finalize()
        self._agent_handles.clear()

    def get_state_store(self) -> StateStore:
        return self._state_store

    def get_run_dir(self) -> Optional[Path]:
        from flo_pro_sdk.core.persistence_backend import FileSystemBackend

        backend = self._store_config.persistence_backend
        if isinstance(backend, FileSystemBackend):
            return backend.run_dir
        return None

    def _build_store(self, config: StoreConfig) -> tuple:
        store: StateStore = InMemoryStateStore(config.cache_size)
        persistence: Optional[PersistenceWriter] = None

        if config.persistence_backend is not None:
            persistence = PersistenceWriter(config.persistence_backend, store)
            store = PersistingStoreWrapper(store, persistence.queue)

        return store, persistence
