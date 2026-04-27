"""Ray execution engine for distributed CPP optimization."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
import ray.util.queue
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.compute import ComputeSpec
from flo_pro_sdk.core.engine import (
    ExecutionEngine,
    QueryExecutor,
    RegistrationExecutor,
    FinalizationExecutor,
)
from flo_pro_sdk.core.handlers import CoordinatorHandler
from flo_pro_sdk.core.persistence import PersistenceWriter
from flo_pro_sdk.core.query import QueryStrategy
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.state_store import StoreConfig, StateStore
from flo_pro_sdk.engine.ray.actors import RayAgentActor, RayCoordinatorActor
from flo_pro_sdk.engine.ray.executors import (
    RayQueryExecutor,
    RayRegistrationExecutor,
    RayFinalizationExecutor,
)
from flo_pro_sdk.engine.ray.options import RayEngineOptions, RayStateStoreType
from flo_pro_sdk.engine.ray.state_store import RayStateStore, RayRefStateStore

logger = logging.getLogger("cpp.ray.engine")


class RayExecutionEngine(ExecutionEngine):
    """Ray-based execution engine for distributed CPP optimization.

    Supports optional state tracking via StoreConfig, using a RayStateStore
    (Ray actor-based) for distributed access to recent iterations.

    The StateStore actor optionally enqueues write events to a Ray queue.
    A driver-side PersistenceWriter thread consumes the queue and writes
    to the PersistenceBackend on the driver's filesystem.

    The StateStore and Coordinator actors are always co-located on the same
    node using a STRICT_PACK placement group. Bundle resources are derived
    from CoordinatorSpec.compute (for the coordinator bundle) and
    StoreConfig.state_store_compute (for the state store bundle). Both
    default to 1 CPU with no memory constraint if not specified.

    Example:
        # Without tracking (backward compatible)
        engine = RayExecutionEngine()

        # With distributed caching only
        config = StoreConfig(cache_size=10)
        engine = RayExecutionEngine(store_config=config)

        # With filesystem persistence on driver
        backend = FileSystemBackend(base_dir="/tmp/runs")
        config = StoreConfig(persistence_backend=backend, cache_size=5)
        engine = RayExecutionEngine(store_config=config)

        # With custom compute sizing for coordinator and state store
        config = StoreConfig(
            cache_size=10,
            state_store_compute=ComputeSpec(num_cpus=1, memory_mb=4096),
        )
        coord_spec = CoordinatorSpec(
            ...,
            compute=ComputeSpec(num_cpus=4, memory_mb=8192),
        )
        engine = RayExecutionEngine(store_config=config)
    """

    def __init__(
        self,
        options: RayEngineOptions = RayEngineOptions(),
        store_config: Optional[StoreConfig] = None,
    ) -> None:
        if not ray.is_initialized():
            ray.init(**options.ray_init_kwargs())
        self._store_config = store_config or StoreConfig()
        self._agent_actors: Dict[str, ray.actor.ActorHandle] = {}
        self._coordinator_actor: Optional[ray.actor.ActorHandle] = None
        self._options = options
        self._state_store: Optional[StateStore] = None
        self._persistence: Optional[PersistenceWriter] = None

    def allocate_agents(self, agent_specs: List[AgentSpec]) -> None:
        """Allocate Ray actors for the agents."""
        for agent_spec in agent_specs:
            options = self._compute_spec_to_ray_options(agent_spec.compute)
            actor = RayAgentActor.options(name=agent_spec.agent_id, **options).remote(
                agent_spec
            )  # type: ignore[attr-defined]
            self._agent_actors[agent_spec.agent_id] = actor

        # Wait for all actors to finish initialization
        ready_futures = {
            actor.ready.remote(): agent_id
            for agent_id, actor in self._agent_actors.items()
        }
        failures: Dict[str, Exception] = {}
        pending = list(ready_futures.keys())

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for future in ready:
                agent_id = ready_futures[future]
                try:
                    ray.get(future)
                except Exception as e:
                    logger.error(f"Agent '{agent_id}' failed to initialize: {e}")
                    failures[agent_id] = e

        if failures:
            failed_ids = list(failures.keys())
            raise RuntimeError(
                f"Failed to initialize {len(failures)} agent(s): {failed_ids}"
            )

    def allocate_coordinator(
        self,
        coordinator_spec: CoordinatorSpec,
        query_strategy: QueryStrategy,
        query_executor: QueryExecutor,
        registry: AgentRegistry,
    ) -> CoordinatorHandler:
        """Allocate a Ray actor for the coordinator.

        Creates a STRICT_PACK placement group to co-locate the StateStore
        and Coordinator actors on the same node, then initializes both.
        Bundle resources are derived from CoordinatorSpec.compute and
        StoreConfig.state_store_compute respectively.
        """
        # Build bundle resource specs from compute configurations.
        # The hard requirement is STRICT_PACK co-location; the resource values
        # for each bundle reflect the actor's own needs independently.
        store_bundle = self._compute_spec_to_bundle(
            self._store_config.state_store_compute
        )
        coord_bundle = self._compute_spec_to_bundle(coordinator_spec.compute)

        pg = placement_group(
            [store_bundle, coord_bundle],
            strategy="STRICT_PACK",
        )
        ray.get(pg.ready())

        state_store_strategy = PlacementGroupSchedulingStrategy(
            pg, placement_group_bundle_index=0
        )
        coord_strategy = PlacementGroupSchedulingStrategy(
            pg, placement_group_bundle_index=1
        )

        self._state_store, self._persistence = self._build_store(
            self._store_config, scheduling_strategy=state_store_strategy
        )

        coord_options = self._compute_spec_to_ray_options(coordinator_spec.compute)
        coord_options["scheduling_strategy"] = coord_strategy

        actor = RayCoordinatorActor.options(**coord_options).remote(  # type: ignore[attr-defined]
            coordinator_spec,
            query_strategy,
            query_executor,
            registry,
            self._state_store,
        )
        self._coordinator_actor = actor

        try:
            ray.get(actor.ready.remote())
        except Exception as e:
            logger.error(f"Coordinator failed to initialize: {e}")
            raise RuntimeError(f"Failed to initialize coordinator: {e}") from e

        return CoordinatorHandler(
            run_iteration=lambda iteration: ray.get(
                actor.run_iteration.remote(iteration)
            ),
            finalize=lambda s: ray.get(actor.finalize.remote(s)),
        )

    def get_query_executor(self) -> QueryExecutor:
        return RayQueryExecutor(self._agent_actors)

    def get_registration_executor(self) -> RegistrationExecutor:
        return RayRegistrationExecutor(self._agent_actors)

    def get_finalization_executor(self) -> FinalizationExecutor:
        return RayFinalizationExecutor(self._agent_actors)

    def shutdown(self) -> None:
        if self._persistence is not None:
            self._persistence.finalize()
        ray.shutdown()

    def get_state_store(self) -> StateStore:
        if self._state_store is None:
            raise RuntimeError(
                "State store not yet initialized. Call allocate_coordinator() first."
            )
        return self._state_store

    def _compute_spec_to_ray_options(
        self, compute: Optional[ComputeSpec]
    ) -> Dict[str, Any]:
        if compute is None:
            return {}
        options: Dict[str, Any] = {}
        if compute.num_cpus is not None:
            options["num_cpus"] = compute.num_cpus
        if compute.num_gpus is not None:
            options["num_gpus"] = compute.num_gpus
        if compute.memory_mb is not None:
            options["memory"] = compute.memory_mb * 1024 * 1024
        return options

    def _compute_spec_to_bundle(
        self, compute: Optional[ComputeSpec]
    ) -> Dict[str, float]:
        """Convert a ComputeSpec to a placement group bundle resource dict.

        Defaults to 1 CPU if num_cpus is not specified. Memory is included
        only when explicitly set, to avoid over-constraining placement.
        """
        bundle: Dict[str, float] = {
            "CPU": compute.num_cpus if compute and compute.num_cpus is not None else 1
        }
        if compute and compute.memory_mb is not None:
            bundle["memory"] = compute.memory_mb * 1024 * 1024
        return bundle

    def _build_store(
        self, config: StoreConfig, scheduling_strategy: Any = None
    ) -> Tuple[StateStore, Optional[PersistenceWriter]]:
        persistence: Optional[PersistenceWriter] = None
        StoreClass = (
            RayRefStateStore
            if self._options.state_store_type == RayStateStoreType.OBJECT_STORE
            else RayStateStore
        )

        if config.persistence_backend is not None:
            persistence_queue = ray.util.queue.Queue()
            store = StoreClass(
                config.cache_size,
                persistence_queue,
                scheduling_strategy=scheduling_strategy,
            )
            persistence = PersistenceWriter(
                config.persistence_backend,
                store,
                queue=persistence_queue,
            )
        else:
            store = StoreClass(
                config.cache_size, scheduling_strategy=scheduling_strategy
            )

        return store, persistence
