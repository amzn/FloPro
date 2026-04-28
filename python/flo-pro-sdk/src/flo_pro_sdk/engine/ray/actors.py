"""Ray actors for distributed agent and coordinator execution.

These actors wrap AgentRuntime and CoordinatorRuntime for execution on Ray workers.
State persists across calls, enabling warm-start optimizations.
"""

from typing import Any, TYPE_CHECKING

import ray

from flo_pro_sdk.agent.agent_definition import AgentSpec, Solution
from flo_pro_sdk.agent.agent_runtime import AgentRuntime
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.coordinator.coordinator_runtime import CoordinatorRuntime
from flo_pro_sdk.core.engine import QueryExecutor
from flo_pro_sdk.core.query import QueryStrategy
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarsMetadata,
    PublicVarValues,
    RhoValues,
)
from flo_pro_sdk.core.observability import Logger, InMemoryMetrics

if TYPE_CHECKING:
    from flo_pro_sdk.core.state_store import StateStore


@ray.remote
class RayAgentActor:
    """Ray actor wrapping AgentRuntime for distributed execution.

    State persists across calls, enabling warm-start optimizations.
    Logger and metrics are created on the worker to ensure proper serialization.
    """

    def __init__(self, agent_spec: AgentSpec):
        self._logger = Logger(f"agent.{agent_spec.agent_id}")
        self._metrics = InMemoryMetrics(f"agent.{agent_spec.agent_id}")

        agent_def = agent_spec.agent_class.create(agent_spec.agent_params or {})
        self._runtime = AgentRuntime(agent_def, agent_spec, self._logger, self._metrics)

    def ready(self) -> bool:
        """Readiness probe. Ray actor __init__ is async, and Ray will queue other method calls
        until __init__ completes. Invoking this ensures initialization has completed."""
        return True

    def query(
        self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues
    ) -> Solution:
        return self._runtime.query(public_vars, prices, rho)

    def register(self) -> PublicVarsMetadata:
        return self._runtime.register()

    def finalize(self, final_state: Any) -> None:
        self._runtime.finalize(final_state)


@ray.remote
class RayCoordinatorActor:
    """Ray actor wrapping CoordinatorRuntime for distributed execution.

    The state_store parameter provides distributed access to algorithm state.
    In the Ray case this is a RayStateStore that delegates to a shared actor,
    so writes from the coordinator (on a worker) go directly to the cache
    without round-tripping to the driver.
    """

    def __init__(
        self,
        coordinator_spec: CoordinatorSpec,
        query_strategy: QueryStrategy,
        query_executor: QueryExecutor,
        registry: AgentRegistry,
        state_store: "StateStore",
    ):
        self._logger = Logger("coordinator")
        self._metrics = InMemoryMetrics("coordinator")

        coord_def = coordinator_spec.instantiate(registry)
        self._runtime = CoordinatorRuntime(
            coordinator_definition=coord_def,
            coordinator_spec=coordinator_spec,
            query_strategy=query_strategy,
            query_executor=query_executor,
            registry=registry,
            state_store=state_store,
        )

    def ready(self) -> bool:
        return True

    def run_iteration(self, iteration: int):
        return self._runtime.run_iteration(iteration)

    def finalize(self, state) -> None:
        self._runtime.finalize(state)
