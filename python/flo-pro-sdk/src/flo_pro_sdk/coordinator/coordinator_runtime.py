import time
from typing import Dict, TYPE_CHECKING

from numpy import ndarray

from flo_pro_sdk.coordinator.coordinator_definition import (
    CoordinatorDefinition,
    CoordinatorSpec,
)
from flo_pro_sdk.core.state import State, AgentId, AgentPlan, IterationResult
from flo_pro_sdk.core.engine import QueryExecutor
from flo_pro_sdk.core.query import QueryResult, QueryStrategy
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.var_layout import VarLayout

if TYPE_CHECKING:
    from flo_pro_sdk.core.state_store import StateStore


class CoordinatorRuntime:
    def __init__(
        self,
        coordinator_definition: CoordinatorDefinition,
        coordinator_spec: CoordinatorSpec,
        query_strategy: QueryStrategy,
        query_executor: QueryExecutor,
        registry: AgentRegistry,
        state_store: "StateStore",
    ) -> None:
        self.coordinator_definition = coordinator_definition
        self.coordinator_spec = coordinator_spec
        self.query_strategy = query_strategy
        self.query_executor = query_executor
        self.registry = registry
        self.state_store = state_store
        self._layout: VarLayout = registry.get_layout()

    def run_iteration(self, iteration: int) -> IterationResult:
        current_state = self.state_store.get_state(iteration)
        if current_state is None:
            raise ValueError(f"No state found for iteration {iteration}")

        query_results: Dict[str, "QueryResult"] = self.query_strategy.query(
            state=current_state,
            registry=self.registry,
            executor=self.query_executor,
        )

        agent_plans: Dict[str, AgentPlan] = {}
        flat_results: Dict[AgentId, ndarray] = {}
        for agent_id, result in query_results.items():
            plan = AgentPlan(
                agent_id=agent_id, iteration=iteration, solution=result.solution
            )
            agent_plans[agent_id] = plan
            flat_results[agent_id] = self._layout.flatten_to_global(
                agent_id, result.solution.preferred_vars
            )

        self.state_store.store_agent_plans(iteration, agent_plans)

        new_state = self.coordinator_definition.update_state(
            flat_results,
            current_state,
            state_store=self.state_store,
        )

        self.state_store.store_state(
            new_state.iteration, new_state, timestamp=time.time()
        )
        converged = self.coordinator_definition.check_convergence(
            new_state.get_core_state()
        )

        return IterationResult(iteration=new_state.iteration, converged=converged)

    def finalize(self, state: State) -> None:
        self.coordinator_definition.finalize(state)
