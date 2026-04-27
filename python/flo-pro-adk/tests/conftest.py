"""Shared test fixtures for V-ADK unit tests."""

from __future__ import annotations

from flo_pro_sdk.agent.agent_definition import (
    AgentDefinition,
    Objective,
    Solution,
)
from flo_pro_sdk.core.types import JsonValue
from flo_pro_sdk.core.variables import Prices, PublicVarValues, RhoValues

from flo_pro_adk.core.counterparty.counterparty_agent import (
    CounterpartyAgent,
)
from flo_pro_adk.core.solver.solver_strategy import (
    SolverStrategy,
)


class _TestStubSolver(SolverStrategy):
    """A no-op solver so stub counterparty agents can be instantiated without xpress."""

    def create_model(  # type: ignore[override]
        self, consensus, prices, rho, *, public_group_metadata,
        sense=None, var_lb=0.0, var_ub=None,
    ):
        raise NotImplementedError("_TestStubSolver is not meant to be invoked")


class StubAgent(AgentDefinition):
    """Minimal agent that returns consensus as preferred. For testing only."""

    def solve(
        self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues,
    ) -> Solution:
        preferred = {g: v.copy() for g, v in public_vars.items()}
        return Solution(
            preferred_vars=preferred,
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        )


class StubCounterpartyAgent(CounterpartyAgent):
    """Minimal counterparty that returns consensus as preferred. For testing only."""

    @classmethod
    def _default_solver(cls) -> SolverStrategy:
        return _TestStubSolver()

    def __init__(
        self,
        agent_params: dict[str, JsonValue] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(agent_params=agent_params or {})

    def solve(
        self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues,
    ) -> Solution:
        preferred = {g: v.copy() for g, v in public_vars.items()}
        return Solution(
            preferred_vars=preferred,
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        )
