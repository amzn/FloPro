"""E2E test runner — entry point for running local end-to-end coordination tests.

Orchestrates build_problem, ProblemRunner, and LocalExecutionEngine into
a single call. Returns a structured E2ETestResult for assertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from flo_pro_sdk.agent.agent_definition import AgentDefinition
from flo_pro_sdk.core.lifecycle import ProblemRunner
from flo_pro_sdk.core.state import State
from flo_pro_sdk.core.types import JsonValue
from flo_pro_sdk.engine.local.engine import LocalExecutionEngine

from flo_pro_adk.core.assembly.problem_assembler import (
    build_problem,
)
from flo_pro_adk.core.counterparty.counterparty_agent import (
    _data_loader_factories,
    _var_metadata_registry,
)
from flo_pro_adk.core.types.scenario_params import (
    ScenarioParams,
)

if TYPE_CHECKING:
    from flo_pro_adk.core.counterparty.counterparty_agent import (
        CounterpartyAgent,
    )
    from flo_pro_adk.core.testing.simulation_data_generator import (
        SimulationDataGenerator,
    )


@dataclass(frozen=True)
class E2ETestResult:
    """Structured output from an E2E coordination test run."""

    final_state: State
    n_iterations: int
    converged: bool

    @property
    def final_residuals(self) -> tuple[float, float] | None:
        """(primal, dual) residuals from the final state, or None."""
        r = self.final_state.residuals
        if r is None:
            return None
        return (r.primal, r.dual)


def run_e2e_test(
    agent_class: type[AgentDefinition],
    counterparty_class: type[CounterpartyAgent],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
    max_iterations: int = 1000,
    convergence_primal_tol: float = 1e-4,
    convergence_dual_tol: float = 1e-4,
) -> E2ETestResult:
    """Run a full E2E coordination test and return structured results.

    Args:
        agent_class: The user's AgentDefinition subclass.
        counterparty_class: The counterparty (RetailerAgent or VendorAgent).
        data_generator_class: Domain-specific SimulationDataGenerator subclass.
        scenario: ScenarioParams configuring the test.
        agent_params: Optional JSON-serializable config for the agent.
        max_iterations: Maximum ADMM iterations before stopping.
        convergence_primal_tol: Primal residual tolerance for convergence.
        convergence_dual_tol: Dual residual tolerance for convergence.
    """
    try:
        problem = build_problem(
            agent_class,
            counterparty_class,
            data_generator_class(scenario),
            max_iterations=max_iterations,
            agent_params=agent_params,
        )
        engine = LocalExecutionEngine()
        final_state = ProblemRunner(problem, engine).run()
    finally:
        for cls in (counterparty_class, agent_class):
            _data_loader_factories.pop(cls, None)  # type: ignore[arg-type]
            _var_metadata_registry.pop(cls, None)  # type: ignore[arg-type]

    n_iterations = final_state.iteration
    residuals = final_state.residuals
    converged = (
        residuals is not None
        and residuals.primal <= convergence_primal_tol
        and residuals.dual <= convergence_dual_tol
    )

    return E2ETestResult(
        final_state=final_state,
        n_iterations=n_iterations,
        converged=converged,
    )
