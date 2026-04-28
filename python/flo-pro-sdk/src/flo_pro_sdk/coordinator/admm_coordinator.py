"""
ADMM (Alternating Direction Method of Multipliers) Coordinator Implementation.

Operates on flat arrays. The runtime (CoordinatorRuntime) handles
translation between grouped agent solutions and flat arrays via VarLayout.

Supports both minimization and maximization problem formats. For
maximization the dual (price) update sign is negated per Boyd §3.3.
"""

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
from numpy import ndarray

from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorDefinition
from flo_pro_sdk.coordinator.problem_format import ProblemFormat, price_sign
from flo_pro_sdk.core.state import AgentId, State, ConsensusState, CoreState
from flo_pro_sdk.core.structure_function import AveragingFunction, StructureFunction
from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import Residuals

if TYPE_CHECKING:
    from flo_pro_sdk.core.state_store import ReadOnlyStore


class ADMMCoordinator(CoordinatorDefinition):
    """Standard ADMM coordinator for consensus-based coordination.

    Operates on flat arrays. Delegates z-update to a pluggable StructureFunction
    (default: AveragingFunction = simple averaging).

    The ADMM algorithm alternates between:
    1. Agent optimization (solve subproblems) — done by agents
    2. Consensus variable update via StructureFunction
    3. Dual variable (price) update
    4. Optional adaptive rho adjustment
    """

    def __init__(
        self,
        layout: VarLayout,
        structure_function: Optional[StructureFunction] = None,
        rho_adaptive: bool = True,
        rho_scale_factor: float = 2.0,
        primal_tol: float = 1e-4,
        dual_tol: float = 1e-4,
        max_iterations: int = 1000,
        problem_format: ProblemFormat = "minimization",
    ) -> None:
        self.layout = layout
        self.structure_function: StructureFunction = (
            structure_function or AveragingFunction(layout=layout)
        )
        self.structure_function.layout = self.layout
        self.rho_adaptive = rho_adaptive
        self.rho_scale_factor = rho_scale_factor
        self.primal_tol = primal_tol
        self.dual_tol = dual_tol
        self.max_iterations = max_iterations
        self.problem_format: ProblemFormat = problem_format
        self._price_sign: int = price_sign(problem_format)

    def update_state(
        self,
        agent_results: Dict[AgentId, ndarray],
        current_state: State,
        state_store: Optional["ReadOnlyStore"] = None,
    ) -> State:
        # Build intermediate with new x_i so structure function sees updated values
        intermediate: ConsensusState = ConsensusState(
            iteration=current_state.iteration,
            consensus_vars=current_state.consensus_vars,
            agent_preferred_vars=agent_results,
            prices={aid: current_state.get_agent_prices(aid) for aid in agent_results},
            rho={aid: current_state.get_rho(aid) for aid in agent_results},
        )
        new_z: ndarray = self.structure_function.solve(intermediate)

        # Price update: y_i = y_i + sign * rho * (x_i - z), only on subscribed indices.
        # sign = +1 for minimization (standard ADMM), −1 for maximization.
        sign = self._price_sign
        new_prices: Dict[AgentId, ndarray] = {}
        for aid, x_i in agent_results.items():
            idx = self.layout.get_global_indices(aid)
            p = current_state.get_agent_prices(aid).copy()
            p[idx] += sign * current_state.get_rho(aid)[idx] * (x_i[idx] - new_z[idx])
            new_prices[aid] = p

        residuals: Residuals = self._compute_residuals(
            current_state, new_z, agent_results
        )

        new_rho: Dict[AgentId, ndarray] = {
            aid: current_state.get_rho(aid).copy() for aid in agent_results
        }
        if self.rho_adaptive:
            new_rho = self._adapt_rho(residuals, new_rho)

        return ConsensusState(
            iteration=current_state.iteration + 1,
            consensus_vars=new_z,
            agent_preferred_vars=agent_results,
            prices=new_prices,
            rho=new_rho,
            residuals=residuals,
            metadata=current_state.metadata,
        )

    def check_convergence(self, core_state: CoreState) -> bool:
        if core_state.iteration >= self.max_iterations:
            return True
        if core_state.residuals is None:
            return False
        return (
            core_state.residuals.primal < self.primal_tol
            and core_state.residuals.dual < self.dual_tol
        )

    def finalize(self, final_state: State) -> None:
        pass

    def _compute_residuals(
        self,
        current_state: State,
        new_z: ndarray,
        agent_results: Dict[AgentId, ndarray],
    ) -> Residuals:
        # Scalar residual norms across all agents (subscribed indices only)
        z_diff = new_z - current_state.consensus_vars
        primal_sq = 0.0
        dual_sq = 0.0
        for aid, x_i in agent_results.items():
            idx = self.layout.get_global_indices(aid)
            primal_sq += float(np.sum((x_i[idx] - new_z[idx]) ** 2))
            dual_sq += float(
                np.sum((current_state.get_rho(aid)[idx] * z_diff[idx]) ** 2)
            )
        return Residuals(primal=float(np.sqrt(primal_sq)), dual=float(np.sqrt(dual_sq)))

    def _adapt_rho(
        self,
        residuals: Residuals,
        current_rho: Dict[AgentId, ndarray],
    ) -> Dict[AgentId, ndarray]:
        """Adapt rho uniformly across all agents based on global residual norms.

        The flat-array coordinator is group-unaware, so adaptation is global
        rather than per-group. Per-group adaptation would require layout
        awareness, deferred to a StructureFunction-based approach if needed.

        # TODO: Allow users to plug in their own adaptive rho strategy.
        """
        primal = residuals.primal
        dual = residuals.dual
        factor: float
        if primal > 10 * dual:
            factor = self.rho_scale_factor
        elif dual > 10 * primal:
            factor = 1.0 / self.rho_scale_factor
        else:
            factor = 1.0
        return {aid: rho * factor for aid, rho in current_rho.items()}
