"""
Sharing ADMM Coordinator

Problem:  minimize  Σ_j f_j(x_j) + g(Σ_j z_j)
          subject to  x_j = z_j,  ∀j

For maximization problems the dual update sign is negated.

ADMM iterations (after simplification, per-variable N_k):
  Step 1. x-update (agents, parallel):
          x_j = argmin f_j(x_j) + λ^T x_j + (ρ/2)||x_j - z_j||²
  Step 2. Structure function (resource update), returns z_bar (average z):
          z_bar = solve(state)
  Step 3. z-update (per-agent):
          Δ_k = N_k * z_bar_k - x_sum_k
          z_j[k] = x_j[k] + Δ_k / N_k   (subscribed indices only)
  Step 4. Dual update:
          λ = λ - sign * (ρ / N_k) * Δ
          where sign = +1 (minimization) or −1 (maximization)
"""

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
from numpy import ndarray

from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorDefinition
from flo_pro_sdk.coordinator.problem_format import ProblemFormat, price_sign
from flo_pro_sdk.core.state import AgentId, State, SharingState, CoreState
from flo_pro_sdk.core.structure_function import ZeroFunction, StructureFunction
from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import Residuals

if TYPE_CHECKING:
    from flo_pro_sdk.core.state_store import ReadOnlyStore


class SharingCoordinator(CoordinatorDefinition):
    """Coordinator for sharing/exchange problems.

    Uses a single global dual variable λ and per-variable subscription
    counts N_k from VarLayout for the gap split.

    Default structure function: ZeroFunction (exchange: g = I(u=0) → z_bar=0).
    """

    def __init__(
        self,
        layout: VarLayout,
        structure_function: Optional[StructureFunction] = None,
        primal_tol: float = 1e-4,
        dual_tol: float = 1e-4,
        max_iterations: int = 1000,
        problem_format: ProblemFormat = "minimization",
    ) -> None:
        self.layout = layout
        self.structure_function: StructureFunction = structure_function or ZeroFunction(
            layout=layout
        )
        self.primal_tol = primal_tol
        self.dual_tol = dual_tol
        self.max_iterations = max_iterations
        self.problem_format: ProblemFormat = problem_format
        self._price_sign: int = price_sign(problem_format)
        self._N_k: ndarray = layout.get_subscription_counts().astype(float)

    def update_state(
        self,
        agent_results: Dict[AgentId, ndarray],
        current_state: State,
        state_store: Optional["ReadOnlyStore"] = None,
    ) -> State:
        if not isinstance(current_state, SharingState):
            raise TypeError(
                f"SharingCoordinator requires SharingState, got {type(current_state).__name__}"
            )

        aids = list(agent_results.keys())
        current_prices = current_state.get_agent_prices(aids[0])  # global λ
        N_k = self._N_k

        # x_sum = Σ x_j on subscribed indices only
        x_sum = np.zeros_like(current_prices)
        for aid in aids:
            idx = self.layout.get_global_indices(aid)
            x_sum[idx] += agent_results[aid][idx]

        # Step 2: Structure function returns z_bar (average z).
        intermediate = SharingState(
            iteration=current_state.iteration,
            consensus_vars=current_state.consensus_vars,
            agent_preferred_vars=agent_results,
            agent_targets={aid: current_state.get_agent_targets(aid) for aid in aids},
            prices=current_prices,
            rho={aid: current_state.get_rho(aid) for aid in aids},
        )
        new_z_bar: ndarray = self.structure_function.solve(intermediate)

        # Step 3: z_j-update
        delta = N_k * new_z_bar - x_sum
        new_targets = self._update_targets(
            aids,
            agent_results,
            delta,
            N_k,
        )

        # Step 4: Dual update — sign flips for maximization problems.
        sign = self._price_sign
        rho = current_state.get_rho(aids[0])
        new_prices = current_prices - sign * np.divide(
            rho * delta,
            N_k,
            out=np.zeros_like(delta),
            where=N_k > 0,
        )

        residuals = self._compute_residuals(current_state, new_z_bar, delta, rho)

        return SharingState(
            iteration=current_state.iteration + 1,
            consensus_vars=new_z_bar,
            agent_preferred_vars=agent_results,
            agent_targets=new_targets,
            prices=new_prices,
            rho={aid: current_state.get_rho(aid) for aid in aids},
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

    def _update_targets(
        self,
        aids: list,
        agent_results: Dict[AgentId, ndarray],
        delta: ndarray,
        N_k: ndarray,
    ) -> Dict[AgentId, ndarray]:
        """z_j[k] = x_j[k] + Δ_k / N_k on subscribed indices."""
        gap_per_var = np.divide(
            delta,
            N_k,
            out=np.zeros_like(delta),
            where=N_k > 0,
        )
        targets: Dict[AgentId, ndarray] = {}
        for aid in aids:
            idx = self.layout.get_global_indices(aid)
            z_j = agent_results[aid].copy()
            z_j[idx] += gap_per_var[idx]
            targets[aid] = z_j
        return targets

    def _compute_residuals(
        self,
        current_state: SharingState,
        new_z_bar: ndarray,
        delta: ndarray,
        rho: ndarray,
    ) -> Residuals:
        """Primal: ||Δ|| (aggregate constraint violation), Dual: ||ρ (z_bar_new - z_bar_old)||."""
        primal = float(np.linalg.norm(delta))
        z_bar_old = current_state.consensus_vars
        dual = float(np.linalg.norm(rho * (new_z_bar - z_bar_old)))
        return Residuals(primal=primal, dual=dual)
