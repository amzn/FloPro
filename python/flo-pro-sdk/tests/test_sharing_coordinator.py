"""Tests for SharingCoordinator and SharingState."""

import numpy as np
import pytest

from flo_pro_sdk.coordinator.sharing_coordinator import SharingCoordinator
from flo_pro_sdk.core.state import SharingState, State
from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import PublicVarGroupName, Residuals

G = PublicVarGroupName("flat")


def _layout(n: int = 2, dim: int = 3) -> VarLayout:
    layout = VarLayout(group_slices={G: slice(0, dim)}, total_size=dim)
    for i in range(n):
        layout.register_agent(f"a{i}", {G: np.arange(dim)})
    return layout


def _sharing_state(n: int = 2, dim: int = 3) -> SharingState:
    return SharingState(
        iteration=0,
        consensus_vars=np.zeros(dim),
        agent_preferred_vars={f"a{i}": np.zeros(dim) for i in range(n)},
        agent_targets={f"a{i}": np.zeros(dim) for i in range(n)},
        prices=np.zeros(dim),
        rho={f"a{i}": np.ones(dim) for i in range(n)},
    )


class TestSharingUpdateState:
    def test_exchange_z_bar_is_zero(self) -> None:
        coord = SharingCoordinator(layout=_layout())
        results = {"a0": np.array([2.0, 4.0, 6.0]), "a1": np.array([4.0, 6.0, 8.0])}
        new = coord.update_state(results, _sharing_state())
        np.testing.assert_array_equal(new.consensus_vars, [0.0, 0.0, 0.0])

    def test_exchange_update(self) -> None:
        """Full update with ZeroFunction: z_bar=0, Δ=-x_sum, z_j=x_j-x_sum/N, λ=-ρΔ/N."""
        coord = SharingCoordinator(layout=_layout(n=2, dim=2))
        results = {"a0": np.array([2.0, 0.0]), "a1": np.array([4.0, 0.0])}
        new = coord.update_state(results, _sharing_state(dim=2))

        assert new.iteration == 1
        # x_sum=[6,0], Δ=[0,0]-[6,0]=[-6,0], gap=[-3,0]
        np.testing.assert_array_almost_equal(new.get_agent_targets("a0"), [-1.0, 0.0])
        np.testing.assert_array_almost_equal(new.get_agent_targets("a1"), [1.0, 0.0])
        # λ = 0 - 1*[-6,0]/2 = [3,0]
        np.testing.assert_array_almost_equal(new.get_agent_prices("a0"), [3.0, 0.0])
        # primal = ||Δ|| = 6
        assert new.residuals is not None
        np.testing.assert_almost_equal(new.residuals.primal, 6.0)

    def test_metadata_preserved(self) -> None:
        coord = SharingCoordinator(layout=_layout(n=1, dim=2))
        state = SharingState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2)},
            agent_targets={"a0": np.zeros(2)},
            prices=np.zeros(2),
            rho={"a0": np.ones(2)},
            metadata={"key": "value"},
        )
        new = coord.update_state({"a0": np.ones(2)}, state)
        assert new.metadata == {"key": "value"}

    def test_partial_subscription(self) -> None:
        """N_k varies per variable with partial subscriptions."""
        layout = VarLayout(group_slices={G: slice(0, 3)}, total_size=3)
        layout.register_agent("a0", {G: np.array([0, 1])})
        layout.register_agent("a1", {G: np.array([1, 2])})
        coord = SharingCoordinator(layout=layout)

        state = SharingState(
            iteration=0,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3), "a1": np.zeros(3)},
            agent_targets={"a0": np.zeros(3), "a1": np.zeros(3)},
            prices=np.zeros(3),
            rho={"a0": np.ones(3), "a1": np.ones(3)},
        )
        results = {"a0": np.array([4.0, 2.0, 0.0]), "a1": np.array([0.0, 3.0, 6.0])}
        new = coord.update_state(results, state)

        # N_k=[1,2,1], x_sum=[4,5,6], Δ=-[4,5,6], gap=[-4,-2.5,-6]
        np.testing.assert_array_almost_equal(
            new.get_agent_targets("a0"), [0.0, -0.5, 0.0]
        )
        np.testing.assert_array_almost_equal(
            new.get_agent_targets("a1"), [0.0, 0.5, 0.0]
        )
        np.testing.assert_array_almost_equal(
            new.get_agent_prices("a0"), [4.0, 2.5, 6.0]
        )

    def test_type_check(self) -> None:
        from flo_pro_sdk.core.state import ConsensusState

        coord = SharingCoordinator(layout=_layout())
        bad = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2)},
            prices={"a0": np.zeros(2)},
            rho={"a0": np.ones(2)},
        )
        with pytest.raises(TypeError):
            coord.update_state({"a0": np.zeros(2)}, bad)


class TestSharingConvergence:
    def test_converged_small_residuals(self) -> None:
        coord = SharingCoordinator(layout=_layout(), primal_tol=1e-3, dual_tol=1e-3)
        s = _sharing_state()
        s._core.iteration = 10
        s._core.residuals = Residuals(primal=1e-5, dual=1e-5)
        assert coord.check_convergence(s.get_core_state()) is True

    def test_not_converged_large_primal(self) -> None:
        coord = SharingCoordinator(layout=_layout())
        s = _sharing_state()
        s._core.iteration = 10
        s._core.residuals = Residuals(primal=1.0, dual=1e-6)
        assert coord.check_convergence(s.get_core_state()) is False

    def test_converged_at_max_iterations(self) -> None:
        coord = SharingCoordinator(layout=_layout(), max_iterations=50)
        s = _sharing_state()
        s._core.iteration = 50
        assert coord.check_convergence(s.get_core_state()) is True


class TestExchangeConvergence:
    @staticmethod
    def _solve_quadratic_agent(desired, z_j, lam, rho):
        """Closed-form solution of agent subproblem with quadratic objective.

        argmin (x - d)² + λᵀx + (ρ/2)||x - z_j||²
        FOC: 2(x-d) + λ + ρ(x - z_j) = 0  →  x = (2d - λ + ρ z_j) / (2 + ρ)
        """
        return (2 * desired - lam + rho * z_j) / (2.0 + rho)

    def test_two_agent_exchange(self) -> None:
        """Agents want [4] and [2]. Exchange Σx_i=0. Optimal: x_0=1, x_1=-1."""
        layout = _layout(n=2, dim=1)
        coord = SharingCoordinator(
            layout=layout,
            primal_tol=1e-6,
            dual_tol=1e-6,
            max_iterations=500,
        )
        desired = {"a0": np.array([4.0]), "a1": np.array([2.0])}
        state: State = _sharing_state(n=2, dim=1)

        for _ in range(500):
            results = {
                aid: self._solve_quadratic_agent(
                    desired[aid],
                    state.get_agent_targets(aid),
                    state.get_agent_prices(aid),
                    state.get_rho(aid),
                )
                for aid in state.agent_ids
            }
            state = coord.update_state(results, state)
            if coord.check_convergence(state.get_core_state()):
                break

        x0 = state.get_agent_preferred_vars("a0")
        x1 = state.get_agent_preferred_vars("a1")
        np.testing.assert_array_almost_equal(x0 + x1, [0.0], decimal=2)
        np.testing.assert_array_almost_equal(x0, [1.0], decimal=1)

    def test_three_agent_exchange(self) -> None:
        """Agents want [6], [3], [3]. Optimal: sum=0.

        Same agent subproblem as test_two_agent_exchange.
        """
        layout = _layout(n=3, dim=1)
        coord = SharingCoordinator(
            layout=layout,
            primal_tol=1e-6,
            dual_tol=1e-6,
            max_iterations=500,
        )
        desired = {
            "a0": np.array([6.0]),
            "a1": np.array([3.0]),
            "a2": np.array([3.0]),
        }
        state: State = SharingState(
            iteration=0,
            consensus_vars=np.zeros(1),
            agent_preferred_vars={aid: np.zeros(1) for aid in desired},
            agent_targets={aid: np.zeros(1) for aid in desired},
            prices=np.zeros(1),
            rho={aid: np.ones(1) for aid in desired},
        )

        for _ in range(500):
            results = {
                aid: self._solve_quadratic_agent(
                    desired[aid],
                    state.get_agent_targets(aid),
                    state.get_agent_prices(aid),
                    state.get_rho(aid),
                )
                for aid in state.agent_ids
            }
            state = coord.update_state(results, state)
            if coord.check_convergence(state.get_core_state()):
                break

        x_sum = sum(state.get_agent_preferred_vars(aid) for aid in desired)
        np.testing.assert_array_almost_equal(x_sum, [0.0], decimal=2)
        np.testing.assert_array_almost_equal(
            state.get_agent_preferred_vars("a0"),
            [2.0],
            decimal=1,
        )
        np.testing.assert_array_almost_equal(
            state.get_agent_preferred_vars("a1"),
            [-1.0],
            decimal=1,
        )
        np.testing.assert_array_almost_equal(
            state.get_agent_preferred_vars("a2"),
            [-1.0],
            decimal=1,
        )
