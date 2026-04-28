"""Tests for ADMMCoordinator"""

import numpy as np
import pytest

from flo_pro_sdk.coordinator.admm_coordinator import ADMMCoordinator
from flo_pro_sdk.core.state import ConsensusState, State
from flo_pro_sdk.core.structure_function import ZeroFunction
from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import PublicVarGroupName, Residuals


# ── Helpers ────────────────────────────────────────────────────────────────

G = PublicVarGroupName("flat")


def _layout(n: int = 2, dim: int = 3) -> VarLayout:
    layout = VarLayout(group_slices={G: slice(0, dim)}, total_size=dim)
    for i in range(n):
        layout.register_agent(f"a{i}", {G: np.arange(dim)})
    return layout


def _state(n: int = 2, dim: int = 3) -> ConsensusState:
    return ConsensusState(
        iteration=0,
        consensus_vars=np.zeros(dim),
        agent_preferred_vars={f"a{i}": np.zeros(dim) for i in range(n)},
        prices={f"a{i}": np.zeros(dim) for i in range(n)},
        rho={f"a{i}": np.ones(dim) for i in range(n)},
    )


def _state_with_rho(rho_val: float, n: int = 2, dim: int = 3) -> ConsensusState:
    return ConsensusState(
        iteration=0,
        consensus_vars=np.zeros(dim),
        agent_preferred_vars={f"a{i}": np.zeros(dim) for i in range(n)},
        prices={f"a{i}": np.zeros(dim) for i in range(n)},
        rho={f"a{i}": np.full(dim, rho_val) for i in range(n)},
    )


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def simple_coordinator() -> ADMMCoordinator:
    return ADMMCoordinator(rho_adaptive=False, layout=_layout())


@pytest.fixture
def adaptive_coordinator() -> ADMMCoordinator:
    return ADMMCoordinator(
        rho_adaptive=True,
        rho_scale_factor=2.0,
        primal_tol=1e-3,
        dual_tol=1e-5,
        max_iterations=50,
        layout=_layout(),
    )


# ── update_state tests ────────────────────────────────────────────────────


class TestUpdateState:
    def test_consensus_update(self, simple_coordinator: ADMMCoordinator) -> None:
        """z = avg(x_i)."""
        results = {"a0": np.array([2.0, 4.0, 6.0]), "a1": np.array([4.0, 6.0, 8.0])}
        new = simple_coordinator.update_state(results, _state())
        assert isinstance(new, ConsensusState)
        assert new.iteration == 1
        np.testing.assert_array_almost_equal(new.consensus_vars, [3.0, 5.0, 7.0])

    def test_price_update(self, simple_coordinator: ADMMCoordinator) -> None:
        """y_i = y_i + rho * (x_i - z)."""
        results = {"a0": np.array([2.0, 0.0, 0.0]), "a1": np.array([0.0, 0.0, 0.0])}
        new = simple_coordinator.update_state(results, _state())
        # z = [1, 0, 0], y_a0 = 0 + 1*(2-1) = [1, 0, 0]
        np.testing.assert_array_almost_equal(
            new.get_agent_prices("a0"), [1.0, 0.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            new.get_agent_prices("a1"), [-1.0, 0.0, 0.0]
        )

    def test_residuals_computed(self, simple_coordinator: ADMMCoordinator) -> None:
        results = {"a0": np.array([2.0, 0.0, 0.0]), "a1": np.zeros(3)}
        new = simple_coordinator.update_state(results, _state())
        assert new.residuals is not None
        # z = [1, 0, 0]; primal: sqrt((2-1)^2 + (0-1)^2) = sqrt(2)
        assert new.residuals.primal == pytest.approx(np.sqrt(2.0))
        # dual: z moved from [0,0,0] to [1,0,0]; sqrt(1^2 + 1^2) = sqrt(2)
        assert new.residuals.dual == pytest.approx(np.sqrt(2.0))

    def test_iteration_increments(self, simple_coordinator: ADMMCoordinator) -> None:
        state = _state()
        new = simple_coordinator.update_state(
            {"a0": np.zeros(3), "a1": np.zeros(3)}, state
        )
        assert new.iteration == 1

    def test_metadata_preserved(self, simple_coordinator: ADMMCoordinator) -> None:
        simple_coordinator = ADMMCoordinator(
            rho_adaptive=False, layout=_layout(n=1, dim=2)
        )
        state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2)},
            prices={"a0": np.zeros(2)},
            rho={"a0": np.ones(2)},
            metadata={"key": "value"},
        )
        new = simple_coordinator.update_state({"a0": np.ones(2)}, state)
        assert new.metadata == {"key": "value"}

    def test_custom_rho(self, simple_coordinator: ADMMCoordinator) -> None:
        """Custom rho values affect price update."""
        simple_coordinator = ADMMCoordinator(
            rho_adaptive=False, layout=_layout(n=2, dim=2)
        )
        state = _state_with_rho(5.0, n=2, dim=2)
        results = {"a0": np.array([2.0, 0.0]), "a1": np.array([0.0, 0.0])}
        new = simple_coordinator.update_state(results, state)
        # z = [1, 0], y_a0 = 0 + 5*(2-1) = [5, 0]
        np.testing.assert_array_almost_equal(new.get_agent_prices("a0"), [5.0, 0.0])
        np.testing.assert_array_almost_equal(new.get_rho("a0"), [5.0, 5.0])

    def test_single_agent(self, simple_coordinator: ADMMCoordinator) -> None:
        state = _state(n=1, dim=2)
        simple_coordinator = ADMMCoordinator(
            rho_adaptive=False, layout=_layout(n=1, dim=2)
        )
        results = {"a0": np.array([3.0, 4.0])}
        new = simple_coordinator.update_state(results, state)
        # z = x_0 (single agent), prices = 0 + 1*(x-z) = 0
        np.testing.assert_array_almost_equal(new.consensus_vars, [3.0, 4.0])
        np.testing.assert_array_almost_equal(new.get_agent_prices("a0"), [0.0, 0.0])

    def test_many_agents(self, simple_coordinator: ADMMCoordinator) -> None:
        state = _state(n=5, dim=2)
        simple_coordinator = ADMMCoordinator(
            rho_adaptive=False, layout=_layout(n=5, dim=2)
        )
        results = {f"a{i}": np.array([float(i), float(i)]) for i in range(5)}
        new = simple_coordinator.update_state(results, state)
        np.testing.assert_array_almost_equal(new.consensus_vars, [2.0, 2.0])

    def test_custom_structure_function(self) -> None:
        """ZeroFunction: z = 0 regardless of agent results."""
        layout = _layout(n=2, dim=2)
        coord = ADMMCoordinator(
            layout=layout,
            structure_function=ZeroFunction(layout=layout),
            rho_adaptive=False,
        )
        results = {"a0": np.array([2.0, 4.0]), "a1": np.array([6.0, 8.0])}
        new = coord.update_state(results, _state(dim=2))
        np.testing.assert_array_equal(new.consensus_vars, [0.0, 0.0])

    def test_partial_subscription_price_update(self) -> None:
        """Prices and residuals only update on subscribed indices."""
        # 3 global vars: a0 subscribes to [0,1], a1 subscribes to [1,2]
        layout = VarLayout(group_slices={G: slice(0, 3)}, total_size=3)
        layout.register_agent("a0", {G: np.array([0, 1])})
        layout.register_agent("a1", {G: np.array([1, 2])})
        coord = ADMMCoordinator(layout=layout, rho_adaptive=False)
        state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3), "a1": np.zeros(3)},
            prices={"a0": np.zeros(3), "a1": np.zeros(3)},
            rho={"a0": np.ones(3), "a1": np.ones(3)},
        )
        # a0 proposes [4, 3, 0], a1 proposes [0, 2, 6]
        results = {"a0": np.array([4.0, 3.0, 0.0]), "a1": np.array([0.0, 2.0, 6.0])}
        new = coord.update_state(results, state)
        # z at idx 0: only a0 subscribes → z[0]=4, idx 1: both → z[1]=avg(3,2)=2.5, idx 2: only a1 → z[2]=6
        np.testing.assert_array_almost_equal(new.consensus_vars, [4.0, 2.5, 6.0])
        # a0 prices: idx 0,1 updated (subscribed), idx 2 stays 0 (unsubscribed)
        # y_a0[0] = 0 + 1*(4-4) = 0, y_a0[1] = 0 + 1*(3-2.5) = 0.5, y_a0[2] = 0 (unsubscribed)
        np.testing.assert_array_almost_equal(
            new.get_agent_prices("a0"), [0.0, 0.5, 0.0]
        )
        # a1 prices: idx 0 stays 0 (unsubscribed), idx 1,2 updated
        # y_a1[1] = 0 + 1*(2-2.5) = -0.5, y_a1[2] = 0 + 1*(6-6) = 0
        np.testing.assert_array_almost_equal(
            new.get_agent_prices("a1"), [0.0, -0.5, 0.0]
        )
        # Residuals only account for subscribed indices per agent
        assert new.residuals is not None
        # Primal: a0 idx[0,1]: (4-4)^2+(3-2.5)^2=0.25; a1 idx[1,2]: (2-2.5)^2+(6-6)^2=0.25 → sqrt(0.5)
        assert new.residuals.primal == pytest.approx(np.sqrt(0.5))
        # Dual (z_diff=[4,2.5,6]): a0 idx[0,1]: 4^2+2.5^2=22.25; a1 idx[1,2]: 2.5^2+6^2=42.25 → sqrt(64.5)
        assert new.residuals.dual == pytest.approx(np.sqrt(64.5))

    def test_rho_weighted_averaging(self) -> None:
        """Rho-weighted averaging with variant rho: consensus, price, and residual updates."""
        layout = VarLayout(group_slices={G: slice(0, 2)}, total_size=2)
        layout.register_agent("a0", {G: np.array([0, 1])})
        layout.register_agent("a1", {G: np.array([0, 1])})
        coord = ADMMCoordinator(layout=layout, rho_adaptive=False)
        state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2), "a1": np.zeros(2)},
            prices={"a0": np.zeros(2), "a1": np.zeros(2)},
            rho={"a0": np.array([3.0, 3.0]), "a1": np.array([1.0, 1.0])},
        )
        # a0 (rho=3) proposes [4, 0], a1 (rho=1) proposes [0, 4]
        results = {"a0": np.array([4.0, 0.0]), "a1": np.array([0.0, 4.0])}
        new = coord.update_state(results, state)
        # Consensus: z = (3*4+1*0)/(3+1), (3*0+1*4)/(3+1) = [3, 1]
        np.testing.assert_array_almost_equal(new.consensus_vars, [3.0, 1.0])
        # Price: y_a0 = 0 + 3*(4-3, 0-1) = [3, -3]; y_a1 = 0 + 1*(0-3, 4-1) = [-3, 3]
        np.testing.assert_array_almost_equal(new.get_agent_prices("a0"), [3.0, -3.0])
        np.testing.assert_array_almost_equal(new.get_agent_prices("a1"), [-3.0, 3.0])
        # Residual: primal per agent: a0: (4-3)^2+(0-1)^2=2; a1: (0-3)^2+(4-1)^2=18 → sqrt(20)
        #           dual (z_diff=[3,1]): a0: (3*3)^2+(3*1)^2=90; a1: (1*3)^2+(1*1)^2=10 → sqrt(100)=10
        assert new.residuals is not None
        assert new.residuals.primal == pytest.approx(np.sqrt(20.0))
        assert new.residuals.dual == pytest.approx(10.0)


# ── check_convergence tests ───────────────────────────────────────────────


class TestCheckConvergence:
    def test_converged_with_small_residuals(
        self, simple_coordinator: ADMMCoordinator
    ) -> None:
        core = ConsensusState(
            iteration=10,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
            residuals=Residuals(primal=1e-6, dual=1e-6),
        ).get_core_state()
        assert simple_coordinator.check_convergence(core) is True

    def test_not_converged_large_primal(
        self, simple_coordinator: ADMMCoordinator
    ) -> None:
        core = ConsensusState(
            iteration=10,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
            residuals=Residuals(primal=1.0, dual=1e-6),
        ).get_core_state()
        assert simple_coordinator.check_convergence(core) is False

    def test_not_converged_large_dual(
        self, simple_coordinator: ADMMCoordinator
    ) -> None:
        core = ConsensusState(
            iteration=10,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
            residuals=Residuals(primal=1e-6, dual=1.0),
        ).get_core_state()
        assert simple_coordinator.check_convergence(core) is False

    def test_converged_at_max_iterations(
        self, simple_coordinator: ADMMCoordinator
    ) -> None:
        core = ConsensusState(
            iteration=1000,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
        ).get_core_state()
        assert simple_coordinator.check_convergence(core) is True

    def test_not_converged_without_residuals(
        self, simple_coordinator: ADMMCoordinator
    ) -> None:
        core = ConsensusState(
            iteration=10,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
        ).get_core_state()
        assert simple_coordinator.check_convergence(core) is False

    def test_custom_tolerances(self, adaptive_coordinator: ADMMCoordinator) -> None:
        # adaptive_coordinator: primal_tol=1e-3, dual_tol=1e-5
        core = ConsensusState(
            iteration=10,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
            residuals=Residuals(primal=5e-4, dual=5e-6),
        ).get_core_state()
        assert adaptive_coordinator.check_convergence(core) is True

    def test_custom_max_iterations(self, adaptive_coordinator: ADMMCoordinator) -> None:
        # adaptive_coordinator: max_iterations=50
        core = ConsensusState(
            iteration=50,
            consensus_vars=np.zeros(3),
            agent_preferred_vars={"a0": np.zeros(3)},
            prices={"a0": np.zeros(3)},
            rho={"a0": np.ones(3)},
        ).get_core_state()
        assert adaptive_coordinator.check_convergence(core) is True


# ── Adaptive rho tests ─────────────────────────────────────────────────────


class TestAdaptiveRho:
    def test_rho_increases_when_primal_large(self) -> None:
        coord = ADMMCoordinator(
            rho_adaptive=True, rho_scale_factor=2.0, layout=_layout(n=2, dim=2)
        )
        results = {"a0": np.array([100.0, 100.0]), "a1": np.array([-100.0, -100.0])}
        new = coord.update_state(results, _state(dim=2))
        np.testing.assert_array_almost_equal(new.get_rho("a0"), [2.0, 2.0])

    def test_rho_decreases_when_dual_large(self) -> None:
        """When dual >> primal, rho should decrease."""
        coord = ADMMCoordinator(
            rho_adaptive=True, rho_scale_factor=2.0, layout=_layout(n=2, dim=1)
        )
        # Create scenario: agents agree (small primal) but z moves a lot (large dual)
        # Start with z far from where agents will land
        state = ConsensusState(
            iteration=0,
            consensus_vars=np.array([100.0]),
            agent_preferred_vars={"a0": np.array([100.0]), "a1": np.array([100.0])},
            prices={"a0": np.zeros(1), "a1": np.zeros(1)},
            rho={"a0": np.array([1.0]), "a1": np.array([1.0])},
        )
        # Agents both want 0 → z moves from 100 to 0 (huge dual), primal = 0
        results = {"a0": np.array([0.0]), "a1": np.array([0.0])}
        new = coord.update_state(results, state)
        np.testing.assert_array_almost_equal(new.get_rho("a0"), [0.5])

    def test_no_adaptation_when_disabled(self) -> None:
        coord = ADMMCoordinator(rho_adaptive=False, layout=_layout(n=2, dim=1))
        results = {"a0": np.array([100.0]), "a1": np.array([-100.0])}
        new = coord.update_state(results, _state(n=2, dim=1))
        np.testing.assert_array_almost_equal(new.get_rho("a0"), [1.0])


# ── Multi-iteration convergence ───────────────────────────────────────────


class TestMultiIteration:
    def test_convergence_to_average(self) -> None:
        """Two agents with fixed targets should converge to their average."""
        coord = ADMMCoordinator(
            rho_adaptive=False, primal_tol=0.1, dual_tol=0.1, layout=_layout(n=2, dim=1)
        )
        state: State = ConsensusState(
            iteration=0,
            consensus_vars=np.array([0.0]),
            agent_preferred_vars={"a0": np.zeros(1), "a1": np.zeros(1)},
            prices={"a0": np.zeros(1), "a1": np.zeros(1)},
            rho={"a0": np.array([1.0]), "a1": np.array([1.0])},
        )
        for _ in range(50):
            results = {
                aid: np.array([3.0 if aid == "a0" else 1.0])
                - state.get_agent_prices(aid) / state.get_rho(aid)
                for aid in state.agent_ids
            }
            state = coord.update_state(results, state)
            if coord.check_convergence(state.get_core_state()):
                break
        np.testing.assert_array_almost_equal(state.consensus_vars, [2.0], decimal=1)
