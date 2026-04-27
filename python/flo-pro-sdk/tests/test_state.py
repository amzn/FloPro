"""Tests for core/state.py — ConsensusState and SharingState."""

import numpy as np

from flo_pro_sdk.core.state import ConsensusState, SharingState
from flo_pro_sdk.core.variables import Residuals


class TestConsensusState:
    def _make(self) -> ConsensusState:
        return ConsensusState(
            iteration=5,
            consensus_vars=np.array([1.0, 2.0, 3.0]),
            agent_preferred_vars={
                "a1": np.array([1.1, 2.1, 3.1]),
                "a2": np.array([0.9, 1.9, 2.9]),
            },
            prices={"a1": np.array([0.1, 0.2, 0.3]), "a2": np.array([0.4, 0.5, 0.6])},
            rho={"a1": np.ones(3), "a2": np.ones(3)},
            residuals=Residuals(primal=0.01, dual=0.02),
        )

    def test_iteration(self) -> None:
        assert self._make().iteration == 5

    def test_agent_targets_returns_z(self) -> None:
        s = self._make()
        np.testing.assert_array_equal(s.get_agent_targets("a1"), s.consensus_vars)

    def test_prices(self) -> None:
        np.testing.assert_array_equal(
            self._make().get_agent_prices("a1"), [0.1, 0.2, 0.3]
        )

    def test_agent_ids(self) -> None:
        assert set(self._make().agent_ids) == {"a1", "a2"}

    def test_core_state(self) -> None:
        core = self._make().get_core_state()
        assert core.iteration == 5
        assert core.residuals is not None

    def test_unknown_agent_raises(self) -> None:
        import pytest

        s = self._make()
        with pytest.raises(KeyError):
            s.get_agent_prices("unknown")
        with pytest.raises(KeyError):
            s.get_rho("unknown")


class TestSharingState:
    def test_global_prices(self) -> None:
        state = SharingState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2), "a1": np.zeros(2)},
            agent_targets={"a0": np.zeros(2), "a1": np.zeros(2)},
            prices=np.zeros(2),
            rho={"a0": np.ones(2), "a1": np.ones(2)},
        )
        assert state.get_agent_prices("a0") is state.get_agent_prices("a1")

    def test_per_agent_targets(self) -> None:
        state = SharingState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2), "a1": np.zeros(2)},
            agent_targets={"a0": np.array([1.0, 2.0]), "a1": np.array([3.0, 4.0])},
            prices=np.zeros(2),
            rho={"a0": np.ones(2), "a1": np.ones(2)},
        )
        np.testing.assert_array_equal(state.get_agent_targets("a0"), [1.0, 2.0])
        np.testing.assert_array_equal(state.get_agent_targets("a1"), [3.0, 4.0])

    def test_core_state(self) -> None:
        state = SharingState(
            iteration=3,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"a0": np.zeros(2)},
            agent_targets={"a0": np.zeros(2)},
            prices=np.zeros(2),
            rho={"a0": np.ones(2)},
        )
        assert state.get_core_state().iteration == 3

    def test_agent_ids(self) -> None:
        state = SharingState(
            iteration=0,
            consensus_vars=np.zeros(1),
            agent_preferred_vars={
                "a0": np.zeros(1),
                "a1": np.zeros(1),
                "a2": np.zeros(1),
            },
            agent_targets={"a0": np.zeros(1), "a1": np.zeros(1), "a2": np.zeros(1)},
            prices=np.zeros(1),
            rho={"a0": np.ones(1), "a1": np.ones(1), "a2": np.ones(1)},
        )
        assert set(state.agent_ids) == {"a0", "a1", "a2"}
