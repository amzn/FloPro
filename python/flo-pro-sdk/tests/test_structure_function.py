"""Tests for core/structure_function.py."""

import numpy as np

from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import PublicVarGroupName
from flo_pro_sdk.core.structure_function import (
    AveragingFunction,
    StructureFunctionSpec,
    ZeroFunction,
)


def _make_state(agent_preferred: dict, consensus: np.ndarray) -> ConsensusState:
    return ConsensusState(
        iteration=0,
        consensus_vars=consensus,
        agent_preferred_vars=agent_preferred,
        prices={aid: np.zeros_like(v) for aid, v in agent_preferred.items()},
        rho={aid: np.ones_like(v) for aid, v in agent_preferred.items()},
    )


class TestAveragingFunction:
    def test_average(self) -> None:
        layout = VarLayout(
            group_slices={PublicVarGroupName("x"): slice(0, 2)}, total_size=2
        )
        layout.register_agent("a1", {PublicVarGroupName("x"): np.array([0, 1])})
        layout.register_agent("a2", {PublicVarGroupName("x"): np.array([0, 1])})
        fn = AveragingFunction(layout=layout)
        state = _make_state(
            {"a1": np.array([2.0, 4.0]), "a2": np.array([6.0, 8.0])}, np.zeros(2)
        )
        np.testing.assert_array_equal(fn.solve(state), [4.0, 6.0])

    def test_empty_agents_returns_zeros(self) -> None:
        layout = VarLayout(
            group_slices={PublicVarGroupName("x"): slice(0, 2)}, total_size=2
        )
        fn = AveragingFunction(layout=layout)
        state = _make_state({}, np.zeros(2))
        result = fn.solve(state)
        np.testing.assert_array_equal(result, [0.0, 0.0])


class TestZeroFunction:
    def test_zero(self) -> None:
        layout = VarLayout(
            group_slices={PublicVarGroupName("x"): slice(0, 2)}, total_size=2
        )
        layout.register_agent("a1", {PublicVarGroupName("x"): np.array([0, 1])})
        fn = ZeroFunction(layout=layout)
        state = _make_state({"a1": np.array([1.0, 2.0])}, np.array([5.0, 6.0]))
        np.testing.assert_array_equal(fn.solve(state), [0.0, 0.0])


class TestStructureFunctionSpec:
    def test_instantiate(self) -> None:
        spec = StructureFunctionSpec(AveragingFunction)
        layout = VarLayout(
            group_slices={PublicVarGroupName("x"): slice(0, 2)}, total_size=2
        )
        fn = spec.instantiate(layout, {})
        assert isinstance(fn, AveragingFunction)
        assert fn.layout is layout
