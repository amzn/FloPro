"""Tests for core/var_layout.py."""

import numpy as np
import pytest

from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import PublicVarGroupName


class TestVarLayout:
    def _make_layout(self) -> VarLayout:
        layout = VarLayout(
            group_slices={
                PublicVarGroupName("x"): slice(0, 3),
                PublicVarGroupName("y"): slice(3, 5),
            },
            total_size=5,
        )
        layout.register_agent("a1", {PublicVarGroupName("x"): np.array([0, 2])})
        return layout

    def test_unflatten_from_global(self) -> None:
        layout = self._make_layout()
        global_flat = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        grouped = layout.unflatten_from_global("a1", global_flat)
        np.testing.assert_array_equal(grouped[PublicVarGroupName("x")], [10.0, 30.0])

    def test_flatten_to_global(self) -> None:
        layout = self._make_layout()
        grouped = {PublicVarGroupName("x"): np.array([7.0, 9.0])}
        result = layout.flatten_to_global("a1", grouped)
        expected = np.array([7.0, 0.0, 9.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_unflatten_returns_copies(self) -> None:
        layout = self._make_layout()
        global_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        grouped = layout.unflatten_from_global("a1", global_flat)
        grouped[PublicVarGroupName("x")][0] = 999
        assert global_flat[0] == 1.0

    def test_get_global_indices(self) -> None:
        layout = self._make_layout()
        np.testing.assert_array_equal(layout.get_global_indices("a1"), [0, 2])

    def test_unregistered_agent_raises(self) -> None:
        layout = self._make_layout()
        with pytest.raises(KeyError):
            layout.get_global_indices("unknown")

    def test_unflatten_unregistered_agent_raises(self) -> None:
        layout = self._make_layout()
        with pytest.raises(KeyError):
            layout.unflatten_from_global("unknown", np.zeros(5))
