"""Property-based tests for registry finalization and VarLayout."""

import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.types import AgentId
from flo_pro_sdk.core.variables import PublicVarGroupMetadata, PublicVarGroupName


def _meta(name: str, size: int, pool_size: int) -> PublicVarGroupMetadata:
    """Create metadata where agent subscribes to `size` variables out of a pool of `pool_size`."""
    # Select `size` indices from [0, pool_size)
    indices = sorted(np.random.choice(pool_size, size=size, replace=False))
    return PublicVarGroupMetadata(
        name=PublicVarGroupName(name),
        var_metadata=pd.DataFrame({"idx": indices}),
    )


@st.composite
def agent_subscriptions(
    draw: st.DrawFn,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Generate agent subscriptions and per-group pool sizes.

    Returns:
        (agents, pool_sizes) where:
        - agents: {agent_id: {group_name: subscribed_count}}
        - pool_sizes: {group_name: k} — total pool size per group
    """
    n_agents: int = draw(st.integers(min_value=1, max_value=5))
    n_groups: int = draw(st.integers(min_value=1, max_value=4))
    group_names: list[str] = [f"g{i}" for i in range(n_groups)]

    # Each group has a pool of k possible variables
    pool_sizes: dict[str, int] = {
        g: draw(st.integers(min_value=1, max_value=10)) for g in group_names
    }

    agents: dict[str, dict[str, int]] = {}
    for i in range(n_agents):
        subscribed: list[str] = draw(
            st.lists(
                st.sampled_from(group_names), min_size=1, max_size=n_groups, unique=True
            )
        )
        sizes: dict[str, int] = {
            g: draw(st.integers(min_value=1, max_value=pool_sizes[g]))
            for g in subscribed
        }
        agents[f"a{i}"] = sizes
    return agents, pool_sizes


def _build_registry(
    subs: dict[str, dict[str, int]], pool_sizes: dict[str, int]
) -> AgentRegistry:
    reg = AgentRegistry()
    for agent_id, groups in subs.items():
        vars_meta = {
            PublicVarGroupName(g): _meta(g, size, pool_sizes[g])
            for g, size in groups.items()
        }
        reg.register_agent(agent_id, vars_meta)
    reg.finalize_registration()
    return reg


AgentSubs = dict[AgentId, dict[str, int]]
PoolSizes = dict[str, int]
RegistryTestData = tuple[AgentSubs, PoolSizes]


class TestRegistryPBT:
    @given(data=agent_subscriptions())
    @settings(max_examples=50)
    def test_total_size_is_sum_of_union_group_sizes(
        self, data: RegistryTestData
    ) -> None:
        subs: AgentSubs
        pool_sizes: PoolSizes
        subs, pool_sizes = data
        reg: AgentRegistry = _build_registry(subs, pool_sizes)
        layout = reg.get_layout()
        global_meta = reg.get_all_subscribed_vars()
        expected = sum(len(m.var_metadata) for m in global_meta.values())
        assert layout.total_size == expected

    @given(data=agent_subscriptions())
    @settings(max_examples=50)
    def test_global_indices_within_bounds(self, data: RegistryTestData) -> None:
        subs: AgentSubs
        pool_sizes: PoolSizes
        subs, pool_sizes = data
        reg: AgentRegistry = _build_registry(subs, pool_sizes)
        layout = reg.get_layout()
        for agent_id in subs:
            indices: np.ndarray = layout.get_global_indices(agent_id)
            assert len(indices) == sum(subs[agent_id].values())
            assert all(0 <= idx < layout.total_size for idx in indices)

    @given(data=agent_subscriptions())
    @settings(max_examples=50)
    def test_flatten_unflatten_roundtrip(self, data: RegistryTestData) -> None:
        subs: AgentSubs
        pool_sizes: PoolSizes
        subs, pool_sizes = data
        reg: AgentRegistry = _build_registry(subs, pool_sizes)
        layout = reg.get_layout()
        for agent_id, groups in subs.items():
            grouped: dict[PublicVarGroupName, np.ndarray] = {
                PublicVarGroupName(g): np.random.randn(size)
                for g, size in groups.items()
            }
            global_flat: np.ndarray = layout.flatten_to_global(agent_id, grouped)
            recovered: dict[PublicVarGroupName, np.ndarray] = (
                layout.unflatten_from_global(agent_id, global_flat)
            )
            for g in grouped:
                np.testing.assert_array_almost_equal(recovered[g], grouped[g])

    @given(data=agent_subscriptions())
    @settings(max_examples=50)
    def test_groups_dont_overlap_in_global(self, data: RegistryTestData) -> None:
        subs: AgentSubs
        pool_sizes: PoolSizes
        subs, pool_sizes = data
        reg: AgentRegistry = _build_registry(subs, pool_sizes)
        layout = reg.get_layout()
        ranges: list[tuple[int, int]] = sorted(
            ((s.start or 0, s.stop or 0) for s in layout.group_slices.values())
        )
        for i in range(len(ranges) - 1):
            assert ranges[i][1] <= ranges[i + 1][0], (
                f"Groups overlap: {ranges[i]} and {ranges[i + 1]}"
            )


class TestPartialSubscription:
    """Deterministic tests for overlapping / partial variable subscriptions."""

    def _register(self, agent_metas: dict[str, pd.DataFrame]) -> AgentRegistry:
        G = PublicVarGroupName("x")
        reg = AgentRegistry()
        for aid, df in agent_metas.items():
            meta = PublicVarGroupMetadata(name=G, var_metadata=df)
            reg.register_agent(aid, {G: meta})
        reg.finalize_registration()
        return reg

    def test_overlapping_subscriptions_merge(self) -> None:
        """Two agents with overlapping rows produce correct global union."""
        reg = self._register(
            {
                "a": pd.DataFrame({"t": [0, 0], "w": [1, 2]}),
                "b": pd.DataFrame({"t": [0, 1], "w": [2, 1]}),
            }
        )
        layout = reg.get_layout()
        # union: (0,1), (0,2), (1,1) → size 3
        assert layout.total_size == 3
        np.testing.assert_array_equal(layout.get_global_indices("a"), [0, 1])
        np.testing.assert_array_equal(layout.get_global_indices("b"), [1, 2])

    def test_disjoint_subscriptions(self) -> None:
        """Two agents with no overlap produce union of both."""
        reg = self._register(
            {
                "a": pd.DataFrame({"t": [0, 0], "w": [1, 2]}),
                "b": pd.DataFrame({"t": [1, 1], "w": [1, 2]}),
            }
        )
        layout = reg.get_layout()
        assert layout.total_size == 4
        np.testing.assert_array_equal(layout.get_global_indices("a"), [0, 1])
        np.testing.assert_array_equal(layout.get_global_indices("b"), [2, 3])

    def test_identical_subscriptions(self) -> None:
        """Two agents subscribing to the same rows produce no duplicates."""
        reg = self._register(
            {
                "a": pd.DataFrame({"t": [0, 1], "w": [1, 2]}),
                "b": pd.DataFrame({"t": [0, 1], "w": [1, 2]}),
            }
        )
        layout = reg.get_layout()
        assert layout.total_size == 2
        np.testing.assert_array_equal(layout.get_global_indices("a"), [0, 1])
        np.testing.assert_array_equal(layout.get_global_indices("b"), [0, 1])

    def test_subset_subscription(self) -> None:
        """One agent subscribes to a subset of another's variables."""
        reg = self._register(
            {
                "a": pd.DataFrame({"t": [0, 0, 1], "w": [1, 2, 1]}),
                "b": pd.DataFrame({"t": [0], "w": [2]}),
            }
        )
        layout = reg.get_layout()
        assert layout.total_size == 3
        np.testing.assert_array_equal(layout.get_global_indices("a"), [0, 1, 2])
        np.testing.assert_array_equal(layout.get_global_indices("b"), [1])

    def test_flatten_unflatten_with_overlap(self) -> None:
        """Flatten/unflatten roundtrip works with overlapping subscriptions."""
        G = PublicVarGroupName("x")
        reg = self._register(
            {
                "a": pd.DataFrame({"t": [0, 0], "w": [1, 2]}),
                "b": pd.DataFrame({"t": [0, 1], "w": [2, 1]}),
            }
        )
        layout = reg.get_layout()

        flat_a = layout.flatten_to_global("a", {G: np.array([10.0, 20.0])})
        np.testing.assert_array_equal(flat_a, [10.0, 20.0, 0.0])

        flat_b = layout.flatten_to_global("b", {G: np.array([30.0, 40.0])})
        np.testing.assert_array_equal(flat_b, [0.0, 30.0, 40.0])

        recovered_a = layout.unflatten_from_global("a", np.array([100.0, 200.0, 300.0]))
        np.testing.assert_array_equal(recovered_a[G], [100.0, 200.0])

        recovered_b = layout.unflatten_from_global("b", np.array([100.0, 200.0, 300.0]))
        np.testing.assert_array_equal(recovered_b[G], [200.0, 300.0])

    def test_indices_preserve_var_metadata_order(self) -> None:
        """Agent indices must follow the row order of the agent's var_metadata."""
        G = PublicVarGroupName("x")
        reg = AgentRegistry()
        # Global union after dedup+sort: (t=0,w=1),(t=1,w=5),(t=2,w=9) → global positions 0,1,2
        reg.register_agent(
            "a",
            {
                G: PublicVarGroupMetadata(
                    name=G,
                    var_metadata=pd.DataFrame({"t": [2, 0, 1], "w": [9, 1, 5]}),
                )
            },
        )
        # Agent "b" registers (t=1,w=5) before (t=0,w=1) — reversed from global order
        reg.register_agent(
            "b",
            {
                G: PublicVarGroupMetadata(
                    name=G,
                    var_metadata=pd.DataFrame({"t": [1, 0], "w": [5, 1]}),
                )
            },
        )
        reg.finalize_registration()
        layout = reg.get_layout()
        # "a" subscribes to all 3; agent order (2,9),(0,1),(1,5) → global positions [2, 0, 1]
        np.testing.assert_array_equal(layout.get_global_indices("a"), [2, 0, 1])
        # "b" subscribes to (1,5) then (0,1) — agent order → global positions [1, 0]
        np.testing.assert_array_equal(layout.get_global_indices("b"), [1, 0])

    def test_unsorted_indices_roundtrip(self) -> None:
        """Global positions must be deterministic regardless of agent registration order."""
        G = PublicVarGroupName("x")
        reg = AgentRegistry()
        # Agent registers indices in reverse order [2, 0, 1]
        reg.register_agent(
            "a",
            {
                G: PublicVarGroupMetadata(
                    name=G,
                    var_metadata=pd.DataFrame({"idx": [2, 0, 1]}),
                )
            },
        )
        reg.finalize_registration()
        layout = reg.get_layout()

        # Agent gives values [10, 20, 30] for its vars (idx=2, idx=0, idx=1)
        flat = layout.flatten_to_global("a", {G: np.array([10.0, 20.0, 30.0])})
        # Global position 0=idx0=20, position 1=idx1=30, position 2=idx2=10
        np.testing.assert_array_equal(flat, [20.0, 30.0, 10.0])

        # Unflatten: global [100, 200, 300] → agent order (idx2, idx0, idx1)
        recovered = layout.unflatten_from_global("a", np.array([100.0, 200.0, 300.0]))
        np.testing.assert_array_equal(recovered[G], [300.0, 100.0, 200.0])
