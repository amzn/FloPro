"""Tests for DashboardMetricsComputer."""

import numpy as np
import pandas as pd
import pytest

from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.persistence_backend import FileSystemBackend
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.state import AgentPlan, ConsensusState
from flo_pro_sdk.core.variables import (
    PublicVarGroupMetadata,
    PublicVarGroupName,
    Residuals,
)
from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider
from flo_pro_sdk.dashboard.metrics import DashboardMetricsComputer


def _make_registry():
    """Build a registry with 2 agents subscribing to 'energy' (3 vars)."""
    registry = AgentRegistry()
    # Agent a1 subscribes to energy vars [0, 1]
    energy_a1 = PublicVarGroupMetadata(
        name=PublicVarGroupName("energy"),
        var_metadata=pd.DataFrame({"t": [0, 1], "node": ["n1", "n2"]}),
    )
    registry.register_agent(
        "a1",
        {PublicVarGroupName("energy"): energy_a1},
        metadata={"type": "generator"},
    )
    # Agent a2 subscribes to energy vars [1, 2]
    energy_a2 = PublicVarGroupMetadata(
        name=PublicVarGroupName("energy"),
        var_metadata=pd.DataFrame({"t": [1, 2], "node": ["n2", "n3"]}),
    )
    registry.register_agent(
        "a2",
        {PublicVarGroupName("energy"): energy_a2},
        metadata={"type": "load"},
    )
    registry.finalize_registration()
    return registry


@pytest.fixture
def metrics_env(tmp_path):
    """Write test data with known values for verifiable metric computation."""
    registry = _make_registry()
    backend = FileSystemBackend(base_dir=str(tmp_path), chunk_size=100)
    backend.write_metadata(registry)

    for i in range(5):
        # Consensus vars: [10.0, 20.0, 30.0] (constant across iterations)
        consensus = np.array([10.0, 20.0, 30.0])
        state = ConsensusState(
            iteration=i,
            consensus_vars=consensus,
            agent_preferred_vars={"a1": np.array([10.0, 20.0])},
            prices={"a1": np.array([0.0, 0.0])},
            rho={"a1": np.array([1.0, 1.0])},
            residuals=Residuals(primal=10.0 / (i + 1), dual=5.0 / (i + 1)),
        )
        backend.write_state(i, state, timestamp=1000.0 + i)

        # Agent a1 preferred_vars for energy: [10+i, 20+i]
        # subscription indices for a1.energy = [0, 1]
        # so residual = norm([10+i, 20+i] - [10, 20]) = norm([i, i]) = i*sqrt(2)
        plan_a1 = AgentPlan(
            agent_id="a1",
            iteration=i,
            solution=Solution(
                preferred_vars={
                    PublicVarGroupName("energy"): np.array([10.0 + i, 20.0 + i])
                },
                objective=Objective(utility=100.0, subsidy=0.0, proximal=0.0),
            ),
        )
        backend.write_agent_plan(i, "a1", plan_a1)

        # Agent a2 preferred_vars for energy: [20, 30] (constant, zero residual)
        plan_a2 = AgentPlan(
            agent_id="a2",
            iteration=i,
            solution=Solution(
                preferred_vars={PublicVarGroupName("energy"): np.array([20.0, 30.0])},
                objective=Objective(utility=50.0, subsidy=0.0, proximal=0.0),
            ),
        )
        backend.write_agent_plan(i, "a2", plan_a2)

    backend.close()

    provider = DashboardDataProvider(backend.run_dir)
    computer = DashboardMetricsComputer(provider)
    return computer


class TestAgentResiduals:
    def test_residual_values(self, metrics_env):
        df = metrics_env.get_agent_residuals("a1")
        assert len(df) == 5
        # residual[i] = norm([i, i]) = i * sqrt(2)
        expected = [i * np.sqrt(2) for i in range(5)]
        np.testing.assert_allclose(df["residual"].values, expected, atol=1e-10)

    def test_zero_residual_agent(self, metrics_env):
        df = metrics_env.get_agent_residuals("a2")
        assert len(df) == 5
        # a2 preferred_vars == consensus_vars[1:3], so residual is 0
        np.testing.assert_allclose(df["residual"].values, 0.0, atol=1e-10)

    def test_unknown_agent(self, metrics_env):
        df = metrics_env.get_agent_residuals("nonexistent")
        assert df.empty

    def test_empty_data(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        computer = DashboardMetricsComputer(provider)
        df = computer.get_agent_residuals("a1")
        assert df.empty


class TestConvergenceRate:
    def test_convergence_rate_values(self, metrics_env):
        df = metrics_env.get_convergence_rate(window=1)
        assert len(df) == 5
        # primal = [10, 5, 3.33, 2.5, 2.0]
        # rate[k] = (primal[k] - primal[k-1]) / 1
        assert np.isnan(df["convergence_rate"].iloc[0])
        # rate[1] = (5 - 10) / 1 = -5
        np.testing.assert_allclose(df["convergence_rate"].iloc[1], -5.0, atol=0.01)

    def test_window_larger_than_data(self, metrics_env):
        df = metrics_env.get_convergence_rate(window=10)
        assert len(df) == 5
        # All should be NaN since window > data length
        assert df["convergence_rate"].isna().all()

    def test_empty_data(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        computer = DashboardMetricsComputer(provider)
        df = computer.get_convergence_rate()
        assert df.empty


class TestCaching:
    def test_cache_returns_same_object(self, metrics_env):
        df1 = metrics_env.get_agent_residuals("a1")
        df2 = metrics_env.get_agent_residuals("a1")
        assert df1 is df2  # Same object, not just equal

    def test_cache_per_agent(self, metrics_env):
        df1 = metrics_env.get_agent_residuals("a1")
        df2 = metrics_env.get_agent_residuals("a2")
        assert df1 is not df2
        assert len(df1) == 5
        assert len(df2) == 5

    def test_cache_invalidates_on_new_data(self, tmp_path):
        """Cache should refresh when new iterations appear."""
        registry = _make_registry()
        backend = FileSystemBackend(base_dir=str(tmp_path), chunk_size=100)
        backend.write_metadata(registry)

        # Write 2 iterations
        for i in range(2):
            consensus = np.array([10.0, 20.0, 30.0])
            state = ConsensusState(
                iteration=i,
                consensus_vars=consensus,
                agent_preferred_vars={"a1": np.array([10.0, 20.0])},
                prices={"a1": np.array([0.0, 0.0])},
                rho={"a1": np.array([1.0, 1.0])},
                residuals=Residuals(primal=10.0 / (i + 1), dual=5.0 / (i + 1)),
            )
            backend.write_state(i, state, timestamp=1000.0 + i)
            plan = AgentPlan(
                agent_id="a1",
                iteration=i,
                solution=Solution(
                    preferred_vars={
                        PublicVarGroupName("energy"): np.array([10.0 + i, 20.0 + i])
                    },
                    objective=Objective(utility=100.0, subsidy=0.0, proximal=0.0),
                ),
            )
            backend.write_agent_plan(i, "a1", plan)

        provider = DashboardDataProvider(backend.run_dir)
        computer = DashboardMetricsComputer(provider)

        df1 = computer.get_agent_residuals("a1")
        assert len(df1) == 2

        # Write 2 more iterations (simulating ongoing run)
        for i in range(2, 4):
            consensus = np.array([10.0, 20.0, 30.0])
            state = ConsensusState(
                iteration=i,
                consensus_vars=consensus,
                agent_preferred_vars={"a1": np.array([10.0, 20.0])},
                prices={"a1": np.array([0.0, 0.0])},
                rho={"a1": np.array([1.0, 1.0])},
                residuals=Residuals(primal=10.0 / (i + 1), dual=5.0 / (i + 1)),
            )
            backend.write_state(i, state, timestamp=1000.0 + i)
            plan = AgentPlan(
                agent_id="a1",
                iteration=i,
                solution=Solution(
                    preferred_vars={
                        PublicVarGroupName("energy"): np.array([10.0 + i, 20.0 + i])
                    },
                    objective=Objective(utility=100.0, subsidy=0.0, proximal=0.0),
                ),
            )
            backend.write_agent_plan(i, "a1", plan)

        df2 = computer.get_agent_residuals("a1")
        assert len(df2) == 4  # Cache refreshed with new data
        assert df1 is not df2


class TestMetadataRetry:
    def test_retries_when_metadata_initially_missing(self, tmp_path):
        """Metadata should be retried if not available on first call."""
        registry = _make_registry()
        backend = FileSystemBackend(base_dir=str(tmp_path), chunk_size=100)

        # Write iteration data but NOT metadata yet
        consensus = np.array([10.0, 20.0, 30.0])
        state = ConsensusState(
            iteration=0,
            consensus_vars=consensus,
            agent_preferred_vars={"a1": np.array([10.0, 20.0])},
            prices={"a1": np.array([0.0, 0.0])},
            rho={"a1": np.array([1.0, 1.0])},
            residuals=Residuals(primal=10.0, dual=5.0),
        )
        backend.write_state(0, state, timestamp=1000.0)
        plan = AgentPlan(
            agent_id="a1",
            iteration=0,
            solution=Solution(
                preferred_vars={PublicVarGroupName("energy"): np.array([11.0, 21.0])},
                objective=Objective(utility=100.0, subsidy=0.0, proximal=0.0),
            ),
        )
        backend.write_agent_plan(0, "a1", plan)

        provider = DashboardDataProvider(backend.run_dir)
        computer = DashboardMetricsComputer(provider)

        # First call — no metadata, should return empty
        df1 = computer.get_agent_residuals("a1")
        assert df1.empty

        # Now write metadata
        backend.write_metadata(registry)

        # Second call — should retry and find metadata
        df2 = computer.get_agent_residuals("a1")
        assert len(df2) == 1
