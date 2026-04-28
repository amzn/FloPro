"""Tests for DashboardDataProvider."""

import numpy as np
import pytest

from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.state import ConsensusState, AgentPlan
from flo_pro_sdk.core.persistence_backend import (
    FileSystemBackend,
)
from flo_pro_sdk.core.variables import PublicVarGroupName, Residuals
from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider


def make_state(
    iteration: int, primal: float = 1.0, dual: float = 0.5
) -> ConsensusState:
    return ConsensusState(
        iteration=iteration,
        consensus_vars=np.array([float(iteration), float(iteration) + 1.0]),
        agent_preferred_vars={
            "a": np.array([float(iteration), float(iteration) + 0.5])
        },
        prices={"a": np.array([float(iteration * 0.1), 0.2])},
        rho={"a": np.array([1.0, 1.0])},
        residuals=Residuals(primal=primal, dual=dual),
    )


def make_plan(agent_id: str, iteration: int) -> AgentPlan:
    return AgentPlan(
        agent_id=agent_id,
        iteration=iteration,
        solution=Solution(
            preferred_vars={PublicVarGroupName("y"): np.array([1.0, 2.0, 3.0])},
            objective=Objective(utility=10.0 + iteration, subsidy=5.0, proximal=2.0),
        ),
    )


@pytest.fixture
def populated_run(tmp_path):
    """Write test data via FileSystemBackend, return (run_dir, backend)."""
    backend = FileSystemBackend(base_dir=str(tmp_path), chunk_size=100)
    for i in range(5):
        state = make_state(i, primal=10.0 / (i + 1), dual=5.0 / (i + 1))
        backend.write_state(i, state, timestamp=1000.0 + i)
        backend.write_agent_plan(i, "agent_a", make_plan("agent_a", i))
        backend.write_agent_plan(i, "agent_b", make_plan("agent_b", i))
    backend.close()
    return backend.run_dir


class TestConvergenceData:
    def test_reads_all_iterations(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_convergence_data()
        assert len(df) == 5
        assert list(df["iteration"]) == [0, 1, 2, 3, 4]

    def test_has_expected_columns(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_convergence_data()
        assert "iteration" in df.columns
        assert "primal_residual" in df.columns
        assert "dual_residual" in df.columns
        assert "timestamp" in df.columns

    def test_residual_values(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_convergence_data()
        np.testing.assert_allclose(df["primal_residual"].iloc[0], 10.0)
        np.testing.assert_allclose(df["primal_residual"].iloc[4], 2.0)

    def test_timestamp_values(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_convergence_data()
        np.testing.assert_allclose(df["timestamp"].iloc[0], 1000.0)
        np.testing.assert_allclose(df["timestamp"].iloc[4], 1004.0)

    def test_empty_when_no_data(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        df = provider.get_convergence_data()
        assert df.empty


class TestAgentSolutions:
    def test_reads_single_agent(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_agent_solutions(agent_id="agent_a")
        assert len(df) == 5
        assert all(df["agent_id"] == "agent_a") if "agent_id" in df.columns else True

    def test_reads_all_agents(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_agent_solutions()
        assert len(df) == 10  # 5 iterations x 2 agents

    def test_column_projection(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_agent_solutions(agent_id="agent_a", columns=["utility"])
        assert "utility" in df.columns
        assert "iteration" in df.columns
        # Binary array columns should not be loaded
        assert "preferred_vars" not in df.columns

    def test_utility_values(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        df = provider.get_agent_solutions(agent_id="agent_a", columns=["utility"])
        assert list(df["utility"]) == [10.0, 11.0, 12.0, 13.0, 14.0]

    def test_empty_when_no_data(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        df = provider.get_agent_solutions(agent_id="nonexistent")
        assert df.empty


class TestConsensusVars:
    def test_reads_single_iteration(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        arr = provider.get_consensus_vars(iteration=3)
        assert arr is not None
        np.testing.assert_allclose(arr, [3.0, 4.0])

    def test_returns_none_for_missing_iteration(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        arr = provider.get_consensus_vars(iteration=999)
        assert arr is None

    def test_returns_none_when_no_data(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        arr = provider.get_consensus_vars(iteration=0)
        assert arr is None

    def test_get_all_consensus_vars(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        all_cv = provider.get_all_consensus_vars()
        assert len(all_cv) == 5
        assert sorted(all_cv.keys()) == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(all_cv[0], [0.0, 1.0])
        np.testing.assert_allclose(all_cv[4], [4.0, 5.0])

    def test_get_all_consensus_vars_empty(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        assert provider.get_all_consensus_vars() == {}


class TestProblemMetadata:
    def test_returns_none_when_no_metadata(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        assert provider.get_problem_metadata() is None

    def test_reads_metadata_round_trip(self, tmp_path):
        import pandas as pd_
        from flo_pro_sdk.core.registry import AgentRegistry
        from flo_pro_sdk.core.variables import (
            PublicVarGroupMetadata,
            PublicVarGroupName,
        )

        registry = AgentRegistry()
        energy_meta = PublicVarGroupMetadata(
            name=PublicVarGroupName("energy"),
            var_metadata=pd_.DataFrame({"t": [0, 1], "node": ["n1", "n2"]}),
        )
        registry.register_agent(
            "a1",
            {PublicVarGroupName("energy"): energy_meta},
            metadata={"type": "generator"},
        )
        registry.finalize_registration()

        backend = FileSystemBackend(base_dir=str(tmp_path), chunk_size=100)
        backend.write_metadata(registry)
        backend.close()

        provider = DashboardDataProvider(backend.run_dir)
        metadata = provider.get_problem_metadata()
        assert metadata is not None
        assert len(metadata["agents"]) == 1
        assert metadata["agents"][0]["agent_id"] == "a1"
        assert "energy" in metadata["var_metadata"]


class TestMidRunReads:
    """Test reading from an unclosed backend (L0 files only, no L2 compaction)."""

    def test_reads_l0_files_without_close(self, tmp_path):
        backend = FileSystemBackend(base_dir=str(tmp_path), chunk_size=100)
        for i in range(3):
            state = make_state(i)
            backend.write_state(i, state, timestamp=1000.0 + i)
            backend.write_agent_plan(i, "agent_a", make_plan("agent_a", i))
        # Do NOT close — simulates a live dashboard reading mid-run

        provider = DashboardDataProvider(backend.run_dir)
        df = provider.get_convergence_data()
        assert len(df) == 3

        df_sol = provider.get_agent_solutions(agent_id="agent_a")
        assert len(df_sol) == 3

        all_cv = provider.get_all_consensus_vars()
        assert len(all_cv) == 3


class TestManifest:
    def test_reads_manifest(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        manifest = provider.get_manifest()
        assert manifest is not None
        assert "coordination_id" in manifest
        assert manifest["status"] == "completed"

    def test_returns_none_when_no_manifest(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        assert provider.get_manifest() is None


class TestListAgentIds:
    def test_discovers_agents(self, populated_run):
        provider = DashboardDataProvider(populated_run)
        ids = provider.list_agent_ids()
        assert sorted(ids) == ["agent_a", "agent_b"]

    def test_empty_when_no_data(self, tmp_path):
        provider = DashboardDataProvider(tmp_path)
        assert provider.list_agent_ids() == []
