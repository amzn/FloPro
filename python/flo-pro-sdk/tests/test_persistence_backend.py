"""Tests for FileSystemBackend with partitioned Parquet dataset layout."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.state import ConsensusState, AgentPlan
from flo_pro_sdk.core.persistence_backend import (
    FileSystemBackend,
    CONVERGENCE_DIR,
    CONSENSUS_VARS_DIR,
    AGENT_SOLUTIONS_DIR,
    METADATA_DIR,
    VAR_METADATA_DIR,
    VAR_METADATA_FILE,
    COMPACTION_MARKER,
    L2_CONVERGENCE_FILE,
    L2_CONSENSUS_VARS_FILE,
    L2_AGENT_SOLUTIONS_FILE,
    _dedup_table,
)
from flo_pro_sdk.core.coordination_run import CoordinationRun, read_manifest
from flo_pro_sdk.core.variables import PublicVarGroupName, Residuals

# Default timestamp for tests that don't care about the specific value
_TS = 1710000000.0


def make_state(
    iteration: int, primal: float = 1.0, dual: float = 0.5
) -> ConsensusState:
    """Create a test ConsensusState with residuals."""
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


def make_state_no_residuals(iteration: int) -> ConsensusState:
    """Create a test ConsensusState without residuals."""
    return ConsensusState(
        iteration=iteration,
        consensus_vars=np.array([float(iteration)]),
        agent_preferred_vars={"a": np.array([float(iteration)])},
        prices={"a": np.array([0.1])},
        rho={"a": np.array([1.0])},
    )


def make_plan(agent_id: str, iteration: int) -> AgentPlan:
    """Create a test agent plan."""
    return AgentPlan(
        agent_id=agent_id,
        iteration=iteration,
        solution=Solution(
            preferred_vars={PublicVarGroupName("y"): np.array([1.0, 2.0, 3.0])},
            objective=Objective(utility=10.0, subsidy=5.0, proximal=2.0),
        ),
    )


class TestFileSystemBackendInit:
    """Tests for FileSystemBackend initialization."""

    def test_creates_run_directory(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            assert store.run_dir.exists()
            assert store.run_dir.is_dir()
            store.close()

    def test_creates_dataset_directories(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            assert (store.run_dir / CONVERGENCE_DIR).is_dir()
            assert (store.run_dir / CONSENSUS_VARS_DIR).is_dir()
            assert (store.run_dir / AGENT_SOLUTIONS_DIR).is_dir()
            store.close()

    def test_custom_identity(self):
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="coord_test", run_id="my_run")
            store = FileSystemBackend(base_dir=base_dir, identity=identity)
            assert store.run_id == "my_run"
            assert store.coordination_id == "coord_test"
            assert store.run_dir.name == "my_run"
            assert store.run_dir.parent.name == "coord_test"
            store.close()

    def test_auto_generated_run_id(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            assert len(store.run_id) == 22  # YYYYMMDD_HHMMSS_ffffff
            store.close()

    def test_unique_run_directories(self):
        with tempfile.TemporaryDirectory() as base_dir:
            stores = [FileSystemBackend(base_dir=base_dir) for _ in range(3)]
            run_dirs = [s.run_dir for s in stores]
            assert len(set(run_dirs)) == 3
            for s in stores:
                s.close()


class TestDirectoryLayout:
    """Tests for the partitioned directory layout."""

    def test_convergence_file_path(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_state(5, make_state(5), timestamp=_TS)
            expected = store.run_dir / CONVERGENCE_DIR / "iter_000005.parquet"
            assert expected.exists()
            store.close()

    def test_consensus_vars_file_path(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_state(5, make_state(5), timestamp=_TS)
            expected = store.run_dir / CONSENSUS_VARS_DIR / "iter_000005.parquet"
            assert expected.exists()
            store.close()

    def test_agent_solutions_hive_partitioned(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_agent_plan(5, "agent1", make_plan("agent1", 5))
            expected = (
                store.run_dir
                / AGENT_SOLUTIONS_DIR
                / "agent_id=agent1"
                / "iter_000005.parquet"
            )
            assert expected.exists()
            store.close()

    def test_multiple_agents_separate_partitions(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for aid in ["agent1", "agent2", "agent3"]:
                store.write_agent_plan(0, aid, make_plan(aid, 0))

            sol_dir = store.run_dir / AGENT_SOLUTIONS_DIR
            partitions = sorted(d.name for d in sol_dir.iterdir() if d.is_dir())
            assert partitions == [
                "agent_id=agent1",
                "agent_id=agent2",
                "agent_id=agent3",
            ]
            store.close()

    def test_iteration_zero_padding(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for it in [0, 1, 10, 100, 1000]:
                store.write_state(it, make_state(it), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            assert (conv_dir / "iter_000000.parquet").exists()
            assert (conv_dir / "iter_000001.parquet").exists()
            assert (conv_dir / "iter_000010.parquet").exists()
            assert (conv_dir / "iter_000100.parquet").exists()
            assert (conv_dir / "iter_001000.parquet").exists()
            store.close()


class TestWriteReadRoundTrip:
    """Tests for write/read round-trip data integrity."""

    def test_convergence_round_trip_with_residuals(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_state(0, make_state(0, primal=1.5, dual=0.3), timestamp=_TS)

            result = store.read_state(0)
            assert result is not None
            assert result["iteration"] == 0
            assert result["primal_residual"] == pytest.approx(1.5)
            assert result["dual_residual"] == pytest.approx(0.3)
            store.close()

    def test_convergence_round_trip_without_residuals(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_state(0, make_state_no_residuals(0), timestamp=_TS)

            result = store.read_state(0)
            assert result is not None
            assert result["iteration"] == 0
            assert result["primal_residual"] is None
            assert result["dual_residual"] is None
            store.close()

    def test_consensus_vars_round_trip(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            state = make_state(7)
            store.write_state(7, state, timestamp=_TS)

            result = store.read_state(7)
            assert result is not None
            np.testing.assert_array_equal(
                result["consensus_vars"],
                state.get_core_state().consensus_vars,
            )
            store.close()

    def test_agent_plan_scalars_round_trip(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            plan = make_plan("agent1", 3)
            store.write_agent_plan(3, "agent1", plan)

            plans = store.read_agent_plans(3)
            assert "agent1" in plans
            record = plans["agent1"]
            assert record["iteration"] == 3
            assert record["utility"] == pytest.approx(10.0)
            assert record["subsidy"] == pytest.approx(5.0)
            assert record["proximal"] == pytest.approx(2.0)
            store.close()

    def test_agent_plan_preferred_vars_round_trip(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            plan = make_plan("agent1", 0)
            store.write_agent_plan(0, "agent1", plan)

            plans = store.read_agent_plans(0)
            record = plans["agent1"]
            # preferred_vars is a struct with group names as fields
            pv = record["preferred_vars"]
            assert "y" in pv
            np.testing.assert_array_almost_equal(pv["y"], [1.0, 2.0, 3.0])
            store.close()

    def test_agent_plan_struct_with_multiple_groups(self):
        """Struct fields are sorted by group name; direct field access works."""
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            plan = AgentPlan(
                agent_id="agent1",
                iteration=0,
                solution=Solution(
                    preferred_vars={
                        PublicVarGroupName("z_reserves"): np.array([9.0]),
                        PublicVarGroupName("a_energy"): np.array([1.0, 2.0]),
                    },
                    objective=Objective(utility=1.0, subsidy=0.0, proximal=0.0),
                ),
            )
            store.write_agent_plan(0, "agent1", plan)

            plans = store.read_agent_plans(0)
            pv = plans["agent1"]["preferred_vars"]
            np.testing.assert_array_almost_equal(pv["a_energy"], [1.0, 2.0])
            np.testing.assert_array_almost_equal(pv["z_reserves"], [9.0])
            store.close()

    def test_timestamp_round_trip(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_state(0, make_state(0), timestamp=1710600000.123)

            result = store.read_state(0)
            assert result is not None
            assert result["timestamp"] == pytest.approx(1710600000.123)
            store.close()

    def test_timestamp_stored_when_provided(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.write_state(0, make_state(0), timestamp=1710600000.0)

            result = store.read_state(0)
            assert result is not None
            assert result["timestamp"] == pytest.approx(1710600000.0)
            store.close()

    def test_timestamp_in_convergence_dataset(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for i in range(3):
                store.write_state(i, make_state(i), timestamp=1000.0 + i)

            table = store.read_convergence_dataset()
            assert "timestamp" in table.column_names
            assert table.column("timestamp").to_pylist() == [
                pytest.approx(1000.0),
                pytest.approx(1001.0),
                pytest.approx(1002.0),
            ]
            store.close()

    def test_timestamp_survives_compaction(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(5):
                store.write_state(i, make_state(i), timestamp=100.0 + i)
            store.close()

            for i in range(5):
                result = store.read_state(i)
                assert result is not None
                assert result["timestamp"] == pytest.approx(100.0 + i)

    def test_read_nonexistent_state(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            assert store.read_state(99) is None
            store.close()

    def test_read_nonexistent_agent_plans(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            assert store.read_agent_plans(99) == {}
            store.close()

    def test_multiple_agents_same_iteration(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for aid in ["agent1", "agent2", "agent3"]:
                store.write_agent_plan(0, aid, make_plan(aid, 0))

            plans = store.read_agent_plans(0)
            assert len(plans) == 3
            assert set(plans.keys()) == {"agent1", "agent2", "agent3"}
            store.close()


class TestDatasetReads:
    """Tests for full dataset reads via pyarrow.dataset."""

    def test_convergence_dataset(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for i in range(5):
                store.write_state(
                    i,
                    make_state(i, primal=float(i), dual=float(i) * 0.1),
                    timestamp=_TS + i,
                )

            table = store.read_convergence_dataset()
            assert table is not None
            assert table.num_rows == 5
            # Verify sorted by iteration
            iterations = table.column("iteration").to_pylist()
            assert iterations == [0, 1, 2, 3, 4]
            store.close()

    def test_convergence_dataset_empty(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            table = store.read_convergence_dataset()
            assert table is not None
            assert table.num_rows == 0
            store.close()

    def test_agent_solutions_dataset_all_agents(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for i in range(3):
                for aid in ["a1", "a2"]:
                    store.write_agent_plan(i, aid, make_plan(aid, i))

            table = store.read_agent_solutions_dataset()
            assert table is not None
            assert table.num_rows == 6  # 3 iterations × 2 agents
            store.close()

    def test_agent_solutions_dataset_filtered(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for i in range(3):
                for aid in ["a1", "a2"]:
                    store.write_agent_plan(i, aid, make_plan(aid, i))

            table = store.read_agent_solutions_dataset(agent_id="a1")
            assert table is not None
            assert table.num_rows == 3  # Only a1's iterations
            store.close()

    def test_agent_solutions_dataset_empty(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            table = store.read_agent_solutions_dataset()
            # No files written, but directory exists
            assert table is not None
            assert table.num_rows == 0
            store.close()


class TestCompaction:
    """Tests for LSM-style tiered compaction."""

    def test_l0_to_l1_triggers_at_chunk_size(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=5)
            for i in range(5):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            # L0 files should be gone, replaced by a chunk
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 0
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 1
            store.close()

    def test_l0_not_compacted_below_chunk_size(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=10)
            for i in range(5):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 5
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 0
            store.close()

    def test_multiple_l1_chunks(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(9):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            # 3 chunks of 3, no remaining L0
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 3
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 0
            store.close()

    def test_l1_chunks_with_remainder(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(7):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            # 2 chunks of 3, 1 remaining L0
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 2
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 1
            store.close()

    def test_agent_solutions_compacted_per_partition(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(3):
                for aid in ["a1", "a2"]:
                    store.write_agent_plan(i, aid, make_plan(aid, i))

            sol_dir = store.run_dir / AGENT_SOLUTIONS_DIR
            for aid in ["a1", "a2"]:
                agent_dir = sol_dir / f"agent_id={aid}"
                assert len(list(agent_dir.glob("chunk_*.parquet"))) == 1
                assert len(list(agent_dir.glob("iter_*.parquet"))) == 0
            store.close()

    def test_l2_compaction_on_close(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(7):
                store.write_state(i, make_state(i), timestamp=_TS)
            store.close()

            conv_dir = store.run_dir / CONVERGENCE_DIR
            # After close, everything merged into convergence.parquet
            assert (conv_dir / L2_CONVERGENCE_FILE).exists()
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 0
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 0

    def test_data_equivalence_across_tiers(self):
        """Data read after L2 compaction equals data written."""
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(7):
                store.write_state(
                    i,
                    make_state(i, primal=float(i), dual=float(i) * 0.1),
                    timestamp=_TS,
                )

            # Read before close (mix of L0 + L1)
            pre_close = {}
            for i in range(7):
                pre_close[i] = store.read_state(i)

            store.close()

            # Read after close (L2 only)
            for i in range(7):
                post = store.read_state(i)
                assert post is not None
                assert post["iteration"] == pre_close[i]["iteration"]
                assert post["primal_residual"] == pytest.approx(
                    pre_close[i]["primal_residual"]
                )
                assert post["dual_residual"] == pytest.approx(
                    pre_close[i]["dual_residual"]
                )

    def test_read_state_works_after_compaction(self):
        """read_state finds data in L1 chunks, not just L0 files."""
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(6):
                store.write_state(i, make_state(i, primal=float(i)), timestamp=_TS)

            # Iterations 0-2 are in chunk_000, 3-5 in chunk_001
            for i in range(6):
                result = store.read_state(i)
                assert result is not None, f"Iteration {i} not found after compaction"
                assert result["iteration"] == i
            store.close()

    def test_read_agent_plans_works_after_compaction(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(6):
                store.write_agent_plan(i, "a1", make_plan("a1", i))

            for i in range(6):
                plans = store.read_agent_plans(i)
                assert "a1" in plans, f"Agent plan for iteration {i} not found"
            store.close()

    def test_dataset_reads_span_tiers(self):
        """Dataset reads transparently span L0 + L1 files."""
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(5):
                store.write_state(i, make_state(i), timestamp=_TS)

            # 1 chunk (iters 0-2) + 2 L0 files (iters 3-4)
            table = store.read_convergence_dataset()
            assert table is not None
            assert table.num_rows == 5
            assert table.column("iteration").to_pylist() == [0, 1, 2, 3, 4]
            store.close()

    def test_l2_agent_solutions_compaction(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(5):
                store.write_agent_plan(i, "a1", make_plan("a1", i))
            store.close()

            agent_dir = store.run_dir / AGENT_SOLUTIONS_DIR / "agent_id=a1"
            assert (agent_dir / L2_AGENT_SOLUTIONS_FILE).exists()
            assert len(list(agent_dir.glob("chunk_*.parquet"))) == 0
            assert len(list(agent_dir.glob("iter_*.parquet"))) == 0

    def test_resume_recovers_l0_counts(self):
        """Resuming into a run_dir with existing L0 files triggers compaction correctly."""
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="c", run_id="r")

            # First backend writes 2 L0 files then "crashes" (no close)
            store1 = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=5
            )
            for i in range(2):
                store1.write_state(i, make_state(i), timestamp=_TS)
            # Simulate crash — don't close

            # Second backend resumes into the same directory
            store2 = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=5
            )
            # Write 3 more — total 5 L0 files should trigger compaction
            for i in range(2, 5):
                store2.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store2.run_dir / CONVERGENCE_DIR
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 1
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 0

            # All 5 iterations readable
            for i in range(5):
                assert store2.read_state(i) is not None
            store2.close()

    def test_resume_recovers_iteration_count(self):
        """Resumed backend reports correct final_iteration in manifest."""
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="c", run_id="r")

            store1 = FileSystemBackend(base_dir=base_dir, identity=identity)
            for i in range(10):
                store1.write_state(i, make_state(i), timestamp=_TS)
            # Simulate crash

            store2 = FileSystemBackend(base_dir=base_dir, identity=identity)
            for i in range(10, 15):
                store2.write_state(i, make_state(i), timestamp=_TS)
            store2.close()

            manifest = read_manifest(store2.run_dir)
            assert manifest["final_iteration"] == 14


class TestCrashRecovery:
    """Tests for marker-based crash recovery and dedup safety net."""

    def test_marker_cleaned_up_after_l0_to_l1(self):
        """No marker file left after successful L0→L1 compaction."""
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(3):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            assert not (conv_dir / COMPACTION_MARKER).exists()
            store.close()

    def test_marker_cleaned_up_after_l2(self):
        """No marker file left after successful L2 compaction."""
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir, chunk_size=3)
            for i in range(3):
                store.write_state(i, make_state(i), timestamp=_TS)
            store.close()

            conv_dir = store.run_dir / CONVERGENCE_DIR
            assert not (conv_dir / COMPACTION_MARKER).exists()

    def test_recover_incomplete_l2_compaction(self):
        """Simulate crash after L2 write but before source deletion."""
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="c", run_id="r")
            store = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=5
            )
            for i in range(5):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            # At this point we have 1 chunk file (L1)
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 1

            # Simulate a crash during L2 compaction:
            # Manually create L2 file + marker + leave chunk behind
            import pyarrow.parquet as pq

            chunk_file = list(conv_dir.glob("chunk_*.parquet"))[0]
            table = pq.read_table(chunk_file)
            pq.write_table(table, conv_dir / L2_CONVERGENCE_FILE)
            (conv_dir / COMPACTION_MARKER).touch()
            # chunk file still exists — simulates crash before delete

            # Resume: new backend should clean up
            store2 = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=5
            )
            assert not (conv_dir / COMPACTION_MARKER).exists()
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 0
            assert (conv_dir / L2_CONVERGENCE_FILE).exists()

            # Data still readable
            for i in range(5):
                assert store2.read_state(i) is not None
            store2.close()

    def test_recover_incomplete_l0_to_l1_compaction(self):
        """Simulate crash after L1 chunk write but before L0 deletion."""
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="c", run_id="r")
            store = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=10
            )
            for i in range(3):
                store.write_state(i, make_state(i), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 3

            # Simulate crash during L0→L1: chunk written, marker written, L0 not deleted
            import pyarrow as pa
            import pyarrow.parquet as pq

            tables = [pq.read_table(f) for f in sorted(conv_dir.glob("iter_*.parquet"))]
            merged = pa.concat_tables(tables)
            pq.write_table(merged, conv_dir / "chunk_000.parquet")
            (conv_dir / COMPACTION_MARKER).touch()

            # Resume: should clean up L0 files
            store2 = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=10
            )
            assert not (conv_dir / COMPACTION_MARKER).exists()
            assert len(list(conv_dir.glob("iter_*.parquet"))) == 0
            assert len(list(conv_dir.glob("chunk_*.parquet"))) == 1

            for i in range(3):
                assert store2.read_state(i) is not None
            store2.close()

    def test_dedup_removes_duplicate_iterations(self):
        """_dedup_table removes duplicate rows by iteration."""
        import pyarrow as pa

        table = pa.table(
            {
                "iteration": [0, 1, 1, 2, 2, 2, 3],
                "value": [10, 20, 21, 30, 31, 32, 40],
            }
        )
        result = _dedup_table(table)
        assert result.num_rows == 4
        assert result.column("iteration").to_pylist() == [0, 1, 2, 3]
        # Last occurrence wins
        assert result.column("value").to_pylist() == [10, 21, 32, 40]

    def test_dedup_noop_when_no_duplicates(self):
        """_dedup_table returns same table when no duplicates."""
        import pyarrow as pa

        table = pa.table({"iteration": [0, 1, 2], "value": [10, 20, 30]})
        result = _dedup_table(table)
        assert result is table  # same object, fast path

    def test_dedup_multi_key(self):
        """_dedup_table with multiple key columns."""
        import pyarrow as pa

        table = pa.table(
            {
                "iteration": [0, 0, 1, 1],
                "agent_id": ["a", "b", "a", "a"],
                "value": [10, 20, 30, 31],
            }
        )
        result = _dedup_table(table, key_columns=("iteration", "agent_id"))
        assert result.num_rows == 3
        # (0,"a")→10, (0,"b")→20, (1,"a")→31 (last occurrence)
        assert result.column("value").to_pylist() == [10, 20, 31]

    def test_dataset_read_dedup_after_simulated_crash(self):
        """Full dataset read deduplicates if crash left duplicate files."""
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="c", run_id="r")
            store = FileSystemBackend(
                base_dir=base_dir, identity=identity, chunk_size=10
            )
            for i in range(5):
                store.write_state(i, make_state(i, primal=float(i)), timestamp=_TS)

            conv_dir = store.run_dir / CONVERGENCE_DIR
            # Simulate crash: duplicate the L0 files into a chunk without deleting them
            # (no marker, so recovery won't clean up — tests the dedup safety net)
            import pyarrow as pa
            import pyarrow.parquet as pq

            tables = [pq.read_table(f) for f in sorted(conv_dir.glob("iter_*.parquet"))]
            merged = pa.concat_tables(tables)
            pq.write_table(merged, conv_dir / "chunk_000.parquet")
            # L0 files still exist → duplicates

            table = store.read_convergence_dataset()
            assert table is not None
            assert table.num_rows == 5  # dedup removed duplicates
            assert table.column("iteration").to_pylist() == [0, 1, 2, 3, 4]
            store.close()


class TestMetadata:
    """Tests for problem metadata persistence."""

    def _make_registry(self):
        """Build a finalized registry with 2 agents and 2 variable groups."""
        import pandas as pd
        from flo_pro_sdk.core.registry import AgentRegistry
        from flo_pro_sdk.core.variables import (
            PublicVarGroupMetadata,
            PublicVarGroupName,
        )

        registry = AgentRegistry()

        # Agent a1 subscribes to energy (rows 0,1) and reserves (row 0)
        energy_meta_a1 = PublicVarGroupMetadata(
            name=PublicVarGroupName("energy"),
            var_metadata=pd.DataFrame({"t": [0, 1], "node": ["n1", "n2"]}),
        )
        reserves_meta_a1 = PublicVarGroupMetadata(
            name=PublicVarGroupName("reserves"),
            var_metadata=pd.DataFrame({"t": [0], "node": ["n1"]}),
        )
        registry.register_agent(
            "a1",
            {
                PublicVarGroupName("energy"): energy_meta_a1,
                PublicVarGroupName("reserves"): reserves_meta_a1,
            },
            metadata={"type": "generator"},
        )

        # Agent a2 subscribes to energy (rows 1,2) — overlaps with a1 on row 1
        energy_meta_a2 = PublicVarGroupMetadata(
            name=PublicVarGroupName("energy"),
            var_metadata=pd.DataFrame({"t": [1, 2], "node": ["n2", "n3"]}),
        )
        registry.register_agent(
            "a2",
            {PublicVarGroupName("energy"): energy_meta_a2},
            metadata={"type": "load"},
        )

        registry.finalize_registration()
        return registry

    def test_write_and_read_problem_json(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            registry = self._make_registry()
            store.write_metadata(registry)

            metadata = store.read_metadata()
            assert metadata is not None

            # Agents
            assert len(metadata["agents"]) == 2
            agent_ids = [a["agent_id"] for a in metadata["agents"]]
            assert "a1" in agent_ids
            assert "a2" in agent_ids

            # Variable groups
            groups = {g["name"]: g for g in metadata["variable_groups"]}
            assert "energy" in groups
            assert "reserves" in groups
            assert groups["energy"]["count"] == 3  # 3 unique rows after merge
            assert groups["reserves"]["count"] == 1
            assert metadata["total_variable_count"] == 4  # 3 energy + 1 reserves

            # Subscriptions
            subs = metadata["subscriptions"]
            assert "a1" in subs
            assert "a2" in subs
            assert "energy" in subs["a1"]
            assert "reserves" in subs["a1"]
            assert "energy" in subs["a2"]
            assert "reserves" not in subs["a2"]

            store.close()

    def test_var_metadata_parquet_round_trip(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            registry = self._make_registry()
            store.write_metadata(registry)

            metadata = store.read_metadata()
            vm = metadata["var_metadata"]
            assert "energy" in vm
            assert "reserves" in vm

            # Energy group has 3 unique rows after merge
            energy_df = vm["energy"]
            assert len(energy_df) == 3
            assert set(energy_df.columns) == {"t", "node"}

            # Reserves group has 1 row
            reserves_df = vm["reserves"]
            assert len(reserves_df) == 1

            store.close()

    def test_var_metadata_directory_structure(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            registry = self._make_registry()
            store.write_metadata(registry)

            vm_dir = store.run_dir / METADATA_DIR / VAR_METADATA_DIR
            assert (vm_dir / "group_name=energy" / VAR_METADATA_FILE).exists()
            assert (vm_dir / "group_name=reserves" / VAR_METADATA_FILE).exists()
            store.close()

    def test_read_metadata_before_write_returns_none(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            assert store.read_metadata() is None
            store.close()

    def test_subscription_indices_are_valid(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            registry = self._make_registry()
            store.write_metadata(registry)

            metadata = store.read_metadata()
            subs = metadata["subscriptions"]

            # a1 subscribes to energy indices within [0, 3) and reserves within [0, 1)
            energy_group = next(
                g for g in metadata["variable_groups"] if g["name"] == "energy"
            )
            for idx in subs["a1"]["energy"]:
                assert 0 <= idx < energy_group["count"]

            reserves_group = next(
                g for g in metadata["variable_groups"] if g["name"] == "reserves"
            )
            for idx in subs["a1"]["reserves"]:
                assert 0 <= idx < reserves_group["count"]

            store.close()

    def test_agent_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            registry = self._make_registry()
            store.write_metadata(registry)

            metadata = store.read_metadata()
            agents_by_id = {a["agent_id"]: a for a in metadata["agents"]}
            assert agents_by_id["a1"]["metadata"] == {"type": "generator"}
            assert agents_by_id["a2"]["metadata"] == {"type": "load"}
            store.close()


class TestLifecycle:
    """Tests for close, manifest, and error handling."""

    def test_close_writes_completed_manifest(self):
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="coord_x", run_id="test_run")
            store = FileSystemBackend(base_dir=base_dir, identity=identity)
            store.close()

            manifest = read_manifest(store.run_dir)
            assert manifest is not None
            assert manifest["run_id"] == "test_run"
            assert manifest["coordination_id"] == "coord_x"
            assert manifest["status"] == "completed"
            assert "completed_at" in manifest
            assert "started_at" in manifest

    def test_manifest_running_on_init(self):
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="c", run_id="r")
            store = FileSystemBackend(base_dir=base_dir, identity=identity)

            manifest = read_manifest(store.run_dir)
            assert manifest["status"] == "running"
            store.close()

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.close()
            store.close()  # Should not raise

    def test_write_after_close_raises(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.close()

            with pytest.raises(RuntimeError):
                store.write_state(0, make_state(0), timestamp=_TS)

            with pytest.raises(RuntimeError):
                store.write_agent_plan(0, "a", make_plan("a", 0))

    def test_write_metadata_after_close_raises(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            store.close()

            import pandas as pd
            from flo_pro_sdk.core.registry import AgentRegistry
            from flo_pro_sdk.core.variables import (
                PublicVarGroupMetadata,
                PublicVarGroupName,
            )

            registry = AgentRegistry()
            registry.register_agent(
                "a1",
                {
                    PublicVarGroupName("x"): PublicVarGroupMetadata(
                        name=PublicVarGroupName("x"),
                        var_metadata=pd.DataFrame({"t": [0]}),
                    )
                },
            )
            registry.finalize_registration()

            with pytest.raises(RuntimeError):
                store.write_metadata(registry)

    def test_final_iteration_in_manifest(self):
        with tempfile.TemporaryDirectory() as base_dir:
            store = FileSystemBackend(base_dir=base_dir)
            for i in range(10):
                store.write_state(i, make_state(i), timestamp=_TS)
            store.close()

            manifest = read_manifest(store.run_dir)
            assert manifest["final_iteration"] == 9

    def test_resumed_from_in_manifest(self):
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(
                coordination_id="coord_x",
                run_id="run_2",
                resumed_from="run_1",
            )
            store = FileSystemBackend(base_dir=base_dir, identity=identity)
            store.close()

            manifest = read_manifest(store.run_dir)
            assert manifest["resumed_from"] == "run_1"


class TestCoordinationRun:
    """Tests for CoordinationRun dataclass and manifest utilities."""

    def test_auto_generated_ids(self):
        identity = CoordinationRun()
        assert identity.coordination_id.startswith("coord_")
        assert len(identity.run_id) == 22

    def test_custom_ids(self):
        identity = CoordinationRun(coordination_id="my_coord", run_id="my_run")
        assert identity.coordination_id == "my_coord"
        assert identity.run_id == "my_run"

    def test_run_dir_structure(self):
        identity = CoordinationRun(coordination_id="c1", run_id="r1")
        base = Path("/tmp/test")
        assert identity.run_dir(base) == base / "c1" / "r1"

    def test_directory_structure_with_identity(self):
        with tempfile.TemporaryDirectory() as base_dir:
            identity = CoordinationRun(coordination_id="energy_q1", run_id="attempt_1")
            store = FileSystemBackend(base_dir=base_dir, identity=identity)

            expected = Path(base_dir) / "energy_q1" / "attempt_1"
            assert store.run_dir == expected
            assert expected.exists()
            store.close()

    def test_multiple_runs_same_coordination(self):
        with tempfile.TemporaryDirectory() as base_dir:
            stores = []
            for i in range(3):
                identity = CoordinationRun(
                    coordination_id="shared_coord", run_id=f"run_{i}"
                )
                stores.append(FileSystemBackend(base_dir=base_dir, identity=identity))

            coord_dir = Path(base_dir) / "shared_coord"
            assert coord_dir.exists()
            assert len(list(coord_dir.iterdir())) == 3

            for s in stores:
                s.close()


class TestIntegration:
    """Integration test with PersistenceWriter pipeline."""

    def test_with_persisting_store_wrapper(self):
        from flo_pro_sdk.core.in_memory_state_store import InMemoryStateStore
        from flo_pro_sdk.core.persistence import (
            PersistenceWriter,
            PersistingStoreWrapper,
        )

        with tempfile.TemporaryDirectory() as base_dir:
            backend = FileSystemBackend(base_dir=base_dir)
            store = InMemoryStateStore(cache_size=3)
            writer = PersistenceWriter(backend, InMemoryStateStore())
            wrapper = PersistingStoreWrapper(store, writer.queue)

            for i in range(5):
                wrapper.store_state(i, make_state(i), timestamp=_TS + i)
                wrapper.store_agent_plan(i, f"agent{i}", make_plan(f"agent{i}", i))

            writer.finalize()

            # After finalize, L2 compaction merges everything
            assert (backend.run_dir / CONVERGENCE_DIR / L2_CONVERGENCE_FILE).exists()
            assert (
                backend.run_dir / CONSENSUS_VARS_DIR / L2_CONSENSUS_VARS_FILE
            ).exists()

            # Verify all data is readable via dataset reads
            for i in range(5):
                state = backend.read_state(i)
                assert state is not None, f"State {i} not readable after compaction"

            assert (backend.run_dir / "manifest.json").exists()
