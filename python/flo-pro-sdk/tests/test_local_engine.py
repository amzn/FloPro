"""Tests for the local execution engine."""

import tempfile

import numpy as np

from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.lifecycle import ProblemRunner
from flo_pro_sdk.core.problem import Problem
from flo_pro_sdk.core.query import DefaultQueryStrategy
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.state_store import StoreConfig
from flo_pro_sdk.core.persistence_backend import FileSystemBackend
from flo_pro_sdk.core.variables import PublicVarGroupName
from flo_pro_sdk.engine.local import LocalExecutionEngine
from flo_pro_sdk.testing import MockAgentDefinition, MockCoordinatorDefinition

G = PublicVarGroupName("g")


def _consensus_state(
    iteration: int = 0, dim: int = 2, agent_id: str = "agent1"
) -> ConsensusState:
    """Helper to build a ConsensusState for tests."""
    return ConsensusState(
        iteration=iteration,
        consensus_vars=np.zeros(dim),
        agent_preferred_vars={agent_id: np.zeros(dim)},
        prices={agent_id: np.zeros(dim)},
        rho={agent_id: np.ones(dim)},
    )


class TestLocalExecutionEngine:
    """Tests for LocalExecutionEngine."""

    def test_allocate_agents(self) -> None:
        engine = LocalExecutionEngine()
        specs = [
            AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1"),
            AgentSpec(agent_class=MockAgentDefinition, agent_id="agent2"),
        ]
        engine.allocate_agents(specs)
        assert len(engine._agent_handles) == 2

    def test_registration_executor(self) -> None:
        engine = LocalExecutionEngine()
        specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(specs)

        executor = engine.get_registration_executor()
        results = executor.execute(["agent1"])

        assert "agent1" in results
        assert isinstance(results["agent1"], dict)

    def test_query_executor(self) -> None:
        engine = LocalExecutionEngine()
        specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(specs)

        from flo_pro_sdk.core.query import AgentInput

        def _input_fn(agent_id, state):
            return AgentInput(
                agent_targets={G: np.array([0.0, 0.0])},
                prices={G: np.array([0.0, 0.0])},
                rho={G: np.array([1.0, 1.0])},
            )

        executor = engine.get_query_executor()
        results = executor.execute(
            agent_ids=["agent1"],
            state=_consensus_state(agent_id="agent1"),
            get_agent_input_fn=_input_fn,
        )

        assert "agent1" in results
        assert results["agent1"].solution.objective.utility == 10.0

    def test_finalization_executor(self) -> None:
        engine = LocalExecutionEngine()
        specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(specs)

        final_state = _consensus_state(iteration=5)

        executor = engine.get_finalization_executor()
        executor.execute(["agent1"], final_state)

    def test_allocate_coordinator(self) -> None:
        engine = LocalExecutionEngine()
        specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(specs)

        registry = AgentRegistry()
        reg_executor = engine.get_registration_executor()
        results = reg_executor.execute(["agent1"])
        registry.register_agent("agent1", results["agent1"])
        registry.finalize_registration()

        coord_spec = CoordinatorSpec(coordinator_class=MockCoordinatorDefinition)
        handler = engine.allocate_coordinator(
            coordinator_spec=coord_spec,
            query_strategy=DefaultQueryStrategy(),
            query_executor=engine.get_query_executor(),
            registry=registry,
        )

        initial_state = _consensus_state()

        engine.get_state_store().store_state(0, initial_state)

        result = handler.run_iteration(0)
        assert result.iteration == 1
        assert not result.converged

    def test_shutdown(self) -> None:
        engine = LocalExecutionEngine()
        specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(specs)

        assert len(engine._agent_handles) == 1
        engine.shutdown()
        assert len(engine._agent_handles) == 0


class TestLocalEngineIntegration:
    """Integration tests for local engine with ProblemRunner."""

    def test_full_problem_run(self) -> None:
        engine = LocalExecutionEngine()
        initial_state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={"agent1": np.zeros(2), "agent2": np.zeros(2)},
            prices={"agent1": np.zeros(2), "agent2": np.zeros(2)},
            rho={"agent1": np.ones(2), "agent2": np.ones(2)},
        )

        problem = Problem(
            agents=[
                AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1"),
                AgentSpec(agent_class=MockAgentDefinition, agent_id="agent2"),
            ],
            coordinator=CoordinatorSpec(coordinator_class=MockCoordinatorDefinition),
            initial_state=initial_state,
            max_iterations=10,
        )

        runner = ProblemRunner(problem, engine)
        final_state = runner.run()

        assert final_state.iteration == 3


class TestLocalEngineStoreConfig:
    """Tests for LocalExecutionEngine store configuration."""

    def test_default_store_config_creates_store(self) -> None:
        engine = LocalExecutionEngine()
        assert engine.get_state_store() is not None

    def test_store_config_with_no_backend_creates_store(self) -> None:
        """In-memory only tracking with StoreConfig but no PersistenceBackend."""
        config = StoreConfig(cache_size=10)
        engine = LocalExecutionEngine(store_config=config)
        store = engine.get_state_store()
        assert store is not None

    def test_store_config_with_filesystem_backend(self) -> None:
        """Full persistence with FileSystemBackend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemBackend(base_dir=tmpdir)
            config = StoreConfig(persistence_backend=backend, cache_size=5)
            engine = LocalExecutionEngine(store_config=config)
            store = engine.get_state_store()
            assert store is not None
            engine.shutdown()

    def test_store_stores_and_retrieves_state(self) -> None:
        """Store can store and retrieve state."""
        config = StoreConfig(cache_size=5)
        engine = LocalExecutionEngine(store_config=config)
        store = engine.get_state_store()
        assert store is not None

        state = _consensus_state()

        store.store_state(0, state)
        retrieved = store.get_state(0)

        assert retrieved is not None
        assert retrieved.iteration == 0

    def test_persistence_via_queue_integration(self) -> None:
        """Test that persistence works via threading queue in a full problem run."""
        from flo_pro_sdk.core.persistence_backend import (
            CONVERGENCE_DIR,
            AGENT_SOLUTIONS_DIR,
            L2_CONVERGENCE_FILE,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileSystemBackend(base_dir=tmpdir)
            config = StoreConfig(persistence_backend=backend, cache_size=3)
            engine = LocalExecutionEngine(store_config=config)

            initial_state = ConsensusState(
                iteration=0,
                consensus_vars=np.zeros(2),
                agent_preferred_vars={"agent1": np.zeros(2), "agent2": np.zeros(2)},
                prices={"agent1": np.zeros(2), "agent2": np.zeros(2)},
                rho={"agent1": np.ones(2), "agent2": np.ones(2)},
            )

            problem = Problem(
                agents=[
                    AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1"),
                    AgentSpec(agent_class=MockAgentDefinition, agent_id="agent2"),
                ],
                coordinator=CoordinatorSpec(
                    coordinator_class=MockCoordinatorDefinition
                ),
                initial_state=initial_state,
                max_iterations=10,
            )

            runner = ProblemRunner(problem, engine)
            final_state = runner.run()

            # Verify the problem ran to convergence
            assert final_state.iteration == 3

            # After finalize, L2 compaction merges into convergence.parquet
            assert (backend.run_dir / CONVERGENCE_DIR / L2_CONVERGENCE_FILE).exists()

            # Verify we can read back the persisted states
            for i in range(4):
                persisted_state = backend.read_state(i)
                assert persisted_state is not None, f"State {i} was not persisted"
                assert persisted_state["iteration"] == i

            # Verify agent solution files were also persisted
            sol_dir = backend.run_dir / AGENT_SOLUTIONS_DIR
            agent_dirs = [d for d in sol_dir.iterdir() if d.is_dir()]
            assert len(agent_dirs) >= 1, "Expected at least 1 agent partition directory"

            # Verify manifest was written
            manifest_file = backend.run_dir / "manifest.json"
            assert manifest_file.exists(), "Manifest file was not created"
