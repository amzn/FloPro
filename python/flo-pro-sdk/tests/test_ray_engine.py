"""Tests for the Ray execution engine."""

import numpy as np
import pytest

# Skip all tests in this module if Ray is not available
ray = pytest.importorskip("ray", reason="Ray is not available in this environment")

from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.compute import ComputeSpec
from flo_pro_sdk.core.lifecycle import ProblemRunner
from flo_pro_sdk.core.problem import Problem
from flo_pro_sdk.core.query import DefaultQueryStrategy
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.state_store import StoreConfig
from flo_pro_sdk.core.variables import PublicVarGroupName
from flo_pro_sdk.engine.ray import RayExecutionEngine
from flo_pro_sdk.testing import (
    FailingAgentDefinition,
    FailingCoordinatorDefinition,
    MockAgentDefinition,
    MockCoordinatorDefinition,
)

G = PublicVarGroupName("g")


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray once for all tests in this module."""
    if not ray.is_initialized():
        ray.init()
    yield
    ray.shutdown()


@pytest.fixture(autouse=True)
def cleanup_actors(ray_context):
    """Clean up all actors after each test to avoid name conflicts."""
    yield
    for actor in ray.util.list_named_actors():
        try:
            ray.kill(ray.get_actor(actor))
        except Exception:
            pass


class TestRayExecutionEngine:
    """Tests for RayExecutionEngine."""

    def test_init_initializes_ray(self, ray_context) -> None:
        engine = RayExecutionEngine()
        assert ray.is_initialized()

    def test_allocate_agents(self, ray_context) -> None:
        engine = RayExecutionEngine()
        specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="test-agent")]
        engine.allocate_agents(specs)

        registration_executor = engine.get_registration_executor()
        metadata = registration_executor.execute(["test-agent"])
        assert "test-agent" in metadata
        assert isinstance(metadata["test-agent"], dict)

    def test_allocate_agents_with_compute_spec(self, ray_context) -> None:
        engine = RayExecutionEngine()
        specs = [
            AgentSpec(
                agent_class=MockAgentDefinition,
                agent_id="test-agent-compute",
                compute=ComputeSpec(num_cpus=0.5),
            )
        ]
        engine.allocate_agents(specs)

        registration_executor = engine.get_registration_executor()
        metadata = registration_executor.execute(["test-agent-compute"])
        assert "test-agent-compute" in metadata

    def test_allocate_coordinator(self, ray_context) -> None:
        engine = RayExecutionEngine()
        agent_specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(agent_specs)

        registration_executor = engine.get_registration_executor()
        registration_results = registration_executor.execute(["agent1"])

        registry = AgentRegistry()
        registry.register_agent("agent1", registration_results["agent1"], {})
        registry.finalize_registration()

        coord_spec = CoordinatorSpec(coordinator_class=MockCoordinatorDefinition)
        query_strategy = DefaultQueryStrategy()
        query_executor = engine.get_query_executor()

        handler = engine.allocate_coordinator(
            coord_spec, query_strategy, query_executor, registry
        )

        initial_state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={
                "agent0": np.zeros(2),
                "agent1": np.zeros(2),
                "agent2": np.zeros(2),
            },
            prices={
                "agent0": np.zeros(2),
                "agent1": np.zeros(2),
                "agent2": np.zeros(2),
            },
            rho={"agent0": np.ones(2), "agent1": np.ones(2), "agent2": np.ones(2)},
        )

        engine.get_state_store().store_state(0, initial_state)
        engine.get_state_store().flush()

        result = handler.run_iteration(0)
        assert result.iteration == 1
        assert not result.converged

    def test_allocate_coordinator_with_compute_spec(self, ray_context) -> None:
        engine = RayExecutionEngine()
        agent_specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(agent_specs)

        registration_executor = engine.get_registration_executor()
        registration_results = registration_executor.execute(["agent1"])

        registry = AgentRegistry()
        registry.register_agent("agent1", registration_results["agent1"], {})
        registry.finalize_registration()

        coord_spec = CoordinatorSpec(
            coordinator_class=MockCoordinatorDefinition,
            compute=ComputeSpec(num_cpus=1.0),
        )
        query_strategy = DefaultQueryStrategy()
        query_executor = engine.get_query_executor()

        handler = engine.allocate_coordinator(
            coord_spec, query_strategy, query_executor, registry
        )
        assert handler is not None

    def test_get_query_executor(self, ray_context) -> None:
        engine = RayExecutionEngine()
        specs = [
            AgentSpec(agent_class=MockAgentDefinition, agent_id=f"agent{i}")
            for i in range(3)
        ]
        engine.allocate_agents(specs)

        query_executor = engine.get_query_executor()

        from flo_pro_sdk.core.query import AgentInput

        def _input_fn(agent_id, state):
            return AgentInput(
                agent_targets={G: np.array([0.0, 0.0])},
                prices={G: np.array([0.0, 0.0])},
                rho={G: np.array([1.0, 1.0])},
            )

        aids = ["agent0", "agent1", "agent2"]
        state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={a: np.zeros(2) for a in aids},
            prices={a: np.zeros(2) for a in aids},
            rho={a: np.ones(2) for a in aids},
        )
        results = query_executor.execute(
            aids, state=state, get_agent_input_fn=_input_fn
        )

        assert len(results) == 3
        for agent_id, result in results.items():
            assert result.agent_id == agent_id
            assert result.solution.objective.utility == 10.0
            assert result.query_time is not None
            assert result.query_time >= 0

    def test_allocate_agents_waits_for_initialization(self, ray_context) -> None:
        """Test that allocate_agents blocks until actors are initialized."""
        engine = RayExecutionEngine()
        specs = [
            AgentSpec(agent_class=MockAgentDefinition, agent_id=f"agent{i}")
            for i in range(3)
        ]
        engine.allocate_agents(specs)

        query_executor = engine.get_query_executor()

        from flo_pro_sdk.core.query import AgentInput

        def _input_fn(agent_id, state):
            return AgentInput(
                agent_targets={G: np.array([0.0, 0.0])},
                prices={G: np.array([0.0, 0.0])},
                rho={G: np.array([1.0, 1.0])},
            )

        aids = ["agent0", "agent1", "agent2"]
        state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={a: np.zeros(2) for a in aids},
            prices={a: np.zeros(2) for a in aids},
            rho={a: np.ones(2) for a in aids},
        )
        results = query_executor.execute(
            aids, state=state, get_agent_input_fn=_input_fn
        )
        assert len(results) == 3

    def test_allocate_agents_raises_on_init_failure(self, ray_context) -> None:
        """Test that allocate_agents raises RuntimeError when an agent fails to initialize."""
        engine = RayExecutionEngine()
        specs = [
            AgentSpec(agent_class=MockAgentDefinition, agent_id="good-agent"),
            AgentSpec(agent_class=FailingAgentDefinition, agent_id="bad-agent"),
        ]

        with pytest.raises(RuntimeError, match="Failed to initialize 1 agent"):
            engine.allocate_agents(specs)

    def test_allocate_agents_reports_all_failures(self, ray_context) -> None:
        """Test that allocate_agents reports all failed agents, not just the first."""
        engine = RayExecutionEngine()
        specs = [
            AgentSpec(agent_class=FailingAgentDefinition, agent_id="bad-agent-1"),
            AgentSpec(agent_class=FailingAgentDefinition, agent_id="bad-agent-2"),
        ]

        with pytest.raises(RuntimeError, match="Failed to initialize 2 agent"):
            engine.allocate_agents(specs)

    def test_allocate_coordinator_raises_on_init_failure(self, ray_context) -> None:
        """Test that allocate_coordinator raises RuntimeError on init failure."""
        engine = RayExecutionEngine()
        agent_specs = [AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1")]
        engine.allocate_agents(agent_specs)

        registration_executor = engine.get_registration_executor()
        registration_results = registration_executor.execute(["agent1"])

        registry = AgentRegistry()
        registry.register_agent("agent1", registration_results["agent1"], {})
        registry.finalize_registration()

        coord_spec = CoordinatorSpec(coordinator_class=FailingCoordinatorDefinition)
        query_strategy = DefaultQueryStrategy()
        query_executor = engine.get_query_executor()

        with pytest.raises(RuntimeError, match="Failed to initialize coordinator"):
            engine.allocate_coordinator(
                coord_spec, query_strategy, query_executor, registry
            )

    def test_compute_spec_to_ray_options_none(self, ray_context) -> None:
        engine = RayExecutionEngine()
        options = engine._compute_spec_to_ray_options(None)
        assert options == {}

    def test_compute_spec_to_ray_options_partial(self, ray_context) -> None:
        engine = RayExecutionEngine()
        spec = ComputeSpec(num_cpus=2.0)
        options = engine._compute_spec_to_ray_options(spec)
        assert options == {"num_cpus": 2.0}

    def test_compute_spec_to_ray_options_full(self, ray_context) -> None:
        engine = RayExecutionEngine()
        spec = ComputeSpec(num_cpus=2.0, num_gpus=1.0, memory_mb=1024)
        options = engine._compute_spec_to_ray_options(spec)
        assert options == {
            "num_cpus": 2.0,
            "num_gpus": 1.0,
            "memory": 1024 * 1024 * 1024,
        }


class TestRayEngineIntegration:
    """Integration tests for Ray engine with ProblemRunner."""

    def test_full_problem_run(self, ray_context) -> None:
        engine = RayExecutionEngine()
        initial_state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={
                "agent0": np.zeros(2),
                "agent1": np.zeros(2),
                "agent2": np.zeros(2),
            },
            prices={
                "agent0": np.zeros(2),
                "agent1": np.zeros(2),
                "agent2": np.zeros(2),
            },
            rho={"agent0": np.ones(2), "agent1": np.ones(2), "agent2": np.ones(2)},
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


class TestRayExecutionEngineStoreConfig:
    """Tests for RayExecutionEngine state store configuration."""

    def _allocate_coordinator(self, engine: RayExecutionEngine) -> None:
        """Helper to allocate a mock coordinator (and thereby initialize the store)."""
        agent_spec = AgentSpec(
            agent_class=MockAgentDefinition, agent_id="store-test-agent"
        )
        engine.allocate_agents([agent_spec])
        reg_executor = engine.get_registration_executor()
        reg_results = reg_executor.execute(["store-test-agent"])
        registry = AgentRegistry()
        registry.register_agent("store-test-agent", reg_results["store-test-agent"], {})
        registry.finalize_registration()
        coord_spec = CoordinatorSpec(coordinator_class=MockCoordinatorDefinition)
        engine.allocate_coordinator(
            coord_spec, DefaultQueryStrategy(), engine.get_query_executor(), registry
        )

    def test_default_store_created(self, ray_context) -> None:
        engine = RayExecutionEngine()
        self._allocate_coordinator(engine)
        assert engine.get_state_store() is not None

    def test_store_with_config_no_backend(self, ray_context) -> None:
        config = StoreConfig(cache_size=10)
        engine = RayExecutionEngine(store_config=config)
        self._allocate_coordinator(engine)
        store = engine.get_state_store()
        assert store is not None

    def test_store_with_filesystem_backend(self, ray_context, tmp_path) -> None:
        from flo_pro_sdk.core.persistence_backend import FileSystemBackend

        backend = FileSystemBackend(base_dir=str(tmp_path))
        config = StoreConfig(persistence_backend=backend, cache_size=5)
        engine = RayExecutionEngine(store_config=config)
        self._allocate_coordinator(engine)
        store = engine.get_state_store()
        assert store is not None
        engine.shutdown()

    def test_store_store_and_retrieve_state(self, ray_context) -> None:
        config = StoreConfig(cache_size=5)
        engine = RayExecutionEngine(store_config=config)
        self._allocate_coordinator(engine)
        store = engine.get_state_store()
        assert store is not None

        state = ConsensusState(
            iteration=0,
            consensus_vars=np.array([1.0, 2.0]),
            agent_preferred_vars={"agent0": np.zeros(2)},
            prices={"agent0": np.array([0.1, 0.2])},
            rho={"agent0": np.ones(2)},
        )

        store.store_state(0, state)
        retrieved = store.get_state(0)

        assert retrieved is not None
        assert retrieved.iteration == 0
        np.testing.assert_array_equal(retrieved.consensus_vars, np.array([1.0, 2.0]))

    def test_store_get_recent_states(self, ray_context) -> None:
        config = StoreConfig(cache_size=5)
        engine = RayExecutionEngine(store_config=config)
        self._allocate_coordinator(engine)
        store = engine.get_state_store()
        assert store is not None

        for i in range(4):
            state = ConsensusState(
                iteration=i,
                consensus_vars=np.array([float(i), float(i)]),
                agent_preferred_vars={"agent0": np.zeros(2)},
                prices={"agent0": np.zeros(2)},
                rho={"agent0": np.ones(2)},
            )
            store.store_state(i, state)

        recent = store.get_recent_states(3)

        assert len(recent) == 3
        assert recent[0].iteration == 3
        assert recent[1].iteration == 2
        assert recent[2].iteration == 1

    def test_persistence_via_queue_integration(self, ray_context, tmp_path) -> None:
        """Test that persistence works via Ray queue in a full problem run."""
        from flo_pro_sdk.core.persistence_backend import (
            FileSystemBackend,
            CONVERGENCE_DIR,
            AGENT_SOLUTIONS_DIR,
            L2_CONVERGENCE_FILE,
        )

        backend = FileSystemBackend(base_dir=str(tmp_path))
        config = StoreConfig(persistence_backend=backend, cache_size=3)
        engine = RayExecutionEngine(store_config=config)

        initial_state = ConsensusState(
            iteration=0,
            consensus_vars=np.zeros(2),
            agent_preferred_vars={
                "agent0": np.zeros(2),
                "agent1": np.zeros(2),
                "agent2": np.zeros(2),
            },
            prices={
                "agent0": np.zeros(2),
                "agent1": np.zeros(2),
                "agent2": np.zeros(2),
            },
            rho={"agent0": np.ones(2), "agent1": np.ones(2), "agent2": np.ones(2)},
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
