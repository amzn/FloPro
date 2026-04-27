"""Tests for agent/coordinator params and factory methods."""

import numpy as np
import pytest

from flo_pro_sdk.agent.agent_definition import AgentDefinition, AgentSpec, Solution
from flo_pro_sdk.coordinator.coordinator_definition import (
    CoordinatorDefinition,
    CoordinatorSpec,
)
from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.query import AgentInput
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.variables import (
    PublicVarGroupMetadata,
    PublicVarGroupName,
)
from flo_pro_sdk.core.var_layout import VarLayout

G = PublicVarGroupName("g")


def _test_layout() -> VarLayout:
    layout = VarLayout(group_slices={G: slice(0, 1)}, total_size=1)
    layout.register_agent("a1", {G: np.array([0])})
    return layout


def _var_meta(size: int) -> "PublicVarGroupMetadata":
    import pandas as pd
    from flo_pro_sdk.core.variables import PublicVarGroupMetadata

    return PublicVarGroupMetadata(
        name=G, var_metadata=pd.DataFrame({"idx": range(size)})
    )


def _test_registry() -> AgentRegistry:
    """Build a registry with one agent for executor unit tests."""
    reg = AgentRegistry()
    reg.register_agent("a1", {G: _var_meta(1)})
    reg.finalize_registration()
    return reg


# ── Concrete test subclasses ───────────────────────────────────────────


class SimpleAgent(AgentDefinition):
    """Agent with no constructor params."""

    def solve(self, public_vars, prices, rho):
        return Solution(preferred_vars={G: np.array([1.0])}, objective=0.0)

    def register(self):
        return {G: _var_meta(1)}


class ParameterizedAgent(AgentDefinition):
    """Agent that accepts constructor params."""

    def __init__(self, alpha: float = 1.0, name: str = "default"):
        self.alpha = alpha
        self.name = name

    def solve(self, public_vars, prices, rho):
        return Solution(preferred_vars={G: np.array([self.alpha])}, objective=0.0)

    def register(self):
        return {G: _var_meta(1)}


class CustomFactoryAgent(AgentDefinition):
    """Agent that overrides create() for custom construction logic."""

    def __init__(self, value: float):
        self.value = value

    @classmethod
    def create(cls, agent_params):
        raw = agent_params.get("raw_value", 0)
        return cls(value=raw * 2)

    def solve(self, public_vars, prices, rho):
        return Solution(preferred_vars={G: np.array([self.value])}, objective=0.0)

    def register(self):
        return {G: _var_meta(1)}


class SimpleCoordinator(CoordinatorDefinition):
    """Coordinator with no constructor params."""

    def __init__(self, **kwargs):
        pass

    def update_state(self, agent_results, current_state, state_store=None):
        return ConsensusState(
            iteration=current_state.iteration + 1,
            consensus_vars=current_state.consensus_vars,
            agent_preferred_vars=agent_results,
            prices={aid: current_state.get_agent_prices(aid) for aid in agent_results},
            rho={aid: current_state.get_rho(aid) for aid in agent_results},
        )

    def check_convergence(self, core_state):
        return False


class ParameterizedCoordinator(CoordinatorDefinition):
    """Coordinator that accepts constructor params."""

    def __init__(
        self,
        tolerance: float = 1e-3,
        max_iter: int = 100,
        layout=None,
        structure_function=None,
    ):
        self.tolerance = tolerance
        self.max_iter = max_iter

    def update_state(self, agent_results, current_state, state_store=None):
        return ConsensusState(
            iteration=current_state.iteration + 1,
            consensus_vars=current_state.consensus_vars,
            agent_preferred_vars=agent_results,
            prices={aid: current_state.get_agent_prices(aid) for aid in agent_results},
            rho={aid: current_state.get_rho(aid) for aid in agent_results},
        )

    def check_convergence(self, core_state):
        return core_state.iteration >= self.max_iter


class CustomFactoryCoordinator(CoordinatorDefinition):
    """Coordinator that overrides create() for custom construction logic."""

    def __init__(self, scaled_tol: float):
        self.scaled_tol = scaled_tol

    @classmethod
    def create(cls, coordinator_params, layout=None, structure_function=None):
        raw = coordinator_params.get("tol", 1e-3)
        return cls(scaled_tol=raw / 10)

    def update_state(self, agent_results, current_state, state_store=None):
        return ConsensusState(
            iteration=current_state.iteration + 1,
            consensus_vars=current_state.consensus_vars,
            agent_preferred_vars=agent_results,
            prices={aid: current_state.get_agent_prices(aid) for aid in agent_results},
            rho={aid: current_state.get_rho(aid) for aid in agent_results},
        )


# ── AgentSpec tests ────────────────────────────────────────────────────


class TestAgentSpecParams:
    def test_agent_params_none_by_default(self):
        spec = AgentSpec(agent_class=SimpleAgent, agent_id="a1")
        assert spec.agent_params is None

    def test_agent_params_accepts_json_serializable(self):
        params = {"alpha": 1.5, "name": "test"}
        spec = AgentSpec(
            agent_class=ParameterizedAgent, agent_id="a1", agent_params=params
        )
        assert spec.agent_params == params

    def test_agent_params_accepts_nested_json(self):
        params = {"config": {"nested": [1, 2, True, None, "str"]}, "value": 3.14}
        spec = AgentSpec(agent_class=SimpleAgent, agent_id="a1", agent_params=params)
        assert spec.agent_params == params

    def test_agent_params_rejects_non_serializable(self):
        with pytest.raises(TypeError, match="agent_params must be JSON-serializable"):
            AgentSpec(
                agent_class=SimpleAgent,
                agent_id="a1",
                agent_params={"bad": object()},
            )

    def test_agent_params_rejects_set(self):
        with pytest.raises(TypeError, match="agent_params must be JSON-serializable"):
            AgentSpec(
                agent_class=SimpleAgent,
                agent_id="a1",
                agent_params={"bad": {1, 2, 3}},
            )

    def test_agent_params_empty_dict_is_valid(self):
        spec = AgentSpec(agent_class=SimpleAgent, agent_id="a1", agent_params={})
        assert spec.agent_params == {}


# ── CoordinatorSpec tests ──────────────────────────────────────────────


class TestCoordinatorSpecParams:
    def test_coordinator_params_none_by_default(self):
        spec = CoordinatorSpec(coordinator_class=SimpleCoordinator)
        assert spec.coordinator_params is None

    def test_coordinator_params_accepts_json_serializable(self):
        params = {"tolerance": 1e-4, "max_iter": 50}
        spec = CoordinatorSpec(
            coordinator_class=ParameterizedCoordinator, coordinator_params=params
        )
        assert spec.coordinator_params == params

    def test_coordinator_params_accepts_nested_json(self):
        params = {"config": {"nested": [1, 2, True, None, "str"]}, "value": 3.14}
        spec = CoordinatorSpec(
            coordinator_class=SimpleCoordinator, coordinator_params=params
        )
        assert spec.coordinator_params == params

    def test_coordinator_params_rejects_non_serializable(self):
        with pytest.raises(
            TypeError, match="coordinator_params must be JSON-serializable"
        ):
            CoordinatorSpec(
                coordinator_class=SimpleCoordinator,
                coordinator_params={"bad": object()},
            )

    def test_coordinator_params_rejects_set(self):
        with pytest.raises(
            TypeError, match="coordinator_params must be JSON-serializable"
        ):
            CoordinatorSpec(
                coordinator_class=SimpleCoordinator,
                coordinator_params={"bad": {1, 2, 3}},
            )

    def test_coordinator_params_empty_dict_is_valid(self):
        spec = CoordinatorSpec(
            coordinator_class=SimpleCoordinator, coordinator_params={}
        )
        assert spec.coordinator_params == {}


# ── AgentDefinition.create() tests ─────────────────────────────────────


class TestAgentDefinitionCreate:
    def test_create_with_empty_params(self):
        agent = SimpleAgent.create({})
        assert isinstance(agent, SimpleAgent)

    def test_create_passes_params_as_kwargs(self):
        agent = ParameterizedAgent.create({"alpha": 2.5, "name": "custom"})
        assert isinstance(agent, ParameterizedAgent)
        assert agent.alpha == 2.5
        assert agent.name == "custom"

    def test_create_with_partial_params_uses_defaults(self):
        agent = ParameterizedAgent.create({"alpha": 3.0})
        assert agent.alpha == 3.0
        assert agent.name == "default"

    def test_create_with_custom_factory(self):
        agent = CustomFactoryAgent.create({"raw_value": 5})
        assert isinstance(agent, CustomFactoryAgent)
        assert agent.value == 10  # 5 * 2

    def test_create_default_factory_raises_on_unknown_param(self):
        with pytest.raises(TypeError):
            SimpleAgent.create({"unknown_param": 42})


# ── CoordinatorDefinition.create() tests ───────────────────────────────


class TestCoordinatorDefinitionCreate:
    def test_create_with_empty_params(self):
        coord = SimpleCoordinator.create({}, layout=_test_layout())
        assert isinstance(coord, SimpleCoordinator)

    def test_create_passes_params_as_kwargs(self):
        coord = ParameterizedCoordinator.create(
            {"tolerance": 1e-5, "max_iter": 200}, layout=_test_layout()
        )
        assert isinstance(coord, ParameterizedCoordinator)
        assert coord.tolerance == 1e-5
        assert coord.max_iter == 200

    def test_create_with_partial_params_uses_defaults(self):
        coord = ParameterizedCoordinator.create(
            {"tolerance": 1e-6}, layout=_test_layout()
        )
        assert coord.tolerance == 1e-6
        assert coord.max_iter == 100

    def test_create_with_custom_factory(self):
        coord = CustomFactoryCoordinator.create({"tol": 1e-2}, layout=_test_layout())
        assert isinstance(coord, CustomFactoryCoordinator)
        assert coord.scaled_tol == pytest.approx(1e-3)  # 1e-2 / 10

    def test_create_default_factory_raises_on_unknown_param(self):
        with pytest.raises(TypeError):
            ParameterizedCoordinator.create(
                {"unknown_param": 42}, layout=_test_layout()
            )


# ── Integration: params flow through engine allocation ──────────────────


class TestParamsIntegration:
    """Test that params are correctly used when engines allocate agents/coordinators."""

    @staticmethod
    def _simple_input_fn(agent_id, state):
        return AgentInput(
            agent_targets={G: np.array([0.0])},
            prices={G: np.array([0.0])},
            rho={G: np.array([1.0])},
        )

    @staticmethod
    def _simple_state() -> ConsensusState:
        return ConsensusState(
            iteration=0,
            consensus_vars=np.array([0.0]),
            agent_preferred_vars={"a1": np.array([0.0])},
            prices={"a1": np.array([0.0])},
            rho={"a1": np.array([1.0])},
        )

    def test_parameterized_agent_via_local_engine(self):
        """Verify agent_params reach the agent by checking solve() output reflects alpha."""
        from flo_pro_sdk.engine.local import LocalExecutionEngine

        spec = AgentSpec(
            agent_class=ParameterizedAgent,
            agent_id="a1",
            agent_params={"alpha": 7.0, "name": "integrated"},
        )
        engine = LocalExecutionEngine()
        engine.allocate_agents([spec])

        executor = engine.get_query_executor()
        results = executor.execute(
            agent_ids=["a1"],
            state=self._simple_state(),
            get_agent_input_fn=self._simple_input_fn,
        )
        # ParameterizedAgent.solve returns array([self.alpha])
        np.testing.assert_array_equal(
            results["a1"].solution.preferred_vars[G], np.array([7.0])
        )

    def test_parameterized_coordinator_via_local_engine(self):
        """Verify coordinator_params reach the coordinator by checking convergence behavior."""
        from flo_pro_sdk.core.query import DefaultQueryStrategy
        from flo_pro_sdk.core.registry import AgentRegistry
        from flo_pro_sdk.engine.local import LocalExecutionEngine

        engine = LocalExecutionEngine()
        spec = AgentSpec(agent_class=SimpleAgent, agent_id="a1")
        engine.allocate_agents([spec])

        registry = AgentRegistry()
        reg_executor = engine.get_registration_executor()
        results = reg_executor.execute(["a1"])
        registry.register_agent("a1", results["a1"])
        registry.finalize_registration()

        coord_spec = CoordinatorSpec(
            coordinator_class=ParameterizedCoordinator,
            coordinator_params={"tolerance": 1e-6, "max_iter": 2},
        )
        handler = engine.allocate_coordinator(
            coordinator_spec=coord_spec,
            query_strategy=DefaultQueryStrategy(),
            query_executor=engine.get_query_executor(),
            registry=registry,
        )

        state = ConsensusState(
            iteration=0,
            consensus_vars=np.array([0.0]),
            agent_preferred_vars={"a1": np.array([0.0])},
            prices={"a1": np.array([0.0])},
            rho={"a1": np.array([1.0])},
        )
        # ParameterizedCoordinator converges at iteration >= max_iter
        engine.get_state_store().store_state(0, state)
        result = handler.run_iteration(0)  # iteration -> 1
        assert not result.converged
        result = handler.run_iteration(result.iteration)  # iteration -> 2
        assert result.converged

    def test_agent_params_none_uses_empty_dict(self):
        """When agent_params is None, create() receives {} and default construction works."""
        from flo_pro_sdk.engine.local import LocalExecutionEngine

        spec = AgentSpec(agent_class=SimpleAgent, agent_id="a1")  # agent_params=None
        engine = LocalExecutionEngine()
        engine.allocate_agents([spec])

        # Verify the agent works (no error during allocation or query)
        executor = engine.get_query_executor()
        results = executor.execute(
            agent_ids=["a1"],
            state=self._simple_state(),
            get_agent_input_fn=self._simple_input_fn,
        )
        assert "a1" in results

    def test_custom_factory_agent_via_local_engine(self):
        """Verify custom create() factory is invoked by checking solve() output."""
        from flo_pro_sdk.engine.local import LocalExecutionEngine

        spec = AgentSpec(
            agent_class=CustomFactoryAgent,
            agent_id="a1",
            agent_params={"raw_value": 3},
        )
        engine = LocalExecutionEngine()
        engine.allocate_agents([spec])

        executor = engine.get_query_executor()
        results = executor.execute(
            agent_ids=["a1"],
            state=self._simple_state(),
            get_agent_input_fn=self._simple_input_fn,
        )
        # CustomFactoryAgent.create doubles raw_value; solve returns array([self.value])
        np.testing.assert_array_equal(
            results["a1"].solution.preferred_vars[G], np.array([6.0])
        )
