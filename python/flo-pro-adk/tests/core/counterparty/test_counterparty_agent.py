"""Tests for CounterpartyAgent."""

from __future__ import annotations

import pytest

from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.variables import PublicVarValues, Prices, RhoValues

from flo_pro_adk.core.counterparty.counterparty_agent import (
    CounterpartyAgent,
    _data_loader_factories,
    _var_metadata_registry,
)
from flo_pro_adk.core.data.in_memory_data_loader import InMemoryDataLoader
from flo_pro_adk.core.exceptions.agent_errors import (
    RegistrationError,
)
from flo_pro_adk.core.solver.solver_strategy import SolverStrategy

_TEST_METADATA = {"test_group": "test_metadata"}


class _DummySolver(SolverStrategy):
    """A no-op solver so the ABC can be exercised without a real solver backend."""

    def create_model(  # type: ignore[override]
        self, consensus, prices, rho, *, public_group_metadata,
        sense=None, var_lb=0.0, var_ub=None,
    ):
        raise NotImplementedError("DummySolver is not meant to be invoked")


class _Stub(CounterpartyAgent):
    @classmethod
    def _default_solver(cls) -> SolverStrategy:
        return _DummySolver()

    def solve(self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues) -> Solution:
        return Solution(
            preferred_vars={g: v.copy() for g, v in public_vars.items()},
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        )


@pytest.fixture(autouse=True)
def _register_stub_metadata():
    """Most tests need _Stub in the registry to construct it."""
    _var_metadata_registry[_Stub] = _TEST_METADATA  # type: ignore[assignment]
    yield
    _var_metadata_registry.pop(_Stub, None)


def test_init_with_params():
    agent = _Stub(agent_params={"key": "value"})
    assert agent.agent_params == {"key": "value"}
    assert agent.data is None


def test_init_with_data_loader():
    agent = _Stub(agent_params={}, data_loader=InMemoryDataLoader({"costs": [1, 2, 3]}))
    assert agent.data == {"costs": [1, 2, 3]}


def test_solver_defaults_to_class_default():
    # __init__ falls back to cls._default_solver() when solver is not passed.
    # For _Stub this returns _DummySolver (no xpress required).
    assert isinstance(_Stub(agent_params={}).solver, _DummySolver)


def test_solver_can_be_overridden_at_construction():
    custom = _DummySolver()
    agent = _Stub(agent_params={}, solver=custom)
    assert agent.solver is custom


def test_base_class_raises_without_default_solver():
    """Subclasses that don't override _default_solver must receive solver explicitly."""

    class _NoDefault(CounterpartyAgent):
        def solve(self, public_vars, prices, rho):
            raise NotImplementedError

    _var_metadata_registry[_NoDefault] = _TEST_METADATA  # type: ignore[assignment]
    try:
        with pytest.raises(NotImplementedError, match="_default_solver"):
            _NoDefault(agent_params={})
    finally:
        _var_metadata_registry.pop(_NoDefault, None)


def test_create_without_factory():
    agent = _Stub.create({"key": "value"})
    assert agent.agent_params == {"key": "value"}
    assert agent.data is None


def test_create_with_factory():
    _data_loader_factories[_Stub] = lambda: InMemoryDataLoader({"x": 42})
    try:
        assert _Stub.create({}).data == {"x": 42}
    finally:
        _data_loader_factories.pop(_Stub, None)


def test_factory_is_per_class():
    class _Other(CounterpartyAgent):
        @classmethod
        def _default_solver(cls) -> SolverStrategy:
            return _DummySolver()

        def solve(self, public_vars, prices, rho):
            raise NotImplementedError

    _var_metadata_registry[_Other] = _TEST_METADATA  # type: ignore[assignment]
    _data_loader_factories[_Stub] = lambda: InMemoryDataLoader({"x": 42})
    try:
        assert _Other.create({}).data is None
    finally:
        _data_loader_factories.pop(_Stub, None)
        _var_metadata_registry.pop(_Other, None)


def test_init_raises_without_metadata():
    _var_metadata_registry.pop(_Stub, None)  # remove autouse fixture's entry
    with pytest.raises(RegistrationError, match="No variable metadata registered for _Stub"):
        _Stub(agent_params={})


def test_register_returns_metadata():
    agent = _Stub(agent_params={})
    result = agent.register()
    assert result is _TEST_METADATA
    assert agent.public_vars_metadata is _TEST_METADATA


def test_metadata_available_without_register():
    """public_vars_metadata works immediately after construction — no register() needed."""
    agent = _Stub(agent_params={})
    assert agent.public_vars_metadata is _TEST_METADATA


def test_error_names_agent_class():
    """Error message includes the concrete class name for debuggability."""
    class _MyCustomAgent(CounterpartyAgent):
        @classmethod
        def _default_solver(cls) -> SolverStrategy:
            return _DummySolver()

        def solve(self, public_vars, prices, rho):
            raise NotImplementedError

    with pytest.raises(RegistrationError, match="_MyCustomAgent"):
        _MyCustomAgent(agent_params={})
