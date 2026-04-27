"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from flo_pro_adk.core.exceptions.vadk_error import VADKError
from flo_pro_adk.core.exceptions.solver_errors import SolverConvergenceError, SolverError
from flo_pro_adk.core.exceptions.assembly_errors import AssemblyError, DuplicateScenarioError, InvalidAssemblyError, ScenarioNotFoundError
from flo_pro_adk.core.exceptions.assertion_errors import VADKAssertionError


def test_vadk_error_base():
    err = VADKError("test")
    assert err.error_code == "VADKError"


def test_solver_convergence_inherits():
    assert issubclass(SolverConvergenceError, SolverError)
    assert issubclass(SolverError, VADKError)


@pytest.mark.parametrize("cls", [
    SolverError, AssemblyError,
    SolverConvergenceError,
    ScenarioNotFoundError, InvalidAssemblyError, DuplicateScenarioError,
])
def test_all_inherit_from_vadk_error(cls):
    assert issubclass(cls, VADKError)


@pytest.mark.parametrize("cls", [VADKAssertionError])
def test_assertion_errors_extend_builtin(cls):
    assert issubclass(cls, AssertionError)


@pytest.mark.parametrize("cls,expected", [
    (ScenarioNotFoundError, "ScenarioNotFoundError"),
    (SolverConvergenceError, "SolverConvergenceError"),
])
def test_error_code_is_class_name(cls, expected):
    err = cls("x")
    assert err.error_code == expected
