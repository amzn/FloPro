"""Tests for XpressSolverStrategy — import handling only."""

from __future__ import annotations

import pytest


def test_import_error_without_xpress():
    try:
        import xpress  # noqa: F401
        pytest.skip("xpress is installed — cannot test import error path")
    except ImportError:
        pass

    from flo_pro_adk.core.solver.xpress_solver_strategy import XpressSolverStrategy

    with pytest.raises(ImportError, match="xpress package not installed"):
        XpressSolverStrategy()
