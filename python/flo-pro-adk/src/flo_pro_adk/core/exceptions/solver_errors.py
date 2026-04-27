"""Solver strategy exceptions."""

from __future__ import annotations

from flo_pro_adk.core.exceptions.vadk_error import VADKError


class SolverError(VADKError):
    """Base for all solver strategy errors."""


class SolverConvergenceError(SolverError):
    """Solver failed to find an optimal or feasible solution.

    The ``status`` attribute carries the raw solver status string
    (e.g. "infeasible", "unbounded") for programmatic inspection.
    """

    def __init__(self, message: str, *, status: str = "") -> None:
        super().__init__(message)
        self.status = status
