"""XpressSolverStrategy — Xpress implementation of SolverStrategy.

Builds LP/QP models using FICO Xpress. ADMM terms are constructed as
native Xpress expressions. Supports both MAXIMIZE and MINIMIZE.

Requires: pip install flo-pro-adk[xpress]
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy import ndarray

from flo_pro_adk.core.exceptions.solver_errors import (
    SolverConvergenceError,
)
from flo_pro_sdk.core.variables import PublicVarGroupMetadata

from flo_pro_adk.core.solver.solver_strategy import (
    OptimizationDirection,
    PublicSolverVariable,
    SolverModel,
    SolverResult,
    SolverStrategy,
    SolverVariable,
)


class XpressSolverModel(SolverModel):
    """Xpress implementation of SolverModel."""

    def __init__(
        self,
        consensus: ndarray,
        prices: ndarray,
        rho: ndarray,
        sense: OptimizationDirection,
        var_lb: float,
        var_ub: float | None,
        public_group_metadata: PublicVarGroupMetadata,
    ) -> None:
        import xpress as xp  # type: ignore[import-untyped]

        self._xp = xp
        self._consensus = consensus
        self._prices = prices
        self._rho = rho
        self._sense = sense
        self._n = len(consensus)
        self._public_group_metadata = public_group_metadata

        # Validate shape against metadata
        expected_n = len(public_group_metadata.var_metadata)
        if expected_n != self._n:
            raise ValueError(
                f"Consensus length {self._n} != metadata rows "
                f"{expected_n} for group '{public_group_metadata.name}'"
            )

        self._prob = xp.problem(name="ADMM")

        ub_val = xp.infinity if var_ub is None else var_ub
        self._x_vars = [
            xp.var(name=f"x_{i}", lb=var_lb, ub=ub_val)
            for i in range(self._n)
        ]
        self._prob.addVariable(self._x_vars)
        self._public = PublicSolverVariable(name="x", refs=self._x_vars, public_group_metadata=public_group_metadata)
        self._private_cost: Any = None

    @property
    def public_vars(self) -> PublicSolverVariable:
        return self._public

    @property
    def expr(self) -> Any:
        """Returns the xpress module."""
        return self._xp

    @property
    def problem(self) -> Any:
        """Direct access to the Xpress problem for advanced usage."""
        return self._prob

    def add_variables(
        self,
        name: str,
        count: int,
        lb: float = 0.0,
        ub: float | None = None,
    ) -> SolverVariable:
        xp = self._xp
        ub_val = xp.infinity if ub is None else ub
        vars_list = [
            xp.var(name=f"{name}_{i}", lb=lb, ub=ub_val)
            for i in range(count)
        ]
        self._prob.addVariable(vars_list)
        return SolverVariable(name=name, refs=vars_list)

    def add_constraint(self, constraint: Any) -> None:
        self._prob.addConstraint(constraint)

    def set_private_cost(self, cost_expr: Any) -> None:
        self._private_cost = cost_expr

    def solve(self) -> SolverResult:
        xp = self._xp
        x = self._x_vars
        z = self._consensus
        p = self._prices
        r = self._rho

        subsidy_expr = xp.Sum(p[i] * x[i] for i in range(self._n))
        proximal_expr = xp.Sum(
            r[i] / 2.0 * (x[i] - z[i]) ** 2 for i in range(self._n)
        )
        private_cost_expr = self._private_cost if self._private_cost is not None else 0

        if self._sense == OptimizationDirection.MAXIMIZE:
            self._prob.setObjective(
                private_cost_expr + subsidy_expr - proximal_expr,
                sense=xp.maximize,
            )
        else:
            self._prob.setObjective(
                private_cost_expr - subsidy_expr + proximal_expr,
                sense=xp.minimize,
            )

        self._prob.solve()

        status = self._prob.getProbStatusString()
        status_lower = status.lower()
        if "infeasible" in status_lower or (
            "optimal" not in status_lower and "feasible" not in status_lower
        ):
            raise SolverConvergenceError(
                f"Xpress solver did not find optimal solution: {status}",
                status=status,
            )

        x_sol = np.array([self._prob.getSolution(v) for v in x])

        subsidy_val = float(np.dot(p, x_sol))
        proximal_val = float(0.5 * np.dot(r, (x_sol - z) ** 2))
        total_obj = float(self._prob.getObjVal())

        if self._sense == OptimizationDirection.MAXIMIZE:
            utility_val = total_obj - subsidy_val + proximal_val
        else:
            utility_val = total_obj + subsidy_val - proximal_val

        return SolverResult(
            preferred_vars=x_sol,
            utility=utility_val,
            subsidy=subsidy_val,
            proximal=proximal_val,
        )


class XpressSolverStrategy(SolverStrategy):
    """Xpress-backed SolverStrategy.

    Args:
        license_path: Optional path to xpauth.xpr.
    """

    def __init__(self, license_path: str | None = None) -> None:
        if license_path is not None:
            os.environ["XPAUTH_PATH"] = license_path
        try:
            import xpress  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "xpress package not installed. Install via: "
                "pip install flo-pro-adk[xpress]"
            ) from exc

    def create_model(
        self,
        consensus: ndarray,
        prices: ndarray,
        rho: ndarray,
        *,
        public_group_metadata: PublicVarGroupMetadata,
        sense: OptimizationDirection = OptimizationDirection.MAXIMIZE,
        var_lb: float = 0.0,
        var_ub: float | None = None,
    ) -> XpressSolverModel:
        return XpressSolverModel(
            consensus=consensus,
            prices=prices,
            rho=rho,
            sense=sense,
            var_lb=var_lb,
            var_ub=var_ub,
            public_group_metadata=public_group_metadata,
        )
