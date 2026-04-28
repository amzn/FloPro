"""SolverStrategy — unified solver interface for ADMM agents.

Agents interact with SolverModel to define variables, constraints, and
private cost. The ADMM terms (subsidy, proximal) and solver backend
(Xpress, Gurobi) are hidden. Agents write solver-agnostic business logic.

Usage in an agent's solve()::

    def solve(self, public_vars, prices, rho):
        results = {}
        for group_name, z in public_vars.items():
            model = self.solver.create_model(z, prices[group_name], rho[group_name])
            inv = model.add_variables("inv", count=n, lb=0.0)
            model.add_constraint(balance(model.public_vars, inv, self.data))
            model.set_private_cost(holding(inv) + backlog(b))
            results[group_name] = model.solve()
        return build_solution(results)
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from numpy import ndarray

from flo_pro_sdk.agent.agent_definition import Objective, Solution
from flo_pro_sdk.core.variables import PublicVarGroupMetadata, PublicVarGroupName, PublicVarValues


class OptimizationDirection(Enum):
    """Optimization direction."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class SolverVariable:
    """Handle to a decision variable array in the model.

    Opaque to the agent — the concrete SolverStrategy defines what
    ``refs`` wraps (Xpress vars, Gurobi vars, etc.). Agents use
    ``refs`` to build constraint and cost expressions.

    Returned by ``model.add_variables()`` for private variables.
    See ``PublicSolverVariable`` for public variables with metadata.
    """

    name: str
    refs: Any


@dataclass(frozen=True)
class PublicSolverVariable(SolverVariable):
    """Public decision variable array with group metadata.

    Returned by ``model.public_vars``. Extends ``SolverVariable``
    with the required ``public_group_metadata`` so vendors can
    inspect variable dimensions (asin, node, week) via the
    metadata DataFrame.
    """

    public_group_metadata: PublicVarGroupMetadata = dataclasses.field()


@dataclass(frozen=True)
class SolverResult:
    """Result from solving a single variable group's model."""

    preferred_vars: ndarray
    utility: float
    subsidy: float
    proximal: float

    @property
    def objective(self) -> Objective:
        return Objective(
            utility=self.utility,
            subsidy=self.subsidy,
            proximal=self.proximal,
        )


class SolverModel(ABC):
    """A single optimization model for one variable group.

    Created by SolverStrategy.create_model(). The public decision
    variables and ADMM terms (subsidy, proximal) are pre-wired.
    The agent adds private variables, constraints, and cost.

    Agents use ``public_vars.refs`` and private variable ``.refs``
    to build expressions. The expression syntax depends on the
    backend but the model structure is solver-agnostic.
    """

    @property
    @abstractmethod
    def public_vars(self) -> PublicSolverVariable:
        """The public decision variables (e.g., PO quantities)."""
        ...

    @property
    @abstractmethod
    def expr(self) -> Any:
        """The backend's expression API module (e.g., xpress, gurobipy).

        Agents use this to build constraint and cost expressions
        without importing the solver library directly.
        """
        ...

    @abstractmethod
    def add_variables(
        self,
        name: str,
        count: int,
        lb: float = 0.0,
        ub: float | None = None,
    ) -> SolverVariable:
        """Add private decision variables to the model."""
        ...

    @abstractmethod
    def add_constraint(self, constraint: Any) -> None:
        """Add a constraint to the model."""
        ...

    @abstractmethod
    def set_private_cost(self, cost_expr: Any) -> None:
        """Set the agent's private cost/profit term.

        For MAXIMIZE: pass negated cost (profit).
        For MINIMIZE: pass positive cost.

        Multiple calls replace (not accumulate).
        """
        ...

    @abstractmethod
    def solve(self) -> SolverResult:
        """Solve and return the result.

        The full objective is assembled from the private cost and
        ADMM terms automatically. The result includes the objective
        decomposition (utility, subsidy, proximal).
        """
        ...


class SolverStrategy(ABC):
    """Factory for creating SolverModel instances.

    Hides the solver backend and ADMM term construction. Agents
    receive a SolverModel and only define their business logic.
    """

    @abstractmethod
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
    ) -> SolverModel:
        """Create a model for one variable group.

        Args:
            consensus: Current consensus values (z).
            prices: Dual prices for this group.
            rho: Penalty weights for this group.
            public_group_metadata: Variable group metadata for shape
                validation and vendor inspection.
            sense: MAXIMIZE for profit maximizing, MINIMIZE for cost minimizing.
            var_lb: Lower bound for public decision variables.
            var_ub: Upper bound (None = unbounded).
        """
        ...


def build_solution(results: dict[PublicVarGroupName, SolverResult]) -> Solution:
    """Assemble a CPP SDK Solution from per-group SolverResults."""
    preferred_vars: PublicVarValues = {}
    total_utility = 0.0
    total_subsidy = 0.0
    total_proximal = 0.0

    for group_name, result in results.items():
        preferred_vars[group_name] = result.preferred_vars
        total_utility += result.utility
        total_subsidy += result.subsidy
        total_proximal += result.proximal

    return Solution(
        preferred_vars=preferred_vars,
        objective=Objective(
            utility=total_utility,
            subsidy=total_subsidy,
            proximal=total_proximal,
        ),
    )
