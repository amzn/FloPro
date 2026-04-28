"""Concrete counterparty input data for RetailerAgent.

Contains demand, holding costs, backlog penalty, and initial inventory
that the RetailerAgent uses in its Xpress optimization formulation.

Array shapes follow the Flo Pro domain dimensions:
    demand:            (n_asins, n_inbound_nodes, n_weeks)
    holding_costs:     (n_asins, n_inbound_nodes)
    initial_inventory: (n_asins, n_inbound_nodes)

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from flo_pro_adk.core.types.counterparty_input_data import (
    CounterpartyInputData,
)
from flo_pro_adk.core.types.validation_result import (
    ValidationResult,
    ValidationSeverity,
)


@dataclass(frozen=True)
class FloProRetailerInputData(CounterpartyInputData):
    """Input data for RetailerAgent (simulates Amazon).

    Fields:
        n_asins: Number of ASINs (dimension I).
        n_inbound_nodes: Number of inbound nodes / 1-DCs (dimension J).
        n_weeks: Number of planning weeks (dimension T).
        demand: Deterministic demand D_ijt, shape (I, J, T).
        holding_costs: Per-unit per-period holding cost H_ij, shape (I, J).
        backlog_penalty: Per-unit per-period backlog penalty P (scalar).
        initial_inventory: On-hand inventory at t=0 I_ij0, shape (I, J).
    """

    n_asins: int
    n_inbound_nodes: int
    n_weeks: int
    demand: ndarray
    holding_costs: ndarray
    backlog_penalty: float
    initial_inventory: ndarray

    def validate(self) -> list[ValidationResult]:
        """Validate shapes, non-negativity, and finiteness."""
        results: list[ValidationResult] = []
        n_asins, n_nodes, n_weeks = self.n_asins, self.n_inbound_nodes, self.n_weeks

        # Shape checks
        if self.demand.shape != (n_asins, n_nodes, n_weeks):
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message=f"demand shape {self.demand.shape} != expected ({n_asins}, {n_nodes}, {n_weeks})",
                field="demand",
            ))
        if self.holding_costs.shape != (n_asins, n_nodes):
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message=f"holding_costs shape {self.holding_costs.shape} != expected ({n_asins}, {n_nodes})",
                field="holding_costs",
            ))
        if self.initial_inventory.shape != (n_asins, n_nodes):
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message=f"initial_inventory shape {self.initial_inventory.shape} != expected ({n_asins}, {n_nodes})",
                field="initial_inventory",
            ))

        # Non-negativity
        if np.any(self.demand < 0):
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message="demand contains negative values",
                field="demand",
            ))
        if np.any(self.holding_costs < 0):
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message="holding_costs contains negative values",
                field="holding_costs",
            ))
        if self.backlog_penalty < 0:
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message=f"backlog_penalty {self.backlog_penalty} is negative",
                field="backlog_penalty",
            ))

        # Finiteness
        for name, arr in [("demand", self.demand), ("holding_costs", self.holding_costs),
                          ("initial_inventory", self.initial_inventory)]:
            if not np.all(np.isfinite(arr)):
                results.append(ValidationResult(
                    valid=False, severity=ValidationSeverity.ERROR,
                    message=f"{name} contains non-finite values",
                    field=name,
                ))

        return results
