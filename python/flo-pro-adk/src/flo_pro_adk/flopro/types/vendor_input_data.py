"""Concrete counterparty input data for VendorAgent.

Contains holding costs, transportation costs, procurement quantities,
and inventory bounds that the VendorAgent uses in its optimization.

Array shapes follow the bounded inventory model dimensions:
    holding_cost_h_it:            (n_vendor_warehouses, n_weeks)
    transportation_cost_r_ijt:    (n_vendor_warehouses, n_inbound_nodes, n_weeks)
    quantity_to_procure_y_ait:    (n_asins, n_vendor_warehouses, n_weeks)
    upper_bound_inv_ait:          (n_asins, n_vendor_warehouses, n_weeks)
    lower_bound_inv_ait:          (n_asins, n_vendor_warehouses, n_weeks)
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
class FloProVendorInputData(CounterpartyInputData):
    """Input data for VendorAgent (simulates a generic vendor).

    Fields:
        n_asins: Number of ASINs (dimension A).
        n_vendor_warehouses: Number of vendor warehouses (dimension I).
        n_inbound_nodes: Number of inbound nodes / 1-DCs (dimension J).
        n_weeks: Number of planning weeks (dimension T).
        holding_cost_h_it: Per-unit holding cost at warehouse, shape (I, T).
        transportation_cost_r_ijt: Per-unit transport cost, shape (I, J, T).
        quantity_to_procure_y_ait: Procurement schedule, shape (A, I, T).
        upper_bound_inv_ait: Inventory upper bound, shape (A, I, T).
        lower_bound_inv_ait: Inventory lower bound, shape (A, I, T).
    """

    n_asins: int
    n_vendor_warehouses: int
    n_inbound_nodes: int
    n_weeks: int
    holding_cost_h_it: ndarray
    transportation_cost_r_ijt: ndarray
    quantity_to_procure_y_ait: ndarray
    upper_bound_inv_ait: ndarray
    lower_bound_inv_ait: ndarray

    def validate(self) -> list[ValidationResult]:
        """Validate shapes, non-negativity, and finiteness."""
        results: list[ValidationResult] = []
        n_asins = self.n_asins
        n_warehouses = self.n_vendor_warehouses
        n_nodes = self.n_inbound_nodes
        n_weeks = self.n_weeks

        shape_checks = [
            ("holding_cost_h_it", self.holding_cost_h_it, (n_warehouses, n_weeks)),
            ("transportation_cost_r_ijt", self.transportation_cost_r_ijt, (n_warehouses, n_nodes, n_weeks)),
            ("quantity_to_procure_y_ait", self.quantity_to_procure_y_ait, (n_asins, n_warehouses, n_weeks)),
            ("upper_bound_inv_ait", self.upper_bound_inv_ait, (n_asins, n_warehouses, n_weeks)),
            ("lower_bound_inv_ait", self.lower_bound_inv_ait, (n_asins, n_warehouses, n_weeks)),
        ]
        for name, arr, expected in shape_checks:
            if arr.shape != expected:
                results.append(ValidationResult(
                    valid=False, severity=ValidationSeverity.ERROR,
                    message=f"{name} shape {arr.shape} != expected {expected}",
                    field=name,
                ))

        for name, arr in [
            ("holding_cost_h_it", self.holding_cost_h_it),
            ("transportation_cost_r_ijt", self.transportation_cost_r_ijt),
            ("quantity_to_procure_y_ait", self.quantity_to_procure_y_ait),
        ]:
            if np.any(arr < 0):
                results.append(ValidationResult(
                    valid=False, severity=ValidationSeverity.ERROR,
                    message=f"{name} contains negative values",
                    field=name,
                ))
            if not np.all(np.isfinite(arr)):
                results.append(ValidationResult(
                    valid=False, severity=ValidationSeverity.ERROR,
                    message=f"{name} contains non-finite values",
                    field=name,
                ))

        # Bound consistency
        if np.any(self.lower_bound_inv_ait > self.upper_bound_inv_ait):
            results.append(ValidationResult(
                valid=False, severity=ValidationSeverity.ERROR,
                message="lower_bound_inv_ait exceeds upper_bound_inv_ait",
                field="lower_bound_inv_ait",
            ))

        return results
