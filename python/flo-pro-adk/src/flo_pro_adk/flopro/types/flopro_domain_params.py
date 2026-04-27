"""Flo Pro domain-specific scenario parameters.

Extends ScenarioDomainParams with Flo Pro public variable dimensions
and retailer cost generation parameters learned from historical data.

Cluster statistics, regional patterns, and temporal patterns are from
330 Newell ASINs across 5 regions and 6 weeks (Nov 2, 2025).

"""

from __future__ import annotations

from typing import Literal, TypedDict

from flo_pro_adk.core.types.scenario_params import (
    ScenarioDomainParams,
)


class RetailerCostParams(TypedDict, total=False):
    """Retailer-side cost and demand generation parameters.

    Controls how demand, holding costs, and backlog penalty are
    generated for RetailerAgent test data.
    """

    # Demand generation: cluster mix probabilities [C0, C1, C2]
    cluster_mix: tuple[float, float, float]

    # Regional distribution pattern
    regional_pattern: Literal["uniform", "concentrated", "dispersed"]

    # Temporal trend pattern
    temporal_pattern: Literal["stable", "growing", "declining", "seasonal"]

    # Holding cost: uniform distribution parameters
    holding_cost_low: float       # lower bound, default 0.006
    holding_cost_high: float      # upper bound, default 0.100

    # Backlog penalty per unit per period
    backlog_penalty: float      # default 10.0

    # Initial on-hand inventory per (ASIN, node)
    initial_inventory: float    # default 0.0


class VendorCostParams(TypedDict, total=False):
    """Vendor-side cost and capacity generation parameters.

    Controls how holding costs, transportation costs, procurement
    quantities, and inventory bounds are generated for VendorAgent.
    """

    # Number of vendor warehouses (dimension I)
    n_vendor_warehouses: int  # default 2

    # Holding cost per unit per period at warehouse, uniform range
    holding_cost_low: float   # default 0.005
    holding_cost_high: float  # default 0.050

    # Transportation cost per unit from warehouse i to node j, uniform range
    transport_cost_low: float   # default 0.01
    transport_cost_high: float  # default 0.10

    # Procurement quantity per (ASIN, warehouse, week), uniform range
    procurement_low: float    # default 10.0
    procurement_high: float   # default 100.0

    # Inventory upper bound per (ASIN, warehouse, week)
    inv_upper_bound: float    # default 500.0

    # Inventory lower bound per (ASIN, warehouse, week)
    inv_lower_bound: float    # default 0.0


class FloProDomainParams(ScenarioDomainParams):
    """Flo Pro-specific scenario parameters.

    Public variable dimensions — these define the shape of the po_qty
    variable group: n_asins * n_inbound_nodes * n_weeks.

    Private variable parameters control cost/demand generation for
    counterparty agents (retailer and vendor).
    """

    n_asins: int
    n_inbound_nodes: int
    n_weeks: int
    retailer_cost_params: RetailerCostParams
    vendor_cost_params: VendorCostParams
