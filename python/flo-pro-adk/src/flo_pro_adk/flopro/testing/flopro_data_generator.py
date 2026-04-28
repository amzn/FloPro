"""FloProSimulationDataGenerator — Flo Pro domain-specific test data generation.

Implements the abstract hooks from SimulationDataGenerator for the Flo Pro domain.
Generates retailer mock input data (demand, holding costs, backlog penalty)
using parameters learned from 330 Newell ASINs.

Demand generation (doc Section 1.3):
    1. Assign each ASIN to a cluster (C0/C1/C2) via cluster_mix probabilities
    2. For each ASIN: base = cluster_mean, std = base * cluster_cv
    3. For each region: regional_demand = base * regional_weight
    4. For each week: demand = regional_demand * temporal_multiplier + N(0, std)

Holding cost generation (doc Section 2.3):
    holding_cost = uniform(low, high), per-ASIN, uniform across regions

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flo_pro_sdk.core.variables import PublicVarGroupName, PublicVarsMetadata

if TYPE_CHECKING:
    from flo_pro_adk.core.data.in_memory_data_loader import (
        InMemoryDataLoader,
    )

from flo_pro_adk.core.testing.simulation_data_generator import (
    SimulationDataGenerator,
)
from flo_pro_adk.core.types.counterparty_input_data import (
    CounterpartyInputData,
)
from flo_pro_adk.core.types.scenario_params import (
    ScenarioParams,
)
from flo_pro_adk.flopro.testing.flopro_defaults import (
    FLOPRO_SCENARIO_DEFAULTS as _D,
)
from flo_pro_adk.flopro.types.flopro_domain_params import (
    FloProDomainParams,
    RetailerCostParams,
    VendorCostParams,
)
from flo_pro_adk.flopro.types.retailer_input_data import (
    FloProRetailerInputData,
)
from flo_pro_adk.flopro.types.vendor_input_data import (
    FloProVendorInputData,
)

from flo_pro_adk.core.types.public_variable_id import (
    PublicVariableId,
)
from flo_pro_adk.flopro.registration import (
    FLOPRO_GROUP_NAME,
    flopro_var_metadata,
)

# --- Learned base parameters (doc Tables 1-3) ---

# Cluster statistics: (mean_demand, cv)
_CLUSTER_STATS: dict[int, tuple[float, float]] = {
    0: (133.31, 0.297),   # Low demand, stable
    1: (1573.45, 0.302),  # High demand, elite
    2: (185.36, 0.658),   # Medium demand, high volatility
}

# Regional weight patterns (doc Table 2)
_REGIONAL_PATTERNS: dict[str, list[float]] = {
    "uniform":       [0.247, 0.158, 0.233, 0.131, 0.231],
    "concentrated":  [0.315, 0.094, 0.258, 0.078, 0.255],
    "dispersed":     [0.230, 0.191, 0.217, 0.147, 0.215],
}

# Temporal multiplier patterns (doc Table 3)
_TEMPORAL_PATTERNS: dict[str, list[float]] = {
    "stable":    [1.014, 0.980, 0.989, 1.000, 1.015, 1.001],
    "growing":   [0.762, 0.791, 0.868, 1.150, 1.156, 1.273],
    "declining": [1.273, 1.156, 1.150, 0.868, 0.791, 0.762],
    "seasonal":  [0.367, 0.406, 1.009, 2.336, 1.093, 0.788],
}

# Default retailer cost generation parameters
_DEFAULT_RETAILER_COST_PARAMS: RetailerCostParams = {
    "cluster_mix": (0.745, 0.042, 0.212),
    "regional_pattern": "uniform",
    "temporal_pattern": "stable",
    "holding_cost_low": 0.006,
    "holding_cost_high": 0.100,
    "backlog_penalty": 10.0,
    "initial_inventory": 0.0,
}

# Default vendor cost generation parameters
_DEFAULT_VENDOR_COST_PARAMS: VendorCostParams = {
    "n_vendor_warehouses": 2,
    "holding_cost_low": 0.005,
    "holding_cost_high": 0.050,
    "transport_cost_low": 0.01,
    "transport_cost_high": 0.10,
    "procurement_low": 10.0,
    "procurement_high": 100.0,
    "inv_upper_bound": 500.0,
    "inv_lower_bound": 0.0,
}


class FloProSimulationDataGenerator(SimulationDataGenerator):
    """Flo Pro domain-specific test data generator.

    Variable groups use a single group "asin_vendor_inbound_periods" containing a flattened
    ndarray indexed by (ASIN, inbound_node, week).
    """

    def __init__(self, params: ScenarioParams) -> None:
        super().__init__(params)

    def get_group_names(self) -> list[PublicVarGroupName]:
        """Flo Pro uses a single variable group."""
        return [FLOPRO_GROUP_NAME]

    def generate_variable_group_metadata(self) -> PublicVarsMetadata:
        """Rich metadata with asin, vendor_code, inbound_node, week columns."""
        dp = self._get_domain_params()
        pids = [
            PublicVariableId(
                asin=f"ASIN_{asin_idx}",
                vendor_code="V0",
                inbound_node=f"NODE_{node_idx}",
            )
            for asin_idx in range(dp["n_asins"])
            for node_idx in range(dp["n_inbound_nodes"])
        ]
        return flopro_var_metadata(pids, dp["n_weeks"])

    def _get_domain_params(self) -> FloProDomainParams:
        """Extract FloProDomainParams from scenario."""
        dp = self._params.domain_params
        # If domain_params is already a FloProDomainParams, use it directly.
        # Otherwise fall back to defaults for missing keys.
        n_asins: int = dp.get("n_asins", _D["n_asins"])  # type: ignore[assignment]
        n_inbound_nodes: int = dp.get("n_inbound_nodes", _D["n_inbound_nodes"])  # type: ignore[assignment]
        n_weeks: int = dp.get("n_weeks", _D["n_weeks"])  # type: ignore[assignment]
        retailer_cost_params: RetailerCostParams = dp.get("retailer_cost_params", {})  # type: ignore[assignment]
        vendor_cost_params: VendorCostParams = dp.get("vendor_cost_params", {})  # type: ignore[assignment]
        return FloProDomainParams(
            n_asins=n_asins,
            n_inbound_nodes=n_inbound_nodes,
            n_weeks=n_weeks,
            retailer_cost_params=retailer_cost_params,
            vendor_cost_params=vendor_cost_params,
        )

    def _resolve_retailer_cost_params(self) -> RetailerCostParams:
        """Merge scenario retailer_cost_params with defaults."""
        dp = self._get_domain_params()
        overrides: RetailerCostParams = dp.get("retailer_cost_params", {})  # type: ignore[assignment]
        merged: RetailerCostParams = {**_DEFAULT_RETAILER_COST_PARAMS, **overrides}  # type: ignore[typeddict-item]
        return merged

    def _resolve_vendor_cost_params(self) -> VendorCostParams:
        """Merge scenario vendor_cost_params with defaults."""
        dp = self._get_domain_params()
        overrides: VendorCostParams = dp.get("vendor_cost_params", {})  # type: ignore[assignment]
        merged: VendorCostParams = {**_DEFAULT_VENDOR_COST_PARAMS, **overrides}  # type: ignore[typeddict-item]
        return merged

    def _generate_demand(
        self,
        n_asins: int,
        n_regions: int,
        n_weeks: int,
        rcp: RetailerCostParams,
    ) -> np.ndarray:
        """Generate demand array following doc Section 1.3.

        Returns:
            demand array, shape (n_asins, n_regions, n_weeks).
        """
        cluster_mix = rcp["cluster_mix"]
        regional_weights = _REGIONAL_PATTERNS[rcp["regional_pattern"]]
        temporal_mults = _TEMPORAL_PATTERNS[rcp["temporal_pattern"]]

        # Adapt patterns to actual dimensions (truncate or tile)
        reg_w = np.array(regional_weights[:n_regions])
        reg_w = reg_w / reg_w.sum()  # renormalize if truncated
        temp_m = np.array(temporal_mults[:n_weeks])

        # Step 1: Assign each ASIN to a cluster
        mix = np.array(cluster_mix, dtype=np.float64)
        mix = mix / mix.sum()  # normalize for floating-point precision
        cluster_ids = self._rng.choice(
            [0, 1, 2], size=n_asins, p=mix,
        )

        # Step 2-4: Generate demand per ASIN
        demand = np.zeros((n_asins, n_regions, n_weeks))
        for a in range(n_asins):
            cid = cluster_ids[a]
            base, cv = _CLUSTER_STATS[cid]
            std = base * cv

            for r in range(n_regions):
                regional_demand = base * reg_w[r]
                for w in range(n_weeks):
                    noise = self._rng.normal(0, std)
                    demand[a, r, w] = max(
                        0.0, regional_demand * temp_m[w] + noise,
                    )

        return demand

    def _generate_holding_costs(
        self,
        n_asins: int,
        n_regions: int,
        rcp: RetailerCostParams,
    ) -> np.ndarray:
        """Generate holding costs from a uniform distribution.

        Per-ASIN, uniform across regions.

        Returns:
            holding_costs array, shape (n_asins, n_regions).
        """
        low = rcp["holding_cost_low"]
        high = rcp["holding_cost_high"]

        per_asin = self._rng.uniform(low, high, size=n_asins)
        # Broadcast across regions (uniform across nodes per doc)
        return np.broadcast_to(
            per_asin[:, np.newaxis], (n_asins, n_regions),
        ).copy()

    def generate_counterparty_input_data(self) -> CounterpartyInputData:
        """Generate FloProRetailerInputData from scenario parameters.

        Produces demand (Section 1), holding costs (Section 2), and
        other input parameters (Section 3) for RetailerAgent.
        """
        dp = self._get_domain_params()
        rcp = self._resolve_retailer_cost_params()

        n_asins = dp["n_asins"]
        n_nodes = dp["n_inbound_nodes"]
        n_weeks = dp["n_weeks"]

        demand = self._generate_demand(n_asins, n_nodes, n_weeks, rcp)
        holding_costs = self._generate_holding_costs(n_asins, n_nodes, rcp)
        backlog_penalty = float(rcp["backlog_penalty"])
        init_inv_val = float(rcp["initial_inventory"])
        initial_inventory = np.full((n_asins, n_nodes), init_inv_val)

        return FloProRetailerInputData(
            n_asins=n_asins,
            n_inbound_nodes=n_nodes,
            n_weeks=n_weeks,
            demand=demand,
            holding_costs=holding_costs,
            backlog_penalty=backlog_penalty,
            initial_inventory=initial_inventory,
        )

    def create_vendor_data_loader(self) -> InMemoryDataLoader:
        """Convenience: generate vendor data and wrap in InMemoryDataLoader."""
        from flo_pro_adk.core.data.in_memory_data_loader import (
            InMemoryDataLoader,
        )
        return InMemoryDataLoader(self.generate_vendor_input_data())

    def create_data_loader_for(self, agent_cls: type) -> InMemoryDataLoader:
        """Create the appropriate DataLoader for a given agent class.

        - RetailerAgent → FloProRetailerInputData
        - VendorAgent → FloProVendorInputData
        """
        from flo_pro_adk.core.data.in_memory_data_loader import (
            InMemoryDataLoader,
        )
        from flo_pro_adk.flopro.counterparty.retailer_agent import (
            RetailerAgent,
        )
        from flo_pro_adk.flopro.counterparty.vendor_agent import (
            VendorAgent,
        )
        if issubclass(agent_cls, VendorAgent):
            return InMemoryDataLoader(self.generate_vendor_input_data())
        if issubclass(agent_cls, RetailerAgent):
            return InMemoryDataLoader(self.generate_counterparty_input_data())
        raise NotImplementedError(f"No data loader for {agent_cls.__name__}")

    def generate_vendor_input_data(self) -> FloProVendorInputData:
        """Generate FloProVendorInputData from scenario parameters.

        Produces holding costs, transportation costs, procurement
        quantities, and inventory bounds for VendorAgent.
        """
        dp = self._get_domain_params()
        vcp = self._resolve_vendor_cost_params()

        n_asins = dp["n_asins"]
        n_nodes = dp["n_inbound_nodes"]
        n_weeks = dp["n_weeks"]
        n_warehouses: int = vcp["n_vendor_warehouses"]

        holding_cost_h_it = self._rng.uniform(
            vcp["holding_cost_low"], vcp["holding_cost_high"],
            size=(n_warehouses, n_weeks),
        )
        transportation_cost_r_ijt = self._rng.uniform(
            vcp["transport_cost_low"], vcp["transport_cost_high"],
            size=(n_warehouses, n_nodes, n_weeks),
        )
        quantity_to_procure_y_ait = self._rng.uniform(
            vcp["procurement_low"], vcp["procurement_high"],
            size=(n_asins, n_warehouses, n_weeks),
        )
        upper_bound_inv_ait = np.full(
            (n_asins, n_warehouses, n_weeks), vcp["inv_upper_bound"],
        )
        lower_bound_inv_ait = np.full(
            (n_asins, n_warehouses, n_weeks), vcp["inv_lower_bound"],
        )

        return FloProVendorInputData(
            n_asins=n_asins,
            n_vendor_warehouses=n_warehouses,
            n_inbound_nodes=n_nodes,
            n_weeks=n_weeks,
            holding_cost_h_it=holding_cost_h_it,
            transportation_cost_r_ijt=transportation_cost_r_ijt,
            quantity_to_procure_y_ait=quantity_to_procure_y_ait,
            upper_bound_inv_ait=upper_bound_inv_ait,
            lower_bound_inv_ait=lower_bound_inv_ait,
        )
