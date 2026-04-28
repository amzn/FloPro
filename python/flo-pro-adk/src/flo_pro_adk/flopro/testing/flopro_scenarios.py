"""Built-in Flo Pro scenarios and auto-registration.

Two categories of scenarios:

1. Unit test scenarios — ADMM-level parameters (n_variables, rho, price
   distribution) for single-iteration directional correctness testing.
   Domain-agnostic, no retailer/vendor cost data needed.
   Used by: run_unit_test(), run_rho_sensitivity(), run_price_sensitivity()

2. E2E scenarios — retailer demand/cost generation parameters for
   multi-iteration coordination testing with RetailerAgent.
   Based on 5 business scenarios from the mock retail agent doc (Section 1.2).
   Used by: run_e2e_test() with RetailerAgent + FloProSimulationDataGenerator

"""

from __future__ import annotations

from flo_pro_adk.core.types.scenario_params import (
    ScenarioParams,
)
from flo_pro_adk.flopro.testing.flopro_defaults import (
    FLOPRO_SCENARIO_DEFAULTS,
)
from flo_pro_adk.flopro.types.flopro_domain_params import (
    FloProDomainParams,
)

_D = FLOPRO_SCENARIO_DEFAULTS

# ============================================================
# Unit test scenarios (ADMM-level, domain-agnostic)
# ============================================================

BASE_SCENARIO = ScenarioParams(
    name="base",
    seed=_D["base_seed"],
    n_variables=_D["n_variables"],
    n_groups=_D["n_groups"],
    price_distribution="uniform",
    price_range=_D["price_range"],
    rho=_D["rho"],
    domain_params={},
)

DEMAND_SPIKE_SCENARIO = ScenarioParams(
    name="demand_spike",
    seed=_D["base_seed"] + 1,
    n_variables=_D["n_variables"],
    n_groups=_D["n_groups"],
    price_distribution="uniform",
    price_range=(0.0, _D["price_range"][1] * 2),
    rho=_D["rho"] * 2,
    domain_params={},
)

SUPPLY_CONSTRAINED_SCENARIO = ScenarioParams(
    name="supply_constrained",
    seed=_D["base_seed"] + 2,
    n_variables=_D["n_variables"],
    n_groups=_D["n_groups"],
    price_distribution="uniform",
    price_range=(0.0, _D["price_range"][1] * 1.5),
    rho=_D["rho"] * 1.5,
    domain_params={},
)

NON_CONVERGENCE_SCENARIO = ScenarioParams(
    name="non_convergence",
    seed=_D["base_seed"] + 3,
    n_variables=_D["n_variables"],
    n_groups=_D["n_groups"],
    price_distribution="lognormal",
    price_range=(0.01, _D["price_range"][1] * 10),
    rho=_D["rho"] * 0.01,
    domain_params={},
)

# ============================================================
# E2E scenarios (retailer demand/cost, doc Section 1.2)
# ============================================================

_N_ASINS = _D["n_asins"]
_N_INBOUND_NODES = _D["n_inbound_nodes"]
_N_WEEKS = _D["n_weeks"]

NORMAL_SCENARIO = ScenarioParams(
    name="normal",
    seed=_D["base_seed"] + 10,
    n_variables=_D["n_variables"],
    n_groups=_D["n_groups"],
    price_distribution="uniform",
    price_range=_D["price_range"],
    rho=_D["rho"],
    domain_params=FloProDomainParams(
        n_asins=_N_ASINS,
        n_inbound_nodes=_N_INBOUND_NODES,
        n_weeks=_N_WEEKS,
        retailer_cost_params={
            "cluster_mix": (0.745, 0.042, 0.212),
            "regional_pattern": "uniform",
            "temporal_pattern": "stable",
        },
        vendor_cost_params={},
    ),
)
