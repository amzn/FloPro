# V-ADK: Vendor Agent Development Kit

Build and test your Flo Pro negotiation agent locally. The V-ADK provides a simulated Amazon agent, a solver framework, and ready-to-run test suites so you can validate your agent before connecting to Amazon's systems.

Your agent and Amazon's agent iteratively negotiate purchase order (PO) quantities. Each round, you receive a proposed plan and price signals, and respond with your preferred quantities. The V-ADK lets you develop and test this logic on your machine — no cloud services, no API calls.

## Requirements

- Python >= 3.11
- FICO Xpress via the `[xpress]` extra below. The `xpress` PyPI package ships
  with FICO's free Community license (5000 rows+columns for LP/MIP — more than
  enough for the built-in test scenarios).

## Quick Start

### 1. Install

Neither `flo-pro-sdk` nor `flo-pro-adk` is on PyPI yet. Clone and install both
from the local paths:

```bash
git clone https://github.com/amzn/FloPro.git
cd FloPro
pip install ./python/flo-pro-sdk "./python/flo-pro-adk[xpress]"
```

The `[xpress]` extra is required for the built-in solver framework and for E2E
tests against the simulated Amazon agent (Xpress-backed) — install it even if
your own agent uses a different solver.

### 2. Implement Your Agent

Subclass `AgentDefinition`. Implement `register()` to declare your public variables and `solve()` to return your preferred PO quantities each round:

```python
from flo_pro_sdk.agent.agent_definition import AgentDefinition, Solution, Objective
from flo_pro_adk.flopro.registration import flopro_var_metadata

class MyVendorAgent(AgentDefinition):
    def __init__(self, **_):
        # Build metadata in __init__ so it's available for both register() and solve()
        # These come from your Flo Pro onboarding — your ASINs, vendor code, and fulfillment nodes
        self._var_metadata = flopro_var_metadata(
            public_variable_ids=[
                {"asin": "B001", "vendor_code": "V1", "inbound_node": "FC1"},
                {"asin": "B001", "vendor_code": "V1", "inbound_node": "FC2"},
            ],
            n_weeks=3,
        )

    def register(self):
        return self._var_metadata

    def solve(self, public_vars, prices, rho):
        # public_vars: proposed PO quantities (group_name -> ndarray)
        # prices: price signals from the negotiation
        # rho: how strongly to stay close to the proposed plan
        max_capacity = 100.0
        preferred = {g: v.clip(0, max_capacity) for g, v in public_vars.items()}
        return Solution(
            preferred_vars=preferred,
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        )
```

### 3. Run Tests

```python
from flo_pro_adk.flopro.testing.simulation_suite import (
    FloProSimulationSuite,
)

suite = FloProSimulationSuite(MyVendorAgent)
suite.run_all()  # runs unit + E2E tests
```

Or run them separately:

```python
suite.run_unit()  # unit tests only
suite.run_e2e()   # E2E tests only
```

Unit tests check that your output has the right shape, is finite, and is deterministic. E2E tests run full negotiations between your agent and the simulated Amazon agent, checking that they converge to agreement. A failing test means your agent's output doesn't meet these properties.

## Solver Framework (Optional)

The `SolverStrategy` framework builds an optimization model where you add your constraints and cost — the framework adds the negotiation terms automatically:

```python
from flo_pro_sdk.agent.agent_definition import AgentDefinition
from flo_pro_adk.flopro.registration import flopro_var_metadata
from flo_pro_adk.core.solver.solver_strategy import build_solution
from flo_pro_adk.core.solver.xpress_solver_strategy import XpressSolverStrategy

class MyVendorAgent(AgentDefinition):
    def __init__(self, **_):
        self._var_metadata = flopro_var_metadata(...)  # same as Quick Start
        self._solver = XpressSolverStrategy()

    def register(self):
        return self._var_metadata

    def solve(self, public_vars, prices, rho):
        results = {}
        for group_name, z in public_vars.items():
            model = self._solver.create_model(
                z, prices[group_name], rho[group_name],
                public_group_metadata=self._var_metadata[group_name],
            )
            xp = model.expr
            x = model.public_vars.refs  # PO quantity variables

            # Add your constraints and cost
            model.add_constraint(xp.Sum(x[i] for i in range(len(z))) <= 1000)
            model.set_private_cost(-xp.Sum(x[i] for i in range(len(z))))

            results[group_name] = model.solve()
        return build_solution(results)
```

If you have your own optimization model (Gurobi, CPLEX, or custom), implement `solve()` directly — your agent just needs to return a `Solution`. Additional solver backends can be added by implementing `SolverStrategy` and `SolverModel`.

## Data Loading

Load your input data in `__init__`. The V-ADK ships `PandasDataLoader` for CSV files:

```python
from flo_pro_adk.core.data.pandas_data_loader import PandasDataLoader

class MyVendorAgent(AgentDefinition):
    def __init__(self, cost_file="data/vendor_costs.csv", **_):
        loader = PandasDataLoader(cost_file)
        self._data = loader.load()  # reads CSV, auto-snapshots to ~/.vadk/snapshots/
```

`PandasDataLoader` caches on first read and automatically snapshots for debugging. For custom data sources, implement the `DataLoader` interface with `load()` and `snapshot()`. The V-ADK also ships `InMemoryDataLoader` for wrapping test data in memory.

## Built-in Scenarios

The test suites include scenarios for different conditions:

- **Base** — normal demand and cost parameters
- **Demand spike** — elevated demand with higher penalty weights
- **Supply constrained** — tighter supply with increased costs
- **Non-convergence** — adversarial parameters to test robustness

## Package Structure

```
core/                          # Domain-independent framework
├── assembly/                  # build_problem()
├── assertions/                # AgentAssertions, CoordinationAssertions
├── counterparty/              # CounterpartyAgent ABC
├── data/                      # DataLoader ABC, InMemoryDataLoader, PandasDataLoader
├── exceptions/                # VADKError, RegistrationError, SolverConvergenceError
├── solver/                    # SolverStrategy, SolverModel, PublicSolverVariable
├── testing/                   # Test runners, SimulationDataGenerator
└── types/                     # ScenarioParams, CounterpartyInputData, PublicVariableId

flopro/                        # Flo Pro domain pack
├── counterparty/              # RetailerAgent, VendorAgent (simulated)
├── testing/                   # FloProSimulationDataGenerator, scenarios, FloProSimulationSuite
└── types/                     # FloProDomainParams, RetailerInputData, VendorInputData
```
