# V-ADK API Reference

The Vendor Agent Development Kit (V-ADK) lets you build, test, and validate your Flo Pro negotiation agent locally. This reference covers every public class, function, and type you interact with.

For a getting-started walkthrough, see the [README](../../README.md).

## Installation

See the [Flo Pro ADK README](../../README.md#1-install) for install instructions.

## Quick Import Reference

```python
# Your agent class and return types
from flo_pro_sdk.agent.agent_definition import AgentDefinition, Solution, Objective

# Variable registration for Flo Pro
from flo_pro_adk.flopro.registration import flopro_var_metadata
from flo_pro_adk.core.types.public_variable_id import PublicVariableId

# Solver framework (optional — only if using the built-in Xpress solver)
from flo_pro_adk.core.solver.solver_strategy import build_solution
from flo_pro_adk.core.solver.xpress_solver_strategy import XpressSolverStrategy

# Testing — run pre-built test suites against your agent
from flo_pro_adk.flopro.testing.simulation_suite import FloProSimulationSuite

# Data loading (optional — for loading your private cost/capacity data)
from flo_pro_adk.core.data.pandas_data_loader import PandasDataLoader
```

The V-ADK (`flo_pro_adk`) depends on the Flo Pro SDK (`flo_pro_sdk`), which is installed automatically. Core types like `AgentDefinition`, `Solution`, and `Objective` come from the SDK; everything else comes from the V-ADK.

---

## Glossary

| Term | Meaning |
|------|---------|
| PO quantities | Purchase order quantities — the numbers being negotiated between you and Amazon. |
| Consensus plan | The current proposed PO quantities that both sides are converging toward. Passed to `solve()` as `public_vars`. |
| Price signals | Feedback from the negotiation indicating how much each PO quantity should change. Passed to `solve()` as `prices`. Higher values push you to increase that quantity. |
| Rho (ρ) | Penalty weight controlling how closely your response should track the consensus plan. Higher rho means stay closer to the proposal. Passed to `solve()` as `rho`. |
| Utility | Your private profit or negated cost — the part of the objective only you know. |
| Subsidy | The price signal contribution to the objective: `Σ prices[i] * x[i]`. |
| Proximal penalty | A penalty for deviating from the consensus: `Σ rho[i]/2 * (x[i] - z[i])²`. Keeps the negotiation stable. |
| Convergence | When both agents agree — the difference between preferred quantities and the consensus plan becomes small enough. |

---

## Contents

- [Installation](#installation)
- [What You're Building](#what-youre-building)
- [Agent Lifecycle](#agent-lifecycle)
- [Complete Working Example](#complete-working-example)
- [Understanding Flat Array Indexing](#understanding-flat-array-indexing)
- [AgentDefinition](#agentdefinition)
- [Solution & Objective](#solution--objective)
- [Flo Pro Registration](#flo-pro-registration)
- [Solver Framework](#solver-framework)
- [Data Loading](#data-loading)
- [Counterparty Agents](#counterparty-agents)
- [Testing](#testing)
- [Assertions](#assertions)
- [Types](#types)
- [Exceptions](#exceptions)
- [Troubleshooting](#troubleshooting)

---

## What You're Building

You're building an agent that negotiates purchase order (PO) quantities with Amazon. The negotiation determines how many units of each product Amazon orders from you, at which fulfillment centers, for each week in a planning horizon.

The negotiation works through rounds. Each round:

1. The coordination system presents a consensus plan — the current proposed PO quantities.
2. Your agent receives the plan and responds with its preferred quantities — the amounts that best serve your business (minimizing your costs, respecting your capacity, etc.).
3. Amazon's agent does the same from its side.
4. The coordinator adjusts the plan based on both responses and starts the next round.

This repeats until both sides converge on quantities they can agree on.

Your job is to implement the `solve()` method — the optimization logic that takes a proposed plan and returns your preferred quantities. The V-ADK handles everything else: the negotiation protocol, the coordination loop, test data, and validation.

---

## Agent Lifecycle

Every negotiation follows this sequence:

```
1. register()  →  Declare what variables you negotiate over
2. solve()     →  Called repeatedly with updated consensus, prices, and rho
3. (repeat 2 until convergence)
```

You implement `register()` and `solve()`. The SDK handles the coordination loop, price updates, and convergence checks.

```python
from flo_pro_sdk.agent.agent_definition import AgentDefinition, Solution, Objective
from flo_pro_adk.flopro.registration import flopro_var_metadata
from flo_pro_adk.core.types.public_variable_id import PublicVariableId

class MyVendorAgent(AgentDefinition):

    def register(self):
        # Declare the variables you're negotiating over
        return flopro_var_metadata(
            public_variable_ids=[
                PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="ABE8"),
                PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="AVP1"),
                PublicVariableId(asin="B02DEF", vendor_code="V0", inbound_node="ABE8"),
            ],
            n_weeks=6,
        )

    def solve(self, public_vars, prices, rho):
        # Stub: just agrees with whatever Amazon proposes.
        # Replace this with your optimization logic.
        preferred = {g: v.copy() for g, v in public_vars.items()}
        return Solution(
            preferred_vars=preferred,
            objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
        )
```

This stub always agrees with the consensus — it's a valid starting point to verify your project setup, but it won't produce meaningful negotiation results. See the [Complete Working Example](#complete-working-example) for a real agent with optimization logic.

---

## Complete Working Example

A minimal but real agent that a vendor can copy, run, and see tests pass. This agent uses the Xpress solver to minimize transportation cost subject to minimum order constraints.

```python
"""my_vendor_agent.py — complete, runnable vendor agent."""

from flo_pro_sdk.agent.agent_definition import AgentDefinition, Solution, Objective
from flo_pro_adk.flopro.registration import flopro_var_metadata
from flo_pro_adk.core.solver.solver_strategy import build_solution
from flo_pro_adk.core.solver.xpress_solver_strategy import XpressSolverStrategy
from flo_pro_adk.core.types.public_variable_id import PublicVariableId

class MyVendorAgent(AgentDefinition):

    def __init__(self):
        self._solver = XpressSolverStrategy()
        # 3 ASINs × 2 inbound nodes × 3 weeks = 18 variables
        self._ids = [
            PublicVariableId(asin=f"ASIN_{a}", vendor_code="V0", inbound_node=f"NODE_{j}")
            for a in range(3)
            for j in range(2)
        ]
        self._n_weeks = 3
        self._metadata = flopro_var_metadata(self._ids, self._n_weeks)

        # Private cost data: transport cost per unit for each variable
        # (in practice, load this from your data source)
        self._transport_cost = [0.05] * (len(self._ids) * self._n_weeks)

        # Minimum order quantity per variable
        self._min_order = [10.0] * (len(self._ids) * self._n_weeks)

    def register(self):
        return self._metadata

    def solve(self, public_vars, prices, rho):
        results = {}

        for group_name, z in public_vars.items():
            n = len(z)
            model = self._solver.create_model(
                consensus=z,
                prices=prices[group_name],
                rho=rho[group_name],
                public_group_metadata=self._metadata[group_name],
            )

            xp = model.expr
            x = model.public_vars.refs  # PO quantity variables

            # Constraint: each PO quantity >= minimum order
            model.add_constraint([
                x[i] >= self._min_order[i] for i in range(n)
            ])

            # Private cost: transportation (negate because framework maximizes)
            model.set_private_cost(
                -xp.Sum(self._transport_cost[i] * x[i] for i in range(n))
            )

            results[group_name] = model.solve()

        return build_solution(results)
```

Run the pre-built tests against it:

```python
from flo_pro_adk.flopro.testing.simulation_suite import FloProSimulationSuite
from my_vendor_agent import MyVendorAgent

suite = FloProSimulationSuite(MyVendorAgent)
suite.run_all()  # 0 = all passed
```

---

## Understanding Flat Array Indexing

The `public_vars`, `prices`, and `rho` arrays passed to `solve()` are flat 1-D numpy arrays. Each index maps to a specific (ASIN, vendor_code, inbound_node, week) combination. The ordering is: iterate over `public_variable_ids` first, then over weeks.

For example, with 2 IDs and 3 weeks:

```
public_variable_ids = [
    PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="ABE8"),  # id 0
    PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="AVP1"),  # id 1
]
n_weeks = 3

# Flat array layout (6 elements):
# Index 0 → (B01ABC, V0, ABE8, week=0)
# Index 1 → (B01ABC, V0, ABE8, week=1)
# Index 2 → (B01ABC, V0, ABE8, week=2)
# Index 3 → (B01ABC, V0, AVP1, week=0)
# Index 4 → (B01ABC, V0, AVP1, week=1)
# Index 5 → (B01ABC, V0, AVP1, week=2)
```

To convert between business dimensions and flat indices:

```python
def flat_index(id_position: int, week: int, n_weeks: int) -> int:
    """Convert (id_position, week) to flat array index."""
    return id_position * n_weeks + week

# Example: id_position=1 (AVP1), week=2 → index 5
idx = flat_index(1, 2, n_weeks=3)  # 5
po_qty = public_vars[group_name][idx]
```

You can also use the metadata DataFrame to look up what each index represents:

```python
meta = model.public_vars.public_group_metadata.var_metadata
print(meta)
#     asin vendor_code inbound_node  week
# 0  B01ABC          V0         ABE8     0
# 1  B01ABC          V0         ABE8     1
# 2  B01ABC          V0         ABE8     2
# 3  B01ABC          V0         AVP1     0
# 4  B01ABC          V0         AVP1     1
# 5  B01ABC          V0         AVP1     2
```

---

## AgentDefinition

```python
from flo_pro_sdk.agent.agent_definition import AgentDefinition
```

Base class for all agents. You subclass this and implement `register()` and `solve()`.

### Methods

#### register()

```python
def register(self) -> PublicVarsMetadata
```

Declare the public variables your agent negotiates over. Called once at the start of a coordination run.

**Returns:** `PublicVarsMetadata` — a dict mapping group names to `PublicVarGroupMetadata`. For Flo Pro, use `flopro_var_metadata()` to build this.

#### solve(public_vars, prices, rho)

```python
def solve(
    self,
    public_vars: PublicVarValues,
    prices: Prices,
    rho: RhoValues,
) -> Solution
```

Your core optimization logic. Called once per negotiation iteration.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `public_vars` | `dict[PublicVarGroupName, ndarray]` | Current consensus PO quantities. Keys are group names (e.g., `"asin_vendor_inbound_periods"`), values are flat numpy arrays. |
| `prices` | `dict[PublicVarGroupName, ndarray]` | Dual price signals. Same structure as `public_vars`. Higher values push you to increase that quantity. |
| `rho` | `dict[PublicVarGroupName, ndarray]` | Penalty weights. Same structure. Higher values mean stay closer to the consensus. |

**Returns:** `Solution` — your preferred PO quantities and objective breakdown.

**Example — echo agent (returns consensus as-is):**

```python
def solve(self, public_vars, prices, rho):
    return Solution(
        preferred_vars={g: v.copy() for g, v in public_vars.items()},
        objective=Objective(utility=0.0, subsidy=0.0, proximal=0.0),
    )
```

**Example — agent with Xpress solver (fragment — see [Complete Working Example](#complete-working-example) for a full runnable version):**

```python
def solve(self, public_vars, prices, rho):
    results = {}

    for group_name, z in public_vars.items():
        model = self._solver.create_model(
            consensus=z,
            prices=prices[group_name],
            rho=rho[group_name],
            # self._metadata is set in __init__ via flopro_var_metadata()
            public_group_metadata=self._metadata[group_name],
        )

        # Add private variables (self.n_items, self.min_stock, self.cost
        # are your own data — set in __init__ or loaded via DataLoader)
        inv = model.add_variables("inv", count=self.n_items, lb=0.0)

        # Add constraints
        xp = model.expr
        model.add_constraint([
            inv.refs[i] >= self.min_stock[i] for i in range(self.n_items)
        ])

        # Set private cost (negated because framework maximizes)
        model.set_private_cost(
            -xp.Sum(self.cost[i] * inv.refs[i] for i in range(self.n_items))
        )

        results[group_name] = model.solve()

    return build_solution(results)
```

#### create(agent_params) — classmethod

```python
@classmethod
def create(cls, agent_params: dict[str, JsonValue]) -> AgentDefinition
```

Factory method. The SDK calls this to instantiate your agent. Override if you need custom construction (e.g., loading data, initializing a solver).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `agent_params` | `dict[str, JsonValue]` | JSON-serializable configuration (vendor code, feature flags, tuning parameters). |

**Returns:** An instance of your agent class.

---

## Solution & Objective

```python
from flo_pro_sdk.agent.agent_definition import Solution, Objective
```

### Solution

The return type of `solve()`. Contains your preferred variable values and an objective breakdown.

```python
@dataclass(frozen=True)
class Solution:
    preferred_vars: PublicVarValues   # dict[PublicVarGroupName, ndarray]
    objective: Objective
```

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `preferred_vars` | `dict[PublicVarGroupName, ndarray]` | Your preferred PO quantities. Must have the same keys and array shapes as the `public_vars` input to `solve()`. |
| `objective` | `Objective` | Breakdown of your objective value into utility, subsidy, and proximal components. |

### Objective

```python
@dataclass(frozen=True)
class Objective:
    utility: float
    subsidy: float
    proximal: float
```

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `utility` | `float` | Your private profit or negated cost. Only you know this value. |
| `subsidy` | `float` | Price signal contribution: `Σ prices[i] * x[i]`. |
| `proximal` | `float` | Penalty for deviating from consensus: `Σ rho[i]/2 * (x[i] - z[i])²`. |

If you use the solver framework, these are computed automatically. If you use your own solver, you must compute them correctly.

---

## Flo Pro Registration

```python
from flo_pro_adk.flopro.registration import flopro_var_metadata
```

### flopro_var_metadata(public_variable_ids, n_weeks)

```python
def flopro_var_metadata(
    public_variable_ids: list[PublicVariableId],
    n_weeks: int,
) -> PublicVarsMetadata
```

Build the variable metadata for Flo Pro agents. Call this in your `register()` method.

Creates a single variable group named `"asin_vendor_inbound_periods"` with `len(public_variable_ids) * n_weeks` variables. Each variable maps to a specific (ASIN, vendor_code, inbound_node, week) combination.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `public_variable_ids` | `list[PublicVariableId]` | List of (asin, vendor_code, inbound_node) triples identifying the PO quantities you negotiate over. |
| `n_weeks` | `int` | Number of planning weeks in the coordination horizon. |

**Returns:** `PublicVarsMetadata` — pass this directly from `register()`.

**Example:**

```python
from flo_pro_adk.core.types.public_variable_id import PublicVariableId

def register(self):
    ids = [
        PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="ABE8"),
        PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="AVP1"),
    ]
    return flopro_var_metadata(public_variable_ids=ids, n_weeks=6)
    # Creates 2 * 6 = 12 variables
```

The resulting metadata DataFrame has columns: `asin`, `vendor_code`, `inbound_node`, `week`. Use this to map flat array indices back to business dimensions.

### FLOPRO_GROUP_NAME

```python
from flo_pro_adk.flopro.registration import FLOPRO_GROUP_NAME
```

A constant equal to `PublicVarGroupName("asin_vendor_inbound_periods")`. This is the canonical group name for Flo Pro public variables. Use it when indexing into `public_vars`, `prices`, or `rho` dicts:

```python
z = public_vars[FLOPRO_GROUP_NAME]
p = prices[FLOPRO_GROUP_NAME]
r = rho[FLOPRO_GROUP_NAME]
```

---

## Solver Framework

The solver framework handles ADMM boilerplate so you only write business logic. It manages public variables, the subsidy term, and the proximal penalty automatically. You define private variables, constraints, and your private cost.

Using the solver framework is optional. If you have your own solver (Gurobi, CPLEX, custom), you can call it directly in `solve()` — but you must compute utility, subsidy, and proximal correctly yourself.

**Example — using your own solver (no SolverStrategy):**

```python
import numpy as np

def solve(self, public_vars, prices, rho):
    preferred = {}
    total_utility = 0.0
    total_subsidy = 0.0
    total_proximal = 0.0

    for group_name, z in public_vars.items():
        p = prices[group_name]
        r = rho[group_name]

        # --- Your optimization here ---
        # Maximize: private_cost + subsidy - proximal
        # where:
        #   private_cost = your negated cost (profit)
        #   subsidy      = Σ p[i] * x[i]
        #   proximal     = Σ r[i]/2 * (x[i] - z[i])²
        x_sol = self._my_solver(z, p, r)  # your solver returns optimal x
        # --- End your optimization ---

        preferred[group_name] = x_sol

        # You MUST compute these correctly
        total_subsidy += float(np.dot(p, x_sol))
        total_proximal += float(0.5 * np.dot(r, (x_sol - z) ** 2))
        # utility = your private cost term (negated cost, i.e. profit)
        total_utility += self._compute_my_profit(x_sol)

    return Solution(
        preferred_vars=preferred,
        objective=Objective(
            utility=total_utility,
            subsidy=total_subsidy,
            proximal=total_proximal,
        ),
    )
```

### SolverStrategy

```python
from flo_pro_adk.core.solver.solver_strategy import SolverStrategy
```

Abstract factory for creating `SolverModel` instances. Hides the solver backend.

#### create_model(consensus, prices, rho, *, public_group_metadata, sense, var_lb, var_ub)

```python
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
) -> SolverModel
```

Create an optimization model for one variable group.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `consensus` | `ndarray` | *required* | Current consensus values (z). Same as `public_vars[group_name]`. |
| `prices` | `ndarray` | *required* | Dual prices for this group. |
| `rho` | `ndarray` | *required* | Penalty weights for this group. |
| `public_group_metadata` | `PublicVarGroupMetadata` | *required* | Variable group metadata. Pass `self._metadata[group_name]` or the metadata from `register()`. |
| `sense` | `OptimizationDirection` | `MAXIMIZE` | `MAXIMIZE` for profit-maximizing agents, `MINIMIZE` for cost-minimizing. |
| `var_lb` | `float` | `0.0` | Lower bound for public decision variables. |
| `var_ub` | `float \| None` | `None` | Upper bound for public decision variables. `None` = unbounded. |

**Returns:** `SolverModel` — add your private variables, constraints, and cost to this model, then call `model.solve()`.

**Raises:** `ValueError` if consensus length doesn't match metadata row count.

---

### SolverModel

```python
from flo_pro_adk.core.solver.solver_strategy import SolverModel
```

A single optimization model for one variable group. Created by `SolverStrategy.create_model()`. The public decision variables and ADMM terms (subsidy, proximal) are pre-wired. You add private variables, constraints, and cost.

#### public_vars

```python
@property
def public_vars(self) -> PublicSolverVariable
```

The public decision variables (PO quantities). Use `.refs` to access the solver variable references for building expressions.

```python
x = model.public_vars.refs  # list of solver variables
# x[0], x[1], ... are the PO quantity variables
```

#### expr

```python
@property
def expr(self) -> Any
```

The solver backend's expression API module (e.g., `xpress`). Use this to build constraint and cost expressions without importing the solver library directly.

```python
xp = model.expr
cost = xp.Sum(c[i] * x[i] for i in range(n))
```

#### add_variables(name, count, lb, ub)

```python
def add_variables(
    self,
    name: str,
    count: int,
    lb: float = 0.0,
    ub: float | None = None,
) -> SolverVariable
```

Add private decision variables to the model.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *required* | Variable name prefix (e.g., `"inv"`, `"backlog"`). |
| `count` | `int` | *required* | Number of variables to create. |
| `lb` | `float` | `0.0` | Lower bound. |
| `ub` | `float \| None` | `None` | Upper bound. `None` = unbounded. |

**Returns:** `SolverVariable` — use `.refs` to access the variable references.

**Example:**

```python
inv = model.add_variables("inv", count=18, lb=0.0, ub=1000.0)
# inv.refs[0], inv.refs[1], ... are the inventory variables
```

#### add_constraint(constraint)

```python
def add_constraint(self, constraint: Any) -> None
```

Add one or more constraints to the model. Accepts a single constraint or a list.

**Example:**

```python
xp = model.expr
x = model.public_vars.refs
inv = model.add_variables("inv", count=n, lb=0.0)

# Single constraint
model.add_constraint(inv.refs[0] >= 100)

# List of constraints
model.add_constraint([
    inv.refs[i] >= demand[i] for i in range(n)
])

# Balance constraint
model.add_constraint([
    inv.refs[t] == inv.refs[t-1] + x[t] - demand[t]
    for t in range(1, T)
])
```

#### set_private_cost(cost_expr)

```python
def set_private_cost(self, cost_expr: Any) -> None
```

Set your private cost or profit term.

The framework assembles the full objective as: `private_cost + subsidy - proximal`, then maximizes it. Since costs are things you want to *minimize*, negate them so that maximizing `(-cost)` is equivalent to minimizing `cost`.

- For `MAXIMIZE` (default): pass negated cost (i.e., profit). Example: `-(holding + transport)`.
- For `MINIMIZE`: pass positive cost. The framework flips the subsidy and proximal signs internally.

Multiple calls replace (do not accumulate). The framework adds subsidy and proximal terms automatically.

**Example:**

```python
xp = model.expr
holding = xp.Sum(h[i] * inv.refs[i] for i in range(n))
transport = xp.Sum(c[i] * ship.refs[i] for i in range(m))

# Maximization: negate costs
model.set_private_cost(-(holding + transport))
```

#### solve()

```python
def solve(self) -> SolverResult
```

Solve the model and return the result. The full objective is assembled from your private cost and the ADMM terms automatically.

**Returns:** `SolverResult` with preferred variable values and objective decomposition.

**Raises:** `SolverConvergenceError` if the solver cannot find an optimal or feasible solution.

---

### SolverVariable & PublicSolverVariable

```python
from flo_pro_adk.core.solver.solver_strategy import (
    SolverVariable, PublicSolverVariable,
)
```

#### SolverVariable

Handle to a decision variable array in the model. Returned by `model.add_variables()`.

```python
@dataclass(frozen=True)
class SolverVariable:
    name: str    # Variable name prefix
    refs: Any    # List of solver-specific variable references
```

Use `refs` to build expressions: `var.refs[i]`.

#### PublicSolverVariable

Extends `SolverVariable` with group metadata. Returned by `model.public_vars`.

```python
@dataclass(frozen=True)
class PublicSolverVariable(SolverVariable):
    public_group_metadata: PublicVarGroupMetadata
```

Access the metadata DataFrame to map flat indices to business dimensions:

```python
meta = model.public_vars.public_group_metadata.var_metadata
# DataFrame with columns: asin, vendor_code, inbound_node, week
print(meta.iloc[0])  # {'asin': 'B01ABC', 'vendor_code': 'V0', 'inbound_node': 'ABE8', 'week': 0}
```

---

### SolverResult

```python
from flo_pro_adk.core.solver.solver_strategy import SolverResult
```

Result from solving a single variable group's model.

```python
@dataclass(frozen=True)
class SolverResult:
    preferred_vars: ndarray   # Optimal public variable values
    utility: float            # Private cost/profit component
    subsidy: float            # Price signal component
    proximal: float           # Proximal penalty component
```

**Properties:**

| Name | Type | Description |
|------|------|-------------|
| `preferred_vars` | `ndarray` | Optimal values for the public decision variables. |
| `utility` | `float` | Your private profit (or negated cost). |
| `subsidy` | `float` | Price signal contribution: `Σ prices[i] * x[i]`. |
| `proximal` | `float` | Proximal penalty: `Σ rho[i]/2 * (x[i] - z[i])²`. |
| `objective` | `Objective` | Convenience property returning `Objective(utility, subsidy, proximal)`. |

---

### XpressSolverStrategy

```python
from flo_pro_adk.core.solver.xpress_solver_strategy import (
    XpressSolverStrategy,
)
```

Xpress-backed implementation of `SolverStrategy`. Builds LP/QP models using FICO Xpress.

```python
class XpressSolverStrategy(SolverStrategy):
    def __init__(self, license_path: str | None = None) -> None: ...
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `license_path` | `str \| None` | `None` | Path to `xpauth.xpr` license file. If `None`, uses the `XPAUTH_PATH` environment variable. |

**Raises:** `ImportError` if the `xpress` package is not installed.

**Install:** `pip install flo-pro-adk[xpress]`

**Example:**

```python
solver = XpressSolverStrategy(license_path="/path/to/xpauth.xpr")
model = solver.create_model(
    consensus=z,
    prices=p,
    rho=r,
    public_group_metadata=metadata[group_name],
)
```

The `XpressSolverModel` returned by `create_model()` also exposes:

- `model.problem` — direct access to the underlying `xpress.problem` for advanced usage.
- `model.expr` — returns the `xpress` module itself.

---

### build_solution

```python
from flo_pro_adk.core.solver.solver_strategy import build_solution
```

```python
def build_solution(results: dict[PublicVarGroupName, SolverResult]) -> Solution
```

Assemble a `Solution` from per-group `SolverResult` objects. Sums utility, subsidy, and proximal across all groups.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `results` | `dict[PublicVarGroupName, SolverResult]` | One `SolverResult` per variable group, keyed by group name. |

**Returns:** `Solution` ready to return from `solve()`.

**Example:**

```python
def solve(self, public_vars, prices, rho):
    results = {}
    for group_name, z in public_vars.items():
        model = self._solver.create_model(
            consensus=z,
            prices=prices[group_name],
            rho=rho[group_name],
            public_group_metadata=self._metadata[group_name],
        )
        # ... add variables, constraints, cost ...
        results[group_name] = model.solve()
    return build_solution(results)
```

### OptimizationDirection

```python
from flo_pro_adk.core.solver.solver_strategy import OptimizationDirection
```

```python
class OptimizationDirection(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
```

Pass to `create_model(sense=...)`. Default is `MAXIMIZE`.

---

## Data Loading

### DataLoader

```python
from flo_pro_adk.core.data.data_loader import DataLoader
```

Abstract base class for agent input data sourcing. Implement this to load your agent's private data (costs, capacities, constraints) from any source.

```python
class DataLoader(ABC):
    @abstractmethod
    def load(self) -> Any: ...

    @abstractmethod
    def snapshot(self, run_id: str) -> None: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `load()` | Read agent input data from the source. Return type is up to you. |
| `snapshot(run_id)` | Persist a point-in-time copy of the data for debugging and reproducibility. |

In production, `snapshot()` is called once at the start of a coordination run to freeze data. In local testing, it's typically a no-op.

---

### PandasDataLoader

```python
from flo_pro_adk.core.data.pandas_data_loader import PandasDataLoader
```

Read a CSV file into a pandas DataFrame with automatic snapshotting.

```python
class PandasDataLoader(DataLoader):
    def __init__(self, path: str | Path, snapshot_dir: Path | None = None) -> None: ...
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `path` | `str \| Path` | *required* | Path to a CSV file. |
| `snapshot_dir` | `Path \| None` | `~/.vadk/snapshots/` | Directory for snapshot files. |

**Behavior:**
- First call to `load()` reads the CSV, auto-snapshots, and caches the DataFrame.
- Subsequent calls return the cached DataFrame.
- Snapshots are written to `{snapshot_dir}/{stem}_{timestamp}/data.csv`.

**Example:**

```python
loader = PandasDataLoader("data/my_costs.csv")
df = loader.load()  # pandas DataFrame
```

**Example — wiring a DataLoader into your agent via `create()`:**

```python
class MyVendorAgent(AgentDefinition):

    def __init__(self, cost_data):
        self._cost_data = cost_data
        # ... set up metadata, solver, etc.

    @classmethod
    def create(cls, agent_params):
        loader = PandasDataLoader(agent_params.get("cost_file", "data/costs.csv"))
        cost_data = loader.load()
        return cls(cost_data=cost_data)

    def solve(self, public_vars, prices, rho):
        # self._cost_data is a DataFrame available here
        ...
```

---

### InMemoryDataLoader

```python
from flo_pro_adk.core.data.in_memory_data_loader import InMemoryDataLoader
```

Wraps pre-generated data for simulation and testing. Used internally by the test framework.

```python
class InMemoryDataLoader(DataLoader):
    def __init__(self, data: Any) -> None: ...
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `data` | `Any` | The data object to return from `load()`. |

`snapshot()` is a no-op.

---

## Counterparty Agents

The V-ADK provides two simulated counterparty agents for local testing. These are not Amazon's production algorithms — they are simplified models that produce behaviorally valid ADMM responses so you can test convergence locally.

### RetailerAgent

```python
from flo_pro_adk.flopro.counterparty.retailer_agent import RetailerAgent
```

Simulates Amazon's retailer side. Used when testing a vendor agent.

Optimization: minimizes holding cost + backlog penalty, subject to inventory balance constraints. Receives PO quantities as public variables and decides how to manage inventory across ASINs and inbound nodes over time.

Use with `FloProSimulationSuite` (automatic) or `run_e2e_test`:

```python
result = run_e2e_test(
    agent_class=MyVendorAgent,
    counterparty_class=RetailerAgent,
    ...
)
```

### VendorAgent

```python
from flo_pro_adk.flopro.counterparty.vendor_agent import VendorAgent
```

Simulates a generic vendor. Used when the Amazon agent team tests their agent.

Optimization: minimizes holding cost + transportation cost, subject to inventory balance, inventory bounds, and lead-time (PO = sum of shipments) constraints. Manages shipping from vendor warehouses to Amazon inbound nodes.

Both agents use `SolverStrategy` internally — their source code serves as a reference implementation for how to structure a `solve()` method with the solver framework.

---

## Testing

### FloProSimulationSuite

```python
from flo_pro_adk.flopro.testing.simulation_suite import (
    FloProSimulationSuite,
)
```

The main entry point for testing your agent. Runs pre-built unit and E2E test suites with zero test code from you.

```python
class FloProSimulationSuite:
    def __init__(self, agent_class: type[AgentDefinition]) -> None: ...
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `agent_class` | `type[AgentDefinition]` | Your agent class. |

#### run_all()

```python
def run_all(self) -> int
```

Run all unit and E2E tests. Returns the pytest exit code (`0` = all passed).

#### run_unit()

```python
def run_unit(self) -> int
```

Run unit tests only. These call `solve()` once per scenario and check:
- Solution shape and finiteness
- Determinism (same inputs → same output)
- Rho sensitivity (L2 distance to consensus decreases as rho increases)

#### run_e2e()

```python
def run_e2e(self) -> int
```

Run E2E tests only. These run full multi-iteration ADMM coordination between your agent and a simulated counterparty, checking convergence across multiple scenarios.

**Example:**

```python
from my_agent import MyVendorAgent

suite = FloProSimulationSuite(MyVendorAgent)
exit_code = suite.run_all()  # 0 = all passed

# Or run selectively:
suite.run_unit()
suite.run_e2e()
```

---

### Unit Test Runners

```python
from flo_pro_adk.core.testing.unit_test_runner import (
    run_unit_test,
    run_unit_test_with_inputs,
    run_rho_sensitivity,
    run_price_sensitivity,
    UnitTestResult,
)
```

Lower-level test runners for custom test scenarios. Use these if you need more control than `FloProSimulationSuite` provides.

#### run_unit_test(agent_class, data_generator_class, scenario, *, agent_params)

```python
def run_unit_test(
    agent_class: type[AgentDefinition],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
) -> UnitTestResult
```

Run a single `solve()` call with generated inputs.

**Returns:** `UnitTestResult` containing the solution, consensus vars, prices, and rho.

#### run_unit_test_with_inputs(agent_class, consensus_vars, prices, rho, *, agent_params)

```python
def run_unit_test_with_inputs(
    agent_class: type[AgentDefinition],
    consensus_vars: PublicVarValues,
    prices: Prices,
    rho: RhoValues,
    *,
    agent_params: dict[str, JsonValue] | None = None,
) -> UnitTestResult
```

Run a single `solve()` call with explicit inputs you provide.

#### run_rho_sensitivity(agent_class, data_generator_class, scenario, *, agent_params, n_points)

```python
def run_rho_sensitivity(
    agent_class: type[AgentDefinition],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
    n_points: int = 10,
) -> list[UnitTestResult]
```

Run `solve()` across a logarithmic range of rho values. Returns one `UnitTestResult` per rho point.

#### run_price_sensitivity(agent_class, data_generator_class, scenario, *, agent_params, n_variants)

```python
def run_price_sensitivity(
    agent_class: type[AgentDefinition],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
    n_variants: int = 5,
) -> list[UnitTestResult]
```

Run `solve()` across different price signal variants. Returns one `UnitTestResult` per variant.

#### UnitTestResult

```python
@dataclass(frozen=True)
class UnitTestResult:
    solution: Solution
    consensus_vars: PublicVarValues
    prices: Prices
    rho: RhoValues
```

**Properties:**

| Name | Type | Description |
|------|------|-------------|
| `solution` | `Solution` | The agent's solve() output. |
| `consensus_vars` | `PublicVarValues` | The consensus inputs used. |
| `prices` | `Prices` | The price inputs used. |
| `rho` | `RhoValues` | The rho inputs used. |
| `objective` | `Objective` | Shortcut for `solution.objective`. |
| `preferred_vars` | `PublicVarValues` | Shortcut for `solution.preferred_vars`. |

---

### E2E Test Runner

```python
from flo_pro_adk.core.testing.e2e_test_runner import (
    run_e2e_test,
    E2ETestResult,
)
```

#### run_e2e_test(agent_class, counterparty_class, data_generator_class, scenario, *, ...)

```python
def run_e2e_test(
    agent_class: type[AgentDefinition],
    counterparty_class: type[CounterpartyAgent],
    data_generator_class: type[SimulationDataGenerator],
    scenario: ScenarioParams,
    *,
    agent_params: dict[str, JsonValue] | None = None,
    max_iterations: int = 1000,
    convergence_primal_tol: float = 1e-4,
    convergence_dual_tol: float = 1e-4,
) -> E2ETestResult
```

Run a full multi-iteration ADMM coordination test.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent_class` | `type[AgentDefinition]` | *required* | Your agent class. |
| `counterparty_class` | `type[CounterpartyAgent]` | *required* | `RetailerAgent` (for vendor testing) or `VendorAgent` (for Amazon agent testing). |
| `data_generator_class` | `type[SimulationDataGenerator]` | *required* | `FloProSimulationDataGenerator` for Flo Pro. |
| `scenario` | `ScenarioParams` | *required* | Test scenario configuration. |
| `agent_params` | `dict \| None` | `None` | Optional agent configuration. |
| `max_iterations` | `int` | `1000` | Maximum ADMM iterations before stopping. |
| `convergence_primal_tol` | `float` | `1e-4` | Primal residual tolerance for convergence. |
| `convergence_dual_tol` | `float` | `1e-4` | Dual residual tolerance for convergence. |

**Returns:** `E2ETestResult`

**Example:**

```python
from flo_pro_adk.flopro.counterparty.retailer_agent import RetailerAgent
from flo_pro_adk.flopro.testing.flopro_data_generator import FloProSimulationDataGenerator
from flo_pro_adk.flopro.testing.flopro_scenarios import BASE_SCENARIO

result = run_e2e_test(
    agent_class=MyVendorAgent,
    counterparty_class=RetailerAgent,
    data_generator_class=FloProSimulationDataGenerator,
    scenario=BASE_SCENARIO,
    max_iterations=500,
)
print(f"Converged: {result.converged} in {result.n_iterations} iterations")
```

#### E2ETestResult

```python
@dataclass(frozen=True)
class E2ETestResult:
    final_state: State
    n_iterations: int
    converged: bool
```

**Properties:**

| Name | Type | Description |
|------|------|-------------|
| `final_state` | `State` | The final coordination state (consensus, prices, residuals). |
| `n_iterations` | `int` | Number of ADMM iterations executed. |
| `converged` | `bool` | Whether primal and dual residuals are within tolerance. |
| `final_residuals` | `tuple[float, float] \| None` | `(primal, dual)` residuals, or `None` if unavailable. |

---

### Simulation Data Generator

```python
from flo_pro_adk.flopro.testing.flopro_data_generator import (
    FloProSimulationDataGenerator,
)
```

Generates test data for Flo Pro scenarios. You typically don't use this directly — `FloProSimulationSuite` and `run_e2e_test` handle it. Documented here for custom test scenarios.

```python
class FloProSimulationDataGenerator(SimulationDataGenerator):
    def __init__(self, params: ScenarioParams) -> None: ...
```

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `generate_consensus_vars()` | `PublicVarValues` | Initial consensus variable values. |
| `generate_prices()` | `Prices` | Initial dual prices (zeros). |
| `generate_rho()` | `RhoValues` | Initial penalty weights (uniform). |
| `generate_variable_group_metadata()` | `PublicVarsMetadata` | Rich metadata with asin, vendor_code, inbound_node, week columns. |
| `generate_counterparty_input_data()` | `FloProRetailerInputData` | Mock retailer demand, holding costs, backlog penalty. |
| `generate_vendor_input_data()` | `FloProVendorInputData` | Mock vendor holding costs, transport costs, procurement. |
| `generate_rho_series(n_points)` | `list[RhoValues]` | Logarithmic range of rho values for sensitivity testing. |
| `generate_price_variants(n_variants)` | `list[Prices]` | Random price variants for directional testing. |

All randomness is seed-controlled via `ScenarioParams.seed` for reproducibility.

---

### Built-in Scenarios

```python
from flo_pro_adk.flopro.testing.flopro_scenarios import (
    BASE_SCENARIO,
    DEMAND_SPIKE_SCENARIO,
    SUPPLY_CONSTRAINED_SCENARIO,
    NON_CONVERGENCE_SCENARIO,
    NORMAL_SCENARIO,
)
```

Pre-configured `ScenarioParams` for common test conditions:

| Scenario | Description |
|----------|-------------|
| `BASE_SCENARIO` | Baseline conditions. Moderate prices, standard rho. |
| `DEMAND_SPIKE_SCENARIO` | Doubled price range and rho, simulating high demand. |
| `SUPPLY_CONSTRAINED_SCENARIO` | 1.5x price range and rho, simulating supply pressure. |
| `NON_CONVERGENCE_SCENARIO` | Lognormal prices, very low rho. Stress test — may not converge. |
| `NORMAL_SCENARIO` | Full domain scenario with 3 ASINs, 2 inbound nodes, 3 weeks. Stable demand. Includes `FloProDomainParams`. |

All scenarios work with both unit and E2E tests. `NORMAL_SCENARIO` includes domain parameters (n_asins, n_inbound_nodes, n_weeks) needed for E2E coordination with counterparty agents. The others use ADMM-level parameters only, but the E2E test suite uses them too.

**Default dimensions** (from `FLOPRO_SCENARIO_DEFAULTS`):

| Parameter | Value |
|-----------|-------|
| `n_variables` | 18 |
| `n_asins` | 3 |
| `n_inbound_nodes` | 2 |
| `n_weeks` | 3 |
| `rho` | 1.0 |
| `price_range` | (0.0, 10.0) |
| `seed` | 42 |

---

## Assertions

### AgentAssertions

```python
from flo_pro_adk.core.assertions.agent_assertions import (
    AgentAssertions,
)
```

Assertions for validating a single agent's `solve()` output. Used by the pre-built test suites and available for your own tests.

```python
assertions = AgentAssertions()
```

#### assert_solution_valid(solution, public_vars, *, check_finiteness, check_shape)

```python
def assert_solution_valid(
    self,
    solution: Solution,
    public_vars: PublicVarValues,
    *,
    check_finiteness: bool = True,
    check_shape: bool = True,
) -> None
```

Assert structural validity of a `Solution`:
- All expected variable groups are present in `preferred_vars`.
- Array shapes match the input `public_vars`.
- All values are finite (no NaN, inf).

**Raises:** `VADKAssertionError` on failure.

#### assert_l2_distance_decreases(solve_fn, public_vars, prices, rho_low, rho_high)

```python
def assert_l2_distance_decreases(
    self,
    solve_fn: Callable,
    public_vars: PublicVarValues,
    prices: Prices,
    rho_low: RhoValues,
    rho_high: RhoValues,
) -> None
```

Assert that L2 distance to consensus decreases when rho increases. This is a fundamental ADMM property — higher penalty should pull the solution closer to the consensus.

**Raises:** `VADKAssertionError` if distance increases with higher rho.

#### assert_deterministic(solve_fn, public_vars, prices, rho, *, n_calls)

```python
def assert_deterministic(
    self,
    solve_fn: Callable,
    public_vars: PublicVarValues,
    prices: Prices,
    rho: RhoValues,
    *,
    n_calls: int = 3,
) -> None
```

Assert that `solve()` is deterministic — same inputs produce identical output across `n_calls` invocations.

**Raises:** `VADKAssertionError` if outputs differ.

---

### CoordinationAssertions

```python
from flo_pro_adk.core.assertions.coordination_assertions import (
    CoordinationAssertions,
)
```

Assertions for multi-agent ADMM coordination results. Used by E2E tests.

```python
assertions = CoordinationAssertions()
```

#### assert_convergence(final_state, *, primal_tol, dual_tol)

```python
def assert_convergence(
    self,
    final_state: State,
    *,
    primal_tol: float = 1e-4,
    dual_tol: float = 1e-4,
) -> None
```

Assert the coordination converged — both primal and dual residuals are within tolerance.

**Raises:** `VADKAssertionError` if residuals exceed tolerance or are missing.

#### assert_gap_narrowing(states, *, window)

```python
def assert_gap_narrowing(
    self,
    states: Sequence[State],
    *,
    window: int = 10,
) -> None
```

Assert the primal-dual gap narrows over a sliding window. Compares the average gap in the first `window` iterations against the last `window` iterations.

Used internally by the pre-built E2E test suite. To use directly, you need a sequence of intermediate `State` objects from the coordination loop (not currently exposed by `run_e2e_test`).

**Raises:** `VADKAssertionError` if the late average is not smaller than the early average.

#### assert_price_stabilization(states, *, tail_fraction, max_oscillation)

```python
def assert_price_stabilization(
    self,
    states: Sequence[State],
    *,
    tail_fraction: float = 0.2,
    max_oscillation: float = 0.01,
) -> None
```

Assert prices stabilize in the tail of the coordination run. Checks that the maximum price change between consecutive iterations in the last `tail_fraction` of the run is below `max_oscillation`.

Used internally by the pre-built E2E test suite. Same caveat as `assert_gap_narrowing` — requires intermediate states.

**Raises:** `VADKAssertionError` if prices are still oscillating.

---

## Types

### PublicVariableId

```python
from flo_pro_adk.core.types.public_variable_id import (
    PublicVariableId,
)
```

Identifies a single public variable in the Flo Pro domain. Matches the API contract.

```python
class PublicVariableId(TypedDict):
    asin: str
    vendor_code: str
    inbound_node: str
```

**Example:**

```python
pid = PublicVariableId(asin="B01ABC", vendor_code="V0", inbound_node="ABE8")
```

---

### ScenarioParams

```python
from flo_pro_adk.core.types.scenario_params import ScenarioParams
```

Immutable parameter set configuring a test scenario.

```python
@dataclass(frozen=True)
class ScenarioParams:
    name: str
    seed: int
    n_variables: int
    n_groups: int
    price_distribution: Literal["uniform", "normal", "lognormal"]
    price_range: tuple[float, float]
    rho: float
    domain_params: ScenarioDomainParams
```

**Fields:**

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Scenario name (must not be empty). |
| `seed` | `int` | Random seed for reproducibility. |
| `n_variables` | `int` | Total number of public variables (must be positive). |
| `n_groups` | `int` | Number of variable groups (must be positive). |
| `price_distribution` | `str` | Distribution for generating prices: `"uniform"`, `"normal"`, or `"lognormal"`. |
| `price_range` | `tuple[float, float]` | `(low, high)` bounds for price generation. |
| `rho` | `float` | Base penalty weight (must be positive). |
| `domain_params` | `ScenarioDomainParams` | Domain-specific parameters (e.g., `FloProDomainParams`). |

**Validation:** Raises `ValueError` if name is empty, n_variables/n_groups are non-positive, price_range is inverted, or rho is non-positive.

**Example — custom scenario:**

```python
custom = ScenarioParams(
    name="my_scenario",
    seed=123,
    n_variables=36,
    n_groups=1,
    price_distribution="uniform",
    price_range=(0.0, 20.0),
    rho=2.0,
    domain_params=FloProDomainParams(
        n_asins=6,
        n_inbound_nodes=2,
        n_weeks=3,
        retailer_cost_params={"temporal_pattern": "seasonal"},
        vendor_cost_params={},
    ),
)
```

---

### FloProDomainParams

```python
from flo_pro_adk.flopro.types.flopro_domain_params import (
    FloProDomainParams,
    RetailerCostParams,
    VendorCostParams,
)
```

Domain-specific scenario parameters for Flo Pro.

```python
class FloProDomainParams(ScenarioDomainParams):
    n_asins: int
    n_inbound_nodes: int
    n_weeks: int
    retailer_cost_params: RetailerCostParams
    vendor_cost_params: VendorCostParams
```

#### RetailerCostParams

Controls demand and cost generation for the simulated Amazon agent. All fields are optional — defaults are used for missing keys.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `cluster_mix` | `tuple[float, float, float]` | `(0.745, 0.042, 0.212)` | Probability of assigning ASINs to demand clusters (low, high, volatile). |
| `regional_pattern` | `str` | `"uniform"` | Regional demand distribution: `"uniform"`, `"concentrated"`, `"dispersed"`. |
| `temporal_pattern` | `str` | `"stable"` | Temporal demand trend: `"stable"`, `"growing"`, `"declining"`, `"seasonal"`. |
| `holding_cost_low` | `float` | `0.006` | Lower bound for holding cost generation. |
| `holding_cost_high` | `float` | `0.100` | Upper bound for holding cost generation. |
| `backlog_penalty` | `float` | `10.0` | Per-unit per-period backlog penalty. |
| `initial_inventory` | `float` | `0.0` | Initial on-hand inventory per (ASIN, node). |

#### VendorCostParams

Controls cost generation for the simulated vendor agent. All fields are optional.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_vendor_warehouses` | `int` | `2` | Number of vendor warehouses. |
| `holding_cost_low` | `float` | `0.005` | Lower bound for warehouse holding cost. |
| `holding_cost_high` | `float` | `0.050` | Upper bound for warehouse holding cost. |
| `transport_cost_low` | `float` | `0.01` | Lower bound for transportation cost. |
| `transport_cost_high` | `float` | `0.10` | Upper bound for transportation cost. |
| `procurement_low` | `float` | `10.0` | Lower bound for procurement quantity. |
| `procurement_high` | `float` | `100.0` | Upper bound for procurement quantity. |
| `inv_upper_bound` | `float` | `500.0` | Inventory upper bound. |
| `inv_lower_bound` | `float` | `0.0` | Inventory lower bound. |

---

### CounterpartyInputData

```python
from flo_pro_adk.core.types.counterparty_input_data import (
    CounterpartyInputData,
)
```

Abstract base for mock counterparty input data. Domain packs provide concrete implementations.

```python
class CounterpartyInputData(ABC):
    @abstractmethod
    def validate(self) -> list[ValidationResult]: ...
```

Concrete implementations for Flo Pro:

- `FloProRetailerInputData` — demand, holding costs, backlog penalty, initial inventory.
- `FloProVendorInputData` — holding costs, transport costs, procurement, inventory bounds.

Both validate shapes, non-negativity, and finiteness.

---

### ValidationResult

```python
from flo_pro_adk.core.types.validation_result import (
    ValidationResult,
    ValidationSeverity,
)
```

```python
@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    severity: ValidationSeverity   # WARNING or ERROR
    message: str
    field: str | None = None

class ValidationSeverity(Enum):
    WARNING = "warning"
    ERROR = "error"
```

Returned by `CounterpartyInputData.validate()`. An empty list means the data is valid.

---

## Exceptions

All V-ADK exceptions inherit from `VADKError`. The hierarchy:

```
VADKError                          # Base for all V-ADK errors
├── AgentError                     # Agent lifecycle errors
│   └── RegistrationError          # Missing variable metadata at registration time
├── AssemblyError                  # Problem assembly errors
│   ├── ScenarioNotFoundError      # Requested scenario not in registry
│   ├── InvalidAssemblyError       # Incompatible components in build_problem()
│   └── DuplicateScenarioError     # Scenario name already registered
├── SolverError                    # Solver strategy errors
│   └── SolverConvergenceError     # Solver failed to find optimal/feasible solution
└── VADKAssertionError             # Test assertion failure (also extends AssertionError)
```

### VADKError

```python
from flo_pro_adk.core.exceptions.vadk_error import VADKError
```

Base exception. Has an `error_code` property returning the class name.

### RegistrationError

```python
from flo_pro_adk.core.exceptions.agent_errors import RegistrationError
```

Raised when agent registration fails due to missing variable metadata. In E2E tests, `build_problem()` handles metadata wiring automatically. In unit tests, you must set `_var_metadata_registry[AgentClass]` before calling `register()`.

### SolverConvergenceError

```python
from flo_pro_adk.core.exceptions.solver_errors import SolverConvergenceError
```

Raised when the solver cannot find an optimal or feasible solution. The `status` attribute carries the raw solver status string (e.g., `"infeasible"`, `"unbounded"`).

```python
try:
    result = model.solve()
except SolverConvergenceError as e:
    print(f"Solver status: {e.status}")
```

### ScenarioNotFoundError

```python
from flo_pro_adk.core.exceptions.assembly_errors import ScenarioNotFoundError
```

Raised when a requested scenario name is not registered. The `scenario_name` attribute carries the missing name.

### VADKAssertionError

```python
from flo_pro_adk.core.exceptions.assertion_errors import VADKAssertionError
```

Test assertion failure. Extends both `VADKError` and Python's `AssertionError`, so pytest catches it naturally.


---

## Troubleshooting

### "My agent doesn't converge"

Convergence depends on your agent producing ADMM-compatible responses. Check these in order:

1. **Rho sensitivity.** Run `suite.run_unit()` — the rho sensitivity test should pass. If your agent ignores rho (doesn't track the consensus more closely as rho increases), ADMM cannot converge. The proximal term `Σ rho[i]/2 * (x[i] - z[i])²` must be part of your objective.

2. **Objective decomposition.** If you're using your own solver (not `SolverStrategy`), verify that `utility`, `subsidy`, and `proximal` are computed correctly. A common mistake: forgetting to include the subsidy term `Σ prices[i] * x[i]` in the objective, or getting the sign wrong.

3. **Solver tolerance.** If using Xpress, the solver framework accepts both `"optimal"` and `"feasible"` as success statuses. If neither appears, you get `SolverConvergenceError`. Check the `status` attribute on the exception for the raw solver status (e.g., `"infeasible"`, `"unbounded"`).

4. **Extreme values.** If your cost data has values spanning many orders of magnitude, numerical conditioning can prevent convergence. Normalize your data.

### "Solver says infeasible"

Your constraints are contradictory given the current consensus values. Common causes:

- **Minimum order > consensus.** If you constrain `x[i] >= min_order` but the consensus is pushing `x[i]` toward zero, the proximal penalty makes the problem infeasible. Consider using soft constraints (penalty terms) instead of hard lower bounds.
- **Capacity < demand.** If your capacity constraints are tighter than what the consensus requires, relax them or add slack variables.
- **Debug tip:** Remove all constraints and re-solve. If it works, add constraints back one at a time to find the conflict.

### "How do I see what prices/rho look like during a run?"

Use the lower-level test runners instead of `FloProSimulationSuite`:

```python
from flo_pro_adk.core.testing.unit_test_runner import run_unit_test
from flo_pro_adk.flopro.testing.flopro_data_generator import FloProSimulationDataGenerator
from flo_pro_adk.flopro.testing.flopro_scenarios import BASE_SCENARIO

result = run_unit_test(MyVendorAgent, FloProSimulationDataGenerator, BASE_SCENARIO)

# Inspect the inputs your agent received
print("Consensus:", result.consensus_vars)
print("Prices:", result.prices)
print("Rho:", result.rho)

# Inspect your agent's response
print("Preferred:", result.preferred_vars)
print("Objective:", result.objective)
```

For E2E runs, `E2ETestResult.final_state` contains the final consensus, prices, and residuals.

### "Shape mismatch" or "Missing variable group"

Your `solve()` is returning arrays with different shapes or group names than what `register()` declared. Ensure:

- `preferred_vars` has the same keys as `public_vars` (the group names).
- Each array in `preferred_vars` has the same length as the corresponding array in `public_vars`.
- You're not accidentally reshaping or slicing the arrays.

### "RegistrationError: No variable metadata registered"

This happens when running counterparty agents in unit tests without wiring the metadata registry. In E2E tests, `build_problem()` handles this automatically. For standalone unit tests of counterparty agents, you need:

```python
from flo_pro_adk.core.counterparty.counterparty_agent import _var_metadata_registry

_var_metadata_registry[RetailerAgent] = generator.generate_variable_group_metadata()
agent = RetailerAgent(agent_params={}, data_loader=InMemoryDataLoader(data))
```

Most vendors won't hit this — it only applies if you're instantiating `RetailerAgent` or `VendorAgent` directly outside the test suite.
