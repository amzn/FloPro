"""VendorAgent — simulates a generic vendor for Amazon agent team E2E testing.

Bounded inventory model optimization (lead time = 0):
    maximize  utility + subsidy - proximal
    where:
        utility = -Σ (holding_cost * inventory + transport_cost * shipping)
        subsidy and proximal are handled by SolverStrategy

    s.t.
        v_ai,t+1 = v_ait + y_ait - Σ_j z_aijt   (inventory balance)
        lower_ait <= v_ait <= upper_ait            (inventory bounds)
        x_ajt = Σ_i z_aijt                        (lead time / PO = shipments)
        z_aijt >= 0, v_ait >= 0

Uses the unified SolverStrategy interface. The agent defines private
variables (shipping, inventory), constraints (balance, bounds, lead time),
and cost (holding + transportation). ADMM terms are handled automatically.
"""

from __future__ import annotations

from flo_pro_sdk.agent.agent_definition import Solution
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarGroupName,
    PublicVarValues,
    RhoValues,
)

from flo_pro_adk.core.counterparty.counterparty_agent import (
    CounterpartyAgent,
)
from flo_pro_adk.core.solver.solver_strategy import (
    SolverResult,
    SolverStrategy,
    build_solution,
)


class VendorAgent(CounterpartyAgent):
    """Simulates a generic vendor agent for Amazon agent team E2E testing.

    Uses SolverStrategy to build the optimization model. The strategy
    handles public variables and ADMM terms; this agent adds shipping
    and inventory variables, balance/bound/lead-time constraints, and
    holding + transportation costs.

    Defaults to XpressSolverStrategy. Pass ``solver=`` to ``__init__``
    (or override ``_default_solver``) to use a different backend.
    """

    @classmethod
    def _default_solver(cls) -> SolverStrategy:
        # Lazy import keeps the optional xpress dependency out of the
        # import graph unless this agent is actually constructed without
        # an explicit solver.
        from flo_pro_adk.core.solver.xpress_solver_strategy import (
            XpressSolverStrategy,
        )
        return XpressSolverStrategy()

    def solve(
        self,
        public_vars: PublicVarValues,
        prices: Prices,
        rho: RhoValues,
    ) -> Solution:
        """Vendor optimization via SolverStrategy.

        Adds shipping/inventory variables, balance/bound/lead-time
        constraints, and holding + transportation costs.
        ADMM terms handled by the solver.
        """
        vd = self.data
        n_asins = vd.n_asins
        n_warehouses = vd.n_vendor_warehouses
        n_nodes = vd.n_inbound_nodes
        n_weeks = vd.n_weeks

        results: dict[PublicVarGroupName, SolverResult] = {}

        for group_name, z in public_vars.items():
            model = self.solver.create_model(
                consensus=z,
                prices=prices[group_name],
                rho=rho[group_name],
                public_group_metadata=self.public_vars_metadata[group_name],
            )

            xp = model.expr
            x = model.public_vars.refs

            # Private variables
            # z_aijt: shipping quantities from warehouse i to node j
            ship = model.add_variables(
                name="z", count=n_asins * n_warehouses * n_nodes * n_weeks, lb=0.0,
            )
            # v_ait: on-hand inventory at warehouse i
            inv = model.add_variables(
                name="v", count=n_asins * n_warehouses * n_weeks, lb=0.0,
            )
            ship_v = ship.refs
            inv_v = inv.refs

            # Index helpers
            def x_idx(a: int, j: int, t: int) -> int:
                return a * n_nodes * n_weeks + j * n_weeks + t

            def ship_idx(a: int, i: int, j: int, t: int) -> int:
                return a * n_warehouses * n_nodes * n_weeks + i * n_nodes * n_weeks + j * n_weeks + t

            def inv_idx(a: int, i: int, t: int) -> int:
                return a * n_warehouses * n_weeks + i * n_weeks + t

            # ---- Constraints ----
            constraints: list = []

            # Inventory balance: v_ai,t+1 = v_ait + y_ait - Σ_j z_aijt
            for a in range(n_asins):
                for i in range(n_warehouses):
                    for t in range(n_weeks - 1):
                        constraints.append(
                            inv_v[inv_idx(a, i, t + 1)]
                            == inv_v[inv_idx(a, i, t)]
                            + vd.quantity_to_procure_y_ait[a, i, t]
                            - xp.Sum(
                                ship_v[ship_idx(a, i, j, t)]
                                for j in range(n_nodes)
                            )
                        )

            # Inventory bounds: lower <= v_ait <= upper
            for a in range(n_asins):
                for i in range(n_warehouses):
                    for t in range(n_weeks):
                        constraints.append(
                            inv_v[inv_idx(a, i, t)] >= vd.lower_bound_inv_ait[a, i, t]
                        )
                        constraints.append(
                            inv_v[inv_idx(a, i, t)] <= vd.upper_bound_inv_ait[a, i, t]
                        )

            # Lead time (zero lead time): x_ajt = Σ_i z_aijt
            for a in range(n_asins):
                for j in range(n_nodes):
                    for t in range(n_weeks):
                        constraints.append(
                            x[x_idx(a, j, t)]
                            == xp.Sum(
                                ship_v[ship_idx(a, i, j, t)]
                                for i in range(n_warehouses)
                            )
                        )

            model.add_constraint(constraints)

            # ---- Private cost: -(holding + transportation) ----
            holding_cost = xp.Sum(
                vd.holding_cost_h_it[i, t] * inv_v[inv_idx(a, i, t)]
                for a in range(n_asins) for i in range(n_warehouses) for t in range(n_weeks)
            )
            transport_cost = xp.Sum(
                vd.transportation_cost_r_ijt[i, j, t] * ship_v[ship_idx(a, i, j, t)]
                for a in range(n_asins) for i in range(n_warehouses) for j in range(n_nodes) for t in range(n_weeks)
            )
            model.set_private_cost(-(holding_cost + transport_cost))

            results[group_name] = model.solve()

        return build_solution(results)
