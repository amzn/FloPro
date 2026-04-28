"""RetailerAgent — simulates Amazon for vendor E2E testing.

Optimization formulation (lead time = 0):
    maximize  utility + subsidy - proximal
    where:
        utility  = -Σ (H_ij * inv_ijt + P * backlog_ijt)
        subsidy and proximal are handled by SolverStrategy

Uses the unified SolverStrategy interface. The agent defines private
variables (inventory, backlog), constraints (balance), and cost
(holding + backlog). ADMM terms are handled automatically.
"""

from __future__ import annotations

from flo_pro_sdk.agent.agent_definition import Solution
from flo_pro_sdk.core.variables import Prices, PublicVarGroupName, PublicVarValues, RhoValues

from flo_pro_adk.core.counterparty.counterparty_agent import (
    CounterpartyAgent,
)
from flo_pro_adk.core.solver.solver_strategy import (
    SolverResult,
    SolverStrategy,
    build_solution,
)


class RetailerAgent(CounterpartyAgent):
    """Simulates Amazon retailer agent for vendor E2E testing.

    Uses SolverStrategy to build the optimization model. The strategy
    handles public variables and ADMM terms; this agent adds inventory
    balance constraints and holding/backlog costs.

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
        """Retailer optimization via SolverStrategy.

        Adds inventory/backlog variables, balance constraints, and
        holding/backlog costs. ADMM terms handled by the solver.
        """
        rd = self.data
        A = rd.n_asins
        J = rd.n_inbound_nodes
        T = rd.n_weeks

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

            inv = model.add_variables(name="inv", count=A * J * T, lb=0.0)
            backlog = model.add_variables(name="b", count=A * J * T, lb=0.0)
            inv_v = inv.refs
            backlog_v = backlog.refs

            def idx(i: int, j: int, t: int) -> int:
                return i * J * T + j * T + t

            # Inventory balance constraints
            constraints = []
            for i in range(A):
                for j in range(J):
                    constraints.append(
                        inv_v[idx(i, j, 0)] - backlog_v[idx(i, j, 0)]
                        == rd.initial_inventory[i, j]
                        + x[idx(i, j, 0)]
                        - rd.demand[i, j, 0]
                    )
                    for t in range(1, T):
                        constraints.append(
                            inv_v[idx(i, j, t)] - backlog_v[idx(i, j, t)]
                            == inv_v[idx(i, j, t - 1)]
                            - backlog_v[idx(i, j, t - 1)]
                            + x[idx(i, j, t)]
                            - rd.demand[i, j, t]
                        )
            model.add_constraint(constraints)

            # Private cost: -(holding + backlog)
            holding_cost = xp.Sum(
                rd.holding_costs[i, j] * inv_v[idx(i, j, t)]
                for i in range(A) for j in range(J) for t in range(T)
            )
            backlog_cost = xp.Sum(
                rd.backlog_penalty * backlog_v[idx(i, j, t)]
                for i in range(A) for j in range(J) for t in range(T)
            )
            model.set_private_cost(-(holding_cost + backlog_cost))

            results[group_name] = model.solve()

        return build_solution(results)
