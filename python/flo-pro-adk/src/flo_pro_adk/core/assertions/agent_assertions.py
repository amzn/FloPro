"""Agent-scoped assertions — validate a single agent's solve() output."""

from __future__ import annotations

from typing import Callable

import numpy as np

from flo_pro_sdk.agent.agent_definition import Solution
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarValues,
    RhoValues,
)

from flo_pro_adk.core.exceptions.assertion_errors import (
    VADKAssertionError,
)


class AgentAssertions:
    """Assertions for a single agent's solve() behavior."""

    def assert_solution_valid(
        self,
        solution: Solution,
        public_vars: PublicVarValues,
        *,
        check_finiteness: bool = True,
        check_shape: bool = True,
    ) -> None:
        """Assert structural validity of a Solution."""
        for group_name, expected in public_vars.items():
            if group_name not in solution.preferred_vars:
                raise VADKAssertionError(
                    f"Missing variable group '{group_name}' in preferred_vars"
                )
            actual = solution.preferred_vars[group_name]
            if check_shape and actual.shape != expected.shape:
                raise VADKAssertionError(
                    f"Shape mismatch for '{group_name}': "
                    f"expected {expected.shape}, got {actual.shape}"
                )
            if check_finiteness and not np.all(np.isfinite(actual)):
                non_finite = np.where(~np.isfinite(actual))[0]
                raise VADKAssertionError(
                    f"Non-finite values in '{group_name}' at indices: "
                    f"{non_finite[:10].tolist()}"
                )

    def assert_l2_distance_decreases(
        self,
        solve_fn: Callable[[PublicVarValues, Prices, RhoValues], Solution],
        public_vars: PublicVarValues,
        prices: Prices,
        rho_low: RhoValues,
        rho_high: RhoValues,
    ) -> None:
        """Assert L2 distance to consensus decreases as rho increases."""
        sol_low = solve_fn(public_vars, prices, rho_low)
        sol_high = solve_fn(public_vars, prices, rho_high)

        dist_low = sum(
            float(np.sum((sol_low.preferred_vars[g] - public_vars[g]) ** 2))
            for g in public_vars
        )
        dist_high = sum(
            float(np.sum((sol_high.preferred_vars[g] - public_vars[g]) ** 2))
            for g in public_vars
        )

        if dist_high > dist_low:
            raise VADKAssertionError(
                f"L2 distance did not decrease with higher rho: "
                f"low_rho={dist_low:.6f}, high_rho={dist_high:.6f}"
            )

    def assert_deterministic(
        self,
        solve_fn: Callable[[PublicVarValues, Prices, RhoValues], Solution],
        public_vars: PublicVarValues,
        prices: Prices,
        rho: RhoValues,
        *,
        n_calls: int = 3,
    ) -> None:
        """Assert solve() is deterministic — same inputs produce same output."""
        solutions = [solve_fn(public_vars, prices, rho) for _ in range(n_calls)]
        first = solutions[0]
        for i, sol in enumerate(solutions[1:], start=1):
            for group_name in first.preferred_vars:
                if not np.array_equal(
                    first.preferred_vars[group_name],
                    sol.preferred_vars[group_name],
                ):
                    raise VADKAssertionError(
                        f"Non-deterministic output for '{group_name}' "
                        f"between call 0 and call {i}"
                    )
