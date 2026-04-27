"""Coordination-scoped assertions — validate multi-agent coordination outcomes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from flo_pro_sdk.core.state import State

from flo_pro_adk.core.exceptions.assertion_errors import (
    VADKAssertionError,
)


class CoordinationAssertions:
    """Assertions for multi-agent ADMM coordination results."""

    def assert_convergence(
        self,
        final_state: State,
        *,
        primal_tol: float = 1e-4,
        dual_tol: float = 1e-4,
    ) -> None:
        """Assert the coordination converged within tolerance."""
        residuals = final_state.residuals
        if residuals is None:
            raise VADKAssertionError("No residuals in final state")
        if residuals.primal > primal_tol:
            raise VADKAssertionError(
                f"Primal residual {residuals.primal:.6f} exceeds tolerance {primal_tol}"
            )
        if residuals.dual > dual_tol:
            raise VADKAssertionError(
                f"Dual residual {residuals.dual:.6f} exceeds tolerance {dual_tol}"
            )

    def assert_gap_narrowing(
        self,
        states: Sequence[State],
        *,
        window: int = 10,
    ) -> None:
        """Assert primal-dual gap narrows over a sliding window."""
        if len(states) < window * 2:
            return

        gaps = []
        for s in states:
            r = s.residuals
            if r is not None:
                gaps.append(r.primal + r.dual)

        if len(gaps) < window * 2:
            return

        early_avg = np.mean(gaps[:window])
        late_avg = np.mean(gaps[-window:])

        if late_avg >= early_avg:
            raise VADKAssertionError(
                f"Gap did not narrow: early_avg={early_avg:.6f}, late_avg={late_avg:.6f}"
            )

    def assert_price_stabilization(
        self,
        states: Sequence[State],
        *,
        tail_fraction: float = 0.2,
        max_oscillation: float = 0.01,
    ) -> None:
        """Assert prices stabilize in the tail of the coordination run."""
        if len(states) < 3:
            return

        tail_start = max(1, int(len(states) * (1 - tail_fraction)))
        tail_states = states[tail_start:]

        for aid in tail_states[0].agent_ids:
            price_diffs = []
            for i in range(1, len(tail_states)):
                prev = tail_states[i - 1].get_agent_prices(aid)
                curr = tail_states[i].get_agent_prices(aid)
                price_diffs.append(float(np.max(np.abs(curr - prev))))

            max_diff = max(price_diffs) if price_diffs else 0.0
            if max_diff > max_oscillation:
                raise VADKAssertionError(
                    f"Prices for agent '{aid}' oscillating in tail: "
                    f"max_diff={max_diff:.6f} > {max_oscillation}"
                )
