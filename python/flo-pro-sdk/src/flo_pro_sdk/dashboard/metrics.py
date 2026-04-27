"""Derived metrics for dashboard visualization.

DashboardMetricsComputer computes metrics that are not stored directly
in the Parquet datasets but are derived from the raw data. Results are
cached and only recomputed when new iterations appear.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider

logger = logging.getLogger(__name__)


class DashboardMetricsComputer:
    """Compute derived metrics from raw dashboard data.

    Args:
        provider: Data provider for reading Parquet datasets.
    """

    def __init__(self, provider: DashboardDataProvider) -> None:
        self._provider = provider
        self._metadata: Optional[Dict] = None
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_iterations: Dict[str, int] = {}

    def _get_subscriptions(self) -> Dict[str, Dict[str, List[int]]]:
        """Get subscription indices from problem metadata, loading lazily.

        Retries on each call if metadata was previously unavailable (e.g.
        registration not yet finalized during a live run).
        """
        if self._metadata is None or not self._metadata:
            self._metadata = self._provider.get_problem_metadata() or {}
        return self._metadata.get("subscriptions", {})

    def _get_group_offsets(self) -> Dict[str, int]:
        """Get the global offset for each variable group.

        Returns a dict mapping group_name -> slice_start so that
        group-local indices can be converted to global consensus vector
        positions via ``slice_start + local_index``.
        """
        if not self._metadata:
            return {}
        return {
            vg["name"]: vg["slice_start"]
            for vg in self._metadata.get("variable_groups", [])
        }

    def _cache_is_valid(self, cache_key: str, current_max_iter: int) -> bool:
        """Return True if cached result covers all available iterations."""
        return (
            cache_key in self._cache
            and self._cache_iterations.get(cache_key, -1) >= current_max_iter
        )

    def get_agent_residuals(self, agent_id: str) -> pd.DataFrame:
        """Compute per-iteration residual norm for an agent.

        residual[k] = norm(preferred_vars[k] - consensus_vars[agent_indices][k])

        Returns a DataFrame with columns [iteration, residual], or empty
        DataFrame if data is missing.
        """
        # All consensus vars in one scan — also used for cache check
        all_cv = self._provider.get_all_consensus_vars()
        if not all_cv:
            return pd.DataFrame()
        current_max = max(all_cv.keys())

        cache_key = f"agent_residuals:{agent_id}"
        if self._cache_is_valid(cache_key, current_max):
            return self._cache[cache_key]

        subscriptions = self._get_subscriptions()
        agent_subs = subscriptions.get(agent_id)
        if not agent_subs:
            logger.warning("No subscription indices for agent %s", agent_id)
            return pd.DataFrame()

        # Agent solutions with preferred_vars struct
        sol_df = self._provider.get_agent_solutions(
            agent_id=agent_id, columns=["preferred_vars"]
        )
        if sol_df.empty:
            return pd.DataFrame()

        # Build the concatenated index array from subscriptions (sorted by group name)
        # Indices in subscriptions are group-local; add group offset for global position
        sorted_groups = sorted(agent_subs.keys())
        group_offsets = self._get_group_offsets()
        global_indices: list[int] = []
        for group in sorted_groups:
            offset = group_offsets.get(group, 0)
            global_indices.extend(offset + idx for idx in agent_subs[group])

        rows = []
        for _, row in sol_df.iterrows():
            iteration = int(row["iteration"])
            cv = all_cv.get(iteration)
            if cv is None:
                continue

            # Extract preferred_vars from the struct (dict of group -> list)
            pv_struct = row["preferred_vars"]
            if pv_struct is None:
                continue

            # Concatenate preferred_vars in sorted group order
            pv_parts = []
            for group in sorted_groups:
                group_vals = pv_struct.get(group)
                if group_vals is not None:
                    pv_parts.extend(group_vals)

            pv = np.array(pv_parts)
            cv_slice = cv[global_indices]

            residual = float(np.linalg.norm(pv - cv_slice))
            rows.append({"iteration": iteration, "residual": residual})

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("iteration").reset_index(drop=True)
        self._cache[cache_key] = result
        self._cache_iterations[cache_key] = current_max
        return result

    def get_convergence_rate(self, window: int = 5) -> pd.DataFrame:
        """Compute sliding-window convergence rate of primal residual.

        rate[k] = (primal[k] - primal[k - window]) / window

        Returns a DataFrame with columns [iteration, convergence_rate].
        The first `window` iterations will have NaN rates.
        """
        convergence = self._provider.get_convergence_data()
        if convergence.empty:
            return pd.DataFrame()
        current_max = int(convergence["iteration"].max())

        cache_key = f"convergence_rate:{window}"
        if self._cache_is_valid(cache_key, current_max):
            return self._cache[cache_key]

        df = convergence[["iteration", "primal_residual"]].copy()
        df = df.sort_values("iteration").reset_index(drop=True)
        df["convergence_rate"] = df["primal_residual"].diff(periods=window) / window
        result = df[["iteration", "convergence_rate"]].copy()

        self._cache[cache_key] = result
        self._cache_iterations[cache_key] = current_max
        return result

    def get_total_objective(self) -> pd.DataFrame:
        """Compute total objective (sum of agent utilities) per iteration.

        Returns a DataFrame with columns [iteration, total_objective].
        """
        sol_df = self._provider.get_agent_solutions(columns=["utility"])
        if sol_df.empty:
            return pd.DataFrame()
        current_max = int(sol_df["iteration"].max())

        cache_key = "total_objective"
        if self._cache_is_valid(cache_key, current_max):
            return self._cache[cache_key]

        result = (
            sol_df.groupby("iteration", as_index=False)
            .agg(total_objective=("utility", "sum"))
            .sort_values("iteration")
            .reset_index(drop=True)
        )
        self._cache[cache_key] = result
        self._cache_iterations[cache_key] = current_max
        return result

    def get_agent_preferred_trajectories(
        self,
        agent_id: str,
        var_labels: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Get per-variable preferred value trajectories for an agent.

        Args:
            agent_id: The agent to query.
            var_labels: List of variable labels like ``"energy[42]"``.

        Returns:
            Dict mapping var_label -> DataFrame with columns
            [iteration, preferred, consensus].
        """
        sol_df = self._provider.get_agent_solutions(
            agent_id=agent_id,
            columns=["preferred_vars"],
        )
        if sol_df.empty:
            return {}

        all_cv = self._provider.get_all_consensus_vars()
        if not all_cv:
            return {}

        # Build label -> (group_name, local_index, global_index) mapping
        metadata = self._provider.get_problem_metadata()
        if not metadata:
            return {}
        var_groups = metadata.get("variable_groups", [])
        label_map: dict[str, tuple[str, int, int]] = {}
        for vg in sorted(var_groups, key=lambda v: v["slice_start"]):
            for j in range(vg["count"]):
                label = f"{vg['name']}[{j}]"
                if label in var_labels:
                    label_map[label] = (vg["name"], j, vg["slice_start"] + j)

        # Also need the agent's subscription mapping to find local offset
        subscriptions = self._get_subscriptions()
        agent_subs = subscriptions.get(agent_id, {})

        results: dict[str, pd.DataFrame] = {}
        for label in var_labels:
            info = label_map.get(label)
            if info is None:
                continue
            group_name, local_idx, global_idx = info

            # Find position of local_idx within agent's subscription for this group
            group_indices = agent_subs.get(group_name, [])
            if local_idx not in group_indices:
                continue
            sub_pos = group_indices.index(local_idx)

            rows = []
            for _, row in sol_df.iterrows():
                iteration = int(row["iteration"])
                pv_struct = row["preferred_vars"]
                if pv_struct is None:
                    continue
                group_vals = pv_struct.get(group_name)
                if group_vals is None or sub_pos >= len(group_vals):
                    continue
                cv = all_cv.get(iteration)
                cv_val = float(cv[global_idx]) if cv is not None else float("nan")
                rows.append(
                    {
                        "iteration": iteration,
                        "preferred": float(group_vals[sub_pos]),
                        "consensus": cv_val,
                    }
                )

            if rows:
                df = pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)
                results[label] = df

        return results

    def get_agent_objective_decomposition(self, agent_id: str) -> pd.DataFrame:
        """Get utility, subsidy, and proximal penalty per iteration for an agent.

        Returns a DataFrame with columns [iteration, utility, subsidy, proximal].
        """
        sol_df = self._provider.get_agent_solutions(
            agent_id=agent_id,
            columns=["utility", "subsidy", "proximal"],
        )
        if sol_df.empty:
            return pd.DataFrame()
        return sol_df.sort_values("iteration").reset_index(drop=True)
