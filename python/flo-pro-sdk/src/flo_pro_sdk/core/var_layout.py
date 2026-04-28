"""VarLayout: stores global layout and all agent subscription mappings."""

__all__ = ["VarLayout"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from numpy import ndarray

from flo_pro_sdk.core.variables import PublicVarGroupName
from flo_pro_sdk.core.types import AgentId


@dataclass(eq=False)
class VarLayout:
    """Translates between grouped Dict[PublicVarGroupName, ndarray] and
    flat global-sized ndarray for any registered agent."""

    group_slices: Dict[PublicVarGroupName, slice]
    total_size: int
    _agent_indices: Dict[AgentId, Dict[PublicVarGroupName, ndarray]] = field(
        default_factory=dict, repr=False
    )
    _agent_global_indices: Dict[AgentId, ndarray] = field(
        default_factory=dict, repr=False
    )
    _subscription_counts_cache: Optional[ndarray] = field(default=None, repr=False)

    def unflatten_from_global(
        self, agent_id: AgentId, global_flat: ndarray
    ) -> Dict[PublicVarGroupName, ndarray]:
        """Slice global flat array to agent's subscribed indices, return grouped dict."""
        indices: Dict[PublicVarGroupName, ndarray] = self._agent_indices[agent_id]
        result: Dict[PublicVarGroupName, ndarray] = {}
        for group_name, group_idx in indices.items():
            s: slice = self.group_slices[group_name]
            if s.start is None:
                raise ValueError(f"group_slices['{group_name}'] has no start")
            result[group_name] = global_flat[s.start + group_idx].copy()
        return result

    def flatten_to_global(
        self, agent_id: AgentId, grouped: Dict[PublicVarGroupName, ndarray]
    ) -> ndarray:
        """Place agent's grouped values into global-sized flat array at correct indices.

        Non-subscribed indices are zero-filled. When averaging across agents with
        different subscriptions, callers should weight by subscription count per index.
        """
        result: ndarray = np.zeros(self.total_size)
        indices: Dict[PublicVarGroupName, ndarray] = self._agent_indices[agent_id]
        for group_name, values in grouped.items():
            s: slice = self.group_slices[group_name]
            if s.start is None:
                raise ValueError(f"group_slices['{group_name}'] has no start")
            result[s.start + indices[group_name]] = values
        return result

    def get_global_indices(self, agent_id: AgentId) -> ndarray:
        """Return the global flat-vector indices this agent subscribes to."""
        return self._agent_global_indices[agent_id]

    def register_agent(
        self, agent_id: AgentId, agent_indices: Dict[PublicVarGroupName, ndarray]
    ) -> None:
        """Register an agent's subscription mapping and pre-compute global indices."""
        sorted_indices: Dict[PublicVarGroupName, ndarray] = dict(
            sorted(agent_indices.items())
        )
        self._agent_indices[agent_id] = sorted_indices
        all_idx: List[ndarray] = []
        for group_name, group_idx in sorted_indices.items():
            s: slice = self.group_slices[group_name]
            if s.start is None:
                raise ValueError(f"group_slices['{group_name}'] has no start")
            all_idx.append(s.start + group_idx)
        self._agent_global_indices[agent_id] = (
            np.concatenate(all_idx) if all_idx else np.array([], dtype=int)
        )

    def get_subscription_counts(self) -> ndarray:
        """Return per-variable count of how many agents subscribe to each index.

        Result is cached after the first call — subscription counts are fixed
        after finalize_registration() and never change.
        """
        if self._subscription_counts_cache is None:
            counts = np.zeros(self.total_size, dtype=int)
            for idx in self._agent_global_indices.values():
                counts[idx] += 1
            self._subscription_counts_cache = counts
        return self._subscription_counts_cache
