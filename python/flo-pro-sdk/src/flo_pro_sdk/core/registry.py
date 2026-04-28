from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from flo_pro_sdk.core.query import AgentInput
    from flo_pro_sdk.core.state import State
from numpy import ndarray

from flo_pro_sdk.core.variables import (
    PublicVarGroupMetadata,
    PublicVarGroupName,
    PublicVarsMetadata,
)
from flo_pro_sdk.core.var_layout import VarLayout

SubscriptionIndices = Dict[PublicVarGroupName, Dict[str, ndarray]]


@dataclass
class AgentRegistryEntry:
    agent_id: str
    subscribed_vars: PublicVarsMetadata
    metadata: Optional[Dict[str, Any]] = None


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, AgentRegistryEntry] = {}
        self._global_public_var_metadata: PublicVarsMetadata = {}
        self._subscription_indices: SubscriptionIndices = {}
        self._layout: Optional[VarLayout] = None

    def register_agent(
        self,
        agent_id: str,
        subscribed_vars: PublicVarsMetadata,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._agents[agent_id] = AgentRegistryEntry(
            agent_id=agent_id, subscribed_vars=subscribed_vars, metadata=metadata
        )

    def finalize_registration(self) -> None:
        """Build global var set by merging metadata, then compute agent indices.

        # TODO: Optimize for millions of variables — the current merge/reset_index
        # approach is O(n*m) per agent. Consider set-based or index-based lookups.
        # TODO: Validate that all agents within a group share the same schema
        # (same columns) and that no agent has duplicate rows in its metadata.
        """
        # Step 1: Collect and merge metadata per group across all agents
        global_dfs: Dict[PublicVarGroupName, pd.DataFrame] = {}
        for entry in self._agents.values():
            for group_name, group_meta in entry.subscribed_vars.items():
                if group_name in global_dfs:
                    global_dfs[group_name] = (
                        pd.concat([global_dfs[group_name], group_meta.var_metadata])
                        .drop_duplicates()
                        .reset_index(drop=True)
                    )
                else:
                    global_dfs[group_name] = (
                        group_meta.var_metadata.drop_duplicates().reset_index(drop=True)
                    )

        # Sort once so global positions are deterministic regardless of registration order
        for group_name in global_dfs:
            df: pd.DataFrame = global_dfs[group_name]
            global_dfs[group_name] = df.sort_values(by=list(df.columns)).reset_index(
                drop=True
            )

        for group_name in sorted(global_dfs.keys()):
            self._global_public_var_metadata[group_name] = PublicVarGroupMetadata(
                name=group_name,
                var_metadata=global_dfs[group_name],
            )

        # Step 2: Build VarLayout from global sizes
        offset: int = 0
        group_slices: Dict[PublicVarGroupName, slice] = {}
        for group_name in sorted(global_dfs.keys()):
            size = len(global_dfs[group_name])
            group_slices[group_name] = slice(offset, offset + size)
            offset += size

        self._layout = VarLayout(group_slices=group_slices, total_size=offset)

        # Step 3: For each agent, find indices into the global merged DataFrame
        _ROW_POS = "__flo_row_pos__"
        for agent_id, entry in self._agents.items():
            agent_indices: Dict[PublicVarGroupName, ndarray] = {}
            for group_name in sorted(group_slices.keys()):
                if group_name in entry.subscribed_vars:
                    global_df = global_dfs[group_name]
                    agent_df = entry.subscribed_vars[group_name].var_metadata
                    if _ROW_POS in global_df.columns:
                        raise ValueError(
                            f"var_metadata for group '{group_name}' must not contain "
                            f"reserved column '{_ROW_POS}'"
                        )
                    positioned = global_df.assign(
                        **{_ROW_POS: np.arange(len(global_df))}
                    )
                    merged = agent_df.merge(positioned, how="inner")
                    indices: ndarray = np.asarray(merged[_ROW_POS].values, dtype=int)
                    agent_indices[group_name] = indices
                    self._subscription_indices.setdefault(group_name, {})[agent_id] = (
                        indices
                    )

            self._layout.register_agent(agent_id, agent_indices)

    def get_layout(self) -> VarLayout:
        if self._layout is None:
            raise RuntimeError("finalize_registration() must be called first")
        return self._layout

    def get_agent_input(self, agent_id: str, state: "State") -> "AgentInput":
        """Build grouped AgentInput from flat state for a specific agent."""
        from flo_pro_sdk.core.query import AgentInput

        layout = self.get_layout()
        return AgentInput(
            agent_targets=layout.unflatten_from_global(
                agent_id, state.get_agent_targets(agent_id)
            ),
            prices=layout.unflatten_from_global(
                agent_id, state.get_agent_prices(agent_id)
            ),
            rho=layout.unflatten_from_global(agent_id, state.get_rho(agent_id)),
        )

    def get_subscribed_vars(self, agent_id: str) -> PublicVarsMetadata:
        entry = self._agents.get(agent_id)
        if not entry:
            raise KeyError(f"Agent {agent_id} not found in registry")
        return entry.subscribed_vars

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def get_agent_indices_by_var_group(
        self, var_name: PublicVarGroupName
    ) -> Dict[str, ndarray]:
        return self._subscription_indices.get(var_name, {})

    def get_all_subscribed_vars(self) -> PublicVarsMetadata:
        return self._global_public_var_metadata

    def get_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        entry = self._agents.get(agent_id)
        if not entry:
            raise KeyError(f"Agent {agent_id} not found in registry")
        return entry.metadata
