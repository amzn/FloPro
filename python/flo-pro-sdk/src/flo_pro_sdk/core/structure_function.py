"""Structure functions for z-update in ADMM coordinators."""

__all__ = [
    "StructureFunction",
    "AveragingFunction",
    "ZeroFunction",
    "StructureFunctionSpec",
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from numpy import ndarray

from flo_pro_sdk.core.var_layout import VarLayout
from flo_pro_sdk.core.variables import PublicVarsMetadata
from flo_pro_sdk.core.types import JsonValue

if TYPE_CHECKING:
    from flo_pro_sdk.core.state import State


class StructureFunction(ABC):
    """Abstract base for z-update structure functions. Operates on flat arrays."""

    def __init__(
        self, layout: VarLayout, metadata: Optional[PublicVarsMetadata] = None
    ) -> None:
        self.layout = layout
        self.metadata = metadata

    @abstractmethod
    def solve(self, state: "State") -> ndarray:
        pass


class AveragingFunction(StructureFunction):
    """Default for consensus: rho-weighted average z = sum(rho_i * x_i) / sum(rho_i),
    weighted by subscription count per index to avoid dilution from non-subscribed agents."""

    def solve(self, state: "State") -> ndarray:
        """All State arrays must be global-sized (from flatten_to_global)."""
        n = len(state.consensus_vars)
        numerator = np.zeros(n)
        denominator = np.zeros(n)
        for aid in state.agent_ids:
            idx = self.layout.get_global_indices(aid)
            rho = state.get_rho(aid)
            numerator[idx] += rho[idx] * state.get_agent_preferred_vars(aid)[idx]
            denominator[idx] += rho[idx]
        # Safe division: indices with no subscribers get 0 (not NaN).
        return np.divide(
            numerator, denominator, out=np.zeros(n), where=denominator != 0
        )


class ZeroFunction(StructureFunction):
    """Default for sharing/exchange: z = 0."""

    def solve(self, state: "State") -> ndarray:
        return np.zeros_like(state.consensus_vars)


class StructureFunctionSpec:
    """Specification for instantiating a structure function."""

    def __init__(
        self,
        definition_class: type[StructureFunction],
        config: Optional[Dict[str, JsonValue]] = None,
    ) -> None:
        self.definition_class = definition_class
        self.config = config or {}

    def instantiate(
        self, layout: VarLayout, metadata: PublicVarsMetadata
    ) -> StructureFunction:
        return self.definition_class(layout=layout, metadata=metadata, **self.config)
