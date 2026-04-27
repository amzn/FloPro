import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

from flo_pro_sdk.core.compute import ComputeSpec
from flo_pro_sdk.core.types import JsonValue
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarsMetadata,
    PublicVarValues,
    RhoValues,
)


@dataclass(frozen=True)
class Objective:
    utility: float
    subsidy: float
    proximal: float

    def total(self) -> float:
        return self.utility + self.subsidy - self.proximal


@dataclass
class AgentSpec:
    agent_class: type["AgentDefinition"]
    agent_id: str
    agent_params: dict[str, JsonValue] | None = None
    compute: Optional[ComputeSpec] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        try:
            self.agent_params is None or json.dumps(self.agent_params)
        except TypeError as e:
            raise TypeError(f"agent_params must be JSON-serializable: {e}")


@dataclass
class Solution:
    preferred_vars: PublicVarValues
    objective: Objective
    metadata: Optional[Dict[str, Any]] = None


class AgentDefinition(ABC):
    @classmethod
    def create(cls, agent_params: dict[str, JsonValue]) -> "AgentDefinition":
        """Factory method used to instantiate the agent. Override for custom construction logic."""
        return cls(**agent_params)  # type: ignore[call-arg]

    @abstractmethod
    def solve(
        self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues
    ) -> Solution:
        pass

    def register(self) -> PublicVarsMetadata:
        return {}

    def finalize(self, final_state: Any) -> None:
        pass
