from typing import List, Optional, Any
from dataclasses import dataclass
from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.state import State


@dataclass
class Problem:
    agents: List[AgentSpec]
    coordinator: CoordinatorSpec
    initial_state: State | Any
    max_iterations: int = 1000
    config: Optional[dict] = None
