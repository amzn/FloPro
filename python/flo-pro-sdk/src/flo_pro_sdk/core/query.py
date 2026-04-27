from typing import Callable, Dict, Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass

from flo_pro_sdk.agent.agent_definition import Solution
from flo_pro_sdk.core.state import State
from flo_pro_sdk.core.variables import PublicVarValues, Prices, RhoValues

if TYPE_CHECKING:
    from flo_pro_sdk.core.engine import QueryExecutor
    from flo_pro_sdk.core.registry import AgentRegistry


@dataclass
class AgentInput:
    """Input data for agent solve. Grouped by variable name."""

    agent_targets: PublicVarValues
    prices: Prices
    rho: RhoValues


# Function that builds AgentInput for a given agent from the current state.
GetAgentInputFn = Callable[[str, State], AgentInput]


@dataclass
class QueryResult:
    agent_id: str
    solution: Solution
    query_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryStrategy(ABC):
    @abstractmethod
    def query(
        self,
        state: State,
        registry: "AgentRegistry",
        executor: "QueryExecutor",
    ) -> Dict[str, QueryResult]:
        pass


class DefaultQueryStrategy(QueryStrategy):
    """Default strategy that queries all registered agents."""

    def query(
        self,
        state: State,
        registry: "AgentRegistry",
        executor: "QueryExecutor",
    ) -> Dict[str, QueryResult]:
        agent_ids: list[str] = registry.list_agents()
        return executor.execute(
            agent_ids=agent_ids,
            state=state,
            get_agent_input_fn=registry.get_agent_input,
        )
