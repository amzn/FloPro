from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.handlers import CoordinatorHandler
from flo_pro_sdk.core.query import GetAgentInputFn, QueryResult
from flo_pro_sdk.core.state import State
from flo_pro_sdk.core.variables import PublicVarsMetadata

if TYPE_CHECKING:
    from flo_pro_sdk.core.query import QueryStrategy
    from flo_pro_sdk.core.registry import AgentRegistry
    from flo_pro_sdk.core.state_store import StateStore


class QueryExecutor(ABC):
    """Interface for executing agent queries."""

    @abstractmethod
    def execute(
        self,
        agent_ids: List[str],
        state: State,
        get_agent_input_fn: GetAgentInputFn,
    ) -> Dict[str, QueryResult]:
        """Execute queries against the specified agents.

        For each agent, calls get_agent_input_fn(agent_id, state) to get
        the grouped AgentInput, then passes it to agent.solve().
        """
        pass


class RegistrationExecutor(ABC):
    @abstractmethod
    def execute(self, agent_ids: List[str]) -> Dict[str, PublicVarsMetadata]:
        pass


class FinalizationExecutor(ABC):
    @abstractmethod
    def execute(self, agent_ids: List[str], final_state: State) -> None:
        pass


class ExecutionEngine(ABC):
    @abstractmethod
    def allocate_agents(self, agent_specs: List[AgentSpec]) -> None:
        pass

    @abstractmethod
    def allocate_coordinator(
        self,
        coordinator_spec: CoordinatorSpec,
        query_strategy: "QueryStrategy",
        query_executor: "QueryExecutor",
        registry: "AgentRegistry",
    ) -> CoordinatorHandler:
        pass

    @abstractmethod
    def get_query_executor(self) -> "QueryExecutor":
        pass

    @abstractmethod
    def get_registration_executor(self) -> RegistrationExecutor:
        pass

    @abstractmethod
    def get_finalization_executor(self) -> FinalizationExecutor:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def get_state_store(self) -> "StateStore":
        pass

    def get_run_dir(self) -> Optional[Path]:
        """Return the persistence run directory, or None if not configured."""
        return None
