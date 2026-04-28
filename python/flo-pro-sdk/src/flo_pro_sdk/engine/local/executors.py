"""Local executor implementations."""

from typing import Dict, List

from flo_pro_sdk.core.engine import (
    QueryExecutor,
    RegistrationExecutor,
    FinalizationExecutor,
)
from flo_pro_sdk.core.query import GetAgentInputFn, QueryResult
from flo_pro_sdk.core.state import State
from flo_pro_sdk.core.variables import PublicVarsMetadata
from flo_pro_sdk.engine.local.handles import LocalAgentHandles


class LocalQueryExecutor(QueryExecutor):
    def __init__(self, handles: Dict[str, LocalAgentHandles]) -> None:
        self._handles = handles

    def execute(
        self,
        agent_ids: List[str],
        state: State,
        get_agent_input_fn: GetAgentInputFn,
    ) -> Dict[str, QueryResult]:
        results: Dict[str, QueryResult] = {}
        for agent_id in agent_ids:
            agent_handles = self._handles.get(agent_id)
            if agent_handles is None:
                raise KeyError(f"Agent {agent_id} not found")
            agent_input = get_agent_input_fn(agent_id, state)
            solution = agent_handles.query(
                agent_input.agent_targets, agent_input.prices, agent_input.rho
            )
            results[agent_id] = QueryResult(agent_id=agent_id, solution=solution)
        return results


class LocalRegistrationExecutor(RegistrationExecutor):
    def __init__(self, handles: Dict[str, LocalAgentHandles]) -> None:
        self._handles = handles

    def execute(self, agent_ids: List[str]) -> Dict[str, PublicVarsMetadata]:
        results: Dict[str, PublicVarsMetadata] = {}
        for agent_id in agent_ids:
            agent_handles = self._handles.get(agent_id)
            if agent_handles is None:
                raise KeyError(f"Agent {agent_id} not found")
            results[agent_id] = agent_handles.register()
        return results


class LocalFinalizationExecutor(FinalizationExecutor):
    def __init__(self, handles: Dict[str, LocalAgentHandles]) -> None:
        self._handles = handles

    def execute(self, agent_ids: List[str], final_state: State) -> None:
        for agent_id in agent_ids:
            agent_handles = self._handles.get(agent_id)
            if agent_handles is None:
                raise KeyError(f"Agent {agent_id} not found")
            agent_handles.finalize(final_state)
