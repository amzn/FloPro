"""Ray executor implementations for distributed agent operations."""

import time
from typing import Dict, List

import ray

from flo_pro_sdk.core.engine import (
    QueryExecutor,
    RegistrationExecutor,
    FinalizationExecutor,
)
from flo_pro_sdk.core.query import GetAgentInputFn, QueryResult
from flo_pro_sdk.core.state import State
from flo_pro_sdk.core.variables import PublicVarsMetadata


class RayQueryExecutor(QueryExecutor):
    """Queries agents in parallel using Ray actors."""

    def __init__(self, agent_actors: Dict[str, ray.actor.ActorHandle]) -> None:
        self._agent_actors = agent_actors

    def execute(
        self,
        agent_ids: List[str],
        state: State,
        get_agent_input_fn: GetAgentInputFn,
    ) -> Dict[str, QueryResult]:
        future_to_agent: Dict[ray.ObjectRef, str] = {}
        start_times: Dict[str, float] = {}

        for agent_id in agent_ids:
            actor = self._agent_actors.get(agent_id)
            if actor is None:
                raise KeyError(f"Agent {agent_id} not found")
            agent_input = get_agent_input_fn(agent_id, state)
            future = actor.query.remote(
                agent_input.agent_targets, agent_input.prices, agent_input.rho
            )
            future_to_agent[future] = agent_id
            start_times[agent_id] = time.time()

        results: Dict[str, QueryResult] = {}
        pending = list(future_to_agent.keys())

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for future in ready:
                agent_id = future_to_agent[future]
                solution = ray.get(future)
                query_time = time.time() - start_times[agent_id]
                results[agent_id] = QueryResult(
                    agent_id=agent_id,
                    solution=solution,
                    query_time=query_time,
                )

        return results


class RayRegistrationExecutor(RegistrationExecutor):
    def __init__(self, agent_actors: Dict[str, ray.actor.ActorHandle]) -> None:
        self._agent_actors = agent_actors

    def execute(self, agent_ids: List[str]) -> Dict[str, PublicVarsMetadata]:
        future_to_agent: Dict[ray.ObjectRef, str] = {}
        for agent_id in agent_ids:
            actor = self._agent_actors.get(agent_id)
            if actor is None:
                raise KeyError(f"Agent {agent_id} not found")
            future_to_agent[actor.register.remote()] = agent_id

        results: Dict[str, PublicVarsMetadata] = {}
        pending = list(future_to_agent.keys())
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            for future in ready:
                agent_id = future_to_agent[future]
                results[agent_id] = ray.get(future)
        return results


class RayFinalizationExecutor(FinalizationExecutor):
    def __init__(self, agent_actors: Dict[str, ray.actor.ActorHandle]) -> None:
        self._agent_actors = agent_actors

    def execute(self, agent_ids: List[str], final_state: State) -> None:
        futures = []
        for agent_id in agent_ids:
            actor = self._agent_actors.get(agent_id)
            if actor is None:
                raise KeyError(f"Agent {agent_id} not found")
            futures.append(actor.finalize.remote(final_state))
        ray.get(futures)
