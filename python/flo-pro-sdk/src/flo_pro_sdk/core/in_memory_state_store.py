"""In-memory state store for single-process execution.

Provides a bounded, dict-based store with O(1) lookup for recent iterations.
Oldest iterations are evicted when cache_size is exceeded.
"""

from typing import Dict, List, Optional

from flo_pro_sdk.core.state import AgentPlan, State
from flo_pro_sdk.core.state_store import StateStore


class InMemoryStateStore(StateStore):
    """In-memory state store for single-process execution.

    Maintains a bounded cache of recent iterations with O(1) lookup.
    Evicts oldest iterations when cache_size is exceeded.
    """

    def __init__(self, cache_size: int = 5) -> None:
        self._cache_size = cache_size
        self._states: Dict[int, State] = {}
        self._plans: Dict[int, Dict[str, AgentPlan]] = {}
        self._iteration_order: List[int] = []

    def store_state(
        self,
        iteration: int,
        state: State,
        timestamp: float | None = None,
        *,
        blocking: bool = False,
    ) -> None:
        if iteration not in self._states:
            self._iteration_order.append(iteration)
        self._states[iteration] = state
        self._evict_if_needed()

    def store_agent_plan(self, iteration: int, agent_id: str, plan: AgentPlan) -> None:
        if iteration not in self._plans:
            self._plans[iteration] = {}
        self._plans[iteration][agent_id] = plan

    def store_agent_plans(self, iteration: int, plans: Dict[str, AgentPlan]) -> None:
        if iteration not in self._plans:
            self._plans[iteration] = {}
        self._plans[iteration].update(plans)

    def get_state(self, iteration: int) -> Optional[State]:
        return self._states.get(iteration)

    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional[AgentPlan]:
        return self._plans.get(iteration, {}).get(agent_id)

    def get_recent_states(self, count: int) -> List[State]:
        recent_iters = self._iteration_order[-count:]
        return [self._states[i] for i in reversed(recent_iters) if i in self._states]

    def flush(self) -> None:
        pass

    def _evict_if_needed(self) -> None:
        while len(self._iteration_order) > self._cache_size:
            oldest = self._iteration_order.pop(0)
            self._states.pop(oldest, None)
            self._plans.pop(oldest, None)
