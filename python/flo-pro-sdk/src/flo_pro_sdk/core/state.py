from dataclasses import dataclass
from typing import Any, Dict, Optional, List, TypeVar, TYPE_CHECKING
from abc import ABC, abstractmethod
from enum import Enum

from numpy import ndarray

from flo_pro_sdk.agent.agent_definition import Solution
from flo_pro_sdk.core.types import AgentId
from flo_pro_sdk.core.variables import Residuals

if TYPE_CHECKING:
    from flo_pro_sdk.core.registry import AgentRegistry


@dataclass
class CoreState:
    """Global state shared by all problem types. Lightweight, easily serialized."""

    iteration: int
    consensus_vars: ndarray  # flat z
    residuals: Optional[Residuals] = None
    metadata: Optional[Dict[str, Any]] = None


class State(ABC):
    """Abstract state interface for all coordinator types.

    Flat-array read interface that unifies consensus and sharing problems.
    Hides whether prices are per-agent or global, whether storage is grouped or flat.
    """

    @abstractmethod
    def get_core_state(self) -> CoreState:
        pass

    @abstractmethod
    def get_agent_prices(self, agent_id: AgentId) -> ndarray:
        # TODO: Consider type alias and sparse array return for memory efficiency.
        pass

    @abstractmethod
    def get_agent_targets(self, agent_id: AgentId) -> ndarray:
        pass

    @abstractmethod
    def get_agent_preferred_vars(self, agent_id: AgentId) -> ndarray:
        pass

    @abstractmethod
    def get_rho(self, agent_id: AgentId) -> ndarray:
        pass

    @property
    @abstractmethod
    def agent_ids(self) -> List[AgentId]:
        pass

    @property
    def iteration(self) -> int:
        return self.get_core_state().iteration

    @property
    def consensus_vars(self) -> ndarray:
        return self.get_core_state().consensus_vars

    @property
    def residuals(self) -> Optional[Residuals]:
        return self.get_core_state().residuals

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self.get_core_state().metadata


class ConsensusState(State):
    """State for consensus problems. All data stored as global-sized flat arrays."""

    def __init__(
        self,
        iteration: int,
        consensus_vars: ndarray,
        agent_preferred_vars: Dict[AgentId, ndarray],
        prices: Dict[AgentId, ndarray],
        rho: Dict[AgentId, ndarray],
        residuals: Optional[Residuals] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._core = CoreState(iteration, consensus_vars, residuals, metadata)
        self._agent_preferred_vars = agent_preferred_vars
        self._prices = prices
        self._rho = rho

    def get_core_state(self) -> CoreState:
        return self._core

    def get_agent_targets(self, agent_id: AgentId) -> ndarray:
        return self._core.consensus_vars.copy()

    def get_agent_preferred_vars(self, agent_id: AgentId) -> ndarray:
        return self._agent_preferred_vars[agent_id]

    def get_agent_prices(self, agent_id: AgentId) -> ndarray:
        return self._prices[agent_id]

    def get_rho(self, agent_id: AgentId) -> ndarray:
        return self._rho[agent_id]

    @property
    def agent_ids(self) -> List[AgentId]:
        return list(self._agent_preferred_vars.keys())


class SharingState(State):
    """State for sharing problems. Prices are global (flat), targets are per-agent (flat)."""

    def __init__(
        self,
        iteration: int,
        consensus_vars: ndarray,  # flat z̄
        agent_preferred_vars: Dict[AgentId, ndarray],  # flat x_i per agent
        agent_targets: Dict[AgentId, ndarray],  # flat z_i per agent
        prices: ndarray,  # flat global λ
        rho: Dict[AgentId, ndarray],  # flat rho per agent
        residuals: Optional[Residuals] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._core = CoreState(iteration, consensus_vars, residuals, metadata)
        self._agent_preferred_vars = agent_preferred_vars
        self._agent_targets = agent_targets
        self._prices = prices
        self._rho = rho

    def get_core_state(self) -> CoreState:
        return self._core

    def get_agent_targets(self, agent_id: AgentId) -> ndarray:
        return self._agent_targets[agent_id]

    def get_agent_preferred_vars(self, agent_id: AgentId) -> ndarray:
        return self._agent_preferred_vars[agent_id]

    def get_agent_prices(self, agent_id: AgentId) -> ndarray:
        return self._prices  # Global λ — same for all agents

    def get_rho(self, agent_id: AgentId) -> ndarray:
        return self._rho[agent_id]

    @property
    def agent_ids(self) -> List[AgentId]:
        return list(self._agent_preferred_vars.keys())


# ── Other state-related types ──────────────────────────────────────────────


@dataclass
class IterationResult:
    iteration: int
    converged: bool


@dataclass
class AgentPlan:
    agent_id: str
    iteration: int
    solution: Solution


class StateLoader(ABC):
    @abstractmethod
    def load(self, registry: "AgentRegistry | None" = None) -> State:
        pass


class ObjectType(Enum):
    STATE = "state"
    AGENT_PLAN = "agent_plan"


@dataclass(frozen=True)
class StateKey:
    object_type: ObjectType
    iteration: int

    def to_tuple(self) -> tuple:
        return (self.object_type.value, self.iteration)


@dataclass(frozen=True)
class AgentPlanKey:
    object_type: ObjectType
    iteration: int
    agent_id: str

    def to_tuple(self) -> tuple:
        return (self.object_type.value, self.iteration, self.agent_id)


TrackerKey = StateKey | AgentPlanKey
T = TypeVar("T")


class StateTracker(ABC):
    @abstractmethod
    def store_state(self, key: StateKey, state: State) -> None:
        pass

    @abstractmethod
    def store_agent_plan(self, key: AgentPlanKey, plan: AgentPlan) -> None:
        pass

    @abstractmethod
    def retrieve_state(self, key: StateKey) -> Optional[State]:
        pass

    @abstractmethod
    def retrieve_agent_plan(self, key: AgentPlanKey) -> Optional[AgentPlan]:
        pass

    @abstractmethod
    def query_states(self, **filters) -> List[State]:
        pass

    @abstractmethod
    def query_agent_plans(self, **filters) -> List[AgentPlan]:
        pass


class InMemoryStateTracker(StateTracker):
    def __init__(self) -> None:
        self._states: Dict[StateKey, State] = {}
        self._agent_plans: Dict[AgentPlanKey, AgentPlan] = {}

    def store_state(self, key: StateKey, state: State) -> None:
        self._states[key] = state

    def store_agent_plan(self, key: AgentPlanKey, plan: AgentPlan) -> None:
        self._agent_plans[key] = plan

    def retrieve_state(self, key: StateKey) -> Optional[State]:
        return self._states.get(key)

    def retrieve_agent_plan(self, key: AgentPlanKey) -> Optional[AgentPlan]:
        return self._agent_plans.get(key)

    def query_states(self, **filters) -> List[State]:
        results = []
        for state in self._states.values():
            match = all(getattr(state, k, None) == v for k, v in filters.items())
            if match:
                results.append(state)
        return results

    def query_agent_plans(self, **filters) -> List[AgentPlan]:
        results = []
        for plan in self._agent_plans.values():
            match = all(getattr(plan, k, None) == v for k, v in filters.items())
            if match:
                results.append(plan)
        return results
