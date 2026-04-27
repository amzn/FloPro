"""State store for shared algorithm state access.

The StateStore is the primary interface for reading and writing iteration
state and agent plans during the coordination loop. It acts as a shared
store — participants post state and plans, and anyone can read them.

Implementations handle the specifics of local vs distributed access:
- InMemoryStateStore: single-process, dict-based cache
- RayStateStore: distributed, wraps a Ray actor (see engine/ray/state_store.py)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from flo_pro_sdk.core.compute import ComputeSpec
    from flo_pro_sdk.core.state import AgentPlan, State
    from flo_pro_sdk.core.persistence_backend import PersistenceBackend


@dataclass
class StoreConfig:
    """Configuration for state store passed to execution engines.

    Attributes:
        persistence_backend: Optional persistence backend. When None, only
            in-memory caching is used. When provided, state is persisted
            for observability.
        cache_size: Number of recent iterations to retain in the store.
            Older iterations are evicted from the store but remain in the
            persistent backend (if configured).
        state_store_compute: Optional compute resources for the StateStore
            actor's placement group bundle. Used by the Ray engine when
            co-locating the StateStore and Coordinator actors on the same
            node. If None, defaults to 1 CPU with no memory constraint.
    """

    persistence_backend: Optional["PersistenceBackend"] = None
    cache_size: int = 5
    state_store_compute: Optional["ComputeSpec"] = None

    def __post_init__(self) -> None:
        if self.cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {self.cache_size}")


@runtime_checkable
class ReadOnlyStore(Protocol):
    """Read-only view of the state store.

    This protocol exposes only the read methods of StateStore. It is passed
    to CoordinatorDefinition.update_state() so that coordinators can access
    historical state without being able to mutate the store.
    """

    def get_state(self, iteration: int) -> Optional["State"]: ...

    def get_agent_plan(
        self, iteration: int, agent_id: str
    ) -> Optional["AgentPlan"]: ...

    def get_recent_states(self, count: int) -> List["State"]: ...


T = TypeVar("T")


class Ref(ABC, Generic[T]):
    """A resolvable reference to a value.

    Abstracts over how values are stored and retrieved. Local engines
    use DirectRef (holds the value in-process). Distributed engines
    use engine-specific subclasses (e.g. RayRef wrapping an ObjectRef).
    """

    @abstractmethod
    def resolve(self) -> T:
        pass


class DirectRef(Ref[T]):
    """A reference that holds the value directly (no indirection)."""

    def __init__(self, value: T) -> None:
        self._value = value

    def resolve(self) -> T:
        return self._value


class StateStore(ABC):
    """Shared store for algorithm state and agent plans.

    Provides bounded, iteration-indexed storage for recent algorithm state.
    Implementations handle the specifics of local vs distributed access.

    The store supports two types of data:
    - State: the full algorithm state after each iteration (consensus vars,
      prices, rho, iteration number)
    - AgentPlan: individual agent solutions for each iteration
    """

    @abstractmethod
    def store_state(
        self,
        iteration: int,
        state: "State",
        timestamp: float | None = None,
        *,
        blocking: bool = False,
    ) -> None:
        """Store state for an iteration.

        Args:
            iteration: The iteration number.
            state: The algorithm state.
            timestamp: Optional UTC epoch timestamp of when the state was produced.
            blocking: If True, block until the write is confirmed.
                Useful for distributed stores where writes are
                fire-and-forget by default.
        """
        pass

    @abstractmethod
    def store_agent_plan(
        self, iteration: int, agent_id: str, plan: "AgentPlan"
    ) -> None:
        """Store an agent's plan for an iteration."""
        pass

    @abstractmethod
    def store_agent_plans(self, iteration: int, plans: Dict[str, "AgentPlan"]) -> None:
        """Store multiple agent plans for an iteration in a single call.

        Args:
            iteration: The iteration number.
            plans: Mapping of agent_id to AgentPlan.
        """
        pass

    @abstractmethod
    def get_state(self, iteration: int) -> Optional["State"]:
        """Get state for an iteration, or None if not cached."""
        pass

    @abstractmethod
    def get_agent_plan(self, iteration: int, agent_id: str) -> Optional["AgentPlan"]:
        """Get an agent's plan, or None if not cached."""
        pass

    @abstractmethod
    def get_recent_states(self, count: int) -> List["State"]:
        """Get the N most recent states in descending iteration order."""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending writes."""
        pass
