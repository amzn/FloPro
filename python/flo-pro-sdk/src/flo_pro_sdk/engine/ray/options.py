"""Configuration options for the Ray execution engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RayStateStoreType(Enum):
    """State store implementation to use with the Ray engine."""

    DIRECT = "direct"
    """Serialize full state objects through the actor. Simple, suitable for small states."""
    OBJECT_STORE = "object_store"
    """Store state in Ray's object store, pass only refs through the actor.
    Recommended for large states (100+ agents, 100k+ variables)."""


@dataclass
class RayEngineOptions:
    """Options for configuring the Ray execution engine.

    Attributes:
        address: The address of the Ray cluster to connect to (e.g.
            ``"ray://my-cluster:10001"``). When ``None`` (default), Ray
            starts a local instance.
        runtime_env: Runtime environment specification forwarded to ``ray.init()``.
        extra_kwargs: Additional keyword arguments forwarded directly to
            ``ray.init()``, for options not yet promoted to named fields.
        state_store_type: Which state store implementation to use.
    """

    address: Optional[str] = None
    runtime_env: Optional[Dict[str, Any]] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    state_store_type: RayStateStoreType = RayStateStoreType.DIRECT

    def ray_init_kwargs(self) -> Dict[str, Any]:
        """Return kwargs to pass to ``ray.init()``, omitting unset fields."""
        kwargs: Dict[str, Any] = dict(self.extra_kwargs)
        if self.address is not None:
            kwargs["address"] = self.address
        if self.runtime_env is not None:
            kwargs["runtime_env"] = self.runtime_env
        return kwargs
