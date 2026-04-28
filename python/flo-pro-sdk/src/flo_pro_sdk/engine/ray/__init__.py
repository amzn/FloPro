"""Ray execution engine for distributed CPP optimization."""

try:
    import ray  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Ray is required for the Ray execution engine. "
        "Install it with: pip install flo-pro-sdk[ray]"
    ) from e

from flo_pro_sdk.engine.ray.engine import RayExecutionEngine
from flo_pro_sdk.engine.ray.options import RayEngineOptions, RayStateStoreType
from flo_pro_sdk.engine.ray.actors import RayAgentActor, RayCoordinatorActor
from flo_pro_sdk.engine.ray.state_store import RayRef, RayStateStore, RayRefStateStore
from flo_pro_sdk.engine.ray.executors import (
    RayQueryExecutor,
    RayRegistrationExecutor,
    RayFinalizationExecutor,
)

__all__ = [
    "RayExecutionEngine",
    "RayEngineOptions",
    "RayStateStoreType",
    "RayAgentActor",
    "RayCoordinatorActor",
    "RayRef",
    "RayStateStore",
    "RayRefStateStore",
    "RayQueryExecutor",
    "RayRegistrationExecutor",
    "RayFinalizationExecutor",
]
