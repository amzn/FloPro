"""Execution engine implementations."""

from flo_pro_sdk.engine.local import LocalExecutionEngine

__all__ = ["LocalExecutionEngine"]


def __getattr__(name: str):
    if name == "RayExecutionEngine":
        from flo_pro_sdk.engine.ray import RayExecutionEngine

        return RayExecutionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
