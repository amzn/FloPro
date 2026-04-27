"""Handler interfaces for coordinator operations."""

from dataclasses import dataclass
from typing import Callable

from flo_pro_sdk.core.state import IterationResult, State


@dataclass
class CoordinatorHandler:
    run_iteration: Callable[[int], IterationResult]
    finalize: Callable[[State], None]
