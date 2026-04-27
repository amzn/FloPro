"""Local agent handle types."""

from dataclasses import dataclass
from typing import Callable

from flo_pro_sdk.agent.agent_definition import Solution
from flo_pro_sdk.core.state import State
from flo_pro_sdk.core.variables import (
    PublicVarValues,
    Prices,
    RhoValues,
    PublicVarsMetadata,
)


@dataclass
class LocalAgentHandles:
    """Container for local agent method references."""

    query: Callable[[PublicVarValues, Prices, RhoValues], Solution]
    register: Callable[[], PublicVarsMetadata]
    finalize: Callable[[State], None]
