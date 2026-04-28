import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any, TYPE_CHECKING

from numpy import ndarray

from flo_pro_sdk.core.compute import ComputeSpec
from flo_pro_sdk.core.types import JsonValue
from flo_pro_sdk.core.state import AgentId, State, CoreState
from flo_pro_sdk.core.var_layout import VarLayout

if TYPE_CHECKING:
    from flo_pro_sdk.core.query import QueryStrategy
    from flo_pro_sdk.core.state_store import ReadOnlyStore
    from flo_pro_sdk.core.structure_function import (
        StructureFunctionSpec,
        StructureFunction,
    )
    from flo_pro_sdk.core.registry import AgentRegistry


@dataclass
class CoordinatorSpec:
    coordinator_class: type["CoordinatorDefinition"]
    query_strategy: Optional["QueryStrategy"] = None
    coordinator_params: dict[str, JsonValue] | None = None
    structure_function_spec: Optional["StructureFunctionSpec"] = None
    compute: Optional[ComputeSpec] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        try:
            self.coordinator_params is None or json.dumps(self.coordinator_params)
        except TypeError as e:
            raise TypeError(f"coordinator_params must be JSON-serializable: {e}")

    def instantiate(self, registry: "AgentRegistry") -> "CoordinatorDefinition":
        """Instantiate coordinator, wiring layout and structure function."""
        layout = registry.get_layout()
        structure_function = None
        if self.structure_function_spec:
            pub_var_metadata = registry.get_all_subscribed_vars()
            structure_function = self.structure_function_spec.instantiate(
                layout, pub_var_metadata
            )
        return self.coordinator_class.create(
            coordinator_params=self.coordinator_params or {},
            layout=layout,
            structure_function=structure_function,
        )


class CoordinatorDefinition(ABC):
    """Base coordinator interface. Operates on flat arrays."""

    @classmethod
    def create(
        cls,
        coordinator_params: dict[str, JsonValue],
        layout: VarLayout,
        structure_function: Optional["StructureFunction"] = None,
    ) -> "CoordinatorDefinition":
        """Factory method for instantiation from JSON-serializable params.

        Subclasses that need layout/structure_function should accept them
        in __init__ and override create() if custom construction is needed.
        """
        # mypy can't verify that subclass __init__ accepts these kwargs
        return cls(
            layout=layout, structure_function=structure_function, **coordinator_params
        )  # type: ignore[call-arg]

    @abstractmethod
    def update_state(
        self,
        agent_results: Dict[AgentId, ndarray],
        current_state: State,
        state_store: Optional["ReadOnlyStore"] = None,
    ) -> State:
        """Update state with flat agent results. Returns new state.

        Use self.layout.get_global_indices(agent_id) for subscription info.
        """
        pass

    def check_convergence(self, core_state: CoreState) -> bool:
        """Check convergence using only lightweight CoreState."""
        return False

    def finalize(self, final_state: State) -> None:
        pass
