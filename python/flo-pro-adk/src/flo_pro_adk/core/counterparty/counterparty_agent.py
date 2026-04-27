"""CounterpartyAgent ABC — abstract base for domain-specific counterparty agents.

Extends AgentDefinition with data loading via DataLoader and pluggable
SolverStrategy. Does not implement solve() — each concrete agent in a
domain pack owns its full optimization formulation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from flo_pro_sdk.agent.agent_definition import (
    AgentDefinition,
    Solution,
)
from flo_pro_sdk.core.types import JsonValue
from flo_pro_sdk.core.variables import Prices, PublicVarsMetadata, PublicVarValues, RhoValues

from flo_pro_adk.core.exceptions.agent_errors import (
    RegistrationError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from flo_pro_adk.core.data.data_loader import DataLoader
    from flo_pro_adk.core.solver.solver_strategy import (
        SolverStrategy,
    )


# Module-level registries keyed by concrete agent class.
# Set by build_problem() before Problem construction, read by create()/register(),
# reset by run_e2e_test after run().
#
# Safe with pytest-xdist (process isolation per worker). Not safe with
# threads sharing a process — use a lock or avoid concurrent mutation.
# Tests that mutate these registries must clean up in a finally block
# or fixture teardown to avoid leaking state between test cases.
_data_loader_factories: dict[type, Callable[[], DataLoader]] = {}
_var_metadata_registry: dict[type, PublicVarsMetadata] = {}


class CounterpartyAgent(AgentDefinition):
    """Abstract base for counterparty agents used in local testing.

    Provides:
    - Data loading via DataLoader (looked up from module registry or
      injected directly via __init__)
    - Pluggable SolverStrategy (concrete subclasses choose their default)
    - Variable metadata for SDK registration (looked up from module registry)

    The base class is solver-agnostic and has no dependency on any specific
    solver library. Concrete domain agents (e.g., RetailerAgent, VendorAgent)
    pick their own solver via ``_default_solver()`` — typically importing the
    solver module lazily so optional dependencies don't load unless needed.
    """

    def __init__(
        self,
        agent_params: dict[str, JsonValue],
        data_loader: DataLoader | None = None,
        solver: SolverStrategy | None = None,
    ) -> None:
        self._agent_params: dict[str, JsonValue] = agent_params
        self._data: Any = data_loader.load() if data_loader is not None else None
        self._solver: SolverStrategy = (
            solver if solver is not None else type(self)._default_solver()
        )

        metadata = _var_metadata_registry.get(type(self))
        if metadata is None:
            raise RegistrationError(
                f"No variable metadata registered for {type(self).__name__}. "
                f"Ensure _var_metadata_registry[{type(self).__name__}] is set "
                f"before constructing the agent. In E2E tests, build_problem() "
                f"handles this automatically."
            )
        self._public_vars_metadata: PublicVarsMetadata = metadata

    @classmethod
    def _default_solver(cls) -> SolverStrategy:
        """Return the default solver for this agent class.

        Concrete subclasses override this to supply their preferred solver.
        The base class raises because it has no opinion — this prevents the
        ABC from coupling to any particular solver library.
        """
        raise NotImplementedError(
            f"{cls.__name__} must override _default_solver() to provide "
            f"a default SolverStrategy, or callers must pass solver= explicitly."
        )

    @property
    def data(self) -> Any:
        return self._data

    @property
    def agent_params(self) -> dict[str, JsonValue]:
        return self._agent_params

    @property
    def solver(self) -> SolverStrategy:
        return self._solver

    @classmethod
    def create(cls, agent_params: dict[str, JsonValue]) -> CounterpartyAgent:
        """SDK-compatible factory. Looks up DataLoader from module registry."""
        data_loader: DataLoader | None = None
        factory = _data_loader_factories.get(cls)
        if factory is not None:
            data_loader = factory()
        return cls(agent_params=agent_params, data_loader=data_loader)

    def register(self) -> PublicVarsMetadata:
        """Return variable metadata. Pure getter — metadata is resolved in __init__."""
        return self._public_vars_metadata

    @property
    def public_vars_metadata(self) -> PublicVarsMetadata:
        """Variable metadata, available immediately after construction."""
        return self._public_vars_metadata

    @abstractmethod
    def solve(
        self,
        public_vars: PublicVarValues,
        prices: Prices,
        rho: RhoValues,
    ) -> Solution: ...
