"""Problem assembly — constructs CPP SDK Problems for E2E testing.

Wires module-level registries (DataLoader, var metadata) and builds a
Problem with sensible defaults. Works symmetrically for both vendor and
Amazon agent teams:

- Vendor: MyVendorAgent (agent) + RetailerAgent (counterparty)
- Amazon: MyAmazonAgent (agent) + VendorAgent (counterparty)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flo_pro_sdk.agent.agent_definition import AgentDefinition, AgentSpec
from flo_pro_sdk.coordinator.admm_coordinator import ADMMCoordinator
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.problem import Problem
from flo_pro_sdk.core.types import JsonValue

from flo_pro_adk.core.counterparty.counterparty_agent import (
    CounterpartyAgent,
    _data_loader_factories,
    _var_metadata_registry,
)

if TYPE_CHECKING:
    from flo_pro_adk.core.testing.simulation_data_generator import (
        SimulationDataGenerator,
    )


def _wire_registries(
    agent_class: type[AgentDefinition],
    counterparty_class: type[CounterpartyAgent],
    generator: SimulationDataGenerator,
) -> None:
    """Wire DataLoader factories and variable metadata into module registries."""
    var_metadata = generator.generate_variable_group_metadata()

    for cls in (agent_class, counterparty_class):
        if not (isinstance(cls, type) and issubclass(cls, CounterpartyAgent)):
            continue
        loader = generator.create_data_loader_for(cls)
        _data_loader_factories[cls] = lambda _l=loader: _l  # type: ignore[misc]
        _var_metadata_registry[cls] = var_metadata


def build_problem(
    agent_class: type[AgentDefinition],
    counterparty_class: type[CounterpartyAgent],
    data_generator: SimulationDataGenerator,
    *,
    max_iterations: int = 1000,
    agent_params: dict[str, JsonValue] | None = None,
) -> Problem:
    """Build a CPP SDK Problem for E2E testing.

    Wires module registries, generates initial state, and returns a
    ready-to-run Problem. Uses ADMMCoordinator with maximization format.

    Args:
        agent_class: The user's AgentDefinition subclass.
        counterparty_class: The counterparty (RetailerAgent or VendorAgent).
        data_generator: Domain-specific SimulationDataGenerator instance.
        max_iterations: Maximum ADMM iterations before stopping.
        agent_params: Optional JSON-serializable config for the agent.
    """
    _wire_registries(agent_class, counterparty_class, data_generator)

    initial_state = data_generator.generate_initial_state(
        agent_ids=("agent", "counterparty"),
    )

    return Problem(
        agents=[
            AgentSpec(
                agent_class=agent_class,
                agent_id="agent",
                agent_params=agent_params,
            ),
            AgentSpec(
                agent_class=counterparty_class,
                agent_id="counterparty",
            ),
        ],
        coordinator=CoordinatorSpec(
            coordinator_class=ADMMCoordinator,
            coordinator_params={"problem_format": "maximization"},
        ),
        initial_state=initial_state,
        max_iterations=max_iterations,
    )
