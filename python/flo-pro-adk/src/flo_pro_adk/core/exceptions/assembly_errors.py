"""Problem assembly exceptions.

Non-retryable configuration errors caught before the coordination loop starts.
"""

from __future__ import annotations

from flo_pro_adk.core.exceptions.vadk_error import VADKError


class AssemblyError(VADKError):
    """Base for all problem assembly errors."""


class ScenarioNotFoundError(AssemblyError):
    """Requested scenario name is not registered in the ScenarioRegistry."""

    def __init__(self, scenario_name: str) -> None:
        super().__init__(f"Scenario '{scenario_name}' not found in registry")
        self.scenario_name = scenario_name


class InvalidAssemblyError(AssemblyError):
    """Incompatible components passed to build_problem."""


class DuplicateScenarioError(AssemblyError):
    """Scenario with this name is already registered."""

    def __init__(self, scenario_name: str) -> None:
        super().__init__(
            f"Scenario '{scenario_name}' is already registered. "
            "Use a unique name or unregister the existing scenario first."
        )
        self.scenario_name = scenario_name
