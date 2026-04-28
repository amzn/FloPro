"""FloProSimulationSuite — vendor-facing entry point for local testing.

Runs the pre-built unit and E2E test suites with the vendor's agent class.
Delegates to the existing test modules via pytest.

Usage::

    from flo_pro_adk.flopro.testing.simulation_suite import (
        FloProSimulationSuite,
    )
    from my_agent import MyVendorAgent

    suite = FloProSimulationSuite(MyVendorAgent)
    suite.run_all()       # all tests (unit + E2E)

    # -- or run selectively --
    # suite.run_unit()  # unit tests only
    # suite.run_e2e()   # E2E tests only
"""

from __future__ import annotations

import pytest

from flo_pro_sdk.agent.agent_definition import AgentDefinition

_UNIT_MODULE = "flo_pro_adk.flopro.testing.test_suites.test_flopro_unit"
_E2E_MODULE = "flo_pro_adk.flopro.testing.test_suites.test_flopro_e2e"


class _AgentClassPlugin:
    """Pytest plugin that overrides the agent_class fixture."""

    def __init__(
        self,
        agent_cls: type[AgentDefinition],
    ) -> None:
        self._agent_cls = agent_cls

    @pytest.fixture(scope="session")
    def agent_class(self):
        return self._agent_cls


class FloProSimulationSuite:
    """Vendor-facing test suite for Flo Pro agents.

    Args:
        agent_class: The vendor's AgentDefinition subclass.
    """

    def __init__(
        self,
        agent_class: type[AgentDefinition],
    ) -> None:
        self._agent_class = agent_class

    def _run_modules(self, *modules: str) -> int:
        """Run pytest against the given test modules. Returns exit code."""
        return pytest.main(
            ["--pyargs", *modules, "-v", "--tb=short", "--no-header",
             "--import-mode=importlib"],
            plugins=[_AgentClassPlugin(self._agent_class)],
        )

    def run_all(self) -> int:
        """Run all unit and E2E tests. Returns pytest exit code (0 = all passed)."""
        return self._run_modules(_UNIT_MODULE, _E2E_MODULE)

    def run_unit(self) -> int:
        """Run unit tests only. Returns pytest exit code."""
        return self._run_modules(_UNIT_MODULE)

    def run_e2e(self) -> int:
        """Run E2E tests only. Returns pytest exit code."""
        return self._run_modules(_E2E_MODULE)
