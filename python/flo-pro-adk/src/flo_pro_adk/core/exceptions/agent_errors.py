"""Agent lifecycle errors — registration and configuration failures."""

from __future__ import annotations

from flo_pro_adk.core.exceptions.vadk_error import (
    VADKError,
)


class AgentError(VADKError):
    """Base for agent lifecycle errors."""


class RegistrationError(AgentError):
    """Raised when agent registration fails due to missing metadata.

    Typically means the variable metadata registry was not populated
    before calling register(). In E2E tests this is handled by
    build_problem(); in unit tests you must set
    ``_var_metadata_registry[AgentClass]`` before calling register().
    """
