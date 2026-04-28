"""Test assertion exceptions.

Extends both VADKError (for programmatic handling) and Python's built-in
AssertionError (so pytest catches them naturally).
"""

from __future__ import annotations

from flo_pro_adk.core.exceptions.vadk_error import VADKError


class VADKAssertionError(VADKError, AssertionError):
    """V-ADK test assertion failure. Caught by pytest as AssertionError."""
