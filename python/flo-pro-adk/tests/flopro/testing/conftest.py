# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for flopro testing."""

import pytest

from flo_pro_adk.core.counterparty.counterparty_agent import (
    _var_metadata_registry,
)


def _xpress_available() -> bool:
    try:
        import xpress  # noqa: F401
        return True
    except ImportError:
        return False


requires_xpress = pytest.mark.skipif(
    not _xpress_available(),
    reason="xpress not installed",
)


@pytest.fixture(autouse=True)
def _clean_var_metadata_registry():
    """Restore _var_metadata_registry to its pre-test state after every test."""
    snapshot = dict(_var_metadata_registry)
    yield
    _var_metadata_registry.clear()
    _var_metadata_registry.update(snapshot)
