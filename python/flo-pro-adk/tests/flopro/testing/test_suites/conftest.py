# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Override agent_class fixture for internal test suite validation.

The pre-built suites expect an agent_class fixture (provided by
FloProSimulationSuite when vendors run tests). For V-ADK internal
testing, we substitute StubAgent.
"""

from __future__ import annotations

import pytest

from tests.conftest import StubAgent


@pytest.fixture(scope="session")
def agent_class():
    """Return StubAgent for internal testing."""
    return StubAgent
