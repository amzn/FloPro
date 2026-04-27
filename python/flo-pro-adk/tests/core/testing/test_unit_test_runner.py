"""Tests for core unit test runner (domain-agnostic)."""

from __future__ import annotations

import numpy as np
from flo_pro_sdk.core.variables import PublicVarGroupName

from flo_pro_adk.core.testing.unit_test_runner import (
    run_unit_test_with_inputs,
)
from tests.conftest import StubAgent

GROUP = PublicVarGroupName("asin_vendor_inbound_periods")


def test_run_unit_test_with_explicit_inputs():
    public_vars = {GROUP: np.array([1.0, 2.0, 3.0])}
    prices = {GROUP: np.zeros(3)}
    rho = {GROUP: np.ones(3)}

    result = run_unit_test_with_inputs(StubAgent, public_vars, prices, rho)

    np.testing.assert_array_equal(result.preferred_vars[GROUP], public_vars[GROUP])
