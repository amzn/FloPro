"""Tests for ScenarioParams."""

from __future__ import annotations

import pytest

from flo_pro_adk.core.types.scenario_params import ScenarioParams


def _params(**overrides) -> ScenarioParams:
    defaults = dict(
        name="t", seed=1, n_variables=10, n_groups=1,
        price_distribution="uniform", price_range=(0.0, 10.0), rho=1.0, domain_params={},
    )
    return ScenarioParams(**{**defaults, **overrides})


def test_valid_params():
    assert _params(name="test").name == "test"


@pytest.mark.parametrize("overrides,match", [
    ({"name": ""}, "name must not be empty"),
    ({"n_variables": 0}, "n_variables must be positive"),
    ({"price_range": (10.0, 0.0)}, "price_range"),
    ({"rho": 0.0}, "rho must be positive"),
])
def test_validation_errors(overrides, match):
    with pytest.raises(ValueError, match=match):
        _params(**overrides)


def test_frozen():
    with pytest.raises(AttributeError):
        _params().name = "changed"  # type: ignore[misc]
