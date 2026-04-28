"""SimulationDataGenerator — produces valid CPP SDK types for all test scenarios.

Uses the Template Method pattern: concrete core generates domain-agnostic
data (consensus vars, prices, rho, initial State), while the abstract hook
generate_counterparty_input_data() is implemented by domain packs.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import ndarray

from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarGroupMetadata,
    PublicVarGroupName,
    PublicVarsMetadata,
    PublicVarValues,
    RhoValues,
)

from flo_pro_adk.core.data.in_memory_data_loader import (
    InMemoryDataLoader,
)
from flo_pro_adk.core.types.scenario_params import (
    ScenarioParams,
)

if TYPE_CHECKING:
    from flo_pro_adk.core.types.counterparty_input_data import (
        CounterpartyInputData,
    )


class SimulationDataGenerator(ABC):
    """Generates valid CPP SDK types for test scenarios.

    The core generation is domain-agnostic, parameterized by ScenarioParams.
    Domain packs subclass and implement:
    - ``generate_counterparty_input_data()`` for mock counterparty data
    - ``get_group_names()`` for domain-specific variable group names

    All randomness is seed-controlled for reproducibility.
    """

    def __init__(self, params: ScenarioParams) -> None:
        self._params = params
        self._rng = np.random.default_rng(params.seed)

    @property
    def params(self) -> ScenarioParams:
        return self._params

    @abstractmethod
    def get_group_names(self) -> list[PublicVarGroupName]:
        """Return the variable group names for this domain.

        Domain packs define what groups exist (e.g., ["asin_vendor_inbound_periods"] for Flo Pro).
        """
        ...

    def _get_variables_per_group(self) -> dict[PublicVarGroupName, int]:
        """Distribute n_variables across groups."""
        groups = self.get_group_names()
        n = self._params.n_variables
        base = n // len(groups)
        remainder = n % len(groups)
        result: dict[PublicVarGroupName, int] = {}
        for i, name in enumerate(groups):
            result[name] = base + (1 if i < remainder else 0)
        return result

    def _generate_array(self, size: int, low: float, high: float) -> ndarray:
        """Generate array using the configured price distribution."""
        dist = self._params.price_distribution
        if dist == "uniform":
            return self._rng.uniform(low, high, size=size)
        elif dist == "normal":
            mean = (low + high) / 2
            std = (high - low) / 4
            return np.clip(self._rng.normal(mean, std, size=size), low, high)
        elif dist == "lognormal":
            mean = np.log((low + high) / 2)
            sigma = 0.5
            return np.clip(self._rng.lognormal(mean, sigma, size=size), low, high)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

    def generate_consensus_vars(self) -> PublicVarValues:
        """Generate initial consensus variable values."""
        low, high = self._params.price_range
        return {
            name: self._generate_array(size, low, high)
            for name, size in self._get_variables_per_group().items()
        }

    def generate_prices(self) -> Prices:
        """Generate initial dual prices (zeros)."""
        return {
            name: np.zeros(size)
            for name, size in self._get_variables_per_group().items()
        }

    def generate_rho(self) -> RhoValues:
        """Generate initial penalty weights (uniform rho)."""
        return {
            name: np.full(size, self._params.rho)
            for name, size in self._get_variables_per_group().items()
        }

    def generate_variable_group_metadata(self) -> PublicVarsMetadata:
        """Generate variable group metadata for diagnostics."""
        return {
            name: PublicVarGroupMetadata(
                name=name,
                var_metadata=pd.DataFrame({"idx": range(size)}),
            )
            for name, size in self._get_variables_per_group().items()
        }

    def generate_initial_state(
        self,
        agent_ids: tuple[str, str] = ("agent", "counterparty"),
    ) -> ConsensusState:
        """Generate a complete initial ConsensusState for E2E testing."""
        sizes = self._get_variables_per_group()
        total = sum(sizes.values())
        consensus = np.zeros(total)
        prices = np.zeros(total)
        rho = np.full(total, self._params.rho)
        preferred = np.zeros(total)

        return ConsensusState(
            iteration=0,
            consensus_vars=consensus,
            agent_preferred_vars={aid: preferred.copy() for aid in agent_ids},
            prices={aid: prices.copy() for aid in agent_ids},
            rho={aid: rho.copy() for aid in agent_ids},
        )

    def generate_rho_series(self, n_points: int = 10) -> list[RhoValues]:
        """Generate a series of rho values for sensitivity testing."""
        base = self._params.rho
        factors = np.logspace(-1, 2, n_points)
        sizes = self._get_variables_per_group()
        return [
            {name: np.full(size, base * f) for name, size in sizes.items()}
            for f in factors
        ]

    def generate_price_variants(self, n_variants: int = 5) -> list[Prices]:
        """Generate price variants for directional testing."""
        low, high = self._params.price_range
        sizes = self._get_variables_per_group()
        return [
            {name: self._generate_array(size, low, high) for name, size in sizes.items()}
            for _ in range(n_variants)
        ]

    @abstractmethod
    def generate_counterparty_input_data(self) -> CounterpartyInputData:
        """Domain hook — produce mock counterparty input dataset.

        TODO: Implement in domain packs when mock solve() is ready.
        """
        ...

    def create_data_loader_for(self, agent_cls: type) -> InMemoryDataLoader:
        """Return a DataLoader for the given agent class.

        Default: wraps generate_counterparty_input_data() in a InMemoryDataLoader.
        Domain packs override to return agent-specific data.
        """
        from flo_pro_adk.core.data.in_memory_data_loader import (
            InMemoryDataLoader,
        )
        return InMemoryDataLoader(self.generate_counterparty_input_data())
