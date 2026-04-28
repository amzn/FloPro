"""Test fixtures for CPP framework.

These classes are defined in the main package so they can be serialized
and deserialized by Ray workers.
"""

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import ndarray

from flo_pro_sdk.agent.agent_definition import AgentDefinition, Objective, Solution
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorDefinition
from flo_pro_sdk.core.state import State, ConsensusState, CoreState, AgentId
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarGroupMetadata,
    PublicVarGroupName,
    PublicVarsMetadata,
    PublicVarValues,
    RhoValues,
)

if TYPE_CHECKING:
    from flo_pro_sdk.core.state_store import ReadOnlyStore

# Standard test variable group name
TEST_VAR_GROUP = PublicVarGroupName("g")


class MockAgentDefinition(AgentDefinition):
    """Mock agent that returns fixed values for testing."""

    def solve(
        self,
        public_vars: PublicVarValues,
        prices: Prices,
        rho: RhoValues,
    ) -> Solution:
        return Solution(
            preferred_vars={TEST_VAR_GROUP: np.array([1.0, 2.0])},
            objective=Objective(utility=10.0, subsidy=0.0, proximal=0.0),
        )

    def register(self) -> PublicVarsMetadata:
        return {
            TEST_VAR_GROUP: PublicVarGroupMetadata(
                name=TEST_VAR_GROUP,
                var_metadata=pd.DataFrame({"idx": range(2)}),
            )
        }


class FailingAgentDefinition(AgentDefinition):
    """Mock agent that raises during initialization for testing error handling."""

    def __init__(self, **kwargs):
        raise RuntimeError("Simulated agent initialization failure")

    def solve(
        self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues
    ) -> Solution:
        raise NotImplementedError


class MockCoordinatorDefinition(CoordinatorDefinition):
    """Mock coordinator that increments iteration and converges after 3 iterations."""

    def __init__(self, **kwargs):
        pass

    def update_state(
        self,
        agent_results: Dict[AgentId, ndarray],
        current_state: State,
        state_store: Optional["ReadOnlyStore"] = None,
    ) -> State:
        return ConsensusState(
            iteration=current_state.iteration + 1,
            consensus_vars=current_state.consensus_vars,
            agent_preferred_vars=agent_results,
            prices={aid: current_state.get_agent_prices(aid) for aid in agent_results},
            rho={aid: current_state.get_rho(aid) for aid in agent_results},
        )

    def check_convergence(self, core_state: CoreState) -> bool:
        return core_state.iteration >= 3


class FailingCoordinatorDefinition(CoordinatorDefinition):
    """Mock coordinator that raises during initialization for testing error handling."""

    def __init__(self, **kwargs):
        raise RuntimeError("Simulated coordinator initialization failure")

    def update_state(
        self,
        agent_results: Dict[AgentId, ndarray],
        current_state: State,
        state_store: Optional["ReadOnlyStore"] = None,
    ) -> State:
        raise NotImplementedError
