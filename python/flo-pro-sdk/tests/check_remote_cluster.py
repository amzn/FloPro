"""Smoke test for RayExecutionEngine against a remote cluster.

This file is intentionally not named test_*.py so pytest won't collect it
during normal test runs. To run it manually:

    hatch test tests/check_remote_cluster.py
"""

import numpy as np

from flo_pro_sdk.agent.agent_definition import AgentSpec
from flo_pro_sdk.coordinator.coordinator_definition import CoordinatorSpec
from flo_pro_sdk.core.lifecycle import ProblemRunner
from flo_pro_sdk.core.problem import Problem
from flo_pro_sdk.core.state import ConsensusState
from flo_pro_sdk.core.variables import PublicVarGroupName
from flo_pro_sdk.engine.ray import RayEngineOptions, RayExecutionEngine
from flo_pro_sdk.testing import MockAgentDefinition, MockCoordinatorDefinition

G = PublicVarGroupName("g")
ADDRESS = "ray://localhost:10001"


def test_remote_cluster():
    engine = RayExecutionEngine(
        RayEngineOptions(
            address=ADDRESS,
            runtime_env={"working_dir": "src"},
        )
    )

    initial_state = ConsensusState(
        iteration=0,
        consensus_vars=np.zeros(2),
        agent_preferred_vars={"agent1": np.zeros(2), "agent2": np.zeros(2)},
        prices={"agent1": np.zeros(2), "agent2": np.zeros(2)},
        rho={"agent1": np.ones(2), "agent2": np.ones(2)},
    )

    problem = Problem(
        agents=[
            AgentSpec(agent_class=MockAgentDefinition, agent_id="agent1"),
            AgentSpec(agent_class=MockAgentDefinition, agent_id="agent2"),
        ],
        coordinator=CoordinatorSpec(coordinator_class=MockCoordinatorDefinition),
        initial_state=initial_state,
        max_iterations=10,
    )

    runner = ProblemRunner(problem, engine)
    final_state = runner.run()
    assert final_state.iteration > 0

    engine.shutdown()
