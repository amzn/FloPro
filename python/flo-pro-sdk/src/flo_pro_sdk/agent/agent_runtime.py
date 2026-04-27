from typing import Any, Optional

from flo_pro_sdk.agent.agent_definition import AgentDefinition, Solution, AgentSpec
from flo_pro_sdk.core.variables import (
    Prices,
    PublicVarsMetadata,
    PublicVarValues,
    RhoValues,
)
from flo_pro_sdk.core.observability import LoggerInterface, MetricInterface


class AgentRuntime:
    def __init__(
        self,
        agent_definition: AgentDefinition,
        agent_spec: AgentSpec,
        logger: Optional[LoggerInterface] = None,
        metrics: Optional[MetricInterface] = None,
    ):
        self.agent_definition = agent_definition
        self.agent_spec = agent_spec
        self._logger = logger
        self._metrics = metrics

    def query(
        self, public_vars: PublicVarValues, prices: Prices, rho: RhoValues
    ) -> Solution:
        return self.agent_definition.solve(public_vars, prices, rho)

    def register(self) -> PublicVarsMetadata:
        return self.agent_definition.register()

    def finalize(self, final_state: Any) -> None:
        self.agent_definition.finalize(final_state)
