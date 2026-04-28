from dataclasses import dataclass
from enum import Enum
import time
from typing import Optional, TYPE_CHECKING
import logging

from flo_pro_sdk.core.problem import Problem
from flo_pro_sdk.core.state import State, StateLoader
from flo_pro_sdk.core.registry import AgentRegistry
from flo_pro_sdk.core.engine import ExecutionEngine
from flo_pro_sdk.core.handlers import CoordinatorHandler
from flo_pro_sdk.core.query import DefaultQueryStrategy

if TYPE_CHECKING:
    from flo_pro_sdk.dashboard.manager import DashboardConfig, DashboardManager

logger = logging.getLogger(__name__)


class PhaseName(Enum):
    AGENT_REGISTRATION = "agent_registration"
    ALGORITHM_STARTUP = "algorithm_startup"
    COORDINATION_LOOP = "coordination_loop"
    FINALIZATION = "finalization"


@dataclass
class RegistrationResult:
    registry: AgentRegistry


@dataclass
class StartupResult:
    state: State
    coordinator_handler: CoordinatorHandler


@dataclass
class CoordinationResult:
    final_state: State


class ProblemRunner:
    def __init__(
        self,
        problem: Problem,
        engine: ExecutionEngine,
        dashboard_config: Optional["DashboardConfig"] = None,
    ):
        self.problem = problem
        self.engine = engine
        self._dashboard_config = dashboard_config
        self._dashboard_mgr: Optional["DashboardManager"] = None

    def run(self) -> State:
        registration_result = self._agent_registration_phase()
        self._start_dashboard(registration_result)
        startup_result = self._algorithm_startup_phase(registration_result)
        coordination_result = self._coordination_loop_phase(startup_result)
        self._finalization_phase(registration_result, coordination_result)
        self._stop_dashboard()

        return coordination_result.final_state

    def _start_dashboard(self, registration_result: RegistrationResult) -> None:
        """Start the live dashboard if configured and persistence is available."""
        if self._dashboard_config is None:
            return
        run_dir = self.engine.get_run_dir()
        if run_dir is None:
            logger.warning(
                "Dashboard requested but no persistence backend configured; "
                "skipping dashboard"
            )
            return
        try:
            from flo_pro_sdk.dashboard.manager import DashboardManager

            self._dashboard_mgr = DashboardManager(run_dir, self._dashboard_config)
            self._dashboard_mgr.start()
        except Exception:
            logger.exception("Failed to start dashboard; continuing without it")
            self._dashboard_mgr = None

    def _stop_dashboard(self) -> None:
        """Linger and then shut down the dashboard."""
        if self._dashboard_mgr is None:
            return
        try:
            self._dashboard_mgr.linger()
            self._dashboard_mgr.shutdown()
        except Exception:
            logger.exception("Error during dashboard shutdown")

    def _agent_registration_phase(self) -> RegistrationResult:
        registry = AgentRegistry()

        self.engine.allocate_agents(self.problem.agents)

        agent_ids = [agent_spec.agent_id for agent_spec in self.problem.agents]
        registration_executor = self.engine.get_registration_executor()
        registration_results = registration_executor.execute(agent_ids)

        for agent_spec in self.problem.agents:
            subscribed_vars = registration_results[agent_spec.agent_id]
            registry.register_agent(
                agent_id=agent_spec.agent_id,
                subscribed_vars=subscribed_vars,
                metadata=agent_spec.metadata,
            )

        registry.finalize_registration()
        return RegistrationResult(registry=registry)

    def _algorithm_startup_phase(
        self, registration_result: RegistrationResult
    ) -> StartupResult:
        if isinstance(self.problem.initial_state, State):
            state = self.problem.initial_state
        elif isinstance(self.problem.initial_state, StateLoader):
            state = self.problem.initial_state.load(
                registry=registration_result.registry
            )
        else:
            raise ValueError("initial_state must be State or StateLoader")

        query_strategy = (
            self.problem.coordinator.query_strategy or DefaultQueryStrategy()
        )

        coordinator_handler = self.engine.allocate_coordinator(
            coordinator_spec=self.problem.coordinator,
            query_strategy=query_strategy,
            query_executor=self.engine.get_query_executor(),
            registry=registration_result.registry,
        )

        return StartupResult(state=state, coordinator_handler=coordinator_handler)

    def _coordination_loop_phase(
        self, startup_result: StartupResult
    ) -> CoordinationResult:
        coordinator_handler = startup_result.coordinator_handler
        store = self.engine.get_state_store()

        # Store initial state so coordinator can read it.
        # blocking=True prevents a race where the coordinator reads
        # before the actor has processed the write.
        store.store_state(
            startup_result.state.iteration,
            startup_result.state,
            timestamp=time.time(),
            blocking=True,
        )

        iteration = startup_result.state.iteration
        while iteration < self.problem.max_iterations:
            result = coordinator_handler.run_iteration(iteration)
            if result.converged:
                break
            iteration = result.iteration

        # Ensure all fire-and-forget writes have landed before reading.
        store.flush()

        final_state = store.get_state(result.iteration)
        if final_state is None:
            raise RuntimeError(
                f"Final state not found for iteration {result.iteration}"
            )
        return CoordinationResult(final_state=final_state)

    def _finalization_phase(
        self,
        registration_result: RegistrationResult,
        coordination_result: CoordinationResult,
    ) -> None:
        final_state = coordination_result.final_state
        registry = registration_result.registry

        agent_ids = registry.list_agents()
        finalization_executor = self.engine.get_finalization_executor()
        finalization_executor.execute(agent_ids, final_state)

        self.engine.shutdown()
