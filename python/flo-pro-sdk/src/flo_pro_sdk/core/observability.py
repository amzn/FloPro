import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LoggerInterface(ABC):
    @abstractmethod
    def log(self, level: LogLevel, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        self.log(LogLevel.DEBUG, message)

    def info(self, message: str) -> None:
        self.log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        self.log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        self.log(LogLevel.ERROR, message)


class MetricInterface(ABC):
    @abstractmethod
    def record_metric(self, name: str, value: Any) -> None:
        pass


class Logger(LoggerInterface):
    """Simple logger using Python's logging module.

    Messages are prefixed with component_id for traceability across components.
    """

    def __init__(self, component_id: str):
        self.component_id = component_id
        self._logger = logging.getLogger(f"cpp.{component_id}")

    def log(self, level: LogLevel, message: str) -> None:
        log_level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }
        python_level = log_level_map.get(level, logging.INFO)
        self._logger.log(python_level, f"[{self.component_id}] {message}")


class InMemoryMetrics(MetricInterface):
    """Simple in-memory metrics store."""

    def __init__(self, component_id: str):
        self.component_id = component_id
        self._metrics: Dict[str, List[Any]] = {}

    def record_metric(self, name: str, value: Any) -> None:
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)

    def get_metrics(self, name: str) -> List[Any]:
        """Retrieve recorded metrics by name."""
        return self._metrics.get(name, [])

    def get_all_metrics(self) -> Dict[str, List[Any]]:
        """Retrieve all recorded metrics."""
        return dict(self._metrics)
