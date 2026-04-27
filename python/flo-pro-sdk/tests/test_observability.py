"""Tests for observability implementations."""

from flo_pro_sdk.core.observability import Logger, InMemoryMetrics


class TestObservability:
    """Tests for observability implementations."""

    def test_logger(self) -> None:
        logger = Logger("test-component")
        # Should not raise
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

    def test_in_memory_metrics(self) -> None:
        metrics = InMemoryMetrics("test-component")
        metrics.record_metric("solve_time", 0.5)
        metrics.record_metric("solve_time", 0.7)
        metrics.record_metric("objective", 100.0)

        assert metrics.get_metrics("solve_time") == [0.5, 0.7]
        assert metrics.get_metrics("objective") == [100.0]
        assert metrics.get_metrics("nonexistent") == []

        all_metrics = metrics.get_all_metrics()
        assert "solve_time" in all_metrics
        assert "objective" in all_metrics
