"""Dashboard lifecycle manager.

Starts and stops the Dash dashboard in a background daemon thread,
isolated from the main coordination loop so dashboard failures never
propagate to the engine.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the live dashboard.

    Attributes:
        host: Bind address for the dashboard server.
        port: Preferred port. If unavailable the manager tries successive
            ports up to ``port + max_port_retries``.
        refresh_interval_ms: Auto-refresh interval in milliseconds passed
            to the Dash app. 0 disables auto-refresh.
        post_run_linger_seconds: How long to keep the dashboard alive
            after the coordination run finishes so the user can inspect
            final results. 0 means shut down immediately.
        max_port_retries: Number of successive ports to try when the
            preferred port is in use.
    """

    host: str = "127.0.0.1"
    port: int = 8050
    refresh_interval_ms: int = 5000
    post_run_linger_seconds: float = 0
    max_port_retries: int = 10


def _find_available_port(host: str, port: int, max_retries: int) -> int:
    """Return *port* if available, otherwise try successive ports."""
    for offset in range(max_retries + 1):
        candidate = port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, candidate))
                if offset > 0:
                    logger.info(
                        "Port %d in use, using %d instead",
                        port,
                        candidate,
                    )
                return candidate
            except OSError:
                continue
    raise OSError(f"No available port in range {port}–{port + max_retries}")


class DashboardManager:
    """Manages the dashboard server lifecycle in a daemon thread.

    Usage::

        mgr = DashboardManager(run_dir, config)
        mgr.start()          # non-blocking
        # ... coordination loop runs ...
        mgr.linger()          # blocks for post_run_linger_seconds
        mgr.shutdown()        # stops the server thread

    All public methods are safe to call even if the dashboard failed to
    start — errors are logged but never raised.
    """

    def __init__(
        self,
        run_dir: Path | str,
        config: DashboardConfig | None = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._config = config or DashboardConfig()
        self._thread: Optional[threading.Thread] = None
        self._actual_port: Optional[int] = None
        self._started = threading.Event()
        self._failed = False

    @property
    def actual_port(self) -> Optional[int]:
        """The port the server is actually listening on, or None."""
        return self._actual_port

    def start(self) -> None:
        """Start the dashboard server in a daemon thread."""
        try:
            self._actual_port = _find_available_port(
                self._config.host,
                self._config.port,
                self._config.max_port_retries,
            )
        except OSError:
            logger.warning(
                "Dashboard: no available port near %d, skipping",
                self._config.port,
            )
            self._failed = True
            return

        self._thread = threading.Thread(
            target=self._run_server,
            name="dashboard-server",
            daemon=True,
        )
        self._thread.start()
        # Wait briefly for the server to bind
        self._started.wait(timeout=5.0)
        if not self._failed:
            logger.info(
                "Dashboard available at http://%s:%d",
                self._config.host,
                self._actual_port,
            )

    def linger(self) -> None:
        """Block for the configured linger period after the run finishes."""
        if self._failed or self._thread is None:
            return
        seconds = self._config.post_run_linger_seconds
        if seconds <= 0:
            return
        logger.info(
            "Dashboard lingering for %.0fs at http://%s:%d",
            seconds,
            self._config.host,
            self._actual_port,
        )
        time.sleep(seconds)

    def shutdown(self) -> None:
        """Stop the dashboard server.

        Since the thread is a daemon it will die when the process exits,
        but we also attempt a clean shutdown of the werkzeug server.
        """
        if self._thread is None:
            return
        # The werkzeug dev server doesn't expose a clean stop API from
        # outside the request context, but as a daemon thread it will be
        # reaped on process exit. We just log and let it go.
        logger.info("Dashboard server shutting down")
        self._thread = None

    # ── internal ────────────────────────────────────────────────────────

    def _run_server(self) -> None:
        """Target for the daemon thread."""
        try:
            from flo_pro_sdk.dashboard.run_browser import RunBrowser
            from flo_pro_sdk.dashboard.dash.app import DashboardDashApp

            browser = RunBrowser(self._run_dir)
            app = DashboardDashApp(
                browser,
                refresh_interval=self._config.refresh_interval_ms,
            )
            self._started.set()
            assert self._actual_port is not None
            app.run(
                host=self._config.host,
                port=self._actual_port,
                debug=False,
            )
        except Exception:
            self._failed = True
            self._started.set()  # unblock start() even on failure
            logger.exception("Dashboard server failed to start")
