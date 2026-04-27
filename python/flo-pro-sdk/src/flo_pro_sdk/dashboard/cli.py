"""CLI entry point for the offline dashboard viewer.

Usage::

    flo-dashboard /path/to/run_dir
    flo-dashboard /path/to/base_dir
    flo-dashboard /path/to/coordination_dir
    flo-dashboard /path/to/run_dir --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flo-dashboard",
        description="Launch the coordination dashboard for completed or in-progress runs.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a run directory, coordination directory, or base directory.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number (default: 8050).",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Disable auto-refresh (useful for completed runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    target = args.path.resolve()
    if not target.exists():
        print(f"Error: path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    from flo_pro_sdk.dashboard.run_browser import RunBrowser

    browser = RunBrowser(target)
    if browser.total_runs == 0:
        print(f"Error: no run directories found under {target}", file=sys.stderr)
        sys.exit(1)

    logger.info(
        "Found %d run(s) across %d coordination(s)",
        browser.total_runs,
        len(browser.coordinations),
    )

    refresh_ms = 0 if args.no_refresh else 5000

    try:
        from flo_pro_sdk.dashboard.dash.app import DashboardDashApp
    except ImportError:
        print(
            "Error: dashboard dependencies not installed. "
            "Install with: pip install flo-pro-sdk[dashboard]",
            file=sys.stderr,
        )
        sys.exit(1)

    app = DashboardDashApp(browser, refresh_interval=refresh_ms)
    print(f"Dashboard: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
