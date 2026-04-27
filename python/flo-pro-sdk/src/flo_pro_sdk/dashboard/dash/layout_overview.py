"""Overview page layout for multi-run browsing.

Shows a summary table of all coordinations and runs, plus a convergence
comparison chart for selected runs.
"""

from __future__ import annotations

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from flo_pro_sdk.dashboard.run_browser import RunBrowser
from flo_pro_sdk.dashboard.dash.constants import (
    CARD_HEADER_STYLE,
    CARD_HEADER_BG,
    TABLE_STYLE_HEADER,
    TABLE_STYLE_CELL,
    TABLE_STYLE_DATA_COND,
)


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def build_overview_layout(browser: RunBrowser) -> dbc.Container:
    """Build the overview page showing all coordinations and runs."""
    browser.refresh()
    sections: list = []

    # Summary stats
    n_coords = len(browser.coordinations)
    n_runs = browser.total_runs
    n_completed = sum(1 for r in browser.all_runs if r.status == "completed")
    n_running = sum(1 for r in browser.all_runs if r.status == "running")

    stats = [
        ("Coordinations", n_coords, "primary"),
        ("Total Runs", n_runs, "info"),
        ("Completed", n_completed, "success"),
        ("Running", n_running, "warning"),
    ]
    sections.append(
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H2(
                                        str(val),
                                        className="card-title text-center mb-0",
                                        style={"color": f"var(--bs-{color})"},
                                    ),
                                    html.P(
                                        label,
                                        className="card-text text-center text-muted small",
                                    ),
                                ]
                            ),
                        ]
                    ),
                    width=3,
                )
                for label, val, color in stats
            ],
            className="mb-3",
        )
    )

    # Runs table — Run column and Resumed From use markdown links
    rows = []
    run_options = []
    # Build a lookup for resumed_from → (coord_id, run_id) so we can link
    run_lookup: dict[str, tuple[str, str]] = {}
    for coord in browser.coordinations:
        for run in coord.runs:
            run_lookup[run.run_id] = (run.coordination_id, run.run_id)

    for coord in browser.coordinations:
        for run in coord.runs:
            link = f"/run/{run.coordination_id}/{run.run_id}"
            # Resumed From: link if the target run exists in our browser
            resumed = "—"
            if run.resumed_from:
                target = run_lookup.get(run.resumed_from)
                if target:
                    resumed = f"[{run.resumed_from[:12]}](/run/{target[0]}/{target[1]})"
                else:
                    resumed = run.resumed_from[:12]
            row = {
                "Coordination": coord.coordination_id[:16],
                "Run": f"[{run.run_id[:16]}]({link})",
                "Status": run.status,
                "Iterations": run.final_iteration
                if run.final_iteration is not None
                else "—",
                "Duration": _fmt_duration(run.duration_seconds),
                "Resumed From": resumed,
                # Hidden columns for comparison dropdown
                "_coord_id": run.coordination_id,
                "_run_id": run.run_id,
            }
            rows.append(row)
            label = f"{coord.coordination_id[:12]} / {run.run_id[:12]} ({run.status})"
            run_options.append(
                {
                    "label": label,
                    "value": f"{run.coordination_id}|{run.run_id}",
                }
            )

    link_cols = {"Run", "Resumed From"}
    columns = [
        {"name": c, "id": c, "presentation": "markdown"}
        if c in link_cols
        else {"name": c, "id": c}
        for c in [
            "Coordination",
            "Run",
            "Status",
            "Iterations",
            "Duration",
            "Resumed From",
        ]
    ]

    sections.append(
        dbc.Card(
            [
                dbc.CardHeader(
                    html.Span("Runs", style=CARD_HEADER_STYLE),
                    style=CARD_HEADER_BG,
                ),
                dbc.CardBody(
                    [
                        dash_table.DataTable(
                            id="overview-runs-table",
                            data=rows,
                            columns=columns,
                            style_table={"overflowX": "auto"},
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data_conditional=TABLE_STYLE_DATA_COND
                            + [
                                {
                                    "if": {
                                        "filter_query": '{Status} = "completed"',
                                        "column_id": "Status",
                                    },
                                    "color": "#27ae60",
                                    "fontWeight": "600",
                                },
                                {
                                    "if": {
                                        "filter_query": '{Status} = "running"',
                                        "column_id": "Status",
                                    },
                                    "color": "#f39c12",
                                    "fontWeight": "600",
                                },
                            ],
                            cell_selectable=False,
                            row_selectable=False,
                            filter_action="native",
                            markdown_options={"link_target": "_self"},
                        ),
                    ]
                ),
            ],
            className="mb-3",
        )
    )

    # Convergence comparison
    sections.append(
        dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.Span("Convergence Comparison", style=CARD_HEADER_STYLE),
                        dcc.Dropdown(
                            id="overview-compare-runs",
                            options=run_options,
                            value=[o["value"] for o in run_options[:5]],
                            multi=True,
                            placeholder="Select runs to compare...",
                            style={
                                "flex": "1",
                                "maxWidth": "600px",
                                "marginLeft": "16px",
                            },
                        ),
                    ],
                    className="d-flex align-items-center",
                    style=CARD_HEADER_BG,
                ),
                dbc.CardBody(dcc.Graph(id="overview-convergence-comparison")),
            ],
            className="mb-3",
        )
    )

    return dbc.Container(sections, fluid=True)
