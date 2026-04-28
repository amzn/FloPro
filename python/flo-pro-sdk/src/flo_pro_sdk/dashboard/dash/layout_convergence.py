"""Convergence tab layout and variable filter helpers for the Dash dashboard."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider
from flo_pro_sdk.dashboard.dash.constants import (
    CARD_HEADER_STYLE,
    CARD_HEADER_BG,
    parse_filter_query,
    row_matches_filters,
)

_SECTION_STYLE = {
    "fontSize": "0.8em",
    "fontWeight": "600",
    "color": "#7f8c8d",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "borderBottom": "1px solid #ecf0f1",
    "paddingBottom": "6px",
}


# ── Variable filter helpers (shared by convergence + agents) ────────


def get_group_names(provider: DashboardDataProvider) -> list[str]:
    """Return sorted list of variable group names."""
    metadata = provider.get_problem_metadata()
    if not metadata:
        return []
    return sorted(vg["name"] for vg in metadata.get("variable_groups", []))


def get_group_max_index(provider: DashboardDataProvider, group_name: str) -> int:
    """Return max index for a variable group (count - 1)."""
    metadata = provider.get_problem_metadata()
    if not metadata:
        return 0
    for vg in metadata.get("variable_groups", []):
        if vg["name"] == group_name:
            return vg["count"] - 1
    return 0


def build_filtered_var_options(
    provider: DashboardDataProvider,
    group_name: str | None = None,
    idx_min: int | None = None,
    idx_max: int | None = None,
    meta_filter: str = "",
    agent_id: str | None = None,
) -> list[dict]:
    """Build variable options filtered by group, index range, and metadata.

    Args:
        provider: Data provider instance.
        group_name: If set, only include variables from this group.
        idx_min: Minimum index within the group.
        idx_max: Maximum index within the group.
        meta_filter: Free-text filter like ``node contains bus_1``.
        agent_id: If set, only include variables the agent subscribes to.

    Returns:
        List of ``{"label": "energy[0] (bus_0)", "value": "energy[0]"}``
        dicts suitable for a Dash dropdown.
    """
    metadata = provider.get_problem_metadata()
    if not metadata:
        return []

    var_groups = metadata.get("variable_groups", [])
    var_metadata_dict = metadata.get("var_metadata", {})
    subs = metadata.get("subscriptions", {})
    agent_subs = subs.get(agent_id, {}) if agent_id else None

    filters = parse_filter_query(meta_filter) if meta_filter else []

    options: list[dict] = []
    for vg in sorted(var_groups, key=lambda v: v["slice_start"]):
        gname = vg["name"]
        if group_name and gname != group_name:
            continue

        vm_df = var_metadata_dict.get(gname)
        vm_cols = list(vm_df.columns) if vm_df is not None and not vm_df.empty else []

        # Determine which indices to iterate
        if agent_id and agent_subs is not None:
            indices = sorted(agent_subs.get(gname, []))
        else:
            indices = list(range(vg["count"]))

        lo = idx_min if idx_min is not None else 0
        hi = idx_max if idx_max is not None else vg["count"] - 1

        for j in indices:
            if j < lo or j > hi:
                continue

            # Build metadata row for filter matching
            if filters and vm_df is not None and j < len(vm_df):
                row = {col: str(vm_df.iloc[j][col]) for col in vm_cols}
                if not row_matches_filters(row, filters):
                    continue

            label = f"{gname}[{j}]"
            # Enrich label with first metadata value for context
            if vm_df is not None and vm_cols and j < len(vm_df):
                display_cols = [c for c in vm_cols if c not in ("t", "index")]
                if display_cols:
                    meta_val = str(vm_df.iloc[j][display_cols[0]])
                    label = f"{gname}[{j}] ({meta_val})"

            options.append({"label": label, "value": f"{gname}[{j}]"})

    return options


# ── Convergence tab layout ──────────────────────────────────────────


def build_convergence_layout(
    provider: DashboardDataProvider,
) -> dbc.Container:
    """Build the Convergence tab content."""
    agents = provider.list_agent_ids()
    group_names = get_group_names(provider)
    default_group = group_names[0] if group_names else None
    default_max_idx = (
        get_group_max_index(provider, default_group) if default_group else 0
    )

    # Build initial filtered options
    initial_opts = build_filtered_var_options(provider, group_name=default_group)
    default_vars = [o["value"] for o in initial_opts[:5]]

    # Compute iteration range for slider
    conv = provider.get_convergence_data()
    max_iter = int(conv["iteration"].max()) if not conv.empty else 100

    return dbc.Container(
        [
            # ── Global controls bar ──
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        "Iteration",
                                        id="x-btn-iter",
                                        n_clicks=0,
                                        color="primary",
                                        outline=False,
                                        size="sm",
                                    ),
                                    dbc.Button(
                                        "Wall-Clock",
                                        id="x-btn-time",
                                        n_clicks=0,
                                        color="primary",
                                        outline=True,
                                        size="sm",
                                    ),
                                ],
                                size="sm",
                            ),
                            dcc.Store(id="x-axis-toggle", data="iteration"),
                        ],
                        width="auto",
                        className="d-flex align-items-center me-4",
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                "Filter Iterations",
                                className="text-center",
                                style={
                                    "fontSize": "0.8em",
                                    "fontWeight": "500",
                                    "color": "#7f8c8d",
                                    "marginBottom": "2px",
                                },
                            ),
                            dcc.RangeSlider(
                                id="iter-range-slider",
                                min=0,
                                max=max_iter,
                                step=1,
                                value=[0, max_iter],
                                marks={0: "0", max_iter: str(max_iter)},
                                tooltip={
                                    "placement": "bottom",
                                    "always_visible": False,
                                },
                            ),
                        ]
                    ),
                    dbc.Col(
                        dbc.Button(
                            "↻",
                            id="refresh-btn",
                            n_clicks=0,
                            color="outline-secondary",
                            size="sm",
                            title="Refresh data",
                        ),
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                ],
                className="mb-3 align-items-center gx-3",
            ),
            # ── Global section ──
            html.Div("Global", style=_SECTION_STYLE, className="mb-3"),
            # Row 1: Convergence | Variable Trajectories
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Span(
                                            "Convergence", style=CARD_HEADER_STYLE
                                        ),
                                        dbc.Checklist(
                                            id="conv-rate-toggle",
                                            options=[
                                                {"label": "Conv. Rate", "value": "show"}
                                            ],
                                            value=[],
                                            inline=True,
                                            inputClassName="me-1",
                                            labelStyle={
                                                "fontSize": "0.8em",
                                                "color": "rgba(255,255,255,0.8)",
                                            },
                                            className="ms-auto",
                                        ),
                                    ],
                                    className="d-flex align-items-center",
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(dcc.Graph(id="convergence-plot")),
                            ],
                            className="h-100",
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span(
                                        "Variable Trajectories", style=CARD_HEADER_STYLE
                                    ),
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(
                                    [
                                        # Filter row: group | index range | metadata search
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id="vt-group-filter",
                                                        options=[
                                                            {"label": g, "value": g}
                                                            for g in group_names
                                                        ],
                                                        value=default_group,
                                                        clearable=False,
                                                        placeholder="Group",
                                                        style={"fontSize": "0.85em"},
                                                    ),
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.InputGroup(
                                                            [
                                                                dcc.Input(
                                                                    id="vt-idx-min",
                                                                    type="number",
                                                                    min=0,
                                                                    max=default_max_idx,
                                                                    value=0,
                                                                    placeholder="min",
                                                                    debounce=True,
                                                                    style={
                                                                        "width": "60px",
                                                                        "fontSize": "0.8em",
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    "–",
                                                                    className="mx-1 align-self-center small",
                                                                ),
                                                                dcc.Input(
                                                                    id="vt-idx-max",
                                                                    type="number",
                                                                    min=0,
                                                                    max=default_max_idx,
                                                                    value=default_max_idx,
                                                                    placeholder="max",
                                                                    debounce=True,
                                                                    style={
                                                                        "width": "60px",
                                                                        "fontSize": "0.8em",
                                                                    },
                                                                ),
                                                            ],
                                                            size="sm",
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    dcc.Input(
                                                        id="vt-meta-filter",
                                                        type="text",
                                                        value="",
                                                        placeholder="e.g. {node} contains bus_1",
                                                        debounce=True,
                                                        style={"fontSize": "0.8em"},
                                                    ),
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-2 align-items-center gx-2",
                                        ),
                                        # Variable multi-select (populated by callback)
                                        dcc.Dropdown(
                                            id="variable-dropdown",
                                            options=initial_opts,
                                            value=default_vars,
                                            multi=True,
                                            placeholder="Select variables...",
                                            className="mb-3",
                                            style={"fontSize": "0.85em"},
                                        ),
                                        dcc.Graph(id="variable-trajectories-plot"),
                                    ]
                                ),
                            ],
                            className="h-100",
                        ),
                        width=6,
                    ),
                ],
                className="mb-4",
            ),
            # ── Agent Comparison section ──
            html.Div("Agent Comparison", style=_SECTION_STYLE, className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="agent-dropdown",
                            options=[{"label": a, "value": a} for a in agents],
                            value=agents[:5] if agents else [],
                            multi=True,
                            placeholder="Select agents to compare...",
                        )
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span("Objectives", style=CARD_HEADER_STYLE),
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(dcc.Graph(id="objectives-plot")),
                            ]
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span("Residuals", style=CARD_HEADER_STYLE),
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(dcc.Graph(id="residuals-plot")),
                            ]
                        ),
                        width=6,
                    ),
                ]
            ),
        ],
        fluid=True,
    )
