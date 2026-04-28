"""Agents tab layout for the Dash dashboard."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc

from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider
from flo_pro_sdk.dashboard.dash.constants import (
    CARD_HEADER_STYLE,
    CARD_HEADER_BG,
)
from flo_pro_sdk.dashboard.dash.layout_convergence import (
    get_group_names,
    build_filtered_var_options,
)


def build_agent_info(
    provider: DashboardDataProvider,
    agent_id: str | None,
) -> list:
    """Build label/value info grid for an agent."""
    if not agent_id:
        return [html.Div("Select an agent", className="text-muted")]

    metadata = provider.get_problem_metadata()
    if not metadata:
        return [html.Div("Metadata not available", className="text-muted")]

    agents_list = metadata.get("agents", [])
    agent_meta = next(
        (a for a in agents_list if a["agent_id"] == agent_id),
        None,
    )
    subs = metadata.get("subscriptions", {}).get(agent_id, {})

    _label_style = {
        "fontSize": "0.75em",
        "color": "#7f8c8d",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
    }
    _value_style = {"fontSize": "0.95em", "fontWeight": "500", "color": "#2c3e50"}

    total_vars = sum(len(idx) for idx in subs.values())
    group_summary = ", ".join(f"{g} ({len(idx)})" for g, idx in sorted(subs.items()))

    fields = [
        ("Agent ID", agent_id),
        ("Groups", group_summary or "—"),
        ("Total Variables", str(total_vars)),
    ]

    # Collect all metadata into a single compact field
    if agent_meta and agent_meta.get("metadata"):
        meta_str = ", ".join(
            f"{k}={v}" for k, v in sorted(agent_meta["metadata"].items())
        )
        fields.append(("Metadata", meta_str))

    # Wrap into rows of 5 columns each
    per_row = 5
    rows: list = []
    for i in range(0, len(fields), per_row):
        chunk = fields[i : i + per_row]
        cols = [
            dbc.Col(
                [
                    html.Div(label, style=_label_style),
                    html.Div(value, style=_value_style),
                ],
                width=True,
            )
            for label, value in chunk
        ]
        rows.append(
            dbc.Row(cols, className="mb-2" if i + per_row < len(fields) else "")
        )
    return rows


def build_agents_layout(
    provider: DashboardDataProvider,
) -> dbc.Container:
    """Build the Agents tab content."""
    agents = provider.list_agent_ids()
    default_agent = agents[0] if agents else None
    group_names = get_group_names(provider)
    default_group = group_names[0] if group_names else None

    # Pre-build variable options for default agent
    default_var_opts = (
        build_filtered_var_options(
            provider,
            group_name=default_group,
            agent_id=default_agent,
        )
        if default_agent
        else []
    )

    # Build agent info for default agent
    agent_info_content = build_agent_info(provider, default_agent)

    return dbc.Container(
        [
            # Refresh button
            dbc.Row(
                dbc.Col(
                    dbc.Button(
                        "↻ Refresh",
                        id="dd-refresh-btn",
                        n_clicks=0,
                        color="outline-secondary",
                        size="sm",
                    ),
                    width="auto",
                    className="ms-auto",
                ),
                className="mb-2",
                justify="end",
            ),
            # Agent selector + info side by side
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.Span("Agent", style=CARD_HEADER_STYLE),
                                        dcc.Dropdown(
                                            id="dd-agent-selector",
                                            options=[
                                                {"label": a, "value": a} for a in agents
                                            ],
                                            value=default_agent,
                                            clearable=False,
                                            style={
                                                "flex": "1",
                                                "maxWidth": "300px",
                                                "marginLeft": "16px",
                                            },
                                        ),
                                    ],
                                    className="d-flex align-items-center",
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(
                                    id="dd-agent-info", children=agent_info_content
                                ),
                            ]
                        ),
                        width=12,
                    ),
                ],
                className="mb-3",
            ),
            # Row 1: Objective decomposition | Agent residual
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span(
                                        "Objective Decomposition",
                                        style=CARD_HEADER_STYLE,
                                    ),
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(dcc.Graph(id="dd-objective-decomp-plot")),
                            ]
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span(
                                        "Agent Residual", style=CARD_HEADER_STYLE
                                    ),
                                    style=CARD_HEADER_BG,
                                ),
                                dbc.CardBody(dcc.Graph(id="dd-residual-plot")),
                            ]
                        ),
                        width=6,
                    ),
                ],
                className="mb-3",
            ),
            # Row 2: Preferred vs Consensus with variable filters
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.Span("Preferred vs Consensus", style=CARD_HEADER_STYLE),
                        style=CARD_HEADER_BG,
                    ),
                    dbc.CardBody(
                        [
                            # Filter row: group | index range | metadata search
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="dd-group-filter",
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
                                                        id="dd-idx-min",
                                                        type="number",
                                                        min=0,
                                                        max=999999,
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
                                                        id="dd-idx-max",
                                                        type="number",
                                                        min=0,
                                                        max=999999,
                                                        value=999999,
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
                                            id="dd-meta-filter",
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
                            dcc.Dropdown(
                                id="dd-variable-dropdown",
                                options=default_var_opts,
                                value=default_var_opts[0]["value"]
                                if default_var_opts
                                else None,
                                multi=False,
                                placeholder="Select a variable to compare...",
                                className="mb-3",
                                style={"fontSize": "0.85em"},
                            ),
                            dcc.Graph(id="dd-pref-vs-consensus-plot"),
                        ]
                    ),
                ],
                className="mb-3",
            ),
        ],
        fluid=True,
    )
