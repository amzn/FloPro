"""Problem Details tab layout for the Dash dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider
from flo_pro_sdk.dashboard.dash.constants import (
    CARD_HEADER_STYLE,
    CARD_HEADER_BG,
    TABLE_STYLE_HEADER,
    TABLE_STYLE_CELL,
    TABLE_STYLE_DATA_COND,
    parse_filter_query,
    row_matches_filters,
)

SUB_MAP_PAGE_SIZE = 20


def build_problem_details_layout(provider: DashboardDataProvider) -> dbc.Container:
    """Build the Problem Details tab content."""
    metadata = provider.get_problem_metadata()
    manifest = provider.get_manifest()
    sections: list = []

    if manifest:
        sections.append(_run_info_card(manifest))

    if not metadata:
        sections.append(
            dbc.Alert(
                "Problem metadata not yet available.",
                color="info",
                className="text-center",
            )
        )
        return dbc.Container(sections, fluid=True)

    # Stat cards
    total_vars = metadata.get("total_variable_count", "?")
    n_agents = len(metadata.get("agents", []))
    n_groups = len(metadata.get("variable_groups", []))
    conv = provider.get_convergence_data()
    n_iters = int(conv["iteration"].max()) + 1 if not conv.empty else 0

    stats = [
        ("Agents", n_agents, "primary"),
        ("Variable Groups", n_groups, "info"),
        ("Total Variables", total_vars, "secondary"),
        ("Iterations", n_iters, "success"),
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

    # Agents table
    agents_list = metadata.get("agents", [])
    subs = metadata.get("subscriptions", {})
    var_groups = {vg["name"]: vg for vg in metadata.get("variable_groups", [])}

    if agents_list:
        sections.append(_agents_card(agents_list, subs))

    # Variable groups summary
    if var_groups:
        vg_rows = [
            {
                "Group": name,
                "Variables": vg["count"],
                "Index Range": f"{vg['slice_start']}–{vg['slice_end'] - 1}",
            }
            for name, vg in sorted(var_groups.items())
        ]
        cols = [{"name": c, "id": c} for c in ["Group", "Variables", "Index Range"]]
        sections.append(
            dbc.Card(
                [
                    dbc.CardHeader(
                        html.Span("Variable Groups", style=CARD_HEADER_STYLE),
                        style=CARD_HEADER_BG,
                    ),
                    dbc.CardBody(
                        dash_table.DataTable(
                            data=vg_rows,
                            columns=cols,
                            style_table={"overflowX": "auto"},
                            style_cell=TABLE_STYLE_CELL,
                            style_header=TABLE_STYLE_HEADER,
                            style_data_conditional=TABLE_STYLE_DATA_COND,
                            cell_selectable=False,
                            row_selectable=False,
                        )
                    ),
                ],
                className="mb-3",
            )
        )

    # Subscription map
    sub_card = build_subscription_map_card(provider, metadata)
    if sub_card is not None:
        sections.append(sub_card)

    return dbc.Container(sections, fluid=True)


def _run_info_card(manifest: dict) -> dbc.Card:
    """Build the Run Info card from manifest data."""

    def _fmt_ts(val):
        try:
            if isinstance(val, (int, float)) and val > 1e9:
                dt = datetime.fromtimestamp(val, tz=timezone.utc)
            elif isinstance(val, str) and val:
                dt = datetime.fromisoformat(val)
            else:
                return "—"
            return dt.strftime("%b %d, %Y %I:%M:%S %p %Z")
        except (ValueError, TypeError):
            return str(val) if val else "—"

    started = manifest.get("started_at", "")
    completed = manifest.get("completed_at", "")

    duration = "—"
    try:
        if started and completed:
            t0 = datetime.fromisoformat(str(started))
            t1 = datetime.fromisoformat(str(completed))
            delta = (t1 - t0).total_seconds()
            if delta < 60:
                duration = f"{delta:.1f}s"
            elif delta < 3600:
                duration = f"{delta / 60:.1f}m"
            else:
                duration = f"{delta / 3600:.1f}h"
    except (ValueError, TypeError):
        pass

    fields = [
        ("Run ID", manifest.get("run_id", "—")),
        ("Status", manifest.get("status", "—")),
        ("Started", _fmt_ts(started)),
        ("Completed", _fmt_ts(completed)),
        ("Duration", duration),
    ]
    if manifest.get("resumed_from"):
        fields.append(("Resumed From", str(manifest["resumed_from"])))

    _label_style = {
        "fontSize": "0.75em",
        "color": "#7f8c8d",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
    }
    _value_style = {"fontSize": "0.95em", "fontWeight": "500", "color": "#2c3e50"}

    info_cols = [
        dbc.Col(
            [
                html.Div(label, style=_label_style),
                html.Div(str(value), style=_value_style),
            ]
        )
        for label, value in fields
    ]
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Span("Run Info", style=CARD_HEADER_STYLE), style=CARD_HEADER_BG
            ),
            dbc.CardBody(dbc.Row(info_cols)),
        ],
        className="mb-3",
    )


def _agents_card(agents_list: list, subs: dict) -> dbc.Card:
    """Build the Agents table card."""
    all_meta_keys: Counter = Counter()
    for a in agents_list:
        for k in a.get("metadata") or {}:
            all_meta_keys[k] += 1

    max_meta_cols = 5
    top_keys = [k for k, _ in all_meta_keys.most_common(max_meta_cols)]
    overflow_keys = set(all_meta_keys) - set(top_keys)

    rows = []
    for a in agents_list:
        aid = a["agent_id"]
        agent_subs = subs.get(aid, {})
        group_summary = ", ".join(
            f"{g}({len(idx)})" for g, idx in sorted(agent_subs.items())
        )
        total_sub_vars = sum(len(idx) for idx in agent_subs.values())
        meta = a.get("metadata") or {}
        row = {
            "Agent": aid,
            "Groups": group_summary or "—",
            "Subscribed Vars": total_sub_vars,
        }
        for k in top_keys:
            row[k] = str(meta[k]) if k in meta else "—"
        if overflow_keys:
            overflow = {k: v for k, v in meta.items() if k in overflow_keys}
            row["Other Metadata"] = (
                ", ".join(f"{k}={v}" for k, v in sorted(overflow.items()))
                if overflow
                else "—"
            )
        rows.append(row)

    fixed_cols = ["Agent", "Groups", "Subscribed Vars"]
    meta_col_names = top_keys[:]
    if overflow_keys:
        meta_col_names.append("Other Metadata")

    all_col_names = ["Agent"] + meta_col_names + ["Groups", "Subscribed Vars"]
    all_cols = [{"name": c, "id": c} for c in all_col_names]
    default_meta: list = []

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span("Agents", style=CARD_HEADER_STYLE),
                    html.Span(
                        "Show metadata:",
                        className="ms-auto me-2 small",
                        style={"color": "rgba(255,255,255,0.7)"},
                    ),
                    dcc.Dropdown(
                        id="agents-col-selector",
                        options=[{"label": c, "value": c} for c in meta_col_names],
                        value=default_meta,
                        multi=True,
                        placeholder="+ Add metadata columns",
                        style={"flex": "1", "maxWidth": "400px"},
                    )
                    if meta_col_names
                    else None,
                ],
                className="d-flex align-items-center",
                style=CARD_HEADER_BG,
            ),
            dbc.CardBody(
                dash_table.DataTable(
                    id="agents-table",
                    data=rows,
                    columns=[
                        c
                        for c in all_cols
                        if c["id"] in fixed_cols or c["id"] in default_meta
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell=TABLE_STYLE_CELL,
                    style_header=TABLE_STYLE_HEADER,
                    style_data_conditional=TABLE_STYLE_DATA_COND,
                    cell_selectable=False,
                    row_selectable=False,
                    filter_action="native",
                )
            ),
            dcc.Store(id="agents-all-columns", data=all_cols),
            dcc.Store(id="agents-fixed-columns", data=fixed_cols),
        ],
        className="mb-3",
    )


# ── Subscription map ────────────────────────────────────────────────


def build_subscription_map_card(
    provider: DashboardDataProvider,
    metadata: dict,
) -> "dbc.Card | None":
    """Build a group-filtered subscription map with var_metadata columns."""
    agents_list = metadata.get("agents", [])
    var_groups_list = metadata.get("variable_groups", [])
    if not agents_list or not var_groups_list:
        return None

    sorted_vgs = sorted(var_groups_list, key=lambda v: v["slice_start"])
    group_names = [vg["name"] for vg in sorted_vgs]
    default_group = group_names[0]

    rows, columns, total = build_sub_map_page(default_group, metadata, page=0)

    default_vg = next(v for v in sorted_vgs if v["name"] == default_group)
    default_max = default_vg["count"] - 1

    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Span("Subscription Map", style=CARD_HEADER_STYLE),
                    dcc.Dropdown(
                        id="sub-map-group-filter",
                        options=[{"label": g, "value": g} for g in group_names],
                        value=default_group,
                        clearable=False,
                        style={"flex": "1", "maxWidth": "300px", "marginLeft": "16px"},
                    ),
                    html.Span(
                        "Index:",
                        className="ms-3 me-1 small",
                        style={"color": "rgba(255,255,255,0.7)"},
                    ),
                    dcc.Input(
                        id="sub-map-idx-min",
                        type="number",
                        min=0,
                        max=default_max,
                        value=0,
                        placeholder="min",
                        debounce=True,
                        style={"width": "80px", "fontSize": "0.85em"},
                    ),
                    html.Span(
                        "–", className="mx-1", style={"color": "rgba(255,255,255,0.7)"}
                    ),
                    dcc.Input(
                        id="sub-map-idx-max",
                        type="number",
                        min=0,
                        max=default_max,
                        value=default_max,
                        placeholder="max",
                        debounce=True,
                        style={"width": "80px", "fontSize": "0.85em"},
                    ),
                ],
                className="d-flex align-items-center",
                style=CARD_HEADER_BG,
            ),
            dbc.CardBody(
                [
                    dash_table.DataTable(
                        id="sub-map-table",
                        data=rows,
                        columns=columns,
                        page_size=SUB_MAP_PAGE_SIZE,
                        page_action="custom",
                        page_current=0,
                        page_count=max(1, -(-total // SUB_MAP_PAGE_SIZE)),
                        style_table={"overflowX": "auto"},
                        style_cell=TABLE_STYLE_CELL,
                        style_header=TABLE_STYLE_HEADER,
                        style_data_conditional=TABLE_STYLE_DATA_COND
                        + [
                            {
                                "if": {"filter_query": "{Subscribers} = 0"},
                                "color": "#ccc",
                            },
                        ],
                        cell_selectable=False,
                        row_selectable=False,
                        filter_action="custom",
                        filter_query="",
                    ),
                    html.Div(
                        id="sub-map-total",
                        className="text-muted small mt-1",
                        children=f"{total:,} variables",
                    ),
                ]
            ),
        ],
        className="mb-3",
    )


def build_sub_map_page(
    group_name: str,
    metadata: dict,
    page: int = 0,
    filter_query: str = "",
    idx_min: int | None = None,
    idx_max: int | None = None,
) -> tuple[list[dict], list[dict], int]:
    """Build one page of subscription map rows for a group."""
    agents_list = metadata.get("agents", [])
    subs = metadata.get("subscriptions", {})
    var_groups_list = metadata.get("variable_groups", [])
    var_metadata_dict = metadata.get("var_metadata", {})

    vg = next((v for v in var_groups_list if v["name"] == group_name), None)
    if vg is None:
        return [], [], 0

    idx_to_agents: dict[int, list[str]] = {}
    for a in agents_list:
        aid = a["agent_id"]
        for idx in subs.get(aid, {}).get(group_name, []):
            idx_to_agents.setdefault(idx, []).append(aid)

    vm_df = var_metadata_dict.get(group_name)
    vm_cols = list(vm_df.columns) if vm_df is not None and not vm_df.empty else []

    filters = parse_filter_query(filter_query)

    lo = idx_min if idx_min is not None else 0
    hi = idx_max if idx_max is not None else vg["count"] - 1
    lo = max(0, lo)
    hi = min(vg["count"] - 1, hi)

    matching_rows: list[dict] = []
    ps = SUB_MAP_PAGE_SIZE
    start = page * ps
    match_count = 0
    for j in range(lo, hi + 1):
        subscribed = idx_to_agents.get(j, [])
        row: dict = {"Variable": f"{group_name}[{j}]", "Index": j}
        if vm_df is not None and j < len(vm_df):
            for col in vm_cols:
                row[col] = str(vm_df.iloc[j][col])
        row["Subscribers"] = len(subscribed)
        row["Agents"] = ", ".join(sorted(subscribed)) or "—"

        if row_matches_filters(row, filters):
            if start <= match_count < start + ps:
                matching_rows.append(row)
            match_count += 1

    col_order = ["Variable", "Index"] + vm_cols + ["Subscribers", "Agents"]
    columns = [{"name": c, "id": c} for c in col_order]
    return matching_rows, columns, match_count
