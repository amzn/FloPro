"""Plotly figure builders for the Dash dashboard.

All functions take a provider/computer and return ``go.Figure`` instances.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flo_pro_sdk.dashboard.data_provider import DashboardDataProvider
from flo_pro_sdk.dashboard.metrics import DashboardMetricsComputer
from flo_pro_sdk.dashboard.dash.constants import (
    COLORS,
    PLOT_LAYOUT,
    crossref_hover,
    get_x_data,
    ts_to_datetime,
)


def convergence_figure(
    provider: DashboardDataProvider,
    computer: DashboardMetricsComputer,
    x_axis: str = "iteration",
    iter_range: list | None = None,
    show_rate: bool = False,
) -> go.Figure:
    df = provider.get_convergence_data()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        fig.add_annotation(
            text="No convergence data yet",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    if iter_range and len(iter_range) == 2:
        df = df[(df["iteration"] >= iter_range[0]) & (df["iteration"] <= iter_range[1])]
    if df.empty:
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    x_vals, x_title, use_time = get_x_data(df, x_axis)
    cd, xref = crossref_hover(df, use_time)

    # Primal residual (first trace gets cross-ref suffix)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["primal_residual"],
            name="Primal",
            line=dict(color=COLORS[0], width=2),
            customdata=cd,
            hovertemplate="%{y:.4g}" + xref + "<extra>Primal</extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df["dual_residual"],
            name="Dual",
            line=dict(color=COLORS[1], width=2),
            hovertemplate="%{y:.4g}<extra>Dual</extra>",
        ),
        secondary_y=False,
    )
    fig.update_yaxes(type="log", title="Residual (log)", secondary_y=False)

    # Convergence rate overlay
    if show_rate:
        rate_df = computer.get_convergence_rate(window=5)
        if not rate_df.empty:
            if iter_range and len(iter_range) == 2:
                rate_df = rate_df[
                    (rate_df["iteration"] >= iter_range[0])
                    & (rate_df["iteration"] <= iter_range[1])
                ]
            if not rate_df.empty:
                rate_x = rate_df["iteration"]
                if use_time:
                    merged = rate_df.merge(
                        df[["iteration", "timestamp"]],
                        on="iteration",
                        how="inner",
                    )
                    rate_x = ts_to_datetime(merged["timestamp"])
                    rate_y = merged["convergence_rate"]
                else:
                    rate_y = rate_df["convergence_rate"]
                fig.add_trace(
                    go.Scatter(
                        x=rate_x,
                        y=rate_y.abs(),
                        name="Conv. Rate",
                        line=dict(color="#9c755f", width=1.5, dash="dashdot"),
                        hovertemplate="%{y:.4g}<extra>Conv. Rate</extra>",
                    ),
                    secondary_y=False,
                )

    # Total objective on secondary y-axis
    obj = computer.get_total_objective()
    if not obj.empty:
        if iter_range and len(iter_range) == 2:
            obj = obj[
                (obj["iteration"] >= iter_range[0])
                & (obj["iteration"] <= iter_range[1])
            ]
        if not obj.empty:
            if use_time:
                obj_merged = obj.merge(
                    df[["iteration", "timestamp"]],
                    on="iteration",
                    how="inner",
                )
                obj_x = ts_to_datetime(obj_merged["timestamp"])
                obj_y = obj_merged["total_objective"]
            else:
                obj_x = obj["iteration"]
                obj_y = obj["total_objective"]
            fig.add_trace(
                go.Scatter(
                    x=obj_x,
                    y=obj_y,
                    name="Total Obj.",
                    line=dict(color=COLORS[4], width=2, dash="dot"),
                    hovertemplate="%{y:.4g}<extra>Total Obj.</extra>",
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title="Total Objective", secondary_y=True)

    fig.update_xaxes(title=x_title)
    if use_time:
        fig.update_xaxes(tickformat="%H:%M:%S")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def objectives_figure(
    provider: DashboardDataProvider,
    computer: DashboardMetricsComputer,
    agents: list,
    x_axis: str = "iteration",
    iter_range: list | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not agents:
        fig.add_annotation(
            text="Select agents above",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    conv_df = provider.get_convergence_data()
    first_trace_added = False
    use_time = (
        x_axis == "timestamp"
        and not conv_df.empty
        and "timestamp" in conv_df.columns
        and conv_df["timestamp"].notna().any()
    )

    for i, aid in enumerate(agents):
        sol = provider.get_agent_solutions(aid, columns=["utility"])
        if sol.empty:
            continue
        if iter_range and len(iter_range) == 2:
            sol = sol[
                (sol["iteration"] >= iter_range[0])
                & (sol["iteration"] <= iter_range[1])
            ]
        if sol.empty:
            continue
        if use_time:
            sol = sol.merge(
                conv_df[["iteration", "timestamp"]], on="iteration", how="inner"
            )
            x_vals = ts_to_datetime(sol["timestamp"])
        else:
            x_vals = sol["iteration"]
        if not first_trace_added:
            cd, xref = crossref_hover(sol, use_time)
            first_trace_added = True
        else:
            cd, xref = None, ""
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=sol["utility"],
                name=aid,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                customdata=cd,
                hovertemplate="%{y:.4g}" + xref + "<extra>" + aid + "</extra>",
            ),
            secondary_y=False,
        )

    # Total objective overlay
    obj = computer.get_total_objective()
    if not obj.empty:
        if iter_range and len(iter_range) == 2:
            obj = obj[
                (obj["iteration"] >= iter_range[0])
                & (obj["iteration"] <= iter_range[1])
            ]
        if not obj.empty:
            obj_x = obj["iteration"]
            if use_time:
                obj = obj.merge(
                    conv_df[["iteration", "timestamp"]], on="iteration", how="inner"
                )
                obj_x = ts_to_datetime(obj["timestamp"])
            fig.add_trace(
                go.Scatter(
                    x=obj_x,
                    y=obj["total_objective"],
                    name="Total Obj.",
                    line=dict(color=COLORS[4], width=2, dash="dot"),
                    hovertemplate="%{y:.4g}<extra>Total Obj.</extra>",
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title="Total Objective", secondary_y=True)

    x_title = "Time" if x_axis == "timestamp" else "Iteration"
    fig.update_xaxes(title=x_title)
    if x_axis == "timestamp":
        fig.update_xaxes(tickformat="%H:%M:%S")
    fig.update_yaxes(title="Utility", secondary_y=False)
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def residuals_figure(
    provider: DashboardDataProvider,
    computer: DashboardMetricsComputer,
    agents: list,
    x_axis: str = "iteration",
    iter_range: list | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if not agents:
        fig.add_annotation(
            text="Select agents above",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    conv_df = provider.get_convergence_data()
    first_trace_added = False

    for i, aid in enumerate(agents):
        res = computer.get_agent_residuals(aid)
        if res.empty:
            continue
        if iter_range and len(iter_range) == 2:
            res = res[
                (res["iteration"] >= iter_range[0])
                & (res["iteration"] <= iter_range[1])
            ]
        if res.empty:
            continue
        x_vals = res["iteration"]
        use_time = (
            x_axis == "timestamp"
            and not conv_df.empty
            and "timestamp" in conv_df.columns
            and conv_df["timestamp"].notna().any()
        )
        if use_time:
            res = res.merge(
                conv_df[["iteration", "timestamp"]], on="iteration", how="inner"
            )
            x_vals = ts_to_datetime(res["timestamp"])
        if not first_trace_added:
            cd, xref = crossref_hover(res, use_time)
            first_trace_added = True
        else:
            cd, xref = None, ""
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=res["residual"],
                name=aid,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                customdata=cd,
                hovertemplate="%{y:.4g}" + xref + "<extra>" + aid + "</extra>",
            ),
            secondary_y=False,
        )

    # Total objective overlay
    obj = computer.get_total_objective()
    if not obj.empty:
        if iter_range and len(iter_range) == 2:
            obj = obj[
                (obj["iteration"] >= iter_range[0])
                & (obj["iteration"] <= iter_range[1])
            ]
        if not obj.empty:
            obj_x = obj["iteration"]
            use_time = (
                x_axis == "timestamp"
                and not conv_df.empty
                and "timestamp" in conv_df.columns
                and conv_df["timestamp"].notna().any()
            )
            if use_time:
                obj = obj.merge(
                    conv_df[["iteration", "timestamp"]], on="iteration", how="inner"
                )
                obj_x = ts_to_datetime(obj["timestamp"])
            fig.add_trace(
                go.Scatter(
                    x=obj_x,
                    y=obj["total_objective"],
                    name="Total Obj.",
                    line=dict(color=COLORS[4], width=2, dash="dot"),
                    hovertemplate="%{y:.4g}<extra>Total Obj.</extra>",
                ),
                secondary_y=True,
            )
            fig.update_yaxes(title="Total Objective", secondary_y=True)

    fig.update_yaxes(
        type="log", title="\u2016x\u1d62 \u2212 z\u2016 (log)", secondary_y=False
    )
    x_title = "Time" if x_axis == "timestamp" else "Iteration"
    fig.update_xaxes(title=x_title)
    if x_axis == "timestamp":
        fig.update_xaxes(tickformat="%H:%M:%S")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def variable_trajectories_figure(
    provider: DashboardDataProvider,
    selected_vars: list,
    x_axis: str = "iteration",
    iter_range: list | None = None,
) -> go.Figure:
    fig = go.Figure()
    if not selected_vars:
        fig.add_annotation(
            text="Select variables above",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    all_cv = provider.get_all_consensus_vars()
    if not all_cv:
        fig.add_annotation(
            text="No consensus data yet",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    metadata = provider.get_problem_metadata()
    var_groups = metadata.get("variable_groups", []) if metadata else []
    label_to_idx = {}
    for vg in sorted(var_groups, key=lambda v: v["slice_start"]):
        for j in range(vg["count"]):
            label = f"{vg['name']}[{j}]"
            label_to_idx[label] = vg["slice_start"] + j

    iterations = sorted(all_cv.keys())
    if iter_range and len(iter_range) == 2:
        iterations = [it for it in iterations if iter_range[0] <= it <= iter_range[1]]

    conv_df = provider.get_convergence_data()
    use_time = (
        x_axis == "timestamp"
        and not conv_df.empty
        and "timestamp" in conv_df.columns
        and conv_df["timestamp"].notna().any()
    )
    iter_to_ts = {}
    if use_time:
        for _, row in conv_df.iterrows():
            iter_to_ts[int(row["iteration"])] = row["timestamp"]

    # Build cross-ref data for the first trace
    if use_time:
        _crossref_cd = [[it] for it in iterations]
        _crossref_suffix = "<br><i>Iter: %{customdata[0]}</i>"
    elif iter_to_ts:
        ts_series = ts_to_datetime(
            pd.Series([iter_to_ts.get(it, float("nan")) for it in iterations])
        )
        _crossref_cd = [
            [t.strftime("%H:%M:%S") if pd.notna(t) else ""] for t in ts_series
        ]
        _crossref_suffix = "<br><i>Time: %{customdata[0]}</i>"
    else:
        _crossref_cd = None
        _crossref_suffix = ""

    first_trace_added = False
    for i, var_label in enumerate(selected_vars):
        idx = label_to_idx.get(var_label)
        if idx is None:
            continue
        values = [float(all_cv[it][idx]) for it in iterations]
        x_vals: Any
        if use_time:
            x_vals = ts_to_datetime(
                pd.Series([iter_to_ts.get(it, float("nan")) for it in iterations])
            )
        else:
            x_vals = list(iterations)
        if not first_trace_added:
            cd, xref = _crossref_cd, _crossref_suffix
            first_trace_added = True
        else:
            cd, xref = None, ""
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=values,
                name=var_label,
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                customdata=cd,
                hovertemplate="%{y:.4g}" + xref + "<extra>" + var_label + "</extra>",
            )
        )

    x_title = "Time" if use_time else "Iteration"
    fig.update_xaxes(title=x_title)
    if use_time:
        fig.update_xaxes(tickformat="%H:%M:%S")
    fig.update_yaxes(title="Value")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def objective_decomposition_figure(
    computer: DashboardMetricsComputer,
    agent_id: str | None,
) -> go.Figure:
    fig = go.Figure()
    if not agent_id:
        fig.add_annotation(
            text="Select an agent", showarrow=False, font=dict(size=14, color="#999")
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    df = computer.get_agent_objective_decomposition(agent_id)
    if df.empty:
        fig.add_annotation(
            text="No solution data", showarrow=False, font=dict(size=14, color="#999")
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    for i, col in enumerate(["utility", "subsidy", "proximal"]):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["iteration"],
                    y=df[col],
                    name=col.capitalize(),
                    line=dict(color=COLORS[i], width=2),
                )
            )
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="Value")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def agent_residual_figure(
    computer: DashboardMetricsComputer,
    agent_id: str | None,
) -> go.Figure:
    fig = go.Figure()
    if not agent_id:
        fig.add_annotation(
            text="Select an agent", showarrow=False, font=dict(size=14, color="#999")
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    res = computer.get_agent_residuals(agent_id)
    if res.empty:
        fig.add_annotation(
            text="No residual data", showarrow=False, font=dict(size=14, color="#999")
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    fig.add_trace(
        go.Scatter(
            x=res["iteration"],
            y=res["residual"],
            name=agent_id,
            line=dict(color=COLORS[0], width=2),
        )
    )
    fig.update_yaxes(type="log", title="\u2016x\u1d62 \u2212 z\u2016 (log)")
    fig.update_xaxes(title="Iteration")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def pref_vs_consensus_figure(
    computer: DashboardMetricsComputer,
    agent_id: str | None,
    var_labels: list,
) -> go.Figure:
    fig = go.Figure()
    if not agent_id or not var_labels:
        msg = "Select an agent" if not agent_id else "Select variables"
        fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#999"))
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    trajectories = computer.get_agent_preferred_trajectories(agent_id, var_labels)
    if not trajectories:
        fig.add_annotation(
            text="No data for selected variables",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    for i, (label, df) in enumerate(sorted(trajectories.items())):
        color = COLORS[i % len(COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["preferred"],
                name=f"{label} (pref)",
                line=dict(color=color, width=2.5),
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["consensus"],
                name=f"{label} (cons)",
                line=dict(color=color, width=1.5, dash="dash"),
                opacity=0.5,
                legendgroup=label,
            )
        )

    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="Value")
    fig.update_layout(**PLOT_LAYOUT)
    return fig


def convergence_comparison_figure(
    providers: list[tuple[str, DashboardDataProvider]],
) -> go.Figure:
    """Overlay primal residual curves from multiple runs.

    Args:
        providers: List of (label, provider) tuples.

    Returns:
        A plotly Figure with one trace per run.
    """
    fig = go.Figure()
    if not providers:
        fig.add_annotation(
            text="Select runs to compare",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )
        fig.update_layout(**PLOT_LAYOUT)
        return fig

    for i, (label, provider) in enumerate(providers):
        df = provider.get_convergence_data()
        if df.empty:
            continue
        color = COLORS[i % len(COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df["iteration"],
                y=df["primal_residual"],
                name=label,
                line=dict(color=color, width=2),
                hovertemplate="%{y:.4g}<extra>" + label + "</extra>",
            )
        )

    if not fig.data:
        fig.add_annotation(
            text="No convergence data",
            showarrow=False,
            font=dict(size=14, color="#999"),
        )

    fig.update_yaxes(type="log", title="Primal Residual (log)")
    fig.update_xaxes(title="Iteration")
    fig.update_layout(**PLOT_LAYOUT)
    return fig
