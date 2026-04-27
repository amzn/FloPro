"""Shared constants and helpers for the Dash dashboard modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tableau 10 accessible palette
COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]

PLOT_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=10, b=40),
    font=dict(size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=320,
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(189,195,199,0.3)",
        gridwidth=1,
        griddash="dot",
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor="#7f8c8d",
        spikedash="dot",
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(189,195,199,0.3)",
        gridwidth=1,
        griddash="dot",
    ),
    hoverlabel=dict(font_size=12, namelength=-1),
    hovermode="x unified",
)

CARD_HEADER_STYLE = {"fontSize": "1.1em", "fontWeight": "600", "color": "white"}
CARD_HEADER_BG = {"backgroundColor": "#5a6c7d", "borderBottom": "none"}

TABLE_STYLE_HEADER = {
    "fontWeight": "600",
    "backgroundColor": "#ecf0f1",
    "color": "#2c3e50",
    "borderBottom": "2px solid #bdc3c7",
    "padding": "10px 8px",
    "fontSize": "0.85em",
}
TABLE_STYLE_CELL = {
    "textAlign": "left",
    "padding": "8px",
    "fontSize": "0.9em",
    "borderBottom": "1px solid #ecf0f1",
}
TABLE_STYLE_DATA_COND = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"},
]


def ts_to_datetime(series: pd.Series) -> pd.Series:
    """Convert unix epoch floats to timezone-aware datetime for display."""
    return pd.to_datetime(series, unit="s", utc=True).dt.tz_convert("US/Pacific")


def get_x_data(df: pd.DataFrame, x_axis: str) -> tuple:
    """Return (x_values, x_title, use_time) for a DataFrame with iteration/timestamp."""
    use_time = (
        x_axis == "timestamp"
        and "timestamp" in df.columns
        and df["timestamp"].notna().any()
    )
    if use_time:
        return ts_to_datetime(df["timestamp"]), "Time", True
    return df["iteration"], "Iteration", False


def crossref_hover(df: pd.DataFrame, use_time: bool) -> tuple:
    """Return (customdata, hover_suffix) for cross-referencing the alternate axis.

    The suffix is appended to the *first* trace's hovertemplate so it
    appears once in the unified hover tooltip beneath that trace's value.
    Subsequent traces use a plain hovertemplate (no suffix).

    Returns ``(None, "")`` when no cross-ref data is available.
    """
    if use_time:
        cd = np.asarray(df["iteration"].values).reshape(-1, 1)
        suffix = "<br><i>Iter: %{customdata[0]}</i>"
    elif "timestamp" in df.columns and df["timestamp"].notna().any():
        ts = ts_to_datetime(df["timestamp"])
        cd = np.asarray(ts.dt.strftime("%H:%M:%S").values).reshape(-1, 1)
        suffix = "<br><i>Time: %{customdata[0]}</i>"
    else:
        return None, ""
    return cd, suffix


def parse_filter_query(query: str) -> list[tuple[str, str, str]]:
    """Parse Dash DataTable filter_query into (col, op, value) triples."""
    import re

    if not query or not query.strip():
        return []
    filters = []
    for part in query.split(" && "):
        part = part.strip()
        m = re.match(
            r'\{([^}]+)\}\s*(contains|eq|=|!=|>=|<=|>|<)\s*"?([^"]*)"?',
            part,
        )
        if m:
            filters.append((m.group(1), m.group(2), m.group(3)))
    return filters


def row_matches_filters(
    row: dict,
    filters: list[tuple[str, str, str]],
) -> bool:
    """Check if a row matches all parsed filter expressions."""
    for col, op, val in filters:
        cell = str(row.get(col, ""))
        if op == "contains":
            if val.lower() not in cell.lower():
                return False
        elif op in ("eq", "="):
            if cell != val:
                return False
        elif op == "!=":
            if cell == val:
                return False
        elif op in (">", ">=", "<", "<="):
            try:
                cv, fv = float(cell), float(val)
            except (ValueError, TypeError):
                return False
            if op == ">" and not (cv > fv):
                return False
            if op == ">=" and not (cv >= fv):
                return False
            if op == "<" and not (cv < fv):
                return False
            if op == "<=" and not (cv <= fv):
                return False
    return True
