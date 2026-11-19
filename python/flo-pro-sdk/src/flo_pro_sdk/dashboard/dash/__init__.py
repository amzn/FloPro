"""Dash/Plotly UI components for the coordination dashboard."""

try:
    import dash  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Dashboard dependencies are required. "
        "Install them with: pip install dash dash-bootstrap-components plotly"
    ) from e
