"""Dash/Plotly UI components for the coordination dashboard."""

try:
    import dash  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Dashboard dependencies are required. "
        "Install them with: pip install flo-pro-sdk[dashboard]"
    ) from e
