"""Dash/Plotly dashboard for monitoring coordination runs.

Uses dash-bootstrap-components for layout. Thin orchestrator that
delegates layout building to ``layout_*`` modules and figure
building to ``figures``.

Two-page structure:
- Overview page (``/``) when multiple runs are discovered
- Detail page (``/run/<coord_id>/<run_id>``) for single-run deep dive

When only one run exists, the overview is skipped and the detail page
is shown at ``/``.

Usage::

    from flo_pro_sdk.dashboard.dash.app import DashboardDashApp
    from flo_pro_sdk.dashboard.run_browser import RunBrowser

    browser = RunBrowser("/path/to/base_dir")
    app = DashboardDashApp(browser)
    app.run(port=8050)
"""

from __future__ import annotations

import logging

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import dash_bootstrap_components as dbc

    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False

from flo_pro_sdk.dashboard.run_browser import RunBrowser

logger = logging.getLogger(__name__)

_MISSING_MSG = (
    "Dash dashboard requires 'dash', 'plotly', and 'dash-bootstrap-components'. "
    "Install with: pip install dash plotly dash-bootstrap-components"
)


# Slider CSS override for Flatly theme
_SLIDER_CSS = """<style>
.dash-slider-range { background-color: #2c3e50 !important; }
.dash-slider-thumb { background-color: #2c3e50 !important; border-color: #2c3e50 !important; }
.dash-slider-thumb:hover, .dash-slider-thumb:focus { box-shadow: 0 0 0 3px rgba(44,62,80,0.25) !important; }
.dash-slider-track { background-color: #dce1e5 !important; }
.dash-slider-dot { border-color: #2c3e50 !important; }
</style></head>"""


class DashboardDashApp:
    """Dash/Plotly dashboard with overview + detail pages.

    Args:
        browser: RunBrowser instance with discovered runs.
        refresh_interval: Auto-refresh interval in milliseconds. 0 disables.
    """

    def __init__(
        self,
        browser: RunBrowser,
        refresh_interval: int = 5000,
    ) -> None:
        if not _HAS_DASH:
            raise ImportError(_MISSING_MSG)

        self._browser = browser
        self._refresh_interval = refresh_interval
        self._app = self._build_app()

    @property
    def browser(self) -> RunBrowser:
        return self._browser

    @property
    def server(self):
        return self._app.server

    @property
    def app(self) -> "dash.Dash":
        return self._app

    # ── Layout ──────────────────────────────────────────────────────────

    def _build_app(self) -> "dash.Dash":
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.FLATLY],
            suppress_callback_exceptions=True,
            title="Coordination Dashboard",
        )
        app.index_string = app.index_string.replace("</head>", _SLIDER_CSS)

        navbar = dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand(
                        "Coordination Dashboard",
                        href="/",
                        className="me-auto",
                        style={"cursor": "pointer"},
                    ),
                    html.Div(id="navbar-breadcrumb"),
                ],
                fluid=True,
                className="d-flex align-items-center",
            ),
            color="primary",
            dark=True,
            className="mb-3",
        )

        app.layout = dbc.Container(
            [
                navbar,
                dcc.Location(id="url", refresh=False),
                html.Div(id="page-content"),
                # Auto-refresh (used by detail page callbacks)
                dcc.Interval(
                    id="auto-refresh",
                    interval=self._refresh_interval
                    if self._refresh_interval > 0
                    else 60_000_000,
                    disabled=self._refresh_interval <= 0,
                ),
            ],
            fluid=True,
        )

        self._register_routing(app)
        self._register_detail_callbacks(app)
        self._register_overview_callbacks(app)
        return app

    # ── Routing ─────────────────────────────────────────────────────────

    def _register_routing(self, app: "dash.Dash") -> None:
        browser = self._browser

        @app.callback(
            Output("page-content", "children"),
            Input("url", "pathname"),
        )
        def route_page(pathname):
            pathname = pathname or "/"

            # /run/<coord_id>/<run_id> → detail page
            if pathname.startswith("/run/"):
                parts = pathname.strip("/").split("/")
                if len(parts) == 3:
                    _, coord_id, run_id = parts
                    run = browser.find_run(coord_id, run_id)
                    if run is not None:
                        return self._build_detail_page(run)

            # Single run → go straight to detail
            if browser.is_single_run:
                run = browser.all_runs[0]
                return self._build_detail_page(run)

            # Default → overview
            return self._build_overview_page()

        @app.callback(
            Output("navbar-breadcrumb", "children"),
            Input("url", "pathname"),
        )
        def update_breadcrumb(pathname):
            pathname = pathname or "/"
            if pathname.startswith("/run/"):
                parts = pathname.strip("/").split("/")
                if len(parts) == 3:
                    _, coord_id, run_id = parts
                    # Look up status from RunInfo
                    run = browser.find_run(coord_id, run_id)
                    status = run.status if run else "unknown"
                    badge_color = {"completed": "success", "running": "warning"}.get(
                        status, "secondary"
                    )
                    return html.Span(
                        [
                            html.Span(
                                " › ", style={"opacity": "0.5", "margin": "0 6px"}
                            ),
                            html.Span(coord_id[:20], style={"opacity": "0.7"}),
                            html.Span(
                                " / ", style={"opacity": "0.5", "margin": "0 4px"}
                            ),
                            html.Span(run_id[:20], style={"opacity": "0.7"}),
                            dbc.Badge(
                                status,
                                color=badge_color,
                                pill=True,
                                className="ms-2",
                                style={"fontSize": "0.7em", "verticalAlign": "middle"},
                            ),
                        ],
                        style={"color": "rgba(255,255,255,0.85)", "fontSize": "0.85em"},
                    )
            return None

    def _build_overview_page(self) -> dbc.Container:
        from flo_pro_sdk.dashboard.dash.layout_overview import (
            build_overview_layout,
        )

        return build_overview_layout(self._browser)

    def _build_detail_page(self, run) -> dbc.Container:
        """Build the detail page for a single run."""
        back_link = []
        if not self._browser.is_single_run:
            back_link = [
                dbc.Button(
                    "← All Runs",
                    href="/",
                    color="link",
                    className="p-0 me-3",
                    style={
                        "textDecoration": "none",
                        "fontSize": "0.9em",
                        "color": "#7f8c8d",
                    },
                ),
            ]

        # Store the active run's coord_id and run_id for callbacks
        return dbc.Container(
            [
                dcc.Store(id="active-coord-id", data=run.coordination_id),
                dcc.Store(id="active-run-id", data=run.run_id),
                html.Div(back_link, className="mb-2") if back_link else None,
                dbc.Tabs(
                    [
                        dbc.Tab(label="Problem Details", tab_id="problem-details"),
                        dbc.Tab(label="Convergence", tab_id="convergence"),
                        dbc.Tab(label="Agents", tab_id="agent-deep-dive"),
                    ],
                    id="main-tabs",
                    active_tab="problem-details",
                    className="mb-3",
                ),
                html.Div(id="tab-content"),
            ],
            fluid=True,
        )

    # ── Detail page callbacks ───────────────────────────────────────────

    def _get_active_provider_and_computer(self, coord_id, run_id):
        """Resolve provider and computer from stored run identifiers."""
        run = self._browser.find_run(coord_id, run_id)
        if run is None:
            # Fallback to first run
            run = self._browser.all_runs[0]
        provider = self._browser.get_provider(run.run_dir)
        computer = self._browser.get_computer(run.run_dir)
        return provider, computer

    def _register_detail_callbacks(self, app: "dash.Dash") -> None:
        from flo_pro_sdk.dashboard.dash import figures
        from flo_pro_sdk.dashboard.dash.layout_details import (
            build_problem_details_layout,
            build_sub_map_page,
            SUB_MAP_PAGE_SIZE,
        )
        from flo_pro_sdk.dashboard.dash.layout_convergence import (
            build_convergence_layout,
            get_group_max_index,
            build_filtered_var_options,
        )
        from flo_pro_sdk.dashboard.dash.layout_agents import (
            build_agents_layout,
            build_agent_info,
        )

        get_pc = self._get_active_provider_and_computer

        # Tab persistence via sessionStorage
        app.clientside_callback(
            """
            function(active_tab) {
                if (active_tab) {
                    sessionStorage.setItem('dash_active_tab', active_tab);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output("url", "hash"),
            Input("main-tabs", "active_tab"),
        )

        app.clientside_callback(
            """
            function(pathname) {
                var storedPath = sessionStorage.getItem('dash_detail_path') || '';
                // Same detail page (e.g. tab switch then refresh) → restore tab
                if (pathname === storedPath) {
                    var stored = sessionStorage.getItem('dash_active_tab');
                    if (stored === 'convergence' || stored === 'problem-details' || stored === 'agent-deep-dive') {
                        return stored;
                    }
                }
                // New detail page → reset to Problem Details
                sessionStorage.setItem('dash_detail_path', pathname);
                sessionStorage.removeItem('dash_active_tab');
                return 'problem-details';
            }
            """,
            Output("main-tabs", "active_tab"),
            Input("url", "pathname"),
        )

        @app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "active_tab"),
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def render_tab(active_tab, coord_id, run_id):
            provider, computer = get_pc(coord_id, run_id)
            if active_tab == "convergence":
                return build_convergence_layout(provider)
            elif active_tab == "agent-deep-dive":
                return build_agents_layout(provider)
            return build_problem_details_layout(provider)

        # Disable auto-refresh for completed runs
        @app.callback(
            Output("auto-refresh", "disabled"),
            [Input("active-coord-id", "data"), Input("active-run-id", "data")],
        )
        def toggle_auto_refresh(coord_id, run_id):
            if not coord_id or not run_id:
                return True
            run = self._browser.find_run(coord_id, run_id)
            if run is None or run.status == "completed":
                return True
            return self._refresh_interval <= 0

        # ── Convergence tab callbacks ───────────────────────────────────

        @app.callback(
            [
                Output("x-axis-toggle", "data"),
                Output("x-btn-iter", "outline"),
                Output("x-btn-time", "outline"),
            ],
            [Input("x-btn-iter", "n_clicks"), Input("x-btn-time", "n_clicks")],
        )
        def toggle_x_axis(n_iter, n_time):
            ctx = dash.callback_context
            if not ctx.triggered:
                return "iteration", False, True
            btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if btn_id == "x-btn-time":
                return "timestamp", True, False
            return "iteration", False, True

        @app.callback(
            Output("convergence-plot", "figure"),
            [
                Input("x-axis-toggle", "data"),
                Input("iter-range-slider", "value"),
                Input("conv-rate-toggle", "value"),
                Input("refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_convergence(
            x_axis, iter_range, rate_toggle, _clicks, _intervals, coord_id, run_id
        ):
            provider, computer = get_pc(coord_id, run_id)
            show_rate = "show" in (rate_toggle or [])
            return figures.convergence_figure(
                provider,
                computer,
                x_axis,
                iter_range=iter_range,
                show_rate=show_rate,
            )

        @app.callback(
            Output("objectives-plot", "figure"),
            [
                Input("agent-dropdown", "value"),
                Input("x-axis-toggle", "data"),
                Input("iter-range-slider", "value"),
                Input("refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_objectives(
            agents, x_axis, iter_range, _clicks, _intervals, coord_id, run_id
        ):
            provider, computer = get_pc(coord_id, run_id)
            return figures.objectives_figure(
                provider,
                computer,
                agents or [],
                x_axis,
                iter_range=iter_range,
            )

        @app.callback(
            Output("residuals-plot", "figure"),
            [
                Input("agent-dropdown", "value"),
                Input("x-axis-toggle", "data"),
                Input("iter-range-slider", "value"),
                Input("refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_residuals(
            agents, x_axis, iter_range, _clicks, _intervals, coord_id, run_id
        ):
            provider, computer = get_pc(coord_id, run_id)
            return figures.residuals_figure(
                provider,
                computer,
                agents or [],
                x_axis,
                iter_range=iter_range,
            )

        @app.callback(
            Output("variable-trajectories-plot", "figure"),
            [
                Input("variable-dropdown", "value"),
                Input("x-axis-toggle", "data"),
                Input("iter-range-slider", "value"),
                Input("refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_variable_trajectories(
            variables, x_axis, iter_range, _clicks, _intervals, coord_id, run_id
        ):
            provider, _ = get_pc(coord_id, run_id)
            return figures.variable_trajectories_figure(
                provider,
                variables or [],
                x_axis,
                iter_range=iter_range,
            )

        # Agents table column selector
        @app.callback(
            Output("agents-table", "columns"),
            [Input("agents-col-selector", "value")],
            [
                State("agents-all-columns", "data"),
                State("agents-fixed-columns", "data"),
            ],
        )
        def update_agents_columns(selected_meta, all_cols, fixed_cols):
            if all_cols is None:
                return dash.no_update
            selected_meta = selected_meta or []
            visible = set(fixed_cols or []) | set(selected_meta)
            return [c for c in all_cols if c["id"] in visible]

        # ── Variable Trajectories filter callbacks ──────────────────────

        @app.callback(
            [
                Output("vt-idx-min", "max"),
                Output("vt-idx-max", "max"),
                Output("vt-idx-min", "value"),
                Output("vt-idx-max", "value"),
            ],
            [Input("vt-group-filter", "value")],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_vt_idx_range(group, coord_id, run_id):
            if not group:
                return (dash.no_update,) * 4
            provider, _ = get_pc(coord_id, run_id)
            mx = get_group_max_index(provider, group)
            return mx, mx, 0, mx

        @app.callback(
            [
                Output("variable-dropdown", "options"),
                Output("variable-dropdown", "value"),
            ],
            [
                Input("vt-group-filter", "value"),
                Input("vt-idx-min", "value"),
                Input("vt-idx-max", "value"),
                Input("vt-meta-filter", "value"),
            ],
            [
                State("variable-dropdown", "value"),
                State("active-coord-id", "data"),
                State("active-run-id", "data"),
            ],
        )
        def update_vt_variable_options(
            group, idx_min, idx_max, meta_filter, current_selection, coord_id, run_id
        ):
            provider, _ = get_pc(coord_id, run_id)
            opts = build_filtered_var_options(
                provider,
                group_name=group,
                idx_min=idx_min,
                idx_max=idx_max,
                meta_filter=meta_filter or "",
            )
            valid_values = {o["value"] for o in opts}
            if current_selection:
                kept = [v for v in current_selection if v in valid_values]
                if kept:
                    return opts, kept
            default = [o["value"] for o in opts[:5]]
            return opts, default

        # ── Subscription map callbacks ──────────────────────────────────

        @app.callback(
            Output("sub-map-table", "page_current"),
            [
                Input("sub-map-group-filter", "value"),
                Input("sub-map-table", "filter_query"),
                Input("sub-map-idx-min", "value"),
                Input("sub-map-idx-max", "value"),
            ],
        )
        def reset_sub_map_page(_group, _fq, _lo, _hi):
            return 0

        @app.callback(
            [
                Output("sub-map-idx-min", "max"),
                Output("sub-map-idx-max", "max"),
                Output("sub-map-idx-min", "value"),
                Output("sub-map-idx-max", "value"),
            ],
            [Input("sub-map-group-filter", "value")],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_idx_range_bounds(group, coord_id, run_id):
            if not group:
                return (dash.no_update,) * 4
            provider, _ = get_pc(coord_id, run_id)
            mx = get_group_max_index(provider, group)
            return mx, mx, 0, mx

        @app.callback(
            [
                Output("sub-map-table", "data"),
                Output("sub-map-table", "columns"),
                Output("sub-map-table", "page_count"),
                Output("sub-map-total", "children"),
            ],
            [
                Input("sub-map-group-filter", "value"),
                Input("sub-map-table", "page_current"),
                Input("sub-map-table", "filter_query"),
                Input("sub-map-idx-min", "value"),
                Input("sub-map-idx-max", "value"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_sub_map(
            group, page, filter_query, idx_min, idx_max, coord_id, run_id
        ):
            if not group:
                return (dash.no_update,) * 4
            provider, _ = get_pc(coord_id, run_id)
            metadata = provider.get_problem_metadata()
            if not metadata:
                return (dash.no_update,) * 4
            rows, columns, total = build_sub_map_page(
                group,
                metadata,
                page=page or 0,
                filter_query=filter_query or "",
                idx_min=idx_min,
                idx_max=idx_max,
            )
            ps = SUB_MAP_PAGE_SIZE
            page_count = max(1, -(-total // ps))
            return rows, columns, page_count, f"{total:,} variables"

        # ── Agents tab callbacks ────────────────────────────────────────

        @app.callback(
            [
                Output("dd-idx-min", "max"),
                Output("dd-idx-max", "max"),
                Output("dd-idx-min", "value"),
                Output("dd-idx-max", "value"),
            ],
            [Input("dd-group-filter", "value")],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_dd_idx_range(group, coord_id, run_id):
            if not group:
                return (dash.no_update,) * 4
            provider, _ = get_pc(coord_id, run_id)
            mx = get_group_max_index(provider, group)
            return mx, mx, 0, mx

        @app.callback(
            [
                Output("dd-variable-dropdown", "options"),
                Output("dd-variable-dropdown", "value"),
            ],
            [
                Input("dd-agent-selector", "value"),
                Input("dd-group-filter", "value"),
                Input("dd-idx-min", "value"),
                Input("dd-idx-max", "value"),
                Input("dd-meta-filter", "value"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_dd_variable_options(
            agent_id, group, idx_min, idx_max, meta_filter, coord_id, run_id
        ):
            if not agent_id:
                return [], None
            provider, _ = get_pc(coord_id, run_id)
            opts = build_filtered_var_options(
                provider,
                group_name=group,
                idx_min=idx_min,
                idx_max=idx_max,
                meta_filter=meta_filter or "",
                agent_id=agent_id,
            )
            default = opts[0]["value"] if opts else None
            return opts, default

        @app.callback(
            Output("dd-agent-info", "children"),
            [Input("dd-agent-selector", "value")],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_dd_agent_info(agent_id, coord_id, run_id):
            provider, _ = get_pc(coord_id, run_id)
            return build_agent_info(provider, agent_id)

        @app.callback(
            Output("dd-objective-decomp-plot", "figure"),
            [
                Input("dd-agent-selector", "value"),
                Input("dd-refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_dd_objective_decomp(agent_id, _clicks, _intervals, coord_id, run_id):
            _, computer = get_pc(coord_id, run_id)
            return figures.objective_decomposition_figure(computer, agent_id)

        @app.callback(
            Output("dd-residual-plot", "figure"),
            [
                Input("dd-agent-selector", "value"),
                Input("dd-refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_dd_residual(agent_id, _clicks, _intervals, coord_id, run_id):
            _, computer = get_pc(coord_id, run_id)
            return figures.agent_residual_figure(computer, agent_id)

        @app.callback(
            Output("dd-pref-vs-consensus-plot", "figure"),
            [
                Input("dd-agent-selector", "value"),
                Input("dd-variable-dropdown", "value"),
                Input("dd-refresh-btn", "n_clicks"),
                Input("auto-refresh", "n_intervals"),
            ],
            [State("active-coord-id", "data"), State("active-run-id", "data")],
        )
        def update_dd_pref_vs_consensus(
            agent_id, variable, _clicks, _intervals, coord_id, run_id
        ):
            _, computer = get_pc(coord_id, run_id)
            variables = [variable] if variable else []
            return figures.pref_vs_consensus_figure(
                computer,
                agent_id,
                variables,
            )

    # ── Overview page callbacks ─────────────────────────────────────────

    def _register_overview_callbacks(self, app: "dash.Dash") -> None:
        from flo_pro_sdk.dashboard.dash import figures

        browser = self._browser

        # Navigation from overview markdown links is handled by dcc.Location
        # (the markdown links point to /run/<coord_id>/<run_id>)

        @app.callback(
            Output("overview-convergence-comparison", "figure"),
            Input("overview-compare-runs", "value"),
        )
        def update_comparison(selected_runs):
            if not selected_runs:
                return figures.convergence_comparison_figure([])
            providers = []
            for key in selected_runs:
                parts = key.split("|", 1)
                if len(parts) != 2:
                    continue
                coord_id, run_id = parts
                run = browser.find_run(coord_id, run_id)
                if run is not None:
                    label = f"{run_id[:12]}"
                    providers.append((label, browser.get_provider(run.run_dir)))
            return figures.convergence_comparison_figure(providers)

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False, **kwargs
    ) -> None:
        self._app.run(host=host, port=port, debug=debug, **kwargs)
