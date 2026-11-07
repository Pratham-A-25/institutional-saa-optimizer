import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from portfolio_engine import (
    DataManager, RiskEstimator, BlackLittermanModel,
    InstitutionalOptimizer, Backtester,
)

# ======================================================================
# PAGE CONFIG & GLOBAL THEME
# ======================================================================
st.set_page_config(
    page_title="Institutional SAA Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* ---- Global ---- */
    .block-container { padding-top: 1.5rem; }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 14px;
        padding: 18px 22px;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(0,0,0,.15);
    }
    [data-testid="stMetricLabel"] { color: #b0c4de; font-size: .85rem; }
    [data-testid="stMetricValue"] { color: #ffffff; font-size: 1.55rem; font-weight: 700; }
    [data-testid="stMetricDelta"]  { color: #06d6a0; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2 {
        font-size: 1.05rem;
        letter-spacing: .03em;
        border-bottom: 2px solid #2c5364;
        padding-bottom: 6px;
        margin-top: 1.2rem;
    }

    /* ---- Tabs ---- */
    button[data-baseweb="tab"] { font-size: .95rem; font-weight: 600; }

    /* ---- Tables ---- */
    [data-testid="stDataFrame"] th {
        background-color: #1b2838 !important;
        color: #ffffff !important;
    }

    /* ---- Success / Info boxes ---- */
    div[data-testid="stAlert"] { border-radius: 10px; }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# ALTAIR CHART HELPERS
# ======================================================================
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#06D6A0", "#EF476F", "#FFD166"]

def _base_theme(chart, height=400):
    """Apply a clean, consistent look to any Altair chart."""
    return (
        chart
        .properties(height=height)
        .configure_axis(
            labelFontSize=12, titleFontSize=13, titleColor="#444",
            gridColor="#eaeaea", domainColor="#999",
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=15, anchor="start", color="#1b2838")
    )


# ======================================================================
# SIDEBAR
# ======================================================================

# --- Section 1: Asset Universe ---
st.sidebar.header("🌍  Asset Universe & Base Data")

# DYNAMIC BENCHMARK NAME ADDED HERE
benchmark_name = st.sidebar.text_input("Benchmark Name", value="Standard 60/40 SAA", help="Name your baseline portfolio for the charts and reports.")

st.sidebar.caption("Edit tickers, benchmark weights, and carbon scores below.")

default_data = pd.DataFrame({
    "Ticker": ["SPY", "EWA", "AGG", "IGF", "GLD"],
    "Benchmark Wt (%)": [40.0, 15.0, 25.0, 10.0, 10.0],
    "Carbon Intensity": [130.0, 210.0, 65.0, 320.0, 25.0],
})
edited_data = st.sidebar.data_editor(
    default_data, num_rows="fixed", hide_index=True, use_container_width=True,
)

universe = edited_data["Ticker"].tolist()
benchmark_allocations = dict(zip(edited_data["Ticker"], edited_data["Benchmark Wt (%)"]))
carbon_scores = dict(zip(edited_data["Ticker"], edited_data["Carbon Intensity"]))

# --- Section 2: Investment Constraints ---
st.sidebar.header("⚙️  Investment Constraints")

OPT_MODES = {
    "Minimum Variance": "min_variance",
    "Maximum Sharpe Ratio": "max_sharpe",
    "Risk Parity": "risk_parity",
}
opt_mode_display = st.sidebar.selectbox(
    "Optimization Objective",
    list(OPT_MODES.keys()),
    help="Min Variance minimises risk. Max Sharpe maximises risk-adjusted return. Risk Parity equalises risk contributions.",
)
opt_mode = OPT_MODES[opt_mode_display]

max_weight = st.sidebar.slider("Max Weight per Asset (%)", 10, 100, 45, 5) / 100.0
risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)", 0.0, 10.0, 4.5, 0.25,
    help="Used for Sharpe ratio calculation and Max Sharpe objective.",
) / 100.0

enforce_return = st.sidebar.checkbox("Enforce Minimum Return Target?", value=False)
if enforce_return and opt_mode == "min_variance":
    target_ret = st.sidebar.number_input("Target Return (%)", 1.0, 15.0, 5.0, 0.5) / 100.0
else:
    target_ret = None

carbon_reduction = st.sidebar.slider(
    f"Carbon Reduction vs {benchmark_name} (%)", 0, 50, 25, 5,
    help="How much cleaner the portfolio must be compared to the benchmark.",
) / 100.0

# --- Section 3: Market Views ---
st.sidebar.header("🔭  Market Views")

st.sidebar.markdown("**Relative View** — *Asset A outperforms Asset B*")
col1, col2 = st.sidebar.columns(2)
with col1:
    view_asset_A = st.selectbox("Outperformer", universe, index=3)
with col2:
    view_asset_B = st.selectbox("Underperformer", universe, index=2)
rel_outperformance = st.sidebar.number_input("Expected Outperformance (%)", value=2.5, step=0.5) / 100.0
conf_relative = st.sidebar.slider(
    "Relative View Confidence", 0.10, 1.00, 0.75, 0.05,
    help="1.0 = full confidence in this view; lower values blend with market equilibrium.",
)

st.sidebar.markdown("**Absolute View** — *Specific asset return*")
view_asset_C = st.sidebar.selectbox("Target Asset", universe, index=1)
abs_return = st.sidebar.number_input("Absolute Return (%)", value=7.5, step=0.5) / 100.0
conf_absolute = st.sidebar.slider("Absolute View Confidence", 0.10, 1.00, 0.60, 0.05)


# ======================================================================
# CACHED DATA FETCH
# ======================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_returns(tickers: tuple, start_date: str, end_date: str):
    dm = DataManager(tickers=list(tickers), start_date=start_date, end_date=end_date)
    return dm.fetch_log_returns()


# ======================================================================
# HERO / LANDING PAGE
# ======================================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
            border-radius:16px; padding:2rem 2.5rem; margin-bottom:1.5rem;">
    <h1 style="color:#ffffff; margin:0 0 .3rem 0; font-size:2rem;">
        📊 Strategic Asset Allocation &amp; Net-Zero Optimizer
    </h1>
    <p style="color:#b0c4de; margin:0; font-size:1.05rem;">
        Black-Litterman · Ledoit-Wolf · Convex Optimization · ESG Carbon Constraints
    </p>
</div>
""", unsafe_allow_html=True)

# Feature cards (shown before results)
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("##### 🧠 Bayesian Expected Returns")
    st.caption("Blend market equilibrium with your own views using the Black-Litterman model.")
with f2:
    st.markdown("##### 📈 Three Optimization Modes")
    st.caption("Minimum Variance · Maximum Sharpe Ratio · Risk Parity — all with ESG constraints.")
with f3:
    st.markdown("##### ⏱️ Walk-Forward Backtest")
    st.caption("Out-of-sample equity curve, drawdowns, and 7 performance metrics vs benchmark.")

st.divider()

# ======================================================================
# RUN BUTTON
# ======================================================================
run_clicked = st.button("🚀  Run Portfolio Optimization", type="primary", use_container_width=True)

if run_clicked:

    # --- Validation ---
    if view_asset_A == view_asset_B:
        st.error("⚠️ **Outperformer and Underperformer cannot be the same asset.** Please select different assets in the sidebar.")
        st.stop()

    with st.spinner("Fetching price data and solving convex optimization …"):

        # 1. Data
        returns = _fetch_returns(
            tickers=tuple(universe),
            start_date="2015-01-01",
            end_date=datetime.today().strftime("%Y-%m-%d"),
        )
        cov_matrix = RiskEstimator.compute_ledoit_wolf_cov(returns)

        # 2. Black-Litterman
        bl = BlackLittermanModel(cov_matrix=cov_matrix, market_caps=benchmark_allocations)
        idx_A, idx_B, idx_C = universe.index(view_asset_A), universe.index(view_asset_B), universe.index(view_asset_C)

        P = np.zeros((2, len(universe)))
        P[0, idx_A], P[0, idx_B], P[1, idx_C] = 1.0, -1.0, 1.0
        Q = np.array([rel_outperformance, abs_return])
        bl_returns = bl.compute_posterior_returns(P=P, Q=Q, confidences=[conf_relative, conf_absolute])

        # 3. Carbon cap
        bmk_weights = np.array([benchmark_allocations[t] for t in universe]) / 100.0
        bmk_carbon = sum(bmk_weights[i] * carbon_scores[universe[i]] for i in range(len(universe)))
        target_carbon_cap = bmk_carbon * (1.0 - carbon_reduction)

        # 4. Optimise
        optimizer = InstitutionalOptimizer(expected_returns=bl_returns, cov_matrix=cov_matrix)
        try:
            if opt_mode == "max_sharpe":
                w_opt, risk_attr, metrics = optimizer.optimize_max_sharpe(
                    risk_free_rate=risk_free_rate, max_weight=max_weight,
                    carbon_intensities=carbon_scores, max_portfolio_carbon=target_carbon_cap,
                )
            elif opt_mode == "risk_parity":
                w_opt, risk_attr, metrics = optimizer.optimize_risk_parity(
                    max_weight=max_weight, risk_free_rate=risk_free_rate,
                    carbon_intensities=carbon_scores, max_portfolio_carbon=target_carbon_cap,
                )
            else:
                w_opt, risk_attr, metrics = optimizer.optimize_min_variance(
                    target_return=target_ret, max_weight=max_weight,
                    carbon_intensities=carbon_scores, max_portfolio_carbon=target_carbon_cap,
                    risk_free_rate=risk_free_rate,
                )
        except ValueError:
            st.error(
                "🚫 **Optimization Infeasible.** The constraints are mathematically impossible to satisfy "
                "simultaneously. Try relaxing the carbon target, lowering the return floor, "
                "or increasing the max weight per asset."
            )
            st.stop()

    # ==================================================================
    # RESULTS
    # ==================================================================
    st.success(f"✅  **{opt_mode_display}** optimization converged successfully!")

    # ---- KPI cards ----
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Expected Return", f"{metrics['Expected Return']:.2%}")
    col_m2.metric("Annualized Vol", f"{metrics['Annualized Volatility']:.2%}")
    col_m3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
    col_m4.metric("Carbon Intensity", f"{metrics['Portfolio Carbon Intensity']:.1f}",
                  f"-{carbon_reduction * 100:.0f}% vs {benchmark_name}")

    st.markdown("")  # spacer

    # ---- Tabs ----
    tab_portfolio, tab_frontier, tab_backtest = st.tabs([
        "📊  Portfolio Allocation",
        "📈  Efficient Frontier",
        "⏱️  Walk-Forward Backtest",
    ])

    # ==============================================================
    # TAB 1 — Portfolio
    # ==============================================================
    with tab_portfolio:

        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown(f"#### Optimal vs {benchmark_name} Allocation")
            results_df = pd.DataFrame({
                benchmark_name: bmk_weights,
                "Optimal": w_opt.values,
                "Active": w_opt.values - bmk_weights,
                "Risk Contribution": risk_attr.values,
            }, index=universe)
            results_df.index.name = "Asset"
            st.dataframe(results_df.style.format("{:.2%}"), use_container_width=True)

        with right_col:
            st.markdown("#### Risk Contribution")
            risk_df = pd.DataFrame({"Asset": universe, "Risk (%)": risk_attr.values * 100})
            risk_bar = (
                alt.Chart(risk_df)
                .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                .encode(
                    x=alt.X("Risk (%):Q", title="Risk Contribution (%)"),
                    y=alt.Y("Asset:N", sort="-x", title=None),
                    color=alt.Color("Asset:N", scale=alt.Scale(range=PALETTE), legend=None),
                    tooltip=[alt.Tooltip("Asset:N"), alt.Tooltip("Risk (%):Q", format=".1f")],
                )
                .properties(height=250)
            )
            st.altair_chart(risk_bar, use_container_width=True)

        # Grouped bar chart — Dynamic Benchmark vs Optimal
        st.markdown("#### Weight Comparison")
        chart_df = results_df[[benchmark_name, "Optimal"]].reset_index()
        chart_melted = chart_df.melt(id_vars="Asset", var_name="Type", value_name="Weight")

        grouped_bar = (
            alt.Chart(chart_melted)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Asset:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Weight:Q", title="Allocation", axis=alt.Axis(format=".0%")),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(domain=[benchmark_name, "Optimal"], range=["#90a4ae", "#2E86AB"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                xOffset="Type:N",
                tooltip=[
                    alt.Tooltip("Asset:N"),
                    alt.Tooltip("Type:N"),
                    alt.Tooltip("Weight:Q", format=".2%"),
                ],
            )
        )
        st.altair_chart(_base_theme(grouped_bar, 340), use_container_width=True)

    # ==============================================================
    # TAB 2 — Efficient Frontier
    # ==============================================================
    with tab_frontier:
        with st.spinner("Tracing the efficient frontier …"):
            frontier_df = optimizer.compute_efficient_frontier(
                n_points=60, max_weight=max_weight, risk_free_rate=risk_free_rate,
                carbon_intensities=carbon_scores, max_portfolio_carbon=target_carbon_cap,
            )

        if frontier_df.empty:
            st.warning("Could not compute the efficient frontier with the current constraints.")
        else:
            st.markdown("#### Constrained Efficient Frontier")
            st.caption(
                "Each point is the minimum-variance portfolio for a given target return "
                "(subject to your weight and carbon constraints). "
                "The **◆ diamond** marks your selected portfolio."
            )

            frontier_scatter = (
                alt.Chart(frontier_df)
                .mark_circle(size=55, opacity=0.85)
                .encode(
                    x=alt.X("Annualized Volatility:Q", title="Annualized Volatility",
                            axis=alt.Axis(format=".1%")),
                    y=alt.Y("Expected Return:Q", title="Expected Return",
                            axis=alt.Axis(format=".1%")),
                    color=alt.Color("Sharpe Ratio:Q",
                                    scale=alt.Scale(scheme="turbo"),
                                    legend=alt.Legend(title="Sharpe", format=".1f")),
                    tooltip=[
                        alt.Tooltip("Expected Return:Q", format=".2%", title="Return"),
                        alt.Tooltip("Annualized Volatility:Q", format=".2%", title="Volatility"),
                        alt.Tooltip("Sharpe Ratio:Q", format=".2f", title="Sharpe"),
                    ],
                )
            )

            opt_pt_df = pd.DataFrame([{
                "Annualized Volatility": metrics["Annualized Volatility"],
                "Expected Return": metrics["Expected Return"],
                "Label": opt_mode_display,
            }])
            opt_pt = (
                alt.Chart(opt_pt_df)
                .mark_point(size=320, shape="diamond", filled=True, color="#EF476F", strokeWidth=2)
                .encode(
                    x="Annualized Volatility:Q",
                    y="Expected Return:Q",
                    tooltip=[
                        alt.Tooltip("Label:N", title="Portfolio"),
                        alt.Tooltip("Expected Return:Q", format=".2%"),
                        alt.Tooltip("Annualized Volatility:Q", format=".2%"),
                    ],
                )
            )

            combined = (frontier_scatter + opt_pt).properties(
                title="Risk-Return Trade-Off", height=480,
            )
            st.altair_chart(_base_theme(combined, 480), use_container_width=True)

    # ==============================================================
    # TAB 3 — Walk-Forward Backtest
    # ==============================================================
    with tab_backtest:
        st.markdown("#### Out-of-Sample Walk-Forward Backtest")
        st.info(
            "📋 **Methodology:** Monthly rebalancing · 3-year rolling lookback · "
            "Equilibrium returns only (no views) to avoid look-ahead bias · "
            "Full Ledoit-Wolf → BL equilibrium → optimizer pipeline re-run at each rebalance."
        )

        with st.spinner("Running walk-forward backtest …"):
            try:
                equity_curve = Backtester.run_walkforward(
                    returns=returns, benchmark_weights=benchmark_allocations,
                    optimization_mode=opt_mode, risk_free_rate=risk_free_rate,
                    max_weight=max_weight, lookback_days=756,
                    carbon_intensities=carbon_scores, max_portfolio_carbon=target_carbon_cap,
                )
                bt_metrics = Backtester.compute_metrics(equity_curve, risk_free_rate)
            except ValueError as e:
                st.warning(f"Backtest could not run: {e}")
                equity_curve = None

        if equity_curve is not None and len(equity_curve) > 1:

            # ---- Equity Curve (Altair) ----
            st.markdown("##### Growth of \\$1")
            
            # Using dynamic benchmark name for the backtest DataFrame
            eq_curve_named = pd.DataFrame({
                "Optimized Portfolio": equity_curve["Optimized Portfolio"],
                benchmark_name: equity_curve["Benchmark"]
            })
            
            eq_long = eq_curve_named.reset_index().melt(
                id_vars="index", var_name="Strategy", value_name="Value"
            ).rename(columns={"index": "Date"})

            eq_chart = (
                alt.Chart(eq_long)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("Date:T", title=None),
                    y=alt.Y("Value:Q", title="Portfolio Value ($)", axis=alt.Axis(format="$.2f")),
                    color=alt.Color(
                        "Strategy:N",
                        scale=alt.Scale(
                            domain=["Optimized Portfolio", benchmark_name],
                            range=["#2E86AB", "#90a4ae"],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    strokeDash=alt.StrokeDash(
                        "Strategy:N",
                        scale=alt.Scale(
                            domain=["Optimized Portfolio", benchmark_name],
                            range=[[0], [6, 4]],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", format="%b %Y"),
                        alt.Tooltip("Strategy:N"),
                        alt.Tooltip("Value:Q", format="$.3f"),
                    ],
                )
            )
            st.altair_chart(_base_theme(eq_chart, 380), use_container_width=True)

            # ---- Drawdown ----
            st.markdown("##### Underwater (Drawdown)")
            drawdown = (eq_curve_named / eq_curve_named.cummax() - 1) * 100
            dd_long = drawdown.reset_index().melt(
                id_vars="index", var_name="Strategy", value_name="Drawdown (%)"
            ).rename(columns={"index": "Date"})

            dd_chart = (
                alt.Chart(dd_long)
                .mark_area(opacity=0.35, line={"strokeWidth": 1.5})
                .encode(
                    x=alt.X("Date:T", title=None),
                    y=alt.Y("Drawdown (%):Q", title="Drawdown (%)", stack=None, scale=alt.Scale(zero=True)),
                    color=alt.Color(
                        "Strategy:N",
                        scale=alt.Scale(
                            domain=["Optimized Portfolio", benchmark_name],
                            range=["#EF476F", "#90a4ae"],
                        ),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", format="%b %Y"),
                        alt.Tooltip("Strategy:N"),
                        alt.Tooltip("Drawdown (%):Q", format=".1f"),
                    ],
                )
            )
            st.altair_chart(_base_theme(dd_chart, 280), use_container_width=True)

            # ---- Performance Summary Table ----
            st.markdown("##### Performance Summary")

            # Renaming the benchmark column in metrics for display
            bt_metrics = bt_metrics.rename(columns={"Benchmark": benchmark_name})

            display_metrics = pd.DataFrame(index=bt_metrics.index, columns=bt_metrics.columns)
            for row in bt_metrics.index:
                for col in bt_metrics.columns:
                    val = bt_metrics.loc[row, col]
                    if row in ("Total Return", "CAGR", "Annualized Volatility", "Max Drawdown"):
                        display_metrics.loc[row, col] = f"{val:.2%}"
                    else:
                        display_metrics.loc[row, col] = f"{val:.2f}"

            st.dataframe(display_metrics, use_container_width=True)

            # Quick verdict
            port_sharpe = bt_metrics.loc["Sharpe Ratio", "Optimized Portfolio"]
            bmk_sharpe  = bt_metrics.loc["Sharpe Ratio", benchmark_name]
            if port_sharpe > bmk_sharpe:
                st.success(
                    f"🏆  The optimized portfolio achieved a **higher Sharpe ratio** "
                    f"({port_sharpe:.2f} vs {bmk_sharpe:.2f}) than {benchmark_name} over "
                    f"the backtest period."
                )
            else:
                st.info(
                    f"ℹ️  The {benchmark_name} Sharpe ({bmk_sharpe:.2f}) exceeded the optimized portfolio "
                    f"({port_sharpe:.2f}) during this period. Consider adjusting constraints."
                )