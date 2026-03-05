import numpy as np
import pandas as pd
import streamlit as st

from src.data      import search_tickers, load_prices, log_returns, port_returns
from src.stats     import compute_metrics, monte_carlo_var
from src.optimizer import run_optimisation
from src.charts    import (price_chart, correlation_heatmap, return_dist_chart,
                            equity_chart, drawdown_chart, rolling_chart,
                            weight_bar, frontier_chart, mc_chart)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "GLD", "TLT"]
RISK_FREE_RATE  = 0.05


# Page config
st.set_page_config(
    page_title="Quant Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background:#0a0f1e; }
  section[data-testid="stSidebar"] { background:#0d1421; }
  div[data-testid="stMetric"] {
    background:#111827; border:1px solid #1e293b;
    border-radius:10px; padding:14px 18px;
  }
  div[data-testid="stMetricLabel"] { color:#64748b !important; font-size:11px; text-transform:uppercase; }
  div[data-testid="stMetricValue"] { color:#e2e8f0 !important; font-size:22px; }
  button[data-baseweb="tab"] { color:#64748b; font-size:14px; }
  button[data-baseweb="tab"][aria-selected="true"] { color:#3b82f6; }
  div[data-baseweb="tab-highlight"] { background-color:#3b82f6; }
  div[data-baseweb="tab-border"]    { background-color:#1e293b; }
  h1, h2, h3 { color:#e2e8f0 !important; }
  #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("## Portfolio Settings")
    st.divider()

    # Ticker search 
    st.markdown("#### Search Tickers")
    query = st.text_input("Search by name or symbol",
                          placeholder="e.g. Apple, Bitcoin, S&P 500…")

    if "tickers" not in st.session_state:
        st.session_state.tickers = list(DEFAULT_TICKERS)

    if query:
        with st.spinner("Searching…"):
            hits = search_tickers(query)
        if hits:
            for h in hits:
                added = h["symbol"] in st.session_state.tickers
                if st.button(
                    ("✓ " if added else "+ ") + h["label"][:40],
                    key=f"btn_{h['symbol']}",
                    disabled=added,
                    use_container_width=True,
                ):
                    st.session_state.tickers.append(h["symbol"])
                    st.rerun()
        else:
            st.caption("No results.")

    # Active tickers
    st.markdown("#### Active Tickers")
    remove = None
    for t in st.session_state.tickers:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"`{t}`")
        if c2.button("✕", key=f"rm_{t}"):
            remove = t
    if remove:
        st.session_state.tickers.remove(remove)
        st.rerun()

    if st.button("↺ Reset to defaults", use_container_width=True):
        st.session_state.tickers = list(DEFAULT_TICKERS)
        st.rerun()

    # Settings
    st.divider()
    st.markdown("#### Date Range")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", value=pd.Timestamp("2020-01-01"))
    end   = c2.date_input("End",   value=pd.Timestamp("2024-12-31"))

    st.divider()
    st.markdown("#### Parameters")
    rf          = st.slider("Risk-Free Rate (%)", 0.0, 10.0, RISK_FREE_RATE * 100, 0.25) / 100
    roll_win    = st.slider("Rolling Window (days)", 10, 63, 21)
    allow_short = st.checkbox("Allow Short-Selling", value=False)

    st.divider()
    st.markdown("#### Monte Carlo")
    mc_sims    = st.slider("Simulations", 500, 10000, 3000, 500)
    mc_horizon = st.slider("Horizon (days)", 1, 63, 21)


# Load & validate data
selected = st.session_state.tickers

if len(selected) < 2:
    st.warning("Add at least 2 tickers using the search above.")
    st.stop()

prices = load_prices(tuple(selected), str(start), str(end))

if prices.empty or len(prices) < 60:
    st.error("Not enough data. Try different tickers or a wider date range.")
    st.stop()

# From this point on, `names` is the single source of truth for asset labels.
# It only contains tickers that actually downloaded successfully.
names   = list(prices.columns)
log_ret = log_returns(prices)   # shape: (n_days, len(names))


# ══════════════════════════════════════════════════════════════════════════════
# Optimisation — cached, so sidebar sliders don't re-trigger the QP solver
# ══════════════════════════════════════════════════════════════════════════════
opt = run_optimisation(log_ret, tuple(names), n_points=60, rf=rf, allow_short=allow_short)

# Portfolio return series — np.dot avoids any pandas alignment issues
tang_ret = port_returns(log_ret, opt["tang"].weights)
eq_ret   = port_returns(log_ret, opt["eq"].weights)

# Rolling metrics — excess return series used consistently in both numerator and denominator
excess_ret = tang_ret - rf / 252
roll_vol   = tang_ret.rolling(roll_win).std() * np.sqrt(252)
roll_sr    = (
    excess_ret.rolling(roll_win).mean()
    / excess_ret.rolling(roll_win).std()
    * np.sqrt(252)
)

# Compute metrics once — reused in Tab 3 and Tab 4
m = compute_metrics(tang_ret, rf)


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='font-size:26px;margin-bottom:2px;'>Quant Portfolio Dashboard</h1>"
    f"<p style='color:#64748b;font-size:13px;margin-top:0;'>"
    f"<b>{' · '.join(names)}</b> &nbsp;|&nbsp; {start} → {end}"
    f" &nbsp;|&nbsp; {len(prices):,} trading days</p>",
    unsafe_allow_html=True,
)
st.divider()


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Optimisation", "Risk Analytics", "VaR & Monte Carlo"
])


# Tab 1: Overview
with tab1:
    st.subheader("Asset Summary")

    rows = []
    for name in names:
        mx = compute_metrics(log_ret[name], rf)
        rows.append({
            "Ticker":       name,
            "Total Return": f"{mx.total_return:.1%}",
            "Ann. Return":  f"{mx.ann_return:.1%}",
            "Ann. Vol":     f"{mx.ann_vol:.1%}",
            "Sharpe":       f"{mx.sharpe:.3f}",
            "Max Drawdown": f"{mx.max_drawdown:.1%}",
            "VaR 95% (1d)": f"{mx.var_95:.3%}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(price_chart(prices), use_container_width=True)
    with col2:
        chosen = st.selectbox("Distribution for:", names)
        st.plotly_chart(return_dist_chart(log_ret, chosen), use_container_width=True)

    st.plotly_chart(correlation_heatmap(log_ret), use_container_width=True)


# Tab 2: Optimisation
with tab2:
    st.subheader("Markowitz Mean-Variance Optimisation")
    st.caption("Minimises portfolio variance for each target return level (SLSQP). "
               "No short-selling by default.")

    c1, c2, c3 = st.columns(3)
    for col, port, label in [
        (c1, opt["gmv"],  "🟢 Min Variance"),
        (c2, opt["tang"], "🟡 Max Sharpe"),
        (c3, opt["eq"],   "🟣 Equal Weight"),
    ]:
        col.markdown(f"**{label}**")
        col.metric("Return", f"{port.ret:.2%}")
        col.metric("Vol",    f"{port.vol:.2%}")
        col.metric("Sharpe", f"{port.sharpe:.3f}")

    st.divider()
    st.plotly_chart(frontier_chart(opt), use_container_width=True)

    st.divider()
    st.subheader("Portfolio Weights")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.plotly_chart(weight_bar(opt["gmv"].weights,  names, "Min Variance"),
                        use_container_width=True)
    with w2:
        st.plotly_chart(weight_bar(opt["tang"].weights, names, "Max Sharpe"),
                        use_container_width=True)
    with w3:
        st.plotly_chart(weight_bar(opt["eq"].weights,   names, "Equal Weight"),
                        use_container_width=True)

    weight_df = pd.DataFrame({
        "Min Variance": opt["gmv"].weights,
        "Max Sharpe":   opt["tang"].weights,
        "Equal Weight": opt["eq"].weights,
    }, index=names)
    st.dataframe(weight_df.style.format("{:.2%}"), use_container_width=True)


# Tab 3: Risk Analytics
with tab3:
    st.subheader("Risk Analytics — Max Sharpe Portfolio")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Return",  f"{m.total_return:.2%}", delta=f"{m.ann_return:.2%} ann.")
    k2.metric("Sharpe Ratio",  f"{m.sharpe:.3f}")
    k3.metric("Sortino Ratio", f"{m.sortino:.3f}")
    k4.metric("Max Drawdown",  f"{m.max_drawdown:.2%}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Ann. Volatility", f"{m.ann_vol:.2%}")
    k6.metric("Calmar Ratio",    f"{m.calmar:.3f}")
    k7.metric("VaR 95% (1d)",    f"{m.var_95:.3%}")
    k8.metric("Hit Rate",        f"{m.hit_rate:.1%}")

    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(equity_chart(tang_ret, eq_ret), use_container_width=True)
    with col2:
        st.markdown("**Definitions**")
        st.markdown("""
            **Sharpe** = (μ − rf) / σ

            **Sortino** = (μ − rf) / σ⁻  
            *(downside vol only)*

            **Calmar** = return / |Max DD|

            **Hit Rate** = % profitable days

            **VaR 95%** = worst 5% of days
            """)

    st.plotly_chart(drawdown_chart(tang_ret), use_container_width=True)
    st.plotly_chart(rolling_chart(roll_vol, roll_sr), use_container_width=True)


# Tab 4: VaR & Monte Carlo
with tab4:
    st.subheader("Value-at-Risk & Monte Carlo")
    st.caption(
        f"**{mc_sims:,} simulations** · **{mc_horizon}-day horizon** · "
        "Cholesky-decomposed multivariate normal"
    )

    with st.spinner("Running Monte Carlo…"):
        mc = monte_carlo_var(log_ret, opt["tang"].weights,
                             n_sims=mc_sims, horizon=mc_horizon)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("MC VaR 95%",    f"{mc['var']:.3%}",
              help=f"Worst 5% of {mc_horizon}-day simulated paths")
    v2.metric("MC CVaR 95%",   f"{mc['cvar']:.3%}",
              help=f"Expected loss beyond VaR over {mc_horizon} days")
    v3.metric("Hist. VaR 95% (1d)", f"{m.var_95:.3%}",
              help="5th percentile of actual historical daily returns")
    v4.metric("Param. VaR 95% (1d)", f"{m.var_param:.3%}",
              help="Normal distribution: μ − 1.645σ on daily returns")

    st.plotly_chart(mc_chart(mc), use_container_width=True)

    st.divider()
    st.subheader("Methods")
    e1, e2, e3 = st.columns(3)
    e1.markdown("**Historical**\nSort past returns, take 5th percentile.\n\n*Simple. No distribution assumption.*")
    e2.markdown("**Parametric**\nAssume r ~ N(μ, σ²), use z-score.\n\n*Fast. Underestimates fat tails.*")
    e3.markdown("**Monte Carlo**\nSimulate correlated paths via Cholesky.\n\n*Most flexible. Captures correlation.*")
