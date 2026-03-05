import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats

COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444",
          "#8b5cf6", "#06b6d4", "#ec4899", "#a3e635"]

# Every chart calls this helper to get consistent dark styling.
# Each chart sets its own margin to avoid duplicate-key issues.
def _layout(**kwargs) -> dict:
    base = dict(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#1e293b"),
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
        margin=dict(t=40, r=20, b=40, l=60),
    )
    base.update(kwargs)
    return base


def price_chart(prices: pd.DataFrame) -> go.Figure:
    norm = prices / prices.iloc[0] * 100
    fig  = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col].round(2),
            mode="lines", name=col,
            line=dict(color=COLORS[i % len(COLORS)], width=1.8),
        ))
    fig.update_layout(**_layout(
        title="Normalised Price History (Rebased = 100)",
        yaxis_title="Price",
    ))
    return fig


def correlation_heatmap(log_ret: pd.DataFrame) -> go.Figure:
    corr = log_ret.corr().round(3)
    fig  = go.Figure()
    fig.add_trace(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale=[[0, "#1e3a8a"], [0.5, "#0f172a"], [1, "#14532d"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(**_layout(
        title="Correlation Matrix",
        hovermode="closest",
        margin=dict(t=40, r=20, b=80, l=80),
    ))
    return fig


def return_dist_chart(log_ret: pd.DataFrame, ticker: str) -> go.Figure:
    r       = log_ret[ticker].dropna()
    mu, sig = float(r.mean()), float(r.std())
    x       = np.linspace(r.min(), r.max(), 300)
    fig     = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=80, histnorm="probability density",
        marker_color="#3b82f6", opacity=0.7, name="Empirical",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sp_stats.norm.pdf(x, mu, sig),
        mode="lines", name="Normal fit",
        line=dict(color="#f59e0b", width=2),
    ))
    fig.update_layout(**_layout(
        title=f"Return Distribution — {ticker}",
        hovermode="closest",
        xaxis_title="Daily Return",
        yaxis_title="Density",
    ))
    return fig


def equity_chart(port_ret: pd.Series, bench_ret: pd.Series,
                 port_label: str = "Max Sharpe",
                 bench_label: str = "Equal Weight") -> go.Figure:
    fig = go.Figure()
    for ret, label, color, dash in [
        (port_ret,  port_label,  "#3b82f6", "solid"),
        (bench_ret, bench_label, "#64748b", "dash"),
    ]:
        cum = (100 * np.exp(ret.cumsum())).round(2)
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum,
            mode="lines", name=label,
            line=dict(color=color, width=2, dash=dash),
        ))
    fig.update_layout(**_layout(
        title="Equity Curve (Rebased = 100)",
        yaxis_title="Value",
    ))
    return fig


def drawdown_chart(returns: pd.Series) -> go.Figure:
    equity = np.exp(returns.cumsum())
    dd     = ((equity - equity.cummax()) / equity.cummax() * 100).round(3)
    fig    = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd,
        mode="lines", fill="tozeroy",
        line=dict(color="#ef4444", width=1),
        fillcolor="rgba(239,68,68,0.12)",
        name="Drawdown",
    ))
    fig.update_layout(**_layout(title="Drawdown", yaxis_title="Drawdown (%)"))
    return fig


def rolling_chart(roll_vol: pd.Series, roll_sr: pd.Series) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=roll_vol.index, y=(roll_vol * 100).round(2),
        mode="lines", name="Volatility (%)",
        line=dict(color="#3b82f6", width=1.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=roll_sr.index, y=roll_sr.round(3),
        mode="lines", name="Sharpe Ratio",
        line=dict(color="#f59e0b", width=1.5),
    ), secondary_y=True)
    fig.update_layout(
        title="Rolling Volatility & Sharpe Ratio",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", size=12),
        legend=dict(bgcolor="#0f172a", bordercolor="#1e293b", borderwidth=1),
        hovermode="x unified",
        margin=dict(t=40, r=60, b=40, l=60),
    )
    fig.update_yaxes(title_text="Volatility (%)", secondary_y=False,
                     gridcolor="#1e293b")
    fig.update_yaxes(title_text="Sharpe", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)")
    return fig


def weight_bar(weights: np.ndarray, names: list, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=(weights * 100).round(1), y=names,
        orientation="h",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(names))],
        text=[f"{w:.1f}%" for w in weights * 100],
        textposition="auto",
    ))
    fig.update_layout(**_layout(
        title=title,
        hovermode="closest",
        xaxis_title="Weight (%)",
        yaxis=dict(autorange="reversed", gridcolor="#1e293b"),
        margin=dict(t=40, r=20, b=40, l=100),
    ))
    return fig


def frontier_chart(result: dict) -> go.Figure:
    front = result["frontier"]
    fig   = go.Figure()

    fig.add_trace(go.Scatter(
        x=[p.vol * 100 for p in front],
        y=[p.ret * 100 for p in front],
        mode="lines", name="Efficient Frontier",
        line=dict(color="#3b82f6", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[v * 100 for v in result["asset_vol"]],
        y=[r * 100 for r in result["asset_ret"]],
        mode="markers+text",
        text=result["names"],
        textposition="top center",
        marker=dict(size=9, color="#64748b"),
        name="Assets",
    ))

    specials = [
        (result["gmv"],  "#10b981", "star",    "Min Variance"),
        (result["tang"], "#f59e0b", "star",    "Max Sharpe"),
        (result["eq"],   "#8b5cf6", "diamond", "Equal Weight"),
    ]
    for p, color, sym, label in specials:
        fig.add_trace(go.Scatter(
            x=[p.vol * 100], y=[p.ret * 100],
            mode="markers", name=label,
            marker=dict(size=14, color=color, symbol=sym,
                        line=dict(color="white", width=1.5)),
            customdata=[[p.sharpe]],
            hovertemplate=(
                f"<b>{label}</b><br>"
                "σ=%{x:.2f}%<br>μ=%{y:.2f}%<br>"
                "SR=%{customdata[0]:.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(**_layout(
        title="Efficient Frontier",
        hovermode="closest",
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
    ))
    return fig


def mc_chart(mc: dict) -> go.Figure:
    rets = mc["terminal"] * 100
    var  = mc["var"]  * 100
    cvar = mc["cvar"] * 100
    fig  = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=100,
        marker_color="#3b82f6", opacity=0.75,
        name="Simulated Returns",
    ))
    fig.add_vline(x=var,  line_color="#ef4444", line_dash="dash", line_width=2,
                  annotation_text=f"VaR 95% = {var:.2f}%",
                  annotation_font_color="#ef4444")
    fig.add_vline(x=cvar, line_color="#f59e0b", line_dash="dash", line_width=2,
                  annotation_text=f"CVaR 95% = {cvar:.2f}%",
                  annotation_font_color="#f59e0b",
                  annotation_position="top left")
    fig.update_layout(**_layout(
        title=f"Monte Carlo Distribution ({mc['horizon']}-Day Horizon)",
        hovermode="closest",
        xaxis_title="Log Return (%)",
        yaxis_title="Count",
    ))
    return fig
