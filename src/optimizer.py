import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize, Bounds
from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class Portfolio:
    label:   str
    weights: np.ndarray
    ret:     float
    vol:     float
    sharpe:  float


def _stats(w, mu, cov, rf):
    r = float(w @ mu) * TRADING_DAYS
    v = float(np.sqrt(max(w @ cov @ w, 0)) * np.sqrt(TRADING_DAYS))
    s = (r - rf) / v if v > 0 else 0.0
    return r, v, s


def _minimise_vol(w0, cov, extra=None, allow_short=False):
    n    = len(w0)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if extra:
        cons += extra
    res = minimize(
        lambda w: w @ cov @ w, w0,
        method="SLSQP",
        constraints=cons,
        bounds=Bounds(lb=-1.0 if allow_short else 0.0, ub=1.0),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.abs(res.x)
    return w / w.sum()


@st.cache_data(show_spinner="Optimising…")
def run_optimisation(
    _log_ret: pd.DataFrame,
    names: tuple,
    n_points: int,
    rf: float,
    allow_short: bool,
) -> dict:
    """
    Compute the efficient frontier and three key portfolios.
    Cached — only re-runs when inputs actually change.
    """
    mu  = _log_ret.mean().values
    cov = _log_ret.cov().values
    n   = len(mu)
    names = list(_log_ret.columns)

    # --- Three special portfolios ---
    w_gmv  = _minimise_vol(np.ones(n) / n, cov, allow_short=allow_short)

    # Max Sharpe via negative Sharpe minimisation
    ann_mu, ann_cov = mu * TRADING_DAYS, cov * TRADING_DAYS
    res = minimize(
        lambda w: -(w @ ann_mu - rf) / np.sqrt(max(w @ ann_cov @ w, 1e-12)),
        np.ones(n) / n,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        bounds=Bounds(lb=-1.0 if allow_short else 0.0, ub=1.0),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w_tang = np.abs(res.x); w_tang /= w_tang.sum()
    w_eq   = np.ones(n) / n

    def make(label, w):
        r, v, s = _stats(w, mu, cov, rf)
        return Portfolio(label, w, r, v, s)

    gmv_port  = make("Min Variance", w_gmv)
    tang_port = make("Max Sharpe",   w_tang)
    eq_port   = make("Equal Weight", w_eq)

    # --- Efficient frontier curve ---
    r_min = gmv_port.ret
    r_max = max(float(mu.max()) * TRADING_DAYS * 0.98, r_min + 0.01)

    frontier = []
    for target in np.linspace(r_min, r_max, n_points):
        t_d = target / TRADING_DAYS
        w   = _minimise_vol(
            w_gmv, cov,
            extra=[{"type": "eq", "fun": lambda w, t=t_d: w @ mu - t}],
            allow_short=allow_short,
        )
        r, v, s = _stats(w, mu, cov, rf)
        frontier.append(Portfolio("Frontier", w, r, v, s))

    return {
        "names":    names,
        "frontier": frontier,
        "gmv":      gmv_port,
        "tang":     tang_port,
        "eq":       eq_port,
        "asset_ret": (mu * TRADING_DAYS).tolist(),
        "asset_vol": [float(np.sqrt(cov[i, i] * TRADING_DAYS)) for i in range(n)],
    }
