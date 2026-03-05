import numpy as np
import pandas as pd
from scipy.stats import norm
from dataclasses import dataclass

TRADING_DAYS = 252


@dataclass
class Metrics:
    total_return: float
    ann_return:   float
    ann_vol:      float
    sharpe:       float
    sortino:      float
    max_drawdown: float
    calmar:       float
    var_95:       float   # historical 1-day VaR
    cvar_95:      float   # historical 1-day CVaR
    var_param:    float   # parametric 1-day VaR (normal dist)
    hit_rate:     float


def compute_metrics(returns: pd.Series, rf: float = 0.05) -> Metrics:
    r = returns.dropna()
    if len(r) < 2:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    total  = float(np.exp(r.sum()) - 1)
    n_yrs  = len(r) / TRADING_DAYS
    ann_r  = float((1 + total) ** (1 / n_yrs) - 1) if n_yrs > 0 else 0.0
    ann_v  = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = (ann_r - rf) / ann_v if ann_v > 0 else 0.0

    downside = r[r < 0]
    down_vol = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else ann_v
    sortino  = (ann_r - rf) / down_vol if down_vol > 0 else 0.0

    equity = np.exp(r.cumsum())
    dd     = (equity - equity.cummax()) / equity.cummax()
    mdd    = float(dd.min())
    calmar = ann_r / abs(mdd) if mdd != 0 else 0.0

    var  = float(np.percentile(r, 5))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var
    # Parametric VaR: assumes returns ~ N(μ, σ²), uses inverse CDF at 5%
    var_param = float(norm.ppf(0.05, loc=float(r.mean()), scale=float(r.std())))

    active   = r[r != 0]
    hit_rate = float((active > 0).mean()) if len(active) > 0 else 0.5

    return Metrics(total, ann_r, ann_v, sharpe, sortino, mdd, calmar,
                   var, cvar, var_param, hit_rate)


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = np.exp(returns.cumsum())
    return (equity - equity.cummax()) / equity.cummax()


def monte_carlo_var(log_ret: pd.DataFrame, weights: np.ndarray,
                    n_sims: int = 3000, horizon: int = 21) -> dict:
    """Vectorized Monte Carlo VaR via Cholesky decomposition."""
    rng = np.random.default_rng(42)
    mu  = log_ret.mean().values
    cov = log_ret.cov().values

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + np.eye(len(mu)) * 1e-8)

    w = np.array(weights, dtype=float)

    # All simulations at once — shape (n_sims, horizon, n_assets)
    Z        = rng.standard_normal((n_sims, horizon, len(mu)))
    sim_ret  = mu + Z @ L.T          # (n_sims, horizon, n_assets)
    port_ret = sim_ret @ w           # (n_sims, horizon)
    terminal = port_ret.sum(axis=1)  # (n_sims,)

    var  = float(np.percentile(terminal, 5))
    cvar = float(terminal[terminal <= var].mean())

    return {"terminal": terminal, "var": var, "cvar": cvar, "horizon": horizon}
