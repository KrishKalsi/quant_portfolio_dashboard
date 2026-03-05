# 📊 Quant Portfolio Dashboard

An interactive quantitative finance dashboard built with Python and Streamlit. Pull real market data from Yahoo Finance, optimise a portfolio using Markowitz Mean-Variance theory, and analyse risk using Monte Carlo simulation, Value-at-Risk, and drawdown analytics — all in a browser-based interface with no server required.

---

## Demo

> Search for any ticker by company name or symbol, pick a date range, and every chart and metric updates instantly.

**4 tabs:**
- **Overview** — asset summary stats, normalised prices, correlation matrix, return distributions
- **Optimisation** — interactive efficient frontier, Min Variance / Max Sharpe / Equal Weight portfolios
- **Risk Analytics** — equity curve, drawdown profile, rolling volatility and Sharpe ratio
- **VaR & Monte Carlo** — 3-method VaR comparison, Monte Carlo simulation distribution

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/KrishKalsi/quant_portfolio_dashboard.git
cd quant-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser. No API key needed — data comes from Yahoo Finance.

---

## Project Structure

```
quant_dashboard/
│
├── app.py               # Main Streamlit app — UI layout and tab logic
├── requirements.txt
│
└── src/
    ├── data.py          # Yahoo Finance data loading, ticker search, return computation
    ├── stats.py         # Risk metrics, drawdown, Monte Carlo VaR
    ├── optimizer.py     # Markowitz efficient frontier, GMV, Max Sharpe portfolios
    └── charts.py        # All Plotly chart builders
```

Each module has a single responsibility. `app.py` only handles layout and wiring — all the maths lives in `src/`.

---

## Features & Methodology

### 1. Data — Yahoo Finance via yfinance

All price data is pulled from Yahoo Finance using the `yfinance` library. Prices are adjusted for splits and dividends (`auto_adjust=True`). The ticker search uses `yf.Search` so users can type company names (e.g. "Apple") rather than having to know the exact symbol.

Returns are computed as **log-returns** throughout:

$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

Log-returns are preferred over simple returns for three reasons:
- They are time-additive: the multi-period return is simply the sum of single-period log-returns
- They are more symmetric around zero
- They map naturally to continuous compounding, which is standard in quantitative finance

---

### 2. Portfolio Optimisation — Markowitz Mean-Variance Theory (1952)

The core idea: for a portfolio of $n$ assets with weight vector $\mathbf{w}$, expected return $\boldsymbol{\mu}$, and covariance matrix $\boldsymbol{\Sigma}$, the portfolio's expected return and variance are:

$$\mu_p = \mathbf{w}^\top \boldsymbol{\mu}$$

$$\sigma_p^2 = \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

**The optimisation problem** for each point on the efficient frontier is:

$$\min_{\mathbf{w}} \quad \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}$$

$$\text{subject to:} \quad \mathbf{w}^\top \boldsymbol{\mu} = \mu_{\text{target}}, \quad \sum_i w_i = 1, \quad w_i \geq 0$$

This is a **Quadratic Programme (QP)** — quadratic objective, linear constraints. It is solved numerically using `scipy.optimize.minimize` with the SLSQP (Sequential Least Squares Programming) method.

By sweeping `μ_target` across a grid from the minimum to maximum achievable return and solving the QP at each point, we trace the **Efficient Frontier** — the set of portfolios that maximise return for a given level of risk.

#### Global Minimum Variance (GMV) Portfolio

The leftmost point on the frontier — the portfolio with the absolute lowest possible volatility regardless of return:

$$\min_{\mathbf{w}} \quad \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} \quad \text{s.t.} \quad \sum_i w_i = 1, \quad w_i \geq 0$$

This is useful as a conservative benchmark — it tells you the floor of achievable risk for this set of assets.

#### Maximum Sharpe Ratio (Tangency) Portfolio

The portfolio that maximises the **Sharpe Ratio** — return per unit of risk:

$$\max_{\mathbf{w}} \quad \frac{\mathbf{w}^\top \boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}}$$

where $r_f$ is the risk-free rate. Geometrically, this is the point on the frontier where a line from the risk-free rate is tangent to the curve — hence "tangency portfolio". This is the portfolio used as the primary subject of the Risk Analytics and VaR tabs.

#### Equal Weight Portfolio

A naive benchmark with $w_i = 1/n$ for all assets. Included because in practice it is surprisingly hard to beat consistently, and it provides a meaningful baseline.

---

### 3. Risk Metrics

All metrics are computed from the daily log-return series of the Max Sharpe portfolio.

#### Annualised Return

$$\mu_{\text{ann}} = (1 + r_{\text{total}})^{252/T} - 1$$

where $T$ is the number of trading days and 252 is the standard number of trading days per year.

#### Annualised Volatility

$$\sigma_{\text{ann}} = \sigma_{\text{daily}} \times \sqrt{252}$$

The square root of time scaling follows from the assumption that daily returns are independent and identically distributed.

#### Sharpe Ratio

$$\text{Sharpe} = \frac{\mu_{\text{ann}} - r_f}{\sigma_{\text{ann}}}$$

Measures how much excess return you receive per unit of total risk. A Sharpe above 1.0 is generally considered good; above 2.0 is exceptional.

#### Sortino Ratio

$$\text{Sortino} = \frac{\mu_{\text{ann}} - r_f}{\sigma_{\text{down}}}$$

where $\sigma_{\text{down}}$ is the annualised standard deviation of **negative returns only**. Unlike Sharpe, this doesn't penalise upside volatility, which is arguably a more realistic measure of investor discomfort.

#### Maximum Drawdown

$$\text{MDD} = \min_t \frac{V_t - \max_{\tau \leq t} V_\tau}{\max_{\tau \leq t} V_\tau}$$

The largest peak-to-trough decline in portfolio value. A critical metric for understanding how painful a strategy is to hold through a bad period.

#### Calmar Ratio

$$\text{Calmar} = \frac{\mu_{\text{ann}}}{|\text{MDD}|}$$

Annualised return divided by maximum drawdown. A Calmar of 1.0 means you earned as much in a year as you lost at your worst point. Higher is better.

#### Hit Rate

$$\text{Hit Rate} = \frac{\text{number of positive return days}}{\text{total trading days}}$$

The fraction of days the portfolio made money. A strategy can have a hit rate below 50% and still be profitable if the winning days are larger than the losing days.

---

### 4. Value-at-Risk (VaR) and CVaR

VaR answers the question: *what is the minimum loss I should expect on the worst X% of days?*

At the 95% confidence level, VaR_95 is the threshold such that losses exceed it only 5% of the time.

Three methods are implemented and displayed side by side:

#### Historical VaR

$$\text{VaR}_{95} = \text{Percentile}(r_1, r_2, \ldots, r_T, \; 5\%)$$

No distribution assumption. Just sort all historical daily returns and read off the 5th percentile. Simple and model-free, but entirely backward-looking.

#### Parametric VaR (Normal Distribution)

$$\text{VaR}_{95} = \mu_{\text{daily}} - 1.645 \cdot \sigma_{\text{daily}}$$

Assumes returns are normally distributed and uses the inverse CDF at the 5% level ($z_{0.05} = -1.645$). Fast but known to underestimate tail risk because real financial returns have fat tails — extreme events happen more often than a normal distribution predicts.

#### Conditional VaR (CVaR / Expected Shortfall)

$$\text{CVaR}_{95} = \mathbb{E}\left[r \;\middle|\; r \leq \text{VaR}_{95}\right]$$

The **expected loss given that we are already in the worst 5%**. Also called Expected Shortfall. CVaR is considered a superior risk measure to VaR because it tells you not just *where* the tail starts, but *how bad* the tail is on average. It is coherent (satisfies subadditivity) whereas VaR is not.

---

### 5. Monte Carlo Simulation

Monte Carlo VaR simulates thousands of possible future portfolio return paths and reads the risk statistics off the resulting distribution. The key challenge is ensuring simulated asset returns respect the real-world correlation structure.

**Step 1 — Cholesky Decomposition**

Given the historical covariance matrix $\boldsymbol{\Sigma}$, compute its Cholesky factor $\mathbf{L}$ such that:

$$\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^\top$$

$\mathbf{L}$ is a lower-triangular matrix. This is the multivariate equivalent of taking a square root.

**Step 2 — Correlated random draws**

Draw an uncorrelated standard normal matrix $\mathbf{Z} \sim \mathcal{N}(0, \mathbf{I})$ of shape $(N_{\text{sims}} \times h \times n_{\text{assets}})$ where $h$ is the horizon in days. Then:

$$\mathbf{R} = \boldsymbol{\mu} + \mathbf{Z} \cdot \mathbf{L}^\top$$

The resulting $\mathbf{R}$ has the same covariance structure as the historical data. This is the mathematically correct way to simulate correlated assets — if you simply drew independent returns per asset, you would miss the fact that e.g. tech stocks move together, which would underestimate portfolio risk.

**Step 3 — Portfolio returns and VaR**

For each simulated path, the portfolio daily return is $\mathbf{R} \cdot \mathbf{w}$, and the terminal $h$-day return is the sum over the horizon. The VaR and CVaR are then computed from the distribution of terminal returns across all $N_{\text{sims}}$ paths.

The entire simulation is **fully vectorised** with NumPy — all $N_{\text{sims}}$ paths are computed in a single matrix operation with no Python loops, making it fast even for 10,000 simulations.

> **Note on horizon comparison**: The Monte Carlo VaR is computed over the selected horizon (default 21 days). The historical and parametric VaR metrics shown alongside it are 1-day metrics. They are labeled clearly in the dashboard — do not compare them directly as they measure different time windows.

---

### 6. Rolling Analytics

Rolling metrics use a sliding window (default 21 days, configurable in the sidebar) to show how risk and return evolved through time rather than just on average.

**Rolling Volatility:**
$$\sigma_t^{\text{roll}} = \text{std}(r_{t-w}, \ldots, r_t) \times \sqrt{252}$$

**Rolling Sharpe Ratio:**
$$\text{SR}_t^{\text{roll}} = \frac{\bar{e}_{t-w:t}}{\text{std}(e_{t-w:t})} \times \sqrt{252}$$

where $e_t = r_t - r_f/252$ is the daily excess return. Note that the denominator uses the standard deviation of **excess returns** (not total returns) — this is the technically correct formulation.

These charts are useful for identifying regime changes — periods when a strategy that looked good historically started to deteriorate.

---

## Technical Implementation Notes

### Why log-returns and not simple returns for the covariance matrix?

Using log-returns for the covariance matrix in Markowitz optimisation is standard practice. Log-returns are approximately normally distributed for short horizons, which is the implicit assumption of mean-variance optimisation. Simple returns are bounded at -100% but unbounded on the upside, making their distribution skewed.

### Why SLSQP?

SLSQP (Sequential Least Squares Programming) is the standard choice for this type of problem because:
- It handles both equality constraints (weights sum to 1) and inequality constraints (no short selling) natively
- It converges reliably for smooth, convex objectives like portfolio variance
- It is available in `scipy.optimize` with no extra dependencies

### Why Cholesky and not eigendecomposition?

Both are valid. Cholesky is preferred for simulation because it is computationally cheaper ($O(n^3/3)$ vs $O(n^3)$) and numerically stable for positive-definite matrices. A small regularisation term ($10^{-8} \cdot \mathbf{I}$) is added to the covariance matrix before decomposition to handle near-singular cases (e.g. highly correlated assets or very short date ranges).

### Caching

`@st.cache_data` is applied to both `load_prices` and `run_optimisation`. The optimisation cache key includes the tuple of asset names, risk-free rate, and short-selling flag — so the expensive QP solver only re-runs when those inputs actually change, not on every sidebar interaction.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `yfinance` | Yahoo Finance data download and ticker search |
| `pandas` | DataFrames, date handling |
| `numpy` | Matrix operations, vectorised simulation |
| `scipy` | SLSQP optimisation, normal distribution |
| `plotly` | Interactive charts |

---

## Limitations & Potential Extensions

**Current limitations:**
- Mean-variance optimisation uses **historical means and covariances** as inputs. Historical mean returns are notoriously noisy estimators of future returns — this is a well-known problem in portfolio construction
- Returns are assumed to be **multivariate normal** in the Monte Carlo simulation. Real returns have fat tails and skewness that this model does not capture
- No **transaction costs** or **rebalancing** logic — weights are assumed to be static over the full period

**Possible extensions:**
- **Black-Litterman model** — combine market equilibrium weights with investor views to produce more stable expected return estimates
- **GARCH volatility** — model time-varying volatility instead of assuming constant variance
- **Hierarchical Risk Parity (HRP)** — an alternative to Markowitz that uses clustering and does not require inverting the covariance matrix, making it more robust with many assets
- **Backtesting with rebalancing** — simulate actually trading the optimal portfolio with periodic rebalancing and transaction costs
- **Regime detection** — use Hidden Markov Models to identify bull/bear market regimes and show how correlations change across regimes

---

## References

1. Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77–91.
2. Sharpe, W. F. (1966). *Mutual Fund Performance*. Journal of Business, 39(1), 119–138.
3. Sortino, F. & van der Meer, R. (1991). *Downside Risk*. Journal of Portfolio Management, 17(4), 27–31.
4. Rockafellar, R. T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3), 21–41.
5. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
