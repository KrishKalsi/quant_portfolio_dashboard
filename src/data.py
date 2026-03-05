import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def search_tickers(query: str) -> list[dict]:
    """Search Yahoo Finance by company name or symbol."""
    if not query:
        return []
    try:
        results = yf.Search(query, max_results=8).quotes
        return [
            {
                "symbol": r.get("symbol", ""),
                "label":  f"{r.get('symbol','')} — {r.get('shortname') or r.get('longname', '')}",
            }
            for r in results if r.get("symbol")
        ]
    except Exception:
        return []


@st.cache_data(show_spinner="Downloading data…")
def load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices. Returns a clean DataFrame."""
    raw = yf.download(list(tickers), start=start, end=end,
                      auto_adjust=True, progress=False)

    # Handle both single and multi-ticker column structures
    if isinstance(raw.columns, pd.MultiIndex):
        # Try "Close" first, then "Price"
        level = "Close" if "Close" in raw.columns.get_level_values(0) else "Price"
        prices = raw[level]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw.iloc[:, [0]]
        prices.columns = [tickers[0]]

    prices = prices.ffill().bfill()
    prices = prices.dropna(axis=1, how="all")  # remove tickers with no data
    prices = prices.dropna()
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def port_returns(log_ret: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Portfolio daily return — always uses numpy dot to avoid shape issues."""
    w = np.array(weights, dtype=float)
    return pd.Series(log_ret.values @ w, index=log_ret.index)
