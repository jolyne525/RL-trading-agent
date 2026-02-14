from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_real_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download and sanitize OHLCV daily data into a 2-column DataFrame:
    Date (datetime64), Close (float).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    if isinstance(close, pd.DataFrame):  # edge case
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    out = pd.DataFrame({"Date": df["Date"], "Close": close}).dropna()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    return out
