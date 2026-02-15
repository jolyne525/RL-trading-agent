from __future__ import annotations

import os
import time
import random
from typing import Optional

import pandas as pd
import yfinance as yf


def _sanitize_price_df(df: pd.DataFrame) -> pd.DataFrame:
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


def _download_yahoo(ticker: str, start: str, end: str, max_retries: int = 4) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,   # keep behavior stable; also keeps Adj Close when available
                threads=False,       # reduce concurrency footprint
            )
            out = _sanitize_price_df(df)
            return out
        except Exception as e:
            last_err = e
            # exponential backoff + jitter
            sleep_s = min(30.0, (2 ** attempt) * 1.0)
            sleep_s *= (1.0 + random.uniform(-0.2, 0.2))
            time.sleep(max(0.5, sleep_s))

    print(f"[WARN] Yahoo download failed for {ticker} after retries: {last_err}")
    return pd.DataFrame()


def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip()
    # If user already provides suffix like AAPL.US, keep it.
    if "." in t:
        return t.lower()
    # Default to US equities for your current use case
    return f"{t.lower()}.us"


def _download_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = _to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"[WARN] Stooq download failed for {ticker} ({symbol}): {e}")
        return pd.DataFrame()

    # Stooq columns typically: Date, Open, High, Low, Close, Volume
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    # Align semantics with yfinance: [start, end) (end exclusive)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df = df[(df["Date"] >= start_dt) & (df["Date"] < end_dt)].copy()

    return df[["Date", "Close"]].reset_index(drop=True)


def get_real_stock_data(
    ticker: str,
    start: str,
    end: str,
    data_source: str = "auto",        # auto | yahoo | stooq
    cache_dir: str = "data_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns 2-column DataFrame: Date, Close.

    data_source:
      - "yahoo": yfinance/Yahoo
      - "stooq": Stooq CSV
      - "auto": try Yahoo first, fallback to Stooq

    Cache:
      - saves/loads to {cache_dir}/{ticker}_{start}_{end}_{source}.csv
    """
    src = data_source.lower().strip()
    if src not in {"auto", "yahoo", "stooq"}:
        raise ValueError("data_source must be one of: auto, yahoo, stooq")

    os.makedirs(cache_dir, exist_ok=True)

    # If auto, cache key should reflect actual source used; we try yahoo cache first, then stooq.
    def cache_path(source_name: str) -> str:
        safe_t = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
        return os.path.join(cache_dir, f"{safe_t}_{start}_{end}_{source_name}.csv")

    if use_cache:
        if src in {"yahoo", "auto"} and os.path.exists(cache_path("yahoo")):
            return pd.read_csv(cache_path("yahoo"), parse_dates=["Date"])
        if src in {"stooq", "auto"} and os.path.exists(cache_path("stooq")):
            return pd.read_csv(cache_path("stooq"), parse_dates=["Date"])

    out = pd.DataFrame()

    if src in {"yahoo", "auto"}:
        out = _download_yahoo(ticker, start, end)
        if not out.empty and use_cache:
            out.to_csv(cache_path("yahoo"), index=False)
            return out

    if src in {"stooq", "auto"}:
        out = _download_stooq(ticker, start, end)
        if not out.empty and use_cache:
            out.to_csv(cache_path("stooq"), index=False)
            return out

    return pd.DataFrame()