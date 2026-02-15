from __future__ import annotations

import time
import random
import pandas as pd
import yfinance as yf

# yfinance 在不同版本里异常类位置可能变动，做个兼容兜底
try:
    from yfinance.exceptions import YFRateLimitError
except Exception:  # pragma: no cover
    class YFRateLimitError(Exception):
        pass


def get_real_stock_data(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 6,
    base_sleep: float = 1.0,
) -> pd.DataFrame:
    """
    Download and sanitize OHLCV daily data into a 2-column DataFrame:
    Date (datetime64), Close (float).

    Added robustness:
    - Retries with exponential backoff (+ jitter) on transient failures / rate limits
    - Explicit auto_adjust to avoid version-dependent defaults
    - threads=False to reduce concurrency-related rate limiting risk
    """
    last_err: Exception | None = None

    for attempt in range(int(max_retries)):
        try:
            # Explicit auto_adjust to avoid FutureWarning / behavior drift
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

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

        except YFRateLimitError as e:
            last_err = e
        except Exception as e:
            # 兜底：有些情况下 rate limit 不一定抛 YFRateLimitError，而是普通 Exception 文本包含 Too Many Requests
            last_err = e
            msg = str(e).lower()
            if "too many requests" not in msg and "rate limit" not in msg:
                # 非限流类错误就不要盲目重试太久，直接返回空
                print(f"[WARN] {ticker}: download failed (non-rate-limit): {e}")
                return pd.DataFrame()

        # rate-limit / transient: exponential backoff + jitter
        sleep_s = min(60.0, base_sleep * (2 ** attempt))
        sleep_s *= (1.0 + random.uniform(-0.2, 0.2))  # jitter
        time.sleep(max(0.5, sleep_s))

    print(f"[WARN] {ticker}: yfinance rate-limited after {max_retries} retries: {last_err}")
    return pd.DataFrame()
