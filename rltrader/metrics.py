from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_metrics(
    history_df: pd.DataFrame,
    initial_balance: float,
    rf_annual: float = 0.02,
    trading_days: int = 252,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.Series]:
    df = history_df.copy()

    init_price = float(df.iloc[0]["price"])
    df["benchmark_nav"] = float(initial_balance) * (df["price"] / init_price)

    df["pct_change"] = df["net_worth"].pct_change().fillna(0.0)

    strat_ret = (float(df.iloc[-1]["net_worth"]) - float(initial_balance)) / float(initial_balance)
    bench_ret = (float(df.iloc[-1]["benchmark_nav"]) - float(initial_balance)) / float(initial_balance)
    alpha = strat_ret - bench_ret

    rf_daily = rf_annual / float(trading_days)
    excess = df["pct_change"] - rf_daily

    sharpe = 0.0
    std = float(np.std(excess))
    if std > 0:
        sharpe = float(np.mean(excess) / std * math.sqrt(trading_days))

    cummax = df["net_worth"].cummax()
    dd = 1.0 - df["net_worth"] / cummax
    mdd = float(dd.max())

    metrics = {
        "Cumulative Return": float(strat_ret),
        "Benchmark Return": float(bench_ret),
        "Alpha": float(alpha),
        "Sharpe": float(sharpe),
        "Max Drawdown": float(mdd),
        "Num Trades": float(df["executed"].sum()) if "executed" in df.columns else float("nan"),
    }
    return metrics, df, dd
