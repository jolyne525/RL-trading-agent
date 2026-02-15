from __future__ import annotations

import math
import numpy as np
import pandas as pd


def compute_metrics(history_df: pd.DataFrame, initial_balance: float, rf_annual: float = 0.02, trading_days: int = 252):
    df = history_df.copy()

    init_price = float(df.iloc[0]["price"])
    df["benchmark_nav"] = float(initial_balance) * (df["price"] / init_price)

    df["pct_change"] = df["net_worth"].pct_change().fillna(0.0).astype(float)

    strat_ret = (float(df.iloc[-1]["net_worth"]) - float(initial_balance)) / float(initial_balance)
    bench_ret = (float(df.iloc[-1]["benchmark_nav"]) - float(initial_balance)) / float(initial_balance)
    alpha = strat_ret - bench_ret

    rf_daily = float(rf_annual) / float(trading_days)
    excess = (df["pct_change"] - rf_daily).astype(float)

    # ---- Sharpe: numerical guard ----
    # If strategy is (near) constant returns, std ~ 0 => Sharpe should be 0 (or undefined).
    std_excess = float(np.std(excess.to_numpy()))
    mean_excess = float(np.mean(excess.to_numpy()))
    eps = 1e-12
    if std_excess < eps:
        sharpe = 0.0
    else:
        sharpe = float(mean_excess / std_excess * math.sqrt(trading_days))

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