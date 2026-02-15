"""
Headless batch runner (no Streamlit) to generate reproducible CSV results.

Example:
  python run_experiments.py --tickers NVDA,AAPL --start 2021-01-01 --end 2021-06-01 --episodes 200 --seed 42

Outputs:
  results/summary.csv
  results/{TICKER}_equity_history.csv
  results/{TICKER}_run_config.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import pandas as pd

import rltrader as rt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="NVDA")
    p.add_argument("--start", type=str, default="2021-01-01")
    p.add_argument("--end", type=str, default="2021-06-01")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # env friction
    p.add_argument("--initial_balance", type=float, default=10_000.0)
    p.add_argument("--trade_size", type=int, default=1)
    p.add_argument("--fixed_cost", type=float, default=0.05)
    p.add_argument("--cost_bps", type=float, default=0.0)
    p.add_argument("--slippage_bps", type=float, default=0.0)

    # dqn
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--buffer_size", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--min_buffer_size", type=int, default=1_000)
    p.add_argument("--target_update_every", type=int, default=1_000)
    p.add_argument("--double_dqn", action="store_true")

    # exploration
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end", type=float, default=0.05)
    p.add_argument("--epsilon_decay_steps", type=int, default=20_000)

    # backtest
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--rf_annual", type=float, default=0.02)

    p.add_argument("--out_dir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rt.set_global_seed(args.seed)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    env_cfg = rt.TradingEnvConfig(
        initial_balance=args.initial_balance,
        trade_size=args.trade_size,
        fixed_cost=args.fixed_cost,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        reward_scale=1.0,
    )

    dqn_cfg = rt.DQNConfig(
        state_size=3,
        action_size=3,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        min_buffer_size=args.min_buffer_size,
        target_update_every=args.target_update_every,
        double_dqn=bool(args.double_dqn),
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )

    bt_cfg = rt.BacktestConfig(train_ratio=args.train_ratio, rf_annual=args.rf_annual, trading_days=252)

    rows = []
    run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for ticker in tickers:
        df = rt.get_real_stock_data(ticker, args.start, args.end)
        if df.empty or len(df) < 40:
            print(f"[WARN] {ticker}: insufficient data. Skipping.")
            continue

        res = rt.train_and_backtest_single(
            ticker=ticker,
            df=df,
            episodes=args.episodes,
            seed=args.seed,
            env_cfg=env_cfg,
            dqn_cfg=dqn_cfg,
            bt_cfg=bt_cfg,
        )

        dfh = res["history_df"]
        metrics = res["metrics"]
        metrics_row = {"Ticker": ticker, **metrics}
        rows.append(metrics_row)

        equity_path = os.path.join(args.out_dir, f"{run_tag}_{ticker}_equity_history.csv")
        dfh.to_csv(equity_path, index=False)

        cfg_path = os.path.join(args.out_dir, f"{run_tag}_{ticker}_run_config.json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(res["configs"], f, ensure_ascii=False, indent=2)

        print(f"[OK] {ticker}: wrote {equity_path} and {cfg_path}")

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, f"{run_tag}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[DONE] summary -> {summary_path}")


if __name__ == "__main__":
    main()
