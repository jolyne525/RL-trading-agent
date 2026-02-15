from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/AAPL_equity_history.csv")
    p.add_argument("--out", type=str, default="results/AAPL_equity.png")
    p.add_argument("--title", type=str, default="AAPL: RL Equity vs Buy&Hold (Test)")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv, parse_dates=["date"])

    if "net_worth" not in df.columns or "benchmark_nav" not in df.columns:
        raise SystemExit("CSV missing required columns: net_worth / benchmark_nav")

    plt.figure()
    plt.plot(df["date"], df["net_worth"], label="RL")
    plt.plot(df["date"], df["benchmark_nav"], label="Buy&Hold")
    plt.title(args.title)
    plt.xlabel("Date")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()