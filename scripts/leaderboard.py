from __future__ import annotations

import argparse
import glob
import os
import re
import math
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--pattern", type=str, default="*_summary.csv")
    p.add_argument("--ticker", type=str, default="ALL")  # e.g., AAPL
    p.add_argument("--min_trades", type=int, default=10)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out_csv", type=str, default="results/leaderboard.csv")
    p.add_argument("--out_md", type=str, default="results/leaderboard.md")
    return p.parse_args()


def extract_run_tag(path: str) -> str:
    # filename like 20260215_065211_summary.csv
    name = os.path.basename(path)
    m = re.match(r"(\d{8}_\d{6})_summary\.csv$", name)
    return m.group(1) if m else name.replace(".csv", "")


def safe_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        if abs(v) > 1e12:
            return float("nan")
        return v
    except Exception:
        return float("nan")


def to_md_table(df: pd.DataFrame) -> str:
    # escape pipes
    f = df.copy()
    for c in f.columns:
        f[c] = f[c].map(lambda s: "" if pd.isna(s) else str(s).replace("|", "\\|"))

    headers = list(f.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in f.iterrows():
        lines.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v*100:.2f}%"


def fmt_num(v: float, nd: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{nd}f}"


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))

    if not paths:
        raise SystemExit(f"No files matched: {os.path.join(args.results_dir, args.pattern)}")

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["RunTag"] = extract_run_tag(p)
            frames.append(df)
        except Exception:
            continue

    all_df = pd.concat(frames, ignore_index=True)

    # normalize numeric cols
    num_cols = [
        "Cumulative Return",
        "Benchmark Return",
        "Alpha",
        "Sharpe",
        "Max Drawdown",
        "Num Trades",
        "Trade Volume",
        "Turnover",
    ]
    for c in num_cols:
        if c in all_df.columns:
            all_df[c] = all_df[c].map(safe_float)

    # filter ticker if requested
    if args.ticker.upper() != "ALL" and "Ticker" in all_df.columns:
        all_df = all_df[all_df["Ticker"].astype(str).str.upper() == args.ticker.upper()].copy()

    # filter pathological runs
    if "Num Trades" in all_df.columns:
        all_df = all_df[all_df["Num Trades"].fillna(0) >= args.min_trades].copy()

    # score: prefer high Sharpe & Alpha, penalize drawdown (you can tune)
    all_df["Score"] = (
        all_df.get("Sharpe", 0).fillna(0)
        + 0.5 * all_df.get("Alpha", 0).fillna(0)
        - 0.75 * all_df.get("Max Drawdown", 0).fillna(0)
    )

    all_df = all_df.sort_values(["Score"], ascending=False).head(args.top).copy()

    # pretty view for markdown
    view = all_df.copy()
    if "Cumulative Return" in view.columns:
        view["Cumulative Return"] = view["Cumulative Return"].map(fmt_pct)
    if "Benchmark Return" in view.columns:
        view["Benchmark Return"] = view["Benchmark Return"].map(fmt_pct)
    if "Alpha" in view.columns:
        view["Alpha"] = view["Alpha"].map(fmt_pct)
    if "Max Drawdown" in view.columns:
        view["Max Drawdown"] = view["Max Drawdown"].map(fmt_pct)
    if "Turnover" in view.columns:
        view["Turnover"] = view["Turnover"].map(fmt_pct)
    if "Sharpe" in view.columns:
        view["Sharpe"] = view["Sharpe"].map(lambda x: fmt_num(x, 2))
    if "Trade Volume" in view.columns:
        view["Trade Volume"] = view["Trade Volume"].map(lambda x: f"{x:,.2f}" if not pd.isna(x) else "NA")
    if "Num Trades" in view.columns:
        view["Num Trades"] = view["Num Trades"].map(lambda x: str(int(x)) if not pd.isna(x) else "NA")
    view["Score"] = view["Score"].map(lambda x: fmt_num(x, 3))

    # choose columns to show
    show_cols = [c for c in ["RunTag", "Ticker", "Score", "Cumulative Return", "Alpha", "Sharpe", "Max Drawdown", "Num Trades", "Turnover"] if c in view.columns]
    view = view[show_cols]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    all_df.to_csv(args.out_csv, index=False)

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(to_md_table(view))

    print(f"[OK] wrote {args.out_csv}")
    print(f"[OK] wrote {args.out_md}")


if __name__ == "__main__":
    main()