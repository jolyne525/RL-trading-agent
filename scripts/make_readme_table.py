from __future__ import annotations

import math
import pandas as pd

SUMMARY_PATH = "results/summary.csv"
OUT_PATH = "results/summary.md"

df = pd.read_csv(SUMMARY_PATH)

preferred = [
    "Ticker",
    "Cumulative Return",
    "Benchmark Return",
    "Alpha",
    "Sharpe",
    "Max Drawdown",
    "Num Trades",
    "Trade Volume",
    "Turnover",
]
cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
df = df[cols]


def _is_bad(x: float) -> bool:
    return math.isnan(x) or math.isinf(x) or abs(x) > 1e6


def fmt_pct(x) -> str:
    if pd.isna(x):
        return ""
    v = float(x)
    if _is_bad(v):
        return "NA"
    return f"{v*100:.2f}%"


def fmt_sharpe(x) -> str:
    if pd.isna(x):
        return ""
    v = float(x)
    if _is_bad(v):
        return "NA"
    return f"{v:.2f}"


def fmt_int(x) -> str:
    if pd.isna(x):
        return ""
    return str(int(round(float(x))))


def fmt_money(x) -> str:
    if pd.isna(x):
        return ""
    v = float(x)
    if _is_bad(v):
        return "NA"
    return f"{v:,.2f}"


for c in ["Cumulative Return", "Benchmark Return", "Alpha", "Max Drawdown", "Turnover"]:
    if c in df.columns:
        df[c] = df[c].map(fmt_pct)

if "Sharpe" in df.columns:
    df["Sharpe"] = df["Sharpe"].map(fmt_sharpe)

if "Num Trades" in df.columns:
    df["Num Trades"] = df["Num Trades"].map(fmt_int)

if "Trade Volume" in df.columns:
    df["Trade Volume"] = df["Trade Volume"].map(fmt_money)

# escape pipes
for c in df.columns:
    df[c] = df[c].map(lambda s: "" if pd.isna(s) else str(s).replace("|", "\\|"))


def to_md_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join(lines) + "\n"


with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(to_md_table(df))

print(f"[OK] wrote {OUT_PATH}")