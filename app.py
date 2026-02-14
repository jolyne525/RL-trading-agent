"""
Streamlit entry for RLtradingagent (DQN with replay + target network).

Run:
  streamlit run impr_agent.py
"""

from __future__ import annotations

import json
import time
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import rltrader as rt


st.set_page_config(page_title="RL Trading Agent", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
div[data-testid="stMetric"]{
  background:#f6f8fa;
  padding:12px;
  border-radius:14px;
  border:1px solid #e5e7eb;
}
.block-container{padding-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)


def beautify_fig(fig: go.Figure, title: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        title=title,
        yaxis_title=ytitle,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


@st.cache_data(show_spinner=False)
def cached_get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    return rt.get_real_stock_data(ticker=ticker, start=start, end=end)


st.title("Reinforcement Learning Quantitative Trader")
st.caption("DQN (Replay + Target Net + optional Double DQN) · Walk-forward Train/Test · Benchmark vs Buy&Hold · Sharpe / MDD / Turnover")

with st.sidebar:
    st.header("⚙️ Settings")

    ticker_input = st.text_input("Tickers (comma-separated)", "NVDA")
    episodes = st.slider("Episodes", 10, 500, 100, step=10)
    seed = st.number_input("Random Seed", value=42, step=1)

    st.subheader("Data Window")
    start_date = st.text_input("Start (YYYY-MM-DD)", "2021-01-01")
    end_date = st.text_input("End (YYYY-MM-DD)", "2021-06-01")

    st.subheader("Execution Model")
    initial_balance = st.number_input("Initial Balance", value=10000, step=1000)
    trade_size = st.number_input("Trade Size (shares)", value=1, step=1, min_value=1)
    fixed_cost = st.number_input("Fixed Cost / Trade", value=0.05, step=0.01, format="%.4f")
    cost_bps = st.number_input("Proportional Cost (bps)", value=0.0, step=1.0, format="%.2f")
    slippage_bps = st.number_input("Slippage (bps)", value=0.0, step=1.0, format="%.2f")

    st.subheader("DQN Hyperparams")
    hidden_size = st.select_slider("Hidden Size", options=[16, 32, 64, 128], value=64)
    gamma = st.slider("Gamma (discount)", 0.80, 0.999, 0.99, step=0.001)
    learning_rate = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
    batch_size = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=64)
    buffer_size = st.select_slider("Replay Buffer Size", options=[10_000, 50_000, 100_000], value=50_000)
    min_buffer_size = st.select_slider("Min Buffer To Train", options=[256, 1_000, 2_000, 5_000], value=1_000)
    target_update_every = st.select_slider("Target Update Every (steps)", options=[250, 500, 1_000, 2_000], value=1_000)
    double_dqn = st.checkbox("Double DQN", value=True)

    st.subheader("Exploration")
    epsilon_start = st.slider("Epsilon Start", 0.1, 1.0, 1.0, step=0.05)
    epsilon_end = st.slider("Epsilon End", 0.0, 0.2, 0.05, step=0.01)
    epsilon_decay_steps = st.select_slider("Epsilon Decay Steps", options=[0, 5_000, 10_000, 20_000, 50_000], value=20_000)

    st.subheader("Backtest / Metrics")
    train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.7, step=0.05)
    rf_annual = st.slider("Risk-free (annual)", 0.0, 0.10, 0.02, step=0.005)

    train_btn = st.button("🚀 Train & Backtest", type="primary")

    with st.expander("MDP Summary", expanded=True):
        st.write("- State: [daily return, position flag, bias] (3D)")
        st.write("- Actions: hold / buy / sell")
        st.write("- Reward: Δ(net worth) scaled by 1/initial_balance")
        st.write("- DQN: Replay Buffer + Target Network (+ Double DQN optional)")
        st.write("- Note: research/education demo, not financial advice.")


tab_overview, tab_trades, tab_perf, tab_training = st.tabs(["Overview", "Trades", "Performance", "Training"])


# Preview (always reflect latest inputs)
preview_ticker = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
preview_ticker = preview_ticker[0] if preview_ticker else "NVDA"
df_preview = cached_get_data(preview_ticker, start_date, end_date)

if not train_btn:
    with tab_overview:
        st.info("👈 Set parameters in the sidebar, then click **Train & Backtest**.")
        if df_preview is not None and not df_preview.empty:
            fig_preview = px.line(df_preview, x="Date", y="Close", title=f"{preview_ticker} Price Preview")
            st.plotly_chart(beautify_fig(fig_preview, title="Price Preview", ytitle="Close"), use_container_width=True)
        else:
            st.warning("No preview data (check network or date range).")
    st.stop()


# Run
rt.set_global_seed(int(seed))

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
results: List[Dict[str, object]] = []

env_cfg = rt.TradingEnvConfig(
    initial_balance=float(initial_balance),
    trade_size=int(trade_size),
    fixed_cost=float(fixed_cost),
    cost_bps=float(cost_bps),
    slippage_bps=float(slippage_bps),
    reward_scale=1.0,  # will be normalized by rl
)

dqn_cfg = rt.DQNConfig(
    state_size=3,
    action_size=3,
    hidden_size=int(hidden_size),
    gamma=float(gamma),
    learning_rate=float(learning_rate),
    buffer_size=int(buffer_size),
    batch_size=int(batch_size),
    min_buffer_size=int(min_buffer_size),
    target_update_every=int(target_update_every),
    double_dqn=bool(double_dqn),
    epsilon_start=float(epsilon_start),
    epsilon_end=float(epsilon_end),
    epsilon_decay_steps=int(epsilon_decay_steps),
)

bt_cfg = rt.BacktestConfig(
    train_ratio=float(train_ratio),
    rf_annual=float(rf_annual),
    trading_days=252,
)

t0 = time.time()
progress = st.progress(0.0)
status = st.empty()

try:
    for i, ticker in enumerate(tickers):
        status.code(f"Running {ticker} ...")
        df = cached_get_data(ticker, start_date, end_date)
        if df.empty or len(df) < 40:
            st.error(f"{ticker}: insufficient data (short date range or download failure).")
            st.stop()

        res = rt.train_and_backtest_single(
            ticker=ticker,
            df=df,
            episodes=int(episodes),
            seed=int(seed),
            env_cfg=env_cfg,
            dqn_cfg=dqn_cfg,
            bt_cfg=bt_cfg,
        )
        results.append(res)
        progress.progress((i + 1) / len(tickers))

    progress.empty()
    status.empty()
    st.success(f"✅ Done. Total time: {time.time() - t0:.2f}s")

except Exception as e:
    st.error("Training/backtest crashed. See details below.")
    st.exception(e)
    st.stop()


def metrics_row(ticker: str, metrics: Dict[str, float]) -> Dict[str, str]:
    return {
        "Ticker": ticker,
        "Return (%)": f"{metrics['Cumulative Return']*100:.2f}",
        "Benchmark (%)": f"{metrics['Benchmark Return']*100:.2f}",
        "Alpha (%)": f"{metrics['Alpha']*100:.2f}",
        "Sharpe": f"{metrics['Sharpe']:.2f}",
        "Max DD (%)": f"{metrics['Max Drawdown']*100:.2f}",
        "Turnover (%)": f"{metrics.get('Turnover', float('nan'))*100:.2f}",
        "Trades": f"{int(metrics.get('Num Trades', 0))}",
    }


# Visualize
if len(results) > 1:
    with tab_overview:
        st.subheader("Equity Curves (RL vs Buy&Hold)")
        fig = go.Figure()
        palette = px.colors.qualitative.Plotly

        for i, res in enumerate(results):
            dfh = res["history_df"]
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(x=dfh["date"], y=dfh["net_worth"], mode="lines", name=f"{res['ticker']} RL", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=dfh["date"], y=dfh["benchmark_nav"], mode="lines", name=f"{res['ticker']} Buy&Hold", line=dict(width=2, dash="dash")))

        st.plotly_chart(beautify_fig(fig, title="Equity Curves", ytitle="Net Worth ($)"), use_container_width=True)

    with tab_perf:
        st.subheader("Metrics Table")
        rows = [metrics_row(r["ticker"], r["metrics"]) for r in results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Download Results")
        for r in results:
            dfh = r["history_df"]
            st.download_button(
                label=f"Download {r['ticker']} equity history CSV",
                data=dfh.to_csv(index=False).encode("utf-8"),
                file_name=f"{r['ticker']}_equity_history.csv",
                mime="text/csv",
            )
            st.download_button(
                label=f"Download {r['ticker']} run config JSON",
                data=json.dumps(r["configs"], ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"{r['ticker']}_run_config.json",
                mime="application/json",
            )

    with tab_trades:
        st.info("Multi-ticker mode: trade markers are hidden to avoid clutter. Use single ticker for detailed signals.")

    with tab_training:
        st.info("Multi-ticker mode: training curves are omitted by default. Use single ticker for episode curves.")

else:
    r = results[0]
    ticker = r["ticker"]
    dfh = r["history_df"]
    dd = r["drawdown"]
    m = r["metrics"]
    curves = r["train_curves"]

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative Return", f"{m['Cumulative Return']*100:.2f}%", delta=f"vs Bench {m['Benchmark Return']*100:.2f}%")
        c2.metric("Sharpe", f"{m['Sharpe']:.2f}")
        c3.metric("Max Drawdown", f"{m['Max Drawdown']*100:.2f}%")

        c4, c5, c6 = st.columns(3)
        c4.metric("Alpha", f"{m['Alpha']*100:.2f}%")
        c5.metric("Turnover", f"{m.get('Turnover', float('nan'))*100:.2f}%")
        c6.metric("#Trades", f"{int(m.get('Num Trades', 0))}")

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(x=dfh["date"], y=dfh["net_worth"], mode="lines", name="RL Equity", line=dict(width=3)))
        fig_nav.add_trace(go.Scatter(x=dfh["date"], y=dfh["benchmark_nav"], mode="lines", name="Buy&Hold", line=dict(dash="dash")))
        st.plotly_chart(beautify_fig(fig_nav, title=f"{ticker} Equity Curve", ytitle="Net Worth ($)"), use_container_width=True)

    with tab_trades:
        st.subheader("Trade Decisions (Buy/Sell markers)")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=dfh["date"], y=dfh["price"], mode="lines", name="Price", line=dict(width=1)))

        buy = dfh[(dfh["action"] == 1) & (dfh["executed"] == True)]
        sell = dfh[(dfh["action"] == 2) & (dfh["executed"] == True)]

        fig_price.add_trace(go.Scatter(x=buy["date"], y=buy["price"], mode="markers", name="Buy", marker=dict(symbol="triangle-up", size=8, opacity=0.85)))
        fig_price.add_trace(go.Scatter(x=sell["date"], y=sell["price"], mode="markers", name="Sell", marker=dict(symbol="triangle-down", size=8, opacity=0.85)))

        st.plotly_chart(beautify_fig(fig_price, title=f"{ticker} Price + Signals", ytitle="Price"), use_container_width=True)

        st.subheader("Download Single-Ticker Results")
        st.download_button("Download equity history CSV", data=dfh.to_csv(index=False).encode("utf-8"), file_name=f"{ticker}_equity_history.csv", mime="text/csv")
        st.download_button("Download run config JSON", data=json.dumps(r["configs"], ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{ticker}_run_config.json", mime="application/json")

    with tab_perf:
        st.subheader("Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dfh["date"], y=dd, mode="lines", name="Drawdown"))
        st.plotly_chart(beautify_fig(fig_dd, title="Drawdown Curve", ytitle="Drawdown"), use_container_width=True)

        st.subheader("Metrics Detail")
        st.json({k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items()})

    with tab_training:
        st.subheader("Training Curves")
        df_curves = pd.DataFrame(
            {
                "episode": range(1, len(curves["episode_reward"]) + 1),
                "reward": curves["episode_reward"],
                "loss": curves["episode_loss"],
            }
        )
        fig_r = px.line(df_curves, x="episode", y="reward", title="Episode Reward (sum of scaled rewards)")
        st.plotly_chart(beautify_fig(fig_r, title="Episode Reward", ytitle="Reward"), use_container_width=True)

        fig_l = px.line(df_curves, x="episode", y="loss", title="Average TD Loss per Episode (NaN before buffer warms up)")
        st.plotly_chart(beautify_fig(fig_l, title="Episode Loss", ytitle="Loss"), use_container_width=True)
