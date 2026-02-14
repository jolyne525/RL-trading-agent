import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time

# -------------------------
# 页面配置
# -------------------------
st.set_page_config(page_title="RL Trading Agent", page_icon="🤖", layout="wide")

# 轻量 UI 美化：metric 卡片化
st.markdown("""
<style>
div[data-testid="stMetric"]{
  background:#f6f8fa;
  padding:12px;
  border-radius:14px;
  border:1px solid #e5e7eb;
}
.block-container{padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

def beautify_fig(fig, title=None, ytitle=None):
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


# -------------------------
# 股票市场环境 MDP 定义
# -------------------------
class StockEnvironment:
    """
    模拟股票市场环境 (MDP)。
    状态: [今日收益率, 持仓标志, 偏置项]
    动作: 0=持有, 1=买入, 2=卖出
    奖励: 净值变动 + 交易成本惩罚 (每次交易 -0.05)
    """
    def __init__(self, data, initial_balance=10000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.reset()

    def reset(self):
        self.step_index = 0
        self.balance = float(self.initial_balance)
        self.shares = 0
        self.net_worth = float(self.initial_balance)
        self.trade_volume = 0.0
        self.history = []
        return self._get_state()

    def _get_state(self):
        # 强制输出为 float32，避免 Streamlit / numpy 组合类型造成 ValueError
        if self.step_index >= len(self.data):
            return np.zeros(3, dtype=np.float32)

        price = float(self.data.loc[self.step_index, "Close"])

        if self.step_index > 0:
            prev_price = float(self.data.loc[self.step_index - 1, "Close"])
            pct_change = (price - prev_price) / prev_price if prev_price != 0 else 0.0
        else:
            pct_change = 0.0

        has_position = 1.0 if self.shares > 0 else 0.0
        return np.array([float(pct_change), float(has_position), 1.0], dtype=np.float32)

    def step(self, action):
        current_price = float(self.data.loc[self.step_index, "Close"])
        prev_net_worth = float(self.net_worth)
        reward = 0.0

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares += 1
                self.balance -= current_price
                reward -= 0.05
                self.trade_volume += current_price

        elif action == 2:  # Sell
            if self.shares > 0:
                self.shares -= 1
                self.balance += current_price
                reward -= 0.05
                self.trade_volume += current_price

        self.net_worth = float(self.balance + self.shares * current_price)
        reward += (self.net_worth - prev_net_worth)

        self.history.append({
            "step": int(self.step_index),
            "date": self.data.loc[self.step_index, "Date"],
            "price": float(current_price),
            "action": int(action),
            "net_worth": float(self.net_worth),
        })

        self.step_index += 1
        done = self.step_index >= len(self.data) - 1
        next_state = self._get_state()
        return next_state, float(reward), bool(done)


# -------------------------
# DQN Agent（你现在这版是手写一层隐藏层的 DQN 近似）
# -------------------------
class DQNAgent:
    """
    深度Q网络代理 (DQN)，使用一层隐藏层近似，ε-贪心探索 + TD 更新
    """
    def __init__(self, state_size, action_size, hidden_size=16):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.w1 = np.random.rand(self.state_size, self.hidden_size) - 0.5
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.rand(self.hidden_size, self.action_size) - 0.5
        self.b2 = np.zeros(self.action_size)

        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        z1 = np.dot(state, self.w1) + self.b1
        hidden = np.where(z1 > 0, z1, 0)  # ReLU
        q_values = np.dot(hidden, self.w2) + self.b2
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state):
        z1 = np.dot(state, self.w1) + self.b1
        hidden = np.where(z1 > 0, z1, 0)
        q_values = np.dot(hidden, self.w2) + self.b2

        z1_next = np.dot(next_state, self.w1) + self.b1
        hidden_next = np.where(z1_next > 0, z1_next, 0)
        q_next = np.dot(hidden_next, self.w2) + self.b2

        target = reward + self.gamma * np.max(q_next)
        error = target - q_values[action]

        # 输出层
        self.w2[:, action] += self.learning_rate * error * hidden
        self.b2[action] += self.learning_rate * error

        # 隐藏层
        hidden_grad = error * self.w2[:, action]
        hidden_grad = hidden_grad * (hidden > 0)
        self.w1 += self.learning_rate * np.outer(state, hidden_grad)
        self.b1 += self.learning_rate * hidden_grad

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# -------------------------
# 数据获取（修复 yfinance MultiIndex / Close 非标量问题）
# -------------------------
@st.cache_data
def get_real_stock_data(ticker="NVDA", start="2021-01-01", end="2021-06-01"):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()

        # MultiIndex 列名摊平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        if isinstance(close, pd.DataFrame):  # 极少数情况
            close = close.iloc[:, 0]

        close = pd.to_numeric(close, errors="coerce")
        out = pd.DataFrame({"Date": df["Date"], "Close": close}).dropna()

        # 防止 Date 不是 datetime
        out["Date"] = pd.to_datetime(out["Date"])
        return out.reset_index(drop=True)

    except Exception as e:
        st.error(f"数据下载失败: {e}")
        return pd.DataFrame()


# -------------------------
# 顶部标题
# -------------------------
st.title("Reinforcement Learning Quantitative Trader")
st.caption("DQN (epsilon-greedy + TD learning) · Walk-forward Train/Test · Benchmark vs Buy&Hold · Sharpe / MDD / Turnover")


# -------------------------
# Sidebar 参数区（页面会立刻更好看）
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    ticker_input = st.text_input("Tickers (comma-separated)", "NVDA")
    episodes = st.slider("Episodes", 10, 150, 50, step=10)
    initial_balance = st.number_input("Initial Balance", value=10000, step=1000)
    start_date = st.text_input("Start (YYYY-MM-DD)", "2021-01-01")
    end_date = st.text_input("End (YYYY-MM-DD)", "2021-06-01")
    train_btn = st.button("🚀 Train & Backtest", type="primary")

    with st.expander("MDP & Reward", expanded=True):
        st.write("- State: [daily return, position flag, bias]")
        st.write("- Actions: hold / buy / sell")
        st.write("- Reward: Δnet_worth - transaction_cost (to reduce over-trading)")


# Tabs（更像作品集）
tab_overview, tab_trades, tab_perf, tab_training = st.tabs(["Overview", "Trades", "Performance", "Training"])


# -------------------------
# 默认预览（未点击训练）
# -------------------------
if 'market_data' not in st.session_state:
    st.session_state.market_data = get_real_stock_data("NVDA", start_date, end_date)

df_preview = st.session_state.market_data

if not train_btn:
    with tab_overview:
        st.info("👈 在左侧设置参数，然后点击 **Train & Backtest**")
        if df_preview is not None and not df_preview.empty:
            fig_preview = px.line(df_preview, x="Date", y="Close", title="Price Preview")
            beautify_fig(fig_preview, title="Price Preview", ytitle="Close")
            st.plotly_chart(fig_preview, use_container_width=True)
        else:
            st.warning("没有预览数据（可能是网络/代码错误或日期范围无数据）")
    st.stop()


# -------------------------
# 训练 + 回测
# -------------------------
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
results = []

try:
    total_iterations = len(tickers) * episodes
    current_iter = 0
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    t0 = time.time()
    for ticker in tickers:
        df = get_real_stock_data(ticker, start_date, end_date)
        if df.empty or len(df) < 20:
            st.error(f"{ticker} 数据不足（可能日期范围太短或下载失败）")
            st.stop()

        # walk-forward: train/test split（70/30）
        train_size = int(len(df) * 0.7)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        env_train = StockEnvironment(train_df, initial_balance=initial_balance)
        agent = DQNAgent(state_size=3, action_size=3)

        first_episode_history = None
        mid_episode_history = None
        final_episode_history = None
        mid_index = episodes // 2

        # train
        for e in range(episodes):
            state = env_train.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = agent.act(state)
                next_state, reward, done = env_train.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            if e == 0:
                first_episode_history = env_train.history.copy()
            if e == mid_index:
                mid_episode_history = env_train.history.copy()
            if e == episodes - 1:
                final_episode_history = env_train.history.copy()

            current_iter += 1
            progress_bar.progress(current_iter / total_iterations)
            status_text.code(f"{ticker} | Episode {e+1}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

        # test（关闭探索）
        env_test = StockEnvironment(test_df, initial_balance=initial_balance)
        state = env_test.reset()
        agent.epsilon = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env_test.step(action)
            state = next_state

        history_df = pd.DataFrame(env_test.history)

        # benchmark：buy&hold
        init_price = float(history_df.iloc[0]["price"])
        history_df["benchmark_nav"] = float(initial_balance) * (history_df["price"] / init_price)

        # metrics
        history_df["pct_change"] = history_df["net_worth"].pct_change().fillna(0.0)

        strat_ret = (float(history_df.iloc[-1]["net_worth"]) - float(initial_balance)) / float(initial_balance)
        bench_ret = (float(history_df.iloc[-1]["benchmark_nav"]) - float(initial_balance)) / float(initial_balance)
        alpha = strat_ret - bench_ret

        rf = 0.02 / 252
        excess = history_df["pct_change"] - rf
        sharpe = 0.0
        if np.std(excess) != 0:
            sharpe = float(np.mean(excess) / np.std(excess) * np.sqrt(252))

        cummax = history_df["net_worth"].cummax()
        dd = 1 - history_df["net_worth"] / cummax
        mdd = float(dd.max())

        turnover = float(env_test.trade_volume) / float(initial_balance)

        results.append({
            "ticker": ticker,
            "history_df": history_df,
            "drawdown": dd,
            "metrics": {
                "Cumulative Return": strat_ret,
                "Benchmark Return": bench_ret,
                "Alpha": alpha,
                "Sharpe": sharpe,
                "Max Drawdown": mdd,
                "Turnover": turnover
            },
            "first_ep": first_episode_history,
            "mid_ep": mid_episode_history,
            "last_ep": final_episode_history
        })

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Done. Total time: {time.time() - t0:.2f}s")

except Exception as e:
    st.error("训练/回测发生异常（已做防护显示，页面不会炸）。请检查日志或按下面异常提示定位。")
    st.exception(e)
    st.stop()


# -------------------------
# 可视化
# -------------------------
if len(results) > 1:
    # 多股票
    with tab_overview:
        st.subheader("Equity Curves (RL vs Buy&Hold)")
        fig = go.Figure()
        palette = px.colors.qualitative.Plotly

        for i, res in enumerate(results):
            color = palette[i % len(palette)]
            dfh = res["history_df"]

            fig.add_trace(go.Scatter(
                x=dfh["date"], y=dfh["net_worth"],
                mode="lines", name=f"{res['ticker']} RL",
                line=dict(color=color, width=3)
            ))
            fig.add_trace(go.Scatter(
                x=dfh["date"], y=dfh["benchmark_nav"],
                mode="lines", name=f"{res['ticker']} Buy&Hold",
                line=dict(color=color, width=2, dash="dash")
            ))

        beautify_fig(fig, title="Equity Curves", ytitle="Net Worth ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_perf:
        st.subheader("Metrics Table")
        rows = []
        for r in results:
            m = r["metrics"]
            rows.append({
                "Ticker": r["ticker"],
                "Return (%)": f"{m['Cumulative Return']*100:.2f}",
                "Benchmark (%)": f"{m['Benchmark Return']*100:.2f}",
                "Alpha (%)": f"{m['Alpha']*100:.2f}",
                "Sharpe": f"{m['Sharpe']:.2f}",
                "Max DD (%)": f"{m['Max Drawdown']*100:.2f}",
                "Turnover (%)": f"{m['Turnover']*100:.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab_trades:
        st.info("多股票模式默认不展示单只股票的买卖点（避免图表太乱）。")

    with tab_training:
        st.info("多股票模式默认不展示训练轮次对比（你需要的话我也可以加一个下拉选择 ticker 来展示）。")

else:
    # 单股票
    res = results[0]
    ticker = res["ticker"]
    dfh = res["history_df"]
    dd = res["drawdown"]
    m = res["metrics"]

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative Return", f"{m['Cumulative Return']*100:.2f}%", delta=f"vs Bench {m['Benchmark Return']*100:.2f}%")
        c2.metric("Sharpe", f"{m['Sharpe']:.2f}")
        c3.metric("Max Drawdown", f"{m['Max Drawdown']*100:.2f}%")

        c4, c5 = st.columns(2)
        c4.metric("Alpha", f"{m['Alpha']*100:.2f}%")
        c5.metric("Turnover", f"{m['Turnover']*100:.2f}%")

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=dfh["date"], y=dfh["net_worth"],
            mode="lines", name="RL Equity", line=dict(width=3)
        ))
        fig_nav.add_trace(go.Scatter(
            x=dfh["date"], y=dfh["benchmark_nav"],
            mode="lines", name="Buy&Hold", line=dict(dash="dash")
        ))
        beautify_fig(fig_nav, title=f"{ticker} Equity Curve", ytitle="Net Worth ($)")
        st.plotly_chart(fig_nav, use_container_width=True)

    with tab_trades:
        st.subheader("Trade Decisions (Buy/Sell markers)")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=dfh["date"], y=dfh["price"], mode="lines",
            name="Price", line=dict(width=1)
        ))

        buy = dfh[dfh["action"] == 1]
        sell = dfh[dfh["action"] == 2]

        fig_price.add_trace(go.Scatter(
            x=buy["date"], y=buy["price"],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=8, opacity=0.85)
        ))
        fig_price.add_trace(go.Scatter(
            x=sell["date"], y=sell["price"],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=8, opacity=0.85)
        ))

        beautify_fig(fig_price, title=f"{ticker} Price + Signals", ytitle="Price")
        st.plotly_chart(fig_price, use_container_width=True)

    with tab_perf:
        st.subheader("Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dfh["date"], y=dd,
            mode="lines", name="Drawdown"
        ))
        beautify_fig(fig_dd, title="Drawdown Curve", ytitle="Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Metrics Detail")
        st.write({
            "Return": f"{m['Cumulative Return']*100:.2f}%",
            "Benchmark": f"{m['Benchmark Return']*100:.2f}%",
            "Alpha": f"{m['Alpha']*100:.2f}%",
            "Sharpe": f"{m['Sharpe']:.2f}",
            "Max Drawdown": f"{m['Max Drawdown']*100:.2f}%",
            "Turnover": f"{m['Turnover']*100:.2f}%"
        })

    with tab_training:
        st.subheader("Learning Progress (Episode 1 / Mid / Last)")
        first_ep = pd.DataFrame(res["first_ep"])
        mid_ep = pd.DataFrame(res["mid_ep"]) if res["mid_ep"] is not None else None
        last_ep = pd.DataFrame(res["last_ep"])

        fig_learn = go.Figure()
        fig_learn.add_trace(go.Scatter(
            x=first_ep["date"], y=first_ep["net_worth"],
            mode="lines", name="Episode 1", line=dict(dash="dash")
        ))
        if mid_ep is not None:
            fig_learn.add_trace(go.Scatter(
                x=mid_ep["date"], y=mid_ep["net_worth"],
                mode="lines", name="Mid Episode", line=dict(dash="dashdot")
            ))
        fig_learn.add_trace(go.Scatter(
            x=last_ep["date"], y=last_ep["net_worth"],
            mode="lines", name="Last Episode", line=dict(width=3)
        ))

        beautify_fig(fig_learn, title="Training Equity Curves", ytitle="Net Worth ($)")
        st.plotly_chart(fig_learn, use_container_width=True)
