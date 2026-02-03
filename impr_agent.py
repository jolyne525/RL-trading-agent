import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time

# 页面配置
st.set_page_config(page_title="算法交易智能体", page_icon="🤖", layout="wide")

# 股票市场环境 MDP 定义
class StockEnvironment:
    """
    模拟股票市场环境 (MDP)。
    状态: [今日收益率, 持仓标志, 偏置项]
    动作: 0=持有, 1=买入, 2=卖出
    奖励: 净值变动 + 交易成本惩罚 (每次交易 -0.05)
    """
    def __init__(self, data, initial_balance=10000):
        # 确保索引连续，包含 Date 列用于记录
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.step_index = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.trade_volume = 0.0  # 重置累计交易额
        self.history = []
        return self._get_state()
        
    def _get_state(self):
        # 返回当前状态向量: [当日价格变动率, 是否持仓, 偏置项]
        if self.step_index >= len(self.data):
            return np.zeros(3)
        price = self.data.iloc[self.step_index]['Close']
        if self.step_index > 0:
            prev_price = self.data.iloc[self.step_index - 1]['Close']
            pct_change = (price - prev_price) / prev_price
        else:
            pct_change = 0.0
        has_position = 1 if self.shares > 0 else 0
        return np.array([pct_change, has_position, 1.0])

    def step(self, action):
        # 执行动作并推进环境一个时间步
        current_price = self.data.iloc[self.step_index]['Close']
        reward = 0.0
        prev_net_worth = self.net_worth
        # 执行买入动作
        if action == 1:
            if self.balance >= current_price:
                self.shares += 1
                self.balance -= current_price
                reward -= 0.05  # 交易成本惩罚
                # 累计交易额
                self.trade_volume += current_price
        # 执行卖出动作
        elif action == 2:
            if self.shares > 0:
                self.shares -= 1
                self.balance += current_price
                reward -= 0.05  # 交易成本惩罚
                self.trade_volume += current_price
        # 计算新的净值（资产净值 = 现金 + 持仓*现价）
        self.net_worth = self.balance + self.shares * current_price
        # 奖励为净值的增减量（包含未实现盈亏）
        reward += (self.net_worth - prev_net_worth)
        # 记录当前时间步信息
        self.history.append({
            'step': self.step_index,
            'date': self.data.iloc[self.step_index]['Date'],
            'price': current_price,
            'action': action,
            'net_worth': self.net_worth
        })
        # 时间步进
        self.step_index += 1
        # 判断是否到达数据末尾（最后一个数据点用于计算净值变化，不执行动作）
        done = self.step_index >= len(self.data) - 1
        next_state = self._get_state()
        return next_state, reward, done

# 深度 Q 网络智能体定义
class DQNAgent:
    """
    深度Q网络代理 (DQN)，使用一层隐藏层进行近似，具备 ε-贪心策略。
    """
    def __init__(self, state_size, action_size, hidden_size=16):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        # 初始化网络参数（权重在 -0.5~0.5 之间均匀分布）
        self.w1 = np.random.rand(self.state_size, self.hidden_size) - 0.5
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.rand(self.hidden_size, self.action_size) - 0.5
        self.b2 = np.zeros(self.action_size)
        # 学习率和折扣因子、探索率参数
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01

    def act(self, state):
        # ε-贪心选择动作
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        # 前向传播计算 Q(s, a) 并选择使 Q 最大的动作
        z1 = np.dot(state, self.w1) + self.b1
        hidden = np.where(z1 > 0, z1, 0)  # ReLU 激活
        q_values = np.dot(hidden, self.w2) + self.b2
        return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state):
        # 计算当前状态和下一状态的 Q 值
        z1 = np.dot(state, self.w1) + self.b1
        hidden = np.where(z1 > 0, z1, 0)
        q_values = np.dot(hidden, self.w2) + self.b2
        z1_next = np.dot(next_state, self.w1) + self.b1
        hidden_next = np.where(z1_next > 0, z1_next, 0)
        q_next = np.dot(hidden_next, self.w2) + self.b2
        # 目标 Q 值和 TD 误差
        target = reward + self.gamma * np.max(q_next)
        error = target - q_values[action]
        # 更新输出层 (针对执行的 action)
        self.w2[:, action] += self.learning_rate * error * hidden
        self.b2[action]     += self.learning_rate * error
        # 更新隐藏层
        hidden_grad = error * self.w2[:, action]
        hidden_grad = hidden_grad * (hidden > 0)  # 仅更新激活的隐藏单元
        self.w1 += self.learning_rate * np.outer(state, hidden_grad)
        self.b1 += self.learning_rate * hidden_grad
        # 衰减探索率 ε
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

@st.cache_data
def get_real_stock_data(ticker="NVDA", start="2021-01-01", end="2021-06-01"):
    """
    获取真实股票收盘价数据（默认NVDA 2021年上半年）。
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # 如果有复权收盘价，则用它作为收盘价
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        # 只保留日期和收盘价列
        return df[['Date', 'Close']]
    except Exception as e:
        st.error(f"数据下载失败: {e}")
        return pd.DataFrame()

# 应用标题和说明
st.title("Reinforcement Learning Quantitative Trader")
st.markdown("""
* **核心技术:** 深度 Q 网络 (DQN) 强化学习, 马尔可夫决策过程 (MDP), 量化分析  
* **数据源:** Yahoo Finance 历史市场数据 (2021 年)
""")
st.divider()

# 布局两列：左侧参数，右侧输出
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("⚙️ 参数设置")
    ticker_input = st.text_input("股票代码（可输入多个，用逗号分隔）", "NVDA")
    episodes = st.slider("训练轮数", 10, 100, 50)
    train_btn = st.button("🚀 开始训练 & 回测", type="primary")
    st.info("""
    **训练原理:**  
    智能体在历史数据上通过 Trial-and-Error 进行学习，尝试在不同波动情况下采取买/卖操作，以最大化长期净值增长（考虑交易成本）。训练完成后，在未来数据上回测策略表现。
    """)

# 如果 session_state 中没有市场数据，则获取默认 NVDA 数据用于初始预览
if 'market_data' not in st.session_state:
    st.session_state.market_data = get_real_stock_data()
df_preview = st.session_state.market_data

# 主逻辑：点击训练按钮后执行
if train_btn:
    with col2:
        # 处理输入的股票列表
        tickers = [t.strip() for t in ticker_input.split(',') if t.strip()]
        results = []
        total_iterations = len(tickers) * episodes
        current_iter = 0
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        start_time = time.time()
        # 遍历每个股票依次训练和测试
        for idx, ticker in enumerate(tickers, start=1):
            df = get_real_stock_data(ticker)
            if df.empty:
                st.error(f"无法获取 {ticker} 的数据，请检查代码是否正确。")
                st.stop()
            # 按时间顺序划分训练集和测试集 (70%/30%)
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            env_train = StockEnvironment(train_df)
            agent = DQNAgent(state_size=3, action_size=3)
            # 保存训练过程中的第一轮、中间轮次、最后一轮记录
            first_episode_history = None
            mid_episode_history = None
            final_episode_history = None
            mid_index = episodes // 2  # 中间轮次索引
            # 训练智能体
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
                # 记录第1轮、中间轮、最后一轮的交易历史
                if e == 0:
                    first_episode_history = env_train.history.copy()
                if e == mid_index:
                    mid_episode_history = env_train.history.copy()
                if e == episodes - 1:
                    final_episode_history = env_train.history.copy()
                # 更新全局训练进度
                current_iter += 1
                progress = current_iter / total_iterations
                progress_bar.progress(progress)
                if len(tickers) > 1:
                    status_text.code(f"{ticker.upper()} - Episode {e+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
                else:
                    status_text.code(f"Episode {e+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")
            # 在测试集上回测策略
            env_test = StockEnvironment(test_df)
            state = env_test.reset()
            agent.epsilon = 0.0  # 测试时关闭探索
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env_test.step(action)
                state = next_state
            # 收集测试结果和指标
            history_df = pd.DataFrame(env_test.history)
            initial_balance = env_test.initial_balance
            initial_price = history_df.iloc[0]['price']
            history_df['benchmark_nav'] = initial_balance * (history_df['price'] / initial_price)
            strategy_return = (history_df.iloc[-1]['net_worth'] - initial_balance) / initial_balance
            benchmark_return = (history_df.iloc[-1]['benchmark_nav'] - initial_balance) / initial_balance
            alpha = strategy_return - benchmark_return
            history_df['pct_change'] = history_df['net_worth'].pct_change().fillna(0)
            risk_free_rate = 0.02
            daily_rf = risk_free_rate / 252
            excess_returns = history_df['pct_change'] - daily_rf
            sharpe_ratio = 0.0
            if np.std(excess_returns) != 0:
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            cum_max = history_df['net_worth'].cummax()
            max_drawdown = (1 - history_df['net_worth'] / cum_max).max()
            turnover = env_test.trade_volume / initial_balance
            results.append({
                'ticker': ticker.upper(),
                'history_df': history_df,
                'metrics': {
                    'Return (%)': f"{strategy_return*100:.1f}%",
                    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown*100:.1f}%",
                    'Turnover (%)': f"{turnover*100:.1f}%",
                    'Alpha (%)': f"{alpha*100:.1f}%"
                },
                'first_ep': first_episode_history,
                'mid_ep': mid_episode_history,
                'last_ep': final_episode_history
            })
        # 清除进度条和状态文本
        progress_bar.empty()
        status_text.empty()
        st.success(f"训练完成！总耗时 {time.time() - start_time:.2f} 秒")
        # 可视化结果
        # 多股票对比模式
        if len(tickers) > 1:
            st.subheader("1. 策略绩效对比（多股票）")
            fig = go.Figure()
            color_palette = px.colors.qualitative.Plotly
            for i, res in enumerate(results):
                ticker = res['ticker']
                history_df = res['history_df']
                color = color_palette[i % len(color_palette)]
                # 策略净值曲线
                fig.add_trace(go.Scatter(
                    x=history_df['date'], y=history_df['net_worth'],
                    mode='lines', name=f"{ticker} RL策略", 
                    line=dict(color=color, width=3)
                ))
                # 基准净值曲线
                fig.add_trace(go.Scatter(
                    x=history_df['date'], y=history_df['benchmark_nav'],
                    mode='lines', name=f"{ticker} 基准", 
                    line=dict(color=color, width=2, dash='dash')
                ))
            fig.update_layout(yaxis_title="Net Worth ($)")
            st.plotly_chart(fig, use_container_width=True)
            # 不同股票的量化指标表格
            st.subheader("2. 关键量化指标对比")
            metrics_rows = []
            for res in results:
                row = {'Ticker': res['ticker']}
                row.update(res['metrics'])
                metrics_rows.append(row)
            metrics_df = pd.DataFrame(metrics_rows).set_index('Ticker')
            st.table(metrics_df)
        # 单股票模式
        else:
            res = results[0]
            ticker = res['ticker']
            history_df = res['history_df']
            # 交易决策可视化图 (买卖点)
            st.subheader("1. 交易决策可视化")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=history_df['date'], y=history_df['price'],
                mode='lines', name=f"{ticker} 收盘价", line=dict(color='gray', width=1)
            ))
            buy_signals = history_df[history_df['action'] == 1]
            sell_signals = history_df[history_df['action'] == 2]
            fig_price.add_trace(go.Scatter(
                x=buy_signals['date'], y=buy_signals['price'],
                mode='markers', name='买入信号',
                marker=dict(symbol='triangle-up', color='green', size=10)
            ))
            fig_price.add_trace(go.Scatter(
                x=sell_signals['date'], y=sell_signals['price'],
                mode='markers', name='卖出信号',
                marker=dict(symbol='triangle-down', color='red', size=10)
            ))
            st.plotly_chart(fig_price, use_container_width=True)
            # 策略净值 vs 基准 净值曲线
            st.subheader("2. 策略绩效对比")
            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(
                x=history_df['date'], y=history_df['net_worth'],
                mode='lines', name='RL 策略净值', line=dict(color='#636EFA', width=3)
            ))
            fig_nav.add_trace(go.Scatter(
                x=history_df['date'], y=history_df['benchmark_nav'],
                mode='lines', name='买入持有净值', line=dict(color='gray', dash='dash')
            ))
            fig_nav.update_layout(yaxis_title="Net Worth ($)")
            st.plotly_chart(fig_nav, use_container_width=True)
            # 关键量化指标 (单股票)
            st.subheader("3. 关键量化指标")
            strategy_return = float(res['metrics']['Return (%)'].strip('%'))
            benchmark_return = (history_df.iloc[-1]['benchmark_nav'] - initial_balance) / initial_balance
            k1, k2, k3 = st.columns(3)
            k1.metric("累计收益", res['metrics']['Return (%)'], delta=f"基准 {benchmark_return*100:.1f}%")
            k2.metric("夏普比率", f"{float(res['metrics']['Sharpe Ratio']):.2f}", help=">1.0 通常被认为是优秀的")
            k3.metric("Alpha (超额收益)", res['metrics']['Alpha (%)'], delta="CV Key Metric")
            k4, k5 = st.columns(2)
            k4.metric("最大回撤", res['metrics']['Max Drawdown (%)'])
            k5.metric("周转率", res['metrics']['Turnover (%)'])
            # 学习过程权益曲线对比图
            st.subheader("4. 训练轮次权益曲线对比")
            first_ep = pd.DataFrame(res['first_ep'])
            mid_ep = pd.DataFrame(res['mid_ep']) if res['mid_ep'] is not None else None
            last_ep = pd.DataFrame(res['last_ep'])
            fig_learning = go.Figure()
            fig_learning.add_trace(go.Scatter(
                x=first_ep['date'], y=first_ep['net_worth'],
                mode='lines', name='第1轮',
                line=dict(color='gray', dash='dash')
            ))
            if mid_ep is not None:
                fig_learning.add_trace(go.Scatter(
                    x=mid_ep['date'], y=mid_ep['net_worth'],
                    mode='lines', name=f"第{mid_index+1}轮",
                    line=dict(color='orange', dash='dashdot')
                ))
            fig_learning.add_trace(go.Scatter(
                x=last_ep['date'], y=last_ep['net_worth'],
                mode='lines', name=f"第{episodes}轮",
                line=dict(color='#636EFA', width=3)
            ))
            fig_learning.update_layout(yaxis_title="Net Worth ($)")
            st.plotly_chart(fig_learning, use_container_width=True)
else:
    # 初次加载或未点击开始时，显示预览图
    with col2:
        st.info("👈 请设置参数并点击 '开始训练 & 回测' 按钮")
        if df_preview is not None and not df_preview.empty:
            fig_preview = px.line(df_preview, x='Date', y='Close', title=f"{df_preview.iloc[0]['Date'].strftime('%Y-%m-%d')} ~ {df_preview.iloc[-1]['Date'].strftime('%Y-%m-%d')} 收盘价")
            st.plotly_chart(fig_preview, use_container_width=True)
        else:
            st.error("没有预览数据可显示，请检查数据源。")
