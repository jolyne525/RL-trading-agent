# 📈 Reinforcement Learning for Algorithmic Trading

This project implements a Deep Q-Network (DQN) agent to learn and optimize trading strategies based on historical stock data via reinforcement learning (RL).

**🔗 Live Demo:** *https://rltradingagent-bpgaqfvcpgg2tnc7mhlxga.streamlit.app/*  
**🧠 GitHub Repo:** https://github.com/jolyne525/RLtradingagent.git

---

## 💡 Project Highlights

- 📊 **MDP Modeling:** Trading is framed as a Markov Decision Process (MDP) with:
  - **States:** [daily return, position flag, bias term]
  - **Actions:** {0: hold, 1: buy, 2: sell}
  - **Rewards:** Change in net worth minus transaction cost penalty

- 🧠 **DQN Agent:** 
  - Epsilon-greedy exploration
  - TD learning with value iteration
  - Linear neural net with one hidden layer

- 🔁 **Walk-forward Backtesting:** 
  - Automatic split into training (70%) and test (30%)
  - Prevents look-ahead bias

- 📈 **Performance Metrics:** 
  - Cumulative return
  - Sharpe ratio
  - Maximum drawdown
  - Turnover (trading volume)

- 📉 **Benchmarked** against Buy & Hold strategy

- 📊 **Interactive Dashboard:**
  - Visualizes buy/sell markers
  - Compares equity curves across episodes
  - Multi-ticker support and training visualization

---

## 🚀 Run It Locally

### 1. Clone the Repository

```bash
git clone https://github.com/jolyne525/RLtradingagent.git
cd RLtradingagent
