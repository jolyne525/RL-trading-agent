# 📈 Reinforcement Learning for Algorithmic Trading

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/jolyne525/RLtradingagent.git)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rltradingagent-bpgaqfvcpgg2tnc7mhlxga.streamlit.app/)

This project implements a Deep Q-Network (DQN) agent to learn and optimize trading strategies based on historical stock data via reinforcement learning (RL).

<p align="center">
  <img src="https://github.com/user-attachments/assets/37c6c58c-21b3-4c3d-944e-dea37b364258" alt="RL Trading Agent Dashboard" width="900">
</p>


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
