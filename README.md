# 📈 Reinforcement Learning for Algorithmic Trading (DQN)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/jolyne525/RLtradingagent)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rltradingagent-bpgaqfvcpgg2tnc7mhlxga.streamlit.app/)

A reinforcement learning project that formulates trading as a **Markov Decision Process (MDP)** and trains a **Deep Q-Network (DQN)** agent to learn a trading policy from historical price trajectories. The Streamlit dashboard visualizes the learned decisions (buy/sell markers), compares **equity curves** against **buy-and-hold**, and reports key quantitative metrics.

**Tech Stack:** Python (NumPy, pandas), Streamlit, Plotly, yfinance

---

## 🖼️ Demo Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/37c6c58c-21b3-4c3d-944e-dea37b364258" alt="RL Trading Agent Dashboard" width="900">
</p>

---

## ✨ What This Project Demonstrates (Resume-Aligned)

- **MDP Modeling:**  
  Trading is formulated as an MDP with a compact state representation, discrete actions, and a reward function that penalizes transaction costs to mitigate over-trading.

- **DQN Agent Implementation:**  
  Uses **epsilon-greedy exploration** and **temporal-difference learning** to optimize a trading policy from historical trajectories.

- **Walk-Forward Train/Test Pipeline:**  
  Uses a reproducible split (train/validation/test or train/test) to mitigate look-ahead bias and evaluate generalization.

- **Benchmark + Metrics Reporting:**  
  Benchmarks against **Buy & Hold** and reports **cumulative return**, **Sharpe ratio**, **max drawdown**, and **turnover**.

- **Interactive Dashboard:**  
  Visualizes buy/sell markers, compares equity curves, supports multi-ticker evaluation, and shows training episode progression.

---

## 🔧 Method Overview

### 1) MDP Design
- **State (example):** `[daily return, position flag, bias]`
- **Action space:** `{0: hold, 1: buy, 2: sell}`
- **Reward:**  
  `Δ(net worth) - transaction_cost * trade_indicator`  
  (encourages profitability while discouraging excessive trading)

### 2) DQN Training
- **Policy:** epsilon-greedy
- **Learning:** TD target with discounted future rewards
- **Network:** lightweight MLP / linear approximator (implementation-specific)

### 3) Evaluation (Walk-forward)
- Train on early period, test on later period (prevents look-ahead bias)
- Compare with baseline (Buy & Hold)

---

## 🚀 Quick Start (Run Locally)

### 1) Clone Repository
```bash
git clone https://github.com/jolyne525/RLtradingagent.git
cd RLtradingagent
````

### 2) Install Dependencies

> Recommended: create a virtual environment to avoid dependency conflicts.

**Option A: venv**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Option B: conda**

```bash
conda create -n rl-trader python=3.10 -y
conda activate rl-trader
pip install -r requirements.txt
```

### 3) Run the Streamlit App

```bash
streamlit run impr_agent.py
```

Open the URL shown in the terminal (usually: [http://localhost:8501](http://localhost:8501)).

---

## 📌 Usage

1. Enter one ticker (or multiple tickers separated by commas).
2. Set training episodes and other parameters.
3. Click **Train & Backtest**.
4. Review:

   * Price chart with buy/sell markers
   * Strategy equity vs Buy & Hold
   * Metrics: return, Sharpe, max drawdown, turnover
   * Learning curves across episodes (if enabled)

---

## 📂 Project Structure (Suggested)

```text
RLtradingagent/
  impr_agent.py              # Streamlit app entry
  requirements.txt
  README.md
  assets/                    # (optional) screenshots
```

> If your entry file is `app.py` instead of `impr_agent.py`, update the run command accordingly:
> `streamlit run app.py`

---

## ☁️ Deploy to Streamlit Cloud

1. Ensure repo includes:

   * `impr_agent.py` (or `app.py`)
   * `requirements.txt`
   * `README.md`
   * (optional) `assets/`

2. Streamlit Community Cloud → **New app**

3. Select this repo + branch

4. Set **Main file path** to:

   * `impr_agent.py` (or your actual entry file)

5. Deploy

---

## 📊 Metrics Reported

* **Cumulative Return**
* **Sharpe Ratio** (annualized)
* **Maximum Drawdown**
* **Turnover** (proxy for trading frequency/volume)

---

## ⚠️ Notes / Limitations

* This is a research/education project, not financial advice.
* Results depend on the data window, ticker volatility regime, and chosen hyperparameters.
* A simplified environment (e.g., 1-share trades, no slippage) may differ from real execution.

---

## 📜 License

MIT

---
