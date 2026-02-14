# 📈 Reinforcement Learning for Algorithmic Trading (DQN)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/jolyne525/RL-trading-agent)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://RL-trading-agent-bpgaqfvcpgg2tnc7mhlxga.streamlit.app/)

This project formulates single-asset trading as a **Markov Decision Process (MDP)** and trains a **Deep Q-Network (DQN)** agent on historical daily prices. It provides:

* A **headless rltrader** (`rltrader.py`) for reproducible training + walk-forward backtesting
* A **Streamlit dashboard** (`app.py`) for interactive analysis (signals, equity curves, metrics)
* A **CLI batch runner** (`run_experiments.py`) that exports results to CSV/JSON under `results/`

**Tech Stack:** Python (NumPy, pandas), Streamlit, Plotly, yfinance

> Note: research/education demo only. Not financial advice.

---

## 🖼️ Demo Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/37c6c58c-21b3-4c3d-944e-dea37b364258" alt="RL Trading Agent Dashboard" width="900">
</p>

---

## ✨ What This Project Demonstrates 

* **MDP Modeling (Trading as Decision Process)**
  Compact state representation with discrete actions (hold/buy/sell), and a reward derived from **Δ(net worth)**.

* **Canonical DQN Components (Implemented from scratch in NumPy)**
  **Experience replay**, **target network** (hard updates), and optional **Double DQN** target computation.

* **More Realistic Execution Model**
  Supports **fixed fee**, **proportional cost (bps)**, and **slippage (bps)**; these affect net worth and therefore learning dynamics.

* **Walk-Forward Train/Test Evaluation**
  Chronological split (train on early period, test on later period) to reduce look-ahead bias.

* **Quant Metrics + Benchmarking**
  Compares against **Buy & Hold** and reports **cumulative return**, **Sharpe (annualized)**, **max drawdown**, **turnover**, and **alpha vs benchmark**.

* **Interactive Dashboard + Batch Experiments**
  Streamlit UI for analysis; CLI runner for reproducible CSV/JSON artifacts under `results/`.

---

## 🔧 Method Overview

### 1) MDP Design

* **State:** `[daily return, position flag, bias]` (3D)

* **Action space:** `{0: hold, 1: buy, 2: sell}`

* **Execution & costs:**
  Buy executes at `price*(1+slippage)`; sell at `price*(1-slippage)`.
  Transaction cost = `fixed_cost + (cost_bps/10000)*notional`.

* **Reward:**
  `reward = Δ(net worth) * reward_scale`
  (net worth already reflects costs/slippage through balance and executed trades)

### 2) DQN Training

* **Exploration:** epsilon-greedy with optional linear decay
* **Learning:** TD loss with discount factor `gamma`
* **Replay Buffer:** uniform sampling
* **Target Network:** hard update every N steps
* **Double DQN:** optional (online selects action, target evaluates)

### 3) Evaluation (Walk-forward)

* Chronological split by `train_ratio` (default 0.7)
* Test uses greedy policy (`eval_mode=True`)
* Benchmark: Buy & Hold NAV

---

## 🚀 Quick Start (Run Locally)

### 0) Requirements

* **Python ≥ 3.10** (required by type syntax like `str | None`)

### 1) Clone

```bash
git clone https://github.com/jolyne525/RL-trading-agent.git
cd RL-trading-agent
```

### 2) Install Dependencies

Recommended: use a virtual environment.

```bash
python -m venv .venv
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3) Run the Streamlit App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 🧪 Run Batch Experiments (Headless CLI)

This generates reproducible artifacts under `results/`.

```bash
python run_experiments.py \
  --tickers NVDA,AAPL \
  --start 2021-01-01 \
  --end 2021-06-01 \
  --episodes 200 \
  --seed 42
```

Optional friction controls:

```bash
python run_experiments.py --tickers NVDA --start 2021-01-01 --end 2021-06-01 \
  --episodes 200 --seed 42 \
  --fixed_cost 0.05 --cost_bps 1.0 --slippage_bps 1.0
```

---

## 📦 Outputs (`results/`)

The CLI runner writes:

* `results/<timestamp>_summary.csv`
  One row per ticker: return, benchmark return, alpha, Sharpe, max drawdown, turnover, trades, etc.

* `results/<timestamp>_<TICKER>_equity_history.csv`
  Step-by-step test history: price, action, executed, net worth, benchmark NAV, etc.

* `results/<timestamp>_<TICKER>_run_config.json`
  Full config snapshot (env / dqn / backtest / seed) for reproducibility.

> Tip: add `results/` to `.gitignore` to avoid committing large experiment dumps.

---

## 📂 Project Structure

```text
RL-trading-agent/
  rltrader/
    __init__.py            # Package exports / public API
    config.py              # Dataclasses for env/DQN/backtest configs
    utils.py               # Reproducibility helpers (global seeding, etc.)
    data.py                # Data ingestion (yfinance download + sanitization)
    env.py                 # Trading MDP environment (state/step/reset + costs/slippage)
    agent.py               # DQN agent + replay buffer + target net (+ optional Double DQN)
    metrics.py             # Performance metrics (Sharpe, MDD, alpha, turnover, etc.)
    train_eval.py          # Walk-forward training + evaluation pipeline
  app.py                   # Streamlit app entry (UI + charts + downloads)
  run_experiments.py        # CLI batch runner -> results/*.csv + *.json
  requirements.txt
  README.md
  .gitignore
  assets/                   # (optional) screenshots
  results/                  # generated by run_experiments.py (recommended gitignore)
```

---

## ⚠️ Notes / Limitations

* Research/education demo, not trading advice.
* Single-asset, simplified state and discrete actions; position sizing is fixed (shares per trade).
* Results can be sensitive to regime, date window, and hyperparameters.
* Execution is simplified (fixed + proportional cost + slippage model), not a full microstructure simulator.

---

## 📜 License

MIT

---
