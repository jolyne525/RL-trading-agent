````md
# 📈 Reinforcement Learning for Algorithmic Trading (DQN)

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/jolyne525/RL-trading-agent)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://RL-trading-agent-bpgaqfvcpgg2tnc7mhlxga.streamlit.app/)

This project formulates single-asset trading as a **Markov Decision Process (MDP)** and trains a **Deep Q-Network (DQN)** agent on historical daily prices. It provides:

- A **headless `rltrader/` package** for reproducible training + walk-forward backtesting
- A **Streamlit dashboard** (`app.py`) for interactive analysis (signals, equity curves, metrics)
- A **CLI batch runner** (`run_experiments.py`) that exports results to CSV/JSON under `results/`

**Tech Stack:** Python (NumPy, pandas), Streamlit, Plotly, matplotlib, Stooq CSV (default), yfinance (optional)

> Note: research/education demo only. Not financial advice.

---

## 🖼️ Demo Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/37c6c58c-21b3-4c3d-944e-dea37b364258" alt="RL Trading Agent Dashboard" width="900">
</p>

---

## ✨ What This Project Demonstrates

- **MDP Modeling (Trading as Decision Process)**  
  Compact state representation with discrete actions (hold/buy/sell), and a reward derived from **Δ(net worth)**.

- **Canonical DQN Components (Implemented from scratch in NumPy)**  
  **Experience replay**, **target network** (hard updates), and optional **Double DQN** target computation.

- **More Realistic Execution Model**  
  Supports **fixed fee**, **proportional cost (bps)**, and **slippage (bps)**; these affect net worth and therefore learning dynamics.

- **Walk-Forward Train/Test Evaluation**  
  Chronological split (train on early period, test on later period) to reduce look-ahead bias.

- **Quant Metrics + Benchmarking**  
  Benchmarks against **Buy & Hold** and reports **cumulative return**, **Sharpe (annualized)**, **max drawdown**, **turnover**, and **alpha vs benchmark**.

- **Interactive Dashboard + Batch Experiments**  
  Streamlit UI for analysis; CLI runner for reproducible CSV/JSON artifacts under `results/`.

---

## 🔧 Method Overview

### 1) MDP Design

- **State:** `[daily return, position flag, bias]` (3D)
- **Action space:** `{0: hold, 1: buy, 2: sell}`
- **Execution & costs:**  
  Buy executes at `price*(1+slippage)`; sell at `price*(1-slippage)`.  
  Transaction cost = `fixed_cost + (cost_bps/10000)*notional`.
- **Reward:**  
  `reward = Δ(net worth) * reward_scale`  
  (net worth already reflects costs/slippage through balance and executed trades)

### 2) DQN Training

- **Exploration:** epsilon-greedy with optional linear decay
- **Learning:** TD loss with discount factor `gamma`
- **Replay Buffer:** uniform sampling
- **Target Network:** hard update every N steps
- **Double DQN:** optional (online selects action, target evaluates)

### 3) Evaluation (Walk-forward)

- Chronological split by `train_ratio` (default 0.7)
- Test uses greedy policy (`eval_mode=True`)
- Benchmark: Buy & Hold NAV

---

## 🚀 Quick Start (Run Locally)

### 0) Requirements

- **Python ≥ 3.10** (required by type syntax like `str | None`)

### 1) Clone

```bash
git clone https://github.com/jolyne525/RL-trading-agent.git
cd RL-trading-agent
````

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

**Default data source:** `stooq` (recommended).
Yahoo via `yfinance` can rate-limit; stooq + caching is more stable for reproducible runs.

```bash
python run_experiments.py \
  --tickers NVDA,AAPL \
  --start 2016-01-01 \
  --end 2024-01-01 \
  --episodes 200 \
  --seed 1 \
  --out_dir results \
  --data_source stooq \
  --cache_dir data_cache
```

Optional friction controls:

```bash
python run_experiments.py --tickers NVDA --start 2016-01-01 --end 2024-01-01 \
  --episodes 200 --seed 1 --out_dir results \
  --data_source stooq --cache_dir data_cache \
  --fixed_cost 0.05 --cost_bps 1.0 --slippage_bps 1.0
```

---

## 📦 Results (Out-of-sample)

Metrics are computed on the **walk-forward test split** and benchmarked against **Buy & Hold**.
All reported values come from `results/summary.csv` and the corresponding test equity curves.

| Ticker | Strategy Return | Buy&Hold Return |   Alpha | Sharpe | Max DD | Trades | Turnover |
| ------ | --------------: | --------------: | ------: | -----: | -----: | -----: | -------: |
| AAPL   |          32.56% |          34.17% |  -1.61% |   0.49 | 30.85% |     69 |    0.997 |
| NVDA   |         128.55% |         143.56% | -15.01% |   1.11 | 35.74% |    461 |    0.994 |

Equity curves on the test split:

![AAPL equity](results/AAPL_equity.png)
![NVDA equity](results/NVDA_equity.png)

Notes:

* Alpha = Strategy return − Buy&Hold return on the same test window.
* Sharpe is annualized from daily returns (252 trading days assumption).
* Turnover is notional traded / initial balance (proxy for trading intensity).
* On this window the policy achieves positive absolute returns, but does not outperform Buy&Hold (negative alpha), highlighting regime-dependence and the need for multi-window evaluation.

---

## Reproducibility

The exact run configuration is stored in `results/*_run_config.json`.
To reproduce the reported results (same window / seed / hyperparameters), run:

```bash
python run_experiments.py --tickers AAPL,NVDA --start 2016-01-01 --end 2024-01-01 \
  --episodes 1200 --seed 1 --out_dir results --data_source stooq --cache_dir data_cache \
  --double_dqn --buffer_size 100000 --batch_size 64 --min_buffer_size 256 --target_update_every 500 \
  --learning_rate 0.0003 --gamma 0.99 --epsilon_end 0.10 --epsilon_decay_steps 80000 \
  --fixed_cost 0.05 --cost_bps 1 --slippage_bps 1
```

---

## 📂 Project Structure

```text
RL-trading-agent/
  rltrader/
    __init__.py            # Public API exports
    config.py              # Dataclasses for env/DQN/backtest configs
    utils.py               # Seed & utility helpers
    data.py                # Data ingestion (stooq/yfinance + caching)
    env.py                 # Trading MDP environment (costs/slippage)
    agent.py               # DQN agent (replay, target net, optional Double DQN)
    metrics.py             # Performance metrics (Sharpe, MDD, alpha, turnover)
    train_eval.py          # Walk-forward training + evaluation pipeline
  app.py                   # Streamlit dashboard
  run_experiments.py       # CLI runner -> results/*.csv + *.json
  scripts/                 # Helper scripts (plot, README table generation)
  requirements.txt
  README.md
  .gitignore
  results/                 # Selected artifacts for README (optional to commit)
  data_cache/              # Local price cache (recommended gitignore)
```

---

## ⚠️ Research Notes & Limitations

This repository is designed as a **research/engineering baseline** for RL-driven trading, emphasizing a clean experimental pipeline and reproducibility.

* **MDP simplification:** single-asset trading with discrete actions (hold/buy/sell) and a compact state; position sizing is fixed (shares per trade).
* **Execution model:** supports fixed fee, proportional cost (bps), and slippage (bps). This is a simplified proxy and not a full market microstructure simulator.
* **Walk-forward evaluation:** chronological train/test split is used to reduce look-ahead bias; results can vary across regimes and time windows.
* **Stability:** RL training is sensitive to seeds/hyperparameters. The repo stores full run configs (`results/*_run_config.json`) to enable exact reproduction.
* **Scope:** this is an educational/research demo and **not financial advice**. Real deployment would require stronger features, risk constraints, and more rigorous validation (multi-window, stress tests, portfolio setting, etc.).

---

## 📜 License

MIT

```
::contentReference[oaicite:0]{index=0}
```