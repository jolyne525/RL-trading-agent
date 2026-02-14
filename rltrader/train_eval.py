from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .agent import DQNAgent
from .config import BacktestConfig, DQNConfig, TradingEnvConfig
from .env import StockEnvironment
from .metrics import compute_metrics


def train_and_backtest_single(
    ticker: str,
    df: pd.DataFrame,
    episodes: int,
    seed: int,
    env_cfg: TradingEnvConfig,
    dqn_cfg: DQNConfig,
    bt_cfg: BacktestConfig,
) -> Dict[str, object]:
    if df is None or df.empty or len(df) < 40:
        raise ValueError(f"{ticker}: insufficient data for training/backtest.")

    # chronological split
    train_size = int(len(df) * bt_cfg.train_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # default reward scaling: normalize by initial balance
    if env_cfg.reward_scale == 1.0:
        env_cfg = TradingEnvConfig(**{**asdict(env_cfg), "reward_scale": 1.0 / float(env_cfg.initial_balance)})

    env_train = StockEnvironment(train_df, env_cfg)
    agent = DQNAgent(dqn_cfg, seed=int(seed))

    first_episode_history = None
    mid_episode_history = None
    last_episode_history = None
    mid_index = int(episodes) // 2

    episode_rewards: List[float] = []
    episode_losses: List[float] = []

    for e in range(int(episodes)):
        state = env_train.reset()
        done = False
        total_reward = 0.0
        loss_acc: List[float] = []

        while not done:
            action = agent.act(state, eval_mode=False)
            next_state, reward, done = env_train.step(action)

            agent.remember(state, action, reward, next_state, done)
            loss, _eps = agent.step_update()
            if loss is not None:
                loss_acc.append(loss)

            state = next_state
            total_reward += reward

        episode_rewards.append(float(total_reward))
        episode_losses.append(float(np.mean(loss_acc)) if loss_acc else float("nan"))

        if e == 0:
            first_episode_history = list(env_train.history)
        if e == mid_index:
            mid_episode_history = list(env_train.history)
        if e == int(episodes) - 1:
            last_episode_history = list(env_train.history)

    # test (greedy)
    env_test = StockEnvironment(test_df, env_cfg)
    state = env_test.reset()
    done = False
    while not done:
        action = agent.act(state, eval_mode=True)
        next_state, reward, done = env_test.step(action)
        state = next_state

    history_df = pd.DataFrame(env_test.history)
    metrics, dfh, dd = compute_metrics(
        history_df,
        env_cfg.initial_balance,
        rf_annual=bt_cfg.rf_annual,
        trading_days=bt_cfg.trading_days,
    )

    metrics["Trade Volume"] = float(env_test.trade_volume)
    metrics["Turnover"] = float(env_test.trade_volume) / float(env_cfg.initial_balance)

    return {
        "ticker": ticker,
        "history_df": dfh,
        "drawdown": dd,
        "metrics": metrics,
        "train_curves": {"episode_reward": episode_rewards, "episode_loss": episode_losses},
        "first_ep": first_episode_history,
        "mid_ep": mid_episode_history,
        "last_ep": last_episode_history,
        "configs": {"env": asdict(env_cfg), "dqn": asdict(dqn_cfg), "backtest": asdict(bt_cfg), "seed": int(seed)},
    }
