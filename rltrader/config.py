from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingEnvConfig:
    initial_balance: float = 10_000.0
    trade_size: int = 1
    fixed_cost: float = 0.05
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    reward_scale: float = 1.0  # if 1.0, will be set to 1/initial_balance automatically


@dataclass(frozen=True)
class DQNConfig:
    state_size: int = 3
    action_size: int = 3
    hidden_size: int = 64

    gamma: float = 0.99
    learning_rate: float = 1e-3

    # replay
    buffer_size: int = 50_000
    batch_size: int = 64
    min_buffer_size: int = 1_000
    train_every: int = 1

    # target net
    target_update_every: int = 1_000
    double_dqn: bool = True

    # exploration (linear decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000  # 0 => constant epsilon


@dataclass(frozen=True)
class BacktestConfig:
    train_ratio: float = 0.7
    rf_annual: float = 0.02
    trading_days: int = 252
