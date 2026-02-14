from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .config import TradingEnvConfig


class StockEnvironment:
    """
    State: [daily return, position flag, bias]
    Action: 0=hold, 1=buy, 2=sell

    Execution model:
      - buy:  exec_price = price * (1 + slippage_bps)
      - sell: exec_price = price * (1 - slippage_bps)
      - transaction cost = fixed_cost + cost_bps/10000 * exec_price * shares
      - balance is updated with exec_price and cost

    Reward:
      reward = (net_worth - prev_net_worth) * reward_scale
    """

    def __init__(self, data: pd.DataFrame, cfg: TradingEnvConfig):
        if data is None or data.empty:
            raise ValueError("Environment data is empty.")
        if "Date" not in data.columns or "Close" not in data.columns:
            raise ValueError("Data must contain Date and Close columns.")
        if int(cfg.trade_size) <= 0:
            raise ValueError("trade_size must be positive.")

        self.data = data.reset_index(drop=True)
        self.cfg = cfg
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_index = 0
        self.balance = float(self.cfg.initial_balance)
        self.shares = 0
        self.net_worth = float(self.cfg.initial_balance)
        self.trade_volume = 0.0
        self.trade_count = 0
        self.history: List[Dict] = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        if self.step_index >= len(self.data):
            return np.zeros(3, dtype=np.float32)

        price = float(self.data.loc[self.step_index, "Close"])

        if self.step_index > 0:
            prev_price = float(self.data.loc[self.step_index - 1, "Close"])
            daily_ret = (price - prev_price) / prev_price if prev_price != 0 else 0.0
        else:
            daily_ret = 0.0

        has_position = 1.0 if self.shares > 0 else 0.0
        return np.array([float(daily_ret), float(has_position), 1.0], dtype=np.float32)

    def _transaction_cost(self, exec_price: float, shares: int) -> float:
        prop = (self.cfg.cost_bps / 10_000.0) * exec_price * shares
        return float(self.cfg.fixed_cost + prop)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        price = float(self.data.loc[self.step_index, "Close"])
        prev_net_worth = float(self.net_worth)

        slip = self.cfg.slippage_bps / 10_000.0
        exec_price = price
        cost_paid = 0.0
        executed = False

        if action == 1:  # buy
            exec_price = price * (1.0 + slip)
            shares = int(self.cfg.trade_size)
            cost_paid = self._transaction_cost(exec_price, shares)
            total_cash_needed = exec_price * shares + cost_paid
            if self.balance >= total_cash_needed:
                self.balance -= total_cash_needed
                self.shares += shares
                self.trade_volume += exec_price * shares
                self.trade_count += 1
                executed = True

        elif action == 2:  # sell
            exec_price = price * (1.0 - slip)
            shares = min(int(self.cfg.trade_size), int(self.shares))
            if shares > 0:
                cost_paid = self._transaction_cost(exec_price, shares)
                proceeds = exec_price * shares - cost_paid
                self.balance += proceeds
                self.shares -= shares
                self.trade_volume += exec_price * shares
                self.trade_count += 1
                executed = True

        # mark-to-market at close price
        self.net_worth = float(self.balance + self.shares * price)

        reward = (self.net_worth - prev_net_worth) * float(self.cfg.reward_scale)

        self.history.append(
            {
                "step": int(self.step_index),
                "date": self.data.loc[self.step_index, "Date"],
                "price": float(price),
                "exec_price": float(exec_price),
                "action": int(action),
                "executed": bool(executed),
                "shares": int(self.shares),
                "balance": float(self.balance),
                "net_worth": float(self.net_worth),
                "cost_paid": float(cost_paid),
                "reward": float(reward),
            }
        )

        self.step_index += 1
        done = self.step_index >= len(self.data) - 1
        next_state = self._get_state()
        return next_state, float(reward), bool(done)
