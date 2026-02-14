from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .config import DQNConfig


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, seed: int = 42):
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self._ptr = 0
        self._size = 0
        self.rng = np.random.default_rng(int(seed))

        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        i = self._ptr
        self.states[i] = s
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_states[i] = s2
        self.dones[i] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._size < batch_size:
            raise ValueError("Not enough samples in buffer.")
        idx = self.rng.integers(0, self._size, size=int(batch_size))
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class DQNAgent:
    """
    Canonical DQN components:
      - Experience Replay
      - Target Network (hard update)
      - Optional Double DQN
    """

    def __init__(self, cfg: DQNConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        # online net
        self.W1 = (self.rng.standard_normal((cfg.state_size, cfg.hidden_size)) * 0.1).astype(np.float32)
        self.b1 = np.zeros((cfg.hidden_size,), dtype=np.float32)
        self.W2 = (self.rng.standard_normal((cfg.hidden_size, cfg.action_size)) * 0.1).astype(np.float32)
        self.b2 = np.zeros((cfg.action_size,), dtype=np.float32)

        # target net
        self.tW1 = self.W1.copy()
        self.tb1 = self.b1.copy()
        self.tW2 = self.W2.copy()
        self.tb2 = self.b2.copy()

        self.buffer = ReplayBuffer(cfg.buffer_size, cfg.state_size, seed=int(seed))

        self.global_step = 0
        self.epsilon = float(cfg.epsilon_start)
        self._eps_slope = 0.0
        if cfg.epsilon_decay_steps and cfg.epsilon_decay_steps > 0:
            self._eps_slope = (cfg.epsilon_start - cfg.epsilon_end) / float(cfg.epsilon_decay_steps)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    def _forward(self, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z1 = S @ self.W1 + self.b1
        H = self._relu(Z1)
        Q = H @ self.W2 + self.b2
        return H, Q

    def _forward_target(self, S: np.ndarray) -> np.ndarray:
        Z1 = S @ self.tW1 + self.tb1
        H = self._relu(Z1)
        Q = H @ self.tW2 + self.tb2
        return Q

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(self.cfg.action_size))
        _, Q = self._forward(state.reshape(1, -1).astype(np.float32))
        return int(np.argmax(Q[0]))

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        self.buffer.add(s.astype(np.float32), int(a), float(r), s2.astype(np.float32), bool(done))

    def update_target(self) -> None:
        self.tW1 = self.W1.copy()
        self.tb1 = self.b1.copy()
        self.tW2 = self.W2.copy()
        self.tb2 = self.b2.copy()

    def _update_epsilon(self) -> None:
        if self.cfg.epsilon_decay_steps and self.cfg.epsilon_decay_steps > 0:
            self.epsilon = max(self.cfg.epsilon_end, self.epsilon - self._eps_slope)

    def train_step(self) -> Optional[float]:
        cfg = self.cfg
        if len(self.buffer) < cfg.min_buffer_size or len(self.buffer) < cfg.batch_size:
            return None

        S, A, R, S2, D = self.buffer.sample(cfg.batch_size)

        # Q(s,a)
        H, Q = self._forward(S)
        q_sa = Q[np.arange(cfg.batch_size), A]

        # target Q(s',·)
        Q_next_target = self._forward_target(S2)
        if cfg.double_dqn:
            _, Q_next_online = self._forward(S2)
            a_star = np.argmax(Q_next_online, axis=1)
            next_q = Q_next_target[np.arange(cfg.batch_size), a_star]
        else:
            next_q = np.max(Q_next_target, axis=1)

        y = R + cfg.gamma * (1.0 - D) * next_q

        err = (q_sa - y).astype(np.float32)
        loss = float(0.5 * np.mean(err ** 2))

        dQ = np.zeros_like(Q, dtype=np.float32)
        dQ[np.arange(cfg.batch_size), A] = err / cfg.batch_size

        dW2 = H.T @ dQ
        db2 = np.sum(dQ, axis=0)

        dH = dQ @ self.W2.T
        dZ1 = dH * (H > 0.0)

        dW1 = S.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        # clip global grad norm
        grad_norm = float(
            math.sqrt(np.sum(dW1**2) + np.sum(db1**2) + np.sum(dW2**2) + np.sum(db2**2)) + 1e-8
        )
        max_norm = 10.0
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            dW1 *= scale
            db1 *= scale
            dW2 *= scale
            db2 *= scale

        lr = cfg.learning_rate
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return loss

    def step_update(self) -> Tuple[Optional[float], float]:
        self.global_step += 1
        loss = None

        if self.global_step % self.cfg.train_every == 0:
            loss = self.train_step()

        if cfg := self.cfg:
            if cfg.target_update_every > 0 and self.global_step % cfg.target_update_every == 0:
                self.update_target()

        self._update_epsilon()
        return loss, float(self.epsilon)
