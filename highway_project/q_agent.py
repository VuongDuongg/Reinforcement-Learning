"""Q-learning dạng bảng: học từ reward, không viết sẵn luật chọn hành động né."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).parent / "models" / "q_learning.npz"
STATE_SHAPE = (3, 4, 3, 3, 3, 2)


class QLearningAgent:
    label = "Q-LEARNING · chính sách đã học"

    def __init__(self, *, seed=42, epsilon=0.0, learning_rate=0.15, gamma=0.98):
        self.rng = np.random.default_rng(seed)
        self.epsilon, self.learning_rate, self.gamma = epsilon, learning_rate, gamma
        self.q = np.zeros((*STATE_SHAPE, 5), dtype=np.float32)
        self.visits = np.zeros(STATE_SHAPE, dtype=np.int32)
        self.last_state = None
        self.training_steps = 0

    @staticmethod
    def state(info):
        sensor = info["sensors"]
        lane, speed = sensor["lane"], sensor["speed"]
        gap, closing = sensor["front_gap"][lane], sensor["front_closing"][lane]
        ttc = gap / max(closing, 0.01)
        front = 3 if gap < 10 or ttc < 1.5 else 2 if gap < 25 or ttc < 3 else 1 if gap < 70 or ttc < 6 else 0

        def side(lane_id):
            if not 0 <= lane_id < 3:
                return 0  # Mép đường.
            front_gap, rear_gap = sensor["front_gap"][lane_id], sensor["rear_gap"][lane_id]
            front_ttc = front_gap / max(sensor["front_closing"][lane_id], 0.01)
            rear_ttc = rear_gap / max(sensor["rear_closing"][lane_id], 0.01)
            return 2 if front_gap > 20 and rear_gap > 15 and min(front_ttc, rear_ttc) > 3 else 1

        return (lane, front, side(lane - 1), side(lane + 1),
                0 if speed < 15 else 1 if speed < 25 else 2, int(sensor["changing"]))

    @staticmethod
    def available(info):
        actions = list(info["available_actions"])
        # Chờ hoàn tất chuyển làn rồi mới cho yêu cầu chuyển làn tiếp;
        # không loại hành động theo nguy cơ va chạm (việc né phải được học).
        if info["sensors"]["changing"]:
            actions = [a for a in actions if a not in (0, 2)]
        return actions

    def reset(self, *, seed):
        self.last_state = None

    def act(self, observation, info):
        state = self.state(info)
        self.last_state = state
        actions = self.available(info)
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        # Khi bằng điểm ưu tiên giữ tốc; không cài sẵn hành vi né.
        preference = [a for a in (1, 3, 4, 0, 2) if a in actions]
        return max(preference, key=lambda action: float(self.q[state][action]))

    def learn(self, state, action, reward, next_state, next_actions, *, terminated):
        target = float(reward)
        if not terminated:
            target += self.gamma * float(np.max(self.q[next_state][next_actions]))
        self.q[state][action] += self.learning_rate * (target - self.q[state][action])
        self.visits[state] += 1
        self.training_steps += 1

    def observe(self, observation, action, reward, next_observation, terminated, truncated, info):
        if self.last_state is not None:
            self.learn(self.last_state, action, reward, self.state(info), self.available(info), terminated=terminated)

    def save(self, path=DEFAULT_MODEL, *, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=self.q, visits=self.visits,
                            metadata=json.dumps({"algorithm": "tabular_q_learning", "schema": 1,
                                                 "training_steps": self.training_steps, **(metadata or {})}))

    @classmethod
    def load(cls, path=DEFAULT_MODEL):
        agent = cls()
        with np.load(path, allow_pickle=False) as model:
            if model["q"].shape != agent.q.shape:
                raise ValueError("Kích thước bảng Q không phù hợp phiên bản môi trường.")
            agent.q = model["q"].copy()
            agent.visits = model["visits"].copy()
            agent.metadata = json.loads(str(model["metadata"]))
            agent.training_steps = agent.metadata["training_steps"]
        return agent
