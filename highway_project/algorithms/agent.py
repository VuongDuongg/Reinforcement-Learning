"""Cầu nối thuật toán Deep Q-Network (DQN) với môi trường Highway."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from .dqn_agent import DEFAULT_MODEL, DQNAgent


class CustomAgent:
    """Agent điều khiển xe tự hành tích hợp mạng nơ-ron Deep Q-Network."""

    def __init__(self, model_path: Path | str | None = None, *, seed: int = 42):
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if path.exists():
            self.agent = DQNAgent.load(path)
            self.agent.epsilon = 0.0  # Chế độ suy luận
        else:
            self.agent = DQNAgent(seed=seed, epsilon=0.0)

    def reset(self, *, seed: int = 42) -> None:
        """Reset bộ nhớ hoặc trạng thái seed cho thuật toán."""
        self.agent.reset(seed=seed)

    def act(self, observation: np.ndarray, info: dict) -> int:
        """Suy luận hành động tối ưu từ mạng Deep Q-Network. Trả về số nguyên 0..4.

        observation: float32, shape (10, 5), chuẩn hóa theo HighwayEnv.
        0: rẽ trái, 1: giữ tốc/giữ làn, 2: rẽ phải, 3: tăng tốc, 4: giảm tốc.
        """
        return self.agent.act(observation, info)

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> None:
        """Ghi nhận kinh nghiệm và cập nhật nếu agent trong chế độ học tập."""
        self.agent.observe(
            observation,
            action,
            reward,
            next_observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
