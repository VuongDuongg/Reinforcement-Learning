"""Kiểm thử thuật toán chuẩn đối sánh Rule-based Baseline Agent."""

from __future__ import annotations

import unittest
import numpy as np

from highway_project.algorithms.baseline_agent import RuleBasedAgent
from highway_project.environment.environment import make_env


class BaselineAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = RuleBasedAgent(safe_gap=30.0, critical_gap=12.0, target_speed=25.0)

    def test_clear_road_accelerates(self):
        """Khi đường phía trước rất thoáng và tốc độ thấp, xe phải tăng tốc (Action 3)."""
        info = {
            "sensors": {
                "lane": 1,
                "lanes_count": 3,
                "speed": 18.0,
                "changing": False,
                "front_gap": [180.0, 100.0, 180.0],
                "front_closing": [0.0, 0.0, 0.0],
                "left_clear": True,
                "right_clear": True,
            },
            "available_actions": [0, 1, 2, 3, 4],
        }
        action = self.agent.act(np.zeros((10, 5)), info)
        self.assertEqual(action, 3, "Xe phải tăng tốc khi đường thoáng")

    def test_obstacle_ahead_changes_lane(self):
        """Khi gặp vật cản trong cự ly cảnh báo và làn trái thoáng, xe phải rẽ trái (Action 0)."""
        info = {
            "sensors": {
                "lane": 1,
                "lanes_count": 3,
                "speed": 22.0,
                "changing": False,
                "front_gap": [100.0, 25.0, 100.0],  # Làn 1 có vật cản ở 25m
                "front_closing": [0.0, 5.0, 0.0],
                "left_clear": True,
                "right_clear": True,
            },
            "available_actions": [0, 1, 2, 3, 4],
        }
        action = self.agent.act(np.zeros((10, 5)), info)
        self.assertEqual(action, 0, "Xe phải ưu tiên đổi sang làn trái để né vật cản")

    def test_obstacle_ahead_turns_right_if_left_blocked(self):
        """Khi làn trái bị chặn nhưng làn phải thoáng, xe phải rẽ phải (Action 2)."""
        info = {
            "sensors": {
                "lane": 1,
                "lanes_count": 3,
                "speed": 22.0,
                "changing": False,
                "front_gap": [10.0, 22.0, 100.0],  # Làn trái bị chặn
                "front_closing": [0.0, 4.0, 0.0],
                "left_clear": False,
                "right_clear": True,
            },
            "available_actions": [0, 1, 2, 3, 4],
        }
        action = self.agent.act(np.zeros((10, 5)), info)
        self.assertEqual(action, 2, "Xe phải đổi sang làn phải khi làn trái không an toàn")

    def test_emergency_braking_when_trapped(self):
        """Khi ở cự ly nguy hiểm cận kề mà cả 2 bên đều bị chặn, xe phải phanh khẩn cấp (Action 4)."""
        info = {
            "sensors": {
                "lane": 1,
                "lanes_count": 3,
                "speed": 25.0,
                "changing": False,
                "front_gap": [10.0, 8.0, 10.0],  # Khoảng cách chỉ còn 8m (< 12m)
                "front_closing": [0.0, 10.0, 0.0],
                "left_clear": False,
                "right_clear": False,
            },
            "available_actions": [0, 1, 2, 3, 4],
        }
        action = self.agent.act(np.zeros((10, 5)), info)
        self.assertEqual(action, 4, "Xe phải phanh khẩn cấp khi bị kẹt hoàn toàn")

    def test_baseline_runs_in_gym_env(self):
        """Kiểm thử chạy tích hợp thực tế 30 bước trong môi trường LaneObstacleEnv."""
        env = make_env(seed=42, config={"vehicles_count": 10, "obstacle_count": 1})
        self.addCleanup(env.close)
        obs, info = env.reset(seed=42)
        self.agent.reset(seed=42)

        total_reward = 0.0
        for _ in range(30):
            action = self.agent.act(obs, info)
            self.assertIn(action, [0, 1, 2, 3, 4])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        self.assertIsInstance(total_reward, float)


if __name__ == "__main__":
    unittest.main()
