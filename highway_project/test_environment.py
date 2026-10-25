"""Kiểm tra hành vi môi trường, không cần cửa sổ hay thư viện học máy."""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from .environment import make_env


class EnvironmentTests(unittest.TestCase):
    def create(self, **config):
        env = make_env(config=config)
        self.addCleanup(env.close)
        return env

    def test_gymnasium_contract_and_seed(self):
        env = self.create()
        check_env(env.unwrapped, skip_render_check=True)
        first, _ = env.reset(seed=17)
        positions = np.array([car.position.copy() for car in env.unwrapped.road.vehicles])
        second, _ = env.reset(seed=17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(positions, [car.position for car in env.unwrapped.road.vehicles])
        self.assertEqual(first.shape, (10, 5))
        self.assertTrue(env.observation_space.contains(first))

    def test_obstacle_visible_stationary_and_collision_terminates(self):
        env = self.create(vehicles_count=0, obstacle_distance=80.0)
        obs, _ = env.reset(seed=1)
        self.assertEqual(int(obs[:, 0].sum()), 2, "Phải nhìn thấy ego và vật cản")
        position = env.unwrapped.obstacles[0].position.copy()
        for _ in range(100):
            _, reward, terminated, truncated, info = env.step(1)
            np.testing.assert_array_equal(position, env.unwrapped.obstacles[0].position)
            if terminated or truncated:
                break
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["crashed"])
        self.assertEqual(info["end_reason"], "collision")
        self.assertLess(reward, 0)

    def test_lane_change_can_pass_obstacle_and_reach_goal(self):
        env = self.create(vehicles_count=0, obstacle_distance=80.0, route_length=160.0)
        env.reset(seed=1)
        # Một lệnh chuyển làn, sau đó chỉ giữ tốc; đây là kịch bản kiểm tra, không phải agent.
        _, _, terminated, truncated, info = env.step(0)
        for _ in range(150):
            if terminated or truncated:
                break
            _, _, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["obstacles_passed"], 1)
        self.assertEqual(info["end_reason"], "success")

    def test_timeout_is_truncation_and_not_success(self):
        env = self.create(vehicles_count=0, duration=0.2)
        env.reset(seed=1)
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(1)
            if terminated or truncated:
                break
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["end_reason"], "time_limit")

    def test_idle_keeps_speed_and_rewards_match_info(self):
        env = self.create(vehicles_count=0)
        env.reset(seed=1)
        for _ in range(10):
            _, reward, _, _, info = env.step(1)
            self.assertAlmostEqual(info["speed"], 20.0)
            self.assertAlmostEqual(reward, sum(info["rewards"].values()))
        _, _, _, _, info = env.step(4)
        self.assertLess(info["speed"], 20.0)


if __name__ == "__main__":
    unittest.main()
