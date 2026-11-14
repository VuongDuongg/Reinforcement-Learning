"""Kiểm tra hành vi môi trường, không cần cửa sổ hay thư viện học máy."""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from ..environment.environment import make_env


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

    def test_random_obstacles_placement_and_safe_spacing(self):
        env = self.create(vehicles_count=0, obstacle_count=2, route_length=1000.0, random_obstacles=True)
        env.reset(seed=42)
        self.assertEqual(len(env.unwrapped.obstacles), 2)
        obs1, obs2 = env.unwrapped.obstacles
        # Cả hai xe hỏng đều có làn hợp lệ 0..2
        self.assertIn(int(obs1.lane_index[2]), range(3))
        self.assertIn(int(obs2.lane_index[2]), range(3))
        # Không quá dày đặc: khoảng cách dọc tuyến đường cách nhau ít nhất 150m
        dist_between = abs(float(obs2.position[0] - obs1.position[0]))
        self.assertGreaterEqual(dist_between, 150.0)
        # Cách điểm xuất phát ít nhất 100m
        self.assertGreater(float(obs1.position[0] - env.unwrapped.start_x), 100.0)

    def test_dynamic_lanes_traffic_presets_and_speed_config(self):
        # 1. Test 2 làn, preset mật độ thưa thớt (sparse), vận tốc ban đầu 25 m/s (~90 km/h)
        env_sparse = self.create(lanes_count=2, vehicles_count="sparse", initial_speed=25.0)
        obs, info = env_sparse.reset(seed=42)
        sensors = info["sensors"]
        self.assertEqual(sensors["lanes_count"], 2)
        self.assertEqual(env_sparse.unwrapped.config["vehicles_count"], 8)
        self.assertEqual(sensors["traffic_density"], "sparse")
        self.assertEqual(len(sensors["front_gap"]), 2)
        self.assertEqual(len(sensors["rear_gap"]), 2)
        self.assertAlmostEqual(sensors["speed"], 25.0, places=1)
        self.assertAlmostEqual(info["speed_kmh"], 90.0, delta=2.0)
        self.assertIn("ego_front_gap", sensors)
        self.assertIn("left_clear", sensors)
        self.assertIn("right_clear", sensors)

        # 2. Test 5 làn, preset mật độ dày đặc (dense = 38 xe), dải tốc độ giao thông tùy biến
        env_dense = self.create(lanes_count=5, vehicles_count="dense", traffic_speed_min=12.0, traffic_speed_max=18.0)
        obs2, info2 = env_dense.reset(seed=42)
        sensors2 = info2["sensors"]
        self.assertEqual(sensors2["lanes_count"], 5)
        self.assertEqual(env_dense.unwrapped.config["vehicles_count"], 38)
        self.assertEqual(sensors2["traffic_density"], "dense")
        self.assertEqual(len(sensors2["front_gap"]), 5)
        self.assertEqual(len(sensors2["rear_gap"]), 5)

    def test_view_renders_various_lanes_and_traffic(self):
        from ..gui.view import HighwayView
        for lanes, traffic in [(2, "sparse"), (4, "normal"), (5, "dense")]:
            env = self.create(lanes_count=lanes, vehicles_count=traffic)
            obs, info = env.reset(seed=42)
            view = HighwayView(headless=True)
            view.draw(env.unwrapped, info, paused=False, mode="agent", last_action="IDLE", total_reward=0.0, episode=1)

    def test_parse_traffic_and_cli_options(self):
        from ..run import parse_traffic
        self.assertEqual(parse_traffic("sparse"), "sparse")
        self.assertEqual(parse_traffic("thua"), "sparse")
        self.assertEqual(parse_traffic("normal"), "normal")
        self.assertEqual(parse_traffic("vua"), "normal")
        self.assertEqual(parse_traffic("dense"), "dense")
        self.assertEqual(parse_traffic("day"), "dense")
        self.assertEqual(parse_traffic("none"), "none")
        self.assertEqual(parse_traffic("15"), "15")


if __name__ == "__main__":
    unittest.main()
