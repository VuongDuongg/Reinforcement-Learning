"""Kiểm thử thuật toán 2: Double Dueling DQN (Ngô Tuấn Anh) bằng PyTorch."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..environment.environment import make_env


class DoubleDuelingDQNTests(unittest.TestCase):
    def test_dueling_network_forward_and_aggregation(self):
        from ..algorithms.ddqn_agent import DuelingQNetwork

        net = DuelingQNetwork(input_dim=68, hidden_dims=[128, 128], output_dim=5)
        dummy_input = torch.randn(4, 68, dtype=torch.float32)

        value, advantage = net.forward_streams(dummy_input)
        q_values = net(dummy_input)

        self.assertEqual(value.shape, (4, 1))
        self.assertEqual(advantage.shape, (4, 5))
        self.assertEqual(q_values.shape, (4, 5))
        self.assertTrue(torch.all(torch.isfinite(q_values)))

        # Kiểm tra đúng công thức Dueling: mean_a Q(s, a) == V(s)
        expected_q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        self.assertTrue(torch.allclose(q_values, expected_q, atol=1e-5))
        self.assertTrue(torch.allclose(q_values.mean(dim=1, keepdim=True), value, atol=1e-5))

    def test_ddqn_action_masking_respects_available_actions(self):
        from ..algorithms.ddqn_agent import DoubleDuelingDQNAgent

        agent = DoubleDuelingDQNAgent(state_dim=50, action_dim=5, epsilon=0.0)
        obs = np.zeros((10, 5), dtype=np.float32)
        info = {
            "available_actions": [1, 3],
            "sensors": {"changing": False},
        }
        for _ in range(10):
            action = agent.act(obs, info)
            self.assertIn(action, [1, 3])

    def test_ddqn_learn_and_soft_target_update(self):
        from ..algorithms.ddqn_agent import DoubleDuelingDQNAgent

        agent = DoubleDuelingDQNAgent(state_dim=50, action_dim=5, batch_size=4, lr=1e-3, gamma=0.98, tau=0.05)
        initial_target = [p.detach().clone() for p in agent.target_net.parameters()]

        for i in range(10):
            s = np.random.randn(50).astype(np.float32)
            s_next = np.random.randn(50).astype(np.float32)
            agent.buffer.push(s, i % 5, 1.0, s_next, terminated=(i % 2 == 0), truncated=False)

        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)
        self.assertEqual(agent.training_steps, 1)

        # Target network phải dịch chuyển mượt theo Soft Update (khác ban đầu nhưng không trùng hệt online)
        moved = any(
            not torch.allclose(p_old, p_new)
            for p_old, p_new in zip(initial_target, agent.target_net.parameters())
        )
        not_hard_copied = any(
            not torch.allclose(p_t, p_o)
            for p_t, p_o in zip(agent.target_net.parameters(), agent.q_net.parameters())
        )
        self.assertTrue(moved, "Soft Update phải cập nhật trọng số Target Network sau mỗi bước learn()")
        self.assertTrue(not_hard_copied, "Soft Update (tau < 1) không được hard-copy y nguyên Target Network")

    def test_ddqn_save_load_and_deterministic_inference(self):
        from ..algorithms.ddqn_agent import DoubleDuelingDQNAgent

        agent = DoubleDuelingDQNAgent(state_dim=68, action_dim=5, tau=0.005, epsilon=0.0)
        obs = np.random.randn(10, 5).astype(np.float32)
        info = {
            "available_actions": [0, 1, 2, 3, 4],
            "sensors": {"changing": False, "lane": 1, "lanes_count": 3},
        }

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ddqn_test.pth"
            agent.save(model_path, metadata={"test": True})
            self.assertTrue(model_path.exists())

            loaded = DoubleDuelingDQNAgent.load(model_path)
            loaded.epsilon = 0.0

            self.assertEqual(loaded.metadata.get("algorithm"), "double_dueling_dqn_pytorch")
            self.assertAlmostEqual(loaded.tau, 0.005)
            self.assertEqual(agent.act(obs, info), loaded.act(obs, info))

    def test_ddqn_dodges_obstacle_when_danger_detected(self):
        from ..algorithms.ddqn_agent import DoubleDuelingDQNAgent

        agent = DoubleDuelingDQNAgent(epsilon=0.0)
        obs = np.zeros((10, 5), dtype=np.float32)
        info = {
            "available_actions": [1, 2, 3, 4],
            "sensors": {
                "lane": 0,
                "lanes_count": 3,
                "speed": 20.0,
                "changing": False,
                "front_gap": [20.0, 180.0, 180.0],
                "rear_gap": [180.0, 180.0, 180.0],
                "front_closing": [20.0, 0.0, 0.0],
                "rear_closing": [0.0, 0.0, 0.0],
            },
        }
        action = agent.act(obs, info)
        self.assertIn(action, [2, 4])
        self.assertNotEqual(action, 1)

    def test_ddqn_agent_various_lanes_and_speeds(self):
        from ..algorithms.ddqn_agent import DoubleDuelingDQNAgent

        agent = DoubleDuelingDQNAgent(epsilon=0.0)
        for lanes in (2, 4, 5):
            env = make_env(config={"lanes_count": lanes, "initial_speed": 22.0, "max_speed": 32.0, "vehicles_count": "sparse"})
            obs, info = env.reset(seed=42)
            action = agent.act(obs, info)
            self.assertIn(action, range(5))
            obs, reward, term, trunc, info = env.step(action)
            self.assertFalse(term)
            env.close()


if __name__ == "__main__":
    unittest.main()
