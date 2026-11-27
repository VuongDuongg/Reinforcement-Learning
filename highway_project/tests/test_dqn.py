"""Kiểm thử thuật toán Deep Q-Network gốc (DQN 2013 - không Target Network) bằng PyTorch."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ..environment.environment import make_env


class DQNTests(unittest.TestCase):
    def test_q_network_forward_and_shapes(self):
        from ..algorithms.dqn_agent import QNetwork

        net = QNetwork(input_dim=50, hidden_dims=[128, 128], output_dim=5)
        dummy_input = torch.randn(4, 50, dtype=torch.float32)
        out = net(dummy_input)
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_replay_buffer_sampling_and_capacity(self):
        from ..algorithms.dqn_agent import ReplayBuffer

        buffer = ReplayBuffer(capacity=10, state_dim=50)
        for i in range(15):
            s = np.zeros(50, dtype=np.float32) + i
            buffer.push(s, i % 5, float(i), s + 1, terminated=False, truncated=False)
        self.assertEqual(len(buffer), 10)

        states, actions, rewards, next_states, terminateds = buffer.sample(batch_size=4)
        self.assertEqual(states.shape, (4, 50))
        self.assertEqual(actions.shape, (4, 1))
        self.assertEqual(rewards.shape, (4, 1))
        self.assertEqual(next_states.shape, (4, 50))
        self.assertEqual(terminateds.shape, (4, 1))

    def test_dqn_action_masking_respects_available_actions(self):
        from ..algorithms.dqn_agent import DQNAgent

        agent = DQNAgent(state_dim=50, action_dim=5, epsilon=0.0)
        obs = np.zeros((10, 5), dtype=np.float32)
        # Chỉ cho phép hành động 1 (IDLE) và 3 (FASTER)
        info = {
            "available_actions": [1, 3],
            "sensors": {"changing": False},
        }
        for _ in range(10):
            action = agent.act(obs, info)
            self.assertIn(action, [1, 3])

    def test_dqn_learn_and_bellman_target(self):
        from ..algorithms.dqn_agent import DQNAgent

        agent = DQNAgent(state_dim=50, action_dim=5, batch_size=4, lr=1e-3, gamma=0.9)
        self.assertFalse(hasattr(agent, "target_net"))
        self.assertIsInstance(agent.criterion, torch.nn.MSELoss)
        # Bơm 10 transitions vào buffer
        for i in range(10):
            s = np.random.randn(50).astype(np.float32)
            s_next = np.random.randn(50).astype(np.float32)
            agent.buffer.push(s, 1, 1.0, s_next, terminated=(i % 2 == 0), truncated=False)

        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)

    def test_dqn_save_load_and_deterministic_inference(self):
        from ..algorithms.dqn_agent import DQNAgent

        agent = DQNAgent(state_dim=50, action_dim=5, epsilon=0.0)
        obs = np.random.randn(10, 5).astype(np.float32)
        info = {
            "available_actions": [0, 1, 2, 3, 4],
            "sensors": {"changing": False},
        }

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "dqn_test.pth"
            agent.save(model_path, metadata={"test": True})
            self.assertTrue(model_path.exists())

            loaded_agent = DQNAgent.load(model_path)
            loaded_agent.epsilon = 0.0
            self.assertEqual(loaded_agent.metadata.get("algorithm"), "dqn_2013_pytorch")

            action_before = agent.act(obs, info)
            action_after = loaded_agent.act(obs, info)
            self.assertEqual(action_before, action_after)

    def test_custom_agent_integration(self):
        from ..algorithms.agent import CustomAgent

        agent = CustomAgent()
        obs = np.zeros((10, 5), dtype=np.float32)
        info = {
            "available_actions": [0, 1, 2, 3, 4],
            "sensors": {"changing": False},
        }
        agent.reset(seed=123)
        action = agent.act(obs, info)
        self.assertIn(action, range(5))
        # Test observe không gây lỗi
        agent.observe(obs, action, 1.0, obs, terminated=False, truncated=False, info=info)

    def test_dqn_dodges_obstacle_when_danger_detected(self):
        from ..algorithms.dqn_agent import DQNAgent

        agent = DQNAgent(epsilon=0.0)
        obs = np.zeros((10, 5), dtype=np.float32)
        # Ego đang ở Làn 0, phía trước 20m có vật cản hỏng tĩnh, làn 1 thông thoáng
        info = {
            "available_actions": [1, 2, 3, 4],  # Làn 0 không được rẽ trái (0)
            "sensors": {
                "lane": 0,
                "speed": 20.0,
                "changing": False,
                "front_gap": [20.0, 180.0, 180.0],
                "rear_gap": [180.0, 180.0, 180.0],
                "front_closing": [20.0, 0.0, 0.0],
                "rear_closing": [0.0, 0.0, 0.0],
            },
        }
        action = agent.act(obs, info)
        # Bắt buộc phải né: chọn rẽ phải sang làn 1 (2) hoặc hãm phanh (4), KHÔNG được đi thẳng (1) hay tăng tốc (3)!
        self.assertIn(action, [2, 4], "AI phải né sang phải hoặc giảm tốc khi gặp vật cản trực diện!")
        self.assertNotEqual(action, 1, "AI không được đi thẳng đâm vào vật cản!")

    def test_dqn_agent_various_lanes_and_speeds(self):
        from ..algorithms.dqn_agent import DQNAgent

        agent = DQNAgent.load()
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

