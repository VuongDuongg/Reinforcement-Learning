"""Kiểm tra cập nhật Bellman, lưu mô hình và nội suy độc lập với FPS."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import math
import unittest

import numpy as np

from .environment import make_env
from .motion import MotionBuffer
from .q_agent import QLearningAgent


class LearningTests(unittest.TestCase):
    def test_terminal_does_not_bootstrap_but_timeout_does(self):
        agent = QLearningAgent(learning_rate=1.0, gamma=0.9)
        state = (1, 0, 2, 2, 1, 0)
        next_state = (1, 1, 2, 2, 1, 0)
        agent.q[next_state][1] = 10.0
        agent.learn(state, 1, -20, next_state, [1], terminated=True)
        self.assertEqual(agent.q[state][1], -20)
        agent.learn(state, 1, 1, next_state, [1], terminated=False)
        self.assertAlmostEqual(agent.q[state][1], 10)
        self.assertEqual(agent.training_steps, 2)

    def test_save_load_and_no_learning_during_inference(self):
        env = make_env(config={"vehicles_count": 0})
        self.addCleanup(env.close)
        obs, info = env.reset(seed=42)
        agent = QLearningAgent()
        state = agent.state(info)
        agent.q[state][0] = 3.5
        agent.visits[state] = 7
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=output_dir) as directory:
            path = Path(directory) / "agent.npz"
            agent.save(path)
            restored = QLearningAgent.load(path)
        before = restored.q.copy()
        self.assertEqual(restored.act(obs, info), 0)
        np.testing.assert_array_equal(before, restored.q)
        np.testing.assert_array_equal(restored.visits, agent.visits)

    def test_motion_interpolates_positions_and_shortest_heading(self):
        car = SimpleNamespace(position=np.array([0.0, 0.0]), heading=math.radians(179))
        env = SimpleNamespace(road=SimpleNamespace(vehicles=[car]), obstacles=[])
        motion = MotionBuffer()
        motion.reset(env)
        car.position[:] = [10, 4]
        car.heading = math.radians(-179)
        motion.capture(env)
        position, heading = motion.sample(car, 0.5)
        np.testing.assert_allclose(position, [5, 2])
        self.assertAlmostEqual(heading, math.pi)
        # Không sửa vị trí mô phỏng để làm mượt hình.
        np.testing.assert_array_equal(car.position, [10, 4])


if __name__ == "__main__":
    unittest.main()
