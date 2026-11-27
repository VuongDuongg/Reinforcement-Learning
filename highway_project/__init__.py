"""highway_project: Gói bài toán học tăng cường điều khiển xe tự hành trên cao tốc."""

from .environment.environment import LaneObstacleEnv, make_env, ACTIONS, ENV_ID, BrokenCar
from .algorithms.agent import CustomAgent
from .algorithms.dqn_agent import DQNAgent, QNetwork, ReplayBuffer, DEFAULT_MODEL as DEFAULT_DQN_MODEL
from .algorithms.ddqn_agent import DoubleDuelingDQNAgent, DuelingQNetwork, DEFAULT_MODEL as DEFAULT_DDQN_MODEL
from .algorithms.q_agent import QLearningAgent, DEFAULT_MODEL as DEFAULT_Q_MODEL
from .gui.view import HighwayView
from .gui.motion import MotionBuffer

__all__ = [
    "LaneObstacleEnv",
    "make_env",
    "ACTIONS",
    "ENV_ID",
    "BrokenCar",
    "CustomAgent",
    "DQNAgent",
    "QNetwork",
    "ReplayBuffer",
    "DEFAULT_DQN_MODEL",
    "DoubleDuelingDQNAgent",
    "DuelingQNetwork",
    "DEFAULT_DDQN_MODEL",
    "QLearningAgent",
    "DEFAULT_Q_MODEL",
    "HighwayView",
    "MotionBuffer",
]
