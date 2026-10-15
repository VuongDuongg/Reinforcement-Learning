"""
rl_bridge.py
Cầu nối trung gian giữa NetLogo và các thuật toán Python trong thư mục algorithms/
Nhóm N1_65KTPM - Đại học Thủy Lợi
"""

import os
import sys

# Đảm bảo đường dẫn import thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from algorithms.dqn_agent import DQNAgent
from algorithms.ddqn_agent import DoubleDuelingDQNAgent
from algorithms.ppo_agent import PPOAgent
from algorithms.sac_agent import SACAgent
from algorithms.baseline_agent import RuleBasedAgent

current_agent = None

def init_agent(policy_name, state_size=5, action_size=4, lr=0.001):
    """Khởi tạo tác nhân AI tương ứng với lựa chọn trên giao diện NetLogo"""
    global current_agent
    policy_name = str(policy_name).strip()
    
    if policy_name in ["Deep Q-Network", "Deep Q-Learning", "DQN"]:
        current_agent = DQNAgent(state_size=state_size, action_size=action_size, lr=lr)
        print(f"[RL-Bridge] Khởi tạo thành công: Deep Q-Network (Dương Quốc Vượng)")
    elif policy_name in ["Double Deep Q-Network", "Double Q-Learning", "DDQN"]:
        current_agent = DoubleDuelingDQNAgent(state_size=state_size, action_size=action_size, lr=lr)
        print(f"[RL-Bridge] Khởi tạo thành công: Double Dueling DQN (Ngô Tuấn Anh)")
    elif policy_name in ["Proximal Policy Optimization", "PPO"]:
        current_agent = PPOAgent(state_size=state_size, action_size=action_size, lr_actor=lr)
        print(f"[RL-Bridge] Khởi tạo thành công: Proximal Policy Optimization (Nguyễn Hồng Thái)")
    elif policy_name in ["Soft Actor-Critic", "SAC", "Naive Actor-Critic", "Advantage Actor-Critic"]:
        current_agent = SACAgent(state_size=state_size, action_size=action_size, lr=lr)
        print(f"[RL-Bridge] Khởi tạo thành công: Soft Actor-Critic (Lê Tuấn Khanh)")
    else:
        current_agent = RuleBasedAgent()
        print(f"[RL-Bridge] Khởi tạo thành công: Rule-based Baseline (Lê Tuấn Khanh)")
    return True

def select_actions(states, epsilon=0.05):
    """Nhận mảng trạng thái từ NetLogo và trả về mảng hành động cho các xe"""
    global current_agent
    if current_agent is None:
        return [1] * len(states)
    return current_agent.act(states, epsilon=float(epsilon))

def update_agent(states, actions, rewards, next_states, batch_size=64, gamma=0.99):
    """Nhận kinh nghiệm sau bước di chuyển từ NetLogo và kích hoạt huấn luyện"""
    global current_agent
    if current_agent is None:
        return 0.0

    batch_size = int(batch_size)
    gamma = float(gamma)

    if isinstance(current_agent, PPOAgent):
        current_agent.remember(states, actions, rewards, next_states)
        loss, _ = current_agent.train(epochs=2, batch_size=batch_size)
        return float(loss)
    elif hasattr(current_agent, "remember"):
        experiences = [(s, a, r, ns) for s, a, r, ns in zip(states, actions, rewards, next_states)]
        current_agent.remember(experiences)
        loss = current_agent.train(batch_size=batch_size, gamma=gamma)
        return float(loss)
    return 0.0
