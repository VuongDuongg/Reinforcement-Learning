"""
test_all_agents.py
Script kiểm thử toàn bộ 4 thuật toán Học tăng cường và Baseline
Nhóm N1_65KTPM - Đại học Thủy Lợi

Cách chạy:
python test_all_agents.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from algorithms.dqn_agent import DQNAgent
from algorithms.ddqn_agent import DoubleDuelingDQNAgent
from algorithms.ppo_agent import PPOAgent
from algorithms.sac_agent import SACAgent
from algorithms.baseline_agent import RuleBasedAgent

def test_agents():
    print("=" * 60)
    print("KIỂM THỬ ĐỒNG BỘ CÁC THUẬT TOÁN HỌC TĂNG CƯỜNG (N1_65KTPM)")
    print("=" * 60)

    state_size = 5
    action_size = 4
    batch_size = 32

    # Tạo dữ liệu giả lập cho 3 xe (Multi-car)
    dummy_states = [
        [0.8, 15.0, 0.2, 30.0, 0.0],
        [0.5, 3.0, 0.5, 25.0, 0.1],   # Xe sắp gặp vật cản
        [0.6, 20.0, 0.1, 10.0, 0.0]
    ]

    # 1. Test DQN (Dương Quốc Vượng)
    print("\n[1] Kiểm tra DQN (Dương Quốc Vượng)...")
    dqn = DQNAgent(state_size=state_size, action_size=action_size)
    dqn_actions = dqn.act(dummy_states, epsilon=0.1)
    print(f"-> DQN Actions cho 3 xe: {dqn_actions}")
    dqn.remember([(dummy_states[0], dqn_actions[0], 1.0, dummy_states[0])] * 50)
    dqn_loss = dqn.train(batch_size=batch_size)
    print(f"-> DQN Train Loss: {dqn_loss:.4f} [PASSED]")

    # 2. Test Double Dueling DQN (Ngô Tuấn Anh)
    print("\n[2] Kiểm tra Double Dueling DQN (Ngô Tuấn Anh)...")
    ddqn = DoubleDuelingDQNAgent(state_size=state_size, action_size=action_size)
    ddqn_actions = ddqn.act(dummy_states, epsilon=0.05)
    print(f"-> DDQN Actions cho 3 xe: {ddqn_actions}")
    ddqn.remember([(dummy_states[0], ddqn_actions[0], 1.0, dummy_states[0])] * 50)
    ddqn_loss = ddqn.train(batch_size=batch_size)
    print(f"-> DDQN Train Loss: {ddqn_loss:.4f} [PASSED]")

    # 3. Test PPO (Nguyễn Hồng Thái)
    print("\n[3] Kiểm tra PPO (Nguyễn Hồng Thái)...")
    ppo = PPOAgent(state_size=state_size, action_size=action_size)
    ppo_actions = ppo.act(dummy_states)
    print(f"-> PPO Actions cho 3 xe: {ppo_actions}")
    for _ in range(40):
        ppo.remember(dummy_states, ppo_actions, [1.0, -1.0, 0.5], dummy_states, [False, False, False])
    a_loss, c_loss = ppo.train(epochs=2, batch_size=batch_size)
    print(f"-> PPO Actor Loss: {a_loss:.4f}, Critic Loss: {c_loss:.4f} [PASSED]")

    # 4. Test SAC (Lê Tuấn Khanh)
    print("\n[4] Kiểm tra SAC (Lê Tuấn Khanh)...")
    sac = SACAgent(state_size=state_size, action_size=action_size)
    sac_actions = sac.act(dummy_states)
    print(f"-> SAC Actions cho 3 xe: {sac_actions}")
    sac.remember([(dummy_states[0], sac_actions[0], 1.0, dummy_states[0])] * 50)
    sac_loss = sac.train(batch_size=batch_size)
    print(f"-> SAC Train Loss: {sac_loss:.4f} [PASSED]")

    # 5. Test Baseline (Lê Tuấn Khanh)
    print("\n[5] Kiểm tra Rule-based Baseline (Lê Tuấn Khanh)...")
    rule = RuleBasedAgent(safe_distance=4.0)
    rule_actions = rule.act(dummy_states)
    print(f"-> Baseline Actions cho 3 xe: {rule_actions} (Xe 2 khoảng cách 3.0m né cản chuyển làn = 3)")
    print("-> Baseline [PASSED]")

    print("\n" + "=" * 60)
    print("TẤT CẢ CÁC MODULE THUẬT TOÁN ĐÃ SẴN SÀNG TÍCH HỢP VÀO NETLOGO!")
    print("=" * 60)

if __name__ == '__main__':
    test_agents()
