"""
algorithms/dqn_agent.py
Thuật toán 1: Deep Q-Network (DQN)
Thành viên phụ trách: Dương Quốc Vượng (MSV: 2351170634)
Nhóm: N1_65KTPM - Đại học Thủy Lợi
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os

class DQNAgent:
    """
    Tác nhân Deep Q-Network với Experience Replay Buffer và Target Network.
    Hỗ trợ cả chế độ điều khiển đơn tác nhân (single-car) và đa tác nhân (multi-car).
    """
    def __init__(self, state_size=5, action_size=4, hidden_size=64, 
                 lr=0.0005, memory_size=50000, update_steps=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.memory_size = memory_size
        self.update_steps = update_steps
        self.step_count = 0

        # Bộ nhớ phát lại kinh nghiệm (Replay Buffer)
        self.memory = []

        # Xây dựng mạng Q-Network chính và mạng Target
        self.q_network = self._build_model()
        self.target_network = keras.models.clone_model(self.q_network)
        self.target_network.set_weights(self.q_network.get_weights())

    def _build_model(self):
        """Khởi tạo kiến trúc mạng nơ-ron sâu xấp xỉ giá trị Q(s, a)"""
        model = keras.Sequential([
            layers.Dense(self.hidden_size, input_shape=(self.state_size,), activation='relu'),
            layers.Dense(self.hidden_size, activation='relu'),
            layers.Dense(self.hidden_size, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def act(self, states, epsilon=0.05):
        """
        Lựa chọn hành động theo chính sách Epsilon-Greedy.
        Hỗ trợ đầu vào là mảng nhiều trạng thái của nhiều xe cùng lúc.
        """
        states_arr = np.array(states, dtype=np.float32)
        if len(states_arr.shape) == 1:
            states_arr = np.expand_dims(states_arr, axis=0)

        num_agents = states_arr.shape[0]
        q_values = self.q_network.predict(states_arr, verbose=0)
        best_actions = np.argmax(q_values, axis=1)

        final_actions = []
        for i in range(num_agents):
            if np.random.rand() < epsilon:
                # Khám phá: chọn ngẫu nhiên hành động
                final_actions.append(int(np.random.randint(self.action_size)))
            else:
                # Khai thác: chọn hành động có Q-value lớn nhất
                final_actions.append(int(best_actions[i]))

        return final_actions

    def remember(self, experiences):
        """
        Lưu các bộ chuyển tiếp mới vào Replay Buffer.
        experiences: list of tuples (state, action, reward, next_state)
        """
        for exp in experiences:
            self.memory.append(exp)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

    def train(self, batch_size=64, gamma=0.99):
        """Huấn luyện mạng Q-Network từ một mini-batch ngẫu nhiên trong Buffer"""
        if len(self.memory) < batch_size:
            return 0.0

        sample_ix = np.random.randint(len(self.memory), size=batch_size)
        states = np.array([self.memory[i][0] for i in sample_ix], dtype=np.float32)
        actions = np.array([self.memory[i][1] for i in sample_ix], dtype=np.int32)
        rewards = np.array([self.memory[i][2] for i in sample_ix], dtype=np.float32)
        next_states = np.array([self.memory[i][3] for i in sample_ix], dtype=np.float32)

        # Tính toán Q-values hiện tại
        targets = self.q_network.predict(states, verbose=0)
        
        # DQN cổ điển: dùng Target Network để lấy max Q ở trạng thái kế tiếp
        next_q_values = self.target_network.predict(next_states, verbose=0)
        max_next_q = np.max(next_q_values, axis=1)

        # Cập nhật giá trị mục tiêu (Bellman Target)
        targets[np.arange(batch_size), actions] = rewards + gamma * max_next_q

        # Tối ưu hóa trọng số mạng qua mini-batch
        loss = self.q_network.train_on_batch(states, targets)

        # Cập nhật định kỳ trọng số sang mạng Target
        self.step_count += 1
        if self.step_count % self.update_steps == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        return float(loss)

    def save(self, filepath):
        """Lưu trọng số mô hình checkpoint"""
        self.q_network.save_weights(filepath)

    def load(self, filepath):
        """Tải trọng số mô hình đã huấn luyện"""
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())


if __name__ == '__main__':
    print("--- Đang kiểm tra độc lập DQNAgent (Dương Quốc Vượng) ---")
    agent = DQNAgent(state_size=5, action_size=4)
    dummy_states = [[0.8, 15.0, 0.2, 30.0, 0.0], [0.5, 12.0, 0.5, 25.0, 0.1]]
    actions = agent.act(dummy_states, epsilon=0.1)
    print("Hành động dự đoán:", actions)
    dummy_exp = [
        (dummy_states[0], actions[0], 1.0, dummy_states[0]),
        (dummy_states[1], actions[1], -1.0, dummy_states[1])
    ] * 40
    agent.remember(dummy_exp)
    loss = agent.train(batch_size=32)
    print(f"Huấn luyện batch thành công, Loss = {loss:.4f}")
