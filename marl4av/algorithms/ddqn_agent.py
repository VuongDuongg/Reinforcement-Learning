"""
algorithms/ddqn_agent.py
Thuật toán 2: Double Dueling Deep Q-Network (Double Dueling DQN)
Thành viên phụ trách: Ngô Tuấn Anh (MSV: 2351170570)
Nhóm: N1_65KTPM - Đại học Thủy Lợi
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

class DoubleDuelingDQNAgent:
    """
    Tác nhân Double Dueling DQN:
    1. Dueling Architecture: Tách nhánh ước tính Giá trị trạng thái V(s) và Lợi thế hành động A(s, a).
    2. Double Q-Learning: Tách rời chọn hành động (mạng chính) và định giá trị (mạng target),
       triệt tiêu hiện tượng ước lượng phóng đại (Overestimation bias).
    """
    def __init__(self, state_size=5, action_size=4, hidden_size=64, 
                 lr=0.0003, memory_size=50000, update_steps=1000, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.memory_size = memory_size
        self.update_steps = update_steps
        self.tau = tau
        self.step_count = 0

        self.memory = []

        # Xây dựng mạng Dueling Q-Network
        self.q_network = self._build_dueling_model()
        self.target_network = self._build_dueling_model()
        self.target_network.set_weights(self.q_network.get_weights())

    def _build_dueling_model(self):
        """Xây dựng mạng Dueling Architecture bằng Keras Functional API"""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = layers.Dense(self.hidden_size, activation='relu')(x)

        # Nhánh 1: Value stream V(s) - 1 giá trị vô hướng
        val_stream = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        val = layers.Dense(1, activation='linear')(val_stream)

        # Nhánh 2: Advantage stream A(s, a) - vector kích thước |A|
        adv_stream = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        adv = layers.Dense(self.action_size, activation='linear')(adv_stream)

        # Kết hợp Dueling: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        norm_adv = layers.Lambda(lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(adv)
        q_values = layers.Add()([val, norm_adv])

        model = keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def act(self, states, epsilon=0.05):
        """Lựa chọn hành động theo chính sách Epsilon-Greedy"""
        states_arr = np.array(states, dtype=np.float32)
        if len(states_arr.shape) == 1:
            states_arr = np.expand_dims(states_arr, axis=0)

        num_agents = states_arr.shape[0]
        q_values = self.q_network.predict(states_arr, verbose=0)
        best_actions = np.argmax(q_values, axis=1)

        final_actions = []
        for i in range(num_agents):
            if np.random.rand() < epsilon:
                final_actions.append(int(np.random.randint(self.action_size)))
            else:
                final_actions.append(int(best_actions[i]))
        return final_actions

    def remember(self, experiences):
        """Lưu trải nghiệm vào Replay Buffer"""
        for exp in experiences:
            self.memory.append(exp)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

    def train(self, batch_size=64, gamma=0.99):
        """Cập nhật mạng chính bằng thuật toán Double Q-Learning"""
        if len(self.memory) < batch_size:
            return 0.0

        sample_ix = np.random.randint(len(self.memory), size=batch_size)
        states = np.array([self.memory[i][0] for i in sample_ix], dtype=np.float32)
        actions = np.array([self.memory[i][1] for i in sample_ix], dtype=np.int32)
        rewards = np.array([self.memory[i][2] for i in sample_ix], dtype=np.float32)
        next_states = np.array([self.memory[i][3] for i in sample_ix], dtype=np.float32)

        targets = self.q_network.predict(states, verbose=0)

        # --- Cơ chế then chốt của Double DQN ---
        # 1. Dùng mạng CHÍNH để chọn hành động tốt nhất ở trạng thái kế tiếp:
        next_q_main = self.q_network.predict(next_states, verbose=0)
        best_next_actions = np.argmax(next_q_main, axis=1)

        # 2. Dùng mạng TARGET để đánh giá giá trị cho hành động tốt nhất đó:
        next_q_target = self.target_network.predict(next_states, verbose=0)
        double_q_values = next_q_target[np.arange(batch_size), best_next_actions]

        # Cập nhật mục tiêu Bellman
        targets[np.arange(batch_size), actions] = rewards + gamma * double_q_values

        loss = self.q_network.train_on_batch(states, targets)

        # Cập nhật mềm (Soft Update) Target Network để tối ưu độ ổn định
        for t_weight, m_weight in zip(self.target_network.weights, self.q_network.weights):
            t_weight.assign(self.tau * m_weight + (1.0 - self.tau) * t_weight)

        self.step_count += 1
        return float(loss)

    def save(self, filepath):
        self.q_network.save_weights(filepath)

    def load(self, filepath):
        self.q_network.load_weights(filepath)
        self.target_network.set_weights(self.q_network.get_weights())


if __name__ == '__main__':
    print("--- Đang kiểm tra độc lập DoubleDuelingDQNAgent (Ngô Tuấn Anh) ---")
    agent = DoubleDuelingDQNAgent(state_size=5, action_size=4)
    dummy_states = [[0.8, 15.0, 0.2, 30.0, 0.0], [0.5, 12.0, 0.5, 25.0, 0.1]]
    actions = agent.act(dummy_states, epsilon=0.05)
    print("Hành động dự đoán:", actions)
    dummy_exp = [
        (dummy_states[0], actions[0], 1.5, dummy_states[0]),
        (dummy_states[1], actions[1], -0.5, dummy_states[1])
    ] * 40
    agent.remember(dummy_exp)
    loss = agent.train(batch_size=32)
    print(f"Huấn luyện batch thành công, Loss = {loss:.4f}")
