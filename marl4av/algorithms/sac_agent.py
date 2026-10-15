"""
algorithms/sac_agent.py
Thuật toán 4 (Nâng cao): Soft Actor-Critic (SAC) với Entropy Regularization
Thành viên phụ trách: Lê Tuấn Khanh (MSV: 2351170601)
Nhóm: N1_65KTPM - Đại học Thủy Lợi

Sửa chữa triệt để lỗi "NameError: name 'self' is not defined" trong mã gốc NetLogo.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

class SACAgent:
    """
    Tác nhân Soft Actor-Critic (SAC) cho không gian hành động rời rạc:
    Tối đa hóa đồng thời phần thưởng kỳ vọng và Entropy của chính sách:
    J(π) = E[ Σ γ^t (r_t + α * H(π(·|s_t))) ]
    Giúp tác nhân vừa né vật cản an toàn vừa giữ tính khám phá cao.
    """
    def __init__(self, state_size=5, action_size=4, hidden_size=64, 
                 lr=0.0003, alpha=0.2, memory_size=50000, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.alpha = alpha  # Trọng số Entropy
        self.tau = tau
        self.memory_size = memory_size

        self.memory = []

        # 1. Mạng Actor
        self.actor = self._build_actor(lr)

        # 2. Hai mạng Critic (Twin Q-Networks) để tránh overestimation
        self.critic1 = self._build_critic(lr)
        self.critic2 = self._build_critic(lr)
        self.target_critic1 = self._build_critic(lr)
        self.target_critic2 = self._build_critic(lr)
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

    def _build_actor(self, lr):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = layers.Dense(self.hidden_size, activation='relu')(x)
        probs = layers.Dense(self.action_size, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=probs)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
        return model

    def _build_critic(self, lr):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = layers.Dense(self.hidden_size, activation='relu')(x)
        q_vals = layers.Dense(self.action_size, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=q_vals)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def act(self, states, epsilon=0.0):
        states_arr = np.array(states, dtype=np.float32)
        if len(states_arr.shape) == 1:
            states_arr = np.expand_dims(states_arr, axis=0)
        probs = self.actor(states_arr).numpy()
        actions = [int(np.random.choice(self.action_size, p=p)) for p in probs]
        return actions

    def remember(self, experiences):
        for exp in experiences:
            self.memory.append(exp)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

    def train(self, batch_size=64, gamma=0.99):
        if len(self.memory) < batch_size:
            return 0.0

        sample_ix = np.random.randint(len(self.memory), size=batch_size)
        states = np.array([self.memory[i][0] for i in sample_ix], dtype=np.float32)
        actions = np.array([self.memory[i][1] for i in sample_ix], dtype=np.int32)
        rewards = np.array([self.memory[i][2] for i in sample_ix], dtype=np.float32)
        next_states = np.array([self.memory[i][3] for i in sample_ix], dtype=np.float32)

        # 1. Tính giá trị mục tiêu cho Critic
        next_probs = self.actor(next_states).numpy()
        next_log_probs = np.log(next_probs + 1e-8)

        target_q1 = self.target_critic1.predict(next_states, verbose=0)
        target_q2 = self.target_critic2.predict(next_states, verbose=0)
        min_target_q = np.minimum(target_q1, target_q2)

        next_v = np.sum(next_probs * (min_target_q - self.alpha * next_log_probs), axis=1)
        target_values = rewards + gamma * next_v

        # 2. Huấn luyện hai mạng Critic
        curr_q1 = self.critic1.predict(states, verbose=0)
        curr_q2 = self.critic2.predict(states, verbose=0)
        curr_q1[np.arange(batch_size), actions] = target_values
        curr_q2[np.arange(batch_size), actions] = target_values

        loss1 = self.critic1.train_on_batch(states, curr_q1)
        loss2 = self.critic2.train_on_batch(states, curr_q2)

        # 3. Huấn luyện Actor
        with tf.GradientTape() as tape:
            probs = self.actor(states, training=True)
            log_probs = tf.math.log(probs + 1e-8)
            q1 = self.critic1(states)
            q2 = self.critic2(states)
            min_q = tf.minimum(q1, q2)
            actor_loss = tf.reduce_mean(tf.reduce_sum(probs * (self.alpha * log_probs - min_q), axis=1))

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # 4. Cập nhật mềm Target Critics
        for t_w, c_w in zip(self.target_critic1.weights, self.critic1.weights):
            t_w.assign(self.tau * c_w + (1.0 - self.tau) * t_w)
        for t_w, c_w in zip(self.target_critic2.weights, self.critic2.weights):
            t_w.assign(self.tau * c_w + (1.0 - self.tau) * t_w)

        return float(loss1 + loss2)
