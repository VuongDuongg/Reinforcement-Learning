"""
algorithms/ppo_agent.py
Thuật toán 3: Proximal Policy Optimization (PPO-Clip)
Thành viên phụ trách: Nguyễn Hồng Thái (MSV: 2351170619)
Nhóm: N1_65KTPM - Đại học Thủy Lợi
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers

class PPOAgent:
    """
    Tác nhân Proximal Policy Optimization (PPO-Clip):
    - Sử dụng mô hình Actor-Critic tách biệt.
    - Cập nhật hàm mục tiêu Clipped Surrogate Loss để kiểm soát bước nhảy chính sách,
      tránh hiện tượng sập chính sách khi xe tự lái gặp tình huống bất ngờ.
    - Tích hợp Generalized Advantage Estimation (GAE) để cân bằng bias-variance.
    """
    def __init__(self, state_size=5, action_size=4, hidden_size=64, 
                 lr_actor=0.0003, lr_critic=0.001, clip_epsilon=0.2, 
                 gamma=0.99, gae_lambda=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Khởi tạo mạng Actor (Chính sách π(a|s))
        self.actor = self._build_actor(lr_actor)
        # Khởi tạo mạng Critic (Hàm giá trị V(s))
        self.critic = self._build_critic(lr_critic)

        # Bộ đệm quỹ đạo (Trajectory Buffer)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.old_log_probs = []

    def _build_actor(self, lr):
        """Mạng Actor xuất phân phối xác suất rời rạc của các hành động (Softmax)"""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = layers.Dense(self.hidden_size, activation='relu')(x)
        probs = layers.Dense(self.action_size, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=probs)
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
        return model

    def _build_critic(self, lr):
        """Mạng Critic ước lượng giá trị kỳ vọng của trạng thái V(s)"""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(self.hidden_size, activation='relu')(inputs)
        x = layers.Dense(self.hidden_size, activation='relu')(x)
        val = layers.Dense(1, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=val)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=lr)
        return model

    def act(self, states, epsilon=0.0):
        """Lấy mẫu hành động từ phân phối xác suất Categorical Distribution"""
        states_arr = np.array(states, dtype=np.float32)
        if len(states_arr.shape) == 1:
            states_arr = np.expand_dims(states_arr, axis=0)

        probs = self.actor(states_arr).numpy()
        dist = tfp.distributions.Categorical(probs=probs, dtype=tf.int32)
        actions = dist.sample().numpy()
        log_probs = dist.log_prob(actions).numpy()

        # Lưu tạm log_prob phục vụ cập nhật PPO
        self._last_log_probs = log_probs
        self._last_states = states_arr
        self._last_actions = actions

        return [int(a) for a in actions]

    def remember(self, states, actions, rewards, next_states, dones=None):
        """Lưu một chu kỳ bước vào bộ đệm quỹ đạo"""
        if dones is None:
            dones = [False] * len(states)
        for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, self._last_log_probs):
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
            self.next_states.append(ns)
            self.dones.append(d)
            self.old_log_probs.append(lp)

    def compute_gae(self, states, next_states, rewards, dones):
        """Tính toán Lợi thế GAE (Generalized Advantage Estimation)"""
        values = self.critic(states).numpy().flatten()
        next_values = self.critic(next_states).numpy().flatten()

        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        # Chuẩn hóa Lợi thế
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std
        return norm_advantages, returns

    def train(self, epochs=4, batch_size=64):
        """Tối ưu hóa hàm mục tiêu PPO-Clip qua nhiều Epochs"""
        if len(self.states) < batch_size:
            return 0.0, 0.0

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.float32)
        next_states = np.array(self.next_states, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.bool_)
        old_log_probs = np.array(self.old_log_probs, dtype=np.float32)

        # Tính Lợi thế và Target cho Critic
        advantages, returns = self.compute_gae(states, next_states, rewards, dones)

        dataset_size = len(states)
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                # Huấn luyện Actor và Critic đồng thời
                with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
                    # 1. Tính toán Actor Loss với Clipped Surrogate
                    probs = self.actor(b_states, training=True)
                    dist = tfp.distributions.Categorical(probs=probs, dtype=tf.int32)
                    new_log_probs = dist.log_prob(b_actions)
                    ratio = tf.exp(new_log_probs - b_old_log_probs)

                    surr1 = ratio * b_advantages
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                    # 2. Tính toán Critic Loss (MSE)
                    values = tf.squeeze(self.critic(b_states, training=True))
                    critic_loss = tf.reduce_mean(tf.square(b_returns - values))

                # Cập nhật Gradient
                actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

                total_actor_loss += float(actor_loss)
                total_critic_loss += float(critic_loss)

        # Xóa sạch bộ đệm sau khi cập nhật On-Policy
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.old_log_probs.clear()

        return total_actor_loss, total_critic_loss

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)


if __name__ == '__main__':
    print("--- Đang kiểm tra độc lập PPOAgent (Nguyễn Hồng Thái) ---")
    agent = PPOAgent(state_size=5, action_size=4)
    dummy_states = [[0.8, 15.0, 0.2, 30.0, 0.0], [0.5, 12.0, 0.5, 25.0, 0.1]]
    actions = agent.act(dummy_states)
    print("Hành động dự đoán:", actions)
    # Giả lập thu thập trajectory
    for _ in range(40):
        agent.remember(dummy_states, actions, [1.0, 0.5], dummy_states, [False, False])
    a_loss, c_loss = agent.train(epochs=2, batch_size=32)
    print(f"Huấn luyện PPO thành công! Actor Loss = {a_loss:.4f}, Critic Loss = {c_loss:.4f}")
