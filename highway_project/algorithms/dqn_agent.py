"""Deep Q-Network (DQN gốc - phiên bản 2013) với PyTorch cho bài toán xe tự lái né vật cản."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "dqn_model.pth"
DEFAULT_STATE_DIM = 68


class QNetwork(nn.Module):
    """Mạng nơ-ron xấp xỉ hàm giá trị hành động Q(s, a)."""

    def __init__(self, input_dim: int = DEFAULT_STATE_DIM, hidden_dims: list[int] | None = None, output_dim: int = 5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = h_dim

        out_linear = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_normal_(out_linear.weight)
        nn.init.constant_(out_linear.bias, 0.0)
        layers.append(out_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Bộ nhớ lưu trữ kinh nghiệm tái nghiệm (Experience Replay)."""

    def __init__(self, capacity: int = 50000, state_dim: int = DEFAULT_STATE_DIM):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.terminateds = np.zeros((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        state_flat = np.asarray(state, dtype=np.float32).reshape(-1)
        next_state_flat = np.asarray(next_state, dtype=np.float32).reshape(-1)

        self.states[self.ptr].fill(0.0)
        self.next_states[self.ptr].fill(0.0)
        dim = min(len(state_flat), self.state_dim)
        self.states[self.ptr, :dim] = state_flat[:dim]
        next_dim = min(len(next_state_flat), self.state_dim)
        self.next_states[self.ptr, :next_dim] = next_state_flat[:next_dim]
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        # Nếu chỉ hết giờ (truncated) chứ không va chạm/đích thì vẫn bootstrap
        self.terminateds[self.ptr] = 1.0 if (terminated and not truncated) else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminateds[indices],
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """Tác nhân Deep Q-Network gốc (DQN 2013 - không Target Network) cho xe tự hành."""

    label = "DQN 2013 (PyTorch) · mạng đơn không Target Network"

    def __init__(
        self,
        *,
        state_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = 5,
        hidden_dims: list[int] | None = None,
        lr: float = 5e-4,
        gamma: float = 0.98,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int | None = None,  # Giữ lại để tương thích script cũ, không dùng trong DQN 2013
        device: str | None = None,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # DQN 2013 chỉ sử dụng duy nhất 1 mạng Q-network
        self.q_net = QNetwork(state_dim, hidden_dims, action_dim).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # DQN 2013 sử dụng Squared TD Error (MSELoss)
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer(capacity=buffer_size, state_dim=state_dim)

        self.training_steps = 0
        self.last_state: np.ndarray | None = None
        self.metadata: dict = {}

    def extract_state(self, observation: np.ndarray, info: dict | None = None) -> np.ndarray:
        """Trích xuất vector trạng thái giàu đặc trưng không gian và radar (68 chiều)."""
        obs_flat = np.asarray(observation, dtype=np.float32).reshape(-1)
        if len(obs_flat) == self.state_dim:
            return obs_flat

        if self.state_dim == 50:
            return obs_flat[:50]

        sensors = info.get("sensors") if info else None
        if not sensors:
            sensors_vec = np.zeros(18, dtype=np.float32)
        else:
            lane = int(sensors.get("lane", 1))
            lanes_count = int(sensors.get("lanes_count", 3))
            speed = float(sensors.get("speed", 20.0))
            changing = 1.0 if sensors.get("changing", False) else 0.0

            raw_fg = list(sensors.get("front_gap", [180.0] * lanes_count))
            raw_rg = list(sensors.get("rear_gap", [180.0] * lanes_count))
            raw_fc = list(sensors.get("front_closing", [0.0] * lanes_count))
            raw_rc = list(sensors.get("rear_closing", [0.0] * lanes_count))

            if lanes_count == 3 and len(raw_fg) == 3:
                fg = np.array(raw_fg[:3], dtype=np.float32) / 180.0
                rg = np.array(raw_rg[:3], dtype=np.float32) / 180.0
                fc = np.array(raw_fc[:3], dtype=np.float32) / 30.0
                rc = np.array(raw_rc[:3], dtype=np.float32) / 30.0
                lane_norm = lane / 2.0
            else:
                if lane > 0 and lane - 1 < len(raw_fg):
                    c0_fg, c0_rg = raw_fg[lane - 1], raw_rg[lane - 1]
                    c0_fc, c0_rc = raw_fc[lane - 1], raw_rc[lane - 1]
                else:
                    c0_fg, c0_rg, c0_fc, c0_rc = 180.0, 180.0, 0.0, 0.0

                if 0 <= lane < len(raw_fg):
                    c1_fg, c1_rg = raw_fg[lane], raw_rg[lane]
                    c1_fc, c1_rc = raw_fc[lane], raw_rc[lane]
                else:
                    c1_fg, c1_rg, c1_fc, c1_rc = 180.0, 180.0, 0.0, 0.0

                if lane < lanes_count - 1 and lane + 1 < len(raw_fg):
                    c2_fg, c2_rg = raw_fg[lane + 1], raw_rg[lane + 1]
                    c2_fc, c2_rc = raw_fc[lane + 1], raw_rc[lane + 1]
                else:
                    c2_fg, c2_rg, c2_fc, c2_rc = 180.0, 180.0, 0.0, 0.0

                fg = np.array([c0_fg, c1_fg, c2_fg], dtype=np.float32) / 180.0
                rg = np.array([c0_rg, c1_rg, c2_rg], dtype=np.float32) / 180.0
                fc = np.array([c0_fc, c1_fc, c2_fc], dtype=np.float32) / 30.0
                rc = np.array([c0_rc, c1_rc, c2_rc], dtype=np.float32) / 30.0
                lane_norm = lane / max(1.0, float(lanes_count - 1))

            front_dist = raw_fg[lane] if 0 <= lane < len(raw_fg) else 180.0
            front_closing = raw_fc[lane] if 0 <= lane < len(raw_fc) else 0.0
            ttc = front_dist / max(front_closing, 0.01)
            danger = 1.0 if (front_dist < 40.0 or ttc < 2.5) else 0.0

            left_clear = 1.0 if sensors.get("left_clear", False) else 0.0
            right_clear = 1.0 if sensors.get("right_clear", False) else 0.0

            sensors_vec = np.array([
                lane_norm, speed / 30.0, changing,
                *fg, *rg, *fc, *rc,
                danger, left_clear, right_clear,
            ], dtype=np.float32)

        return np.concatenate([sensors_vec, obs_flat[:50]], dtype=np.float32)

    @staticmethod
    def available_actions(info: dict) -> list[int]:
        actions = list(info.get("available_actions", range(5)))
        sensors = info.get("sensors")
        if sensors and sensors.get("changing"):
            actions = [a for a in actions if a not in (0, 2)]
        return actions if actions else [1]

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.last_state = None

    def act(self, observation: np.ndarray, info: dict) -> int:
        actions = self.available_actions(info)
        state_vec = self.extract_state(observation, info)
        self.last_state = state_vec

        # Thăm dò e-greedy
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))

        # Khai thác: suy luận mạng nơ-ron Q-network
        self.q_net.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor).cpu().numpy()[0]

        # Action masking
        masked_q = {a: float(q_values[a]) for a in actions}

        # Cơ chế an toàn chủ động & Chống chết máy (Anti-stall)
        sensors = info.get("sensors") if info else None
        if sensors:
            lane = int(sensors.get("lane", 1))
            lanes_count = int(sensors.get("lanes_count", 3))
            speed = float(sensors.get("speed", 20.0))
            is_changing = bool(sensors.get("changing", False))
            target_lane = int(sensors.get("target_lane", lane))

            raw_fg = list(sensors.get("front_gap", [180.0] * lanes_count))
            raw_rg = list(sensors.get("rear_gap", [180.0] * lanes_count))
            raw_fc = list(sensors.get("front_closing", [0.0] * lanes_count))

            check_lane = target_lane if is_changing else lane
            fg = raw_fg[check_lane] if 0 <= check_lane < len(raw_fg) else 180.0
            closing = raw_fc[check_lane] if 0 <= check_lane < len(raw_fc) else 0.0
            is_danger = fg < 42.0 or (closing > 0.1 and fg / closing < 2.5)

            if is_changing:
                if 1 in masked_q:
                    masked_q[1] += 15.0
                if 4 in masked_q:
                    masked_q[4] -= 30.0
                if speed < 20.0 and 3 in masked_q:
                    masked_q[3] += 10.0
            else:
                if is_danger:
                    if 1 in masked_q:
                        masked_q[1] -= 30.0
                    if 3 in masked_q:
                        masked_q[3] -= 40.0

                    if lane > 0 and 0 in masked_q:
                        left_fg = raw_fg[lane - 1] if lane - 1 < len(raw_fg) else 180.0
                        left_rg = raw_rg[lane - 1] if lane - 1 < len(raw_rg) else 180.0
                        if left_fg > 25.0 and left_rg > 15.0:
                            masked_q[0] += 25.0

                    if lane < lanes_count - 1 and 2 in masked_q:
                        right_fg = raw_fg[lane + 1] if lane + 1 < len(raw_fg) else 180.0
                        right_rg = raw_rg[lane + 1] if lane + 1 < len(raw_rg) else 180.0
                        if right_fg > 25.0 and right_rg > 15.0:
                            masked_q[2] += 25.0

            if lane > 0 and 0 in masked_q:
                left_fg = raw_fg[lane - 1] if lane - 1 < len(raw_fg) else 180.0
                left_rg = raw_rg[lane - 1] if lane - 1 < len(raw_rg) else 180.0
                if left_fg < 35.0 or left_rg < 12.0:
                    masked_q[0] -= 60.0

            if lane < lanes_count - 1 and 2 in masked_q:
                right_fg = raw_fg[lane + 1] if lane + 1 < len(raw_fg) else 180.0
                right_rg = raw_rg[lane + 1] if lane + 1 < len(raw_rg) else 180.0
                if right_fg < 35.0 or right_rg < 12.0:
                    masked_q[2] -= 60.0

            if speed < 16.0:
                if 4 in masked_q:
                    masked_q[4] -= 60.0
                if 3 in masked_q:
                    masked_q[3] += 30.0
            if speed < 12.0:
                if 4 in masked_q:
                    masked_q[4] -= 120.0
                if 3 in masked_q:
                    masked_q[3] += 60.0

        preference = [a for a in (1, 3, 4, 0, 2) if a in actions]
        best_action = max(preference, key=lambda a: (masked_q[a], -preference.index(a)))
        return int(best_action)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, terminateds = self.buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        terminateds_t = torch.from_numpy(terminateds).to(self.device)

        self.q_net.train()
        current_q = self.q_net(states_t).gather(1, actions_t)

        # DQN 2013: dùng trực tiếp q_net để tính max Q(s', a'), không có Target Network
        with torch.no_grad():
            next_q_max = self.q_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + (1.0 - terminateds_t) * self.gamma * next_q_max

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        return float(loss.item())

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict | None = None,
        next_info: dict | None = None,
    ) -> None:
        state = self.extract_state(observation, info)
        next_state = self.extract_state(next_observation, next_info or info)
        self.buffer.push(
            state,
            action,
            reward,
            next_state,
            terminated=terminated,
            truncated=truncated,
        )
        self.learn()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path = DEFAULT_MODEL, *, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "algorithm": "dqn_2013_pytorch",
            "schema": 2,
            "training_steps": self.training_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            **(metadata or {}),
        }
        torch.save(
            {
                "model_state_dict": self.q_net.state_dict(),
                "training_steps": self.training_steps,
                "metadata": json.dumps(meta),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL, device: str = "cpu") -> DQNAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        metadata = json.loads(str(checkpoint.get("metadata", "{}")))
        state_dim = metadata.get("state_dim", DEFAULT_STATE_DIM)
        action_dim = metadata.get("action_dim", 5)

        agent = cls(state_dim=state_dim, action_dim=action_dim, device=device)
        try:
            agent.q_net.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            old_dim = checkpoint["model_state_dict"]["net.0.weight"].shape[1]
            agent = cls(state_dim=old_dim, action_dim=action_dim, device=device)
            agent.q_net.load_state_dict(checkpoint["model_state_dict"])

        agent.training_steps = checkpoint.get("training_steps", 0)
        agent.metadata = metadata
        agent.q_net.eval()
        return agent