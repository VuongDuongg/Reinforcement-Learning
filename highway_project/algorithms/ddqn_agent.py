"""Double Dueling Deep Q-Network (Double Dueling DQN) với PyTorch.

Thuật toán 2 - Thành viên phụ trách: Ngô Tuấn Anh (MSV: 2351170570)
Nhóm: N1_65KTPM - Đại học Thủy Lợi

Đặc trưng kiến trúc:
1. Dueling Network Architecture:
   Tách luồng đặc trưng chung thành 2 nhánh độc lập:
   - Nhánh Giá trị trạng thái V(s) (1 giá trị vô hướng)
   - Nhánh Lợi thế hành động A(s, a) (vector kích thước |A|)
   Hợp nhất: Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
2. Double Q-Learning:
   Mạng chính (online) chọn hành động tốt nhất a* = argmax Q(s', a; theta),
   mạng đích (target) định giá trị Q(s', a*; theta^-) để triệt tiêu Overestimation Bias.
3. Soft Target Update (Polyak Averaging):
   theta^- <- tau * theta + (1 - tau) * theta^- sau mỗi bước học (tau = 0.005).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_agent import DEFAULT_STATE_DIM, DQNAgent, ReplayBuffer

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "ddqn_model.pth"


class DuelingQNetwork(nn.Module):
    """Mạng Dueling Q-Network tách nhánh Value V(s) và Advantage A(s, a)."""

    def __init__(
        self,
        input_dim: int = DEFAULT_STATE_DIM,
        hidden_dims: list[int] | None = None,
        output_dim: int = 5,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.output_dim = int(output_dim)

        feature_layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.constant_(linear.bias, 0.0)
            feature_layers.append(linear)
            feature_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.feature_net = nn.Sequential(*feature_layers)

        stream_dim = max(32, prev_dim // 2)

        # Nhánh 1: Value stream V(s) -> (batch, 1)
        val_fc1 = nn.Linear(prev_dim, stream_dim)
        val_fc2 = nn.Linear(stream_dim, 1)
        nn.init.kaiming_normal_(val_fc1.weight, nonlinearity="relu")
        nn.init.constant_(val_fc1.bias, 0.0)
        nn.init.xavier_normal_(val_fc2.weight)
        nn.init.constant_(val_fc2.bias, 0.0)
        self.value_stream = nn.Sequential(val_fc1, nn.ReLU(), val_fc2)

        # Nhánh 2: Advantage stream A(s, a) -> (batch, action_dim)
        adv_fc1 = nn.Linear(prev_dim, stream_dim)
        adv_fc2 = nn.Linear(stream_dim, self.output_dim)
        nn.init.kaiming_normal_(adv_fc1.weight, nonlinearity="relu")
        nn.init.constant_(adv_fc1.bias, 0.0)
        nn.init.xavier_normal_(adv_fc2.weight)
        nn.init.constant_(adv_fc2.bias, 0.0)
        self.advantage_stream = nn.Sequential(adv_fc1, nn.ReLU(), adv_fc2)

    def forward_streams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Trả về riêng biệt V(s) và A(s, a) phục vụ phân tích/kiểm thử."""
        features = self.feature_net(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, advantage = self.forward_streams(x)
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, .)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DoubleDuelingDQNAgent(DQNAgent):
    """Tác nhân Double Dueling DQN (Ngô Tuấn Anh - Thuật toán 2) cho xe tự hành."""

    label = "Double Dueling DQN · Ngô Tuấn Anh"

    def __init__(
        self,
        *,
        state_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = 5,
        hidden_dims: list[int] | None = None,
        lr: float = 3e-4,
        gamma: float = 0.98,
        tau: float = 0.005,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        device: str | None = None,
        seed: int = 42,
    ):
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [128, 128]
        self.tau = float(tau)

        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
            lr=lr,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=1,
            device=device,
            seed=seed,
        )

        # Thay thế mạng MLP thường bằng kiến trúc Dueling Q-Network
        torch.manual_seed(seed)
        self.q_net = DuelingQNetwork(state_dim, self.hidden_dims, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, self.hidden_dims, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(capacity=buffer_size, state_dim=state_dim)

    def soft_update_target(self) -> None:
        """Cập nhật mềm (Soft Update / Polyak Averaging): theta^- <- tau*theta + (1-tau)*theta^-."""
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def learn(self) -> float | None:
        """Cập nhật mạng chính theo Double Q-Learning và Soft Update mạng đích."""
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

        # Double Q-Learning: Online chọn hành động tốt nhất, Target định giá trị
        with torch.no_grad():
            next_online_q = self.q_net(next_states_t)
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - terminateds_t) * self.gamma * next_target_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft Update Target Network sau mỗi bước cập nhật
        self.soft_update_target()
        self.training_steps += 1

        return float(loss.item())

    def save(self, path: str | Path = DEFAULT_MODEL, *, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "algorithm": "double_dueling_dqn_pytorch",
            "author": "Ngo Tuan Anh (2351170570)",
            "schema": 1,
            "training_steps": self.training_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "tau": self.tau,
            **(metadata or {}),
        }
        torch.save(
            {
                "model_state_dict": self.q_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "training_steps": self.training_steps,
                "metadata": json.dumps(meta),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL, device: str = "cpu") -> DoubleDuelingDQNAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        metadata = json.loads(str(checkpoint.get("metadata", "{}")))
        state_dim = metadata.get("state_dim", DEFAULT_STATE_DIM)
        action_dim = metadata.get("action_dim", 5)
        hidden_dims = metadata.get("hidden_dims", [128, 128])
        tau = float(metadata.get("tau", 0.005))

        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            tau=tau,
            device=device,
        )
        agent.q_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
        agent.training_steps = checkpoint.get("training_steps", 0)
        agent.metadata = metadata
        agent.q_net.eval()
        agent.target_net.eval()
        return agent
