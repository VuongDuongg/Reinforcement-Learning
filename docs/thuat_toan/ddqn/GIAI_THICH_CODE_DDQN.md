# GIẢI THÍCH CHI TIẾT MÃ NGUỒN THUẬT TOÁN DOUBLE DUELING DQN

> **Tài liệu kỹ thuật chuyên sâu — Thuật toán 2**  
> **Thành viên phụ trách:** Ngô Tuấn Anh (MSV: `2351170570`)  
> **Dự án:** Học Tăng Cường Cho Xe Tự Hành Né Chướng Ngại Vật Trên Cao Tốc Đa Làn (`highway_project/`)  
> **Nền tảng:** PyTorch + Gymnasium API + Highway-Env + Pygame 60 FPS  
> **Mã nguồn lõi:**
> * [`highway_project/algorithms/ddqn_agent.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/algorithms/ddqn_agent.py) (`DuelingQNetwork`, `DoubleDuelingDQNAgent`, Soft Update)
> * [`highway_project/training/train_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/training/train_ddqn.py) (Warmup, Curriculum Learning, Đánh giá & Xuất báo cáo)
> * [`highway_project/run.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/run.py) & [`highway_project/gui/view.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/gui/view.py) (Tích hợp CLI `--algo ddqn` & chuyển đổi thuật toán trực tiếp trên GUI)
> * [`highway_project/tests/test_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/tests/test_ddqn.py) (Bộ 6 unit tests tự động kiểm chứng kiến trúc Dueling và Soft Update)

---

## 1. LÝ DO CHỌN THUẬT TOÁN & TỔNG QUAN KIẾN TRÚC

### 1.1. Tại Sao Chọn Double Dueling DQN Cho Bài Toán Xe Tự Hành Cao Tốc?

Trong bài toán xe tự hành né chướng ngại vật (xe hỏng đứng yên $v = 0$ + dòng xe giao thông di chuyển ngẫu nhiên) trên cao tốc 2..6 làn, **Double Dueling DQN** được lựa chọn vì giải quyết trực tiếp 5 đặc thù kỹ thuật của bài toán:

1. **Xử lý trực tiếp không gian trạng thái liên tục 68 chiều (Giải quyết lời nguyền số chiều):**
   * Môi trường trả về khoảng cách radar liên tục ($0\text{–}180\text{ m}$), tốc độ tiếp cận, vị trí và vận tốc của 10 phương tiện lân cận. Bảng Q rời rạc (Tabular Q-Learning) không thể lưu nổi tổ hợp trạng thái này, còn mạng nơ-ron sâu xấp xỉ trực tiếp $Q(s, a; \theta)$ và tổng quát hóa tốt sang các kịch bản chưa từng gặp.
2. **Phù hợp hoàn hảo với không gian hành động rời rạc cấp chiến thuật (5 lệnh điều khiển):**
   * Bộ điều khiển cấp thấp của `HighwayEnv` đảm nhiệm góc lái/chân ga; AI chỉ cần ra 1 trong 5 quyết định rời rạc `{LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}`. Phép `argmax` trên 5 đầu ra tốn $< 1\text{ ms}$ trên CPU, đáp ứng dễ dàng tần số điều khiển $10\text{ Hz}$ và hiển thị $60\text{ FPS}$.
3. **Kiến trúc Dueling ($V(s)$ + $A(s, a)$) tối ưu cho đặc thù đường cao tốc:**
   * Trên cao tốc, phần lớn thời gian đường phía trước thông thoáng — giá trị an toàn chủ yếu nằm ở **trạng thái hiện tại $s$** chứ không khác biệt lớn giữa các hành động. Tách nhánh $V(s)$ giúp mạng học giá trị trạng thái nhanh hơn và ổn định hơn mạng MLP đơn nhánh thông thường.
4. **Double Q-Learning triệt tiêu thiên kiến thổi phồng (Overestimation Bias):**
   * Tách biệt mạng chọn hành động (Online $\theta$) và mạng định giá (Target $\theta^-$), ngăn xe đánh giá quá cao giá trị $Q$ dẫn đến hành vi vượt ẩu qua khe hẹp.
5. **Off-policy + Soft Target Update ($\tau = 0.005$):**
   * Tính chất Off-policy cho phép tái sử dụng dữ liệu trong `ReplayBuffer` ($50\,000$ mẫu) và khởi tạo mồi kinh nghiệm (`warmup_replay_buffer`). Cập nhật mềm Polyak giúp mục tiêu TD dịch chuyển êm ái sau mỗi bước học, hạn chế dao động so với cập nhật cứng mỗi 400 bước.

---

### 1.2. Đặc Tả Chi Tiết Đầu Vào (Input) & Đầu Ra (Output) Của Hệ Thống

#### A. Luồng Đầu Vào — Xử Lý — Đầu Ra Tổng Thể

```text
ĐẦU VÀO TỪ MÔI TRƯỜNG (LaneObstacleEnv)
 ├── observation: np.ndarray shape (10, 5), float32   (Ma trận Kinematics 10 xe gần nhất)
 └── info: dict
      ├── available_actions: list[int] ⊆ {0,1,2,3,4}  (Các hành động hợp lệ theo biên làn)
      └── sensors: dict                               (Radar 180m đa làn: lane, speed, front_gap, rear_gap, ...)
                           │
                           ▼  extract_state(observation, info)
              Vector trạng thái chuẩn hóa: state_vec ∈ R^68 (float32)
                           │
                           ▼  DuelingQNetwork.forward(state_tensor [1, 68])
              ├── Nhánh Value:     V(s)   ∈ R^(1, 1)
              ├── Nhánh Advantage: A(s,a) ∈ R^(1, 5)
              └── Tổng hợp Q:      Q(s,a) = V(s) + (A(s,a) - mean(A)) ∈ R^(1, 5)
                           │
                           ▼  act(): ε-greedy + Action Masking + An toàn chủ động (Anti-stall)
ĐẦU RA ĐIỀU KHIỂN
 └── action: int ∈ {0, 1, 2, 3, 4}  --> Gửi vào env.step(action)
```

#### B. Chi Tiết Đầu Vào: Vector Trạng Thái 68 Chiều (`extract_state`)

| Chỉ số (Index) | Số chiều | Thành phần | Phạm vi gốc | Chuẩn hóa | Ý nghĩa nghiệp vụ |
| :---: | :---: | :--- | :---: | :---: | :--- |
| `0` | 1 | `lane_norm` | $0 \dots N-1$ | $/ \max(1, N-1)$ | Vị trí làn hiện tại của xe Ego (hỗ trợ động $N \in [2..6]$ làn) |
| `1` | 1 | `speed_norm` | $0 \dots 35\text{ m/s}$ | $/ 30.0$ | Vận tốc tức thời chuẩn hóa theo vận tốc thiết kế ($108\text{ km/h}$) |
| `2` | 1 | `changing` | $\{0, 1\}$ | Nhị phân | `1.0` nếu xe đang trong quá trình chuyển làn, `0.0` nếu đang đi thẳng |
| `3..5` | 3 | `fg` (`front_gap`) | $0 \dots 180\text{ m}$ | $/ 180.0$ | Khoảng cách tới vật cản/xe phía trước trên 3 kênh `[Làn trái, Làn hiện tại, Làn phải]` |
| `6..8` | 3 | `rg` (`rear_gap`) | $0 \dots 180\text{ m}$ | $/ 180.0$ | Khoảng cách tới xe phía sau trên 3 kênh `[Làn trái, Làn hiện tại, Làn phải]` |
| `9..11` | 3 | `fc` (`front_closing`) | $-30 \dots +30\text{ m/s}$ | $/ 30.0$ | Tốc độ tiếp cận tương đối với xe trước ($v_{\text{ego}} - v_{\text{front}}$) trên 3 kênh |
| `12..14` | 3 | `rc` (`rear_closing`) | $-30 \dots +30\text{ m/s}$ | $/ 30.0$ | Tốc độ xe sau đang lao tới ($v_{\text{rear}} - v_{\text{ego}}$) trên 3 kênh |
| `15` | 1 | `danger` | $\{0, 1\}$ | Nhị phân | `1.0` khi khoảng cách trước $< 40\text{ m}$ hoặc thời gian va chạm $\text{TTC} < 2.5\text{ s}$ |
| `16` | 1 | `left_clear` | $\{0, 1\}$ | Nhị phân | `1.0` nếu làn bên trái tồn tại và đủ khoảng trống an toàn để chuyển làn |
| `17` | 1 | `right_clear` | $\{0, 1\}$ | Nhị phân | `1.0` nếu làn bên phải tồn tại và đủ khoảng trống an toàn để chuyển làn |
| `18..67` | 50 | `obs_flat[:50]` | Ma trận `10 × 5` | Chuẩn hóa sẵn | `[presence, x, y, vx, vy]` của 10 xe gần nhất quanh xe Ego từ `HighwayEnv` |

#### C. Chi Tiết Đầu Ra: 5 Hành Động Rời Rạc & Giá Trị Trả Về Của Từng Hàm

**1. Đầu ra hành động điều khiển (`action: int`):**

| Mã (`int`) | Tên hành động | Ý nghĩa điều khiển trong môi trường `LaneObstacleEnv` |
| :---: | :--- | :--- |
| `0` | `LANE_LEFT` | Chuyển sang làn bên trái (`lane - 1`, chỉ hợp lệ khi `lane > 0` và không đang đổi làn) |
| `1` | `IDLE` | Giữ nguyên làn và duy trì vận tốc mục tiêu (hoặc tiếp tục bám quỹ đạo đang đổi làn) |
| `2` | `LANE_RIGHT` | Chuyển sang làn bên phải (`lane + 1`, chỉ hợp lệ khi `lane < lanes_count - 1`) |
| `3` | `FASTER` | Tăng vận tốc mục tiêu thêm $+5\text{ m/s}$ (giới hạn bởi `max_speed`) |
| `4` | `SLOWER` | Giảm vận tốc mục tiêu đi $-5\text{ m/s}$ (phanh giảm tốc tránh va chạm) |

**2. Bảng tóm tắt Input / Output của các phương thức chính:**

| Phương thức | Đầu vào (Input) | Đầu ra (Output) |
| :--- | :--- | :--- |
| `DuelingQNetwork.forward_streams(x)` | `x: Tensor(B, 68)` | `(value: Tensor(B, 1), advantage: Tensor(B, 5))` |
| `DuelingQNetwork.forward(x)` | `x: Tensor(B, 68)` | `q_values: Tensor(B, 5)` thỏa mãn $\text{mean}_a Q(s, a) = V(s)$ |
| `agent.extract_state(obs, info)` | `obs: ndarray(10, 5)`, `info: dict` | `state_vec: ndarray(68,), float32` |
| `agent.act(obs, info)` | `obs: ndarray(10, 5)`, `info: dict` | `action: int ∈ {0, 1, 2, 3, 4}` |
| `agent.observe(obs, a, r, next_obs, term, trunc, info, next_info)` | Bộ chuyển tiếp đầy đủ từ `env.step()` | `None` (ghi vào `ReplayBuffer` và kích hoạt `learn()`) |
| `agent.learn()` | Lấy ngẫu nhiên 64 mẫu từ `self.buffer` | `loss: float` (Smooth L1 Loss) hoặc `None` nếu chưa đủ 64 mẫu |
| `agent.soft_update_target()` | Trọng số `q_net` ($\theta$) và `target_net` ($\theta^-$) | `None` (cập nhật tại chỗ $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$) |
| `agent.save(path, metadata)` | Đường dẫn file `.pth`, `metadata: dict` | File checkpoint `models/ddqn_model.pth` (chứa `model_state_dict`, `target_state_dict`, `training_steps`, `metadata`) |
| `DoubleDuelingDQNAgent.load(path)` | Đường dẫn file `.pth`, `device` | Đối tượng `DoubleDuelingDQNAgent` đã khôi phục đủ trọng số và siêu tham số |

---

### 1.3. Luồng Tích Hợp Trong `highway_project`

Thay vì sao chép lại toàn bộ logic trích xuất radar 68 chiều và tầng an toàn chủ động (dễ gây sai lệch khi bảo trì), `DoubleDuelingDQNAgent` được thiết kế **kế thừa trực tiếp từ `DQNAgent`** và ghi đè đúng 4 thành phần tạo nên bản sắc của **Double Dueling DQN**:

1. **Mạng nơ-ron:** Thay `QNetwork` (MLP 1 nhánh) bằng `DuelingQNetwork` (tách nhánh Value $V(s)$ và Advantage $A(s, a)$).
2. **Học & Cập nhật Target:** Ghi đè `learn()` sử dụng Double Q-Learning kết hợp **Soft Update** (`soft_update_target()`) với $\tau = 0.005$ sau mỗi bước học.
3. **Siêu tham số mặc định:** $lr = 3 \times 10^{-4}$, $\gamma = 0.98$, $\tau = 0.005$.
4. **Lưu/Tải Checkpoint:** Tách riêng file trọng số tại `highway_project/models/ddqn_model.pth` kèm metadata riêng (`algorithm = "double_dueling_dqn_pytorch"`, `tau`, `hidden_dims`, `author`).

### 1.4. Bản Đồ Các File Liên Quan

| File | Vai trò trong hệ thống |
| :--- | :--- |
| [`highway_project/algorithms/ddqn_agent.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/algorithms/ddqn_agent.py) | Cài đặt `DuelingQNetwork`, `DoubleDuelingDQNAgent`, `DEFAULT_MODEL = models/ddqn_model.pth`. |
| [`highway_project/ddqn_agent.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/ddqn_agent.py) | Shim tương thích ngược cho lệnh import ngắn `from highway_project.ddqn_agent import ...`. |
| [`highway_project/training/train_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/training/train_ddqn.py) | Script huấn luyện độc lập với Replay Warmup + Curriculum Learning, xuất `results/ddqn_report.json`. |
| [`highway_project/train_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/train_ddqn.py) | Shim cho phép chạy trực tiếp bằng `python -m highway_project.train_ddqn`. |
| [`highway_project/run.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/run.py) | Hỗ trợ `--algo ddqn`, tự nhận diện checkpoint `ddqn`, và bộ nhớ đệm `agent_cache` để đổi thuật toán trực tiếp không reset xe. |
| [`highway_project/gui/view.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/gui/view.py) | Hiển thị tiêu đề `DOUBLE DUELING DQN`, nút `Tab · AI: DDQN` và nút chọn `Double Dueling DQN (Tuấn Anh)` trong Menu Cài đặt (`M`). |
| [`highway_project/launcher.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/launcher.py) | Thêm tùy chọn `[2]` chạy nhanh Double Dueling DQN và chọn thuật toán trong cấu hình tùy biến `[7]`. |
| [`highway_project/training/plot_results.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/training/plot_results.py) | Tự động vẽ thêm cột benchmark **Double Dueling DQN** khi có `results/ddqn_report.json`. |
| [`highway_project/tests/test_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/tests/test_ddqn.py) | Bộ 6 bài unit test kiểm chứng kiến trúc Dueling, Soft Update, Save/Load, Action Masking và chạy đa làn. |

---

## 2. MẠNG DUELING Q-NETWORK: `class DuelingQNetwork(nn.Module)`

Mã nguồn tại [`highway_project/algorithms/ddqn_agent.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/algorithms/ddqn_agent.py#L34-L86):

```python
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
        ...
        self.value_stream = nn.Sequential(val_fc1, nn.ReLU(), val_fc2)

        # Nhánh 2: Advantage stream A(s, a) -> (batch, action_dim)
        adv_fc1 = nn.Linear(prev_dim, stream_dim)
        adv_fc2 = nn.Linear(stream_dim, self.output_dim)
        ...
        self.advantage_stream = nn.Sequential(adv_fc1, nn.ReLU(), adv_fc2)
```

### 2.1. Giải Thích Chi Tiết Từng Khối
1. **`self.feature_net` (Khối trích xuất đặc trưng chung):**
   * Nhận tensor đầu vào kích thước `(B, 68)` và đi qua 2 tầng `Linear(68, 128) -> ReLU -> Linear(128, 128) -> ReLU`.
   * Dùng khởi tạo **Kaiming Normal** (`nonlinearity="relu"`) và `bias = 0.0` để giữ ổn định phương sai tín hiệu qua ReLU.
2. **`self.value_stream` (Nhánh Giá trị trạng thái $V(s)$):**
   * Chiếu từ `128 -> 64 (ReLU) -> 1 (Linear)`.
   * Xuất ra 1 giá trị vô hướng cho mỗi trạng thái trong batch: kích thước `(B, 1)`.
3. **`self.advantage_stream` (Nhánh Lợi thế hành động $A(s, a)$):**
   * Chiếu từ `128 -> 64 (ReLU) -> 5 (Linear)`.
   * Xuất ra điểm lợi thế tương đối cho 5 hành động: kích thước `(B, 5)`.
   * Cả hai tầng đầu ra (`val_fc2`, `adv_fc2`) dùng khởi tạo **Xavier Normal** để đầu ra ban đầu cân bằng, không thiên vị hành động nào.

### 2.2. Phương Thức Lan Truyền Xuôi `forward_streams()` và `forward()`

```python
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
```

* `advantage.mean(dim=1, keepdim=True)` tính trung bình trên chiều hành động (dimension 1), trả về tensor `(B, 1)` để trừ theo cơ chế broadcasting khỏi `advantage` `(B, 5)`.
* Tách hàm `forward_streams()` giúp bộ kiểm thử `test_ddqn.py` xác minh trực tiếp bất biến toán học $\text{mean}_a Q(s, a) = V(s)$.

---

## 3. TÁC NHÂN `class DoubleDuelingDQNAgent(DQNAgent)`

### 3.1. Khởi Tạo Mạng Dueling & Kế Thừa Giao Diện Môi Trường

Mã nguồn tại [`highway_project/algorithms/ddqn_agent.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/algorithms/ddqn_agent.py#L89-L139):

```python
class DoubleDuelingDQNAgent(DQNAgent):
    """Tác nhân Double Dueling DQN (Ngô Tuấn Anh - Thuật toán 2) cho xe tự hành."""

    label = "Double Dueling DQN (PyTorch) · Ngô Tuấn Anh"

    def __init__(
        self,
        *,
        state_dim: int = DEFAULT_STATE_DIM,
        action_dim: int = 5,
        hidden_dims: list[int] | None = None,
        lr: float = 3e-4,
        gamma: float = 0.98,
        tau: float = 0.005,
        ...
    ):
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [128, 128]
        self.tau = float(tau)
        super().__init__(...)

        # Thay thế mạng MLP thường bằng kiến trúc Dueling Q-Network
        torch.manual_seed(seed)
        self.q_net = DuelingQNetwork(state_dim, self.hidden_dims, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, self.hidden_dims, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(capacity=buffer_size, state_dim=state_dim)
```

* Nhờ kế thừa `DQNAgent`, `DoubleDuelingDQNAgent` tự động sở hữu đầy đủ:
  * `extract_state()`: Ghép 18 đặc trưng radar (tự động chiếu ego-centric 3 kênh cho mọi cao tốc 2..6 làn) + 50 đặc trưng kinematics = 68 chiều.
  * `available_actions()` và `act()`: Chọn hành động $\epsilon$-greedy, Action Masking (khóa đổi làn khi đang chuyển làn), bộ lọc làn an toàn và hệ thống chống chết máy (Anti-stall).
  * `observe()`: Đẩy `(s, a, r, s', terminated, truncated)` vào `ReplayBuffer` và gọi `self.learn()`.

### 3.2. Cập Nhật Mềm Mạng Đích: `soft_update_target()`

```python
    def soft_update_target(self) -> None:
        """Cập nhật mềm (Soft Update / Polyak Averaging): theta^- <- tau*theta + (1-tau)*theta^-."""
        with torch.no_grad():
            for target_param, online_param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)
```

* Thực hiện trong ngữ cảnh `with torch.no_grad():` để không tạo đồ thị tính đạo hàm thừa cho `target_net`.
* Dùng các phép toán tại chỗ (`mul_`, `add_`) giúp cập nhật nhanh, không cấp phát thêm bộ nhớ tensor tạm.
* Với mặc định `tau = 0.005`, mỗi bước học mạng đích giữ lại $99.5\%$ trọng số cũ và hấp thụ $0.5\%$ trọng số mới của mạng chính.

### 3.3. Vòng Học Double Q-Learning: `learn()`

Mã nguồn tại [`highway_project/algorithms/ddqn_agent.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/algorithms/ddqn_agent.py#L147-L181):

```python
    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, terminateds = self.buffer.sample(self.batch_size)
        ...
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
```

Luồng tính toán trong từng bước `learn()`:
1. **Kiểm tra dữ liệu:** Nếu số mẫu trong `ReplayBuffer` nhỏ hơn `batch_size` (`64`), trả về `None`.
2. **Dự đoán $Q(s_i, a_i; \theta)$:** `self.q_net(states_t).gather(1, actions_t)` chọn đúng giá trị Q ứng với hành động `actions_t` đã thực thi.
3. **Tính mục tiêu Double Q-Learning:**
   * `next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)` chọn hành động tối ưu $a^*$ bằng mạng Online.
   * `next_target_q = self.target_net(next_states_t).gather(1, next_actions)` lấy điểm đánh giá $Q(s', a^*; \theta^-)$ từ mạng Target.
   * Nhân với `(1.0 - terminateds_t)` để đảm bảo trạng thái kết thúc vật lý không cộng thêm giá trị tương lai.
4. **Lan truyền ngược ổn định:** Tính `SmoothL1Loss` (Huber Loss), cắt chuẩn gradient tại `max_norm=10.0`, cập nhật trọng số bằng `Adam`, sau đó gọi ngay `self.soft_update_target()`.

### 3.4. Lưu & Nạp Checkpoint: `save()` và `load()`

* `save()` lưu `model_state_dict`, `target_state_dict`, `training_steps`, và chuỗi JSON `metadata` chứa `"algorithm": "double_dueling_dqn_pytorch"`, `"author": "Ngo Tuan Anh (2351170570)"`, `"hidden_dims"`, `"tau"`.
* `load()` dùng `torch.load(..., weights_only=True)` an toàn, tự động đọc lại `state_dim`, `action_dim`, `hidden_dims`, `tau` từ `metadata`, khởi tạo đúng `DuelingQNetwork` và chuyển cả hai mạng sang `.eval()`.

---

## 4. QUY TRÌNH HUẤN LUYỆN: `train_ddqn.py`

Mã nguồn tại [`highway_project/training/train_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/training/train_ddqn.py):

1. **Khởi động bộ đệm kinh nghiệm (`warmup_replay_buffer`):**
   * Nếu có sẵn mô hình `models/q_learning.npz`, chạy 40 episodes mẫu (mật độ 0/3/6 xe, luân phiên cả 3 làn `0, 1, 2`) để nạp trước các kinh nghiệm né vật cản vào `ReplayBuffer`, rút ngắn giai đoạn khám phá mù ban đầu.
2. **Huấn luyện tăng tiến độ khó (Curriculum Learning):**
   * $25\%$ số episode đầu chạy không có xe nền (`traffic = 0`) để mạng tập trung học né xe hỏng tĩnh ở cả 3 làn.
   * $75\%$ số episode sau tăng dần mật độ xe nền ngẫu nhiên `[3, 6, 12]` xe với khoảng cách vật cản thay đổi ngẫu nhiên `[65m, 140m]`.
   * Giảm dần hệ số khám phá theo đường cong lũy thừa: $\epsilon = \max\big(0.05,\; 0.8 \cdot (1 - \text{progress})^{1.8}\big)$.
3. **Đánh giá khách quan & Xuất báo cáo:**
   * Đánh giá đối sánh giữa mô hình chưa huấn luyện (`untrained`) và mô hình sau huấn luyện trên cùng tập `eval_seeds` ở 3 mức giao thông `(0, 6, 12)` xe.
   * Lưu checkpoint tại [`highway_project/models/ddqn_model.pth`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/models/ddqn_model.pth) và dữ liệu thống kê tại [`highway_project/results/ddqn_report.json`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/results/ddqn_report.json).

### 4.1. Đầu Ra Thực Nghiệm Đã Đạt Được (160 Episodes, 22.135 Bước Cập Nhật)

| Mức giao thông (10 episodes/mức) | Tỷ lệ đến đích | Tỷ lệ va chạm | Vận tốc trung bình |
| :---: | :---: | :---: | :---: |
| **0 xe** (chỉ xe hỏng tĩnh) | **100%** | **0%** | **99.8 km/h** |
| **6 xe** | **100%** | **0%** | **99.2 km/h** |
| **12 xe** | **100%** | **0%** | **95.3 km/h** |

---

## 5. TÍCH HỢP VÀO MÔI TRƯỜNG CHẠY & GIAO DIỆN PYGAME

### 5.1. Bộ Điều Phối `run.py`
* Mở rộng cờ CLI: `--algo {auto, dqn, ddqn, q_learning}`.
* Khi truyền `--algo auto --model <path>`, nếu tên file chứa `"ddqn"` hoặc `"dueling"`, hệ thống tự động nạp bằng `DoubleDuelingDQNAgent`.
* Hàm `build_agent(target_algo)` tích hợp **bộ nhớ đệm `agent_cache`**:
  * Cho phép chuyển đổi trực tiếp giữa **DQN** (`menu_mode_agent`), **Double Dueling DQN** (`menu_mode_ddqn`) và **Bàn phím** (`menu_mode_keyboard`) ngay trong khi xe đang chạy mà không reset vị trí hay phải tải lại file `.pth` từ ổ đĩa mỗi lần bấm.
  * Nếu người dùng chọn Double Dueling DQN trên GUI trước khi huấn luyện `ddqn_model.pth`, hệ thống tự động fallback sang `DoubleDuelingDQNAgent(epsilon=0.0)` kết hợp tầng an toàn chủ động để vẫn chạy mượt, không crash.

### 5.2. Giao Diện `gui/view.py` & `launcher.py`
* `view.py` nhận diện động qua `is_ddqn = "Dueling" in str(agent_label)` để hiển thị:
  * Header: `RL / GYMNASIUM HIGHWAY-ENV  ·  DOUBLE DUELING DQN`.
  * Dòng phụ: `Thuật toán: Double Dueling DQN (Ngô Tuấn Anh)`.
  * Nút chế độ thanh đáy: `Tab · AI: DDQN`.
  * Menu Cài đặt (`M` → Mục 6): Chia 3 nút ngang vừa khít khung `904 px` gồm `DQN (Deep Q-Network)`, `Double Dueling DQN (Tuấn Anh)` và `Bàn phím (← → ↑ ↓)`.
* `launcher.py`: Bổ sung mục `[2]` chạy trực tiếp Thuật toán 2 và cho phép chọn thuật toán khi tùy biến chi tiết ở mục `[7]`.

---

## 6. BỘ KIỂM THỬ TỰ ĐỘNG: `tests/test_ddqn.py`

Bộ kiểm thử tại [`highway_project/tests/test_ddqn.py`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/tests/test_ddqn.py) gồm 6 bài test độc lập:

| Tên Test Case | Mục tiêu kiểm chứng |
| :--- | :--- |
| `test_dueling_network_forward_and_aggregation` | Xác nhận kích thước `V(s) = (4, 1)`, `A(s,a) = (4, 5)`, `Q = (4, 5)` và kiểm chứng đúng công thức toán học $\text{mean}_a Q(s, a) = V(s)$. |
| `test_ddqn_action_masking_respects_available_actions` | Đảm bảo `act()` chỉ trả về các hành động nằm trong `available_actions`. |
| `test_ddqn_learn_and_soft_target_update` | Xác nhận `learn()` trả về loss dương, tăng `training_steps`, và `target_net` dịch chuyển mượt theo Soft Update ($\tau < 1$: khác trọng số ban đầu nhưng không trùng y hệt `q_net`). |
| `test_ddqn_save_load_and_deterministic_inference` | Xác nhận lưu/tải file `.pth` bảo toàn `metadata`, hệ số `tau` và cho ra hành động giống hệt nhau khi `epsilon = 0.0`. |
| `test_ddqn_dodges_obstacle_when_danger_detected` | Kiểm chứng tình huống có vật cản đứng yên phía trước 20 m: AI bắt buộc né làn hoặc phanh, không đâm thẳng. |
| `test_ddqn_agent_various_lanes_and_speeds` | Đảm bảo `DoubleDuelingDQNAgent` tương thích hoàn hảo khi chạy trên cao tốc 2, 4 và 5 làn ở vận tốc tùy biến. |
