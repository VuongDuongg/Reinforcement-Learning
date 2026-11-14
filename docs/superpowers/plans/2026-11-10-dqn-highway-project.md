# Deep Q-Network (PyTorch DQN) Implementation Plan for Highway Project

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng, huấn luyện và áp dụng thuật toán Deep Q-Network (Double DQN) sử dụng PyTorch cho xe tự hành né vật cản trên cao tốc trong thư mục `highway_project`.

**Architecture:** Sử dụng PyTorch `nn.Module` (mạng MLP: 50 -> 128 -> 128 -> 5), kết hợp Replay Memory buffer, Double DQN Bellman target, Smooth L1 (Huber) Loss và Adam optimizer. Hỗ trợ action masking loại trừ các hành động không hợp lệ theo `info["available_actions"]`.

**Tech Stack:** Python 3.13, PyTorch (`torch`), Gymnasium, Highway-Env, Pygame.

**Spec:** Triển khai theo Phương án 1 (PyTorch): module `dqn_agent.py`, cập nhật `agent.py`, kịch bản huấn luyện `train_dqn.py`, kiểm thử tự động `test_dqn.py`, tích hợp vào `run.py` và script khởi chạy `CHAY_XE_DQN.cmd`.

## Global Constraints

- Môi trường: Python 3.13 trên Windows (`.venv-highway`).
- Thư viện mạng nơ-ron: PyTorch (`torch`).
- Không gian quan sát: Kinematics `(10, 5)` phẳng hóa thành tensor 1 chiều 50 phần tử kiểu `torch.float32`.
- Không gian hành động: 5 hành động rời rạc (`0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER`).
- Phân biệt chặt chẽ `terminated` (va chạm / đến đích -> không bootstrap) và `truncated` (hết thời gian -> bootstrap TD target).
- Lưu trữ mô hình theo chuẩn PyTorch state_dict (`.pth` hoặc `.pt`).

## Review Focus

1. Cài đặt thành công `torch` vào `.venv-highway` tương thích Python 3.13 trên Windows.
2. Tensor shape & Type: Tensor quan sát đầu vào `torch.FloatTensor` kích thước `(batch_size, 50)`, đầu ra Q-values `(batch_size, 5)`.
3. Action Masking: Khi chọn hành động greedy ($\arg\max Q$), gán giá trị $-\infty$ cho các hành động không có trong `available_actions`.
4. Double DQN Target Calculation: Lựa chọn hành động tốt nhất bằng Online Network và tính Q-value tương ứng bằng Target Network: $y = r + \gamma (1 - \text{terminated}) Q_{\text{target}}(s', \arg\max_{a'} Q_{\text{online}}(s', a'))$.
5. Save & Load: Lưu và nạp `torch.save(..., path)` / `torch.load(..., weights_only=True)`, suy luận với `model.eval()` và `torch.no_grad()`.

---

### Task 1: Cài đặt PyTorch và Cấu hình Môi trường

**Files:**
- Modify: `highway_project/requirements.txt`
- Modify: `highway_project/requirements-lock.txt`

- [ ] **Step 1: Thêm `torch` vào `highway_project/requirements.txt`**
- [ ] **Step 2: Cài đặt `torch` vào `.venv-highway`**
- [ ] **Step 3: Kiểm tra import `torch` thành công trong `.venv-highway`**

---

### Task 2: Xây dựng QNetwork và ReplayBuffer bằng PyTorch

**Files:**
- Create: `highway_project/dqn_agent.py`
- Create: `highway_project/test_dqn.py`

**Interfaces:**
- Produces: `QNetwork(nn.Module)`:
  - `forward(x: torch.Tensor) -> torch.Tensor`
- Produces: `ReplayBuffer`:
  - `push(state, action, reward, next_state, terminated, truncated)`
  - `sample(batch_size: int)`
  - `__len__() -> int`
- Produces: `DQNAgent`:
  - `__init__(state_dim=50, action_dim=5, lr=5e-4, gamma=0.98, epsilon_start=1.0, epsilon_end=0.05, buffer_size=50000, batch_size=64, target_update_freq=500)`
  - `act(observation: np.ndarray, info: dict) -> int`
  - `observe(obs, action, reward, next_obs, terminated, truncated, info) -> None`
  - `learn() -> float | None`
  - `update_target_network() -> None`
  - `save(path: str | Path, metadata: dict | None = None) -> None`
  - `load(path: str | Path, device: str = "cpu") -> DQNAgent`

- [ ] **Step 1: Viết test cho `QNetwork`, `ReplayBuffer` và `DQNAgent` trong `test_dqn.py`**
- [ ] **Step 2: Chạy test để xác nhận FAIL trước khi triển khai**
- [ ] **Step 3: Cài đặt `QNetwork`, `ReplayBuffer` và `DQNAgent` trong `dqn_agent.py`**
- [ ] **Step 4: Chạy test để xác nhận PASS**

---

### Task 3: Cập nhật `agent.py` làm cầu nối mặc định

**Files:**
- Modify: `highway_project/agent.py`
- Modify: `highway_project/test_dqn.py`

- [ ] **Step 1: Viết test kiểm tra `CustomAgent` khởi tạo và tương thích với `DQNAgent`**
- [ ] **Step 2: Cập nhật `CustomAgent` trong `agent.py` để sử dụng `DQNAgent`**
- [ ] **Step 3: Chạy test kiểm tra `CustomAgent` PASS**

---

### Task 4: Xây dựng Kịch bản Huấn luyện `train_dqn.py`

**Files:**
- Create: `highway_project/train_dqn.py`

- [ ] **Step 1: Cài đặt quy trình huấn luyện DQN (Curriculum Learning + Epsilon Decay + Target Update + Evaluation)**
- [ ] **Step 2: Chạy smoke test 2 episodes với `train_dqn.py` để xác thực luồng lưu mô hình `.pth` và báo cáo JSON**

---

### Task 5: Cập nhật `run.py` và Script Khởi chạy `CHAY_XE_DQN.cmd`

**Files:**
- Modify: `highway_project/run.py`
- Create: `CHAY_XE_DQN.cmd`

- [ ] **Step 1: Cập nhật `run.py` để hỗ trợ cờ `--algo {q_learning,dqn}` và tải mô hình PyTorch `.pth`**
- [ ] **Step 2: Chạy kiểm thử `run.py --headless --algo dqn`**
- [ ] **Step 3: Tạo script `CHAY_XE_DQN.cmd` để người dùng click chạy một chạm**

---

### Task 6: Huấn luyện Mô hình DQN và Xác thực Toàn diện

**Files:**
- Output: `highway_project/models/dqn_model.pth`
- Output: `highway_project/results/dqn_report.json`

- [ ] **Step 1: Huấn luyện mô hình DQN (150-200 episodes) trên môi trường `LaneObstacleEnv`**
- [ ] **Step 2: Chạy toàn bộ test suite (`test_environment`, `test_learning`, `test_dqn`)**
- [ ] **Step 3: Chạy thử nghiệm đánh giá và xuất báo cáo tỷ lệ thành công**
