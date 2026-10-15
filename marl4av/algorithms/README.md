# HƯỚNG DẪN LÀM VIỆC NHÓM VỚI CÁC MODULE THUẬT TOÁN (N1_65KTPM)

Thư mục `algorithms/` này đã được chia tách thành các file Python độc lập cho từng thành viên, giúp các bạn code trực tiếp bằng VS Code/PyCharm, tránh xung đột Git (Merge Conflict):

---

## 1. Phân công file cụ thể:

| Thành viên | File đảm nhiệm | Thuật toán | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Dương Quốc Vượng** | `algorithms/dqn_agent.py` | **Deep Q-Network (DQN)** | Tinh chỉnh số nơ-ron (hidden size), chu kỳ cập nhật Target Network, kích thước Replay Buffer. |
| **Ngô Tuấn Anh** | `algorithms/ddqn_agent.py` | **Double Dueling DQN** | Tối ưu nhánh Value $V(s)$ và Advantage $A(s, a)$, hệ số cập nhật mềm $\tau$. |
| **Nguyễn Hồng Thái** | `algorithms/ppo_agent.py` | **Proximal Policy Optimization (PPO)** | Cân chỉnh hệ số xén $\epsilon = 0.2$, số epoch huấn luyện và hệ số GAE $\lambda = 0.95$. |
| **Lê Tuấn Khanh** | `algorithms/sac_agent.py`<br>`algorithms/baseline_agent.py` | **Soft Actor-Critic (SAC)**<br>**Rule-based Baseline** | Tinh chỉnh hệ số Entropy $\alpha$, chạy thực nghiệm đối sánh các thuật toán và lập bảng số liệu. |
| **Nguyễn Đình Tuấn Sơn** & **Nguyễn Khắc Sỹ** | `Expressways.nlogo` | **Quản lý Môi trường & Giao diện** | Tinh chỉnh số làn đường (3 làn), tần suất xe hỏng `prob-damaged-car` và hàm phần thưởng `get-reward`. |

---

## 2. Cách chạy kiểm thử độc lập (Không cần bật NetLogo):

Trước khi gắn vào mô phỏng NetLogo, mỗi thành viên có thể tự test file của mình bằng lệnh:

```bash
# Test riêng từng thuật toán:
python algorithms/dqn_agent.py
python algorithms/ddqn_agent.py
python algorithms/ppo_agent.py

# Test toàn bộ các thuật toán cùng lúc:
python test_all_agents.py
```

---

## 3. Cách NetLogo kết nối với các file này qua `rl_bridge.py`:

Khi chạy NetLogo, file `rl_bridge.py` sẽ đóng vai trò trung gian:
1. Khi bấm nút `Setup`: `rl_bridge.init_agent()` sẽ tự động khởi tạo đúng class của thuật toán được chọn trên giao diện NetLogo.
2. Khi xe chạy: NetLogo gọi `rl_bridge.select_actions(states)` để AI chọn hành động.
3. Khi xe nhận phần thưởng: NetLogo gọi `rl_bridge.update_agent(...)` để cập nhật trọng số mạng nơ-ron.
