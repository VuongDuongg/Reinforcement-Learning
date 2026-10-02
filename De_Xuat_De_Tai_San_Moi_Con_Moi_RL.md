# ĐỀ XUẤT ĐỀ TÀI / DỰ ÁN HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)
## Chuyên đề: Cuộc chiến Săn mồi - Con mồi trong Mê cung (Predator vs. Prey Ecosystem)

---

## 1. Đề xuất Tên Đề tài & Tổng quan Dự án

- **Tên đề tài dự kiến**: *Nghiên cứu và Xây dựng Hệ thống Đối kháng Đa Tác nhân Kẻ săn mồi - Con mồi (Predator-Prey) trong Môi trường Mê cung Động sử dụng Học tăng cường.*
- **Tên phần mềm / sản phẩm**: **MazeHunter-RL (Multi-Agent Predator vs. Prey Arena)**
- **Bối cảnh & Ý tưởng cốt lõi**:
  - Môi trường là một mê cung 2D dạng lưới (Grid-world) với chướng ngại vật/tường ngăn cố định hoặc được tạo ngẫu nhiên.
  - Tham gia vào môi trường gồm có 2 tác nhân (Agents):
    - **Kẻ săn mồi (Hunter/Predator)**: Mục tiêu là định vị, truy đuổi và bắt giữ con mồi trong thời gian ngắn nhất (tối ưu hóa số bước di chuyển).
    - **Con mồi (Prey)**: Mục tiêu là lẩn trốn, tận dụng cấu trúc góc khuất của mê cung để sinh tồn lâu nhất có thể, hoặc tìm cách ăn các vật phẩm năng lượng để thoát khỏi màn chơi / sống sót qua số bước quy định.
  - Sử dụng phương pháp **Học tăng cường Đa tác nhân (Multi-Agent Reinforcement Learning - MARL)** hoặc Huấn luyện đối kháng (Self-Play / Competitive Co-Evolution) để cả hai bên liên tục thích nghi và nâng cao chiến thuật qua từng epoch.

---

## 2. Đề xuất Tính năng Phần mềm

Hệ thống được thiết kế dạng ứng dụng tương tác hoàn chỉnh (kết hợp Python/Pygame hoặc Web UI/Streamlit):

### 2.1. Môi trường mô phỏng & Cấu hình (Environment & Simulation Engine)
- **Tùy biến cấu hình mê cung**:
  - Hỗ trợ tạo mê cung ngẫu nhiên (thuật toán DFS, Prim, hoặc Kruskal) hoặc chọn các bản đồ tĩnh mẫu (Classic, Labyrinth, Open Field).
  - Tùy chỉnh kích thước bản đồ ($10 \times 10$, $15 \times 15$, $20 \times 20$).
  - Cho phép thêm/bớt chướng ngại vật động hoặc các trạm tiếp tế thức ăn/bình năng lượng cho Con mồi.
- **Cơ chế tầm nhìn (Field of View - FOV)**:
  - Tùy chọn: *Tầm nhìn toàn cảnh (Full Observation)* hoặc *Tầm nhìn hạn chế (Partial Observation - POMDP, chỉ nhìn thấy trong bán kính $k$ ô)*.

### 2.2. Huấn luyện & Đối đầu Mô hình (Training & Benchmarking)
- **Đa dạng hóa Thuật toán RL phục vụ Thực nghiệm & Đối sánh (Ablation & Benchmarking)**:
  - **Nhóm thuật toán Dạng bảng (Tabular RL - Baseline)**: **Q-Learning** và **SARSA** (thử nghiệm trên mê cung kích thước nhỏ, rời rạc $10 \times 10$).
  - **Nhóm thuật toán Dựa trên Giá trị Sâu (Value-based Deep RL)**: **DQN (Deep Q-Network)** kết hợp Replay Buffer và Target Network.
  - **Nhóm thuật toán Dựa trên Chính sách Sâu (Policy-based / Actor-Critic Deep RL)**: **PPO (Proximal Policy Optimization)** hoặc **MADDPG (Multi-Agent DDPG)** cho môi trường liên tục hoặc không gian quan sát lớn.
- **Cơ chế Benchmark & So sánh Hiệu năng**:
  - Thử nghiệm chéo (Cross-tournament): Hunter (DQN) vs Prey (PPO), Hunter (Q-Learning) vs Prey (DQN), Hunter (PPO) vs Prey (Rule-based A*).
  - So sánh tốc độ hội tụ, độ ổn định của chính sách và khả năng ứng phó khi đổi layout mê cung.
- **Chế độ đối kháng linh hoạt (Versus Modes)**:
  - `AI vs AI`: Thi đấu giữa các thuật toán khác nhau.
  - `Human vs AI`: Người chơi điều khiển Con mồi trốn chạy Hunter (AI) hoặc ngược lại.
  - `AI vs Rule-based`: Đánh giá AI chống lại thuật toán cổ điển (A* Search, Dijkstra).

---

## 3. Mô hình hóa Bài toán thành Quy trình Quyết định Markov (MDP / SG)

Vì đây là môi trường đa tác nhân đối kháng, bài toán được mô hình hóa theo dạng **Markov Game (Stochastic Game)** - mở rộng của MDP cho 2 tác nhân $i \in \{\text{Hunter } (H), \text{Prey } (P)\}$.

Mô hình được biểu diễn bằng bộ giá trị: $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}^H, \mathcal{A}^P, \mathcal{P}, \mathcal{R}^H, \mathcal{R}^P, \gamma \rangle$.

### 3.1. Không gian Trạng thái (State Space $\mathcal{S}$)
Trạng thái $s_t \in \mathcal{S}$ tại thời điểm $t$ có thể biểu diễn theo 2 dạng tùy thuộc vào thuật toán:

1. **Dạng tọa độ rút gọn (Vector Feature - Dùng cho Tabular RL & MLP-DQN/PPO)**:
   $$s_t = \Big( (x_H, y_H), (x_P, y_P), \Delta x, \Delta y, d_{\text{Manhattan}}(H, P), V_{\text{obstacles}} \Big)$$
   - $(x_H, y_H)$: Tọa độ hiện tại của Kẻ săn mồi.
   - $(x_P, y_P)$: Tọa độ hiện tại của Con mồi.
   - $(\Delta x, \Delta y) = (x_P - x_H, y_P - y_H)$: Véc-tơ khoảng cách tương đối giữa hai bên.
   - $V_{\text{obstacles}}$: Vector nhị phân biểu thị 4 hướng liền kề (Lên, Xuống, Trái, Phải) có phải là tường chắn hay không.

2. **Dạng lưới ma trận (Grid Matrix - thích hợp cho CNN-DQN / PPO)**:
   - Ma trận kích thước $C \times W \times H$ (với $C$ là các kênh: Kênh 0 = Tường, Kênh 1 = Vị trí Hunter, Kênh 2 = Vị trí Prey, Kênh 3 = Thức ăn/vật phẩm).

### 3.2. Không gian Hành động (Action Space $\mathcal{A}$)
Cả Hunter và Prey đều có không gian hành động rời rạc:
$$\mathcal{A}^H = \mathcal{A}^P = \{\text{ĐỨNG YÊN (0)}, \text{LÊN (1)}, \text{XUỐNG (2)}, \text{TRÁI (3)}, \text{PHẢI (4)}\}$$

*Lưu ý:* Nếu tác nhân chọn hành động đâm vào tường, vị trí giữ nguyên và nhận phạt va chạm.

### 3.3. Hàm Xác suất Chuyển trạng thái (Transition Probability $\mathcal{P}$)
$$\mathcal{P}(s_{t+1} \mid s_t, a_t^H, a_t^P)$$
- Vị trí mới của mỗi tác nhân được cập nhật dựa trên hành động được chọn và bản đồ vật lý (nếu không va tường).
- Môi trường mang tính chất xác định (Deterministic) hoặc có thể bổ sung yếu tố ngẫu nhiên (Stochastic slip probability $\epsilon$) để tăng độ thử thách.

### 3.4. Hàm Phần thưởng (Reward Function $\mathcal{R}$)
Hàm phần thưởng được thiết kế nhằm khuyến khích tối đa hóa mục tiêu đối lập:

#### Cho Kẻ săn mồi (Hunter - $R_t^H$):
- **Bắt được con mồi (Terminal)**: $+100$ điểm (khi tọa độ trùng nhau: $(x_H, y_H) = (x_P, y_P)$).
- **Phạt thời gian (Step penalty)**: $-1$ điểm/bước (thúc đẩy bắt mồi nhanh nhất có thể).
- **Phạt va vào tường**: $-5$ điểm.
- **Thưởng định hướng (Reward Shaping - tùy chọn)**: Thưởng dương nhỏ nếu khoảng cách Manhattan tới Prey giảm: $+0.5 \times (d_{t-1} - d_t)$.

#### Cho Con mồi (Prey - $R_t^P$):
- **Bị bắt (Terminal)**: $-100$ điểm.
- **Sống sót mỗi bước**: $+1$ điểm (khuyến khích kéo dài trận đấu).
- **Sống sót hết thời gian quy định ($T_{\text{max}}$ - Terminal)**: $+50$ điểm.
- **Phạt va vào tường**: $-5$ điểm.
- **Ăn được thức ăn / vật phẩm (nếu có)**: $+10$ điểm.
- **Thưởng né tránh (Reward Shaping - tùy chọn)**: Thưởng dương nhỏ nếu khoảng cách tới Hunter tăng lên: $+0.5 \times (d_t - d_{t-1})$.

### 3.5. Hệ số Chiết khấu (Discount Factor $\gamma$)
- $\gamma \in [0.95, 0.99]$: Đảm bảo các hành động dài hạn (lập kế hoạch đường đi, phục kích, chạy vòng mê cung) được coi trọng.

---

## 4. Phân công Công việc Cho Nhóm 6 Thành Viên (Tối ưu theo Đa Thuật toán RL)

Để tận dụng tối đa thế mạnh của nhóm 6 người và triển khai nhiều thuật toán RL (Q-Learning, SARSA, DQN, PPO) phục vụ so sánh đối chuẩn (Benchmarking), công việc được phân chia theo cấu trúc mô-đun rõ ràng:

```
┌─────────────────────────────────────────────────────────────┐
│              Thành viên 1: Trưởng nhóm & Môi trường         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Cung cấp Gym/PettingZoo API
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌────────────────┐    ┌─────────────────┐    ┌────────────────┐
│   Nhánh 1:     │    │    Nhánh 2:     │    │   Nhánh 3:     │
│  Tabular RL    │    │    Value DRL    │    │   Policy DRL   │
│(Q-Learn/SARSA) │    │      (DQN)      │    │     (PPO)      │
│  Thành viên 2  │    │  Thành viên 3   │    │  Thành viên 4  │
└───────┬────────┘    └────────┬────────┘    └────────┬───────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼ Model Checkpoints
┌─────────────────────────────────────────────────────────────┐
│  Thành viên 5: Pipeline Đối kháng, Benchmarking & Thống kê  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Dữ liệu thi đấu & Đồ thị
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Thành viên 6: Giao diện 2D (Pygame/Web), Human Play & Báo cáo│
└─────────────────────────────────────────────────────────────┘
```

### Bảng Phân công Chi tiết:

| STT | Thành viên & Vai trò | Trọng tâm Nghiên cứu & Kỹ thuật | Nhiệm vụ cụ thể | Deliverables / Sản phẩm bàn giao |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Trưởng nhóm & Kỹ sư Môi trường** *(Env & Core Architect)* | - PettingZoo / Gymnasium<br>- Thuật toán sinh mê cung | - Xây dựng lõi môi trường mô phỏng mê cung 2D (Grid-world).<br>- Lập trình thuật toán sinh bản đồ động (DFS/Prim/Kruskal).<br>- Xây dựng luật va chạm, FOV (tầm nhìn toàn phần/hạn chế), hàm sinh tọa độ.<br>- Thiết lập chuẩn hóa API chung (Observation & Action space) cho cả 5 thành viên còn lại. | - `environment/maze_env.py`<br>- `environment/maze_generator.py`<br>- Tài liệu API Specs chuẩn cho nhóm. |
| **2** | **Kỹ sư RL 1: Nhóm Tabular (Baseline)** *(Q-Learning & SARSA)* | - Q-Learning (Off-policy)<br>- SARSA (On-policy)<br>- Rule-based (A* Search) | - Lập trình tác nhân theo phương pháp bảng Q-Table.<br>- Cài đặt thuật toán đường đi kinh điển A* Search làm Baseline mẫu để so sánh.<br>- Thử nghiệm Hunter/Prey với Q-Learning vs SARSA trên các bản đồ nhỏ.<br>- Khảo sát tốc độ hội tụ và độ nhớ của bảng Q. | - `agents/tabular_q.py`<br>- `agents/sarsa.py`<br>- `agents/a_star_baseline.py`<br>- Notebook đánh giá hiệu năng Tabular RL. |
| **3** | **Kỹ sư RL 2: Nhóm Deep Q-Network** *(DQN & biến thể)* | - DQN, Double-DQN<br>- Experience Replay<br>- Epsilon-greedy decay | - Xây dựng mạng nơ-ron Q-Network (PyTorch) xử lý vector trạng thái / lưới ma trận.<br>- Tối ưu hóa Target Network và Replay Buffer để ổn định huấn luyện.<br>- Huấn luyện Hunter (DQN) đối đầu với Rule-based Prey và lưu checkpoint.<br>- Tinh chỉnh siêu tham số (Learning rate, Discount factor, Buffer size). | - `agents/dqn_agent.py`<br>- `models/q_network.py`<br>- Checkpoints mô hình DQN huấn luyện sẵn. |
| **4** | **Kỹ sư RL 3: Nhóm Policy Gradient** *(PPO / Actor-Critic)* | - PPO (Proximal Policy Optimization)<br>- Generalized Advantage Estimator (GAE)<br>- Actor-Critic Architecture | - Xây dựng mạng Actor (Policy) và Critic (Value) bằng PyTorch.<br>- Cài đặt thuật toán PPO cắt ngưỡng (Clipped Objective) xử lý không gian trạng thái phức tạp.<br>- Huấn luyện Prey (PPO) né tránh linh hoạt, tận dụng góc khuất mê cung.<br>- Khảo sát khả năng thích nghi của PPO khi thay đổi bản đồ ngẫu nhiên. | - `agents/ppo_agent.py`<br>- `models/actor_critic.py`<br>- Checkpoints mô hình PPO huấn luyện sẵn. |
| **5** | **Kỹ sư MARL & Benchmarking** *(Self-play & Cross-Tournament)* | - Multi-Agent Self-Play<br>- Ma trận tỉ lệ thắng (Elo / Win-rate Matrix)<br>- Thống kê & Phân tích dữ liệu | - Xây dựng vòng lặp thi đấu đối kháng đa tác nhân (MARL Runner).<br>- Tạo giải đấu chéo: Đấu vòng tròn giữa các thuật toán (DQN Hunter vs PPO Prey; Q-Learning Hunter vs DQN Prey...).<br>- Thu thập log huấn luyện: Độ dài bước trung bình (Step count), Tỷ lệ săn thành công, Đồ thị Reward.<br>- Vẽ ma trận nhiệt (Heatmap) và bảng xếp hạng hiệu năng thuật toán. | - `training/train_marl.py`<br>- `evaluation/tournament.py`<br>- `evaluation/plot_metrics.py`<br>- Bảng số liệu đối soát (Benchmark Results). |
| **6** | **Kỹ sư UI/UX & Tích hợp Sản phẩm** *(Visualizer, Human Play & Docs)* | - Pygame / Streamlit<br>- Game Loop & Event Handling<br>- Kỹ thuật Viết báo cáo & Demo | - Thiết kế giao diện mô phỏng 2D sinh động (Sprite nhân vật, màu sắc, hiệu ứng bắt mồi).<br>- Tích hợp chế độ `Human vs AI`: Cho phép người chơi dùng phím mũi tên điều khiển Con mồi trốn Hunter (DQN/PPO) hoặc ngược lại.<br>- Xây dựng bảng điều khiển trực quan (Control Panel: chọn map, chỉnh tốc độ FPS, chọn mô hình nạp vào).<br>- Phụ trách hoàn thiện báo cáo khoa học, slide thuyết trình và dựng video demo. | - `ui/gui_game.py` (hoặc Web Dashboard)<br>- Video demo đối đầu trực quan<br>- Báo cáo đồ án & Slide thuyết trình. |

---

## 5. Lợi ích của Cách phân chia Đa Thuật toán này

1. **Không bị trùng lặp công việc**: Mỗi thành viên phụ trách một mảng công nghệ chuyên biệt (Môi trường, Tabular RL, Value-based Deep RL, Policy-based Deep RL, Evaluation/Tournament, UI/Product).
2. **Giá trị học thuật và thực nghiệm cao**: Thay vì chỉ dùng 1 thuật toán duy nhất, nhóm có ngay một bài báo cáo khoa học so sánh toàn diện:
   - *Tabular (Q-Learning, SARSA)* vs *Deep RL (DQN, PPO)* vs *Rule-based (A\*)*.
3. **Dễ dàng ghép nối (Loose Coupling)**: Các module agents tuân thủ chung format hàm `select_action(state)` và `update(...)`, giúp Thành viên 5 (Tournament) và Thành viên 6 (UI) dễ dàng nạp bất kỳ cặp thuật toán nào vào thi đấu mà không bị lỗi tương thích.

---

## 6. Kế hoạch Triển khai Dự kiến (Roadmap)

| Giai đoạn | Thời gian | Thành viên phụ trách | Nhiệm vụ chính | Kết quả đầu ra |
| :--- | :---: | :--- | :--- | :--- |
| **Giai đoạn 1: Chuẩn bị & Môi trường** | Tuần 1 - 2 | Cả nhóm, TV1, TV2 | - Chốt kiến trúc và phân công kỹ thuật.<br>- **TV1**: Xây dựng Core Environment (`maze_env.py`) và quy chuẩn API.<br>- **TV2**: Xây dựng thuật toán Baseline A* Search. | Module môi trường hoàn chỉnh, API Specs cho nhóm. |
| **Giai đoạn 2: Thuật toán RL** | Tuần 3 - 4 | TV2, TV3, TV4, TV6 | - **TV2**: Cài đặt & Huấn luyện Tabular RL (Q-Learning, SARSA).<br>- **TV3**: Xây dựng mạng DQN + Replay Buffer + Target Net.<br>- **TV4**: Xây dựng mạng Actor-Critic + thuật toán PPO.<br>- **TV6**: Dựng khung giao diện UI 2D sơ bộ. | Các module agent độc lập (`tabular_q.py`, `dqn_agent.py`, `ppo_agent.py`). |
| **Giai đoạn 3: Huấn luyện Đối kháng & Benchmark** | Tuần 5 - 6 | TV5, TV3, TV4 | - **TV5**: Xây dựng pipeline tự động thi đấu chéo (Tournament Runner).<br>- Huấn luyện đối kháng (Self-Play / Alternate training).<br>- Thu thập log, vẽ biểu đồ Reward, Win-rate, Heatmap. | Checkpoints mô hình tối ưu, Bảng số liệu Benchmark đa thuật toán. |
| **Giai đoạn 4: Hoàn thiện & Báo cáo** | Tuần 7 | Cả nhóm, TV6 | - **TV6**: Tích hợp các model vào UI, hoàn thiện chế độ `Human vs AI`.<br>- Cả nhóm: Hoàn thiện báo cáo kỹ thuật, slide thuyết trình và video demo. | Ứng dụng mô phỏng hoàn chỉnh, Báo cáo & Slide nghiệm thu đồ án. |

