# Thuật Toán Double Dueling Deep Q-Network (Double Dueling DQN) — Xe Tự Hành Né Vật Cản

> **Thuật toán 2 — Thành viên phụ trách:** Ngô Tuấn Anh (MSV: `2351170570`)  
> **Nhóm thực hiện:** Nhóm 1 - Lớp 65KTPM - Đại học Thủy Lợi  
> **Nền tảng tích hợp:** `highway_project/` (PyTorch + Gymnasium + Highway-Env + Pygame 60 FPS)

Tài liệu này trình bày cơ sở lý thuyết toán học, động cơ cải tiến so với DQN truyền thống, kiến trúc mạng hai nhánh (Dueling Architecture), cơ chế cập nhật mềm (Soft Target Update), so sánh đối sánh và hướng dẫn vận hành **Double Dueling DQN** trên môi trường cao tốc đa làn 2..6 làn.

---

## 1. Đặt Vấn Đề: Tại Sao Cần Nâng Cấp Từ DQN Lên Double Dueling DQN?

Trong bài toán xe tự hành trên cao tốc (`LaneObstacleEnv`), không gian trạng thái liên tục gồm **68 chiều** (18 đặc trưng radar đa làn + 50 đặc trưng kinematics của 10 xe xung quanh) và không gian hành động rời rạc gồm **5 hành động**:

$$\mathcal{A} = \{0: \text{LANE\_LEFT},\; 1: \text{IDLE},\; 2: \text{LANE\_RIGHT},\; 3: \text{FASTER},\; 4: \text{SLOWER}\}$$

Thuật toán **DQN nguyên bản** (Mnih et al., 2015) gặp 3 hạn chế lớn trong bối cảnh lái xe tốc độ cao:

1. **Thiên kiến ước lượng phóng đại (Overestimation Bias):**
   Toán tử $\max_{a'} Q(s', a'; \theta^-)$ vừa chọn hành động tốt nhất vừa dùng chính mạng đó để chấm điểm, khiến sai số dương bị khuếch đại liên tục:
   $$\mathbb{E}\left[\max_{a'} Q(s', a')\right] \ge \max_{a'} \mathbb{E}\left[Q(s', a')\right]$$
   Hệ quả là xe tự tin thái quá vào các tình huống nguy hiểm (ví dụ vượt gấp qua khoảng trống hẹp).

2. **Không tách biệt giá trị tình huống và lợi thế hành động:**
   Trên đường cao tốc, có rất nhiều thời điểm **đường hoàn toàn thông thoáng** (phía trước trống $> 150\text{ m}$). Khi đó, việc giữ làn (`IDLE`), tăng nhẹ tốc độ (`FASTER`) hay đi làn khác đều an toàn gần như nhau — giá trị cốt lõi phụ thuộc vào **bản thân trạng thái $s$** chứ không phải từng hành động riêng lẻ. Mạng MLP đơn nhánh phải học lại toàn bộ $Q(s, a)$ riêng rẽ cho cả 5 hành động nên hội tụ chậm hơn và dễ nhiễu.

3. **Dao động khi cập nhật cứng mạng Target (Hard Target Update):**
   Sao chép cứng toàn bộ trọng số $\theta^- \leftarrow \theta$ sau mỗi $C$ bước (ví dụ 400 hoặc 1000 bước) tạo ra các "cú nhảy bậc" (step jumps) ở nhãn mục tiêu TD, có thể gây dao động loss tạm thời.

**Double Dueling DQN** giải quyết đồng thời cả 3 vấn đề trên bằng sự kết hợp của 3 kỹ thuật nền tảng.

---

## 2. Cơ Sở Toán Học Của Double Dueling DQN

### 2.1. Trụ Cột 1: Kiến Trúc Mạng Dueling (Wang et al., 2016)

Thay vì xuất trực tiếp vector $Q(s, a)$ từ một đầu ra duy nhất, mạng nơ-ron tách thành **2 nhánh chuyên biệt** sau khối trích xuất đặc trưng chung $\phi(s; \theta_{\text{feat}})$:

* **Nhánh Giá trị trạng thái (State Value Stream) $V(s; \theta_{\text{val}}) \in \mathbb{R}$:** Ước lượng mức độ tốt/xấu tổng quát của tình huống giao thông hiện tại $s$ (ví dụ: đường thoáng vs sắp kẹt giữa 2 xe).
* **Nhánh Lợi thế hành động (Action Advantage Stream) $A(s, a; \theta_{\text{adv}}) \in \mathbb{R}^{|\mathcal{A}|}$:** Đánh giá hành động $a$ tốt hơn hay tệ hơn bao nhiêu so với mặt bằng chung các hành động khác tại cùng trạng thái $s$.

#### Vấn đề không định danh (Unidentifiability) và công thức chuẩn hóa:
Nếu chỉ cộng trực tiếp $Q(s, a) = V(s) + A(s, a)$, với mọi hằng số $c$, cặp $(V(s) + c,\; A(s, a) - c)$ đều cho ra cùng một giá trị $Q(s, a)$ $\Rightarrow$ không thể xác định duy nhất $V$ và $A$ khi lan truyền ngược, gây mất ổn định huấn luyện.

Để khắc phục, mạng áp dụng phép **trừ giá trị trung bình của nhánh Advantage (Mean Centering)**:

$$Q(s, a; \theta) = V(s; \theta_{\text{feat}}, \theta_{\text{val}}) + \left( A(s, a; \theta_{\text{feat}}, \theta_{\text{adv}}) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta_{\text{feat}}, \theta_{\text{adv}}) \right)$$

Tính chất toán học quan trọng:
$$\frac{1}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} Q(s, a; \theta) = V(s; \theta_{\text{feat}}, \theta_{\text{val}})$$

* **Lợi ích khi lái xe cao tốc:** Mỗi lần cập nhật gradient từ bất kỳ hành động $a_t$ nào, nhánh chung $V(s)$ đều được cập nhật — nhờ đó trạng thái $s$ được học nhanh và tổng quát hơn nhiều ngay cả với các hành động ít được chọn.

---

### 2.2. Trụ Cột 2: Double Q-Learning (van Hasselt et al., 2016)

Để triệt tiêu Overestimation Bias, quá trình tính nhãn mục tiêu TD được tách làm 2 bước độc lập giữa mạng chính (Online $\theta$) và mạng đích (Target $\theta^-$):

1. **Chọn hành động tốt nhất ở trạng thái kế tiếp bằng mạng Online $\theta$:**
   $$a^* = \arg\max_{a' \in \mathcal{A}} Q(s'_i, a'; \theta)$$
2. **Định giá trị cho hành động $a^*$ bằng mạng Target $\theta^-$:**
   $$y_i = r_i + (1 - d_i) \cdot \gamma \cdot Q\left(s'_i,\; a^*;\; \theta^-\right)$$

Trong đó:
* $\gamma = 0.98$: Hệ số chiết khấu dài hạn giúp xe nhìn xa để chuyển làn sớm trước vật cản tĩnh.
* $d_i \in \{0, 1\}$: Chỉ bằng $1$ khi `terminated == True` và `truncated == False` (va chạm hoặc cán đích). Nếu chỉ hết thời gian (`truncated`), $d_i = 0$ để tiếp tục bootstrap đúng chuẩn Gymnasium.

---

### 2.3. Trụ Cột 3: Hàm Mất Mát Huber Loss & Cập Nhật Mềm (Soft Target Update)

#### Hàm mất mát Smooth L1 (Huber Loss):
Với sai số TD $\delta_i = y_i - Q(s_i, a_i; \theta)$:
$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B \mathcal{L}_{\text{Huber}}(\delta_i), \quad \mathcal{L}_{\text{Huber}}(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{nếu } |\delta| \le 1 \\ |\delta| - \frac{1}{2} & \text{nếu } |\delta| > 1 \end{cases}$$

Kết hợp **Gradient Clipping** ($\|\nabla_\theta \mathcal{L}\|_2 \le 10.0$) và bộ tối ưu **Adam** ($lr = 3 \times 10^{-4}$) giúp chống nổ gradient khi xe nhận điểm phạt va chạm lớn.

#### Cập nhật mềm Target Network (Polyak Averaging):
Thay vì đợi $400$ bước mới sao chép cứng, sau **mỗi bước huấn luyện**, mạng đích dịch chuyển mượt mà theo mạng chính với hệ số $\tau = 0.005$:

$$\theta^- \leftarrow \tau \cdot \theta + (1 - \tau) \cdot \theta^-$$

* **Tác dụng:** Mục tiêu $y_i$ thay đổi liên tục nhưng cực kỳ êm ái (tương đương trung bình trượt hàm mũ với cửa sổ hiệu dụng $\approx \frac{1}{\tau} = 200$ bước), giúp đường cong học tập ít dao động hơn so với Hard Update.

---

## 3. Sơ Đồ Kiến Trúc Mạng `DuelingQNetwork`

```text
               State Vector Input s_t [Dim = 68]
        (18 đặc trưng Radar ego-centric + 50 Kinematics)
                               │
                               ▼
              ┌─────────────────────────────────┐
              │  Feature Trunk (feature_net)    │
              │  Linear(68 -> 128) + ReLU       │  (Kaiming Normal)
              │  Linear(128 -> 128) + ReLU      │  (Kaiming Normal)
              └───────────────┬─────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   Value Stream V(s)      │    │   Advantage Stream A(s, a)   │
│   Linear(128 -> 64)+ReLU │    │   Linear(128 -> 64) + ReLU   │
│   Linear(64 -> 1)        │    │   Linear(64 -> 5)            │
│   Output: (B, 1)         │    │   Output: (B, 5)             │
└─────────────┬────────────┘    └──────────────┬───────────────┘
              │                                │
              │        ┌───────────────────────┴───────────────┐
              │        ▼                                       ▼
              │   A(s, a)                          mean_a(A(s, a))
              │        └───────────────┬───────────────────────┘
              │                        ▼
              │              A(s, a) - mean(A)
              └──────────┬─────────────┘
                         ▼
         Q(s, a) = V(s) + (A(s, a) - mean(A))   [Output: (B, 5)]
                         │
                         ▼
        [Q(s, LANE_LEFT), Q(s, IDLE), Q(s, LANE_RIGHT),
         Q(s, FASTER), Q(s, SLOWER)]
```

---

## 4. So Sánh Đối Sánh Các Thuật Toán Trong `highway_project`

| Tiêu chí | Tabular Q-Learning (`q_agent.py`) | Thuật toán 1: Double DQN (`dqn_agent.py` - Dương Quốc Vượng) | Thuật toán 2: Double Dueling DQN (`ddqn_agent.py` - Ngô Tuấn Anh) |
| :--- | :--- | :--- | :--- |
| **Biểu diễn hàm $Q$** | Bảng rời rạc (6 biến trạng thái rời rạc hóa) | Mạng MLP đơn nhánh `68 → 128 → 128 → 5` | Mạng Dueling 2 nhánh: Trunk `68 → 128 → 128`, tách `V(s): 64→1` & `A(s,a): 64→5` |
| **Chống Overestimation** | Không ($\max$ trực tiếp) | Có (Double Q-Learning) | Có (Double Q-Learning) |
| **Tách $V(s)$ và $A(s,a)$** | Không | Không | **Có** (học giá trị trạng thái nhanh khi đường thoáng) |
| **Cập nhật Target Net** | Không có | **Hard Update** mỗi `400` bước | **Soft Update** (Polyak) mỗi bước với $\tau = 0.005$ |
| **Learning Rate ($lr$)** | `0.25` | `5e-4` | `3e-4` |
| **Hỗ trợ 2..6 làn động** | Hạn chế (3 làn) | Có (chiếu 3 kênh ego-centric 68 chiều) | **Có** (kế thừa chiếu 3 kênh ego-centric 68 chiều) |
| **Checkpoint mặc định** | `models/q_learning.npz` | `models/dqn_model.pth` | `models/ddqn_model.pth` |

---

## 5. Đặc Tả Đầu Vào — Đầu Ra & Phạm Vi Áp Dụng

### 5.1. Đặc Tả Đầu Vào (Input) & Đầu Ra (Output)

* **Đầu vào (State Input — 68 chiều `float32`):**
  * **18 đặc trưng Radar ego-centric (`index 0..17`):** vị trí làn chuẩn hóa, vận tốc tức thời, cờ đang chuyển làn, khoảng cách trước/sau (`fg`, `rg`) và tốc độ tiếp cận trước/sau (`fc`, `rc`) trên 3 kênh `[Làn trái, Làn hiện tại, Làn phải]` (tự động thích ứng cho cao tốc 2..6 làn), cùng 3 cờ an toàn (`danger`, `left_clear`, `right_clear`).
  * **50 đặc trưng Kinematics (`index 18..67`):** ma trận `10 × 5` của 10 phương tiện lân cận gần nhất (`[presence, x, y, vx, vy]`).
* **Đầu ra mạng nơ-ron (`DuelingQNetwork`):**
  * Nhánh Value $V(s) \in \mathbb{R}^{B \times 1}$ và nhánh Advantage $A(s, a) \in \mathbb{R}^{B \times 5}$ → hợp nhất thành vector 5 giá trị $Q(s, a) \in \mathbb{R}^{B \times 5}$.
* **Đầu ra quyết định (`DoubleDuelingDQNAgent.act`):**
  * Chỉ số hành động nguyên $a_t \in \{0: \text{LANE\_LEFT},\; 1: \text{IDLE},\; 2: \text{LANE\_RIGHT},\; 3: \text{FASTER},\; 4: \text{SLOWER}\}$ sau khi đi qua Action Masking và lớp an toàn chủ động.
* **Đầu ra huấn luyện (`train_ddqn.py`):**
  * Trọng số mô hình [`highway_project/models/ddqn_model.pth`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/models/ddqn_model.pth) và báo cáo đánh giá [`highway_project/results/ddqn_report.json`](file:///d:/PTAPP/Reinforcement-Learning/highway_project/results/ddqn_report.json) (**100% đến đích, 0% va chạm** trên cả 3 mức giao thông 0 / 6 / 12 xe với tốc độ trung bình $95.3 - 99.8\text{ km/h}$).

### 5.2. Ưu Điểm & Điều Kiện Áp Dụng

#### Ưu điểm nổi bật
* **Hội tụ ổn định hơn trong môi trường nhiều trạng thái "trung tính":** Trên cao tốc, phần lớn các bước chạy là đi thẳng trên làn trống; nhánh $V(s)$ đánh giá nhanh mức độ an toàn của trạng thái mà không cần thử hết mọi hành động.
* **Ít dao động TD Target:** Nhờ Soft Update ($\tau = 0.005$) kết hợp Smooth L1 Loss và Gradient Clipping.
* **Tốc độ suy luận thời gian thực:** Dưới $1\text{ ms}$/bước trên CPU, đảm bảo giao diện Pygame mượt 60 FPS ở tần số điều khiển 10 Hz.

#### Hạn chế & Điều kiện áp dụng
* Chỉ áp dụng cho **không gian hành động rời rạc** (5 hành động chiến thuật); không dùng trực tiếp để điều khiển góc đánh lái hoặc lực phanh liên tục (cần chuyển sang PPO/SAC/TD3).
* Số tham số nhỉnh hơn mạng MLP đơn nhánh một chút do có 2 nhánh ẩn `64` nơ-ron sau khối đặc trưng chung.

---

## 6. Hướng Dẫn Huấn Luyện, Chạy & Kiểm Thử

```powershell
# 1. Huấn luyện Double Dueling DQN (lưu models/ddqn_model.pth và results/ddqn_report.json)
.\.venv-highway\Scripts\python.exe -m highway_project.train_ddqn --episodes 160 --tau 0.005

# 2. Chạy mô phỏng trực tiếp với Double Dueling DQN
.\CHAY_MOI_TRUONG.cmd --algo ddqn --mode agent --auto-reset

# 3. Hoặc mở giao diện mặc định rồi bấm phím M -> Mục 6 -> chọn "Double Dueling DQN (Tuấn Anh)" để chuyển trực tiếp không reset xe
.\CHAY_MOI_TRUONG.cmd

# 4. Vẽ biểu đồ so sánh Q-Learning vs DQN vs Double Dueling DQN
.\.venv-highway\Scripts\python.exe -m highway_project.plot_results

# 5. Chạy bộ kiểm thử tự động cho Double Dueling DQN
.\.venv-highway\Scripts\python.exe -m unittest highway_project/tests/test_ddqn.py
```

> Xem thêm phần giải thích chi tiết từng dòng mã nguồn tại: [**`GIAI_THICH_CODE_DDQN.md`**](file:///d:/PTAPP/Reinforcement-Learning/docs/thuat_toan/ddqn/GIAI_THICH_CODE_DDQN.md).
