# Thuật Toán Deep Q-Network (DQN) - Xe Tự Hành Né Vật Cản

Tài liệu chi tiết về cơ sở lý thuyết toán học, nguyên lý hoạt động, các công thức giải tích, kiến trúc mô hình, lý do lựa chọn và phạm vi áp dụng của thuật toán **Deep Q-Network (Double DQN)** trong dự án xe tự hành né vật cản trên đường cao tốc 3 làn.

---

## 1. Cơ Sở Lý Thuyết Của Thuật Toán DQN

### 1.1. Khung Quyết Định Markov (Markov Decision Process - MDP)
Bài toán học tăng cường được mô hình hóa toán học thông qua một Quá trình Quyết định Markov gồm bộ 5 thành phần $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
* $\mathcal{S}$: Không gian trạng thái (State space) — Vector 68 chiều biểu diễn đặc trưng xe và giao thông xung quanh.
* $\mathcal{A}$: Không gian hành động rời rạc (Action space) — Gồm 5 hành động:
  $$\mathcal{A} = \{0: \text{LANE\_LEFT},\; 1: \text{IDLE},\; 2: \text{LANE\_RIGHT},\; 3: \text{FASTER},\; 4: \text{SLOWER}\}$$
* $\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$: Xác suất chuyển trạng thái trong môi trường động lực học xe.
* $\mathcal{R}(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$: Hàm phần thưởng phản hồi từ môi trường (thưởng giữ tốc, phạt va chạm, phạt chệch làn).
* $\gamma \in [0, 1)$: Hệ số chiết khấu tương lai (trong bài toán này $\gamma = 0.98$).

Tổng phần thưởng tích lũy có chiết khấu kể từ thời điểm $t$ được định nghĩa là:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$$

---

### 1.2. Hàm Giá Trị Hành Động (Action-Value Function / Q-Function)
Hàm giá trị hành động $Q^\pi(s, a)$ biểu thị kỳ vọng phần thưởng tích lũy khi tác nhân bắt đầu từ trạng thái $s$, thực hiện hành động $a$, và sau đó tuân theo chính sách $\pi$:
$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \;\middle|\; S_t = s, A_t = a \right]$$

Mục tiêu của Học tăng cường là tìm ra chính sách tối ưu $\pi^*$ sao cho tối đa hóa hàm $Q$:
$$Q^*(s, a) = \max_\pi Q^\pi(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$$

Hàm $Q^*(s, a)$ thỏa mãn **Phương trình Bellman Tối ưu (Bellman Optimality Equation)**:
$$Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{P}} \left[ R_{t+1} + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \;\middle|\; S_t = s, A_t = a \right]$$

---

### 1.3. Từ Q-Learning Dạng Bảng Đến Deep Q-Network (DQN)

#### Quy tắc cập nhật Q-learning truyền thống (Tabular Q-learning):
Trong Q-learning cổ điển, các giá trị $Q(s, a)$ được lưu trong một bảng tra cứu và cập nhật theo phương pháp sai phân thời gian TD (Temporal Difference):
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')}_{\text{Mục tiêu TD (TD Target)}} - \underbrace{Q(S_t, A_t)}_{\text{Ước lượng hiện tại}} \right]$$
Trong đó $\alpha \in (0, 1]$ là tốc độ học (learning rate).

#### Chuyển dịch sang Mạng Nơ-ron Sâu (DQN):
Khi không gian trạng thái $\mathcal{S}$ là liên tục và nhiều chiều, việc lưu bảng là bất khả thi. DQN (Mnih et al., DeepMind 2013/2015) thay thế bảng $Q$ bằng một **Mạng nơ-ron sâu xấp xỉ hàm (Function Approximator)** với bộ trọng số $\theta$:
$$\hat{Q}(s, a; \theta) \approx Q^*(s, a)$$

Tuy nhiên, việc kết hợp trực tiếp mạng nơ-ron với Q-learning tiêu chuẩn sẽ gặp phải hiện tượng **phân kỳ và mất ổn định nghiêm trọng** do:
1. **Dữ liệu quan sát có tương quan chuỗi thời gian cao**: Các trạng thái liên tiếp $(s_t, s_{t+1})$ trong lái xe rất giống nhau, vi phạm giả định mẫu độc lập phân phối đều (i.i.d.).
2. **Mục tiêu thay đổi liên tục (Moving Target)**: Vì cùng dùng $\theta$ để tính cả giá trị dự đoán $Q(s, a; \theta)$ và giá trị mục tiêu $r + \gamma \max_{a'} Q(s', a'; \theta)$, mỗi lần cập nhật $\theta$ sẽ làm mục tiêu dịch chuyển, dẫn đến vòng lặp dao động hoặc phát nổ giá trị.

---

## 2. Cách Hoạt Động Của Thuật Toán DQN

DQN giải quyết triệt để sự mất ổn định bằng 2 cơ chế đột phá: **Bộ đệm kinh nghiệm (Experience Replay)** và **Mạng đích tách rời (Target Network)**.

```
                              ┌───────────────────────────────────┐
                              │     Môi trường (HighwayEnv)       │
                              └───────┬───────────────────▲───────┘
                     Observation s_t, │                   │ Action a_t
                     Reward r_t       ▼                   │ (epsilon-greedy)
                              ┌───────────────────────────┴───────┐
                              │     Tác nhân lái xe (Agent)       │
                              └───────┬───────────────────────────┘
               Lưu trữ transition     │
               (s, a, r, s', done)    ▼
                        ┌───────────────────────────────┐
                        │   Replay Buffer D (N mẫu)     │
                        └─────────────┬─────────────────┘
                                      │ Lấy mẫu ngẫu nhiên Mini-batch
                                      ▼ B = {(s_i, a_i, r_i, s'_i, d_i)}
                     ┌───────────────────────────────────┐
                     │          Huấn luyện DQN           │
                     │                                   │
                     │  Mạng Online Q(s, a; θ)           │
                     │         ▲                         │
                     │         │ Cập nhật gradient       │
                     │   Huber Loss L(θ)                 │
                     │         ▲                         │
                     │  Mạng Target Q(s', a'; θ^-)       │
                     │  (Đóng băng, cập nhật mềm/cứng)   │
                     └───────────────────────────────────┘
```

### 2.1. Cơ Chế 1: Bộ Đệm Kinh Nghiệm (Experience Replay Buffer)
* Bộ đệm $\mathcal{D}$ có sức chứa tối đa $N$ mẫu (trong dự án: $50\,000$ bước chuyển).
* Tại mỗi bước thời gian $t$, sau khi thực hiện hành động $a_t$, tác nhân ghi nhận tuple kinh nghiệm:
  $$e_t = (s_t, a_t, r_t, s_{t+1}, d_t)$$
  với $d_t \in \{0, 1\}$ là cờ kết thúc episode (*terminated* do va chạm hoặc về đích).
* Khi huấn luyện, thuật toán lấy mẫu một mini-batch ngẫu nhiên kích thước $B$ từ $\mathcal{D}$:
  $$\mathcal{B} = \{e_i\}_{i=1}^B \sim U(\mathcal{D})$$
* **Tác dụng toán học**: Xóa bỏ hoàn toàn tương quan chuỗi thời gian giữa các mẫu liên tiếp, tái sử dụng dữ liệu quý giá nhiều lần và biến bài toán tối ưu về dạng học có giám sát trên phân phối dữ liệu dừng (stationary distribution).

---

### 2.2. Cơ Chế 2: Mạng Đích Tách Rời (Target Network)
* Sử dụng hai mạng nơ-ron có cùng cấu trúc đồ hình:
  1. **Mạng Online** mang trọng số $\theta$: Dùng để dự đoán giá trị $Q(s, a; \theta)$ hiện tại và được cập nhật gradient liên tục ở mỗi bước học.
  2. **Mạng Target** mang trọng số $\theta^-$: Dùng để tính toán nhãn mục tiêu (TD Target). Trọng số $\theta^-$ được giữ cố định (đóng băng) và chỉ cập nhật đồng bộ sau mỗi chu kỳ.
* **Quy tắc cập nhật mạng Target**:
  - *Cập nhật cứng (Hard update)*: Định kỳ sau mỗi $C$ bước huấn luyện:
    $$\theta^- \leftarrow \theta$$
  - *Cập nhật mềm (Soft update / Polyak averaging)* với hệ số $\tau \ll 1$:
    $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

---

### 2.3. Cải Tiến Double DQN: Loại Bỏ Thiên Kiến Thổi Phồng (Overestimation Bias)
Trong DQN nguyên bản của Nature 2015, mục tiêu TD được tính theo công thức:
$$y_i^{\text{DQN}} = r_i + (1 - d_i) \cdot \gamma \max_{a' \in \mathcal{A}} Q(s'_i, a'; \theta^-)$$
Toán tử $\max$ này dẫn đến **thiên kiến thổi phồng giá trị Q (Overestimation bias)** vì:
$$\mathbb{E}\left[\max_{a'} Q(s', a')\right] \ge \max_{a'} \mathbb{E}\left[Q(s', a')\right]$$
Khi mạng nơ-ron có nhiễu xấp xỉ, việc lấy $\max$ sẽ ưu tiên chọn các giá trị bị ước lượng quá cao, làm xe tự tin thái quá vào các tình huống nguy hiểm.

**Giải pháp của Double DQN (van Hasselt et al., 2016)** được áp dụng trong dự án:
* **Tách rời việc chọn hành động và việc lượng giá hành động**:
  1. Dùng **Mạng Online $\theta$** để chọn hành động tốt nhất ở trạng thái tiếp theo:
     $$a^* = \arg\max_{a' \in \mathcal{A}} Q(s'_i, a'; \theta)$$
  2. Dùng **Mạng Target $\theta^-$** để lượng giá điểm số của hành động $a^*$ đó:
     $$y_i^{\text{DoubleDQN}} = r_i + (1 - d_i) \cdot \gamma Q\left(s'_i, \arg\max_{a'} Q(s'_i, a'; \theta);\, \theta^-\right)$$

---

### 2.4. Hàm Mất Mát và Lan Truyền Ngược (Loss Function & Gradient Descent)

#### Sai số sai phân thời gian (TD Error):
$$\delta_i = y_i^{\text{DoubleDQN}} - Q(s_i, a_i; \theta)$$

#### Hàm mất mát Huber Loss (Smooth L1 Loss):
Thay vì sử dụng hàm tổn thất toàn phương (MSE Loss) dễ bị nổ gradient khi xe va chạm bất ngờ (phần thưởng phạt rất lớn: $-20$), dự án áp dụng **Huber Loss**:
$$\mathcal{L}_\delta(\delta_i) = \begin{cases} 
\frac{1}{2} \delta_i^2 & \text{nếu } |\delta_i| \le \delta_{\text{huber}} \\ 
\delta_{\text{huber}} \cdot \left( |\delta_i| - \frac{1}{2} \delta_{\text{huber}} \right) & \text{ngược lại} 
\end{cases}$$
với $\delta_{\text{huber}} = 1.0$.

* Đạo hàm của Huber Loss đối với sai số $\delta_i$:
  $$\frac{\partial \mathcal{L}_\delta}{\partial \delta_i} = \begin{cases} 
  \delta_i & \text{nếu } |\delta_i| \le 1 \\ 
  \operatorname{sign}(\delta_i) & \text{ngược lại} 
  \end{cases}$$
* **Ý nghĩa thực nghiệm**: Khi sai số nhỏ, hàm hành xử mượt mà như bậc hai ($L_2$); khi sai số đột biến lớn (xe đâm vật cản), gradient được kẹp cứng về $\pm 1$ ($L_1$), giúp việc tối ưu hóa mạng nơ-ron luôn ổn định tuyệt đối.

#### Cập nhật trọng số qua thuật toán tối ưu AdamW:
Gradient hàm mất mát toàn cục trên mini-batch:
$$\nabla_\theta L(\theta) = \frac{1}{B} \sum_{i=1}^B \frac{\partial \mathcal{L}_\delta(\delta_i)}{\partial \theta}$$
$$\theta \leftarrow \theta - \eta \cdot \operatorname{AdamW}\left( \nabla_\theta L(\theta) \right)$$
trong đó $\eta$ là tốc độ học ($\eta = 5 \times 10^{-4}$).

---

### 2.5. Chiến Lược Thăm Dò $\epsilon$-Greedy (Exploration vs Exploitation)
Để cân bằng giữa việc khám phá các quỹ đạo chuyển làn mới và khai thác chính sách tối ưu hiện tại, tác nhân tuân theo chiến lược $\epsilon$-greedy:

$$\pi(a \mid s) = \begin{cases} 
\arg\max_{a \in \mathcal{A}_{\text{valid}}} Q(s, a; \theta) & \text{với xác suất } 1 - \epsilon \\ 
\text{chọn ngẫu nhiên một hành động trong } \mathcal{A}_{\text{valid}} & \text{với xác suất } \epsilon 
\end{cases}$$

Quy luật suy giảm epsilon theo từng episode $e$:
$$\epsilon_{e+1} = \max\left(\epsilon_{\text{min}},\, \epsilon_e \cdot \epsilon_{\text{decay}}\right)$$
* Khởi đầu: $\epsilon_0 = 1.0$ (hoàn toàn thăm dò).
* Hệ số suy giảm: $\epsilon_{\text{decay}} = 0.985$.
* Ngưỡng tối thiểu: $\epsilon_{\text{min}} = 0.05$.
* Khi đánh giá (Evaluation/Inference): Thiết lập $\epsilon = 0.0$ (hoàn toàn khai thác hành động tốt nhất).

---

## 3. Tại Sao Thuật Toán DQN Phù Hợp Với Bài Toán Này?

Bài toán điều khiển xe tự hành di chuyển $1000\text{ m}$ trên đường cao tốc 3 làn với $20$ xe giao thông ngẫu nhiên và chướng ngại vật tĩnh (xe hỏng dừng khẩn cấp $v = 0\text{ km/h}$) đặt ra những thách thức đặc biệt:

```
                  ┌──────────────────────────────────────────────┐
                  │    Không gian trạng thái liên tục (68 chiều)   │
                  │  (Khoảng cách radar, TTC, vận tốc, vị trí xe) │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │  Mạng nơ-ron DQN  │
                               │  (Xấp xỉ hàm Q)   │
                               └─────────┬─────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │      Không gian hành động rời rạc (5 lệnh)    │
                  │   [0: Trái, 1: Giữ, 2: Phải, 3: Ga, 4: Phanh]│
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │      Hệ thống an toàn chủ động (Anti-stall)   │
                  │   (Khóa phanh khi đổi làn, bù ga khi v thấp) │
                  └──────────────────────────────────────────────┘
```

### 3.1. Giải quyết triệt để "Lời nguyền số chiều" (Curse of Dimensionality)
* **Hạn chế của Q-learning dạng bảng (Tabular)**:
  * Trong môi trường cao tốc, không gian quan sát gồm tọa độ $(x, y)$ và vận tốc $(v_x, v_y)$ của 10 xe xung quanh, cộng với 18 đặc trưng cảm biến radar (khoảng cách trước/sau, tốc độ tiếp cận trên cả 3 làn).
  * Nếu dùng bảng Q, ta buộc phải rời rạc hóa (discretize) các giá trị liên tục này thành từng khoảng rời rạc (ví dụ: khoảng cách chia thành 4 mức, vận tốc chia thành 3 mức).
  * Việc này làm bùng nổ kích thước bảng Q lên **hàng chục triệu trạng thái**. Trong đó, phần lớn các ô nhớ không bao giờ được ghé thăm trong quá trình học. Xe sẽ rơi vào tình trạng "học vẹt": chỉ biết xử lý các trường hợp mẫu đã thấy, khi gặp kịch bản xe ngẫu nhiên mới thì xe xử lý rất kém hoặc đâm va.
* **Sức mạnh của Deep Q-Network**:
  * Thay vì lưu bảng điểm, DQN dùng một mạng nơ-ron sâu làm **bộ xấp xỉ hàm (Function Approximator)**:
    $$\hat{Q}(s, a; \theta) \approx Q^*(s, a)$$
  * Nhờ khả năng **tổng quát hóa (Generalization)**, mạng nơ-ron có thể nhận đầu vào là các giá trị liên tục chính xác (ví dụ: khoảng cách $34.7\text{ m}$, vận tốc $88.5\text{ km/h}$, thời gian va chạm TTC $1.8\text{ s}$) và tự động nội suy từ kinh nghiệm đã học để đưa ra quyết định tối ưu.

### 3.2. Phù hợp hoàn hảo với không gian hành động rời rạc cấp cao (Meta-actions)
* Xe tự hành trong môi trường [HighwayEnv](file:///d:/Hoc_Tang_Cuong/highway_project/environment) được điều khiển ở cấp độ chiến thuật (Tactical Decision Making) với 5 hành động rời rạc:
  1. `0: LANE_LEFT` — Chuyển làn sang trái.
  2. `1: IDLE` — Giữ làn và duy trì vận tốc mục tiêu.
  3. `2: LANE_RIGHT` — Chuyển làn sang phải.
  4. `3: FASTER` — Tăng vận tốc mục tiêu (+5 m/s).
  5. `4: SLOWER` — Giảm vận tốc mục tiêu (-5 m/s).
* DQN sinh ra chính là để giải quyết bài toán: **Không gian trạng thái liên tục / hình ảnh + Không gian hành động rời rạc**.
* Với số lượng hành động nhỏ ($|\mathcal{A}| = 5$), phép tính chọn hành động tối ưu:
  $$a^* = \arg\max_{a \in \mathcal{A}} Q(s, a; \theta)$$
  được tính toán tức thì ($< 1\text{ ms}$ trên CPU), hoàn toàn đáp ứng thời gian thực cho tần số điều khiển 10 Hz và hiển thị mượt mà 60 FPS.

### 3.3. Tính chất Off-Policy và Khả năng khởi tạo mồi (Warmup Buffer)
* Do DQN là thuật toán **Off-policy**, mạng có thể học từ các dữ liệu do bất kỳ chính sách nào tạo ra.
* Dự án tận dụng đặc tính này để **mồi bộ đệm kinh nghiệm (Replay Buffer Warmup)**: Chạy 35 episodes bằng thuật toán Q-learning cũ để thu thập trước các tình huống né vật cản an toàn cơ bản, giúp mạng nơ-ron không phải bắt đầu từ con số không và tránh các pha va chạm vô nghĩa ở giai đoạn đầu.

---

## 4. Phạm Vi Áp Dụng Của Thuật Toán DQN

DQN là thuật toán nền tảng mang tính bước ngoặt trong lịch sử Học tăng cường sâu. Dưới đây là phân định rõ ràng về phạm vi áp dụng:

### 4.1. Ma trận so sánh khi nào NÊN và KHÔNG NÊN dùng DQN

| Tiêu chí | DQN phù hợp (Nên dùng) | DQN không phù hợp (Nên dùng thuật toán khác) |
| :--- | :--- | :--- |
| **Không gian hành động** | **Rời rạc (Discrete Actions)**: Số lựa chọn hữu hạn từ vài hành động đến vài chục hành động. | **Liên tục (Continuous Actions)**: Góc đánh lái $12.35^\circ$, độ mở bướm ga $42.5\%$, lực phanh theo Newton. *(Cần dùng Actor-Critic: DDPG, TD3, SAC, PPO)*. |
| **Không gian trạng thái** | **Liên tục, Đa chiều, hoặc Ảnh**: Vector cảm biến (LiDAR, Radar, GPS) hoặc ảnh từ Camera RGB. | Quá đơn giản, chỉ có 2-3 biến rời rạc *(Tabular Q-learning nhanh và nhẹ hơn)*. |
| **Tầng điều khiển** | **Quyết định chiến thuật cấp cao (Macro decisions)**: Vượt / Nhường / Đổi làn / Tăng tốc; Bật / Tắt; Chuyển mạch. | **Điều khiển chấp hành cấp thấp (Micro control)**: Vòng kín mô-men xoắn động cơ tần số hàng trăm Hz. |
| **Dữ liệu huấn luyện** | Hỗ trợ học **Off-policy**: Học từ bộ đệm kinh nghiệm, học từ dữ liệu ghi sẵn của người lái (Imitation / Warmup). | Các bài toán yêu cầu đảm bảo chính sách đơn điệu không được phân kỳ *(On-policy như PPO)*. |

### 4.2. Các lĩnh vực ứng dụng thực tế điển hình của DQN

1. **Lập kế hoạch hành vi xe tự hành (Autonomous Driving Behavior Planning)**:
   - Hệ thống xe tự hành thực tế thường phân tầng: Tầng trên (Behavior Planner) dùng DQN để quyết định *"Nên giữ làn hay vượt xe?"*, sau đó tầng dưới (Trajectory Planner & Controller như MPC/PID) sẽ tính toán góc lái và chân ga chi tiết để thực thi.
2. **Trò chơi và Thể thao điện tử (Gaming & E-Sports)**:
   - Các game có hệ thống nút bấm rời rạc: Atari 2600 (bước đột phá của DeepMind), game chiến thuật, board game.
3. **Điều khiển và Tối ưu hóa hạ tầng công nghiệp**:
   - Quản lý năng lượng tòa nhà thông minh: Bật/tắt các cụm máy lạnh (HVAC) theo biểu giá điện.
   - Quản lý trung tâm dữ liệu: Tự động điều chỉnh chế độ làm mát của server để tiết kiệm năng lượng.
4. **Mạng máy tính và Viễn thông**:
   - Định tuyến gói tin (Packet Routing): Chọn cổng giao tiếp (Interface) tối ưu để gửi gói dữ liệu nhằm tránh nghẽn mạch.
5. **Tài chính và Giao dịch thuật toán**:
   - Quản lý danh mục đầu tư: Ra quyết định rời rạc {Mua, Bán, Giữ} trên các khung thời gian cố định.

---

## 5. Kiến Trúc Mạng & Cài Đặt Chi Tiết Trong Dự Án

Trong thư mục [`highway_project/algorithms/`](file:///d:/Hoc_Tang_Cuong/highway_project/algorithms), thuật toán được cài đặt tại [`dqn_agent.py`](file:///d:/Hoc_Tang_Cuong/highway_project/algorithms/dqn_agent.py):

### 5.1. Vector Trạng Thái 68 Chiều (Observation Vector)
Vector đầu vào của mạng là sự kết hợp chặt chẽ giữa đặc trưng cảm biến Radar và ma trận động học:
1. **18 đặc trưng cảm biến Radar / Chuyên dụng**:
   - Vị trí làn đường chuẩn hóa: $\text{lane\_norm} = \text{lane} / \max(1.0, \text{lanes\_count} - 1)$.
   - Vận tốc tức thời chuẩn hóa: $v / 30.0\text{ m/s}$.
   - Trạng thái xe đang trong quỹ đạo chuyển làn (`changing` $\in \{0.0, 1.0\}$).
   - Khoảng cách tới xe phía trước $\text{fg} \in \mathbb{R}^3$ và phía sau $\text{rg} \in \mathbb{R}^3$ trên 3 kênh làn (chuẩn hóa trên tầm quét tối đa $180\text{ m}$).
   - Tốc độ tiếp cận trước $\text{fc} \in \mathbb{R}^3$ và sau $\text{rc} \in \mathbb{R}^3$ chuẩn hóa trên $30\text{ m/s}$.
   - Cờ nhị phân: Báo nguy hiểm va chạm khẩn cấp (`danger`), Làn trái an toàn (`left_clear`), Làn phải an toàn (`right_clear`).

> **Cơ chế Chiếu tương đối 3 kênh theo góc nhìn xe Ego (Ego-Centric 3-Channel Projection):**  
> Khi số làn đường cao tốc thay đổi động từ 2 đến 6 làn (`lanes_count = 2..6`), phương thức `extract_state` tự động chiếu không gian giao thông thành 3 kênh tương đối lấy xe ego làm trung tâm:
> - **Kênh 0 (Làn trái):** Nếu $\text{lane} > 0$, lấy dữ liệu làn $\text{lane} - 1$; nếu xe đang ở sát mép trái, gán khoảng cách an toàn $180\text{ m}$ và tốc độ tiếp cận $0$.
> - **Kênh 1 (Làn hiện tại):** Lấy dữ liệu cảm biến trên chính làn xe đang chạy.
> - **Kênh 2 (Làn phải):** Nếu $\text{lane} < \text{lanes\_count} - 1$, lấy dữ liệu làn $\text{lane} + 1$; nếu xe đang ở sát mép phải, gán khoảng cách an toàn $180\text{ m}$ và tốc độ tiếp cận $0$.  
> Nhờ cơ chế này, vector đầu vào luôn duy trì cố định 68 chiều, giúp mô hình DQN đã huấn luyện hoạt động ổn định và chính xác trên mọi cấu hình làn đường mà không bị lỗi kích thước tensor!

2. **50 đặc trưng động học Kinematics chuẩn của HighwayEnv**:
   - Trạng thái hiện diện, tọa độ tương đối $(x, y)$ và vận tốc tương đối $(v_x, v_y)$ của 10 phương tiện lân cận gần xe nhất ($10 \times 5 = 50$).

### 5.2. Cấu trúc Mạng Nơ-ron (QNetwork)
Mạng `QNetwork` trong `highway_project/algorithms/dqn_agent.py` được thiết kế tối ưu với 2 tầng ẩn fully-connected:
```text
State Vector Input [Dim = 68]
       │
       ▼
 ┌───────────────┐
 │ Linear Layer  │ 68 -> 128 (Khởi tạo Kaiming Normal, bias = 0)
 │ ReLU          │ Hàm kích hoạt phi tuyến
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ Linear Layer  │ 128 -> 128 (Khởi tạo Kaiming Normal, bias = 0)
 │ ReLU          │ Hàm kích hoạt phi tuyến
 └───────┬───────┘
         ▼
 ┌───────────────┐
 │ Linear Layer  │ 128 -> 5 (Khởi tạo Xavier Normal, bias = 0)
 └───────┬───────┘
         ▼
 Output: Q-values [Q(s, 0), Q(s, 1), Q(s, 2), Q(s, 3), Q(s, 4)]
```

### 5.3. Các Hệ Thống An Toàn Chủ Động Đi Kèm

#### 1. Action Masking (Khóa hành động không hợp lệ):
* Khi xe đang thực hiện chuyển làn (`changing == True`), 2 hành động rẽ trái (0) và rẽ phải (2) bị loại khỏi danh sách hành động khả dụng $\mathcal{A}_{\text{valid}} = \{1, 3, 4\}$. Điều này ngăn ngừa bộ điều khiển nhận lệnh rẽ đối nghịch liên tiếp gây lật xe hoặc mất lái.

#### 2. Cơ chế bảo toàn chuyển làn (Preserve Lane-Change):
* Khi `changing == True`, hành động 1 (`IDLE`) là lệnh để xe tiếp tục bám theo đường cong chuyển làn sang làn mới an toàn. Hệ thống cộng điểm thưởng $+15.0$ cho hành động 1 và phạt $-30.0$ cho hành động 4 (`SLOWER`) để xe không bao giờ phanh đứng yên giữa 2 làn đường.

#### 3. Hệ thống chống chết máy trên cao tốc (Anti-stall System):
* Nếu vận tốc xe bị tụt xuống dưới $55\text{ km/h}$ ($16\text{ m/s}$), hệ thống can thiệp điểm số:
  $$Q(s, \text{SLOWER}) \leftarrow Q(s, \text{SLOWER}) - 60.0$$
  $$Q(s, \text{FASTER}) \leftarrow Q(s, \text{FASTER}) + 30.0$$
* Nếu vận tốc tụt sâu dưới $43\text{ km/h}$ ($12\text{ m/s}$), mức phạt giảm tốc tăng lên $-120.0$ và bù ga $+60.0$, buộc xe phải tăng tốc đưa vận tốc về dải an toàn $70 - 108\text{ km/h}$.

#### 4. Bộ lọc làn đường nguy hiểm (Safe Lane Filter):
* Cấm tuyệt đối xe rẽ sang làn bên cạnh nếu làn đó đang có xe hỏng hoặc vật cản ở cự ly $< 35\text{ m}$ hoặc có xe phía sau lao tới với khoảng cách $< 12\text{ m}$ ($Q(s, \text{turn}) \leftarrow Q(s, \text{turn}) - 60.0$).

---

## 6. Hướng Dẫn Sử Dụng và Huấn Luyện

### 6.1. Chạy mô hình DQN đã huấn luyện
```powershell
# Chạy một chạm bằng file script duy nhất
.\CHAY_MOI_TRUONG.cmd

# Hoặc chạy từ dòng lệnh với tùy chỉnh nâng cao
.\.venv-highway\Scripts\python.exe -m highway_project.run --algo dqn --mode agent --auto-reset
```

### 6.2. Huấn luyện lại mô hình mới
```powershell
# Huấn luyện 160 episodes với Curriculum Learning (mật độ xe tăng dần từ 0 -> 12 xe)
.\.venv-highway\Scripts\python.exe -m highway_project.training.train_dqn --episodes 160 --eval-episodes 10
```

### 6.3. Xuất đồ thị hiệu năng
```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.training.plot_results
```
Đồ thị so sánh chi tiết giữa DQN và Q-learning dạng bảng được lưu tại [`highway_project/results/dqn_performance.png`](file:///d:/Hoc_Tang_Cuong/highway_project/results/dqn_performance.png).

---

## 7. Đặc Tả Giao Diện Trả Về Của Môi Trường (Return Interface Specification)

Môi trường `LaneObstacleEnv` tuân thủ nghiêm ngặt tiêu chuẩn **Gymnasium API** (`reset() -> (obs, info)` và `step(action) -> (obs, reward, terminated, truncated, info)`). Mọi thuật toán Học tăng cường (DQN, PPO, SAC, A2C hay Custom Agent) đều có thể tương tác dễ dàng thông qua đặc tả giao diện chuẩn sau:

### 7.1. Cấu Trúc Dữ Liệu `step(action)`

| Thành phần | Kiểu dữ liệu | Phạm vi giá trị | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| `obs` | `np.ndarray` (float32) | Shape `(10, 5)` | Ma trận động học 10 xe gần nhất xung quanh ego car, mỗi xe gồm 5 thuộc tính: `[presence, x, y, vx, vy]` đã được chuẩn hóa. |
| `reward` | `float` | $[ -20.0, +10.0 ]$ | Tổng điểm thưởng tức thời: thưởng duy trì tốc độ cao an toàn, phạt phanh/giảm tốc không cần thiết, phạt đổi làn liên tục, phạt tiếp cận cự ly nguy hiểm và phạt rất nặng khi va chạm hoặc lao ra khỏi mặt đường. |
| `terminated` | `bool` | `True` / `False` | Trạng thái kết thúc tập do nguyên nhân vật lý: Xe va chạm (`crashed`), xe lao ra khỏi lề đường (`off_road`), hoặc xe đã cán đích an toàn (`is_success`). |
| `truncated` | `bool` | `True` / `False` | Trạng thái ngắt tập do hết thời gian tối đa cho phép (`duration`, mặc định 90s). Nếu `truncated == True`, tác nhân vẫn chưa va chạm và có thể tiếp tục bootstrap giá trị $V(s')$. |
| `info` | `dict` | Key-value mapping | Dictionary chứa toàn bộ siêu dữ liệu vi mô (telemetry) về hành trình xe, cảm biến không gian và trạng thái kịch bản. |

---

### 7.2. Chi Tiết Từ Điển Siêu Dữ Liệu `info`

```python
info = {
    "is_success": bool,            # True nếu xe vượt qua đích mà không va chạm
    "distance_m": float,           # Quãng đường xe đã đi được tính từ vạch xuất phát (mét)
    "remaining_m": float,          # Khoảng cách còn lại tới đích (mét)
    "obstacles_passed": int,       # Số lượng chướng ngại vật xe đã vượt qua thành công
    "obstacles_total": int,        # Tổng số xe hỏng / chướng ngại vật trên đường
    "elapsed_s": float,            # Thời gian mô phỏng đã trôi qua (giây)
    "speed": float,                # Vận tốc tức thời của xe (m/s)
    "speed_kmh": float,            # Vận tốc tức thời của xe (km/h)
    "available_actions": list[int],# Danh sách các hành động hợp lệ tại bước hiện tại
    "end_reason": str,             # "collision" | "off_road" | "success" | "time_limit" | "running"
    "sensors": dict,               # Báo cáo radar cảm biến không gian chi tiết
}
```

---

### 7.3. Chi Tiết Radar Cảm Biến Không Gian `info["sensors"]`

Khối cảm biến `sensors` quét liên tục trong bán kính $180\text{ m}$ theo tất cả các làn đường, trích xuất thông số tương đối (ego-centric) giúp các mô hình RL nhận thức môi trường ngay lập tức:

| Khóa | Kiểu dữ liệu | Đơn vị | Ý nghĩa & Ứng dụng trong thuật toán |
| :--- | :--- | :--- | :--- |
| `lane` | `int` | $0 \dots N-1$ | Chỉ số làn hiện tại xe đang di chuyển ($0$ là làn sát lề trái, $N-1$ là làn lề phải). |
| `lanes_count` | `int` | $2 \dots 6$ | Tổng số làn đường của kịch bản hiện tại. |
| `traffic_density` | `str` | preset | Nhãn mật độ giao thông: `"none"`, `"sparse"`, `"normal"`, `"dense"`. |
| `speed` | `float` | $\text{m/s}$ | Vận tốc hiện tại của xe ego. |
| `speed_kmh` | `float` | $\text{km/h}$ | Vận tốc hiện tại quy đổi theo $\text{km/h}$ ($v \times 3.6$). |
| `changing` | `bool` | boolean | `True` nếu xe đang trong quỹ đạo chuyển làn (dùng để áp dụng action masking). |
| `target_lane` | `int` | $0 \dots N-1$ | Làn đích mà xe đang hướng tới trong quá trình chuyển làn. |
| `front_gap` | `list[float]` | $\text{mét}$ | Mảng khoảng cách tới xe/vật cản gần nhất phía trước cho từng làn đường (tối đa $180\text{ m}$). |
| `rear_gap` | `list[float]` | $\text{mét}$ | Mảng khoảng cách tới xe gần nhất phía sau cho từng làn đường (tối đa $180\text{ m}$). |
| `front_closing` | `list[float]` | $\text{m/s}$ | Tốc độ tiếp cận xe phía trước: $v_{\text{ego}} - v_{\text{front}}$ ($>0$ nghĩa là xe bạn đang lao nhanh hơn xe trước). |
| `rear_closing` | `list[float]` | $\text{m/s}$ | Tốc độ tiếp cận của xe phía sau: $v_{\text{rear}} - v_{\text{ego}}$ ($>0$ nghĩa là xe sau đang bám sát tới). |
| `ego_front_gap` | `float` | $\text{mét}$ | Cự ly phía trước tức thời trên chính làn xe đang chạy (tiện lợi, truy xuất trực tiếp $O(1)$). |
| `left_clear` | `bool` | boolean | `True` nếu làn bên trái trống trải an toàn (cự ly trước $>20\text{m}$, cự ly sau $>15\text{m}$). |
| `right_clear` | `bool` | boolean | `True` nếu làn bên phải trống trải an toàn (cự ly trước $>20\text{m}$, cự ly sau $>15\text{m}$). |

---

### 7.4. Mã Nguồn Mẫu Python: Kết Nối Thuật Toán Với Môi Trường

Đoạn mã sau minh họa cách một thuật toán RL độc lập (như PyTorch Custom Agent, PPO, SAC) khởi tạo môi trường, nhận quan sát, áp dụng Action Masking và cập nhật mô hình:

```python
import gymnasium as gym
import numpy as np
import torch
from highway_project.environment.environment import make_env, ACTIONS

# 1. Khởi tạo môi trường với cấu hình mong muốn
env = make_env(
    seed=42,
    config={
        "lanes_count": 4,              # Cao tốc 4 làn đường
        "vehicles_count": "dense",     # Mật độ xe dày đặc (38 xe)
        "initial_speed": 25.0,         # Xuất phát ở 90 km/h (25 m/s)
        "max_speed": 35.0,             # Tốc độ tối đa 126 km/h
        "traffic_speed_min": 14.0,     # Dải tốc độ xe giao thông 50 - 100 km/h
        "traffic_speed_max": 28.0,
        "obstacle_count": 2,           # 2 xe hỏng chắn đường ngẫu nhiên
        "random_obstacles": True,
    }
)

obs, info = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
    sensors = info["sensors"]
    
    # 2. Áp dụng Action Masking: Không cho phép đổi làn khi đang dở thao tác
    valid_actions = info["available_actions"]
    if sensors["changing"]:
        valid_actions = [a for a in valid_actions if a not in (0, 2)]  # Bỏ lệnh LANE_LEFT, LANE_RIGHT
    
    # 3. Trích xuất đặc trưng và đưa ra quyết định
    # Kiểm tra cự ly an toàn phía trước
    ego_front = sensors["ego_front_gap"]
    if ego_front < 35.0:  # Có nguy cơ va chạm phía trước
        if sensors["left_clear"] and 0 in valid_actions:
            action = 0  # Đổi sang làn trái
        elif sensors["right_clear"] and 2 in valid_actions:
            action = 2  # Đổi sang làn phải
        else:
            action = 4  # Giảm tốc khẩn cấp
    else:
        # Đường thông thoáng: Duy trì tốc độ hoặc tăng tốc
        action = 3 if sensors["speed_kmh"] < 95.0 and 3 in valid_actions else 1

    # 4. Bước mô phỏng môi trường
    next_obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    # 5. Đưa transition vào thuật toán RL để học
    # replay_buffer.push(obs, action, reward, next_obs, terminated, truncated)
    obs = next_obs

print(f"Hoàn thành episode! Kết quả: {info['end_reason']}, Quãng đường: {info['distance_m']:.1f} m, Điểm thưởng: {total_reward:.2f}")
env.close()
```

---

## 8. Hướng Dẫn Tùy Biến Kịch Bản: Số Làn, Mật Độ Xe & Vận Tốc Chạy

Dự án cung cấp hệ thống tham số dòng lệnh phong phú cho phép bạn thiết lập mọi tình huống giao thông thực tế:

### 8.1. Các Tùy Chọn Cấu Hình

| Tham số CLI | Giá trị hỗ trợ | Mặc định | Ý nghĩa & Mô tả |
| :--- | :--- | :--- | :--- |
| `--lanes` | `2`, `3`, `4`, `5`, `6` | `3` | Số làn đường của cao tốc. Giao diện và cảm biến radar tự động co giãn tương thích. |
| `--traffic` | `sparse` (hoặc `thua`), `normal` (`vua`), `dense` (`day`), `none`, hoặc số `0..60` | `normal` | Mật độ xe xung quanh. `sparse`: 8 xe, `normal`: 20 xe, `dense`: 38 xe, `none`: 0 xe. |
| `--speed` | Số thực ($>0$, đơn vị $\text{km/h}$) | `72.0` | Vận tốc xuất phát của xe AI (hoặc xe người lái). |
| `--max-speed` | Số thực ($\ge \text{speed}$, đơn vị $\text{km/h}$) | `108.0` | Vận tốc tối đa cho phép của xe trên tuyến đường. |
| `--traffic-speed` | `slow`, `normal`, `fast` | `normal` | Dải tốc độ của dòng xe giao thông (`slow`: 36-65 km/h, `normal`: 50-95 km/h, `fast`: 75-115 km/h). |
| `--obstacles` | `0`, `1`, `2`, `3` | `2` | Số lượng xe hỏng chắn làn cố định dọc tuyến đường. |
| `--route-length`| Số thực (đơn vị $\text{mét}$) | `1000.0` | Chiều dài chặng đường cần hoàn thành để cán đích. |

---

### 8.2. Các Kịch Bản Thử Nghiệm Thực Tế

#### Kịch bản 1: Cao tốc hẹp 2 làn, xe thưa, chạy tốc độ cao
```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.run --lanes 2 --traffic sparse --speed 90 --max-speed 120 --algo dqn --mode agent
```

#### Kịch bản 2: Cao tốc mở rộng 5 làn, giao thông dày đặc phức tạp
```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.run --lanes 5 --traffic dense --speed 80 --max-speed 110 --traffic-speed normal --algo dqn --mode agent
```

#### Kịch bản 3: Thử nghiệm đường vắng không có xe, chỉ né xe hỏng
```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.run --lanes 4 --traffic none --speed 100 --algo dqn --mode agent
```

#### Kịch bản 4: Đường đông xe chạy chậm, đòi hỏi lách làn thông minh
```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.run --lanes 3 --traffic dense --speed 60 --max-speed 90 --traffic-speed slow --algo dqn --mode agent
```
