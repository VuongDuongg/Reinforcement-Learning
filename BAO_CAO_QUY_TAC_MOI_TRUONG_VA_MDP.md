# BÁO CÁO ĐẶC TẢ QUY TẮC MÔI TRƯỜNG MÔ PHỎNG VÀ MÔ HÌNH HÓA QUY TRÌNH QUYẾT ĐỊNH MARKOV (MDP)

**ĐỀ TÀI:** NGHIÊN CỨU VÀ XÂY DỰNG HỆ THỐNG XE TỰ LÁI NÉ TRÁNH CHƯỚNG NGẠI VẬT THEO LÀN ĐƯỜNG CAO TỐC SỬ DỤNG HỌC TĂNG CƯỜNG  
**HỌC PHẦN:** HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)  
**ĐƠN VỊ:** KHOA CÔNG NGHỆ THÔNG TIN – ĐẠI HỌC THỦY LỢI  
**NHÓM THỰC HIỆN:** NHÓM 1 - LỚP 65KTPM  

---

## MỤC LỤC TỔNG QUAN
1. [QUY TẮC MÔI TRƯỜNG (ENVIRONMENT RULES & CONSTRAINTS)](#i-quy-tắc-môi-trường-environment-rules--constraints)
   - 1.1. Cấu trúc Không gian Đường cao tốc 1 chiều (One-Way Directional Topology)
   - 1.2. Luật Phân cấp Tốc độ theo từng làn ($v_{\text{limit, lane}}$)
   - 1.3. Quy tắc An toàn & Điều kiện Dừng (Safety Rules & Terminal Conditions)
2. [CƠ CHẾ VẬN HÀNH MÔI TRƯỜNG (ENVIRONMENT DYNAMICS & MECHANISMS)](#ii-cơ-chế-vận-hành-môi-trường-environment-dynamics--mechanisms)
   - 2.1. Cơ chế Dòng xe Nguồn - Đích (Source-Sink Traffic Spawner) – *Không có cơ chế quay đầu*
   - 2.2. Cơ chế Sự cố & Xe hỏng tấp vào làn khẩn cấp (Breakdown & Rescue Mechanism)
   - 2.3. Cơ chế Cảm biến hình nón (Cone Sensor) & Phân tích cải tiến tầm nhìn
   - 2.4. Động lực học Chuyển trạng thái ($\mathcal{P}$)
3. [XÂY DỰNG MÔ HÌNH QUY TRÌNH QUYẾT ĐỊNH MARKOV (MDP)](#iii-xây-dựng-mô-hình-quy-trình-quyết-định-markov-mdp)
   - 3.1. Không gian Trạng thái ($\mathcal{S}$)
   - 3.2. Không gian Hành động ($\mathcal{A}$)
   - 3.3. Xác suất Chuyển trạng thái ($\mathcal{P}$)
   - 3.4. Thiết kế Hàm Phần thưởng Đa mục tiêu ($\mathcal{R}$)
   - 3.5. Hệ số Chiết khấu ($\gamma$)
4. [TỔNG KẾT & TÍNH KHẢ THI CHO BÀI TẬP NHÓM](#iv-tổng-kết--tính-khả-thi-cho-bài-tập-nhóm)

---

# I. QUY TẮC MÔI TRƯỜNG (ENVIRONMENT RULES & CONSTRAINTS)

Để giải quyết bài toán xe tự hành né tránh chướng ngại vật một cách thực tế, hiệu quả và phù hợp với giới hạn thời gian, tài nguyên tính toán của môn học, nhóm thiết kế một bộ quy tắc môi trường tinh gọn, chuẩn mực như sau:

### 1.1. Cấu trúc Không gian Đường cao tốc 1 chiều (One-Way Directional Topology)
*   **Chỉ có một chiều đi duy nhất (One-way forward only)**:
    *   Toàn bộ luồng phương tiện chỉ di chuyển theo **một chiều cố định từ trái sang phải** (tọa độ trục $X$ tăng dần, hướng la bàn $\text{heading} = 90^\circ$).
    *   **Loại bỏ hoàn toàn chiều ngược lại**: Không mô phỏng dải phân cách giữa và chiều đối diện. Thiết kế này giúp tiết kiệm **50% tài nguyên tính toán của CPU**, loại bỏ toàn bộ các điều kiện rẽ nhánh kiểm tra tọa độ âm/dương phức tạp, tập trung 100% mật độ tương tác giao thông vào hành trình của Agent xe tự hành (Ego Car).
*   **Mặt cắt ngang gồm 4 làn đường rõ rệt (tính từ trái qua phải theo hướng di chuyển)**:
    1.  **Làn 0 (Làn vượt / Làn trong cùng bên trái - $y = 3$)**: Làn dành cho các phương tiện di chuyển với tốc độ tối đa cao nhất.
    2.  **Làn 1 (Làn giữa - $y = 1$)**: Làn lưu thông tiêu chuẩn với tốc độ trung bình.
    3.  **Làn 2 (Làn ngoài cùng bên phải - $y = -1$)**: Làn tốc độ thấp dành cho xe tải hoặc xe chạy chậm.
    4.  **Làn 3 (Làn dừng khẩn cấp / Shoulder - $y = -3$)**: Làn nằm sát mép đường bên phải, được phân tách bằng **vạch sơn trắng liền nét**. Làn này **nghiêm cấm xe lưu thông bình thường**, chỉ dùng làm nơi cứu hộ tập kết các xe gặp sự cố chết máy ($v = 0$).

```
[Mép trên đường y = 4] ================= VẠCH VÀNG LIỀN =================
  LÀN 0 (Làn vượt - Tốc độ cao 120 km/h)          y = 3   ---> Xe chạy
------------------------ - - - VẠCH TRẮNG ĐỨT ĐOẠN - - - ----------------
  LÀN 1 (Làn giữa - Tốc độ chuẩn 100 km/h)        y = 1   ---> Xe chạy
------------------------ - - - VẠCH TRẮNG ĐỨT ĐOẠN - - - ----------------
  LÀN 2 (Làn phải - Tốc độ chậm 80 km/h)          y = -1  ---> Xe chạy
======================== ===== VẠCH TRẮNG LIỀN NÉT ===== ================
  LÀN 3 (Làn dừng khẩn cấp - Chỉ đỗ xe hỏng)      y = -3  [ Xe hỏng v = 0 ]
[Mép dưới đường y = -4] ================= VẠCH VÀNG LIỀN =================
```

---

### 1.2. Luật Phân cấp Tốc độ theo từng làn ($v_{\text{limit, lane}}$)
Môi trường áp dụng giới hạn tốc độ tối đa khác nhau trên 3 làn chạy chính nhằm mô phỏng sát thực tế cao tốc Việt Nam (ví dụ cao tốc Hà Nội - Hải Phòng):
*   **Làn 0**: $v_{\text{limit}, 0} = 33.3\text{ m/s}$ ($\sim 120\text{ km/h}$).
*   **Làn 1**: $v_{\text{limit}, 1} = 27.8\text{ m/s}$ ($\sim 100\text{ km/h}$).
*   **Làn 2**: $v_{\text{limit}, 2} = 22.2\text{ m/s}$ ($\sim 80\text{ km/h}$).
*   **Làn khẩn cấp (Làn 3)**: $v_{\text{limit}, 3} = 0\text{ m/s}$ (Cấm lưu thông).

> **Ý nghĩa học tăng cường (RL Strategy)**: Quy tắc này tạo ra **sự đánh đổi đa mục tiêu (Trade-off: Speed vs Safety)**:
> - Muốn đạt điểm thưởng tốc độ cao nhất, xe AI phải chuyển sang Làn 0.
> - Tuy nhiên, Làn 0 có mật độ xe chạy rất nhanh, nếu gặp xe hỏng đột ngột thì quãng đường phanh cần thiết rất lớn, rủi ro va chạm cực cao.
> - Chạy ở Làn 2 an toàn hơn nhưng tốc độ bị khống chế ở 80 km/h, tích lũy phần thưởng thấp hơn. Thuật toán RL buộc phải học chiến lược chuyển làn thông minh theo thời gian thực.

---

### 1.3. Quy tắc An toàn & Điều kiện Dừng (Terminal Conditions)
*   **Cự ly an toàn & Va chạm**:
    *   Khoảng cách vật lý giữa hai tâm xe: $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$.
    *   Nếu cự ly $d < d_{\text{collision}} = 1.0\text{ patch}$ ($\sim 1.75\text{ m}$ - tương đương chiều rộng xe), môi trường xác định **xảy ra va chạm (Crash)**.
*   **Điều kiện kết thúc ván chơi (Terminal States - Episode Done)**:
    1.  **Va chạm (Collision Failure)**: Khi xe AI đâm vào xe khác hoặc đâm vào xe hỏng $\to \text{Done} = \text{True}$, môi trường áp dụng hình phạt va chạm cực nặng ($R = -100$) và lập tức kết thúc episode.
    2.  **Hoàn thành hành trình (Goal Reached / Success)**: Khi xe AI di chuyển an toàn từ đầu tuyến ($x = x_{\min}$) đến cuối tuyến ($x = x_{\max}$) mà không gây tai nạn $\to \text{Done} = \text{True}$, môi trường cấp điểm thưởng về đích lớn ($R = +50$) và kết thúc episode.
    3.  **Hết thời gian tối đa (Truncation / Max Steps)**: Nếu đạt giới hạn số bước mô phỏng mà xe chưa về đích, episode kết thúc để thu thập dữ liệu huấn luyện.

---

# II. CƠ CHẾ VẬN HÀNH MÔI TRƯỜNG (ENVIRONMENT DYNAMICS & MECHANISMS)

### 2.1. Cơ chế Dòng xe Nguồn - Đích (Source-Sink Traffic Spawner)
> **ĐIỂM NHẤN QUAN TRỌNG: TUYỆT ĐỐI KHÔNG CÓ CƠ CHẾ QUAY ĐẦU (NO U-TURN / NO LOOPING)**

*   **Vì sao loại bỏ cơ chế xe quay đầu trên đường cao tốc?**
    *   Về mặt luật pháp và thực tế giao thông: Xe trên cao tốc **tuyệt đối không được phép quay đầu chạy ngược chiều**, kể cả chạy trên làn khẩn cấp.
    *   Về mặt mô phỏng: Nếu cho xe đi hết đường rồi quay ngược lại trên làn khẩn cấp, các xe này sẽ đâm trực diện vào các xe gặp sự cố đang đỗ trên làn khẩn cấp, gây ra xung đột logic và tắc nghẽn giao thông vô nghĩa.
*   **Cơ chế Nguồn - Đích chuẩn (Source-Sink Mechanism)**:
    *   **Nguồn sinh xe (Source at $x = x_{\min}$)**: Các phương tiện giao thông nền (Background Vehicles) được sinh ra tại vạch xuất phát bên trái với xác suất $P_{\text{spawn}}$. Xe chỉ được sinh nếu vị trí xuất phát còn khoảng trống an toàn và tổng số xe trên đường chưa vượt quá ngưỡng giới hạn trần $N_{\max}$ (đảm bảo không xảy ra bùng nổ dân số xe gây kẹt đường).
    *   **Điểm tiêu hủy (Sink at $x = x_{\max}$)**: Khi bất kỳ phương tiện giao thông nền nào hoàn thành quãng đường và chạm mép cuối đường cao tốc ($x \ge x_{\max}$), phương tiện đó được xóa khỏi mô phỏng một cách tự nhiên.
    *   **Đối với Ego Car (Xe AI)**: Khi chạm $x \ge x_{\max}$, xe hoàn thành 1 Episode học tập, nhận điểm thưởng Goal Reward và được môi trường Reset về vạch xuất phát cho Episode tiếp theo.

---

### 2.2. Cơ chế Sự cố & Xe hỏng tấp vào làn khẩn cấp (Breakdown & Rescue Mechanism)
Đây là cơ chế tạo ra **chướng ngại vật động và tĩnh bất ngờ**, buộc thuật toán học tăng cường phải học cách phanh gấp hoặc chuyển làn né tránh:

1.  **Giai đoạn 1: Kích hoạt sự cố ngẫu nhiên**:
    *   Mỗi bước thời gian (tick), mỗi xe nền đang chạy có xác suất gặp sự cố: $P_{\text{breakdown}} = 0.00002$.
    *   Khi sự cố xảy ra: Xe chết máy lập tức ($v = 0$), đổi sang màu **ĐỎ CẢNH BÁO** (`damaged-color`).
    *   Lúc này xe hỏng trở thành **chướng ngại vật tĩnh nằm chắn ngay giữa làn chạy chính**. Các xe chạy phía sau (kể cả Ego Car) phải dùng cảm biến phát hiện từ xa để giảm tốc hoặc chuyển làn khẩn cấp.
2.  **Giai đoạn 2: Cứu hộ dạt xe vào làn khẩn cấp**:
    *   Xe hỏng nằm trên làn chính đếm thời gian: $t_{\text{inlane}}$.
    *   Khi $t_{\text{inlane}} \ge T_{\text{inlane\_max}}$ (khoảng 30 giây - 1 phút trong thế giới mô phỏng), hệ thống cứu hộ bắt đầu kéo tịnh tiến xe dạt dần sang bên phải để vào **Làn 3 (Làn dừng khẩn cấp $y = -3$)**.
3.  **Giai đoạn 3: Giải tỏa làn khẩn cấp**:
    *   Khi đã nằm gọn trong Làn 3, xe hỏng đỗ tại đây để không làm gián đoạn luồng giao thông chính trên 3 làn chạy.
    *   Sau thời gian cứu hộ hoàn tất ($t_{\text{rescue}} \ge T_{\text{rescue\_max}}$), xe cứu hộ kéo xe rời khỏi đường cao tốc (`die`), giải phóng không gian làn khẩn cấp.

---

### 2.3. Cơ chế Cảm biến hình nón (Cone Sensor) & Phân tích cải tiến tầm nhìn
Để xe AI có thể đưa ra quyết định ở từng bước thời gian, môi trường trang bị cho xe hệ thống cảm biến Radar/LiDAR quét dạng hình nón phía trước:

#### A. Cấu trúc Cảm biến Nón cơ bản:
*   **Góc mở quan sát (Field of View - FoV)**: $60^\circ$ (quét $\pm 30^\circ$ đối xứng qua trục mũi xe $\text{heading} = 90^\circ$).
*   **Dữ liệu trích xuất từ nón quan sát**:
    *   Khoảng cách đến xe gần nhất phía trước ($d_{\text{ahead}}$).
    *   Vận tốc của xe gần nhất phía trước ($v_{\text{ahead}}$).
    *   **Phân loại chướng ngại vật phía trước**:
        $$\text{type}_{\text{ahead}} = \begin{cases} 0 & \text{Không có vật cản} \\ 1 & \text{Xe bình thường đang chạy } (v > 0) \\ 2 & \text{Xe gặp sự cố chết máy } (v = 0) \end{cases}$$
    *   Khoảng cách và vận tốc xe phía sau trên cùng làn ($d_{\text{behind}}, v_{\text{behind}}$).

#### B. Phân tích điểm yếu & Đề xuất cải tiến tầm nhìn động (Dynamic Look-ahead):
*   **Phân tích hạn chế**: Trong mô hình NetLogo ban đầu, tầm quét radar chỉ cố định ở mức $2.5\text{ patches}$ ($\sim 8.75\text{ m}$). 
    *   Khi xe chạy ở Làn 0 với tốc độ $33\text{ m/s}$ ($120\text{ km/h}$), thời gian từ lúc phát hiện xe hỏng đến khi đâm chỉ là:
        $$t = \frac{8.75\text{ m}}{33\text{ m/s}} \approx 0.26\text{ giây}$$
    *   Với thời gian $0.26$s, quán tính vật lý không cho phép xe phanh hay chuyển làn kịp, dẫn tới **tai nạn không thể tránh khỏi (Unavoidable Crash)** khiến AI không học được chính sách hội tụ.
*   **Giải pháp cải tiến (Tầm nhìn thích ứng theo tốc độ)**:
    Áp dụng công thức tầm quét radar tỉ lệ thuận với vận tốc hiện tại của xe:
    $$d_{\text{sensor}}(v_t) = d_0 + v_t \cdot \tau_{\text{preview}}$$
    *Trong đó: $d_0 = 3.0\text{ patches}$ là tầm nhìn cơ sở; $\tau_{\text{preview}} = 1.5\text{ s}$ là khoảng thời gian dự báo an toàn. Khi xe chạy càng nhanh, tầm radar càng quét xa (lên tới $8 - 10\text{ patches} \approx 30\text{ m}$), giúp AI có đủ thời gian phản xạ đưa ra quyết định chuyển làn.*

---

### 2.4. Động lực học Chuyển trạng thái ($\mathcal{P}$)
Sau khi Agent xe tự hành phát ra một hành động điều khiển $a_t \in \mathcal{A}$, môi trường tính toán cập nhật tọa độ vật lý theo mô hình động học điểm khối (Point-mass Kinematic Model):

1.  **Cập nhật Tọa độ Dọc (Trục $X$)**:
    $$x_{t+1} = x_t + v_t \cdot \Delta t$$
2.  **Cập nhật Vận tốc Dọc ($v$)**:
    $$v_{t+1} = \text{clip}\Big(v_t + a_{\text{long}}(a_t) \cdot \Delta t,\; v_{\min},\; v_{\text{limit, lane}}\Big)$$
    *Trong đó: $a_{\text{long}}(a_t)$ là gia tốc tương ứng với hành động ($+a_{\text{acc}}$ khi tăng tốc, $-a_{\text{dec}}$ khi phanh, $0$ khi giữ tốc).*
3.  **Cập nhật Tọa độ Ngang (Trục $Y$ - Chuyển làn trơn tru)**:
    Khi xe thực hiện hành động chuyển làn sang làn mục tiêu $y_{\text{target}}$, môi trường tịnh tiến tọa độ $Y$ mượt mà qua các bước đệm:
    $$y_{t+1} = y_t + \text{sign}(y_{\text{target}} - y_t) \cdot \min\Big(\Delta y_{\text{step}},\; |y_{\text{target}} - y_t|\Big)$$
    *Mũi xe vẫn duy trì hướng cố định $\text{heading} = 90^\circ$, giúp xe chuyển làn song song mượt mà như thực tế thay vì bị quay ngoắt đầu xe.*

---

# III. XÂY DỰNG MÔ HÌNH QUY TRÌNH QUYẾT ĐỊNH MARKOV (MDP)

Bài toán điều khiển xe tự hành né tránh chướng ngại vật được mô hình hóa toán học chặt chẽ dưới dạng bộ 5 thành phần MDP: $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### 3.1. Không gian Trạng thái ($\mathcal{S}$ - State Space)
Để mạng nơ-ron xấp xỉ giá trị (DQN/DDQN) và mạng chính sách (PPO/SAC) có đủ thông tin ra quyết định, vector trạng thái đầu vào được chuẩn hóa liên tục gồm 6 thành phần đặc trưng:

$$\mathbf{s}_t = \big[ \tilde{v}_t,\; \tilde{d}_{\text{ahead}},\; \tilde{v}_{\text{ahead}},\; \text{type}_{\text{ahead}},\; \tilde{d}_{\text{behind}},\; \tilde{v}_{\text{limit}} \big] \in \mathbb{R}^6$$

| Ký hiệu | Thành phần thông tin | Khoảng giá trị chuẩn hóa | Ý nghĩa nghiệp vụ |
| :---: | :--- | :---: | :--- |
| $\tilde{v}_t$ | Vận tốc hiện tại của Ego Car | $[0, 1]$ | Chuẩn hóa theo $v_{\max} = 33.3\text{ m/s}$ |
| $\tilde{d}_{\text{ahead}}$ | Khoảng cách tới vật cản phía trước | $[0, 1]$ | Chuẩn hóa theo tầm cảm biến $d_{\text{sensor}}$ |
| $\tilde{v}_{\text{ahead}}$ | Vận tốc của vật cản phía trước | $[0, 1]$ | Xác định tốc độ tương đối giữa hai xe |
| $\text{type}_{\text{ahead}}$ | Phân loại chướng ngại vật trước mặt | $\{0, 0.5, 1.0\}$ | $0$: Đường trống; $0.5$: Xe đang chạy; $1.0$: Xe hỏng ($v=0$) |
| $\tilde{d}_{\text{behind}}$ | Khoảng cách tới xe phía sau cùng làn | $[0, 1]$ | Đảm bảo an toàn không bị đâm đuôi khi giảm tốc |
| $\tilde{v}_{\text{limit}}$ | Giới hạn tốc độ của làn hiện tại | $\{0.67, 0.83, 1.0\}$ | 80 km/h (Làn 2), 100 km/h (Làn 1), 120 km/h (Làn 0) |

---

### 3.2. Không gian Hành động ($\mathcal{A}$ - Action Space)
Môi trường cung cấp không gian hành động rời rạc gồm 4 lựa chọn trực quan, dễ huấn luyện cho các thuật toán:

$$\mathcal{A} = \{0, 1, 2, 3\}$$

*   **$a = 0$ (Accelerate - Tăng tốc)**: Xe tăng thêm một lượng gia tốc $+a_{\text{acc}}$, tô màu **Xanh lá**.
*   **$a = 1$ (Cruise / Stay same - Giữ nguyên tốc độ)**: Xe duy trì vận tốc ổn định, tô màu **Vàng**.
*   **$a = 2$ (Decelerate / Brake - Giảm tốc độ)**: Xe phanh giảm một lượng $-a_{\text{dec}}$, tô màu **Tím**.
*   **$a = 3$ (Change Lane - Chuyển làn né vật cản)**: Xe kích hoạt thuật toán chuyển sang làn bên cạnh an toàn hơn, tô màu **Xanh dương**.

---

### 3.3. Xác suất Chuyển trạng thái ($\mathcal{P}$)
Do môi trường chứa các yếu tố bất định thực tế (xe khác đổi làn ngẫu nhiên, xác suất sự cố xe hỏng ngẫu nhiên $P_{\text{breakdown}}$), xác suất chuyển trạng thái mang tính ngẫu nhiên (Stochastic MDP):

$$\mathcal{P}(\mathbf{s}_{t+1} \mid \mathbf{s}_t, a_t) = \mathbb{P}\big(\mathbf{s}_{t+1} \mid \mathbf{s}_t, a_t\big)$$

Trong đó trạng thái tiếp theo $\mathbf{s}_{t+1}$ là sự kết hợp giữa:
1.  Động lực học tất định của xe AI (Deterministic Motion Update).
2.  Hành vi ngẫu nhiên của các xe giao thông xung quanh (Stochastic Traffic Behavior).

---

### 3.4. Thiết kế Hàm Phần thưởng Đa mục tiêu ($\mathcal{R}$ - Reward Function)
Hàm phần thưởng đóng vai trò kim chỉ nam hướng dẫn các thuật toán RL hội tụ về hành vi mong muốn: **Lái xe an toàn, duy trì tốc độ cao theo đúng quy định làn, chuyển làn dứt khoát không lạng lách**.

Tổng phần thưởng tại bước $t$ được định nghĩa:

$$\mathcal{R}_t = R_{\text{safety}} + R_{\text{speed}} + R_{\text{rule}} + R_{\text{comfort}} + R_{\text{terminal}}$$

| Thành phần | Công thức / Giá trị điểm | Mục đích và Hành vi khuyến khích |
| :--- | :---: | :--- |
| **Phạt va chạm ($R_{\text{crash}}$)** | **$-100$** | Phạt cực nặng khi $d < d_{\text{collision}}$ (kết thúc episode ngay) |
| **Thưởng tốc độ ($R_{\text{speed}}$)** | $+3 \cdot \Big(\frac{v_t}{v_{\text{limit, lane}}}\Big)$ | Khuyến khích xe di chuyển thông suốt, áp sát tốc độ tối đa cho phép |
| **Phạt quá tốc độ làn ($R_{\text{overspeed}}$)**| $-10$ | Phạt nếu $v_t > v_{\text{limit, lane}}$ (bắt buộc tuân thủ luật phân cấp tốc độ) |
| **Phạt lấn làn khẩn cấp ($R_{\text{shoulder}}$)** | $-5$ | Phạt nếu xe đi vào Làn 3 (ngăn AI lạm dụng làn khẩn cấp để trốn giao thông) |
| **Phạt phanh xe ($R_{\text{brake}}$)** | $-1$ | Hạn chế tình trạng phanh giật cục khi không có nguy hiểm |
| **Phạt chuyển làn ($R_{\text{lane\_change}}$)** | $-2$ | Phạt chi phí chuyển làn, ngăn xe lạng lách chuyển làn liên tục vô tội vạ |
| **Thưởng hoàn thành tuyến ($R_{\text{goal}}$)** | **$+50$** | Thưởng lớn khi xe chạy an toàn đến cuối đường ($x \ge x_{\max}$) |

---

### 3.5. Hệ số Chiết khấu ($\gamma$ - Discount Factor)
*   Nhóm lựa chọn: **$\gamma = 0.98$**.
*   **Cơ sở khoa học**: Lái xe tự hành trên cao tốc là một bài toán có tầm nhìn dài hạn (Long-horizon planning). Hệ số $\gamma = 0.98$ đảm bảo xe AI không chỉ quan tâm đến phần thưởng trước mắt ở bước hiện tại ($t$) mà phải tính toán đến nguy cơ tiềm ẩn phía trước (ví dụ: nhìn thấy từ xa có xe hỏng ở Làn 0 thì phải có kế hoạch giảm tốc và chuyển làn sớm từ trước).

---

# IV. TỔNG KẾT & TÍNH KHẢ THI CHO BÀI TẬP NHÓM

1.  **Tính tinh gọn & Hiệu năng**:
    *   Bằng việc thiết lập đường **1 chiều duy nhất**, bài toán giảm được $50\%$ khối lượng tính toán trên NetLogo.
    *   Loại bỏ hoàn toàn cơ chế quay đầu xe ngược chiều giúp mô hình không bị xung đột vật lý với các xe hỏng trên làn khẩn cấp.
2.  **Tính thực tế & Khoa học**:
    *   Cơ chế **phân cấp tốc độ làn (120 - 100 - 80 km/h)** kết hợp **xe hỏng đỗ tại làn khẩn cấp** tái hiện đầy đủ các tình huống giao thông thực tế tại Việt Nam.
    *   Cải tiến **cảm biến hình nón động** giải quyết dứt điểm nghịch lý thời gian phản xạ ở vận tốc cao, đảm bảo xe AI hoàn toàn có khả năng học được chính sách an toàn.
3.  **Tính sẵn sàng cho các thành viên nhóm**:
    *   Vector trạng thái $\mathbf{s}_t \in \mathbb{R}^6$ và không gian hành động rời rạc $|\mathcal{A}| = 4$ được chuẩn hóa đồng bộ cho cả 4 thành viên cài đặt các thuật toán: **DQN (Dương Quốc Vượng)**, **Double Dueling DQN (Ngô Tuấn Anh)**, **PPO (Nguyễn Hồng Thái)**, và **SAC / Baseline (Lê Tuấn Khanh)**.
    *   Môi trường đã hoàn thiện, sẵn sàng cho việc huấn luyện và vẽ đồ thị so sánh đánh giá hiệu năng trong báo cáo cuối kỳ.
