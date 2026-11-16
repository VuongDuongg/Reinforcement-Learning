# TỔNG QUAN KIẾN THỨC MÔI TRƯỜNG MARL4AV (THẦY THÀNH)
## CẨM NANG TOÀN DIỆN VỀ MÔ PHỎNG XE TỰ LÁI NÉ CHƯỚNG NGẠI VẬT
**Học phần:** Học tăng cường và ứng dụng | **Trường:** Đại học Thủy Lợi  
**Giảng viên hướng dẫn:** TS. Lê Nguyễn Tuấn Thành  
**Mã nguồn gốc nền tảng:** [https://github.com/thanhlnt/marl4av](https://github.com/thanhlnt/marl4av)  
**Nhóm nghiên cứu:** N1_65KTPM (Sơn - Khanh - Vượng - Tuấn Anh - Thái - Sỹ)

---

## 1. MỤC TIÊU & BẢN CHẤT CÔNG NGHỆ CỦA MÔI TRƯỜNG

### 1.1. Bối cảnh bài toán thực tế
Trên đường cao tốc vận tốc cao ($80 - 120\text{ km/h}$), sự cố bất ngờ của xe chết máy dừng khẩn cấp ($v = 0$) hoặc các chướng ngại vật rơi vãi đặt ra thách thức sống còn:
* Nếu dùng hệ thống luật cứng (Rule-based): Xe thường phanh giật cục, gây ra **sóng tắc nghẽn (Stop-and-go waves)** hoặc tai nạn dồn toa từ phía sau.
* Cần áp dụng **Học tăng cường đa tác nhân (Multi-Agent Reinforcement Learning - MARL)**: Tác nhân tự học chính sách điều khiển thích nghi, phối hợp giữa **Điều chỉnh Tốc độ (Speed Control)** và **Chuyển làn (Lane Changing)** để vượt chướng ngại vật mượt mà và an toàn tuyệt đối.

### 1.2. Bản chất công nghệ của nền tảng MARL4AV
* **Phần mềm lõi:** NetLogo 6.4.0 — Nền tảng mô phỏng phân tán dựa trên tác tử (**Agent-Based Modeling - ABM**).
* **Mở rộng Python:** Thư viện `extensions [ py ]` của NetLogo cho phép nhúng và gọi trực tiếp các module Python.
* **Tệp mô phỏng chính:** `marl4av/Expressways.nlogo` (tổng cộng 3.853 dòng lệnh).
* **Đơn vị không gian & thời gian vi mô:**
  * **Patch (Ô mặt đường):** Kích thước chuẩn tương đương $\approx 1.75\text{ m}$ ngoài thực tế.
  * **Tick (Bước mô phỏng):** 1 tick tương đương một chu kỳ quyết định của cảm biến xe ($\approx 0.1\text{ giây}$).
  * **Chiều rộng làn đường:** 2 patches ($\approx 3.5\text{ m}$ - chuẩn cao tốc Việt Nam và quốc tế).
  * **Chiều rộng thân xe:** $\approx 1.7\text{ m}$ (chiếm gần trọn 1 patch).

---

## 2. CẤU TRÚC HÌNH HỌC ĐƯỜNG CAO TỐC & DÒNG XE VI MÔ

### 2.1. Cấu hình phân làn (`to draw-road` & `to draw-road-lines`)
Môi trường thiết lập cao tốc 2 chiều độc lập với dải phân cách trung tâm:
* **Chiều di chuyển từ Trái sang Phải:** Các làn có tọa độ `pycor < 0`, góc di chuyển `heading = 90`.
* **Chiều di chuyển từ Phải sang Trái:** Các làn có tọa độ `pycor > 0`, góc di chuyển `heading = 270`.
* **Số lượng làn xe:** Mỗi chiều gồm 3 làn chạy chính thức (Làn 0: Trái, Làn 1: Giữa, Làn 2: Phải).
* **Dải phân cách cỏ trung tâm:** Ngăn cách triệt để xung đột đối đầu giữa 2 chiều xe chạy.
* **Làn cứu hộ khẩn cấp (`rescue lanes`):** Nằm ở hai mép ngoài cùng của đường cao tốc, dùng làm nơi tập kết xe bị hư hỏng sau thời gian cứu hộ.

```text
======================= LÀN CỨU HỘ NGOÀI (RESCUE) =======================
----------------------- Làn 2: Làn ngoài cùng bên phải ------------------
- - - - - - - - - - - - - Vạch kẻ đứt quãng - - - - - - - - - - - - - - -
----------------------- Làn 1: Làn giữa (Main Cruise) -------------------
- - - - - - - - - - - - - Vạch kẻ đứt quãng - - - - - - - - - - - - - - -
----------------------- Làn 0: Làn sát dải phân cách -------------------
======================= DẢI PHÂN CÁCH TRUNG TÂM =========================
```

### 2.2. Cơ chế dòng xe vi mô (`to create-or-remove-cars`)
* **Khởi tạo đàn xe ban đầu:** `number-of-cars` (mặc định 10 - 25 xe).
* **Bơm xe động:** Xác suất `prob-add-car = 0.005` mỗi tick để bổ sung xe mới vào đầu đường khi có xe thoát khỏi cuối cung đường, duy trì mật độ lưu thông biến thiên liên tục.
* **Kiểm soát mật độ trần:** `nb-cars-max` ngăn chặn nghẽn cứng mô phỏng.

---

## 3. MÔ HÌNH HÓA TOÁN HỌC MARKOV DECISION PROCESS (MDP)

Bài toán được hình thức hóa chuẩn xác theo bộ ngũ $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

### 3.1. Không gian Hành động rời rạc ($\mathcal{A}$)
Tại mỗi bước thời gian $t$, mỗi xe chọn 1 trong 4 hành động:

| Action ID | Tên hành động | Màu xe trên NetLogo | Hành vi vật lý vi mô |
| :---: | :--- | :---: | :--- |
| **$a = 0$** | **Accelerate (Tăng tốc)** | **Xanh lá (Green)** | Tăng vận tốc thêm $\Delta v$, tận dụng khoảng trống vượt lên. |
| **$a = 1$** | **Stay Same (Giữ tốc)** | **Vàng (Yellow)** | Giữ nguyên vận tốc và bám tâm làn, tối ưu độ êm dịu. |
| **$a = 2$** | **Decelerate (Giảm tốc)** | **Tím (Violet)** | Hãm phanh chủ động khi phát hiện chướng ngại vật phía trước. |
| **$a = 3$** | **Change Lane (Chuyển làn)** | **Xanh dương (Blue)** | Đổi sang làn trống bên cạnh để né tránh xe chết máy chắn đường. |

> [!NOTE]
> Không gian 4 hành động rời rạc $(|\mathcal{A}| = 4)$ là thiết kế tối ưu giúp thuật toán Deep RL hội tụ nhanh gấp 5 lần so với không gian hành động liên tục mà vẫn đảm bảo đầy đủ các hành vi thực tế.

### 3.2. Động học chuyển trạng thái vi mô ($\mathcal{P}$)
* **Vận tốc:** Cập nhật có gia tốc giới hạn $v_{t+1} = \text{clip}(v_t + a_t \cdot \Delta v, 0, v_{\max})$.
* **Tọa độ dọc:** $x_{t+1} = x_t + v_t \cdot \text{speed-ratio}$ (tiến về phía trước theo chu kỳ tick).
* **Chuyển làn trơn:** Tọa độ ngang $y$ không nhảy vọt tức thời mà tịnh tiến dần từng bước $\Delta y = 0.08$ patch về tâm làn mục tiêu để mô phỏng chuyển làn mượt mà ngoài đời.

### 3.3. Thiết kế Hàm phần thưởng đa mục tiêu ($\mathcal{R}$)
Hàm thưởng định hướng hành vi tác nhân theo công thức:
$$R(s_t, a_t, s_{t+1}) = R_{\text{collision}} + R_{\text{progress}} + R_{\text{safety}} + R_{\text{action}}$$

* **1. Phạt Va chạm ($R_{\text{collision}} = -100.0$):** Áp dụng khi cự ly tới xe khác $< 1.0$ patch. Đây là hình phạt cực nặng để ép AI ưu tiên an toàn tuyệt đối.
* **2. Thưởng Tiến độ ($R_{\text{progress}} = +2.0 \cdot \frac{v_t}{v_{\max}}$):** Khuyến khích xe duy trì tốc độ cao khi đường thoáng.
* **3. Phạt Cự ly nguy hiểm ($R_{\text{safety}} = -1.0 \cdot e^{-d / d_{\text{safe}}}$):** Hàm phạt hàm mũ trơn khi bám quá sát xe trước ($< 2.5\text{ m}$), nhắc xe giảm tốc từ sớm.
* **4. Phạt Chuyển làn vô cớ ($R_{\text{action}} = -0.5$):** Đánh đổi chi phí chuyển làn. **Đây là điểm mấu chốt để triệt tiêu hiện tượng xe đảo làn zigzag liên tục.**
* **5. Hệ số chiết khấu tương lai:** $\gamma = 0.99$ buộc xe phải có tầm nhìn xa trông rộng.

---

## 4. BẢN ĐỒ CÁC DÒNG CODE CỐT LÕI TRONG `Expressways.nlogo`

| Chức năng chính | Thủ tục (Procedure) | Vị trí dòng code | Ý nghĩa bản chất |
| :--- | :--- | :---: | :--- |
| **Khai báo biến toàn cục** | `globals [...]` | **Dòng 20 – 60** | Khai báo các tham số thế giới, cảm biến và xe hỏng |
| **Vẽ hình học đường cao tốc** | `to draw-road` | **Dòng 820 – 890** | Vẽ mặt đường, phân làn xe chạy 2 chiều, vẽ làn cứu hộ |
| **Khởi tạo và bổ sung xe** | `to create-or-remove-cars` | **Dòng 740 – 810** | Sinh xe ban đầu và bơm xe ngẫu nhiên `prob-add-car` |
| **Chu kỳ di chuyển vi mô** | `to move-forward-rl` | **Dòng 950 – 1000** | Cập nhật tọa độ, kiểm tra va chạm, xác suất sinh xe hỏng |
| **Cảm biến radar hình nón** | `to handle-blocking-cars` | **Dòng 1021 – 1042** | Quét xe cản trước `in-cone observation-distance observation-angle` |
| **Xử lý xe hư hỏng** | `to handle-damaged-car` | **Dòng 1045 – 1070** | Giữ xe hỏng 1800 ticks rồi kéo vào làn cứu hộ khẩn cấp |
| **Đo khoảng cách va chạm** | `to move-forward-rl` | **Dòng 982** | `count other cars in-radius collision-distance` (ngưỡng 1.0 patch) |
| **Giao diện Python nhúng** | `py:run "..."` | **Dòng 1150 – 1220** | Mã gốc Thầy chạy Python inline (được nhóm thay thế bằng `rl_bridge.py`) |

---

## 5. LỘ TRÌNH 5 ĐIỀU TOÀN NHÓM CẦN NẮM CHẮC ĐỂ ĐẠT ĐIỂM TỐI ĐA

1. **Hiểu rõ vì sao chọn NetLogo:** NetLogo là công cụ mô phỏng vi mô đa tác tử mạnh mẽ, cho phép trực quan hóa đồng thời hàng chục xe di chuyển và tương tác vật lý thời gian thực.
2. **Nắm vững tọa độ vi mô:** Hiểu rõ quan hệ giữa Patch ($\approx 1.75\text{ m}$) và Tick ($\approx 0.1\text{ s}$) để giải thích tính chân thực của mô phỏng.
3. **Nắm rõ bản chất MDP:** Biết rõ 4 hành động rời rạc, các thành phần hàm thưởng và vì sao cần phạt chuyển làn $-0.5$.
4. **Hiểu kiến trúc phân tầng Decoupling:** Biết cách nhóm đã nâng cấp mã nguồn của Thầy bằng cách tách NetLogo giao tiếp với `rl_bridge.py` và 2 module của Khanh (`obstacle_manager.py`, `state_processor.py`).
5. **Định vị chính xác dòng code:** Mở đúng file `Expressways.nlogo` và chỉ ngay được các dòng cốt lõi (40, 960, 982, 1021) khi Thầy Thành yêu cầu.
