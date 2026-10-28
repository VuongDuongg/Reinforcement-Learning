# Xe tự lái né chướng ngại vật theo làn

Nhóm N1_65KTPM · Học tăng cường và ứng dụng

**Giảng viên:** TS. Lê Nguyễn Tuấn Thành · Cập nhật 28/10/2026

1. Các thành phần MDP của bài toán.
2. Môi trường thử nghiệm đã chọn và đã triển khai.
3. Thuật toán dự kiến tích hợp và mã giả.
4. Tiến độ từng thành viên.

---

## 1. MDP: xe quan sát, hành động và nhận thưởng

`M = (S, A, P, R, γ)` · Mục tiêu: tới đích an toàn, duy trì lưu thông, hạn chế đổi làn không cần thiết.

| Thành phần | Áp dụng vào bài toán |
|---|---|
| S — trạng thái | Vị trí, tốc độ, hướng, mục tiêu điều khiển của xe; giao thông, vật cản và thời gian |
| A — hành động | Trái, giữ, phải, tăng tốc, giảm tốc |
| P — chuyển trạng thái | Động học HighwayEnv và phản ứng của xe giao thông |
| R — phần thưởng | Sáu thành phần, cùng tên trong công thức và bảng chi tiết |
| γ — chiết khấu | Q-learning thử dùng 0,98; thuật toán mới ghi rõ cấu hình được chọn |

Trạng thái toàn môi trường là `s_t`; quan sát agent nhận là `o_t`, chỉ bao quát một phần giao thông.

---

## 1.1. S và quan sát: agent biết những gì?

**Quan sát chuẩn:** `o_t ∈ R^(10×5)`; mỗi dòng gồm `presence, x, y, vx, vy`.

- Dòng đầu là xe điều khiển; các dòng sau là đối tượng lân cận, gồm xe hỏng; dòng thiếu đệm 0.
- Dữ liệu được chuẩn hóa. Vị trí/vận tốc của đối tượng xung quanh là tương đối với xe điều khiển; dòng ego giữ đại lượng tuyệt đối đã chuẩn hóa.
- Mười dòng không có nghĩa là toàn môi trường chỉ có mười xe.

**Q-learning thử:** dùng thêm `info.sensors` trong phạm vi 180 m: làn, tốc độ, khoảng trống trước/sau và tốc độ khép khoảng cách của ba làn.

Trạng thái Q: `(làn, nguy cơ trước, tình trạng trái, tình trạng phải, nhóm tốc độ, đang đổi làn)`.

`3 × 4 × 3 × 3 × 3 × 2 = 648 trạng thái` · Đây là xấp xỉ quan sát một phần, không khẳng định đầy đủ tính Markov.

---

## 1.2. A: năm hành động rời rạc

| ID | Hành động | Tác động |
|---|---|---|
| 0 | LANE_LEFT | Đặt làn mục tiêu sang trái |
| 1 | IDLE | Giữ tốc độ mục tiêu và tiếp tục bám làn |
| 2 | LANE_RIGHT | Đặt làn mục tiêu sang phải |
| 3 | FASTER | Tăng một mức tốc độ mục tiêu |
| 4 | SLOWER | Giảm một mức tốc độ mục tiêu |

Các mức tốc độ: `0, 5, 10, 15, 20, 25, 30 m/s`.

**IDLE không có nghĩa là đứng yên.** SLOWER không phải phanh dừng tức thời. Không dùng bảng bốn hành động cũ của NetLogo.

---

## 1.3. P và γ: cập nhật trạng thái

**Một quyết định 0,1 giây:** chọn hành động → bám làn/tốc độ → cập nhật động học → xét va chạm → trả quan sát và thưởng mới.

- Quyết định **10 Hz**; vật lý **30 Hz**: ba bước vật lý cho một quyết định.
- Khởi tạo xe giao thông theo seed; cùng seed tái lập được khởi tạo.
- Xe giao thông dùng IDM; xe hỏng đứng yên, không có bộ điều khiển tự tăng tốc.
- `P(s'|s,a)` do mô phỏng xác định; nhóm không tự lập bảng xác suất chuyển.

`G_t = r_t + γ r_(t+1) + γ² r_(t+2) + …`

Q-learning thử dùng `γ = 0,98`. Chiết khấu giúp cân nhắc thưởng tương lai; bản thân γ không bảo đảm xe né an toàn.

---

## 1.4. R: một công thức thống nhất với mã

`r_t = r_collision + r_offroad + r_speed + r_lane_change + r_goal + r_danger`

| Thành phần | Giá trị khi kích hoạt | Điều kiện |
|---|---|---|
| r_collision | −20 | Va chạm |
| r_offroad | −10 | Ra khỏi đường, chưa va chạm |
| r_speed | +0,03 × v / 30 | Còn trên đường, chưa va chạm |
| r_lane_change | −0,02 | Yêu cầu hành động trái hoặc phải |
| r_goal | +10 | Tới đích an toàn |
| r_danger | −0,15 | Khoảng trống trước <10 m hoặc đang tiến gần với TTC <2 s |

**Ưu tiên:** va chạm chỉ nhận −20; ra khỏi đường chỉ nhận −10. Khi đó các thành phần khác bằng 0; thành phần không kích hoạt luôn bằng 0.

TTC = khoảng trống / tốc độ khép khoảng cách; chỉ xét TTC khi tốc độ khép >0,1 m/s.

---

## 1.5. Kết thúc lượt chạy và ví dụ thưởng

| Tình huống | Cờ kết thúc | Ý nghĩa |
|---|---|---|
| Đi đủ 650 m, an toàn, còn trên đường | terminated; is_success | Hoàn thành |
| Va chạm hoặc ra khỏi đường | terminated; không thành công | Thất bại |
| Đạt 60 giây | truncated | Hết giờ; kiểm tra riêng is_success |
| Vượt xe hỏng nhưng chưa tới đích | Tiếp tục chạy | Chưa hoàn thành |

Đường thoáng, giữ 20 m/s: `r_t = 0,03 × 20/30 = +0,02`.

Đổi làn ở cùng tốc độ, không có nguy cơ: `r_t = +0,02 − 0,02 = 0`.

Xe dừng và đồng hồ vẫn tăng là lựa chọn của chính sách; môi trường không kết thúc ngay sau khi né vật cản.

---

## 2. Môi trường đã chọn: HighwayEnv + Gymnasium

| Công cụ | Vai trò |
|---|---|
| HighwayEnv | Đường, động học, xe giao thông và va chạm |
| Gymnasium | API reset / step; tách môi trường khỏi thuật toán |
| Python + NumPy | Kịch bản và Q-learning thử trên CPU |
| Pygame | Hiển thị và bàn phím; không quyết định thay agent |

**Lý do chọn:** cùng hệ Python để tích hợp thuật toán; có sẵn mô phỏng cao tốc; tạo được lớp riêng cho xe hỏng.

`marl4av / NetLogo` dùng tham khảo ý tưởng. Môi trường mới không cần NetLogo để chạy.

Nguồn: https://highway-env.farama.org/ · API được đối chiếu với bộ thư viện đã cài.

---

## 2.1. Kịch bản đã triển khai

**Đường một chiều, ba làn chính; một xe do agent điều khiển.** Các xe giao thông dùng IDM, không cùng học.

| Tham số mặc định | Giá trị |
|---|---|
| Xe giao thông / xe hỏng | 12 / 1 |
| Xe điều khiển xuất phát | Làn giữa, 20 m/s (72 km/h) |
| Xe hỏng đầu tiên | Cách điểm xuất phát 160 m, vận tốc 0 |
| Đích / thời hạn | 650 m / 60 giây |
| Tùy chỉnh dòng lệnh | 0–40 xe giao thông, 0–3 vật cản, thay seed |

Chưa có làn khẩn cấp, giới hạn tốc độ riêng từng làn, cứu hộ xe hỏng hoặc học đa tác nhân trong môi trường mới.

---

## 2.2. Kiến trúc để thêm thuật toán

`reset(seed) → (observation, info)`

`agent.act(observation, info) → action ∈ {0,1,2,3,4}`

`step(action) → (next_observation, reward, terminated, truncated, info)`

- **environment.py:** khởi tạo, vật cản, cảm biến, thưởng và kết thúc episode.
- **q_agent.py / train.py:** chính sách thử, vòng học và đánh giá; có thể thay bằng thuật toán khác.
- **run.py / view.py / motion.py:** demo, vẽ và nội suy 60 FPS; trễ hình tối đa một bước.

Khi tích hợp DQN/PPO: chọn rõ đầu vào ma trận hay cảm biến, cập nhật kích thước mạng và dùng cùng đầu vào khi học/đánh giá.

---

## 2.3. Đã làm được gì, còn giới hạn gì?

**Đã có:** mô phỏng, bàn phím, xe hỏng đứng yên, va chạm, đích/hết giờ, reset theo seed và chỗ nối thuật toán.

**Đã kiểm tra:** hợp đồng Gymnasium; seed; quan sát/va chạm vật cản; tới đích; hết giờ; Bellman; lưu/tải Q; nội suy không sửa vật lý.

**Q-learning thử:** 180 episode, 23.994 bước cập nhật. Kiểm tra ban đầu với 10 seed mới và 0 xe giao thông: **0/10 tới đích, 0/10 va chạm**; xe có thể dừng tới hết giờ.

Môi trường có thể dùng để phát triển thuật toán; model thử chưa đạt mục tiêu tới đích. Chưa có đối sánh DQN/DDQN/PPO trên môi trường mới.

---

## 3. Thuật toán dự kiến tích hợp

| Thuật toán | Vai trò | Trạng thái trên môi trường mới |
|---|---|---|
| Rule-based | Mốc so sánh theo luật | Cần chuyển và kiểm tra |
| Q-learning | Thử vòng tương tác/học | Có bản thử, chưa hội tụ |
| DQN | Mạng ước lượng Q | Có mã nền trong marl4av, chưa tích hợp xác minh |
| Double Dueling DQN | Tách V/A, giảm thiên lệch Q | Có mã nền, cần chuyển đầu vào và terminal |
| PPO-Clip | Chính sách rời rạc, học từ rollout | Có mã nền, cần kiểm tra rollout/GAE và tích hợp |

Ưu tiên: **baseline → DQN → Double Dueling DQN → PPO**.

**Phân công thuật toán:** Vượng — DQN; Tuấn Anh — Double Dueling DQN; Thái — PPO; Sỹ — baseline/đánh giá. **Lê Tuấn Khanh chỉ phụ trách môi trường.**

SAC rời rạc là hướng mở rộng nếu còn thời gian; không bắt buộc trong bản thử này.

---

## 3.1. Mã giả: baseline và Q-learning

**Baseline dự kiến** — quy tắc đối chiếu, không phải học:

```text
1. Quan sát xe/vật cản phía trước và khoảng trống hai bên.
2. Nếu đang đổi làn: tiếp tục hoàn thành lần đổi làn đó.
3. Nếu phía trước bị cản: đổi sang làn đủ trống; nếu không có thì giảm tốc.
4. Nếu đường thoáng: giữ hoặc tăng tốc tới mức mong muốn.
```

**Q-learning** — trạng thái cảm biến đã chia nhóm:

```text
1. Tạo bảng điểm Q cho từng tình huống và hành động, ban đầu bằng 0.
2. Bắt đầu lượt chạy; quan sát tình huống hiện tại.
3. Thỉnh thoảng thử ngẫu nhiên; còn lại chọn hành động có điểm cao nhất.
4. Cho xe thực hiện, nhận thưởng/phạt và quan sát tình huống mới.
5. Sửa điểm Q bằng thưởng vừa nhận và điểm dự đoán của bước tiếp theo.
6. Lặp đến khi kết thúc lượt; dần giảm số lần thử ngẫu nhiên.
```

---

## 3.2. Mã giả: Deep Q-Network — DQN

**Đề xuất:** làm phẳng quan sát `(10,5)` thành 50 giá trị; mạng xuất năm Q-value. Đầu vào khác phải được ghi rõ.

```text
1. Tạo mạng chấm điểm năm hành động, một bản mạng đích và bộ nhớ.
2. Quan sát đường; chọn hành động điểm cao hoặc thử ngẫu nhiên.
3. Thực hiện và lưu: quan sát cũ, hành động, thưởng, quan sát mới, cờ kết thúc.
4. Lấy ngẫu nhiên một nhóm kinh nghiệm từ bộ nhớ để học lại.
5. Điểm cần học = thưởng vừa nhận + điểm tương lai từ mạng đích.
6. Sửa mạng chính để điểm dự đoán gần với điểm cần học.
7. Định kỳ chép mạng chính sang mạng đích; lặp qua nhiều lượt chạy.
```

Nếu kết thúc do va chạm/đích, không cộng điểm tương lai. Hết giờ được ghi riêng; mạng đích là bản sao cập nhật chậm giúp việc học ổn định hơn.

---

## 3.3. Mã giả: Double Dueling DQN

**Dueling:** mạng tách giá trị trạng thái V và lợi thế A.

`Q(s,a) = V(s) + A(s,a) − mean_b A(s,b)`

```text
1. Thu thập và lưu kinh nghiệm giống DQN.
2. Mạng có hai nhánh: tình huống tốt đến đâu và hành động nào tốt hơn.
3. Với tình huống kế tiếp, mạng chính chọn hành động có điểm cao nhất.
4. Mạng đích chấm điểm đúng hành động vừa được chọn.
5. Cộng thưởng và điểm tương lai để tạo điểm cần học; sửa mạng chính.
6. Cập nhật mạng đích định kỳ rồi tiếp tục các lượt chạy.
```

**Double:** online chọn hành động kế tiếp, target đánh giá hành động đó. So sánh với DQN trên cùng đầu vào, kịch bản và ngân sách bước.

---

## 3.4. Mã giả: PPO-Clip

**Đề xuất:** Actor xuất phân phối năm hành động; Critic ước lượng V(o).

```text
1. Tạo Actor để chọn hành động và Critic để đánh giá tình huống.
2. Cho xe chạy một đoạn; ghi các quan sát, hành động và thưởng/phạt.
3. Dùng Critic để ước lượng hành động nào tốt hơn dự kiến.
4. Tăng khả năng chọn hành động tốt, giảm khả năng chọn hành động kém.
5. Giới hạn mức thay đổi chính sách mỗi lần cập nhật bằng phép clip.
6. Sửa Critic theo kết quả thực nhận của đoạn chạy.
7. Thu đoạn chạy mới bằng chính sách vừa cập nhật, rồi lặp lại.
```

Khi tính lợi thế (GAE), tách riêng từng lượt chạy; phân biệt kết thúc thực sự và hết giờ. Khi đánh giá, giữ nguyên chính sách và không học thêm.

---

## 4. Cập nhật tiến độ từng thành viên

| Thành viên | Phần việc | Đối chiếu hiện tại / việc tiếp theo |
|---|---|---|
| Nguyễn Đình Tuấn Sơn | Lõi môi trường, tích hợp | Nhóm có HighwayEnv và API; rà soát cấu hình, nối thuật toán |
| Lê Tuấn Khanh | Chỉ phụ trách môi trường | Đường, vật cản, cảm biến, thưởng và điều kiện kết thúc; không phụ trách thuật toán |
| Dương Quốc Vượng | DQN | Có dqn_agent.py; chuyển đầu vào/năm hành động, chạy môi trường mới |
| Ngô Tuấn Anh | Double Dueling DQN | Có ddqn_agent.py; kiểm tra target, terminal và tích hợp |
| Nguyễn Hồng Thái | PPO-Clip | Có ppo_agent.py; kiểm tra rollout/GAE và tích hợp |
| Nguyễn Khắc Sỹ | Baseline, đánh giá, báo cáo | Có baseline_agent.py; chuyển baseline, lập phép đo đối sánh |

Theo mã cục bộ và phân công cũ; chưa quy mã mới cho cá nhân hoặc xác nhận tỷ lệ hoàn thành. Nhóm xác nhận tiến độ cá nhân trước khi nộp.

---

## 4.1. Kế hoạch tiếp theo và tiêu chí đạt

1. Chốt cấu hình môi trường, hàm thưởng và đầu vào chung.
2. Chạy baseline; lần lượt tích hợp DQN, Double Dueling DQN và PPO.
3. Đánh giá trên seed tách khỏi huấn luyện, cùng số xe/vật cản và ngân sách bước.
4. Ghi tỷ lệ tới đích, va chạm, hết giờ; tốc độ, yêu cầu đổi làn và tổng thưởng.

**Điều kiện đạt:** vượt vật cản và tới đích an toàn; đứng yên để tránh va chạm chưa đạt.

**Bàn giao:** môi trường chạy và mở rộng được; có Q-learning thử và tài liệu code. Tối ưu huấn luyện và đối sánh thuật toán là phần việc tiếp theo.
