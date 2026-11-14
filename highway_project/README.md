# Highway RL Project: Xe Tự Lái Né Vật Cản Trên Cao Tốc

Dự án áp dụng Học tăng cường (Reinforcement Learning) cho bài toán xe tự hành (Autonomous Vehicle) điều hướng, chuyển làn và né chướng ngại vật cố định trên đường cao tốc 3 làn dài 1000m với dòng xe dày đặc và tính cách lái xe đa dạng, khó đoán.

---

## 1. Cấu trúc Thư mục

```text
highway_project/
├── DQN.md                          # Tài liệu lý thuyết chuyên sâu về thuật toán DQN & phạm vi áp dụng
├── README.md                       # Hướng dẫn tổng quan dự án highway_project
├── run.py                          # Script khởi chạy mô phỏng đồ họa chính
├── algorithms/                     # [CÁC THUẬT TOÁN RL]
│   ├── dqn_agent.py                # Cài đặt Double DQN, QNetwork, Replay Buffer (PyTorch)
│   ├── q_agent.py                  # Cài đặt thuật toán Tabular Q-learning
│   └── agent.py                    # CustomAgent làm cầu nối chung
├── environment/                    # [MÔI TRƯỜNG MÔ PHỎNG]
│   └── environment.py              # Môi trường LaneObstacleEnv (Gymnasium + HighwayEnv)
├── gui/                            # [GIAO DIỆN & HIỂN THỊ ĐỒ HỌA]
│   ├── view.py                     # Giao diện hiển thị Pygame HUD & Cao tốc 3 làn
│   └── motion.py                   # Bộ đệm nội suy chuyển động 60 FPS
├── training/                       # [HUẤN LUYỆN & PHÂN TÍCH]
│   ├── train_dqn.py                # Script huấn luyện Deep Q-Network (Curriculum Learning)
│   ├── train.py                    # Script huấn luyện Tabular Q-learning
│   └── plot_results.py             # Script xuất đồ thị theo dõi & so sánh hiệu năng
├── tests/                          # [BỘ KIỂM THỬ TỰ ĐỘNG]
│   ├── test_dqn.py                 # Kiểm thử mạng nơ-ron và thuật toán DQN
│   ├── test_environment.py         # Kiểm thử hợp đồng môi trường Gymnasium
│   └── test_learning.py            # Kiểm thử cập nhật Bellman và nội suy
├── models/                         # [TRỌNG SỐ MÔ HÌNH ĐÃ HUẤN LUYỆN]
│   ├── dqn_model.pth               # Checkpoint trọng số mô hình Deep Q-Network
│   └── q_learning.npz              # Bảng giá trị Q-learning
└── results/                        # [KẾT QUẢ ĐÁNH GIÁ & ĐỒ THỊ]
    ├── dqn_report.json             # Báo cáo chi tiết quá trình học & đánh giá DQN
    ├── q_learning_report.json      # Báo cáo đánh giá Q-learning
    └── dqn_performance.png         # Đồ thị so sánh trực quan hóa
```

> 📖 **Xem tài liệu chi tiết về thuật toán:** [**`DQN.md`**](file:///d:/Hoc_Tang_Cuong/highway_project/DQN.md) giải thích cặn kẽ tại sao DQN phù hợp với bài toán xe tự hành, phạm vi áp dụng, so sánh với Q-learning dạng bảng và kiến trúc mạng nơ-ron.

---

## 2. Cài đặt Môi trường

Yêu cầu Python 3.10+ (Đã thử nghiệm và tối ưu trên Python 3.13 Windows).

```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv-highway
.\.venv-highway\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install -r highway_project\requirements.txt
```

---

## 3. Hướng dẫn Chạy Mô phỏng

### 3.1. Chạy nhanh một chạm bằng file `.cmd`
- **Chạy môi trường ngay lập tức (DQN AI tự lái)**: Click đúp [`CHAY_MOI_TRUONG.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_MOI_TRUONG.cmd) hoặc [`CHAY_XE_DQN.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_XE_DQN.cmd). Cửa sổ Pygame sẽ bật lên ngay lập tức trong vòng chưa đến 1 giây!
- **Chạy menu kịch bản tương tác trong Console**: Click đúp [`CHAY_TUY_CHON.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_TUY_CHON.cmd) để trả lời trắc nghiệm chọn làn, mật độ xe, vận tốc trước khi chạy.
- **Chạy xe tự lái Q-learning dạng bảng**: Click đúp [`CHAY_XE_TU_HOC.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_XE_TU_HOC.cmd).

### 3.2. Điều khiển trực tiếp trên giao diện Pygame (Nút bấm & Phím tắt)
Bạn có thể tùy biến số làn, mật độ xe và kịch bản **trực tiếp khi đang xem mô phỏng** bằng chuột hoặc phím tắt:
- **Phím `L`** hoặc nút **`[L · Làn]`**: Đổi số làn đường cao tốc theo vòng lặp: `2 -> 3 -> 4 -> 5 -> 6 -> 2` làn.
- **Phím `T`** hoặc nút **`[T · Xe]`**: Đổi mật độ xe theo vòng lặp: `Thưa (8 xe) -> Vừa (20 xe) -> Dày (38 xe) -> Không (0 xe)`.
- **Phím `S`** hoặc nút **`[S · Tốc độ]`**: Đổi dải tốc độ xuất phát: `54 -> 72 -> 90 -> 108 km/h`.
- **Phím `1`**: Kịch bản 1 — Cao tốc hẹp 2 làn, xe thưa 8 xe, 90 km/h.
- **Phím `2`**: Kịch bản 2 — Cao tốc 3 làn chuẩn, xe vừa 20 xe, 72 km/h.
- **Phím `3`**: Kịch bản 3 — Cao tốc 5 làn mở rộng, xe dày đặc 38 xe, dòng xe nhanh.
- **Phím `4`**: Kịch bản 4 — Cao tốc vắng 4 làn, không có xe giao thông (0 xe), 90 km/h.
- **Phím `Tab`** hoặc nút **`[Tab · Đổi chế độ]`**: Chuyển đổi tức thì giữa **AI tự lái (DQN)** và **Người tự lái bằng phím**.
- **Phím `Space`** hoặc nút **`[Space · Dừng]`**: Tạm dừng / Tiếp tục mô phỏng.
- **Phím `R`** hoặc nút **`[R · Chạy lại]`**: Khởi động lại tập mô phỏng hiện tại.
- **Phím `N`** hoặc nút **`[N · Hạt giống]`**: Sinh tình huống giao thông ngẫu nhiên mới.
- **Phím mũi tên `← / → / ↑ / ↓`**: Chuyển làn và tăng/giảm tốc độ khi tự lái.

### 3.3. Chạy từ dòng lệnh với các tham số tùy chọn

Người dùng có thể linh hoạt điều chỉnh cấu hình kịch bản: **số làn đường (2..6)**, **mật độ xe (thưa thớt / bình thường / dày đặc)**, **vận tốc xe** và **dải tốc độ giao thông**:

```powershell
# Chạy mô hình DQN mặc định (3 làn đường, mật độ bình thường 20 xe, vận tốc 72 km/h)
python -m highway_project.run --algo dqn --mode agent --auto-reset

# 1. Tùy chỉnh số làn đường (từ 2 đến 6 làn)
# Cao tốc 2 làn đường:
python -m highway_project.run --lanes 2 --traffic sparse --speed 80 --algo dqn --mode agent

# Cao tốc mở rộng 5 làn đường:
python -m highway_project.run --lanes 5 --traffic dense --speed 90 --max-speed 120 --algo dqn --mode agent

# 2. Tùy chỉnh mật độ giao thông trực quan theo 3 cấp độ:
# - Mật độ thưa thớt (sparse / thua: 8 xe)
python -m highway_project.run --traffic sparse --algo dqn --mode agent

# - Mật độ bình thường (normal / vua: 20 xe - mặc định)
python -m highway_project.run --traffic normal --algo dqn --mode agent

# - Mật độ dày đặc (dense / day: 38 xe)
python -m highway_project.run --traffic dense --algo dqn --mode agent

# - Đường vắng chỉ có xe hỏng (none / 0 xe)
python -m highway_project.run --traffic none --algo dqn --mode agent

# 3. Tùy chỉnh vận tốc xuất phát và dòng xe:
# Xe xuất phát ở 90 km/h, max 130 km/h, dòng xe chạy nhanh (fast: 75-115 km/h)
python -m highway_project.run --speed 90 --max-speed 130 --traffic-speed fast --algo dqn --mode agent

# Chế độ tự lái bằng bàn phím (Người điều khiển)
python -m highway_project.run --mode keyboard
```

**Phím tắt trong cửa sổ Pygame:**
- `Space`: Tạm dừng / Tiếp tục.
- `Tab`: Chuyển đổi giữa chế độ AI (`DQN`) và Người lái (`Bàn phím`).
- `R`: Chạy lại kịch bản hiện tại.
- `N`: Đổi kịch bản / hạt giống ngẫu nhiên mới.
- `Esc`: Thoát cửa sổ.

---

## 4. Huấn luyện Mô hình

### 4.1. Huấn luyện Deep Q-Network (DQN)
```powershell
# Huấn luyện mô hình mới 160 episodes (Curriculum 0 -> 12 xe)
python -m highway_project.train_dqn --episodes 160 --eval-episodes 10
```

### 4.2. Huấn luyện Q-learning Dạng Bảng
```powershell
python -m highway_project.train --episodes 180
```

### 4.3. Xuất đồ thị theo dõi và so sánh hiệu năng
```powershell
python -m highway_project.plot_results
```
Đồ thị so sánh sẽ được lưu tại `highway_project/results/dqn_performance.png`.

---

## 5. Kết quả Đánh giá Benchmark

Thử nghiệm độc lập trên tập seed kiểm thử ($1.000.000 - 1.000.009$):

| Kịch bản giao thông | Q-Learning Dạng Bảng | **Deep Q-Network (DQN)** | Vận tốc TB (DQN) |
| :--- | :---: | :---: | :---: |
| **Đường vắng (Traffic = 0)** | 0% | **100% (10/10)** | 106.6 km/h |
| **Giao thông vừa (Traffic = 6)** | 0% | **60% (6/10)** | 105.8 km/h |
| **Giao thông đông (Traffic = 12)** | 10% | **30% (3/10)** | 100.6 km/h |

---

## 6. Chạy Kiểm thử Tự động (Unit Tests)

```powershell
python -m unittest discover -s highway_project -t .
```
*(Tất cả 19/19 tests bao gồm Gymnasium environment, dynamic lanes, traffic density presets, speed scaling, Bellman target, Action Masking, Save/Load và Replay Buffer đều PASS).*

