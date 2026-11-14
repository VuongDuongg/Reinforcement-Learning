# Hệ Thống Xe Tự Lái Né Tránh Chướng Ngại Vật Trên Cao Tốc (Học Tăng Cường)

Dự án nghiên cứu và triển khai các thuật toán Học tăng cường (Reinforcement Learning) cho bài toán điều khiển xe tự hành né vật cản trên đường cao tốc.

## Danh mục Dự án

```text
Hoc_Tang_Cuong/
├── highway_project/           # Dự án chính: Deep RL trên nền Gymnasium & Highway-Env
│   ├── algorithms/            # Cài đặt thuật toán Double DQN và Q-learning
│   ├── environment/           # Môi trường cao tốc 2..6 làn, mật độ xe và tốc độ tùy chỉnh
│   ├── gui/                   # Giao diện HUD 60 FPS (Pygame)
│   ├── training/              # Huấn luyện mô hình và xuất đồ thị
│   ├── tests/                 # 19 unit tests tự động
│   ├── models/                # Checkpoints mạng nơ-ron (.pth) và bảng Q (.npz)
│   ├── results/               # Biểu đồ hiệu năng và báo cáo JSON
│   ├── DQN.md                 # Lý thuyết DQN, cấu thức giải tích, đặc tả dữ liệu trả về & tùy biến kịch bản
│   └── README.md              # Hướng dẫn chi tiết mã nguồn highway_project
├── docs/                      # Tài liệu lý thuyết, báo cáo thực hành & hướng dẫn
│   ├── THUAT_TOAN_DQN.md      # Phân tích thuật toán DQN, đặc tả dữ liệu trả về & hướng dẫn kịch bản
│   ├── BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md
│   ├── BAO_CAO_TH2.md
│   └── KIEN_THUC_CHUNG_MOI_TRUONG_THAY_THANH.md
├── marl4av/                   # Mô phỏng đa tác nhân NetLogo (Báo cáo TH2)
├── CAI_DAT_MOI_TRUONG.cmd     # Script cài đặt môi trường tự động
├── CHAY_MOI_TRUONG.cmd        # Mở ngay giao diện mô phỏng Pygame (DQN AI tự lái)
├── CHAY_XE_DQN.cmd            # Chạy xe tự lái mô hình Deep Q-Network
├── CHAY_TUY_CHON.cmd          # Menu tương tác chọn số làn, mật độ xe, tốc độ trong console
└── CHAY_XE_TU_HOC.cmd         # Chạy xe tự lái mô hình Tabular Q-learning
```

1. **`highway_project/` (Khuyên dùng - Nền tảng Gymnasium & Highway-Env):**
   - Triển khai thuật toán **Deep Q-Network (Double DQN với PyTorch)** và **Tabular Q-learning**.
   - Môi trường cao tốc linh hoạt tùy chỉnh từ **2 đến 6 làn đường** (`--lanes` hoặc phím **`L`**), mật độ giao thông 3 cấp trực quan: **thưa thớt / bình thường / dày đặc** (`--traffic` hoặc phím **`T`**), tùy biến vận tốc xuất phát và dòng xe (`--speed` hoặc phím **`S`**).
   - Giao diện trực quan Pygame mượt mà 60 FPS, hệ thống HUD hiển thị vận tốc, quãng đường, vật cản, mật độ và nút bấm tương tác trực tiếp trên màn hình.
   - Hỗ trợ chạy một chạm qua [`CHAY_MOI_TRUONG.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_MOI_TRUONG.cmd) và [`CHAY_XE_DQN.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_XE_DQN.cmd).
   - Xem chi tiết tại [**`highway_project/README.md`**](file:///d:/Hoc_Tang_Cuong/highway_project/README.md) và tài liệu thuật toán [**`highway_project/DQN.md`**](file:///d:/Hoc_Tang_Cuong/highway_project/DQN.md).

2. **`docs/` (Kho lưu trữ tài liệu & báo cáo):**
   - [**`THUAT_TOAN_DQN.md`**](file:///d:/Hoc_Tang_Cuong/docs/THUAT_TOAN_DQN.md): Giải thích tại sao DQN phù hợp với bài toán, phân tích phạm vi áp dụng (khi nào nên dùng, khi nào không nên dùng) và chi tiết kiến trúc mạng nơ-ron.
   - Các báo cáo quy tắc môi trường MDP và lý thuyết củng cố.

3. **`marl4av/` (Nền tảng NetLogo 6.4.0 Engine & Python Extension):**
   - Mô phỏng đa tác nhân NetLogo theo báo cáo thực hành TH2.

## Bắt đầu Nhanh với DQN trong `highway_project`

```powershell
# Chạy xe tự lái với mô hình Deep Q-Network đã huấn luyện
.\CHAY_XE_DQN.cmd

# Chạy kiểm thử tự động
.\.venv-highway\Scripts\python.exe -m unittest discover -s highway_project -t .

# Xuất đồ thị phân tích kết quả huấn luyện và so sánh
.\.venv-highway\Scripts\python.exe -m highway_project.plot_results
```
