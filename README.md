# Hệ Thống Xe Tự Lái Né Tránh Chướng Ngại Vật Trên Cao Tốc (Học Tăng Cường)

Dự án nghiên cứu và triển khai các thuật toán Học tăng cường (Reinforcement Learning) cho bài toán điều khiển xe tự hành né vật cản trên đường cao tốc.

## Danh mục Dự án

```text
Hoc_Tang_Cuong/
├── highway_project/           # Dự án chính: Deep RL trên nền Gymnasium & Highway-Env
│   ├── algorithms/            # Cài đặt thuật toán DQN 2013, Double Dueling DQN và Q-learning
│   ├── environment/           # Môi trường cao tốc 2..6 làn, mật độ xe và tốc độ tùy chỉnh
│   ├── gui/                   # Giao diện HUD 60 FPS (Pygame)
│   ├── training/              # Huấn luyện mô hình và xuất đồ thị
│   ├── tests/                 # 19 unit tests tự động
│   ├── models/                # Checkpoints mạng nơ-ron (.pth) và bảng Q (.npz)
│   ├── results/               # Biểu đồ hiệu năng và báo cáo JSON
│   ├── DQN.md                 # Lý thuyết DQN, cấu thức giải tích, đặc tả dữ liệu trả về & tùy biến kịch bản
│   └── README.md              # Hướng dẫn chi tiết mã nguồn highway_project
├── docs/                      # Kho lưu trữ tài liệu phân tầng theo chuyên đề
│   ├── README.md              # Mục lục & hướng dẫn tra cứu tổng quan
│   ├── moi_truong/            # Tài liệu giải thích mã nguồn môi trường & mô hình MDP
│   ├── thuat_toan/            # Phân tích thuật toán Deep Q-Network & MARL
│   ├── bao_cao_thuyet_trinh/  # Slide thuyết trình nhóm buổi 2
│   ├── hinh_anh/              # Ảnh chụp giao diện Dashboard & Menu Cài Đặt
│   └── cong_cu/               # Script tự động xuất tài liệu sang file Word (.docx)
├── marl4av/                   # Mô phỏng đa tác nhân NetLogo (Báo cáo TH2)
├── CAI_DAT_MOI_TRUONG.cmd     # Script cài đặt môi trường tự động (Tách riêng)
└── CHAY_MOI_TRUONG.cmd        # File chạy duy nhất: Mở ngay giao diện mô phỏng Pygame tích hợp đầy đủ tính năng
```

1. **`highway_project/` (Khuyên dùng - Nền tảng Gymnasium & Highway-Env):**
   - Triển khai thuật toán **Deep Q-Network (DQN 2013 với PyTorch)**, **Double Dueling DQN (Thuật toán 2 - Ngô Tuấn Anh)** và **Tabular Q-learning**.
   - Môi trường cao tốc linh hoạt tùy chỉnh từ **2 đến 6 làn đường** (`--lanes` hoặc phím **`L`**), mật độ giao thông 3 cấp trực quan: **thưa thớt / bình thường / dày đặc** (`--traffic` hoặc phím **`T`**), tùy biến vận tốc xuất phát và dòng xe (`--speed` hoặc phím **`S`**).
   - Giao diện trực quan Pygame mượt mà 60 FPS, hệ thống HUD hiển thị vận tốc, quãng đường, vật cản, mật độ và Bảng Menu Cài Đặt nổi trung tâm không reset xe.
   - Hỗ trợ chạy một chạm qua file duy nhất: [`CHAY_MOI_TRUONG.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_MOI_TRUONG.cmd).
   - Xem chi tiết tại [**`highway_project/README.md`**](file:///d:/Hoc_Tang_Cuong/highway_project/README.md) và tài liệu thuật toán [**`highway_project/DQN.md`**](file:///d:/Hoc_Tang_Cuong/highway_project/DQN.md).

2. **`docs/` (Kho lưu trữ tài liệu & báo cáo - Đã phân loại theo chuyên đề):**
   - Xem mục lục tổng thể tại [**`docs/README.md`**](file:///d:/Hoc_Tang_Cuong/docs/README.md).
   - **Môi trường & MDP**: [**`docs/moi_truong/GIAI_THICH_CODE_MOI_TRUONG.md`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/GIAI_THICH_CODE_MOI_TRUONG.md) (bản Word: [`.docx`](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/GIAI_THICH_CODE_MOI_TRUONG.docx)), [**`BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md).
   - **Thuật toán RL (DQN, Double Dueling DQN & MARL)**: [**`docs/thuat_toan/dqn/GIAI_THICH_CODE_DQN.md`**](file:///d:/PTAPP/Reinforcement-Learning/docs/thuat_toan/dqn/GIAI_THICH_CODE_DQN.md), [**`docs/thuat_toan/dqn/THUAT_TOAN_DQN.md`**](file:///d:/PTAPP/Reinforcement-Learning/docs/thuat_toan/dqn/THUAT_TOAN_DQN.md), [**`docs/thuat_toan/ddqn/THUAT_TOAN_DDQN.md`**](file:///d:/PTAPP/Reinforcement-Learning/docs/thuat_toan/ddqn/THUAT_TOAN_DDQN.md), [**`docs/thuat_toan/ddqn/GIAI_THICH_CODE_DDQN.md`**](file:///d:/PTAPP/Reinforcement-Learning/docs/thuat_toan/ddqn/GIAI_THICH_CODE_DDQN.md), [**`docs/thuat_toan/dqn/BAO_CAO_TH2.md`**](file:///d:/PTAPP/Reinforcement-Learning/docs/thuat_toan/dqn/BAO_CAO_TH2.md).
   - **Thuyết trình**: [**`docs/bao_cao_thuyet_trinh/SLIDE_THUYET_TRINH_BUOI_2.md`**](file:///d:/Hoc_Tang_Cuong/docs/bao_cao_thuyet_trinh/SLIDE_THUYET_TRINH_BUOI_2.md).

3. **`marl4av/` (Nền tảng NetLogo 6.4.0 Engine & Python Extension):**
   - Mô phỏng đa tác nhân NetLogo theo báo cáo thực hành TH2.

## Bắt đầu Nhanh với DQN trong `highway_project`

```powershell
# Chạy xe tự lái với mô hình Deep Q-Network đã huấn luyện (File chạy duy nhất)
.\CHAY_MOI_TRUONG.cmd

# Chạy kiểm thử tự động
.\.venv-highway\Scripts\python.exe -m unittest discover -s highway_project -t .

# Xuất đồ thị phân tích kết quả huấn luyện và so sánh
.\.venv-highway\Scripts\python.exe -m highway_project.plot_results
```
