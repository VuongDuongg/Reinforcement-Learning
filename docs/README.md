# Danh Mục Tài Liệu Nghiên Cứu & Báo Cáo Dự Án (Docs Directory)

Kho tài liệu được tổ chức phân tầng rõ ràng theo từng chủ đề chuyên môn, giúp việc tra cứu mã nguồn, lý thuyết toán học và báo cáo trở nên thuận tiện và nhanh chóng.

```text
docs/
├── README.md                      # Mục lục & hướng dẫn tra cứu tổng quan (File này)
├── moi_truong/                    # 1. TÀI LIỆU VỀ MÔI TRƯỜNG & QUY TẮC MDP
│   ├── GIAI_THICH_CODE_MOI_TRUONG.md     # Giải thích chi tiết từng dòng code môi trường cao tốc
│   ├── GIAI_THICH_CODE_MOI_TRUONG.docx   # Bản Word định dạng chuyên nghiệp sẵn sàng in ấn
│   ├── BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md   # Lý thuyết toán học Markov Decision Process (MDP)
│   ├── BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.docx # Bản Word báo cáo quy tắc môi trường & MDP
│   ├── DE_XUAT_MOI_TRUONG.md             # Đề xuất cải tiến môi trường (làn, xe, chướng ngại vật)
│   └── KIEN_THUC_CHUNG_MOI_TRUONG_THAY_THANH.md # Ghi chú & yêu cầu bài toán từ giảng viên
├── thuat_toan/                    # 2. TÀI LIỆU VỀ THUẬT TOÁN HỌC TĂNG CƯỜNG (RL)
│   ├── THUAT_TOAN_DQN.md                 # Lý thuyết Double DQN, kiến trúc mạng & phân tích so sánh
│   └── BAO_CAO_TH2.md                    # Thuật toán đa tác nhân (MARL) & Q-learning (NetLogo)
├── bao_cao_thuyet_trinh/          # 3. SLIDE BÁO CÁO & THUYẾT TRÌNH
│   └── SLIDE_THUYET_TRINH_BUOI_2.md      # Đề cương & nội dung slide thuyết trình nhóm buổi 2
├── hinh_anh/                      # 4. HÌNH ẢNH GIAO DIỆN & MINH HỌA KẾT QUẢ
│   ├── preview_main_view.png             # Ảnh giao diện chính tinh gọn 4 nút điều khiển
│   ├── preview_menu_view.png             # Ảnh Bảng Menu Cài Đặt nổi trung tâm (Popup Settings)
│   ├── preview_gui_dashboard.png         # Ảnh bảng đo cảm biến radar và telemetry
│   └── preview_interactive.png           # Ảnh minh họa xe né vật cản trên cao tốc
├── cong_cu/                       # 5. CÔNG CỤ XUẤT TÀI LIỆU
│   └── export_docx.py                    # Script tự động xuất markdown sang Word docx chuẩn đẹp
└── superpowers/                   # 6. KẾ HOẠCH PHÁT TRIỂN & TIẾN ĐỘ THỰC HIỆN
    └── plans/                            # Kế hoạch chi tiết theo từng nhiệm vụ
```

---

## 1. Tài Liệu Môi Trường & Quy Tắc MDP (`docs/moi_truong/`)

| Tên Tài Liệu | Định Dạng | Mô Tả Tóm Tắt |
| :--- | :---: | :--- |
| [**`GIAI_THICH_CODE_MOI_TRUONG.md`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/GIAI_THICH_CODE_MOI_TRUONG.md) | Markdown | Phân tích chi tiết toàn bộ mã nguồn `LaneObstacleEnv`: không gian trạng thái, không gian hành động, hàm phần thưởng đa mục tiêu, cảm biến radar 180m và cơ chế vật lý. |
| [**`GIAI_THICH_CODE_MOI_TRUONG.docx`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/GIAI_THICH_CODE_MOI_TRUONG.docx) | Word .docx | Bản Word thiết kế theo phong cách Corporate Tech Palette, kẻ bảng và định dạng code block sẵn sàng in ấn/nộp bài. |
| [**`BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md) | Markdown | Đặc tả mô hình Markov Decision Process (MDP): Bộ 5 thành phần $(S, A, P, R, \gamma)$, phương trình Bellman và các ràng buộc biên cao tốc. |
| [**`BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.docx`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.docx) | Word .docx | Bản Word báo cáo quy tắc môi trường và MDP. |
| [**`DE_XUAT_MOI_TRUONG.md`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/DE_XUAT_MOI_TRUONG.md) | Markdown | Bản đề xuất thiết kế môi trường: chuyển đổi sang HighwayEnv + Gymnasium, hỗ trợ 2..6 làn và 3 cấp độ mật độ xe. |
| [**`KIEN_THUC_CHUNG_MOI_TRUONG_THAY_THANH.md`**](file:///d:/Hoc_Tang_Cuong/docs/moi_truong/KIEN_THUC_CHUNG_MOI_TRUONG_THAY_THANH.md) | Markdown | Tổng hợp kiến thức nền tảng, ghi chú và các lưu ý quan trọng từ bài giảng của thầy. |

---

## 2. Tài Liệu Về Thuật Toán Học Tăng Cường (`docs/thuat_toan/`)

| Tên Tài Liệu | Mô Tả Tóm Tắt |
| :--- | :--- |
| [**`THUAT_TOAN_DQN.md`**](file:///d:/Hoc_Tang_Cuong/docs/thuat_toan/THUAT_TOAN_DQN.md) | **Tài liệu chuyên sâu về thuật toán Deep Q-Network (Double DQN):**<br>• Tại sao DQN phù hợp với bài toán xe né vật cản.<br>• Phân tích phạm vi áp dụng (khi nào nên dùng DQN, khi nào không nên dùng).<br>• So sánh chi tiết hiệu năng giữa Tabular Q-learning và DQN.<br>• Kiến trúc mạng nơ-ron (68 đầu vào, 2 tầng ẩn ReLU [128, 128], 5 đầu ra $Q$).<br>• Cơ chế chiếu tương đối 3 kênh thích ứng động cho 2..6 làn đường.<br>• Đặc tả dữ liệu trả về `(obs, reward, terminated, truncated, info, sensors)` chuẩn Gymnasium API.<br>• Hướng dẫn tùy biến kịch bản qua Menu Cài Đặt và tham số dòng lệnh CLI. |
| [**`BAO_CAO_TH2.md`**](file:///d:/Hoc_Tang_Cuong/docs/thuat_toan/BAO_CAO_TH2.md) | **Báo cáo thực hành 2 (NetLogo MARL4AV):**<br>• Thuật toán học tăng cường đa tác nhân (Multi-Agent RL).<br>• Bảng Q-learning cục bộ phân tán.<br>• Phân tích hành vi hợp tác và tránh va chạm giữa các xe tự hành. |

---

## 3. Slide Báo Cáo & Thuyết Trình (`docs/bao_cao_thuyet_trinh/`)

| Tên Tài Liệu | Mô Tả Tóm Tắt |
| :--- | :--- |
| [**`SLIDE_THUYET_TRINH_BUOI_2.md`**](file:///d:/Hoc_Tang_Cuong/docs/bao_cao_thuyet_trinh/SLIDE_THUYET_TRINH_BUOI_2.md) | Kịch bản trình bày, cấu trúc slide, số liệu thực nghiệm và kết luận phân tích cho buổi báo cáo nhóm trước hội đồng/lớp học. |

---

## 4. Hình Ảnh Giao Diện & Kết Quả (`docs/hinh_anh/`)

- [**`preview_main_view.png`**](file:///d:/Hoc_Tang_Cuong/docs/hinh_anh/preview_main_view.png): Màn hình chính khi đóng Menu — siêu tinh gọn chỉ với 4 nút điều khiển ở đáy.
- [**`preview_menu_view.png`**](file:///d:/Hoc_Tang_Cuong/docs/hinh_anh/preview_menu_view.png): Bảng Menu Cài Đặt nổi trung tâm (Popup Settings) — phân nhóm rõ ràng (số làn, mật độ xe, tốc độ, kịch bản mẫu, chế độ lái).
- [**`preview_gui_dashboard.png`**](file:///d:/Hoc_Tang_Cuong/docs/hinh_anh/preview_gui_dashboard.png): Toàn cảnh giao diện HUD 60 FPS với thẻ telemetry và cảm biến an toàn.

---

## 5. Công Cụ Hỗ Trợ Xuất Tài Liệu (`docs/cong_cu/`)

- [**`export_docx.py`**](file:///d:/Hoc_Tang_Cuong/docs/cong_cu/export_docx.py): Script Python sử dụng thư viện `python-docx` để tự động render toàn bộ tài liệu giải thích code sang file Word `.docx` chuẩn xuất bản.
  ```powershell
  & "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" docs\cong_cu\export_docx.py
  ```
