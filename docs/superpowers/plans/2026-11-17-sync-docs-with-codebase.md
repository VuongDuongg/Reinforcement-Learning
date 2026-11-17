# Kế Hoạch Chỉnh Sửa Và Đồng Bộ Toàn Bộ Tài Liệu Trong `docs/` Khớp 100% Với Codebase

> **Mục tiêu:** Rà soát, chuẩn hóa và chỉnh sửa toàn diện tất cả các tệp tài liệu trong thư mục `docs/` để phản ánh chính xác 100% các tính năng, thông số kỹ thuật, cấu trúc mã nguồn, giao diện Menu Modal, cơ chế tái cấu hình trực tiếp (Live Reconfiguration), và bộ 21 unit tests hiện có trong dự án.  
> **Ngày lập:** 17/11/2026  
> **Tài liệu áp dụng:** Toàn bộ thư mục `docs/` (`docs/moi_truong/`, `docs/thuat_toan/`, `docs/bao_cao_thuyet_trinh/`, `docs/cong_cu/`, `docs/README.md`).

---

## 1. Danh Mục Các Điểm Cần Đồng Bộ Giữa Code Và Docs

| Thành phần Code thực tế | Tình trạng trong Docs cũ | Hành động đồng bộ cần thực hiện |
| :--- | :--- | :--- |
| **Số lượng Unit Tests** | Ghi "Bộ 20 bài test tự động" | Cập nhật thành **21 bài test tự động**, bổ sung đặc tả chi tiết bài test `test_live_reconfiguration_preserves_vehicle_progress`. |
| **Cơ chế Tái cấu hình trực tiếp (Live Reconfiguration)** | Chưa được mô tả trong tài liệu môi trường | Thêm mục phân tích chuyên sâu về hàm `live_reconfigure(...)`: cập nhật `RoadNetwork`, clamp tọa độ $y$ các xe, bảo toàn vị trí $x$, quãng đường `distance_m` và điểm số mà **không reset xe**. |
| **Giao diện Menu Cài Đặt nổi trung tâm (Popup Modal)** | Docs mô tả giao diện cũ các nút bấm tràn lan dưới đáy | Mô tả kiến trúc Menu Modal mở bằng phím `M` / nút `☰`, lớp phủ mờ tối `dim_surface`, 6 phân mục cài đặt, và thanh điều khiển đáy siêu tinh gọn chỉ 4 nút. |
| **Thông báo nổi Toast thời gian thực** | Chưa có tài liệu | Thêm mô tả phương thức `show_toast(text, duration_ms, color)` hiển thị phản hồi tức thì góc dưới bên phải màn hình. |
| **Tập tin chạy chương trình** | Chưa đồng bộ với `CHAY_MOI_TRUONG.cmd` duy nhất & `launcher.py` | Bổ sung `CHAY_MOI_TRUONG.cmd` (script chạy duy nhất cho Windows) và `highway_project/launcher.py` (menu console hỗ trợ UTF-8). |
| **Thông số `default_config()`** | Một số tài liệu còn ghi `route_length: 650m`, `duration: 60s`, `vehicles_count: 12` | Đồng bộ toàn bộ về: `route_length: 1000.0m`, `duration: 90.0s`, `vehicles_count: 20`, `obstacle_distance: 180.0m`, `vehicle_spacing: 28.0m`. |
| **Đường dẫn `rl_bridge.py` trong MARL4AV** | Ghi nhầm `marl4av/algorithms/rl_bridge.py` | Đổi thành `marl4av/rl_bridge.py` (vị trí file thực tế). |
| **Liên kết tương đối trong `DE_XUAT_MOI_TRUONG.md`** | Liên kết `[hướng dẫn](highway_project/README.md)` bị sai cấp thư mục | Sửa thành `../../highway_project/README.md` hoặc clickable markdown link chuẩn. |
| **Tài liệu Word `.docx`** | Chưa cập nhật các nội dung mới | Cập nhật mã nguồn `docs/cong_cu/export_docx.py` và chạy xuất lại file `GIAI_THICH_CODE_MOI_TRUONG.docx` chuẩn đẹp. |

---

## 2. Kế Hoạch Triển Khai Chi Tiết Từng Bước

### Task 1: Cập nhật `docs/moi_truong/GIAI_THICH_CODE_MOI_TRUONG.md`
- [ ] Cập nhật bảng phân công trách nhiệm file ở Mục 1.3: Thêm `CHAY_MOI_TRUONG.cmd`, `CAI_DAT_MOI_TRUONG.cmd`, `highway_project/launcher.py`, đổi 20 -> 21 unit tests.
- [ ] Kiểm tra và chuẩn hóa cấu hình `default_config()` ở Mục 2.
- [ ] Bổ sung Mục 8.3: Bảng Menu Cài Đặt nổi trung tâm (Modal Popup Settings).
- [ ] Bổ sung Mục 8.4: Cơ chế Tái cấu hình trực tiếp (Live Reconfiguration) không reset xe.
- [ ] Bổ sung Mục 8.5: Hệ thống thông báo nổi Toast thời gian thực (`show_toast`).
- [ ] Bổ sung Mục 8.6: Thanh điều khiển chính tinh gọn 4 nút ở màn hình chính.
- [ ] Cập nhật Mục 10: Nâng lên 21 bài kiểm thử tự động, thêm bài test số 13 `test_live_reconfiguration_preserves_vehicle_progress`.

### Task 2: Cập nhật `docs/cong_cu/export_docx.py` & Xuất lại `GIAI_THICH_CODE_MOI_TRUONG.docx`
- [ ] Cập nhật nội dung trong `export_docx.py` tương ứng với các mục mới (21 tests, Menu Modal, Live Reconfiguration).
- [ ] Chạy script `docs/cong_cu/export_docx.py` để tạo lại file Word `.docx`.

### Task 3: Cập nhật `docs/thuat_toan/THUAT_TOAN_DQN.md`
- [ ] Rà soát vector 68 chiều ở Mục 5.1: 18 chiều cảm biến + 50 chiều ma trận động học.
- [ ] Mô tả chi tiết cơ chế chiếu tương đối 3 kênh ego-centric `[trái, hiện tại, phải]` để mô hình DQN thích ứng linh hoạt từ 2 đến 6 làn mà không lỗi kích thước tensor.
- [ ] Cập nhật Mục 6.1: Hướng dẫn chạy thông qua file `CHAY_MOI_TRUONG.cmd` và giao diện Pygame Menu Modal.
- [ ] Cập nhật Mục 8: Hướng dẫn tương tác đa kịch bản qua Menu Cài Đặt nổi (`M`), phím tắt nhanh (`L`, `T`, `S`, `F`, `1..4`) và CLI.

### Task 4: Chuẩn hóa đường dẫn trong `BAO_CAO_QUY_TAC_MOI_TRUONG_VA_MDP.md`, `BAO_CAO_TH2.md` và `DE_XUAT_MOI_TRUONG.md`
- [ ] Đổi `marl4av/algorithms/rl_bridge.py` thành `marl4av/rl_bridge.py` trong cả 2 file báo cáo.
- [ ] Thêm ghi chú liên kết chéo (cross-reference note) tới thư mục `highway_project` (nền tảng Gymnasium & PyTorch DQN mới nhất).
- [ ] Sửa các đường dẫn tương đối trong `DE_XUAT_MOI_TRUONG.md`.

### Task 5: Cập nhật `docs/bao_cao_thuyet_trinh/SLIDE_THUYET_TRINH_BUOI_2.md` và `docs/README.md`
- [ ] Cập nhật các slide trong `SLIDE_THUYET_TRINH_BUOI_2.md` cho khớp các thông số thực tế của dự án (`1000m`, `90s`, `20 xe`, hỗ trợ `2..6 làn`, Menu Modal, 21 unit tests).
- [ ] Cập nhật `docs/README.md` phản ánh đầy đủ cấu trúc tài liệu, tính năng mới và các đường dẫn.

### Task 6: Kiểm thử toàn diện & Đẩy code lên GitHub
- [ ] Chạy `python -m unittest discover -s highway_project -t .` xác nhận 21/21 tests PASS.
- [ ] Kiểm tra `git status` và `git diff`.
- [ ] Commit và push lên GitHub remote `origin main`.
