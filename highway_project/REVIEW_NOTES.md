# Ghi chú review bản thử

- Môi trường HighwayEnv 3 làn, xe giao thông và xe hỏng đứng yên; điều khiển bàn phím hoặc chính sách Q-learning.
- Nội suy vị trí/hướng để giảm giật giữa các bước mô phỏng 10 Hz và vòng vẽ 60 FPS.
- Bảng Q đã chạy 180 episode, 23.994 bước cập nhật. Không có luật chọn hành động né viết sẵn.
- 8 kiểm tra môi trường, Bellman, lưu/tải model và nội suy đã qua trên Windows/Python 3.13.
- Đây là bản thử chưa hội tụ: kiểm tra ban đầu với 0 xe giao thông, 10 seed mới cho tỷ lệ đến đích 0%, va chạm 0%; xe có thể giảm tốc/dừng và hết giờ. Chưa hoàn tất đánh giá 6/12 xe. Không coi model là chính sách lái xe đã đạt yêu cầu.
- Mở CAI_DAT_MOI_TRUONG.cmd một lần rồi CHAY_XE_TU_HOC.cmd. Space dừng/chạy, Tab đổi qua bàn phím, R chạy lại, N đổi seed.
- Không cần TensorFlow/PyTorch/GPU. Không kèm môi trường ảo, cache hoặc log.
