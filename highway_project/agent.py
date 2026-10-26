"""Chỗ nối thuật toán sau này. Hiện tại CHỈ giữ tốc, không học hay tự né."""

from __future__ import annotations

import numpy as np


class CustomAgent:
    def reset(self, *, seed: int) -> None:
        """TODO: reset bộ nhớ/quỹ đạo nếu thuật toán cần."""

    def act(self, observation: np.ndarray, info: dict) -> int:
        """TODO: thay bằng suy luận mô hình; trả về một số nguyên 0..4.

        observation: float32, shape (10, 5), chuẩn hóa theo HighwayEnv.
        Các cột: presence, x, y, vx, vy; dòng đầu là xe được điều khiển.
        0 trái, 1 giữ, 2 phải, 3 tăng tốc, 4 giảm tốc.
        info có available_actions để nhận biết hành động khả dụng ở mép làn.
        """
        return 1

    def observe(self, observation, action, reward, next_observation, terminated, truncated, info) -> None:
        """TODO: ghi kinh nghiệm nếu cần; giữ riêng hai cờ kết thúc.

        Không mặc định truncated (hết giờ) là terminal khi tính bootstrap.
        Hàm này hiện không huấn luyện hay lưu dữ liệu.
        """
