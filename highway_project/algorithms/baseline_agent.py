"""Thuật toán đối sánh chuẩn: Rule-based Heuristic Baseline Agent.

Tác nhân điều khiển xe dựa trên tập luật an toàn giao thông cố định:
- Cảm biến khoảng cách phía trước và thời gian va chạm ước tính (TTC).
- Kiểm tra tính thông thoáng hai bên làn để chuyển làn vượt chướng ngại vật.
- Phanh khẩn cấp khi gặp nguy cơ va chạm mà không thể chuyển làn.
- Tự động duy trì và tăng tốc khi đường thông thoáng.
"""

from __future__ import annotations

import numpy as np


class RuleBasedAgent:
    """Tác nhân chuẩn đối sánh (Baseline) dựa trên tập luật Heuristic."""

    label = "RULE-BASED BASELINE · Tập luật an toàn (Heuristic)"

    def __init__(
        self,
        *,
        safe_gap: float = 30.0,
        critical_gap: float = 12.0,
        target_speed: float = 26.0,  # ~93.6 km/h
        seed: int = 42,
    ):
        self.safe_gap = safe_gap
        self.critical_gap = critical_gap
        self.target_speed = target_speed
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset trạng thái của agent."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        """Quyết định hành động 0..4 dựa trên tập luật heuristic.

        Hành động:
        0: LANE_LEFT  (Chuyển làn sang trái)
        1: IDLE       (Duy trì làn và vận tốc)
        2: LANE_RIGHT (Chuyển làn sang phải)
        3: FASTER     (Tăng tốc)
        4: SLOWER     (Giảm tốc / Phanh)
        """
        sensors = info.get("sensors") if info else None
        available_actions = set(info.get("available_actions", [0, 1, 2, 3, 4])) if info else {0, 1, 2, 3, 4}

        # Nếu không có thông tin sensors chi tiết từ info, suy luận từ ma trận observation (10, 5)
        if not sensors:
            return self._act_from_observation(observation, available_actions)

        lane = sensors.get("lane", 1)
        lanes_count = sensors.get("lanes_count", 3)
        speed = float(sensors.get("speed", 20.0))
        changing = bool(sensors.get("changing", False))
        front_gaps = sensors.get("front_gap", [180.0] * lanes_count)
        front_closings = sensors.get("front_closing", [0.0] * lanes_count)

        ego_gap = front_gaps[lane] if 0 <= lane < len(front_gaps) else 180.0
        closing = front_closings[lane] if 0 <= lane < len(front_closings) else 0.0

        # Ước tính thời gian tới điểm va chạm (Time-to-Collision - TTC)
        ttc = ego_gap / max(closing, 0.01) if closing > 0.1 else 999.0

        left_clear = bool(sensors.get("left_clear", False)) and (0 in available_actions)
        right_clear = bool(sensors.get("right_clear", False)) and (2 in available_actions)

        # Nếu đang trong quá trình chuyển làn dở dang, không ra lệnh chuyển làn tiếp
        if changing:
            left_clear = False
            right_clear = False

        # LUẬT 1: Nguy cơ va chạm cực kỳ nguy hiểm (Critical Emergency)
        if ego_gap < self.critical_gap or ttc < 1.4:
            if left_clear:
                return 0  # Chuyển làn trái khẩn cấp
            if right_clear:
                return 2  # Chuyển làn phải khẩn cấp
            if 4 in available_actions:
                return 4  # Không thể đổi làn -> Phanh khẩn cấp
            return 1

        # LUẬT 2: Gặp vật cản / xe chậm trong khoảng cách cảnh báo (Obstacle Ahead)
        if ego_gap < self.safe_gap or ttc < 3.2:
            # Ưu tiên vượt sang trái (quy tắc giao thông chuẩn)
            if left_clear:
                return 0
            if right_clear:
                return 2
            # Bị kẹt cả 2 bên -> Giảm tốc độ bám đuôi an toàn
            if speed > 14.0 and 4 in available_actions:
                return 4
            return 1

        # LUẬT 3: Đường rất thông thoáng -> Tăng tốc tới tốc độ mục tiêu
        if ego_gap > 50.0:
            if speed < self.target_speed and 3 in available_actions and not changing:
                return 3  # Tăng tốc
            return 1  # Đã đạt tốc độ mục tiêu, giữ đều ga

        # LUẬT 4: Mặc định giữ làn và ổn định hành trình
        return 1

    def _act_from_observation(self, observation: np.ndarray, available_actions: set[int]) -> int:
        """Fallback tính toán luật từ mảng Kinematics khi không có info."""
        if observation is None or len(observation) < 2:
            return 1

        # observation[0]: ego (presence, x, y, vx, vy)
        # observation[1:]: neighboring objects
        min_front_dist = 180.0
        for obj in observation[1:]:
            presence, dx, dy, dvx, _ = obj
            if presence <= 0.5:
                continue
            # Vật cản cùng làn (khoảng lệch dy nhỏ) và ở phía trước (dx > 0)
            if abs(dy) < 0.1 and dx > 0:
                dist = dx * 180.0  # giải chuẩn hóa khoảng cách tương đối
                if dist < min_front_dist:
                    min_front_dist = dist

        if min_front_dist < self.critical_gap:
            return 4 if 4 in available_actions else 1
        if min_front_dist < self.safe_gap:
            if 0 in available_actions:
                return 0
            if 2 in available_actions:
                return 2
            return 4 if 4 in available_actions else 1
        return 1

    def observe(self, *args, **kwargs) -> None:
        """Tác nhân dựa trên luật cố định, không cập nhật trọng số trong quá trình chạy."""
        pass
