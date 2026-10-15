"""
algorithms/baseline_agent.py
Thuật toán Đối sánh: Rule-based Heuristic Baseline
Thành viên phụ trách: Lê Tuấn Khanh (MSV: 2351170601)
Nhóm: N1_65KTPM - Đại học Thủy Lợi
"""

import numpy as np

class RuleBasedAgent:
    """
    Tác nhân chuẩn đối sánh (Baseline) dựa trên tập luật an toàn giao thông cố định:
    - Nếu khoảng cách tới xe phía trước lớn (> safe_distance): Tăng tốc (Action 0).
    - Nếu có chướng ngại vật ở cự ly nguy hiểm: Chuyển làn (Action 3).
    - Nếu không thể chuyển làn: Giảm tốc/phanh (Action 2).
    """
    def __init__(self, safe_distance=4.0):
        self.safe_distance = safe_distance

    def act(self, states, epsilon=0.0):
        """
        Duyệt qua từng xe và quyết định hành động bằng luật.
        Giả định state vector: [speed, dist_ahead, speed_ahead, dist_behind, speed_behind]
        """
        actions = []
        for s in states:
            dist_ahead = s[1] if len(s) > 1 else 100.0
            
            if dist_ahead > self.safe_distance * 2:
                # Đường rất thoáng -> Tăng tốc
                action = 0  # Accelerate
            elif dist_ahead > self.safe_distance:
                # Đường bình thường -> Duy trì tốc độ
                action = 1  # Stay same
            elif dist_ahead > 1.5:
                # Bắt đầu gặp chướng ngại vật -> Chủ động đổi làn
                action = 3  # Change lane
            else:
                # Quá gần nguy cơ va chạm -> Phanh khẩn cấp
                action = 2  # Decelerate

            actions.append(action)
        return actions

    def remember(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        return 0.0
