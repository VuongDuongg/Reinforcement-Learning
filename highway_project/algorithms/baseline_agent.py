"""Thuật toán đối sánh chuẩn: Physics-informed Rule-based Heuristic Baseline Agent.

Mô hình kết hợp 3 nền tảng điều khiển giao thông kinh điển:
1. Intelligent Driver Model (IDM - Treiber et al.):
   - Tính toán khoảng cách an toàn động lực học: s*(v, Δv) = s0 + v*T + (v*Δv) / (2*sqrt(a*b))
   - Xác định gia tốc tối ưu a_IDM để tự động điều chỉnh FASTER / SLOWER / IDLE.
2. MOBIL Lane Changing Model (Kesting et al.):
   - Tiêu chuẩn an toàn (Safety Criterion): Đảm bảo xe sau tại làn mới không phải phanh gấp.
   - Tiêu chuẩn động cơ (Incentive Criterion): Đổi làn khi gia tốc tăng đáng kể và không gây cản trở.
3. Finite State Machine (FSM - Máy trạng thái an toàn):
   - Quản lý 4 trạng thái hành vi: CRUISING, CAR_FOLLOWING, OVERTAKING, EMERGENCY_BRAKING.
"""

from __future__ import annotations

from enum import Enum
import numpy as np


class AgentState(Enum):
    """Các trạng thái hành vi của xe tự hành trong máy trạng thái FSM."""
    CRUISING = "CRUISING (Tuần hành tự do)"
    CAR_FOLLOWING = "CAR_FOLLOWING (Bám đuôi an toàn)"
    OVERTAKING = "OVERTAKING (Chuyển làn né cản / Vượt xe)"
    EMERGENCY_BRAKING = "EMERGENCY_BRAKING (Phanh khẩn cấp)"


class RuleBasedAgent:
    """Tác nhân chuẩn đối sánh (Baseline) dựa trên tập luật IDM + MOBIL + FSM."""

    label = "RULE-BASED BASELINE · IDM & MOBIL (Heuristic)"

    def __init__(
        self,
        *,
        desired_speed: float = 26.5,     # Tốc độ mong muốn v0 (~95.4 km/h)
        target_speed: float | None = None, # Tương thích ngược
        safe_gap: float = 30.0,          # Tương thích ngược
        critical_gap: float = 12.0,      # Tương thích ngược
        time_headway: float = 1.2,       # Khoảng đệm thời gian an toàn T (giây)
        min_gap: float = 5.0,            # Khoảng cách tối thiểu khi dừng s0 (m)
        max_accel: float = 2.5,          # Gia tốc cực đại a (m/s^2)
        comf_decel: float = 2.0,         # Gia tốc phanh thoải mái b (m/s^2)
        safe_decel: float = 4.0,         # Ngưỡng phanh an toàn b_safe của MOBIL (m/s^2)
        politeness: float = 0.1,         # Hệ số lịch sự p trong giải thuật MOBIL
        lane_change_thresh: float = 0.2, # Ngưỡng gia tốc kích hoạt đổi làn Δa_th (m/s^2)
        seed: int = 42,
    ):
        self.v0 = target_speed if target_speed is not None else desired_speed
        self.safe_gap = safe_gap
        self.critical_gap = critical_gap
        self.T = time_headway
        self.s0 = min_gap
        self.a_max = max_accel
        self.b_comf = comf_decel
        self.b_safe = safe_decel
        self.politeness = politeness
        self.delta_a_th = lane_change_thresh

        self.state = AgentState.CRUISING
        self.rng = np.random.default_rng(seed)
        self.current_action = 1
        self.last_transition_reason = "Khởi tạo"

    def reset(self, *, seed: int | None = None) -> None:
        """Reset trạng thái của agent."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = AgentState.CRUISING
        self.current_action = 1
        self.last_transition_reason = "Khởi động hành trình"

    def calculate_idm_accel(self, speed: float, front_gap: float, closing_speed: float) -> float:
        """Tính toán gia tốc theo công thức Intelligent Driver Model (IDM).

        s*(v, Δv) = s0 + max(0, v*T + (v * Δv) / (2 * sqrt(a * b)))
        a_IDM = a_max * [1 - (v / v0)^4 - (s* / s)^2]
        """
        v = max(0.0, float(speed))
        s = max(0.5, float(front_gap))
        delta_v = float(closing_speed)  # Δv = v_ego - v_lead

        # Khoảng cách bám đuôi động lực học mong muốn s*
        braking_term = (v * delta_v) / (2.0 * np.sqrt(self.a_max * self.b_comf) + 1e-6)
        s_star = self.s0 + max(0.0, v * self.T + braking_term)

        # Công thức gia tốc IDM vi phân
        free_road_term = (v / max(1.0, self.v0)) ** 4
        interaction_term = (s_star / s) ** 2
        accel = self.a_max * (1.0 - free_road_term - interaction_term)
        return float(accel)

    def evaluate_mobil_lane_change(
        self,
        current_lane: int,
        target_lane: int,
        ego_speed: float,
        front_gaps: list[float],
        rear_gaps: list[float],
        front_closings: list[float],
        rear_closings: list[float],
        available_actions: set[int],
    ) -> bool:
        """Kiểm tra điều kiện đổi làn theo giải thuật MOBIL (Kesting & Treiber).

        1. Safety Criterion: Gia tốc phanh của xe phía sau làn mới không vượt ngưỡng nguy hiểm.
        2. Incentive Criterion: Gia tốc của xe tự hành tại làn mới tăng đủ lớn so với làn cũ.
        """
        # Kiểm tra tính khả thi của hành động
        req_action = 0 if target_lane < current_lane else 2
        if req_action not in available_actions:
            return False

        if not (0 <= target_lane < len(front_gaps)):
            return False

        target_front_gap = front_gaps[target_lane]
        target_rear_gap = rear_gaps[target_lane]

        # Kiểm tra khoảng cách vật lý tối thiểu để đảm bảo an toàn tuyệt đối
        if target_front_gap < 18.0 or target_rear_gap < 14.0:
            return False

        # Gia tốc dự kiến ở làn hiện tại
        curr_gap = front_gaps[current_lane]
        curr_closing = front_closings[current_lane]
        accel_curr = self.calculate_idm_accel(ego_speed, curr_gap, curr_closing)

        # Gia tốc dự kiến ở làn mục tiêu
        target_closing = front_closings[target_lane]
        accel_target = self.calculate_idm_accel(ego_speed, target_front_gap, target_closing)

        # 1. Tiêu chuẩn an toàn (Safety Criterion)
        rear_closing = rear_closings[target_lane]
        rear_speed = max(5.0, ego_speed + rear_closing)
        rear_accel_after = self.calculate_idm_accel(rear_speed, target_rear_gap, -rear_closing)
        if rear_accel_after < -self.b_safe:
            return False  # Làm xe sau phải phanh gấp vượt ngưỡng cho phép -> Bác bỏ đổi làn

        # 2. Tiêu chuẩn động cơ (Incentive Criterion)
        advantage = accel_target - accel_curr
        # Ưu tiên đổi sang làn trái nếu thoáng hơn (luật vượt cao tốc chuẩn)
        lane_bias = 0.15 if target_lane < current_lane else 0.0

        return bool((advantage + lane_bias) > self.delta_a_th)

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        """Hàm ra quyết định tối ưu kết hợp FSM + IDM + MOBIL.

        Trả về mã hành động:
        0: LANE_LEFT  (Chuyển làn trái)
        1: IDLE       (Duy trì làn và vận tốc)
        2: LANE_RIGHT (Chuyển làn phải)
        3: FASTER     (Tăng tốc)
        4: SLOWER     (Giảm tốc / Phanh)
        """
        sensors = info.get("sensors") if info else None
        available_actions = set(info.get("available_actions", [0, 1, 2, 3, 4])) if info else {0, 1, 2, 3, 4}

        # Nếu không có sensors chi tiết, chạy hàm suy luận dự phòng từ observation
        if not sensors:
            return self._act_from_observation(observation, available_actions)

        lane = int(sensors.get("lane", 1))
        lanes_count = int(sensors.get("lanes_count", 3))
        speed = float(sensors.get("speed", 20.0))
        changing = bool(sensors.get("changing", False))
        front_gaps = sensors.get("front_gap", [180.0] * lanes_count)
        front_closings = sensors.get("front_closing", [0.0] * lanes_count)
        rear_gaps = sensors.get("rear_gap", [180.0] * lanes_count)
        rear_closings = sensors.get("rear_closing", [0.0] * lanes_count)

        ego_gap = front_gaps[lane] if 0 <= lane < len(front_gaps) else 180.0
        closing = front_closings[lane] if 0 <= lane < len(front_closings) else 0.0

        # Tính thời gian va chạm Time-to-Collision (TTC)
        ttc = ego_gap / max(closing, 0.01) if closing > 0.1 else 999.0

        # Tính toán gia tốc mong muốn theo mô hình IDM
        desired_accel = self.calculate_idm_accel(speed, ego_gap, closing)

        # =========================================================================
        # MÁY TRẠNG THÁI HỮU HẠN (SAFETY FINITE STATE MACHINE)
        # =========================================================================

        # 1. TRẠNG THÁI KHẨN CẤP (EMERGENCY BRAKING)
        if ego_gap < 12.0 or ttc < 1.4 or desired_accel < -self.b_safe:
            self.state = AgentState.EMERGENCY_BRAKING
            # Nếu đang ở cự ly nguy cấp, kiểm tra xem có thể lách khẩn cấp không
            if not changing:
                if self.evaluate_mobil_lane_change(lane, lane - 1, speed, front_gaps, rear_gaps, front_closings, rear_closings, available_actions):
                    self.last_transition_reason = "Lách làn trái tránh va chạm khẩn cấp"
                    return 0
                if self.evaluate_mobil_lane_change(lane, lane + 1, speed, front_gaps, rear_gaps, front_closings, rear_closings, available_actions):
                    self.last_transition_reason = "Lách làn phải tránh va chạm khẩn cấp"
                    return 2

            # Không thể đổi làn -> Phanh khẩn cấp bằng mọi giá
            self.last_transition_reason = f"Phanh khẩn cấp (Cự ly: {ego_gap:.1f}m, TTC: {ttc:.1f}s)"
            return 4 if 4 in available_actions else 1

        # 2. XEM XÉT CHUYỂN LÀN NÉ CẢN / VƯỢT XE (OVERTAKING)
        # Chỉ kích hoạt đổi làn khi làn hiện tại thực sự bị cản trở bởi xe chậm hoặc vật cản
        need_overtake = (ego_gap < 45.0) or (ttc < 3.8) or (closing > 1.5 and ego_gap < 70.0)
        if not changing and need_overtake:
            # Thử đổi làn sang trái (ưu tiên cao)
            can_left = self.evaluate_mobil_lane_change(
                lane, lane - 1, speed, front_gaps, rear_gaps, front_closings, rear_closings, available_actions
            )
            if can_left:
                self.state = AgentState.OVERTAKING
                self.last_transition_reason = "Kích hoạt đổi làn trái (MOBIL hợp lệ)"
                return 0

            # Thử đổi làn sang phải nếu làn trái không thỏa mãn
            can_right = self.evaluate_mobil_lane_change(
                lane, lane + 1, speed, front_gaps, rear_gaps, front_closings, rear_closings, available_actions
            )
            if can_right:
                self.state = AgentState.OVERTAKING
                self.last_transition_reason = "Kích hoạt đổi làn phải (MOBIL hợp lệ)"
                return 2

        # 3. TRẠNG THÁI BÁM ĐUÔI AN TOÀN (CAR FOLLOWING)
        if ego_gap < 60.0 or closing > 0.5:
            self.state = AgentState.CAR_FOLLOWING
            if desired_accel < -0.4 and 4 in available_actions:
                self.last_transition_reason = f"IDM yêu cầu giảm tốc (a = {desired_accel:.2f} m/s^2)"
                return 4
            elif desired_accel > 0.6 and speed < self.v0 and 3 in available_actions and not changing:
                self.last_transition_reason = f"IDM yêu cầu tăng tốc (a = {desired_accel:.2f} m/s^2)"
                return 3
            else:
                self.last_transition_reason = "IDM duy trì ổn định cự ly bám đuôi"
                return 1

        # 4. TRẠNG THÁI TUẦN HÀNH TỰ DO (CRUISING)
        self.state = AgentState.CRUISING
        if speed < (self.v0 - 1.0) and 3 in available_actions and not changing:
            self.last_transition_reason = f"Đường thoáng, tăng tốc đạt v0={self.v0*3.6:.0f} km/h"
            return 3
        elif speed > (self.v0 + 2.0) and 4 in available_actions:
            self.last_transition_reason = "Vượt tốc độ cho phép, giảm tốc"
            return 4
        else:
            self.last_transition_reason = "Duy trì tốc độ hành trình tối ưu"
            return 1

    def _act_from_observation(self, observation: np.ndarray, available_actions: set[int]) -> int:
        """Dự phòng suy luận từ ma trận Kinematics khi không có telemetry info."""
        if observation is None or len(observation) < 2:
            return 1

        min_gap = 180.0
        for obj in observation[1:]:
            presence, dx, dy, dvx, _ = obj
            if presence <= 0.5:
                continue
            if abs(dy) < 0.1 and dx > 0:
                dist = dx * 180.0
                if dist < min_gap:
                    min_gap = dist

        if min_gap < 12.0:
            return 4 if 4 in available_actions else 1
        if min_gap < 32.0:
            if 0 in available_actions:
                return 0
            if 2 in available_actions:
                return 2
            return 4 if 4 in available_actions else 1
        return 3 if min_gap > 50.0 and 3 in available_actions else 1

    def observe(self, *args, **kwargs) -> None:
        """Tác nhân dựa trên luật và công thức động lực học vi phân, không cập nhật trọng số."""
        pass
