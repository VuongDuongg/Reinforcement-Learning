"""Giao diện Pygame tương tác chuyên nghiệp cho HighwayEnv.

Hỗ trợ tùy biến trực tiếp số làn, mật độ xe, tốc độ chạy mà không reset vị trí xe.
Tích hợp thanh đo cảm biến radar, telemetry AI, và bảng nút bấm tương tác thời gian thực.
"""

from __future__ import annotations

import math
from pathlib import Path

import pygame

# Bảng màu giao diện chuẩn Cyber-Tech tối ưu cho hiển thị đồ họa khoa học
BG = (13, 20, 30)
PANEL = (22, 33, 47)
PANEL_HOVER = (34, 50, 70)
PANEL_BORDER = (42, 60, 82)
TEXT = (235, 242, 250)
MUTED = (140, 160, 180)
MINT = (46, 213, 162)
BLUE = (80, 150, 250)
YELLOW = (255, 204, 0)
RED = (248, 92, 92)


class HighwayView:
    WIDTH, HEIGHT = 1280, 800
    SCALE, EGO_X = 4.6, 260

    def __init__(self, *, headless: bool = False):
        pygame.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT)) if headless else pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Học Tăng Cường | Điều Khiển Xe Tự Hành Né Vật Cản Cao Tốc (DQN)")
        
        # Nạp các cỡ font chuẩn giao diện
        self.fonts = {size: pygame.font.SysFont("segoeui,arial,dejavusans", size) for size in (11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 26, 28, 34, 36)}
        self.buttons: dict[str, pygame.Rect] = {}
        
        # Hệ thống thông báo nổi (Toast Notification) thời gian thực
        self.toast_text = ""
        self.toast_expire = 0
        self.toast_color = MINT

    def get_font(self, size: int) -> pygame.font.Font:
        return self.fonts.get(size) or self.fonts.get(16) or next(iter(self.fonts.values()))

    def text(self, text, x, y, size=18, color=TEXT):
        font = self.get_font(size)
        self.screen.blit(font.render(str(text), True, color), (x, y))

    def show_toast(self, text: str, duration_ms: int = 2500, color=MINT):
        self.toast_text = text
        self.toast_expire = pygame.time.get_ticks() + duration_ms
        self.toast_color = color

    def draw(self, env, info, *, paused, mode, last_action, total_reward, episode,
             motion=None, alpha=1.0, agent_label="AGENT CHỜ CÀI ĐẶT"):
        self.screen.fill(BG)
        mouse_pos = pygame.mouse.get_pos()

        lanes_count = int(env.config.get("lanes_count", 3))
        # Chiều cao và vị trí mặt đường co giãn theo số làn đường
        if lanes_count <= 2:
            road_top = 215
            lane_height = 68
        elif lanes_count == 3:
            road_top = 215
            lane_height = 64
        elif lanes_count == 4:
            road_top = 210
            lane_height = 56
        elif lanes_count == 5:
            road_top = 205
            lane_height = 50
        else:  # 6 làn
            road_top = 200
            lane_height = 44

        road_height = lanes_count * lane_height
        road_bottom = road_top + road_height

        # ---------------------------------------------------------
        # 1. TIÊU ĐỀ CHÍNH & HUY HIỆU TRẠNG THÁI (HEADER)
        # ---------------------------------------------------------
        self.text("RL / GYMNASIUM HIGHWAY-ENV  ·  DOUBLE DQN", 38, 16, 12, MINT)
        self.text(f"Cao tốc {lanes_count} làn · Né chướng ngại vật tự động", 36, 34, 30)

        algo_title = "Deep Q-Network (PyTorch)" if "DQN" in str(agent_label) else "Q-learning (Bảng Q)"
        density = info.get("sensors", {}).get("traffic_density", "normal")
        density_vi = {
            "sparse": "Thưa thớt",
            "normal": "Bình thường",
            "dense": "Dày đặc",
            "none": "Không có xe"
        }.get(density, "Bình thường")
        
        cfg_speed = int(round(float(env.config.get("initial_speed", 20.0)) * 3.6))
        vehicles_total = len([v for v in env.road.vehicles if v is not env.vehicle])
        self.text(
            f"Thuật toán: {algo_title}  |  Mật độ xe: {density_vi} ({vehicles_total} xe)  |  Vận tốc xuất phát: {cfg_speed} km/h",
            38, 72, 14, MUTED
        )

        # Huy hiệu trạng thái mô phỏng góc phải trên
        badge_rect = pygame.Rect(1060, 24, 184, 32)
        pygame.draw.rect(self.screen, PANEL, badge_rect, border_radius=16)
        pygame.draw.rect(self.screen, (34, 52, 70), badge_rect, width=1, border_radius=16)
        pygame.draw.circle(self.screen, RED if paused else MINT, (1076, 40), 5)
        badge_text = "TẠM DỪNG" if paused else "MÔ PHỎNG 60 FPS"
        self.text(badge_text, 1088, 31, 12, TEXT)

        # ---------------------------------------------------------
        # 2. BỐN THẺ CHỈ SỐ HOẠT ĐỘNG (TELEMETRY CARDS)
        # ---------------------------------------------------------
        sensors = info.get("sensors", {})
        ego_speed_kmh = env.vehicle.speed * 3.6
        target_speed_kmh = env.vehicle.target_speed * 3.6 if hasattr(env.vehicle, "target_speed") else ego_speed_kmh
        ego_front_gap = sensors.get("ego_front_gap", 180.0)
        gap_str = f"{ego_front_gap:.1f} m" if ego_front_gap < 170 else "> 180 m"

        cards = [
            ("VẬN TỐC HIỆN TẠI", f"{ego_speed_kmh:.0f}", "km/h", f"Mục tiêu: {target_speed_kmh:.0f} km/h", MINT if not env.vehicle.crashed else RED),
            ("QUÃNG ĐƯỜNG", f"{info['distance_m']:.0f}", "m", f"Còn lại: {info['remaining_m']:.0f} m / {env.config.get('route_length', 1000):.0f} m", TEXT),
            ("VẬT CẢN CAO TỐC", f"{info['obstacles_passed']}/{info['obstacles_total']}", "xe", f"Xe trước: {gap_str}", YELLOW if ego_front_gap < 35 else TEXT),
            ("THỜI GIAN & ĐIỂM", f"{info['elapsed_s']:.1f}", "s", f"Reward: {total_reward:+.2f}  ·  Ep {episode}", TEXT),
        ]
        for index, (label, value, unit, subtext, val_color) in enumerate(cards):
            x = 36 + index * 308
            rect = pygame.Rect(x, 100, 292, 82)
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
            pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=10)
            self.text(label, x + 16, 110, 12, MUTED)
            self.text(value, x + 16, 128, 30, val_color)
            self.text(unit, x + 16 + len(value) * 18 + 8, 142, 14, MUTED)
            self.text(subtext, x + 16, 162, 11, MUTED)

        # ---------------------------------------------------------
        # 3. MẶT ĐƯỜNG CAO TỐC & XE LƯU THÔNG
        # ---------------------------------------------------------
        self.text("TRÁI / LÀN 0", 38, road_top - 20, 12, MUTED)
        self.text("Chiều di chuyển xe  →", 1085, road_top - 20, 13, MINT)

        # Lề cỏ và mặt đường nhựa
        pygame.draw.rect(self.screen, (28, 48, 42), (0, road_top - 6, self.WIDTH, road_height + 12))
        pygame.draw.rect(self.screen, (46, 56, 68), (0, road_top, self.WIDTH, road_height))

        # Vạch liền màu vàng hai bên lề
        for y in (road_top + 3, road_bottom - 3):
            pygame.draw.line(self.screen, (220, 195, 100), (0, y), (self.WIDTH, y), 3)

        def pose(obj):
            return motion.sample(obj, alpha) if motion else (obj.position, obj.heading)

        camera_x = float(pose(env.vehicle)[0][0])
        offset = int(camera_x * self.SCALE) % 66

        # Vạch đứt phân làn
        for row in range(1, lanes_count):
            y = road_top + row * lane_height
            for x in range(-offset, self.WIDTH, 66):
                pygame.draw.line(self.screen, (176, 189, 204), (x, y), (x + 32, y), 2)

        self.text(f"PHẢI / LÀN {lanes_count - 1}", 38, road_bottom + 5, 12, MUTED)

        def coords(obj):
            position, _ = pose(obj)
            return (self.EGO_X + (position[0] - camera_x) * self.SCALE,
                    road_top + lane_height / 2 + position[1] / 4 * lane_height)

        # Mốc mặt đường theo quãng đường thực tế
        marker_y = road_bottom + 18
        for marker in range(int(camera_x // 50) * 50 - 50, int(camera_x + 300), 50):
            x = self.EGO_X + (marker - camera_x) * self.SCALE
            self.text(f"{marker - env.start_x:.0f} m", x, marker_y, 12, MUTED)

        # Vẽ vật cản và xe lưu thông
        for obj in [*env.road.vehicles, *env.obstacles]:
            x, y = coords(obj)
            if x < -80 or x > self.WIDTH + 80:
                continue
            obstacle = obj in env.obstacles
            ego = obj is env.vehicle
            color = RED if obstacle or obj.crashed else MINT if ego else BLUE

            car = pygame.Surface((46, 30), pygame.SRCALPHA)
            pygame.draw.rect(car, color, (2, 2, 42, 26), border_radius=6)
            # Kính chắn gió trước/sau
            pygame.draw.rect(car, (20, 35, 48), (29, 6, 7, 18), border_radius=2)
            pygame.draw.rect(car, (20, 35, 48), (8, 6, 6, 18), border_radius=2)

            if obstacle:
                # Dấu chữ X cảnh báo xe hỏng
                pygame.draw.line(car, (255, 235, 205), (17, 9), (26, 21), 2)
                pygame.draw.line(car, (255, 235, 205), (26, 9), (17, 21), 2)

            rotated = pygame.transform.rotate(car, -math.degrees(pose(obj)[1]))
            self.screen.blit(rotated, rotated.get_rect(center=(x, y)))

            if ego or obstacle:
                label_color = color
                label_text = "XE CỦA BẠN" if ego else "XE HỎNG"
                self.text(label_text, x - 38, y - 34, 12, label_color)

        # Vạch đích đến
        goal_x = self.EGO_X + (env.goal_x - camera_x) * self.SCALE
        if 0 < goal_x < self.WIDTH:
            pygame.draw.line(self.screen, MINT, (goal_x, road_top), (goal_x, road_bottom), 4)
            self.text("ĐÍCH ĐẾN", goal_x + 8, road_top + 6, 14, MINT)

        # Hộp thoại trạng thái kết thúc hoặc tạm dừng
        reason = info.get("end_reason", "running")
        status = {
            "collision": "VA CHẠM — Bấm R để thử lại",
            "success": "ĐẾN ĐÍCH AN TOÀN — Bấm N để sang kịch bản mới",
            "time_limit": "HẾT THỜI GIAN — Bấm R để thử lại",
            "off_road": "RA KHỎI ĐƯỜNG — Bấm R để thử lại",
        }.get(reason)

        popup_y = road_top + (road_height - 50) // 2
        if status:
            pygame.draw.rect(self.screen, BG, (310, popup_y, 660, 50), border_radius=10)
            pygame.draw.rect(self.screen, MINT if reason == "success" else RED, (310, popup_y, 660, 50), width=2, border_radius=10)
            self.text(status, 330, popup_y + 13, 20, MINT if reason == "success" else RED)
        elif paused:
            pygame.draw.rect(self.screen, BG, (380, popup_y, 520, 50), border_radius=10)
            pygame.draw.rect(self.screen, YELLOW, (380, popup_y, 520, 50), width=2, border_radius=10)
            self.text("TẠM DỪNG — Bấm Space để xe chạy tiếp", 405, popup_y + 13, 20, YELLOW)

        # ---------------------------------------------------------
        # 4. BẢNG ĐIỀU KHIỂN RADAR CẢM BIẾN & TELEMETRY AI (HUD PANEL)
        # ---------------------------------------------------------
        panel_y = 485
        # 4.1. Panel Trái: Radar cảm biến làn đường
        left_p = pygame.Rect(36, panel_y, 594, 76)
        pygame.draw.rect(self.screen, PANEL, left_p, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, left_p, width=1, border_radius=10)
        
        self.text("RADAR CẢM BIẾN AN TOÀN", 52, panel_y + 8, 12, MUTED)
        curr_lane = sensors.get("lane", 1)
        self.text(f"Làn hiện tại: {curr_lane} / {lanes_count - 1}", 52, panel_y + 28, 15, TEXT)
        self.text(f"Khoảng cách xe trước: {gap_str}", 52, panel_y + 48, 14, YELLOW if ego_front_gap < 35 else MINT)

        left_clear = sensors.get("left_clear", False)
        right_clear = sensors.get("right_clear", False)
        
        lc_color = MINT if left_clear else RED
        rc_color = MINT if right_clear else RED
        lc_text = "Làn trái: AN TOÀN" if left_clear else "Làn trái: CÓ XE / BIÊN"
        rc_text = "Làn phải: AN TOÀN" if right_clear else "Làn phải: CÓ XE / BIÊN"
        self.text(lc_text, 360, panel_y + 28, 14, lc_color)
        self.text(rc_text, 360, panel_y + 48, 14, rc_color)

        # 4.2. Panel Phải: Quyết định & Trạng thái điều khiển AI
        right_p = pygame.Rect(650, panel_y, 594, 76)
        pygame.draw.rect(self.screen, PANEL, right_p, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_BORDER, right_p, width=1, border_radius=10)

        self.text("QUYẾT ĐỊNH & TRẠNG THÁI AI", 666, panel_y + 8, 12, MUTED)
        mode_text = "BÀN PHÍM (TỰ LÁI)" if mode == "keyboard" else agent_label
        self.text(f"Chế độ: {mode_text}", 666, panel_y + 28, 15, MINT)
        
        action_vi = {
            "LANE_LEFT": "← Chuyển làn trái",
            "IDLE": "● Giữ làn & tốc độ",
            "LANE_RIGHT": "→ Chuyển làn phải",
            "FASTER": "↑ Tăng tốc",
            "SLOWER": "↓ Giảm tốc / Phanh",
        }.get(str(last_action), str(last_action))
        self.text(f"Hành động AI: {action_vi}", 666, panel_y + 48, 14, TEXT)

        is_changing = sensors.get("changing", False)
        status_text = "Đang đổi làn..." if is_changing else ("Cảnh báo va chạm!" if ego_front_gap < 25 else "Đường thông thoáng")
        status_col = YELLOW if is_changing else (RED if ego_front_gap < 25 else MUTED)
        self.text(f"Trạng thái: {status_text}", 980, panel_y + 28, 13, status_col)

        # ---------------------------------------------------------
        # 5. HỆ THỐNG NÚT BẤM TƯƠNG TÁC THỜI GIAN THỰC (KHÔNG RESET XE)
        # ---------------------------------------------------------
        # Hàng 1: Tùy biến môi trường trực tiếp
        density_short = {"sparse": "Thưa", "normal": "Vừa", "dense": "Dày", "none": "Không"}.get(density, "Vừa")
        tspd_min = float(env.config.get("traffic_speed_min", 14.0))
        flow_short = "Nhanh" if tspd_min >= 20.0 else ("Chậm" if tspd_min <= 11.0 else "Vừa")

        row1_buttons = [
            ("lanes", f"L · Làn: {lanes_count}", 36, 135),
            ("traffic", f"T · Xe: {density_short}", 181, 150),
            ("speed", f"S · {cfg_speed} km/h", 341, 145),
            ("traffic_speed", f"F · Dòng: {flow_short}", 496, 140),
            ("preset_1", "1: 2 Làn Thưa", 646, 140),
            ("preset_2", "2: 3 Làn Chuẩn", 796, 145),
            ("preset_3", "3: 5 Làn Dày", 951, 145),
            ("preset_4", "4: 4 Làn Vắng", 1106, 138),
        ]

        for key, label, x, width in row1_buttons:
            rect = pygame.Rect(x, 578, width, 42)
            self.buttons[key] = rect
            is_hover = rect.collidepoint(mouse_pos)
            bg_col = PANEL_HOVER if is_hover else PANEL
            border_col = MINT if is_hover else PANEL_BORDER
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_col, rect, width=1, border_radius=8)
            self.text(label, x + 10, 590, 14, MINT if is_hover else TEXT)

        # Hàng 2: Điều khiển chạy & thông báo nổi Toast
        pause_label = "Space · Tiếp tục" if paused else "Space · Tạm dừng"
        mode_btn_label = "Tab · AI: DQN" if mode == "agent" else "Tab · Lái: Phím"

        row2_buttons = [
            ("pause", pause_label, 36, 165),
            ("mode", mode_btn_label, 211, 195),
            ("reset", "R · Chạy lại từ đầu", 416, 180),
            ("new", "N · Hạt giống mới", 606, 185),
        ]

        for key, label, x, width in row2_buttons:
            rect = pygame.Rect(x, 630, width, 42)
            self.buttons[key] = rect
            is_hover = rect.collidepoint(mouse_pos)
            bg_col = PANEL_HOVER if is_hover else PANEL
            border_col = YELLOW if key == "pause" and paused else (MINT if is_hover else PANEL_BORDER)
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_col, rect, width=1, border_radius=8)
            self.text(label, x + 12, 642, 14, MINT if is_hover else TEXT)

        # Khu vực thông báo nổi Toast thời gian thực
        toast_rect = pygame.Rect(801, 630, 443, 42)
        pygame.draw.rect(self.screen, PANEL, toast_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, toast_rect, width=1, border_radius=8)

        if pygame.time.get_ticks() < self.toast_expire and self.toast_text:
            pygame.draw.rect(self.screen, (28, 44, 56), toast_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.toast_color, toast_rect, width=1, border_radius=8)
            self.text(f"✓  {self.toast_text}", 815, 642, 13, self.toast_color)
        else:
            self.text("Trực tiếp: Thao tác không reset xe đang chạy", 815, 642, 13, MUTED)

        # ---------------------------------------------------------
        # 6. DÒNG CHÚ THÍCH PHÍM TẮT & HƯỚNG DẪN Ở CHÂN TRANG
        # ---------------------------------------------------------
        self.text(
            "Tùy biến trực tiếp không reset xe: Bấm L đổi làn (2..6) · T đổi mật độ xe · S đổi tốc độ · F đổi tốc độ dòng xe · 1..4 kịch bản mẫu",
            38, 686, 13, MUTED
        )
        if mode == "keyboard":
            self.text(
                "Lái xe bằng bàn phím: ← / → chuyển làn  ·  ↑ tăng tốc  ·  ↓ giảm tốc/phanh  |  Bấm Tab để chuyển lại cho AI tự lái",
                38, 706, 13, MINT
            )
        else:
            self.text(
                "Điều khiển mô phỏng: Bấm Space tạm dừng  ·  Bấm Tab đổi sang tự lái  ·  Bấm R chạy lại từ đầu  ·  Bấm N sinh hạt giống mới",
                38, 706, 13, (120, 180, 240)
            )

    def save(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(target))
