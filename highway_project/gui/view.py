"""Giao diện Pygame tương tác chuyên nghiệp cho HighwayEnv.

Hỗ trợ hệ thống Popup / Overlay Menu trung tâm tinh gọn, loại bỏ dàn trải nút bấm.
Tùy biến trực tiếp số làn, mật độ xe, tốc độ chạy mà không reset xe.
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
MODAL_BG = (18, 27, 39)
MODAL_BORDER = (54, 82, 114)
TEXT = (235, 242, 250)
MUTED = (140, 160, 180)
MINT = (46, 213, 162)
MINT_DIM = (20, 52, 52)
BLUE = (80, 150, 250)
YELLOW = (255, 204, 0)
RED = (248, 92, 92)


class HighwayView:
    WIDTH, HEIGHT = 1280, 800
    SCALE, EGO_X = 4.6, 260

    def __init__(self, *, headless: bool = False):
        pygame.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT)) if headless else pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        if not headless and hasattr(pygame.key, "stop_text_input"):
            pygame.key.stop_text_input()
        pygame.display.set_caption("Học Tăng Cường | Điều Khiển Xe Tự Hành Né Vật Cản Cao Tốc (RL)")
        
        # Nạp các cỡ font chuẩn giao diện
        self.fonts = {size: pygame.font.SysFont("segoeui,arial,dejavusans", size) for size in (11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 32, 36)}
        self.buttons: dict[str, pygame.Rect] = {}
        
        # Trạng thái mở Bảng Cài Đặt (Popup / Overlay Menu)
        self.menu_open: bool = False
        self.modal_rect = pygame.Rect(160, 75, 960, 650)

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

    def toggle_menu(self):
        self.menu_open = not self.menu_open

    def draw(self, env, info, *, paused, mode, last_action, total_reward, episode,
             motion=None, alpha=1.0, agent_label="AGENT CHỜ CÀI ĐẶT"):
        self.screen.fill(BG)
        mouse_pos = pygame.mouse.get_pos()
        self.buttons.clear()

        lanes_count = int(env.config.get("lanes_count", 3))
        # Chiều cao và vị trí mặt đường co giãn theo số làn đường
        if lanes_count <= 2:
            road_top = 220
            lane_height = 68
        elif lanes_count == 3:
            road_top = 220
            lane_height = 64
        elif lanes_count == 4:
            road_top = 215
            lane_height = 56
        elif lanes_count == 5:
            road_top = 210
            lane_height = 50
        else:  # 6 làn
            road_top = 205
            lane_height = 44

        road_height = lanes_count * lane_height
        road_bottom = road_top + road_height

        # ---------------------------------------------------------
        # 1. TIÊU ĐỀ CHÍNH & HUY HIỆU TRẠNG THÁI (HEADER)
        # ---------------------------------------------------------
        is_ddqn = "Dueling" in str(agent_label)
        is_dqn = "DQN" in str(agent_label)
        header_algo = "DOUBLE DUELING DQN" if is_ddqn else ("DOUBLE DQN" if is_dqn else "Q-LEARNING")
        self.text(f"RL / GYMNASIUM HIGHWAY-ENV  ·  {header_algo}", 38, 16, 12, MINT)
        self.text(f"Cao tốc {lanes_count} làn · Né chướng ngại vật tự động", 36, 34, 30)

        if is_ddqn:
            algo_title = "Double Dueling DQN (Ngô Tuấn Anh)"
        elif is_dqn:
            algo_title = "Deep Q-Network (PyTorch)"
        else:
            algo_title = "Q-learning (Bảng Q)"
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
            rect = pygame.Rect(x, 102, 292, 80)
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
            pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=10)
            self.text(label, x + 16, 111, 12, MUTED)
            self.text(value, x + 16, 127, 28, val_color)
            self.text(unit, x + 16 + len(value) * 17 + 8, 139, 14, MUTED)
            self.text(subtext, x + 16, 160, 11, MUTED)

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
            pygame.draw.rect(car, (20, 35, 48), (29, 6, 7, 18), border_radius=2)
            pygame.draw.rect(car, (20, 35, 48), (8, 6, 6, 18), border_radius=2)

            if obstacle:
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

        # Hộp thoại thông báo va chạm hoặc về đích
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
        elif paused and not self.menu_open:
            pygame.draw.rect(self.screen, BG, (380, popup_y, 520, 50), border_radius=10)
            pygame.draw.rect(self.screen, YELLOW, (380, popup_y, 520, 50), width=2, border_radius=10)
            self.text("TẠM DỪNG — Bấm Space để xe chạy tiếp", 405, popup_y + 13, 20, YELLOW)

        # ---------------------------------------------------------
        # 4. BẢNG ĐIỀU KHIỂN RADAR CẢM BIẾN & TELEMETRY AI (HUD PANEL)
        # ---------------------------------------------------------
        panel_y = 485
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
        self.text("Làn trái: AN TOÀN" if left_clear else "Làn trái: CÓ XE / BIÊN", 360, panel_y + 28, 14, lc_color)
        self.text("Làn phải: AN TOÀN" if right_clear else "Làn phải: CÓ XE / BIÊN", 360, panel_y + 48, 14, rc_color)

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
        self.text(f"Trạng thái: {status_text}", 965, panel_y + 48, 13, status_col)

        # ---------------------------------------------------------
        # 5. THANH ĐIỀU KHIỂN CHÍNH GỌN GÀNG (MAIN VIEW - CHỈ 4 NÚT)
        # ---------------------------------------------------------
        pause_label = "Space · Tiếp tục" if paused else "Space · Tạm dừng"
        ai_short = "DDQN" if is_ddqn else ("DQN" if is_dqn else "Q-Learn")
        mode_btn_label = f"Tab · AI: {ai_short}" if mode == "agent" else "Tab · Lái: Phím"

        main_buttons = [
            ("pause", pause_label, 36, 175, False),
            ("mode", mode_btn_label, 226, 210, False),
            ("menu_toggle", "≡ CÀI ĐẶT & KỊCH BẢN (M)", 451, 270, True),
            ("reset", "R · Chạy lại từ đầu", 736, 185, False),
        ]

        for key, label, x, width, is_highlight in main_buttons:
            rect = pygame.Rect(x, 580, width, 44)
            if not self.menu_open:
                self.buttons[key] = rect
            is_hover = (not self.menu_open) and rect.collidepoint(mouse_pos)
            bg_col = PANEL_HOVER if is_hover else (PANEL if not is_highlight else (25, 42, 58))
            border_col = MINT if (is_hover or is_highlight) else PANEL_BORDER
            pygame.draw.rect(self.screen, bg_col, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_col, rect, width=2 if is_highlight else 1, border_radius=8)
            text_col = MINT if (is_hover or is_highlight) else TEXT
            self.text(label, x + 14, 593, 14, text_col)

        # Khu vực thông báo nổi Toast thời gian thực
        toast_rect = pygame.Rect(936, 580, 308, 44)
        pygame.draw.rect(self.screen, PANEL, toast_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, toast_rect, width=1, border_radius=8)

        if pygame.time.get_ticks() < self.toast_expire and self.toast_text:
            pygame.draw.rect(self.screen, (24, 42, 54), toast_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.toast_color, toast_rect, width=1, border_radius=8)
            self.text(f"● {self.toast_text}", 946, 594, 12, self.toast_color)
        else:
            self.text("Bấm M hoặc nút ≡ để mở Cài đặt", 946, 594, 12, MUTED)

        # Chú thích phím tắt chân trang
        self.text(
            "Phím tắt: Bấm M mở Menu cài đặt  ·  Tab đổi AI/Phím  ·  Space tạm dừng  ·  R chạy lại từ đầu  ·  Esc đóng menu/thoát",
            38, 642, 13, MUTED
        )
        if mode == "keyboard":
            self.text(
                "Lái xe bằng bàn phím: ← / → chuyển làn  ·  ↑ tăng tốc  ·  ↓ giảm tốc/phanh  |  Bấm Tab để chuyển AI tự lái",
                38, 664, 13, MINT
            )
        else:
            self.text(
                "Phím tắt nhanh: L đổi số làn  ·  T đổi mật độ xe  ·  S đổi tốc độ  ·  F đổi dòng xe  ·  1..4 kịch bản mẫu",
                38, 664, 13, (120, 180, 240)
            )

        # ---------------------------------------------------------
        # 6. BẢNG CÀI ĐẶT & KỊCH BẢN DẠNG MENU MODAL (KHI MENU ĐANG MỞ)
        # ---------------------------------------------------------
        if self.menu_open:
            # Lớp phủ mờ nền tối (Dim Backdrop)
            dim_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            dim_surface.fill((8, 14, 22, 215))
            self.screen.blit(dim_surface, (0, 0))

            # Khung hộp thoại chính Modal
            modal_rect = self.modal_rect
            pygame.draw.rect(self.screen, MODAL_BG, modal_rect, border_radius=14)
            pygame.draw.rect(self.screen, MODAL_BORDER, modal_rect, width=2, border_radius=14)

            # --- Header Menu ---
            self.text("≡ BẢNG CÀI ĐẶT & KỊCH BẢN MÔ PHỎNG", modal_rect.x + 28, modal_rect.y + 20, 20, MINT)
            self.text("Thay đổi có hiệu lực ngay lập tức — Xe tiếp tục chạy không bị reset", modal_rect.x + 28, modal_rect.y + 46, 13, MUTED)

            # Nút đóng góc phải
            close_top_rect = pygame.Rect(modal_rect.right - 170, modal_rect.y + 18, 142, 34)
            self.buttons["menu_close"] = close_top_rect
            is_ct_hover = close_top_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, PANEL_HOVER if is_ct_hover else PANEL, close_top_rect, border_radius=6)
            pygame.draw.rect(self.screen, RED if is_ct_hover else PANEL_BORDER, close_top_rect, width=1, border_radius=6)
            self.text("× Đóng (M / Esc)", close_top_rect.x + 14, close_top_rect.y + 8, 13, RED if is_ct_hover else TEXT)

            pygame.draw.line(self.screen, PANEL_BORDER, (modal_rect.x + 28, modal_rect.y + 76), (modal_rect.right - 28, modal_rect.y + 76), 1)

            # --- Mục 1: Số Làn Đường (2..6 Làn) ---
            sec1_y = modal_rect.y + 88
            self.text("1. SỐ LÀN ĐƯỜNG CAO TỐC:", modal_rect.x + 28, sec1_y, 13, MINT)
            lane_opts = [(2, "2 Làn"), (3, "3 Làn (Chuẩn)"), (4, "4 Làn"), (5, "5 Làn"), (6, "6 Làn")]
            for idx, (ln, ln_lbl) in enumerate(lane_opts):
                bx = modal_rect.x + 28 + idx * 180
                b_rect = pygame.Rect(bx, sec1_y + 22, 170, 36)
                btn_key = f"menu_lane_{ln}"
                self.buttons[btn_key] = b_rect
                is_active = (lanes_count == ln)
                is_hover = b_rect.collidepoint(mouse_pos)
                bg = MINT_DIM if is_active else (PANEL_HOVER if is_hover else PANEL)
                bd = MINT if is_active else (MINT if is_hover else PANEL_BORDER)
                pygame.draw.rect(self.screen, bg, b_rect, border_radius=6)
                pygame.draw.rect(self.screen, bd, b_rect, width=2 if is_active else 1, border_radius=6)
                prefix = "● " if is_active else ""
                self.text(f"{prefix}{ln_lbl}", bx + 14, sec1_y + 31, 13, MINT if is_active else (TEXT if not is_hover else MINT))

            # --- Mục 2: Mật Độ Xe Giao Thông ---
            sec2_y = modal_rect.y + 160
            self.text("2. MẬT ĐỘ XE GIAO THÔNG XUNG QUANH:", modal_rect.x + 28, sec2_y, 13, MINT)
            dens_opts = [
                ("sparse", "Thưa thớt (8 xe)"),
                ("normal", "Bình thường (20 xe - Chuẩn)"),
                ("dense", "Dày đặc (38 xe)"),
                ("none", "Không có xe (0 xe)"),
            ]
            for idx, (dens_code, dens_lbl) in enumerate(dens_opts):
                bx = modal_rect.x + 28 + idx * 228
                b_rect = pygame.Rect(bx, sec2_y + 22, 218, 36)
                btn_key = f"menu_traffic_{dens_code}"
                self.buttons[btn_key] = b_rect
                is_active = (density == dens_code)
                is_hover = b_rect.collidepoint(mouse_pos)
                bg = MINT_DIM if is_active else (PANEL_HOVER if is_hover else PANEL)
                bd = MINT if is_active else (MINT if is_hover else PANEL_BORDER)
                pygame.draw.rect(self.screen, bg, b_rect, border_radius=6)
                pygame.draw.rect(self.screen, bd, b_rect, width=2 if is_active else 1, border_radius=6)
                prefix = "● " if is_active else ""
                self.text(f"{prefix}{dens_lbl}", bx + 12, sec2_y + 31, 13, MINT if is_active else (TEXT if not is_hover else MINT))

            # --- Mục 3: Vận Tốc Xe & Dòng Xe ---
            sec3_y = modal_rect.y + 232
            self.text("3. VẬN TỐC XE CỦA BẠN:", modal_rect.x + 28, sec3_y, 13, MINT)
            self.text("4. TỐC ĐỘ DÒNG XE:", modal_rect.x + 520, sec3_y, 13, MINT)

            # Tốc độ xe
            cur_speed = int(round(float(env.config.get("initial_speed", 20.0)) * 3.6))
            spd_opts = [(54, "54 km/h"), (72, "72 km/h (Chuẩn)"), (90, "90 km/h"), (108, "108 km/h")]
            for idx, (spd_val, spd_lbl) in enumerate(spd_opts):
                bx = modal_rect.x + 28 + idx * 118
                b_rect = pygame.Rect(bx, sec3_y + 22, 110, 36)
                btn_key = f"menu_spd_{spd_val}"
                self.buttons[btn_key] = b_rect
                is_active = abs(cur_speed - spd_val) <= 5
                is_hover = b_rect.collidepoint(mouse_pos)
                bg = MINT_DIM if is_active else (PANEL_HOVER if is_hover else PANEL)
                bd = MINT if is_active else (MINT if is_hover else PANEL_BORDER)
                pygame.draw.rect(self.screen, bg, b_rect, border_radius=6)
                pygame.draw.rect(self.screen, bd, b_rect, width=2 if is_active else 1, border_radius=6)
                self.text(spd_lbl, bx + 10, sec3_y + 31, 12, MINT if is_active else (TEXT if not is_hover else MINT))

            # Tốc độ dòng xe
            cur_flow_min = float(env.config.get("traffic_speed_min", 14.0))
            cur_flow = "fast" if cur_flow_min >= 20.0 else ("slow" if cur_flow_min <= 11.0 else "normal")
            flow_opts = [("slow", "Chậm (slow)"), ("normal", "Bình thường (normal)"), ("fast", "Nhanh (fast)")]
            for idx, (flw_code, flw_lbl) in enumerate(flow_opts):
                bx = modal_rect.x + 520 + idx * 128
                b_rect = pygame.Rect(bx, sec3_y + 22, 120, 36)
                btn_key = f"menu_flow_{flw_code}"
                self.buttons[btn_key] = b_rect
                is_active = (cur_flow == flw_code)
                is_hover = b_rect.collidepoint(mouse_pos)
                bg = MINT_DIM if is_active else (PANEL_HOVER if is_hover else PANEL)
                bd = MINT if is_active else (MINT if is_hover else PANEL_BORDER)
                pygame.draw.rect(self.screen, bg, b_rect, border_radius=6)
                pygame.draw.rect(self.screen, bd, b_rect, width=2 if is_active else 1, border_radius=6)
                self.text(flw_lbl, bx + 10, sec3_y + 31, 12, MINT if is_active else (TEXT if not is_hover else MINT))

            # --- Mục 4: Kịch Bản Mẫu Tích Hợp Sẵn (Presets) ---
            sec4_y = modal_rect.y + 304
            self.text("5. CHỌN NHANH KỊCH BẢN MẪU (PRESETS):", modal_rect.x + 28, sec4_y, 13, MINT)
            presets = [
                ("menu_preset_1", "Kịch bản 1: Cao tốc hẹp 2 làn · Mật độ thưa (8 xe) · Vận tốc 90 km/h", modal_rect.x + 28, sec4_y + 22, 442),
                ("menu_preset_2", "Kịch bản 2: Cao tốc chuẩn 3 làn · Mật độ bình thường (20 xe) · 72 km/h", modal_rect.x + 482, sec4_y + 22, 442),
                ("menu_preset_3", "Kịch bản 3: Cao tốc mở rộng 5 làn · Mật độ dày đặc (38 xe) · Dòng xe nhanh", modal_rect.x + 28, sec4_y + 68, 442),
                ("menu_preset_4", "Kịch bản 4: Cao tốc 4 làn đường vắng · Kiểm tra né chướng ngại vật hỏng", modal_rect.x + 482, sec4_y + 68, 442),
            ]
            for btn_key, desc, px, py, pw in presets:
                p_rect = pygame.Rect(px, py, pw, 38)
                self.buttons[btn_key] = p_rect
                is_hover = p_rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.screen, PANEL_HOVER if is_hover else PANEL, p_rect, border_radius=6)
                pygame.draw.rect(self.screen, MINT if is_hover else PANEL_BORDER, p_rect, width=1, border_radius=6)
                self.text(desc, px + 12, py + 11, 12, MINT if is_hover else TEXT)

            # --- Mục 5: Chế Độ Điều Khiển & Thuật Toán ---
            sec5_y = modal_rect.y + 424
            self.text("6. CHẾ ĐỘ ĐIỀU KHIỂN & THUẬT TOÁN:", modal_rect.x + 28, sec5_y, 13, MINT)
            mode_opts = [
                ("menu_mode_agent", "DQN (Deep Q-Network)", modal_rect.x + 28, sec5_y + 22, 294, mode == "agent" and not is_ddqn),
                ("menu_mode_ddqn", "Double Dueling DQN (Tuấn Anh)", modal_rect.x + 333, sec5_y + 22, 294, mode == "agent" and is_ddqn),
                ("menu_mode_keyboard", "Bàn phím (← → ↑ ↓)", modal_rect.x + 638, sec5_y + 22, 294, mode == "keyboard"),
            ]
            for btn_key, m_lbl, mx, my, mw, is_active in mode_opts:
                m_rect = pygame.Rect(mx, my, mw, 38)
                self.buttons[btn_key] = m_rect
                is_hover = m_rect.collidepoint(mouse_pos)
                bg = MINT_DIM if is_active else (PANEL_HOVER if is_hover else PANEL)
                bd = MINT if is_active else (MINT if is_hover else PANEL_BORDER)
                pygame.draw.rect(self.screen, bg, m_rect, border_radius=6)
                pygame.draw.rect(self.screen, bd, m_rect, width=2 if is_active else 1, border_radius=6)
                prefix = "● " if is_active else ""
                self.text(f"{prefix}{m_lbl}", mx + 14, my + 11, 13, MINT if is_active else (TEXT if not is_hover else MINT))

            # --- Nút lớn đóng menu ở dưới đáy modal ---
            foot_btn_rect = pygame.Rect(modal_rect.x + 28, modal_rect.bottom - 60, modal_rect.width - 56, 44)
            self.buttons["menu_close_btn"] = foot_btn_rect
            is_foot_hover = foot_btn_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (32, 60, 52) if is_foot_hover else (22, 42, 38), foot_btn_rect, border_radius=8)
            pygame.draw.rect(self.screen, MINT, foot_btn_rect, width=1, border_radius=8)
            self.text("×  ĐÓNG CÀI ĐẶT & TIẾP TỤC CHẠY (Bấm phím M hoặc Esc)", foot_btn_rect.x + 240, foot_btn_rect.y + 13, 15, MINT)

    def save(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(target))
