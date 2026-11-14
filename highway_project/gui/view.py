"""Giao diện Pygame; chỉ vẽ trạng thái, vật lý do HighwayEnv xử lý."""

from __future__ import annotations

import math
from pathlib import Path

import pygame

BG = (14, 22, 33)
PANEL = (23, 35, 49)
TEXT = (231, 238, 245)
MUTED = (146, 165, 183)
MINT = (54, 211, 179)
RED = (247, 107, 107)


class HighwayView:
    WIDTH, HEIGHT = 1280, 800
    ROAD_TOP, LANE_HEIGHT, SCALE, EGO_X = 294, 72, 4.6, 260

    def __init__(self, *, headless: bool = False):
        pygame.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT)) if headless else pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("RL | Môi trường xe né vật cản")
        self.fonts = {size: pygame.font.SysFont("segoeui,arial", size) for size in (12, 14, 15, 16, 18, 22, 28, 36)}
        self.buttons: dict[str, pygame.Rect] = {}

    def text(self, text, x, y, size=18, color=TEXT):
        self.screen.blit(self.fonts[size].render(str(text), True, color), (x, y))

    def draw(self, env, info, *, paused, mode, last_action, total_reward, episode,
             motion=None, alpha=1.0, agent_label="AGENT CHỜ CÀI ĐẶT"):
        self.screen.fill(BG)
        lanes_count = int(env.config.get("lanes_count", 3))
        if lanes_count <= 3:
            road_top = self.ROAD_TOP
            lane_height = self.LANE_HEIGHT
        elif lanes_count == 4:
            road_top = 278
            lane_height = 64
        elif lanes_count == 5:
            road_top = 268
            lane_height = 54
        else:  # 6
            road_top = 260
            lane_height = 46

        road_height = lanes_count * lane_height
        road_bottom = road_top + road_height

        self.text("RL / MÔ PHỎNG GIAO THÔNG", 38, 24, 16, MINT)
        self.text(f"Cao tốc {lanes_count} làn · Né chướng ngại vật", 36, 50, 36)
        algo_title = "Deep Q-Network (PyTorch)" if "DQN" in str(agent_label) else "Q-learning trên CPU"
        density = info.get("sensors", {}).get("traffic_density", "normal")
        density_vi = {
            "sparse": "Thưa thớt",
            "normal": "Bình thường",
            "dense": "Dày đặc",
            "none": "Không có xe"
        }.get(density, "Bình thường")
        self.text(
            f"HighwayEnv + Gymnasium  |  {algo_title}  |  Mật độ: {density_vi} ({env.config.get('vehicles_count', 0)} xe)",
            38, 103, 16, MUTED
        )

        for index, (label, value, unit) in enumerate([
            ("VẬN TỐC", f"{env.vehicle.speed * 3.6:.0f}", "km/h"),
            ("ĐÃ ĐI", f"{info['distance_m']:.0f}", "m"),
            ("VƯỢT VẬT CẢN", f"{info['obstacles_passed']}/{info['obstacles_total']}", ""),
            ("THỜI GIAN", f"{info['elapsed_s']:.1f}", "s"),
        ]):
            x = 36 + index * 308
            pygame.draw.rect(self.screen, PANEL, (x, 150, 292, 102), border_radius=12)
            self.text(label, x + 18, 163, 14, MUTED)
            self.text(value, x + 18, 186, 36, MINT if index == 0 else TEXT)
            self.text(unit, x + 200, 207, 16, MUTED)

        self.text("TRÁI / LÀN 0", 38, road_top - 24, 14, MUTED)
        self.text("Chiều di chuyển  →", 1065, road_top - 24, 16, MINT)
        pygame.draw.rect(self.screen, (31, 55, 49), (0, road_top - 7, self.WIDTH, road_height + 14))
        pygame.draw.rect(self.screen, (49, 59, 72), (0, road_top, self.WIDTH, road_height))
        for y in (road_top + 3, road_bottom - 3):
            pygame.draw.line(self.screen, (218, 192, 107), (0, y), (self.WIDTH, y), 3)

        def pose(obj):
            return motion.sample(obj, alpha) if motion else (obj.position, obj.heading)

        camera_x = float(pose(env.vehicle)[0][0])
        offset = int(camera_x * self.SCALE) % 66
        for row in range(1, lanes_count):
            y = road_top + row * lane_height
            for x in range(-offset, self.WIDTH, 66):
                pygame.draw.line(self.screen, (176, 189, 204), (x, y), (x + 32, y), 2)

        self.text(f"PHẢI / LÀN {lanes_count - 1}", 38, road_bottom + 6, 14, MUTED)

        def coords(obj):
            position, _ = pose(obj)
            return (self.EGO_X + (position[0] - camera_x) * self.SCALE,
                    road_top + lane_height / 2 + position[1] / 4 * lane_height)

        # Mốc mặt đường thay đổi theo quãng đường thực của xe.
        marker_y = road_bottom + 24
        for marker in range(int(camera_x // 50) * 50 - 50, int(camera_x + 300), 50):
            x = self.EGO_X + (marker - camera_x) * self.SCALE
            self.text(f"{marker - env.start_x:.0f} m", x, marker_y, 14, MUTED)
        for obj in [*env.road.vehicles, *env.obstacles]:
            x, y = coords(obj)
            if x < -60 or x > self.WIDTH + 60:
                continue
            obstacle = obj in env.obstacles
            ego = obj is env.vehicle
            color = RED if obstacle or obj.crashed else MINT if ego else (117, 151, 201)
            # Kích thước xe phóng đại ở giao diện để dễ nhìn, va chạm vẫn theo mét trong engine.
            car = pygame.Surface((46, 30), pygame.SRCALPHA)
            pygame.draw.rect(car, color, (2, 2, 42, 26), border_radius=7)
            pygame.draw.rect(car, (25, 42, 55), (29, 6, 7, 18), border_radius=2)
            pygame.draw.rect(car, (25, 42, 55), (8, 6, 6, 18), border_radius=2)
            if obstacle:
                pygame.draw.line(car, (255, 235, 205), (17, 9), (26, 21), 2)
                pygame.draw.line(car, (255, 235, 205), (26, 9), (17, 21), 2)
            rotated = pygame.transform.rotate(car, -math.degrees(pose(obj)[1]))
            self.screen.blit(rotated, rotated.get_rect(center=(x, y)))
            if ego or obstacle:
                self.text("XE CỦA BẠN" if ego else "XE HỎNG", x - 42, y - 37, 14, color)
        goal_x = self.EGO_X + (env.goal_x - camera_x) * self.SCALE
        if 0 < goal_x < self.WIDTH:
            pygame.draw.line(self.screen, MINT, (goal_x, road_top), (goal_x, road_bottom), 4)
            self.text("ĐÍCH", goal_x + 8, road_top + 8, 16, MINT)

        reason = info["end_reason"]
        status = {"collision": "VA CHẠM — nhấn R để thử lại", "success": "ĐẾN ĐÍCH AN TOÀN — nhấn N để đổi kịch bản",
                  "time_limit": "HẾT THỜI GIAN — nhấn R để thử lại", "off_road": "RA KHỎI ĐƯỜNG — nhấn R để thử lại"}.get(reason)
        popup_y = road_top + (road_height - 54) // 2
        if status:
            pygame.draw.rect(self.screen, BG, (335, popup_y, 790, 54), border_radius=10)
            self.text(status, 354, popup_y + 12, 22, MINT if reason == "success" else RED)
        elif paused:
            pygame.draw.rect(self.screen, BG, (430, popup_y, 600, 54), border_radius=10)
            self.text("TẠM DỪNG — nhấn Space để xe chạy", 452, popup_y + 12, 22)

        mode_text = "BÀN PHÍM" if mode == "keyboard" else agent_label
        hud_info_y = max(570, min(585, road_bottom + 32))
        self.text(f"Chế độ: {mode_text}", 38, hud_info_y, 18, MINT)
        self.text(f"Episode {episode}  ·  Còn {info['remaining_m']:.0f} m  ·  Reward {total_reward:.2f}", 38, hud_info_y + 26, 16, MUTED)
        self.text(f"Lệnh gần nhất: {last_action}", 38, hud_info_y + 52, 16, MUTED)

        # Nhãn mật độ giao thông hiển thị gọn
        density_label_short = {
            "sparse": "Thưa",
            "normal": "Vừa",
            "dense": "Dày",
            "none": "Không",
        }.get(density, "Vừa")

        pause_label = "Space · Tiếp tục" if paused else "Space · Dừng"
        mode_label = "Tab · DQN (AI)" if mode == "agent" else "Tab · Tự lái"
        lanes_label = f"L · {lanes_count} Làn"
        traffic_label = f"T · Xe: {density_label_short}"

        current_cfg_speed = int(round(env.config.get("initial_speed", 20.0) * 3.6))
        speed_label = f"S · {current_cfg_speed} km/h"

        labels = [
            ("pause", pause_label, 36, 150),
            ("reset", "R · Chạy lại", 202, 140),
            ("new", "N · Hạt giống", 358, 150),
            ("mode", mode_label, 524, 170),
            ("lanes", lanes_label, 710, 145),
            ("traffic", traffic_label, 871, 165),
            ("speed", speed_label, 1052, 180),
        ]
        for key, label, x, width in labels:
            rect = pygame.Rect(x, 689, width, 46)
            self.buttons[key] = rect
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
            pygame.draw.rect(self.screen, (45, 62, 80), rect, width=1, border_radius=8)
            self.text(label, x + 12, 701, 16)

        if mode == "keyboard":
            self.text("Lái xe: ← / → đổi làn  ·  ↑ tăng tốc  ·  ↓ giảm tốc  |  Phím tắt: L đổi làn  ·  T mật độ  ·  S tốc độ", 38, 757, 15, MINT)
        else:
            self.text("Phím tắt: L đổi số làn (2..6)  ·  T đổi mật độ  ·  S đổi tốc độ  ·  1-4 kịch bản mẫu  ·  Tab đổi AI/Phím  ·  Space dừng", 38, 757, 15, MUTED)

    def save(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(target))
