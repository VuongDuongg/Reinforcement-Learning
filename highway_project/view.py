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
        self.fonts = {size: pygame.font.SysFont("segoeui,arial", size) for size in (14, 16, 18, 22, 28, 36)}
        self.buttons: dict[str, pygame.Rect] = {}

    def text(self, text, x, y, size=18, color=TEXT):
        self.screen.blit(self.fonts[size].render(str(text), True, color), (x, y))

    def draw(self, env, info, *, paused, mode, last_action, total_reward, episode,
             motion=None, alpha=1.0, agent_label="AGENT CHỜ CÀI ĐẶT"):
        self.screen.fill(BG)
        self.text("RL / MÔ PHỎNG GIAO THÔNG", 38, 24, 16, MINT)
        self.text("Cao tốc 3 làn · Né chướng ngại vật", 36, 50, 36)
        self.text("HighwayEnv + Gymnasium  |  Q-learning trên CPU  |  Nội suy chuyển động 60 FPS", 38, 103, 16, MUTED)

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

        self.text("TRÁI / LÀN 0", 38, 265, 14, MUTED)
        self.text("Chiều di chuyển  →", 1065, 265, 16, MINT)
        road_bottom = self.ROAD_TOP + 3 * self.LANE_HEIGHT
        pygame.draw.rect(self.screen, (31, 55, 49), (0, 287, self.WIDTH, 243))
        pygame.draw.rect(self.screen, (49, 59, 72), (0, self.ROAD_TOP, self.WIDTH, 216))
        for y in (self.ROAD_TOP + 3, road_bottom - 3):
            pygame.draw.line(self.screen, (218, 192, 107), (0, y), (self.WIDTH, y), 3)
        def pose(obj):
            return motion.sample(obj, alpha) if motion else (obj.position, obj.heading)

        camera_x = float(pose(env.vehicle)[0][0])
        offset = int(camera_x * self.SCALE) % 66
        for row in (1, 2):
            y = self.ROAD_TOP + row * self.LANE_HEIGHT
            for x in range(-offset, self.WIDTH, 66):
                pygame.draw.line(self.screen, (176, 189, 204), (x, y), (x + 32, y), 2)

        def coords(obj):
            position, _ = pose(obj)
            return (self.EGO_X + (position[0] - camera_x) * self.SCALE,
                    self.ROAD_TOP + self.LANE_HEIGHT / 2 + position[1] / 4 * self.LANE_HEIGHT)

        # Mốc mặt đường thay đổi theo quãng đường thực của xe.
        for marker in range(int(camera_x // 50) * 50 - 50, int(camera_x + 300), 50):
            x = self.EGO_X + (marker - camera_x) * self.SCALE
            self.text(f"{marker - env.start_x:.0f} m", x, 538, 14, MUTED)
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
            pygame.draw.line(self.screen, MINT, (goal_x, self.ROAD_TOP), (goal_x, road_bottom), 4)
            self.text("ĐÍCH", goal_x + 8, self.ROAD_TOP + 8, 16, MINT)

        reason = info["end_reason"]
        status = {"collision": "VA CHẠM — nhấn R để thử lại", "success": "ĐẾN ĐÍCH AN TOÀN — nhấn N để đổi kịch bản",
                  "time_limit": "HẾT THỜI GIAN — nhấn R để thử lại", "off_road": "RA KHỎI ĐƯỜNG — nhấn R để thử lại"}.get(reason)
        if status:
            pygame.draw.rect(self.screen, BG, (335, 376, 790, 54), border_radius=10)
            self.text(status, 354, 388, 22, MINT if reason == "success" else RED)
        elif paused:
            pygame.draw.rect(self.screen, BG, (430, 376, 600, 54), border_radius=10)
            self.text("TẠM DỪNG — nhấn Space để xe chạy", 452, 388, 22)

        mode_text = "BÀN PHÍM" if mode == "keyboard" else agent_label
        self.text(f"Chế độ: {mode_text}", 38, 582, 18, MINT)
        self.text(f"Episode {episode}  ·  Còn {info['remaining_m']:.0f} m  ·  Reward {total_reward:.2f}", 38, 611, 16, MUTED)
        self.text(f"Lệnh gần nhất: {last_action}", 38, 638, 16, MUTED)
        labels = [("pause", "Chạy / Dừng", 36, 190), ("reset", "R · Chạy lại", 240, 175),
                  ("new", "N · Kịch bản mới", 429, 215), ("mode", "Tab · Đổi chế độ", 658, 225)]
        for key, label, x, width in labels:
            rect = pygame.Rect(x, 689, width, 46)
            self.buttons[key] = rect
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
            self.text(label, x + 14, 700, 18)
        self.text("← / → đổi làn    ↑ tăng tốc    ↓ giảm tốc    Space tạm dừng    Esc thoát", 38, 759, 16, MUTED)

    def save(self, path: str | Path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.screen, str(target))
