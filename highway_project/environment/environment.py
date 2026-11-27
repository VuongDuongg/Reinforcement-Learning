
"""HighwayEnv với vật cản cố định, episode hữu hạn và thông tin đánh giá."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle

ENV_ID = "LaneObstacle-v0"
ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

TRAFFIC_PRESETS: dict[str, int] = {
    "none": 0, "0": 0, "khong": 0,
    "sparse": 8, "thua": 8, "thua_thot": 8,
    "normal": 20, "vua": 20, "binh_thuong": 20,
    "dense": 38, "day": 38, "day_dac": 38,
}


class BrokenCar(Obstacle):
    """RoadObject đứng yên: không có bộ điều khiển tự tăng tốc."""

    LENGTH = 5.0
    WIDTH = 2.0


class LaneObstacleEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "lanes_count": 3,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "initial_speed": 20.0,
            "max_speed": 30.0,
            "traffic_speed_min": 14.0,
            "traffic_speed_max": 28.0,
            "duration": 90.0,
            "simulation_frequency": 30,
            "policy_frequency": 10,
            "route_length": 1000.0,
            "obstacle_count": 1,
            "obstacle_distance": 180.0,
            "obstacle_spacing": 280.0,
            "random_obstacles": False,
            "offroad_terminal": True,
            "unpredictable_traffic": True,
            "vehicle_spacing": 28.0,
            # MDPVehicle nội suy chỉ số tốc độ theo lưới đều.
            "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20, 25, 30]},
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "normalize": True,
                "clip": True,
                "see_behind": True,
                "include_obstacles": True,
                "order": "sorted",
            },
            "collision_penalty": -20.0,
            "offroad_penalty": -10.0,
            "speed_weight": 0.15,
            "low_speed_penalty": -0.25,
            "timeout_penalty": -10.0,
            "danger_penalty": -0.15,
            "lane_change_penalty": -0.02,
            "success_reward": 10.0,
        })
        return config

    def _create_vehicles(self) -> None:
        lanes_count = int(self.config.get("lanes_count", 3))
        if not (2 <= lanes_count <= 6) or self.config.get("controlled_vehicles", 1) != 1:
            raise ValueError("Bản môi trường này hỗ trợ từ 2 đến 6 làn và 1 xe điều khiển.")

        raw_count = self.config.get("vehicles_count", 20)
        if isinstance(raw_count, str):
            key = raw_count.strip().lower()
            if key in TRAFFIC_PRESETS:
                vcount = TRAFFIC_PRESETS[key]
            else:
                try:
                    vcount = int(key)
                except ValueError:
                    raise ValueError(
                        f"vehicles_count '{raw_count}' không hợp lệ. Chọn 'sparse', 'normal', 'dense', 'none' hoặc số nguyên 0..60."
                    )
        else:
            vcount = int(raw_count)

        if not 0 <= vcount <= 60:
            raise ValueError(f"vehicles_count phải từ 0 đến 60, nhận được: {vcount}")

        self.config["vehicles_count"] = vcount

        if vcount == 0:
            self.traffic_density = "none"
        elif vcount <= 10:
            self.traffic_density = "sparse"
        elif vcount >= 30:
            self.traffic_density = "dense"
        else:
            self.traffic_density = "normal"

        count = int(self.config.get("obstacle_count", 1))
        if not 0 <= count <= 3:
            raise ValueError("obstacle_count phải từ 0 đến 3.")
        last_obstacle = self.config["obstacle_distance"] + max(0, count - 1) * self.config["obstacle_spacing"]
        if count and (self.config["obstacle_distance"] < 60 or self.config["route_length"] <= last_obstacle + 20):
            raise ValueError("Cần vật cản cách điểm xuất phát >= 60 m và đích sau vật cản cuối > 20 m.")

        lane_id = int(self.config.get("initial_lane_id", 1))
        if lane_id not in range(lanes_count):
            raise ValueError(f"initial_lane_id phải nằm trong khoảng 0 đến {lanes_count - 1}.")
        lane = self.road.network.get_lane(("0", "1", lane_id))
        init_speed = float(self.config.get("initial_speed", 20.0))
        ego = self.action_type.vehicle_class(self.road, lane.position(80.0, 0), heading=0, speed=init_speed)
        ego.target_speed = init_speed
        ego.color = (54, 211, 179)
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)
        self.start_x = float(ego.position[0])
        self.goal_x = self.start_x + self.config["route_length"]
        self.obstacles: list[BrokenCar] = []
        if count > 0:
            random_obstacles = bool(self.config.get("random_obstacles", False))
            if random_obstacles:
                route_len = float(self.config["route_length"])
                min_start = min(130.0, max(60.0, route_len * 0.2))
                safe_goal = min(70.0, max(30.0, route_len * 0.1))
                span = max(40.0, route_len - min_start - safe_goal)
                sector_len = span / count
                for i in range(count):
                    obstacle_lane = int(self.np_random.integers(0, lanes_count))
                    margin = min(15.0, sector_len * 0.1)
                    x_min = self.start_x + min_start + i * sector_len + margin
                    x_max = self.start_x + min_start + (i + 1) * sector_len - margin
                    x_pos = float(self.np_random.uniform(x_min, max(x_min + 1.0, x_max)))
                    obstacle = BrokenCar.make_on_lane(
                        self.road, ("0", "1", obstacle_lane),
                        x_pos,
                        speed=0.0,
                    )
                    self.obstacles.append(obstacle)
                    self.road.objects.append(obstacle)
            else:
                for i in range(count):
                    obstacle_lane = lane_id if i == 0 else int(self.np_random.integers(0, lanes_count))
                    obstacle = BrokenCar.make_on_lane(
                        self.road, ("0", "1", obstacle_lane),
                        self.start_x + self.config["obstacle_distance"] + i * self.config["obstacle_spacing"],
                        speed=0.0,
                    )
                    self.obstacles.append(obstacle)
                    self.road.objects.append(obstacle)

        # Không sinh xe đè lên ego, vật cản hoặc xe vừa tạo.
        attempts = 0
        min_spacing = float(self.config.get("vehicle_spacing", 28.0))
        unpredictable = bool(self.config.get("unpredictable_traffic", True))
        spd_min = float(self.config.get("traffic_speed_min", 14.0))
        spd_max = float(self.config.get("traffic_speed_max", 28.0))
        while len(self.road.vehicles) - 1 < int(self.config["vehicles_count"]):
            attempts += 1
            if attempts > 5000:
                raise ValueError("Quá nhiều xe cho khoảng sinh hiện tại; giảm vehicles_count.")
            traf_lane_id = int(self.np_random.integers(0, lanes_count))
            x = float(self.np_random.uniform(0, self.goal_x + 100))
            occupied = [*self.road.vehicles, *self.obstacles]
            if any(obj.lane_index[2] == traf_lane_id and abs(obj.position[0] - x) < min_spacing for obj in occupied):
                continue
            if abs(x - self.start_x) < 40:
                continue

            if unpredictable:
                speed = float(self.np_random.uniform(spd_min, spd_max))
                vehicle = IDMVehicle.make_on_lane(
                    self.road, ("0", "1", traf_lane_id), x,
                    speed=speed,
                )
                vehicle.target_speed = float(self.np_random.uniform(spd_min, max(spd_min + 1.0, spd_max)))
                vehicle.randomize_behavior()
                vehicle.POLITENESS = float(self.np_random.uniform(0.0, 0.35))
                vehicle.LANE_CHANGE_MIN_ACC_GAIN = float(self.np_random.uniform(0.1, 0.35))
                vehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = float(self.np_random.uniform(1.8, 3.8))
                vehicle.enable_lane_change = True
            else:
                vehicle = IDMVehicle.make_on_lane(
                    self.road, ("0", "1", traf_lane_id), x,
                    speed=float(self.np_random.uniform(spd_min, spd_max)),
                )
                vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _success(self) -> bool:
        return bool(self.vehicle.position[0] >= self.goal_x and not self.vehicle.crashed and self.vehicle.on_road)

    def _is_terminated(self) -> bool:
        return bool(super()._is_terminated() or self._success())

    def _rewards(self, action: int | None) -> dict[str, float]:
        terms = {
            "collision": 0.0,
            "offroad": 0.0,
            "speed": 0.0,
            "low_speed": 0.0,
            "lane_change": 0.0,
            "success": 0.0,
            "timeout": 0.0,
            "danger": 0.0,
        }
        if self.vehicle.crashed:
            terms["collision"] = float(self.config["collision_penalty"])
        elif not self.vehicle.on_road:
            terms["offroad"] = float(self.config["offroad_penalty"])
        else:
            speed = max(0.0, float(self.vehicle.speed))
            terms["speed"] = float(self.config["speed_weight"]) * speed / 30.0
            if speed < 10.0:
                terms["low_speed"] = float(self.config.get("low_speed_penalty", -0.25)) * (1.0 - speed / 10.0)
            if action in (0, 2):
                terms["lane_change"] = float(self.config["lane_change_penalty"])
            if self._success():
                terms["success"] = float(self.config["success_reward"])
            elif self._is_truncated():
                terms["timeout"] = float(self.config.get("timeout_penalty", -10.0))
            sensor = self._sensors()
            lane = sensor["lane"]
            if 0 <= lane < len(sensor["front_gap"]):
                gap = sensor["front_gap"][lane]
                closing = sensor["front_closing"][lane]
                if gap < 10 or (closing > 0.1 and gap / closing < 2.0):
                    terms["danger"] = float(self.config["danger_penalty"])
        return terms

    def _reward(self, action: int) -> float:
        return float(sum(self._rewards(action).values()))

    def _sensors(self) -> dict:
        """Cảm biến 180 m theo các làn, không cung cấp vị trí vật cản ngoài tầm."""
        ego = self.vehicle
        lanes_count = int(self.config.get("lanes_count", 3))
        front = [180.0] * lanes_count
        rear = [180.0] * lanes_count
        front_closing = [0.0] * lanes_count
        rear_closing = [0.0] * lanes_count
        for obj in [*self.road.vehicles, *self.road.objects]:
            if obj is ego:
                continue
            dx = float(obj.position[0] - ego.position[0])
            if abs(dx) > 180:
                continue
            gap = max(0.0, abs(dx) - (obj.LENGTH + ego.LENGTH) / 2)
            for lane_id in range(lanes_count):
                # Xe chuyển làn có thể chiếm cả hai làn.
                if abs(float(obj.position[1]) - lane_id * 4) > 2.7:
                    continue
                if dx >= 0 and gap < front[lane_id]:
                    front[lane_id] = gap
                    front_closing[lane_id] = float(ego.speed - obj.speed)
                elif dx < 0 and gap < rear[lane_id]:
                    rear[lane_id] = gap
                    rear_closing[lane_id] = float(obj.speed - ego.speed)
        lane_id = int(ego.lane_index[2])
        target_lane = int(ego.target_lane_index[2])
        changing = target_lane != lane_id or abs(float(ego.position[1]) - target_lane * 4) > 0.25
        ego_front_gap = front[lane_id] if 0 <= lane_id < lanes_count else 180.0
        left_clear = (lane_id > 0) and (front[lane_id - 1] > 20.0) and (rear[lane_id - 1] > 15.0)
        right_clear = (lane_id < lanes_count - 1) and (front[lane_id + 1] > 20.0) and (rear[lane_id + 1] > 15.0)
        density = getattr(self, "traffic_density", "normal")
        return {
            "lane": lane_id,
            "lanes_count": lanes_count,
            "traffic_density": density,
            "speed": float(ego.speed),
            "speed_kmh": float(ego.speed * 3.6),
            "changing": bool(changing),
            "target_lane": target_lane,
            "front_gap": front,
            "rear_gap": rear,
            "front_closing": front_closing,
            "rear_closing": rear_closing,
            "ego_front_gap": float(ego_front_gap),
            "left_clear": bool(left_clear),
            "right_clear": bool(right_clear),
        }

    def _info(self, obs: np.ndarray, action: int | None = None) -> dict:
        info = super()._info(obs, action)
        success = self._success()
        sensors = self._sensors()
        info.update({
            "is_success": success,
            "distance_m": max(0.0, float(self.vehicle.position[0] - self.start_x)),
            "remaining_m": max(0.0, float(self.goal_x - self.vehicle.position[0])),
            "obstacles_passed": int(sum(self.vehicle.position[0] > obj.position[0] + 5 for obj in self.obstacles)),
            "obstacles_total": len(self.obstacles),
            "elapsed_s": float(self.time),
            "speed": float(self.vehicle.speed),
            "speed_kmh": float(self.vehicle.speed * 3.6),
            "available_actions": self.action_type.get_available_actions(),
            "sensors": sensors,
            "end_reason": (
                "collision" if self.vehicle.crashed else
                "off_road" if not self.vehicle.on_road else
                "success" if success else
                "time_limit" if self._is_truncated() else "running"
            ),
        })
        return info


if ENV_ID not in gym.registry:
    gym.register(ENV_ID, entry_point=LaneObstacleEnv)


def make_env(*, seed: int = 42, config: dict | None = None, render_mode: str | None = None):
    """Trả về env Gymnasium; dùng reset/step chuẩn khi nối thuật toán."""
    env_config = config.copy() if config else {}
    if "max_speed" in env_config and "action" not in env_config:
        max_s = float(env_config["max_speed"])
        speeds = [round(float(s), 1) for s in np.linspace(0, max_s, 7)]
        env_config["action"] = {"type": "DiscreteMetaAction", "target_speeds": speeds}
    env = gym.make(ENV_ID, config=env_config if env_config else None, render_mode=render_mode)
    env.action_space.seed(seed)
    return env
