"""HighwayEnv với vật cản cố định, episode hữu hạn và thông tin đánh giá."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle

ENV_ID = "LaneObstacle-v0"
ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


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
            "vehicles_count": 12,
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "duration": 60.0,
            "simulation_frequency": 30,
            "policy_frequency": 10,
            "route_length": 650.0,
            "obstacle_count": 1,
            "obstacle_distance": 160.0,
            "obstacle_spacing": 180.0,
            "offroad_terminal": True,
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
            "speed_weight": 0.03,
            "danger_penalty": -0.15,
            "lane_change_penalty": -0.02,
            "success_reward": 10.0,
        })
        return config

    def _create_vehicles(self) -> None:
        if self.config["lanes_count"] != 3 or self.config["controlled_vehicles"] != 1:
            raise ValueError("Bản môi trường này hỗ trợ 3 làn và 1 xe điều khiển.")
        count = int(self.config["obstacle_count"])
        if not 0 <= count <= 3 or self.config["vehicles_count"] < 0:
            raise ValueError("obstacle_count phải từ 0 đến 3; vehicles_count phải >= 0.")
        last_obstacle = self.config["obstacle_distance"] + max(0, count - 1) * self.config["obstacle_spacing"]
        if count and (self.config["obstacle_distance"] < 60 or self.config["route_length"] <= last_obstacle + 20):
            raise ValueError("Cần vật cản cách điểm xuất phát >= 60 m và đích sau vật cản cuối > 20 m.")

        lane_id = int(self.config["initial_lane_id"])
        if lane_id not in range(3):
            raise ValueError("initial_lane_id phải là 0, 1 hoặc 2.")
        lane = self.road.network.get_lane(("0", "1", lane_id))
        ego = self.action_type.vehicle_class(self.road, lane.position(80.0, 0), heading=0, speed=20.0)
        ego.color = (54, 211, 179)
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)
        self.start_x = float(ego.position[0])
        self.goal_x = self.start_x + self.config["route_length"]
        self.obstacles: list[BrokenCar] = []
        for i in range(count):
            obstacle_lane = lane_id if i == 0 else int(self.np_random.integers(0, 3))
            obstacle = BrokenCar.make_on_lane(
                self.road, ("0", "1", obstacle_lane),
                self.start_x + self.config["obstacle_distance"] + i * self.config["obstacle_spacing"],
                speed=0.0,
            )
            self.obstacles.append(obstacle)
            self.road.objects.append(obstacle)

        # Không sinh xe đè lên ego, vật cản hoặc xe vừa tạo.
        attempts = 0
        while len(self.road.vehicles) - 1 < int(self.config["vehicles_count"]):
            attempts += 1
            if attempts > 5000:
                raise ValueError("Quá nhiều xe cho khoảng sinh hiện tại; giảm vehicles_count.")
            lane_id = int(self.np_random.integers(0, 3))
            x = float(self.np_random.uniform(0, self.goal_x + 100))
            occupied = [*self.road.vehicles, *self.obstacles]
            if any(obj.lane_index[2] == lane_id and abs(obj.position[0] - x) < 38 for obj in occupied):
                continue
            if abs(x - self.start_x) < 45:
                continue
            vehicle = IDMVehicle.make_on_lane(
                self.road, ("0", "1", lane_id), x,
                speed=float(self.np_random.uniform(16, 24)),
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _success(self) -> bool:
        return bool(self.vehicle.position[0] >= self.goal_x and not self.vehicle.crashed and self.vehicle.on_road)

    def _is_terminated(self) -> bool:
        return bool(super()._is_terminated() or self._success())

    def _rewards(self, action: int | None) -> dict[str, float]:
        terms = {"collision": 0.0, "offroad": 0.0, "speed": 0.0, "lane_change": 0.0, "success": 0.0, "danger": 0.0}
        if self.vehicle.crashed:
            terms["collision"] = float(self.config["collision_penalty"])
        elif not self.vehicle.on_road:
            terms["offroad"] = float(self.config["offroad_penalty"])
        else:
            terms["speed"] = self.config["speed_weight"] * max(0.0, float(self.vehicle.speed)) / 30.0
            if action in (0, 2):
                terms["lane_change"] = float(self.config["lane_change_penalty"])
            if self._success():
                terms["success"] = float(self.config["success_reward"])
            sensor = self._sensors()
            lane = sensor["lane"]
            gap = sensor["front_gap"][lane]
            closing = sensor["front_closing"][lane]
            if gap < 10 or (closing > 0.1 and gap / closing < 2.0):
                terms["danger"] = float(self.config["danger_penalty"])
        return terms

    def _reward(self, action: int) -> float:
        return float(sum(self._rewards(action).values()))

    def _sensors(self) -> dict:
        """Cảm biến 180 m theo ba làn, không cung cấp vị trí vật cản ngoài tầm."""
        ego = self.vehicle
        front, rear = [180.0] * 3, [180.0] * 3
        front_closing, rear_closing = [0.0] * 3, [0.0] * 3
        for obj in [*self.road.vehicles, *self.road.objects]:
            if obj is ego:
                continue
            dx = float(obj.position[0] - ego.position[0])
            if abs(dx) > 180:
                continue
            gap = max(0.0, abs(dx) - (obj.LENGTH + ego.LENGTH) / 2)
            for lane_id in range(3):
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
        return {"lane": lane_id, "speed": float(ego.speed), "changing": bool(changing),
                "front_gap": front, "rear_gap": rear,
                "front_closing": front_closing, "rear_closing": rear_closing}

    def _info(self, obs: np.ndarray, action: int | None = None) -> dict:
        info = super()._info(obs, action)
        success = self._success()
        info.update({
            "is_success": success,
            "distance_m": max(0.0, float(self.vehicle.position[0] - self.start_x)),
            "remaining_m": max(0.0, float(self.goal_x - self.vehicle.position[0])),
            "obstacles_passed": int(sum(self.vehicle.position[0] > obj.position[0] + 5 for obj in self.obstacles)),
            "obstacles_total": len(self.obstacles),
            "elapsed_s": float(self.time),
            "available_actions": self.action_type.get_available_actions(),
            "sensors": self._sensors(),
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
    env = gym.make(ENV_ID, config=config, render_mode=render_mode)
    env.action_space.seed(seed)
    return env
