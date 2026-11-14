"""Nội suy hai trạng thái mô phỏng để vẽ ở 60 FPS, trễ tối đa một bước."""

import math

import numpy as np


class MotionBuffer:
    def reset(self, env):
        self.current = self.snapshot(env)
        self.previous = self.current

    @staticmethod
    def snapshot(env):
        return {id(obj): (obj.position.copy(), float(obj.heading))
                for obj in [*env.road.vehicles, *env.obstacles]}

    def capture(self, env):
        self.previous = self.current
        self.current = self.snapshot(env)

    def sample(self, obj, alpha):
        current = self.current.get(id(obj), (obj.position, obj.heading))
        previous = self.previous.get(id(obj), current)
        alpha = float(np.clip(alpha, 0, 1))
        heading_delta = (current[1] - previous[1] + math.pi) % (2 * math.pi) - math.pi
        return previous[0] * (1 - alpha) + current[0] * alpha, previous[1] + alpha * heading_delta
