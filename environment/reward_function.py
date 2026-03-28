from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from config.env_config import EnvConfig, ENV_CONFIG


@dataclass
class RewardInfo:
    total: float = 0.0
    goal: float = 0.0
    collision: float = 0.0
    boundary: float = 0.0
    progress: float = 0.0
    step: float = 0.0
    altitude: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "reward/total": self.total,
            "reward/goal": self.goal,
            "reward/collision": self.collision,
            "reward/boundary": self.boundary,
            "reward/progress": self.progress,
            "reward/step": self.step,
            "reward/altitude": self.altitude,
        }


class RewardFunction:
    def __init__(self, cfg: EnvConfig = ENV_CONFIG):
        self.cfg = cfg
        self._target = cfg.target_position

    def compute(
        self,
        position: tuple[float, float, float],
        prev_position: tuple[float, float, float],
        has_collided: bool,
        out_of_bounds: bool,
        goal_reached: bool,
    ) -> tuple[float, RewardInfo]:
        info = RewardInfo()

        if goal_reached:
            info.goal = self.cfg.reward_goal_reached
            info.total = info.goal
            return info.total, info

        if has_collided:
            info.collision = self.cfg.reward_collision
            info.total = info.collision
            return info.total, info

        if out_of_bounds:
            info.boundary = self.cfg.reward_boundary_violation

        prev_dist = self._dist_to_target(prev_position)
        curr_dist = self._dist_to_target(position)
        delta = prev_dist - curr_dist
        info.progress = self.cfg.reward_progress_scale * delta

        info.step = self.cfg.reward_step_penalty

        altitude = -position[2]
        info.altitude = self._altitude_penalty(altitude)

        info.total = (
            info.goal
            + info.collision
            + info.boundary
            + info.progress
            + info.step
            + info.altitude
        )
        return info.total, info


    def _dist_to_target(self, position: tuple[float, float, float]) -> float:
        tx, ty, tz = self._target
        px, py, pz = position
        return math.sqrt((px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2)

    def _altitude_penalty(self, altitude: float) -> float:
        """
        Returns a negative penalty when the drone flies outside [2, 25] metres.
        Linear ramp with the configured scale coefficient.
        """
        lo, hi = 2.0, 25.0
        if altitude < lo:
            excess = lo - altitude
        elif altitude > hi:
            excess = altitude - hi
        else:
            return 0.0
        return self.cfg.reward_altitude_penalty_scale * excess
