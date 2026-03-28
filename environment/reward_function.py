"""
environment/reward_function.py
Computes the scalar reward for each environment step.

Reward components
─────────────────
  +  goal_reached      : large positive when drone reaches target
  -  collision         : large negative on obstacle / ground hit
  -  boundary          : negative if drone leaves the navigable box
  +  progress          : proportional to reduction in distance to goal
  -  step_penalty      : small constant cost per step (encourages speed)
  -  altitude_penalty  : penalises flying too high or too low
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from config.env_config import EnvConfig, ENV_CONFIG


@dataclass
class RewardInfo:
    """Breakdown of each reward component (useful for logging)."""
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
    """
    Computes per-step reward given the current and previous environment state.

    Parameters
    ----------
    cfg : EnvConfig
        Environment configuration that holds all reward coefficients.
    """

    def __init__(self, cfg: EnvConfig = ENV_CONFIG):
        self.cfg = cfg
        self._target = cfg.target_position  # (x, y, z)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        position: tuple[float, float, float],
        prev_position: tuple[float, float, float],
        has_collided: bool,
        out_of_bounds: bool,
        goal_reached: bool,
    ) -> tuple[float, RewardInfo]:
        """
        Compute the scalar reward and its breakdown.

        Parameters
        ----------
        position       : Current (x, y, z) in NED metres.
        prev_position  : Previous (x, y, z) in NED metres.
        has_collided   : True if a collision was detected this step.
        out_of_bounds  : True if drone left the navigable bounding box.
        goal_reached   : True if distance to target < goal_tolerance.

        Returns
        -------
        (total_reward, RewardInfo)
        """
        info = RewardInfo()

        # 1. Goal reached (terminal positive)
        if goal_reached:
            info.goal = self.cfg.reward_goal_reached
            info.total = info.goal
            return info.total, info

        # 2. Collision (terminal negative)
        if has_collided:
            info.collision = self.cfg.reward_collision
            info.total = info.collision
            return info.total, info

        # 3. Boundary violation (non-terminal negative)
        if out_of_bounds:
            info.boundary = self.cfg.reward_boundary_violation

        # 4. Progress reward  (Δ distance × scale)
        prev_dist = self._dist_to_target(prev_position)
        curr_dist = self._dist_to_target(position)
        delta = prev_dist - curr_dist          # positive → drone moved closer
        info.progress = self.cfg.reward_progress_scale * delta

        # 5. Step penalty
        info.step = self.cfg.reward_step_penalty

        # 6. Altitude penalty  (z is negative-up in NED, so altitude = -z)
        altitude = -position[2]
        info.altitude = self._altitude_penalty(altitude)

        # Sum
        info.total = (
            info.goal
            + info.collision
            + info.boundary
            + info.progress
            + info.step
            + info.altitude
        )
        return info.total, info

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

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
