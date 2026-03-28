"""
environment/state_processing.py
Converts raw AirSim sensor data into a normalised flat observation vector
that can be fed directly to the policy network.

Observation vector layout (total = 3 + 3 + 3 + N_sensors + 1 = 10 + N)
────────────────────────────────────────────────────────────────────────
  [0:3]       normalised relative position to target  (dx, dy, dz)
  [3:6]       normalised linear velocity              (vx, vy, vz)
  [6:9]       normalised orientation angles           (roll, pitch, yaw)
  [9:9+N]     normalised distance-sensor readings     (0-1 each)
  [9+N]       normalised distance to target           (scalar)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from config.env_config import EnvConfig, ENV_CONFIG


class StateProcessor:
    """
    Transforms raw AirSim MultirotorState + sensor readings into a
    normalised numpy observation vector.

    Parameters
    ----------
    cfg : EnvConfig
    """

    def __init__(self, cfg: EnvConfig = ENV_CONFIG):
        self.cfg = cfg
        self._target = np.array(cfg.target_position, dtype=np.float32)

        self.obs_dim = (
            3
            + 3
            + 3
            + cfg.num_distance_sensors
            + 1
        )


    def process(
        self,
        multirotor_state: Any,
        sensor_readings: List[float],
    ) -> np.ndarray:
        """
        Build the observation vector from AirSim state objects.

        Parameters
        ----------
        multirotor_state : airsim.MultirotorState
        sensor_readings  : list of floats, one per distance sensor

        Returns
        -------
        np.ndarray, shape (obs_dim,), dtype float32, values clipped to [-1, 1]
        """
        pos = self._extract_position(multirotor_state)
        vel = self._extract_velocity(multirotor_state)
        ori = self._extract_orientation(multirotor_state)
        sensors = self._normalise_sensors(sensor_readings)

        rel_pos = (self._target - pos) / max(
            math.sqrt(
                self.cfg.boundary_x ** 2
                + self.cfg.boundary_y ** 2
                + self.cfg.boundary_z ** 2
            ),
            1e-6,
        )
        dist = float(np.linalg.norm(rel_pos))

        obs = np.concatenate(
            [rel_pos, vel, ori, sensors, [dist]], dtype=np.float32
        )
        return np.clip(obs, -1.0, 1.0)

    def process_mock(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, -5.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sensor_readings: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Build an observation vector from raw tuples (useful when AirSim is not
        running, e.g., during unit testing or the mock environment).
        """
        if sensor_readings is None:
            sensor_readings = [self.cfg.sensor_max_range] * self.cfg.num_distance_sensors

        pos = np.array(position, dtype=np.float32)
        vel = np.array(velocity, dtype=np.float32) / self.cfg.max_velocity
        ori_rad = np.array(orientation, dtype=np.float32)
        ori_norm = ori_rad / math.pi
        sensors = self._normalise_sensors(sensor_readings)

        diag = math.sqrt(
            self.cfg.boundary_x ** 2
            + self.cfg.boundary_y ** 2
            + self.cfg.boundary_z ** 2
        )
        rel_pos = (self._target - pos) / max(diag, 1e-6)
        dist = float(np.linalg.norm(rel_pos))

        obs = np.concatenate(
            [rel_pos, vel, ori_norm, sensors, [dist]], dtype=np.float32
        )
        return np.clip(obs, -1.0, 1.0)


    @staticmethod
    def _extract_position(state: Any) -> np.ndarray:
        p = state.kinematics_estimated.position
        return np.array([p.x_val, p.y_val, p.z_val], dtype=np.float32)

    def _extract_velocity(self, state: Any) -> np.ndarray:
        v = state.kinematics_estimated.linear_velocity
        vel = np.array([v.x_val, v.y_val, v.z_val], dtype=np.float32)
        return np.clip(vel / self.cfg.max_velocity, -1.0, 1.0)

    @staticmethod
    def _extract_orientation(state: Any) -> np.ndarray:
        """Convert quaternion → Euler (roll, pitch, yaw) normalised to [-1,1]."""
        q = state.kinematics_estimated.orientation
        w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        euler = np.array([roll, pitch, yaw], dtype=np.float32)
        return euler / math.pi

    def _normalise_sensors(self, readings: List[float]) -> np.ndarray:
        arr = np.array(readings[: self.cfg.num_distance_sensors], dtype=np.float32)
        if len(arr) < self.cfg.num_distance_sensors:
            pad = np.full(
                self.cfg.num_distance_sensors - len(arr),
                self.cfg.sensor_max_range,
                dtype=np.float32,
            )
            arr = np.concatenate([arr, pad])
        return np.clip(arr / self.cfg.sensor_max_range, 0.0, 1.0)
