from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config.env_config import EnvConfig, ENV_CONFIG
from environment.state_processing import StateProcessor
from environment.reward_function import RewardFunction, RewardInfo

class _MockVector3r:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_val, self.y_val, self.z_val = x, y, z

class _MockQuaternion:
    def __init__(self):
        self.w_val, self.x_val, self.y_val, self.z_val = 1.0, 0.0, 0.0, 0.0

class _MockKinematics:
    def __init__(self, pos=(0., 0., -5.)):
        self.position   = _MockVector3r(*pos)
        self.linear_velocity = _MockVector3r()
        self.orientation = _MockQuaternion()

class _MockState:
    def __init__(self, pos=(0., 0., -5.)):
        self.kinematics_estimated = _MockKinematics(pos)
        self.collision_info = type("CI", (), {"has_collided": False})()

class _LocalDroneClient:

    def __init__(self, cfg: EnvConfig):
        self._cfg = cfg
        self._pos  = list(cfg.spawn_position)
        self._vel  = [0.0, 0.0, 0.0]
        self._collided = False

    def confirmConnection(self):          pass
    def enableApiControl(self, *a, **k):  pass
    def armDisarm(self, *a, **k):         pass
    def takeoffAsync(self):               return self
    def join(self):                       return self

    def getMultirotorState(self):
        return _MockState(tuple(self._pos))

    def getDistanceSensorData(self, name, vehicle=""):
        return type("DS", (), {"distance": float(np.random.uniform(3, 20))})()

    def moveByVelocityAsync(self, vx, vy, vz, duration, **kwargs):
        dt = duration
        self._vel = [vx, vy, vz]
        self._pos[0] += vx * dt
        self._pos[1] += vy * dt
        self._pos[2] += vz * dt
        if self._pos[2] > -1.0:
            self._collided = True
        return self

    def simGetCollisionInfo(self):
        info = type("CI", (), {"has_collided": self._collided})()
        self._collided = False
        return info

    def reset(self):
        self._pos   = list(self._cfg.spawn_position)
        self._vel   = [0.0, 0.0, 0.0]
        self._collided = False

    def ping(self):
        return True

class DroneNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        cfg: EnvConfig = ENV_CONFIG,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cfg         = cfg
        self.render_mode = render_mode or cfg.render_mode

        self._processor  = StateProcessor(cfg)
        self._reward_fn  = RewardFunction(cfg)

        obs_dim = self._processor.obs_dim
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._client = self._connect()

        self._step_count  = 0
        self._prev_pos    = cfg.spawn_position
        self._episode_reward = 0.0
        self._reward_log: List[RewardInfo] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._client.reset()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self._client.takeoffAsync().join()

        self._step_count     = 0
        self._episode_reward = 0.0
        self._reward_log     = []
        self._prev_pos       = self.cfg.spawn_position

        obs = self._get_obs()
        info: Dict[str, Any] = {"episode_step": 0}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        vx, vy, vz = (action * self.cfg.max_velocity).tolist()
        self._client.moveByVelocityAsync(
            vx, vy, vz,
            duration=self.cfg.action_duration_sec,
        ).join()
        time.sleep(max(0.0, self.cfg.step_duration_sec - self.cfg.action_duration_sec))

        state       = self._client.getMultirotorState()
        col_info    = self._client.simGetCollisionInfo()
        curr_pos    = self._get_position(state)
        sensors     = self._read_sensors()

        collided    = col_info.has_collided
        oob         = self._is_out_of_bounds(curr_pos)
        goal        = self._dist(curr_pos, self.cfg.target_position) < self.cfg.goal_tolerance

        reward, r_info = self._reward_fn.compute(
            curr_pos, self._prev_pos, collided, oob, goal
        )
        self._prev_pos       = curr_pos
        self._step_count    += 1
        self._episode_reward += reward
        self._reward_log.append(r_info)

        terminated = collided or goal
        truncated  = self._step_count >= self.cfg.max_steps_per_episode

        obs  = self._processor.process_mock(
            position=curr_pos,
            velocity=(vx, vy, vz),
            sensor_readings=sensors,
        )

        info = {
            "episode_step":    self._step_count,
            "episode_reward":  self._episode_reward,
            "goal_reached":    goal,
            "collision":       collided,
            "out_of_bounds":   oob,
            "reward_breakdown": r_info.as_dict(),
            "position":        curr_pos,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pos = self._get_position(self._client.getMultirotorState())
            dist = self._dist(pos, self.cfg.target_position)
            print(
                f"Step {self._step_count:4d} | "
                f"pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                f"dist={dist:6.2f}m | "
                f"reward={self._episode_reward:8.2f}"
            )

    def close(self):
        try:
            self._client.enableApiControl(False)
        except Exception:
            pass


    def _connect(self):
        return _LocalDroneClient(self.cfg)

    def _get_obs(self) -> np.ndarray:
        state   = self._client.getMultirotorState()
        sensors = self._read_sensors()
        return self._processor.process_mock(
            position=self._get_position(state),
            sensor_readings=sensors,
        )

    def _read_sensors(self) -> List[float]:
        sensor_names = [f"Distance{i}" for i in range(self.cfg.num_distance_sensors)]
        readings = []
        for name in sensor_names:
            try:
                data = self._client.getDistanceSensorData(name)
                readings.append(min(data.distance, self.cfg.sensor_max_range))
            except Exception:
                readings.append(self.cfg.sensor_max_range)
        return readings

    @staticmethod
    def _get_position(state: Any) -> Tuple[float, float, float]:
        p = state.kinematics_estimated.position
        return (p.x_val, p.y_val, p.z_val)

    def _is_out_of_bounds(self, pos: Tuple[float, float, float]) -> bool:
        x, y, z = pos
        return (
            abs(x) > self.cfg.boundary_x
            or abs(y) > self.cfg.boundary_y
            or abs(z) > self.cfg.boundary_z
        )

    @staticmethod
    def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
