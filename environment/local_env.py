from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.env_config import EnvConfig, ENV_CONFIG
from environment.reward_function import RewardFunction, RewardInfo
from environment.state_processing import StateProcessor


DIFFICULTY_PRESETS: Dict[str, Dict[str, float]] = {
    "easy": {
        "obstacle_density": 0.35,
        "obstacle_radius_min": 0.9,
        "obstacle_radius_max": 1.6,
        "wind_sigma": 0.08,
        "sensor_noise_sigma": 0.12,
        "goal_tolerance_scale": 1.2,
        "max_steps_scale": 1.1,
    },
    "medium": {
        "obstacle_density": 0.65,
        "obstacle_radius_min": 1.1,
        "obstacle_radius_max": 2.0,
        "wind_sigma": 0.22,
        "sensor_noise_sigma": 0.2,
        "goal_tolerance_scale": 1.0,
        "max_steps_scale": 1.0,
    },
    "hard": {
        "obstacle_density": 1.0,
        "obstacle_radius_min": 1.4,
        "obstacle_radius_max": 2.8,
        "wind_sigma": 0.35,
        "sensor_noise_sigma": 0.32,
        "goal_tolerance_scale": 0.8,
        "max_steps_scale": 0.9,
    },
}


class _MockVector3r:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x_val, self.y_val, self.z_val = x, y, z


class _MockQuaternion:
    def __init__(self):
        self.w_val, self.x_val, self.y_val, self.z_val = 1.0, 0.0, 0.0, 0.0


class _MockKinematics:
    def __init__(
        self,
        pos: Tuple[float, float, float] = (0.0, 0.0, -5.0),
        vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.position = _MockVector3r(*pos)
        self.linear_velocity = _MockVector3r(*vel)
        self.orientation = _MockQuaternion()


class _MockState:
    def __init__(
        self,
        pos: Tuple[float, float, float] = (0.0, 0.0, -5.0),
        vel: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.kinematics_estimated = _MockKinematics(pos, vel)
        self.collision_info = type("CI", (), {"has_collided": False})()


class _LocalDroneClient:
    def __init__(self, cfg: EnvConfig):
        self._cfg = cfg
        self._pos = list(cfg.spawn_position)
        self._vel = [0.0, 0.0, 0.0]
        self._collided = False

    def confirmConnection(self):
        pass

    def enableApiControl(self, *a, **k):
        pass

    def armDisarm(self, *a, **k):
        pass

    def takeoffAsync(self):
        return self

    def join(self):
        return self

    def getMultirotorState(self):
        return _MockState(tuple(self._pos), tuple(self._vel))

    def moveByVelocityAsync(self, vx: float, vy: float, vz: float, duration: float, **kwargs):
        dt = duration
        self._vel = [vx, vy, vz]
        self._pos[0] += vx * dt
        self._pos[1] += vy * dt
        self._pos[2] += vz * dt

        # Simple "ground hit" proxy collision.
        if self._pos[2] > -1.0:
            self._collided = True
        return self

    def simGetCollisionInfo(self):
        info = type("CI", (), {"has_collided": self._collided})()
        self._collided = False
        return info

    def reset(self):
        self._pos = list(self._cfg.spawn_position)
        self._vel = [0.0, 0.0, 0.0]
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
        self.cfg = cfg
        self.render_mode = render_mode or cfg.render_mode

        self._processor = StateProcessor(cfg)
        self._reward_fn = RewardFunction(cfg)

        obs_dim = self._processor.obs_dim
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self._client = self._connect()

        self._step_count = 0
        self._prev_pos = cfg.spawn_position
        self._episode_reward = 0.0
        self._reward_log: List[RewardInfo] = []

        self._rng = np.random.default_rng(cfg.seed)
        self._difficulty = self._resolve_difficulty(cfg.difficulty_level)
        self._goal_tolerance = cfg.goal_tolerance
        self._max_steps = cfg.max_steps_per_episode
        self._obstacles: List[Tuple[float, float, float, float]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._difficulty = self._resolve_difficulty(self.cfg.difficulty_level)
        self._goal_tolerance = self.cfg.goal_tolerance * self._difficulty["goal_tolerance_scale"]
        self._max_steps = max(
            50,
            int(self.cfg.max_steps_per_episode * self._difficulty["max_steps_scale"]),
        )

        self._client.reset()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)
        self._client.takeoffAsync().join()

        self._step_count = 0
        self._episode_reward = 0.0
        self._reward_log = []
        self._prev_pos = self.cfg.spawn_position
        self._obstacles = self._spawn_obstacles()

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "episode_step": 0,
            "difficulty_level": self.cfg.difficulty_level,
            "obstacle_count": len(self._obstacles),
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        vx, vy, vz = (action * self.cfg.max_velocity).tolist()

        wind = self._rng.normal(0.0, self._difficulty["wind_sigma"], size=3)
        vx = float(np.clip(vx + wind[0], -self.cfg.max_velocity, self.cfg.max_velocity))
        vy = float(np.clip(vy + wind[1], -self.cfg.max_velocity, self.cfg.max_velocity))
        vz = float(np.clip(vz + wind[2], -self.cfg.max_velocity, self.cfg.max_velocity))

        self._client.moveByVelocityAsync(
            vx,
            vy,
            vz,
            duration=self.cfg.action_duration_sec,
        ).join()
        time.sleep(max(0.0, self.cfg.step_duration_sec - self.cfg.action_duration_sec))

        state = self._client.getMultirotorState()
        col_info = self._client.simGetCollisionInfo()
        curr_pos = self._get_position(state)
        sensors = self._read_sensors(curr_pos)

        collided = col_info.has_collided or self._collides_with_obstacle(curr_pos)
        oob = self._is_out_of_bounds(curr_pos)
        goal = self._dist(curr_pos, self.cfg.target_position) < self._goal_tolerance

        reward, r_info = self._reward_fn.compute(
            curr_pos,
            self._prev_pos,
            collided,
            oob,
            goal,
        )
        self._prev_pos = curr_pos
        self._step_count += 1
        self._episode_reward += reward
        self._reward_log.append(r_info)

        terminated = collided or goal
        truncated = self._step_count >= self._max_steps

        obs = self._processor.process_mock(
            position=curr_pos,
            velocity=(vx, vy, vz),
            sensor_readings=sensors,
        )

        info = {
            "episode_step": self._step_count,
            "episode_reward": self._episode_reward,
            "goal_reached": goal,
            "collision": collided,
            "out_of_bounds": oob,
            "reward_breakdown": r_info.as_dict(),
            "position": curr_pos,
            "difficulty_level": self.cfg.difficulty_level,
            "obstacle_count": len(self._obstacles),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pos = self._get_position(self._client.getMultirotorState())
            dist = self._dist(pos, self.cfg.target_position)
            print(
                f"Step {self._step_count:4d} | "
                f"lvl={self.cfg.difficulty_level:>6s} | "
                f"obs={len(self._obstacles):2d} | "
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
        state = self._client.getMultirotorState()
        pos = self._get_position(state)
        sensors = self._read_sensors(pos)
        return self._processor.process_mock(
            position=pos,
            sensor_readings=sensors,
        )

    def _resolve_difficulty(self, level: str) -> Dict[str, float]:
        return DIFFICULTY_PRESETS.get(level, DIFFICULTY_PRESETS["medium"])

    def _spawn_obstacles(self) -> List[Tuple[float, float, float, float]]:
        if not self.cfg.dynamic_obstacles:
            return []

        density = self._difficulty["obstacle_density"]
        count = max(0, int(round(self.cfg.num_random_obstacles * density)))
        if count == 0:
            return []

        r_min = self._difficulty["obstacle_radius_min"]
        r_max = self._difficulty["obstacle_radius_max"]

        sx, sy, sz = self.cfg.spawn_position
        tx, ty, tz = self.cfg.target_position

        obstacles: List[Tuple[float, float, float, float]] = []
        max_attempts = count * 45

        for _ in range(max_attempts):
            if len(obstacles) >= count:
                break

            x = float(self._rng.uniform(min(sx, tx) + 4.0, max(sx, tx) + 6.0))
            y = float(self._rng.uniform(-self.cfg.boundary_y * 0.7, self.cfg.boundary_y * 0.7))
            z = float(self._rng.uniform(min(sz, tz) - 5.0, max(sz, tz) + 3.0))
            r = float(self._rng.uniform(r_min, r_max))

            if self._dist((x, y, z), self.cfg.spawn_position) < r + 4.0:
                continue
            if self._dist((x, y, z), self.cfg.target_position) < r + 4.0:
                continue

            overlap = False
            for ox, oy, oz, orad in obstacles:
                if self._dist((x, y, z), (ox, oy, oz)) < (r + orad + 1.2):
                    overlap = True
                    break
            if overlap:
                continue

            obstacles.append((x, y, z, r))

        return obstacles

    def _read_sensors(self, position: Optional[Tuple[float, float, float]] = None) -> List[float]:
        if position is None:
            state = self._client.getMultirotorState()
            position = self._get_position(state)

        x, y, z = position
        noise_sigma = self._difficulty["sensor_noise_sigma"]

        readings: List[float] = []
        n = self.cfg.num_distance_sensors
        for i in range(n):
            angle = (2.0 * math.pi * i) / max(n, 1)
            dx, dy = math.cos(angle), math.sin(angle)

            dist_to_boundary = self._ray_distance_to_boundary(x, y, dx, dy)
            dist_to_obstacle = self._ray_distance_to_obstacle(x, y, z, dx, dy)
            dist = min(self.cfg.sensor_max_range, dist_to_boundary, dist_to_obstacle)

            noisy = dist + float(self._rng.normal(0.0, noise_sigma))
            readings.append(float(np.clip(noisy, 0.2, self.cfg.sensor_max_range)))

        return readings

    def _ray_distance_to_boundary(self, x: float, y: float, dx: float, dy: float) -> float:
        candidates: List[float] = []

        if abs(dx) > 1e-8:
            tx = ((self.cfg.boundary_x if dx > 0 else -self.cfg.boundary_x) - x) / dx
            if tx >= 0.0:
                candidates.append(float(tx))
        if abs(dy) > 1e-8:
            ty = ((self.cfg.boundary_y if dy > 0 else -self.cfg.boundary_y) - y) / dy
            if ty >= 0.0:
                candidates.append(float(ty))

        if not candidates:
            return self.cfg.sensor_max_range
        return min(candidates)

    def _ray_distance_to_obstacle(self, x: float, y: float, z: float, dx: float, dy: float) -> float:
        best = self.cfg.sensor_max_range

        for ox, oy, oz, r in self._obstacles:
            dz = z - oz
            if abs(dz) >= r:
                continue

            effective_r2 = max(0.0, (r * r) - (dz * dz))
            fx = x - ox
            fy = y - oy

            b = 2.0 * (fx * dx + fy * dy)
            c = (fx * fx + fy * fy) - effective_r2
            disc = (b * b) - (4.0 * c)
            if disc < 0.0:
                continue

            sq = math.sqrt(disc)
            t1 = (-b - sq) / 2.0
            t2 = (-b + sq) / 2.0
            candidates = [t for t in (t1, t2) if t >= 0.0]
            if not candidates:
                continue
            best = min(best, min(candidates))

        return best

    def _collides_with_obstacle(self, pos: Tuple[float, float, float]) -> bool:
        drone_radius = 0.35
        for ox, oy, oz, r in self._obstacles:
            if self._dist(pos, (ox, oy, oz)) <= (r + drone_radius):
                return True
        return False

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
