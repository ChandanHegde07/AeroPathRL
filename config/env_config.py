"""
config/env_config.py
Environment configuration for the AirSim drone RL navigation project.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EnvConfig:
    # ── AirSim connection ────────────────────────────────────────────────────
    airsim_ip: str = "127.0.0.1"
    airsim_port: int = 41451

    # ── World / spawn settings ───────────────────────────────────────────────
    # Target position the drone must reach (x, y, z) in NED metres
    target_position: Tuple[float, float, float] = (30.0, 0.0, -10.0)

    # Spawn position of the drone at episode reset
    spawn_position: Tuple[float, float, float] = (0.0, 0.0, -5.0)

    # Tolerance (metres) to consider the goal reached
    goal_tolerance: float = 2.0

    # Half-extents of the navigable bounding box (m)
    boundary_x: float = 60.0
    boundary_y: float = 60.0
    boundary_z: float = 30.0

    # ── Episode limits ───────────────────────────────────────────────────────
    max_steps_per_episode: int = 500
    step_duration_sec: float = 0.1          # real-time per env step

    # ── State space ──────────────────────────────────────────────────────────
    # Lidar / distance sensor rays
    num_distance_sensors: int = 8
    sensor_max_range: float = 20.0          # metres

    # Whether to include RGB camera frames in the state
    use_camera: bool = False
    camera_resolution: Tuple[int, int] = (84, 84)

    # ── Action space ─────────────────────────────────────────────────────────
    # Continuous velocity commands (vx, vy, vz) in m/s
    max_velocity: float = 5.0
    action_duration_sec: float = 0.1

    # ── Reward shaping (see reward_function.py) ──────────────────────────────
    reward_goal_reached: float = 200.0
    reward_collision: float = -100.0
    reward_boundary_violation: float = -50.0
    reward_step_penalty: float = -0.1
    reward_progress_scale: float = 5.0      # multiplied by Δdistance-to-goal
    reward_altitude_penalty_scale: float = -0.5  # penalise extreme altitudes

    # ── Obstacle settings ───────────────────────────────────────────────────
    dynamic_obstacles: bool = False         # enable moving obstacles
    num_random_obstacles: int = 10

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    render_mode: str = "none"               # "none" | "human"
    verbose: bool = False


# Singleton used throughout the project
ENV_CONFIG = EnvConfig()
