from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class EnvConfig:
    airsim_ip: str = "127.0.0.1"
    airsim_port: int = 41451

    target_position: Tuple[float, float, float] = (30.0, 0.0, -10.0)

    spawn_position: Tuple[float, float, float] = (0.0, 0.0, -5.0)

    goal_tolerance: float = 2.0

    boundary_x: float = 60.0
    boundary_y: float = 60.0
    boundary_z: float = 30.0

    max_steps_per_episode: int = 500
    step_duration_sec: float = 0.1

    num_distance_sensors: int = 8
    sensor_max_range: float = 20.0

    use_camera: bool = False
    camera_resolution: Tuple[int, int] = (84, 84)

    max_velocity: float = 5.0
    action_duration_sec: float = 0.1

    reward_goal_reached: float = 200.0
    reward_collision: float = -100.0
    reward_boundary_violation: float = -50.0
    reward_step_penalty: float = -0.1
    reward_progress_scale: float = 5.0
    reward_altitude_penalty_scale: float = -0.5

    dynamic_obstacles: bool = False
    num_random_obstacles: int = 10

    seed: int = 42
    render_mode: str = "none"
    verbose: bool = False


ENV_CONFIG = EnvConfig()
