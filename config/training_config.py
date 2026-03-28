from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    algorithm: str = "PPO"

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1

    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    save_freq: int = 50_000

    policy_type: str = "MlpPolicy"
    net_arch: list = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"

    n_envs: int = 1

    log_dir: Path = Path("logs/")
    model_dir: Path = Path("models/saved_models/")
    tensorboard_log: Path = Path("logs/tensorboard/")
    best_model_name: str = "best_drone_ppo"
    final_model_name: str = "final_drone_ppo"

    device: str = "auto"

    use_curriculum: bool = False
    curriculum_stages: list = field(default_factory=lambda: [
        {"timesteps": 0,        "target_distance": 10.0, "num_obstacles": 2},
        {"timesteps": 200_000,  "target_distance": 20.0, "num_obstacles": 5},
        {"timesteps": 500_000,  "target_distance": 30.0, "num_obstacles": 10},
    ])

    seed: int = 42
    verbose: int = 1
    deterministic_eval: bool = True


TRAIN_CONFIG = TrainingConfig()
