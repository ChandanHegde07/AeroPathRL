from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config.env_config import EnvConfig, ENV_CONFIG
from config.training_config import TrainingConfig, TRAIN_CONFIG
from environment.local_env import DroneNavigationEnv
from utils.visualization import plot_episode_trajectory, plot_evaluation_summary

console = Console()



class LiveTrajectory2D:

    def __init__(self, env_cfg: EnvConfig):
        self.cfg = env_cfg
        self.path_x: List[float] = []
        self.path_z: List[float] = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.canvas.manager.set_window_title("Drone Live Simulation (2D)")
        self.ax.set_title("Drone Trajectory (X-Z Plane)")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-2, self.cfg.boundary_x + 2)
        self.ax.set_ylim(-self.cfg.boundary_z - 2, 2)

        sx, _, sz = self.cfg.spawn_position
        tx, _, tz = self.cfg.target_position
        self.ax.scatter([sx], [sz], marker="^", s=80, label="Spawn", color="#c1121f")
        self.ax.scatter([tx], [tz], marker="*", s=130, label="Target", color="#2d6a4f")

        (self.path_line,) = self.ax.plot([], [], lw=2.0, color="#006d77", label="Path")
        self.current_point = self.ax.scatter([], [], s=70, color="#ff7f50", label="Drone")
        self.info_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, va="top", fontsize=10
        )
        self.ax.legend(loc="best")

    def update(self, position: Tuple[float, float, float], step: int, total_reward: float):
        x, _, z = position
        dist = np.linalg.norm(np.array(position) - np.array(self.cfg.target_position))

        self.path_x.append(x)
        self.path_z.append(z)
        self.path_line.set_data(self.path_x, self.path_z)
        self.current_point.set_offsets(np.array([[x, z]]))
        self.info_text.set_text(
            f"step: {step}\nreward: {total_reward:.2f}\ndist to target: {dist:.2f} m"
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()


class EpisodeResult:
    def __init__(self):
        self.positions:   List[Tuple[float, float, float]] = []
        self.rewards:     List[float]  = []
        self.actions:     List[List[float]] = []
        self.step_infos:  List[Dict]   = []
        self.total_reward: float = 0.0
        self.steps:        int   = 0
        self.goal_reached: bool  = False
        self.collision:    bool  = False
        self.success:      bool  = False

    @property
    def path_length(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.positions)):
            a, b = self.positions[i - 1], self.positions[i]
            total += np.linalg.norm(np.array(b) - np.array(a))
        return float(total)


def _run_episode(
    env: DroneNavigationEnv,
    model: PPO,
    deterministic: bool = True,
    render: bool = False,
    live_viewer: Optional[LiveTrajectory2D] = None,
) -> EpisodeResult:
    result = EpisodeResult()
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        result.rewards.append(float(reward))
        result.actions.append(action.tolist())
        result.step_infos.append(info)
        result.total_reward += float(reward)
        result.steps += 1

        pos = info.get("position")
        if pos:
            result.positions.append(pos)

        if render:
            env.render()
        if live_viewer is not None and pos:
            live_viewer.update(pos, result.steps, result.total_reward)

    result.goal_reached = any(i.get("goal_reached", False) for i in result.step_infos)
    result.collision    = any(i.get("collision",    False) for i in result.step_infos)
    result.success      = result.goal_reached
    return result



class DroneEvaluator:
    def __init__(
        self,
        model_path: str,
        env_cfg: EnvConfig = ENV_CONFIG,
    ):
        self.env_cfg    = env_cfg
        self.model_path = Path(model_path)
        console.print(f"[cyan]Loading model from {self.model_path}[/cyan]")
        self.model = PPO.load(str(self.model_path))
        console.print("[green]Model loaded.[/green]")


    def evaluate_single(self, render: bool = True, render_2d: bool = False) -> EpisodeResult:
        env = DroneNavigationEnv(cfg=self.env_cfg)
        viewer = LiveTrajectory2D(self.env_cfg) if render_2d else None
        result = _run_episode(
            env,
            self.model,
            deterministic=True,
            render=render,
            live_viewer=viewer,
        )
        env.close()
        if viewer is not None:
            viewer.close()
        self._print_single_result(result)
        return result

    def evaluate_batch(
        self,
        n_episodes: int = 20,
        deterministic: bool = True,
        save_trajectories: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict:
        results: List[EpisodeResult] = []
        env = DroneNavigationEnv(cfg=self.env_cfg)

        for _ in track(range(n_episodes), description="Evaluating…"):
            r = _run_episode(env, self.model, deterministic=deterministic)
            results.append(r)

        env.close()

        stats = self._compute_stats(results)
        self._print_batch_stats(stats, n_episodes)

        if save_trajectories and output_dir:
            self._save_results(results, stats, Path(output_dir))

        return stats


    @staticmethod
    def _compute_stats(results: List[EpisodeResult]) -> Dict:
        rewards      = [r.total_reward  for r in results]
        steps        = [r.steps         for r in results]
        path_lengths = [r.path_length   for r in results]
        successes    = [r.success        for r in results]
        collisions   = [r.collision      for r in results]

        return {
            "n_episodes":      len(results),
            "success_rate":    float(np.mean(successes)),
            "collision_rate":  float(np.mean(collisions)),
            "mean_reward":     float(np.mean(rewards)),
            "std_reward":      float(np.std(rewards)),
            "min_reward":      float(np.min(rewards)),
            "max_reward":      float(np.max(rewards)),
            "mean_steps":      float(np.mean(steps)),
            "mean_path_length":float(np.mean(path_lengths)),
        }


    @staticmethod
    def _print_single_result(result: EpisodeResult):
        status = "GOAL REACHED" if result.goal_reached else \
                 "COLLISION"   if result.collision     else "⏱  TIMEOUT"
        console.print(
            f"\n[bold]{status}[/bold]\n"
            f"  Steps        : {result.steps}\n"
            f"  Total reward : {result.total_reward:.2f}\n"
            f"  Path length  : {result.path_length:.2f} m\n"
        )

    @staticmethod
    def _print_batch_stats(stats: Dict, n: int):
        table = Table(title=f"Evaluation results — {n} episodes", border_style="cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value",  style="magenta")
        for k, v in stats.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
            else:
                table.add_row(k, str(v))
        console.print(table)


    @staticmethod
    def _save_results(
        results: List[EpisodeResult],
        stats: Dict,
        output_dir: Path,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "eval_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        rows = []
        for ep_idx, r in enumerate(results):
            for step_idx, pos in enumerate(r.positions):
                rows.append({
                    "episode": ep_idx,
                    "step": step_idx,
                    "x": pos[0], "y": pos[1], "z": pos[2],
                    "reward": r.rewards[step_idx] if step_idx < len(r.rewards) else 0.0,
                })
        pd.DataFrame(rows).to_csv(output_dir / "trajectories.csv", index=False)
        console.print(f"[green]Results saved to {output_dir}[/green]")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained drone agent")
    parser.add_argument("--model",   required=True, help="Path to .zip model file")
    parser.add_argument("--mode",    default="batch", choices=["single", "batch"])
    parser.add_argument("--n",       type=int, default=20, help="Episodes (batch mode)")
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--render2d", action="store_true",
                        help="Show live 2D trajectory window (single mode)")
    parser.add_argument("--save",    action="store_true", help="Save trajectory data")
    parser.add_argument("--out_dir", default="eval_results/")
    args = parser.parse_args()

    evaluator = DroneEvaluator(model_path=args.model)

    if args.mode == "single":
        evaluator.evaluate_single(render=args.render, render_2d=args.render2d)
    else:
        evaluator.evaluate_batch(
            n_episodes=args.n,
            save_trajectories=args.save,
            output_dir=args.out_dir,
        )
