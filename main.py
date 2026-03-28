from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_train(args: argparse.Namespace):
    from config.env_config import ENV_CONFIG
    from config.training_config import TRAIN_CONFIG
    from agent.train import DroneTrainer

    if args.timesteps:
        TRAIN_CONFIG.total_timesteps = args.timesteps
    if args.n_envs:
        TRAIN_CONFIG.n_envs = args.n_envs

    trainer = DroneTrainer(env_cfg=ENV_CONFIG, train_cfg=TRAIN_CONFIG)
    trainer.train(resume_from=args.resume)


def cmd_evaluate(args: argparse.Namespace):
    from agent.evaluate import DroneEvaluator
    from config.env_config import ENV_CONFIG

    evaluator = DroneEvaluator(model_path=args.model, env_cfg=ENV_CONFIG)

    if args.mode == "single":
        evaluator.evaluate_single(render=args.render)
    else:
        evaluator.evaluate_batch(
            n_episodes=args.n,
            deterministic=not args.stochastic,
            save_trajectories=args.save,
            output_dir=args.out_dir,
        )


def cmd_demo(args: argparse.Namespace):
    from agent.evaluate import DroneEvaluator
    from config.env_config import ENV_CONFIG

    evaluator = DroneEvaluator(model_path=args.model, env_cfg=ENV_CONFIG)
    console.print(
        Panel(
            "Running live demo episode — watch the console for step output.",
            title="🚁  Demo Mode",
            border_style="yellow",
        )
    )
    evaluator.evaluate_single(render=True)


def cmd_info(_args: argparse.Namespace):
    from config.env_config import ENV_CONFIG
    from config.training_config import TRAIN_CONFIG

    env_table = Table(title="Environment Config", border_style="blue")
    env_table.add_column("Parameter", style="cyan")
    env_table.add_column("Value",     style="magenta")
    for k, v in vars(ENV_CONFIG).items():
        env_table.add_row(k, str(v))

    train_table = Table(title="Training Config", border_style="green")
    train_table.add_column("Parameter", style="cyan")
    train_table.add_column("Value",     style="magenta")
    for k, v in vars(TRAIN_CONFIG).items():
        train_table.add_row(k, str(v))

    console.print(env_table)
    console.print(train_table)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone_rl",
        description="Autonomous Drone Navigation using Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ────────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Train a PPO agent")
    p_train.add_argument("--resume",     type=str, default=None,
                         help="Path to saved model to resume training from")
    p_train.add_argument("--timesteps",  type=int, default=None,
                         help="Override total training timesteps")
    p_train.add_argument("--n_envs",     type=int, default=None,
                         help="Override number of parallel environments")

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate a trained agent")
    p_eval.add_argument("--model",      required=True, help="Path to .zip model")
    p_eval.add_argument("--mode",       default="batch", choices=["single", "batch"])
    p_eval.add_argument("--n",          type=int, default=20, help="Episodes (batch)")
    p_eval.add_argument("--render",     action="store_true")
    p_eval.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (default: deterministic)")
    p_eval.add_argument("--save",       action="store_true",
                        help="Save trajectory CSV and stats JSON")
    p_eval.add_argument("--out_dir",    default="eval_results/")

    # ── demo ─────────────────────────────────────────────────────────────────
    p_demo = sub.add_parser("demo", help="Run one rendered episode")
    p_demo.add_argument("--model", required=True, help="Path to .zip model")

    # ── info ─────────────────────────────────────────────────────────────────
    sub.add_parser("info", help="Print current configuration")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    console.print(
        Panel(
            "[bold cyan]Autonomous Drone Navigation[/bold cyan] — "
            "[green]Reinforcement Learning[/green]",
            subtitle="PPO · AirSim · Gymnasium",
            border_style="cyan",
        )
    )

    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "demo":     cmd_demo,
        "info":     cmd_info,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
