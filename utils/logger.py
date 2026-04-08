from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

console = Console()

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class TrainingLogger:
    def __init__(
        self,
        log_dir: str = "logs/",
        run_name: Optional[str] = None,
        use_tb: bool = True,
    ):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        run_name = run_name or f"run_{int(time.time())}"
        self._csv_path = self.log_dir / f"{run_name}.csv"
        self._step     = 0
        self._start_t  = time.time()
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None

        self._tb: Optional[Any] = None
        if use_tb and _TB_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / run_name
            self._tb = SummaryWriter(log_dir=str(tb_dir))
            console.print(f"[blue]TensorBoard log: {tb_dir}[/blue]")


    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        if step is None:
            step = self._step
        self._step = step + 1

        elapsed = time.time() - self._start_t
        row = {"step": step, "elapsed_s": f"{elapsed:.1f}", **metrics}

        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(k, float(v), global_step=step)
                except Exception:
                    pass

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        goal_reached: bool,
        collision: bool,
    ):
        self.log(
            {
                "episode/total_reward": total_reward,
                "episode/steps":        steps,
                "episode/goal_reached": int(goal_reached),
                "episode/collision":    int(collision),
            },
            step=episode,
        )

    def log_text(self, message: str, prefix: str = "ℹ"):
        elapsed = time.time() - self._start_t
        console.print(f"[dim]{elapsed:8.1f}s[/dim]  {prefix}  {message}")

    def close(self):
        self._csv_file.close()
        if self._tb is not None:
            self._tb.close()
