"""
Thin W&B wrapper so other modules never import wandb directly.
Keeps experiment tracking swappable without touching training code.
"""

import os
from typing import Any

import wandb


def init_run(config: dict[str, Any], project: str, name: str | None = None) -> wandb.run:
    """Initialize a W&B run. Call once at the start of a script."""
    return wandb.init(
        project=project,
        name=name,
        config=config,
    )


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dict of metrics. step=None lets W&B auto-increment."""
    wandb.log(metrics, step=step)


def finish_run() -> None:
    """Mark the current run as finished."""
    wandb.finish()
