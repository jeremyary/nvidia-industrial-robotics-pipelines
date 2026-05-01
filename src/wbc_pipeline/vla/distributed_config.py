# This project was developed with assistance from AI tools.
"""Distributed VLA fine-tuning configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from wbc_pipeline.config import _int_env
from wbc_pipeline.vla.config import VlaTrainingConfig


def _float_env(name: str, default: float) -> float:
    """Read a float from an environment variable."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid float") from None


@dataclass
class DistributedTrainingConfig(VlaTrainingConfig):
    """Extended VLA config for multi-node training and scaling trials."""

    num_nodes: int = field(default_factory=lambda: _int_env("VLA_NUM_NODES", 1))
    rdzv_endpoint: str = field(default_factory=lambda: os.environ.get("VLA_RDZV_ENDPOINT", ""))
    learning_rate: float = field(default_factory=lambda: _float_env("VLA_LEARNING_RATE", 1e-4))
    warmup_ratio: float = field(default_factory=lambda: _float_env("VLA_WARMUP_RATIO", 0.05))
    state_dropout_prob: float = field(default_factory=lambda: _float_env("VLA_STATE_DROPOUT_PROB", 0.2))
    dataloader_num_workers: int = field(default_factory=lambda: _int_env("VLA_DATALOADER_WORKERS", 4))
    trial_name: str = field(default_factory=lambda: os.environ.get("VLA_TRIAL_NAME", ""))
