# This project was developed with assistance from AI tools.
"""SONIC training configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from wbc_pipeline.config import MLflowConfig, S3ConfigBase, _int_env


@dataclass
class SonicS3Config(S3ConfigBase):
    """S3-compatible storage for SONIC checkpoints and data."""

    data_prefix: str = field(default_factory=lambda: os.environ.get("S3_DATA_PREFIX", "bones-seed/processed"))
    checkpoint_prefix: str = field(default_factory=lambda: os.environ.get("S3_CHECKPOINT_PREFIX", "sonic-checkpoints"))


@dataclass
class SonicMLflowConfig(MLflowConfig):
    """MLflow tracking with SONIC-specific default experiment name."""

    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "g1-sonic-locomotion")
    )


@dataclass
class SonicTrainingConfig:
    """Top-level SONIC training configuration."""

    s3: SonicS3Config = field(default_factory=SonicS3Config)
    mlflow: SonicMLflowConfig = field(default_factory=SonicMLflowConfig)
    num_gpus: int = field(default_factory=lambda: _int_env("SONIC_NUM_GPUS", 4))
    num_envs: int = field(default_factory=lambda: _int_env("SONIC_NUM_ENVS", 4096))
    max_iterations: int = field(default_factory=lambda: _int_env("SONIC_MAX_ITERATIONS", 10000))
    hydra_experiment: str = field(default_factory=lambda: os.environ.get("SONIC_HYDRA_EXPERIMENT", "sonic_release"))
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    # 10K iters / 100 interval = 100 checkpoints × ~500MB each ≈ 50GB per run
    checkpoint_interval: int = field(default_factory=lambda: _int_env("CHECKPOINT_INTERVAL", 100))
