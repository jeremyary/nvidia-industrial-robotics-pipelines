# This project was developed with assistance from AI tools.
"""SONIC training configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from wbc_pipeline.config import MLflowConfig, S3ConfigBase


@dataclass
class SonicS3Config(S3ConfigBase):
    """S3-compatible storage for SONIC checkpoints and data."""

    data_prefix: str = field(default_factory=lambda: os.environ.get("S3_DATA_PREFIX", "bones-seed/processed"))
    checkpoint_prefix: str = field(default_factory=lambda: os.environ.get("S3_CHECKPOINT_PREFIX", "sonic-checkpoints"))

    def __post_init__(self) -> None:
        from wbc_pipeline.onnx_validation import validate_s3_prefix

        validate_s3_prefix(self.data_prefix, "S3_DATA_PREFIX")
        validate_s3_prefix(self.checkpoint_prefix, "S3_CHECKPOINT_PREFIX")


@dataclass
class SonicMLflowConfig(MLflowConfig):
    """MLflow tracking with SONIC-specific default experiment name."""

    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "g1-sonic-locomotion")
    )


@dataclass
class SonicTrainingConfig:
    """Top-level SONIC configuration (import pipeline + optional training)."""

    s3: SonicS3Config = field(default_factory=SonicS3Config)
    mlflow: SonicMLflowConfig = field(default_factory=SonicMLflowConfig)
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
