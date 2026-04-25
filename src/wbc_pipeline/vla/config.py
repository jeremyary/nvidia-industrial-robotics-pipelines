# This project was developed with assistance from AI tools.
"""VLA fine-tuning configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from wbc_pipeline.config import MLflowConfig, S3ConfigBase, _int_env


@dataclass
class VlaS3Config(S3ConfigBase):
    """S3-compatible storage for VLA base model, dataset, and fine-tuned checkpoints."""

    model_prefix: str = field(default_factory=lambda: os.environ.get("VLA_S3_MODEL_PREFIX", "vla/base-model"))
    dataset_prefix: str = field(default_factory=lambda: os.environ.get("VLA_S3_DATASET_PREFIX", "vla/dataset"))
    checkpoint_prefix: str = field(default_factory=lambda: os.environ.get("VLA_S3_CHECKPOINT_PREFIX", "vla-finetune"))

    def __post_init__(self) -> None:
        from wbc_pipeline.onnx_validation import validate_s3_prefix

        validate_s3_prefix(self.model_prefix, "VLA_S3_MODEL_PREFIX")
        validate_s3_prefix(self.dataset_prefix, "VLA_S3_DATASET_PREFIX")
        validate_s3_prefix(self.checkpoint_prefix, "VLA_S3_CHECKPOINT_PREFIX")


@dataclass
class VlaMLflowConfig(MLflowConfig):
    """MLflow tracking with VLA-specific default experiment name."""

    experiment_name: str = field(default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "g1-vla-finetune"))


@dataclass
class VlaTrainingConfig:
    """Top-level VLA fine-tuning configuration."""

    s3: VlaS3Config = field(default_factory=VlaS3Config)
    mlflow: VlaMLflowConfig = field(default_factory=VlaMLflowConfig)
    base_model_repo: str = field(default_factory=lambda: os.environ.get("VLA_BASE_MODEL_REPO", "nvidia/GR00T-N1.7-3B"))
    dataset_repo: str = field(
        default_factory=lambda: os.environ.get("VLA_DATASET_REPO", "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1")
    )
    embodiment_tag: str = field(default_factory=lambda: os.environ.get("VLA_EMBODIMENT_TAG", "UNITREE_G1"))
    num_gpus: int = field(default_factory=lambda: _int_env("VLA_NUM_GPUS", 2))
    max_steps: int = field(default_factory=lambda: _int_env("VLA_MAX_STEPS", 2000))
    global_batch_size: int = field(default_factory=lambda: _int_env("VLA_GLOBAL_BATCH_SIZE", 64))
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
