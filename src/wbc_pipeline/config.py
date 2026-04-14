# This project was developed with assistance from AI tools.
"""Training configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_env(name: str, default: int) -> int:
    """Read an integer from an environment variable with a descriptive error."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid integer") from None


@dataclass
class S3ConfigBase:
    """Base S3-compatible storage config shared across backends."""

    endpoint: str = field(default_factory=lambda: os.environ.get("S3_ENDPOINT", ""))
    bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", "wbc-training"))
    access_key: str = field(default_factory=lambda: os.environ.get("AWS_ACCESS_KEY_ID", ""), repr=False)
    secret_key: str = field(default_factory=lambda: os.environ.get("AWS_SECRET_ACCESS_KEY", ""), repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.endpoint and self.access_key and self.secret_key)

    def create_client(self):
        """Create a boto3 S3 client from this config."""
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )


@dataclass
class S3Config(S3ConfigBase):
    """S3-compatible storage for checkpoints."""

    prefix: str = field(default_factory=lambda: os.environ.get("S3_PREFIX", "checkpoints"))


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", ""))
    experiment_name: str = field(
        default_factory=lambda: os.environ.get("MLFLOW_EXPERIMENT_NAME", "g1-29dof-locomotion")
    )
    insecure_tls: bool = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "false").lower() == "true"
    )

    @property
    def enabled(self) -> bool:
        return bool(self.tracking_uri)


@dataclass
class TrainingConfig:
    """Top-level training configuration."""

    s3: S3Config = field(default_factory=S3Config)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    checkpoint_interval: int = field(default_factory=lambda: _int_env("CHECKPOINT_INTERVAL", 50))
    resume_checkpoint: str = field(default_factory=lambda: os.environ.get("RESUME_CHECKPOINT", ""))
