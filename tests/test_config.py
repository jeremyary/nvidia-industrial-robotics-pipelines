# This project was developed with assistance from AI tools.
"""Tests for training configuration."""

from __future__ import annotations

from wbc_pipeline.config import MLflowConfig, S3Config, TrainingConfig, VideoConfig


class TestS3Config:
    """Validate S3 config from env vars."""

    def test_disabled_by_default(self):
        """S3 is disabled when env vars are not set."""
        cfg = S3Config(endpoint="", access_key="", secret_key="")
        assert cfg.enabled is False

    def test_enabled_with_all_fields(self):
        """S3 is enabled when endpoint and credentials are provided."""
        cfg = S3Config(endpoint="https://s3.example.com", access_key="key", secret_key="secret")
        assert cfg.enabled is True

    def test_disabled_without_endpoint(self):
        """S3 requires endpoint."""
        cfg = S3Config(endpoint="", access_key="key", secret_key="secret")
        assert cfg.enabled is False

    def test_default_bucket(self):
        """Default bucket is wbc-training."""
        cfg = S3Config()
        assert cfg.bucket == "wbc-training"

    def test_default_prefix(self):
        """Default prefix is checkpoints."""
        cfg = S3Config()
        assert cfg.prefix == "checkpoints"

    def test_from_env_vars(self, monkeypatch):
        """Config reads from environment variables."""
        monkeypatch.setenv("S3_ENDPOINT", "https://minio.example.com")
        monkeypatch.setenv("S3_BUCKET", "my-bucket")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "mykey")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "mysecret")
        cfg = S3Config()
        assert cfg.endpoint == "https://minio.example.com"
        assert cfg.bucket == "my-bucket"
        assert cfg.access_key == "mykey"
        assert cfg.secret_key == "mysecret"
        assert cfg.enabled is True


class TestMLflowConfig:
    """Validate MLflow config from env vars."""

    def test_disabled_by_default(self):
        """MLflow is disabled when tracking URI is not set."""
        cfg = MLflowConfig(tracking_uri="")
        assert cfg.enabled is False

    def test_enabled_with_uri(self):
        """MLflow is enabled when tracking URI is provided."""
        cfg = MLflowConfig(tracking_uri="http://mlflow.example.com")
        assert cfg.enabled is True

    def test_default_experiment_name(self):
        """Default experiment name."""
        cfg = MLflowConfig()
        assert cfg.experiment_name == "g1-29dof-locomotion"

    def test_from_env_vars(self, monkeypatch):
        """Config reads from environment variables."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "my-experiment")
        cfg = MLflowConfig()
        assert cfg.tracking_uri == "http://mlflow:5000"
        assert cfg.experiment_name == "my-experiment"


class TestVideoConfig:
    """Validate video recording config."""

    def test_disabled_by_default(self):
        """Video is disabled without VIDEO_ENABLED env var."""
        cfg = VideoConfig()
        assert cfg.enabled is False

    def test_enabled_with_env_var(self, monkeypatch):
        """Video is enabled when VIDEO_ENABLED=true."""
        monkeypatch.setenv("VIDEO_ENABLED", "true")
        cfg = VideoConfig()
        assert cfg.enabled is True

    def test_enabled_with_1(self, monkeypatch):
        """VIDEO_ENABLED=1 also works."""
        monkeypatch.setenv("VIDEO_ENABLED", "1")
        cfg = VideoConfig()
        assert cfg.enabled is True

    def test_default_num_recordings(self):
        """Default is 10 recordings."""
        cfg = VideoConfig()
        assert cfg.num_recordings == 10

    def test_default_steps_per_video(self):
        """Default is 200 steps per video."""
        cfg = VideoConfig()
        assert cfg.steps_per_video == 200

    def test_from_env_vars(self, monkeypatch):
        """Config reads from environment variables."""
        monkeypatch.setenv("VIDEO_NUM_RECORDINGS", "5")
        monkeypatch.setenv("VIDEO_STEPS", "300")
        cfg = VideoConfig()
        assert cfg.num_recordings == 5
        assert cfg.steps_per_video == 300


class TestTrainingConfig:
    """Validate top-level training config."""

    def test_default_checkpoint_interval(self):
        """Default checkpoint interval is 50."""
        cfg = TrainingConfig()
        assert cfg.checkpoint_interval == 50

    def test_checkpoint_interval_from_env(self, monkeypatch):
        """Checkpoint interval reads from env var."""
        monkeypatch.setenv("CHECKPOINT_INTERVAL", "100")
        cfg = TrainingConfig()
        assert cfg.checkpoint_interval == 100

    def test_s3_and_mlflow_disabled_by_default(self):
        """Both S3 and MLflow are disabled without env vars."""
        cfg = TrainingConfig()
        assert cfg.s3.enabled is False
        assert cfg.mlflow.enabled is False

    def test_video_disabled_by_default(self):
        """Video recording is disabled without env vars."""
        cfg = TrainingConfig()
        assert cfg.video.enabled is False
