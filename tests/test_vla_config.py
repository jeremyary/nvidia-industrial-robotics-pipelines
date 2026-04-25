# This project was developed with assistance from AI tools.
"""Tests for VLA fine-tuning configuration."""

from __future__ import annotations

import os
from unittest import mock

from wbc_pipeline.vla.config import VlaMLflowConfig, VlaS3Config, VlaTrainingConfig


class TestVlaS3Config:
    """Validate VLA S3 configuration defaults and env overrides."""

    def test_defaults(self):
        """S3 config has sensible defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = VlaS3Config()
        assert cfg.bucket == "wbc-training"
        assert cfg.model_prefix == "vla/base-model"
        assert cfg.dataset_prefix == "vla/dataset"
        assert cfg.checkpoint_prefix == "vla-finetune"

    def test_enabled_requires_endpoint_and_keys(self):
        """S3 is enabled only when endpoint and both keys are set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = VlaS3Config()
        assert not cfg.enabled

    def test_enabled_with_all_fields(self):
        """S3 is enabled when endpoint, access key, and secret key are set."""
        env = {"S3_ENDPOINT": "http://minio:9000", "AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = VlaS3Config()
        assert cfg.enabled

    def test_env_overrides(self):
        """Env vars override S3 defaults."""
        env = {
            "VLA_S3_MODEL_PREFIX": "custom/model",
            "VLA_S3_CHECKPOINT_PREFIX": "custom/ckpt",
            "S3_BUCKET": "my-bucket",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = VlaS3Config()
        assert cfg.model_prefix == "custom/model"
        assert cfg.checkpoint_prefix == "custom/ckpt"
        assert cfg.bucket == "my-bucket"


class TestVlaMLflowConfig:
    """Validate VLA MLflow configuration."""

    def test_defaults(self):
        """MLflow config has VLA-specific experiment name default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = VlaMLflowConfig()
        assert cfg.experiment_name == "g1-vla-finetune"
        assert not cfg.insecure_tls

    def test_enabled_requires_tracking_uri(self):
        """MLflow is enabled only when tracking URI is set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = VlaMLflowConfig()
        assert not cfg.enabled

    def test_enabled_with_uri(self):
        """MLflow is enabled when tracking URI is set."""
        with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow:8080"}, clear=True):
            cfg = VlaMLflowConfig()
        assert cfg.enabled

    def test_insecure_tls_from_env(self):
        """Insecure TLS parsed from env var."""
        with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_INSECURE_TLS": "true"}, clear=True):
            cfg = VlaMLflowConfig()
        assert cfg.insecure_tls


class TestVlaTrainingConfig:
    """Validate top-level VLA training configuration."""

    def test_defaults(self):
        """Training config has expected defaults for B200."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = VlaTrainingConfig()
        assert cfg.base_model_repo == "nvidia/GR00T-N1.7-3B"
        assert cfg.dataset_repo == "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1"
        assert cfg.embodiment_tag == "UNITREE_G1"
        assert cfg.num_gpus == 2
        assert cfg.max_steps == 2000
        assert cfg.global_batch_size == 64

    def test_env_overrides(self):
        """Env vars override training defaults."""
        env = {
            "VLA_NUM_GPUS": "4",
            "VLA_MAX_STEPS": "500",
            "VLA_GLOBAL_BATCH_SIZE": "16",
            "VLA_EMBODIMENT_TAG": "CUSTOM_ROBOT",
            "VLA_DATASET_REPO": "org/custom-dataset",
            "VLA_BASE_MODEL_REPO": "org/custom-model",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = VlaTrainingConfig()
        assert cfg.num_gpus == 4
        assert cfg.max_steps == 500
        assert cfg.global_batch_size == 16
        assert cfg.embodiment_tag == "CUSTOM_ROBOT"
        assert cfg.dataset_repo == "org/custom-dataset"
        assert cfg.base_model_repo == "org/custom-model"

    def test_hf_token_from_env(self):
        """HF_TOKEN is read from environment."""
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token_123"}, clear=True):
            cfg = VlaTrainingConfig()
        assert cfg.hf_token == "hf_test_token_123"

    def test_nested_configs_created(self):
        """S3 and MLflow sub-configs are created."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = VlaTrainingConfig()
        assert isinstance(cfg.s3, VlaS3Config)
        assert isinstance(cfg.mlflow, VlaMLflowConfig)

    def test_invalid_int_env_raises(self):
        """Non-integer env var raises ValueError."""
        with mock.patch.dict(os.environ, {"VLA_NUM_GPUS": "not_a_number"}, clear=True):
            try:
                VlaTrainingConfig()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "VLA_NUM_GPUS" in str(e)
