# This project was developed with assistance from AI tools.
"""Tests for SONIC training configuration."""

from __future__ import annotations

import os
from unittest import mock

from wbc_pipeline.sonic.config import SonicMLflowConfig, SonicS3Config, SonicTrainingConfig


class TestSonicS3Config:
    """Validate SONIC S3 configuration defaults and env overrides."""

    def test_defaults(self):
        """S3 config has sensible defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicS3Config()
        assert cfg.bucket == "wbc-training"
        assert cfg.data_prefix == "bones-seed/processed"
        assert cfg.checkpoint_prefix == "sonic-checkpoints"

    def test_enabled_requires_endpoint_and_keys(self):
        """S3 is enabled only when endpoint and both keys are set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicS3Config()
        assert not cfg.enabled

    def test_enabled_with_all_fields(self):
        """S3 is enabled when endpoint, access key, and secret key are set."""
        env = {"S3_ENDPOINT": "http://minio:9000", "AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = SonicS3Config()
        assert cfg.enabled

    def test_env_overrides(self):
        """Env vars override S3 defaults."""
        env = {"S3_DATA_PREFIX": "custom/data", "S3_CHECKPOINT_PREFIX": "custom/ckpt", "S3_BUCKET": "my-bucket"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = SonicS3Config()
        assert cfg.data_prefix == "custom/data"
        assert cfg.checkpoint_prefix == "custom/ckpt"
        assert cfg.bucket == "my-bucket"


class TestSonicMLflowConfig:
    """Validate SONIC MLflow configuration."""

    def test_defaults(self):
        """MLflow config has SONIC-specific experiment name default."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicMLflowConfig()
        assert cfg.experiment_name == "g1-sonic-locomotion"
        assert not cfg.insecure_tls

    def test_enabled_requires_tracking_uri(self):
        """MLflow is enabled only when tracking URI is set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicMLflowConfig()
        assert not cfg.enabled

    def test_enabled_with_uri(self):
        """MLflow is enabled when tracking URI is set."""
        with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow:8080"}, clear=True):
            cfg = SonicMLflowConfig()
        assert cfg.enabled

    def test_insecure_tls_from_env(self):
        """Insecure TLS parsed from env var."""
        with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_INSECURE_TLS": "true"}, clear=True):
            cfg = SonicMLflowConfig()
        assert cfg.insecure_tls


class TestSonicTrainingConfig:
    """Validate top-level SONIC training configuration."""

    def test_defaults(self):
        """Training config has expected defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicTrainingConfig()
        assert cfg.num_gpus == 4
        assert cfg.num_envs == 4096
        assert cfg.max_iterations == 10000
        assert cfg.hydra_experiment == "sonic_release"
        assert cfg.checkpoint_interval == 100

    def test_env_overrides(self):
        """Env vars override training defaults."""
        env = {
            "SONIC_NUM_GPUS": "2",
            "SONIC_NUM_ENVS": "512",
            "SONIC_MAX_ITERATIONS": "100",
            "SONIC_HYDRA_EXPERIMENT": "sonic_bones_seed",
            "CHECKPOINT_INTERVAL": "50",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = SonicTrainingConfig()
        assert cfg.num_gpus == 2
        assert cfg.num_envs == 512
        assert cfg.max_iterations == 100
        assert cfg.hydra_experiment == "sonic_bones_seed"
        assert cfg.checkpoint_interval == 50

    def test_hf_token_from_env(self):
        """HF_TOKEN is read from environment."""
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token_123"}, clear=True):
            cfg = SonicTrainingConfig()
        assert cfg.hf_token == "hf_test_token_123"

    def test_nested_configs_created(self):
        """S3 and MLflow sub-configs are created."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicTrainingConfig()
        assert isinstance(cfg.s3, SonicS3Config)
        assert isinstance(cfg.mlflow, SonicMLflowConfig)
