# This project was developed with assistance from AI tools.
"""Tests for SONIC configuration."""

from __future__ import annotations

import os
from unittest import mock

import pytest

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

    def test_path_traversal_rejected(self):
        """S3 prefix with path traversal is rejected."""
        env = {"S3_CHECKPOINT_PREFIX": "../../evil"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="must not contain"):
                SonicS3Config()


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
    """Validate top-level SONIC configuration."""

    def test_defaults(self):
        """Config has expected defaults after pivot to import-only."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = SonicTrainingConfig()
        assert cfg.hf_token == ""

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
