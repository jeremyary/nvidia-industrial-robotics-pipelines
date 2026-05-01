# This project was developed with assistance from AI tools.
"""Tests for distributed VLA fine-tuning configuration."""

from __future__ import annotations

import os
from unittest import mock

from wbc_pipeline.vla.distributed_config import DistributedTrainingConfig


class TestDistributedTrainingConfig:
    """Validate distributed training config defaults and env overrides."""

    def test_defaults(self):
        """Distributed config has expected defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = DistributedTrainingConfig()
        assert cfg.num_nodes == 1
        assert cfg.rdzv_endpoint == ""
        assert cfg.learning_rate == 1e-4
        assert cfg.warmup_ratio == 0.05
        assert cfg.state_dropout_prob == 0.2
        assert cfg.dataloader_num_workers == 4
        assert cfg.trial_name == ""

    def test_inherits_base_defaults(self):
        """Distributed config inherits VlaTrainingConfig defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            cfg = DistributedTrainingConfig()
        assert cfg.num_gpus == 2
        assert cfg.max_steps == 2000
        assert cfg.global_batch_size == 64
        assert cfg.base_model_repo == "nvidia/GR00T-N1.7-3B"
        assert cfg.embodiment_tag == "UNITREE_G1"

    def test_env_overrides(self):
        """Env vars override distributed training defaults."""
        env = {
            "VLA_NUM_NODES": "2",
            "VLA_RDZV_ENDPOINT": "node-0.vla-dist:29500",
            "VLA_LEARNING_RATE": "2.8e-4",
            "VLA_WARMUP_RATIO": "0.1",
            "VLA_STATE_DROPOUT_PROB": "0.5",
            "VLA_DATALOADER_WORKERS": "8",
            "VLA_TRIAL_NAME": "C3",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = DistributedTrainingConfig()
        assert cfg.num_nodes == 2
        assert cfg.rdzv_endpoint == "node-0.vla-dist:29500"
        assert abs(cfg.learning_rate - 2.8e-4) < 1e-8
        assert cfg.warmup_ratio == 0.1
        assert cfg.state_dropout_prob == 0.5
        assert cfg.dataloader_num_workers == 8
        assert cfg.trial_name == "C3"

    def test_base_env_overrides_still_work(self):
        """Base VlaTrainingConfig env vars still work through inheritance."""
        env = {
            "VLA_NUM_GPUS": "4",
            "VLA_MAX_STEPS": "5000",
            "VLA_GLOBAL_BATCH_SIZE": "512",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = DistributedTrainingConfig()
        assert cfg.num_gpus == 4
        assert cfg.max_steps == 5000
        assert cfg.global_batch_size == 512

    def test_invalid_float_env_raises(self):
        """Non-float env var raises ValueError."""
        with mock.patch.dict(os.environ, {"VLA_LEARNING_RATE": "not_a_float"}, clear=True):
            try:
                DistributedTrainingConfig()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "VLA_LEARNING_RATE" in str(e)

    def test_invalid_int_env_raises(self):
        """Non-integer env var for int field raises ValueError."""
        with mock.patch.dict(os.environ, {"VLA_NUM_NODES": "abc"}, clear=True):
            try:
                DistributedTrainingConfig()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "VLA_NUM_NODES" in str(e)

    def test_s3_config_inherited(self):
        """S3 sub-config is accessible through inheritance."""
        env = {"S3_ENDPOINT": "http://minio:9000", "AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = DistributedTrainingConfig()
        assert cfg.s3.enabled
        assert cfg.s3.bucket == "wbc-training"
