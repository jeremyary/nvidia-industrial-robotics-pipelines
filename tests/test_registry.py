# This project was developed with assistance from AI tools.
"""Tests for RHOAI Model Registry integration."""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

from wbc_pipeline.registry import _read_sa_token, register_model


class _FakeRegisteredModel:
    """Minimal mock for the return value of ModelRegistry.register_model."""

    def __init__(self, name: str, model_id: str = "model-123"):
        self.name = name
        self.id = model_id


def _mock_model_registry():
    """Create a mock model_registry module with a ModelRegistry class."""
    fake_module = types.ModuleType("model_registry")
    fake_module.ModelRegistry = mock.MagicMock()
    return fake_module


class TestReadSaToken:
    """Validate SA token reading."""

    def test_returns_none_when_no_file(self, tmp_path):
        """Returns None when token file does not exist."""
        with mock.patch("wbc_pipeline.registry.os.path.exists", return_value=False):
            assert _read_sa_token() is None

    def test_reads_token_from_file(self, tmp_path):
        """Reads and strips token from SA token file."""
        token_file = tmp_path / "token"
        token_file.write_text("  my-token-value  \n")
        with mock.patch("builtins.open", mock.mock_open(read_data="  my-token-value  \n")):
            with mock.patch("wbc_pipeline.registry.os.path.exists", return_value=True):
                assert _read_sa_token() == "my-token-value"


class TestRegisterModel:
    """Validate model registration logic."""

    def test_registers_model(self):
        """Registers model with correct parameters."""
        fake_mod = _mock_model_registry()
        mock_registry = fake_mod.ModelRegistry.return_value
        mock_registry.register_model.return_value = _FakeRegisteredModel("test-model")

        env = {"MODEL_REGISTRY_ADDRESS": "http://localhost:8080"}
        with (
            mock.patch.dict(sys.modules, {"model_registry": fake_mod}),
            mock.patch("wbc_pipeline.registry._read_sa_token", return_value=None),
            mock.patch.dict(os.environ, env, clear=False),
        ):
            result = register_model(
                name="test-model",
                uri="s3://bucket/model.onnx",
                version="v1",
            )

        assert result == "model-123"
        fake_mod.ModelRegistry.assert_called_once_with(
            server_address="http://localhost:8080",
            author="wbc-pipeline",
            is_secure=False,
            user_token=None,
        )
        mock_registry.register_model.assert_called_once_with(
            name="test-model",
            uri="s3://bucket/model.onnx",
            version="v1",
            model_format_name="onnx",
            model_format_version="1",
            version_description=None,
            metadata={},
        )

    def test_passes_sa_token(self):
        """Passes SA token to ModelRegistry constructor."""
        fake_mod = _mock_model_registry()
        mock_registry = fake_mod.ModelRegistry.return_value
        mock_registry.register_model.return_value = _FakeRegisteredModel("m")

        env = {"MODEL_REGISTRY_ADDRESS": "https://registry.svc:443"}
        with (
            mock.patch.dict(sys.modules, {"model_registry": fake_mod}),
            mock.patch("wbc_pipeline.registry._read_sa_token", return_value="sa-token"),
            mock.patch.dict(os.environ, env, clear=False),
        ):
            register_model(name="m", uri="s3://b/m.onnx", version="v1")

        fake_mod.ModelRegistry.assert_called_once_with(
            server_address="https://registry.svc:443",
            author="wbc-pipeline",
            is_secure=True,
            user_token="sa-token",
        )

    def test_custom_metadata(self):
        """Passes custom metadata to register_model."""
        fake_mod = _mock_model_registry()
        mock_registry = fake_mod.ModelRegistry.return_value
        mock_registry.register_model.return_value = _FakeRegisteredModel("m")

        env = {"MODEL_REGISTRY_ADDRESS": "http://localhost:8080"}
        with (
            mock.patch.dict(sys.modules, {"model_registry": fake_mod}),
            mock.patch("wbc_pipeline.registry._read_sa_token", return_value=None),
            mock.patch.dict(os.environ, env, clear=False),
        ):
            register_model(
                name="m",
                uri="s3://b/m.onnx",
                version="v1",
                metadata={"obs_dim": 103, "action_dim": 29},
            )

        call_kwargs = mock_registry.register_model.call_args[1]
        assert call_kwargs["metadata"] == {"obs_dim": 103, "action_dim": 29}

    def test_default_server_address(self):
        """Uses default server address when env var not set."""
        fake_mod = _mock_model_registry()
        mock_registry = fake_mod.ModelRegistry.return_value
        mock_registry.register_model.return_value = _FakeRegisteredModel("m")

        with (
            mock.patch.dict(sys.modules, {"model_registry": fake_mod}),
            mock.patch("wbc_pipeline.registry._read_sa_token", return_value=None),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("MODEL_REGISTRY_ADDRESS", None)
            register_model(name="m", uri="s3://b/m.onnx", version="v1")

        call_kwargs = fake_mod.ModelRegistry.call_args[1]
        assert "wbc-model-registry" in call_kwargs["server_address"]
        assert call_kwargs["is_secure"] is False
