# This project was developed with assistance from AI tools.
"""Tests for shared ONNX validation logic used by VLA and SONIC pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wbc_pipeline.onnx_validation import (
    _build_feed,
    _resolve_dynamic_dim,
    validate_onnx_model,
    validate_s3_prefix,
)


def _create_simple_onnx(path: Path, input_dims: int = 10, output_dims: int = 5) -> Path:
    """Create a minimal ONNX model (linear layer) for testing."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dims])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dims])

    w_data = np.ones(input_dims * output_dims).tolist()
    W = helper.make_tensor("W", TensorProto.FLOAT, [input_dims, output_dims], w_data)
    B = helper.make_tensor("B", TensorProto.FLOAT, [output_dims], np.zeros(output_dims).tolist())

    matmul = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "B"], ["output"])

    graph = helper.make_graph([matmul, add], "test_model", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.save(model, str(path))
    return path


def _create_int_input_onnx(path: Path) -> Path:
    """Create an ONNX model with integer input (simulating token IDs)."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("token_ids", TensorProto.INT64, [1, 8])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    cast = helper.make_node("Cast", ["token_ids"], ["output"], to=TensorProto.FLOAT)

    graph = helper.make_graph([cast], "int_input_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    onnx.save(model, str(path))
    return path


class TestValidateOnnxModel:
    """Validate ONNX model structural check, inference, and determinism."""

    def test_valid_model_passes(self, tmp_path):
        """Valid ONNX model passes all checks."""
        model_path = _create_simple_onnx(tmp_path / "model.onnx")
        result = validate_onnx_model(model_path)
        assert result["passed"] is True
        assert result["errors"] == []
        assert result["name"] == "model.onnx"

    def test_invalid_file_fails_structural_check(self, tmp_path):
        """Non-ONNX file fails structural validation."""
        bad_path = tmp_path / "bad.onnx"
        bad_path.write_text("not an onnx model")
        result = validate_onnx_model(bad_path)
        assert result["passed"] is False
        assert any("structural check failed" in e for e in result["errors"])

    def test_int_input_gets_nonzero_values(self, tmp_path):
        """Integer inputs get randint values, not truncated randn zeros."""
        model_path = _create_int_input_onnx(tmp_path / "int_model.onnx")
        result = validate_onnx_model(model_path)
        assert result["passed"] is True

    def test_determinism_tracked_independently(self, tmp_path):
        """Determinism check reports separately from other failures."""
        model_path = _create_simple_onnx(tmp_path / "model.onnx")
        result = validate_onnx_model(model_path)
        assert not any("Non-deterministic" in e for e in result["errors"])

    def test_reports_io_metadata(self, tmp_path):
        """Result includes input and output shape metadata."""
        model_path = _create_simple_onnx(tmp_path / "model.onnx", input_dims=16, output_dims=29)
        result = validate_onnx_model(model_path)
        assert result["inputs"][0]["shape"] == [1, 16]
        assert result["outputs"][0]["shape"] == [1, 29]


class TestBuildFeed:
    """Test input tensor generation for ONNX validation."""

    def test_float_inputs_use_randn(self):
        """Float-typed inputs get continuous random values."""

        class FakeInput:
            name = "obs"
            shape = [1, 10]
            type = "tensor(float)"

        feed = _build_feed([FakeInput()])
        assert feed["obs"].dtype == np.float32
        assert feed["obs"].shape == (1, 10)

    def test_int_inputs_use_randint(self):
        """Integer-typed inputs get random integers, not truncated floats."""

        class FakeInput:
            name = "token_ids"
            shape = [1, 8]
            type = "tensor(int64)"

        feed = _build_feed([FakeInput()])
        assert feed["token_ids"].dtype == np.int64
        assert np.any(feed["token_ids"] != 0)


class TestResolveDynamicDim:
    """Test dynamic dimension resolution for multimodal model inputs."""

    def test_image_input_gets_spatial_dims(self):
        """Inputs with image-related names get >= 16 spatial dims."""
        assert _resolve_dynamic_dim("pixel_values") >= 16

    def test_token_input_gets_sequence_dims(self):
        """Inputs with token-related names get >= 4 sequence dims."""
        assert _resolve_dynamic_dim("input_ids") >= 4

    def test_unknown_input_gets_1(self):
        """Unrecognized input names default to dim 1."""
        assert _resolve_dynamic_dim("some_other") == 1


class TestValidateS3Prefix:
    """Test S3 prefix validation for path traversal and unsafe characters."""

    def test_valid_prefix_passes(self):
        """Normal prefix with alphanumeric, dots, hyphens, and slashes passes."""
        validate_s3_prefix("checkpoints/v1.0", "test")

    def test_path_traversal_rejected(self):
        """Prefix containing '..' is rejected."""
        import pytest

        with pytest.raises(ValueError, match="must not contain"):
            validate_s3_prefix("../../other-data", "test")

    def test_leading_slash_rejected(self):
        """Prefix starting with '/' is rejected."""
        import pytest

        with pytest.raises(ValueError, match="must not start with"):
            validate_s3_prefix("/absolute/path", "test")

    def test_special_chars_rejected(self):
        """Prefix with shell-unsafe characters is rejected."""
        import pytest

        with pytest.raises(ValueError, match="invalid characters"):
            validate_s3_prefix("prefix;rm -rf /", "test")
