# This project was developed with assistance from AI tools.
"""Tests for SONIC ONNX validation logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wbc_pipeline.sonic.validate import validate_onnx_model


def _create_simple_onnx(path: Path, input_dims: int = 10, output_dims: int = 5) -> Path:
    """Create a minimal ONNX model (linear layer) for testing."""
    import onnx
    from onnx import TensorProto, helper

    # Build a simple graph: Y = X @ W + B
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


class TestValidateOnnxModel:
    """Validate ONNX model inspection, inference, and determinism checks."""

    def test_valid_model_passes(self, tmp_path):
        """Valid ONNX model passes all checks."""
        model_path = _create_simple_onnx(tmp_path / "encoder.onnx", input_dims=8, output_dims=4)
        result = validate_onnx_model(model_path)
        assert result["passed"] is True
        assert result["errors"] == []
        assert result["name"] == "encoder.onnx"
        assert len(result["inputs"]) == 1
        assert len(result["outputs"]) == 1

    def test_reports_input_output_shapes(self, tmp_path):
        """Result includes correct input and output shape metadata."""
        model_path = _create_simple_onnx(tmp_path / "decoder.onnx", input_dims=16, output_dims=29)
        result = validate_onnx_model(model_path)
        assert result["inputs"][0]["shape"] == [1, 16]
        assert result["outputs"][0]["shape"] == [1, 29]

    def test_invalid_file_fails(self, tmp_path):
        """Non-ONNX file fails to load."""
        bad_path = tmp_path / "bad.onnx"
        bad_path.write_text("not an onnx model")
        result = validate_onnx_model(bad_path)
        assert result["passed"] is False
        assert any("Failed to load" in e for e in result["errors"])

    def test_multiple_models(self, tmp_path):
        """Each model validated independently."""
        enc = _create_simple_onnx(tmp_path / "encoder.onnx", input_dims=64, output_dims=32)
        dec = _create_simple_onnx(tmp_path / "decoder.onnx", input_dims=32, output_dims=29)
        r_enc = validate_onnx_model(enc)
        r_dec = validate_onnx_model(dec)
        assert r_enc["passed"] is True
        assert r_dec["passed"] is True
        assert r_enc["name"] == "encoder.onnx"
        assert r_dec["name"] == "decoder.onnx"

    def test_determinism_verified(self, tmp_path):
        """Same input produces identical output (determinism check)."""
        model_path = _create_simple_onnx(tmp_path / "model.onnx")
        result = validate_onnx_model(model_path)
        assert result["passed"] is True
        # No non-determinism errors
        assert not any("Non-deterministic" in e for e in result["errors"])
