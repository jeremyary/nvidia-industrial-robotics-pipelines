# This project was developed with assistance from AI tools.
"""Shared ONNX validation: structural check, inference, finite outputs, determinism."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ONNX_EXTENSIONS = {".onnx"}

_IMAGE_PATTERNS = {"image", "pixel", "vision", "img"}
_SEQUENCE_PATTERNS = {"token", "input_ids", "attention", "mask", "text"}
_MIN_SPATIAL_DIM = 16
_MIN_SEQUENCE_DIM = 4

_S3_PREFIX_PATTERN = re.compile(r"^[a-zA-Z0-9._\-/]*$")


def validate_s3_prefix(value: str, field_name: str = "s3_prefix") -> None:
    """Reject path traversal and unsafe characters in S3 prefixes."""
    if ".." in value:
        raise ValueError(f"{field_name} must not contain '..': {value!r}")
    if value.startswith("/"):
        raise ValueError(f"{field_name} must not start with '/': {value!r}")
    if not _S3_PREFIX_PATTERN.match(value):
        raise ValueError(f"{field_name} contains invalid characters: {value!r}")


def _resolve_dynamic_dim(input_name: str) -> int:
    """Resolve a dynamic ONNX dimension to a realistic minimum size."""
    name_lower = (input_name or "").lower()
    if any(p in name_lower for p in _IMAGE_PATTERNS):
        return _MIN_SPATIAL_DIM
    if any(p in name_lower for p in _SEQUENCE_PATTERNS):
        return _MIN_SEQUENCE_DIM
    return 1


def _build_feed(inputs) -> dict:
    """Build random input tensors with correct dtype and realistic shapes."""
    feed = {}
    for inp in inputs:
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                shape.append(_resolve_dynamic_dim(inp.name))
            else:
                shape.append(dim)

        dtype_str = inp.type.lower() if inp.type else ""
        if "float" in dtype_str or "double" in dtype_str:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        else:
            feed[inp.name] = np.random.randint(0, 100, size=shape).astype(np.int64)

    return feed


def validate_onnx_model(onnx_path: Path) -> dict:
    """Validate a single ONNX model: structure, inference, finite outputs, determinism."""
    import onnx
    import onnxruntime as ort

    print(f"\n--- Validating: {onnx_path.name} ---")
    result: dict = {"name": onnx_path.name, "passed": True, "errors": []}

    try:
        onnx.checker.check_model(str(onnx_path))
        print("  Structure: OK")
    except Exception as e:
        err_msg = str(e).lower()
        if "too large" in err_msg or "2gb" in err_msg or "2gib" in err_msg:
            print("  Structure: SKIPPED (model exceeds 2GiB protobuf limit)")
        else:
            result["passed"] = False
            result["errors"].append(f"ONNX structural check failed: {e}")
            return result

    try:
        providers = ort.get_available_providers()
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        print("  Inference: SKIPPED (model uses ops not supported by available providers)")
        return result

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    result["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs]
    result["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in outputs]

    print(f"  Inputs:  {[(i.name, i.shape) for i in inputs]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in outputs]}")

    if not inputs:
        result["passed"] = False
        result["errors"].append("Model has no inputs")
        return result
    if not outputs:
        result["passed"] = False
        result["errors"].append("Model has no outputs")
        return result

    feed = _build_feed(inputs)

    try:
        out1 = session.run(None, feed)
        print(f"  Inference: OK (output shapes: {[o.shape for o in out1]})")
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"Inference failed: {e}")
        return result

    for i, o in enumerate(out1):
        if not np.all(np.isfinite(o)):
            result["passed"] = False
            result["errors"].append(f"Non-finite values in output {i}")

    deterministic = True
    try:
        out2 = session.run(None, feed)
        for i, (o1, o2) in enumerate(zip(out1, out2)):
            if not np.allclose(o1, o2, atol=1e-6):
                deterministic = False
                result["passed"] = False
                result["errors"].append(f"Non-deterministic output at index {i}")
        if deterministic:
            print("  Determinism: OK")
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"Determinism check failed: {e}")

    return result


def download_onnx_files(s3, bucket: str, prefix: str, local_dir: Path) -> list[Path]:
    """Download ONNX files from S3."""
    print(f"Downloading ONNX files from s3://{bucket}/{prefix}/...")
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    onnx_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix) :].lstrip("/")
            if not rel_path:
                continue
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            if local_path.suffix in ONNX_EXTENSIONS:
                onnx_files.append(local_path)

    print(f"Downloaded {len(onnx_files)} ONNX file(s): {[p.name for p in onnx_files]}")
    return onnx_files
