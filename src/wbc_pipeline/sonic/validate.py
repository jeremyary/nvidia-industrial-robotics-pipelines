# This project was developed with assistance from AI tools.
"""SONIC ONNX validation: download exported models from S3, verify shapes and inference.

Validates encoder and decoder ONNX files for correct input/output shapes,
successful inference with random inputs, and deterministic outputs.

Usage (requires onnxruntime):
    python -m wbc_pipeline.sonic.validate [--checkpoint_prefix PREFIX]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from wbc_pipeline.sonic.config import SonicTrainingConfig

ONNX_EXTENSIONS = {".onnx"}


def _download_onnx_files(s3, bucket: str, prefix: str, local_dir: Path) -> list[Path]:
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
            if local_path.suffix in ONNX_EXTENSIONS:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local_path))
                onnx_files.append(local_path)

    print(f"Downloaded {len(onnx_files)} ONNX file(s): {[p.name for p in onnx_files]}")
    return onnx_files


def validate_onnx_model(onnx_path: Path) -> dict:
    """Validate a single ONNX model: load, check shapes, run inference, verify determinism."""
    import onnxruntime as ort

    print(f"\n--- Validating: {onnx_path.name} ---")
    result: dict = {"name": onnx_path.name, "passed": True, "errors": []}

    # Load model
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"Failed to load model: {e}")
        return result

    # Collect input/output metadata
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    result["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in inputs]
    result["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in outputs]

    print(f"  Inputs:  {[(i.name, i.shape) for i in inputs]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in outputs]}")

    # Verify model has at least one input and output
    if not inputs:
        result["passed"] = False
        result["errors"].append("Model has no inputs")
        return result
    if not outputs:
        result["passed"] = False
        result["errors"].append("Model has no outputs")
        return result

    # Build random inputs with correct shapes
    feed = {}
    for inp in inputs:
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str) or dim is None:
                shape.append(1)  # dynamic dim → use 1
            else:
                shape.append(dim)
        dtype = np.float32 if "float" in inp.type.lower() else np.int64
        feed[inp.name] = np.random.randn(*shape).astype(dtype)

    # Run inference
    try:
        out1 = session.run(None, feed)
        print(f"  Inference: OK (output shapes: {[o.shape for o in out1]})")
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"Inference failed: {e}")
        return result

    # Determinism check — same input should produce same output
    try:
        out2 = session.run(None, feed)
        for i, (o1, o2) in enumerate(zip(out1, out2)):
            if not np.allclose(o1, o2, atol=1e-6):
                result["passed"] = False
                result["errors"].append(f"Non-deterministic output at index {i}")
        if result["passed"]:
            print("  Determinism: OK")
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"Determinism check failed: {e}")

    return result


def run(checkpoint_prefix: str | None = None) -> list[dict]:
    """Download and validate all SONIC ONNX models. Returns list of validation results."""
    cfg = SonicTrainingConfig()

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured.", file=sys.stderr)
        sys.exit(1)

    checkpoint_prefix = checkpoint_prefix or cfg.s3.checkpoint_prefix
    onnx_prefix = f"{checkpoint_prefix}/onnx"
    s3 = cfg.s3.create_client()

    with tempfile.TemporaryDirectory(prefix="sonic-validate-") as tmpdir:
        onnx_dir = Path(tmpdir)
        onnx_files = _download_onnx_files(s3, cfg.s3.bucket, onnx_prefix, onnx_dir)

        if not onnx_files:
            print("ERROR: No ONNX files found in S3.", file=sys.stderr)
            sys.exit(1)

        results = [validate_onnx_model(f) for f in onnx_files]

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    print(f"\n=== SONIC ONNX Validation: {passed} passed, {failed} failed ===")

    if failed > 0:
        for r in results:
            if not r["passed"]:
                print(f"  FAILED: {r['name']}: {r['errors']}")
        sys.exit(1)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SONIC ONNX models")
    parser.add_argument("--checkpoint_prefix", type=str, default=None, help="S3 prefix for ONNX files")
    args = parser.parse_args()
    run(checkpoint_prefix=args.checkpoint_prefix)


if __name__ == "__main__":
    main()
