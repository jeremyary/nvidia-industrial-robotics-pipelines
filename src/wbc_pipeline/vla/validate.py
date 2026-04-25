# This project was developed with assistance from AI tools.
"""VLA ONNX validation: download exported model from S3, verify shapes and inference.

Uses dynamic I/O shape detection since VLA models have multimodal inputs
(unlike the fixed 103→29 WBC policy).

Usage (requires onnxruntime):
    python -m wbc_pipeline.vla.validate [--checkpoint-prefix PREFIX]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from wbc_pipeline.onnx_validation import download_onnx_files, validate_onnx_model
from wbc_pipeline.vla.config import VlaTrainingConfig


def run(checkpoint_prefix: str | None = None) -> list[dict]:
    """Download and validate all VLA ONNX models."""
    cfg = VlaTrainingConfig()

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured.", file=sys.stderr)
        sys.exit(1)

    checkpoint_prefix = checkpoint_prefix or cfg.s3.checkpoint_prefix
    onnx_prefix = f"{checkpoint_prefix}/onnx"
    s3 = cfg.s3.create_client()

    with tempfile.TemporaryDirectory(prefix="vla-validate-") as tmpdir:
        onnx_dir = Path(tmpdir)
        onnx_files = download_onnx_files(s3, cfg.s3.bucket, onnx_prefix, onnx_dir)

        if not onnx_files:
            print("ERROR: No ONNX files found in S3.", file=sys.stderr)
            sys.exit(1)

        results = [validate_onnx_model(f) for f in onnx_files]

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    print(f"\n=== VLA ONNX Validation: {passed} passed, {failed} failed ===")

    if failed > 0:
        for r in results:
            if not r["passed"]:
                print(f"  FAILED: {r['name']}: {r['errors']}")
        sys.exit(1)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VLA ONNX models")
    parser.add_argument("--checkpoint-prefix", type=str, default=None, help="S3 prefix for ONNX files")
    args = parser.parse_args()
    run(checkpoint_prefix=args.checkpoint_prefix)


if __name__ == "__main__":
    main()
