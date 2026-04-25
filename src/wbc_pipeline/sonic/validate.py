# This project was developed with assistance from AI tools.
"""SONIC ONNX validation: download exported models from S3, verify shapes and inference.

Validates encoder and decoder ONNX files for correct input/output shapes,
successful inference with random inputs, and deterministic outputs.

Usage (requires onnxruntime):
    python -m wbc_pipeline.sonic.validate [--checkpoint-prefix PREFIX]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from wbc_pipeline.onnx_validation import download_onnx_files, validate_onnx_model
from wbc_pipeline.sonic.config import SonicTrainingConfig

EXPECTED_MIN_ONNX_FILES = 3


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
        onnx_files = download_onnx_files(s3, cfg.s3.bucket, onnx_prefix, onnx_dir)

        if not onnx_files:
            print("ERROR: No ONNX files found in S3.", file=sys.stderr)
            sys.exit(1)

        if len(onnx_files) < EXPECTED_MIN_ONNX_FILES:
            print(
                f"WARNING: Expected at least {EXPECTED_MIN_ONNX_FILES} ONNX files "
                f"(encoder, decoder, planner), got {len(onnx_files)}",
                file=sys.stderr,
            )

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
    parser.add_argument("--checkpoint-prefix", type=str, default=None, help="S3 prefix for ONNX files")
    args = parser.parse_args()
    run(checkpoint_prefix=args.checkpoint_prefix)


if __name__ == "__main__":
    main()
