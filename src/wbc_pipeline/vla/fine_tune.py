# This project was developed with assistance from AI tools.
"""VLA fine-tuning entrypoint: run GR00T fine-tuning, export ONNX, upload to S3.

Usage (inside VLA container):
    python -m wbc_pipeline.vla.fine_tune [--max-steps N] [--num-gpus N]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from wbc_pipeline.vla.config import VlaTrainingConfig

GROOT_ROOT = Path("/workspace/Isaac-GR00T")
FINETUNE_SCRIPT = GROOT_ROOT / "gr00t" / "experiment" / "launch_finetune.py"
EXPORT_SCRIPT = GROOT_ROOT / "scripts" / "deployment" / "build_trt_pipeline.py"
MODALITY_CONFIG = Path(__file__).parent / "g1_teleop_modality.py"
OUTPUT_DIR = Path("/tmp/vla-output")
DATASET_DIR = Path("/tmp/vla-dataset")
ONNX_DIR = Path("/tmp/vla-onnx")


def _upload_artifacts_to_s3(s3, bucket: str, prefix: str, local_dir: Path) -> list[str]:
    """Upload fine-tuned checkpoint and ONNX files to S3."""
    uploaded: list[str] = []
    for local_path in sorted(local_dir.rglob("*")):
        if not local_path.is_file():
            continue
        rel_path = local_path.relative_to(local_dir)
        s3_key = f"{prefix}/{rel_path}"
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {rel_path} ({size_mb:.1f} MB) -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_path), bucket, s3_key)
        uploaded.append(s3_key)
    return uploaded


def _download_from_s3(s3, bucket: str, prefix: str, local_dir: Path) -> None:
    """Download all files under an S3 prefix to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix) :].lstrip("/")
            if not rel_path:
                continue
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  s3://{bucket}/{key} -> {local_path}")
            s3.download_file(bucket, key, str(local_path))


def run(
    max_steps: int | None = None,
    num_gpus: int | None = None,
    global_batch_size: int | None = None,
) -> None:
    """Run GR00T fine-tuning, ONNX export, and S3 upload."""
    cfg = VlaTrainingConfig()

    max_steps = max_steps if max_steps is not None else cfg.max_steps
    num_gpus = num_gpus if num_gpus is not None else cfg.num_gpus
    global_batch_size = global_batch_size if global_batch_size is not None else cfg.global_batch_size

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured. Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    s3 = cfg.s3.create_client()

    # Download base model from S3
    base_model_dir = Path("/tmp/base-model")
    print(f"\n=== Downloading base model from S3 ({cfg.s3.model_prefix}) ===")
    _download_from_s3(s3, cfg.s3.bucket, cfg.s3.model_prefix, base_model_dir)

    # Download dataset from S3
    dataset_prefix = cfg.s3.dataset_prefix
    print(f"\n=== Downloading dataset from S3 ({dataset_prefix}) ===")
    _download_from_s3(s3, cfg.s3.bucket, dataset_prefix, DATASET_DIR)

    # GR00T expects --dataset-path to point at the task subdirectory containing meta/info.json
    dataset_path = DATASET_DIR
    task_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()]
    if task_dirs:
        dataset_path = task_dirs[0]
        print(f"Using task subdirectory: {dataset_path}")
    elif not (DATASET_DIR / "meta" / "info.json").exists():
        print("ERROR: No meta/info.json found in dataset directory or subdirectories.", file=sys.stderr)
        sys.exit(1)

    # Configure MLflow via env vars
    env = os.environ.copy()
    if cfg.mlflow.enabled:
        env["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
        env["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
        if cfg.mlflow.insecure_tls:
            env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

    print("\n=== VLA Fine-Tuning ===")
    print(f"Base model: {base_model_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Embodiment: {cfg.embodiment_tag}")
    print(f"GPUs: {num_gpus}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {global_batch_size}")
    print(f"MLflow: {'enabled' if cfg.mlflow.enabled else 'disabled'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        str(FINETUNE_SCRIPT),
        "--base-model-path",
        str(base_model_dir),
        "--dataset-path",
        str(dataset_path),
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--modality-config-path",
        str(MODALITY_CONFIG),
        "--output-dir",
        str(OUTPUT_DIR),
        "--max-steps",
        str(max_steps),
        "--global-batch-size",
        str(global_batch_size),
    ]

    print(f"Command: {' '.join(train_cmd)}\n")
    subprocess.run(train_cmd, env=env, cwd=str(GROOT_ROOT), stdin=subprocess.DEVNULL, check=True)

    # ONNX export (N1.7: build_trt_pipeline.py with --steps export for ONNX-only)
    print("\n=== ONNX Export ===")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        "python",
        str(EXPORT_SCRIPT),
        "--model-path",
        str(OUTPUT_DIR),
        "--output-dir",
        str(ONNX_DIR),
        "--export-mode",
        "full_pipeline",
        "--precision",
        "bf16",
        "--steps",
        "export",
        "--embodiment-tag",
        "NEW_EMBODIMENT",
        "--dataset-path",
        str(dataset_path),
    ]

    print(f"Command: {' '.join(export_cmd)}\n")
    subprocess.run(export_cmd, env=env, cwd=str(GROOT_ROOT), stdin=subprocess.DEVNULL, check=True)

    # Upload ONNX artifacts to S3 (checkpoint upload skipped — only ONNX needed for validation + registration)
    checkpoint_prefix = cfg.s3.checkpoint_prefix
    print(f"\n=== Uploading ONNX to S3 ({checkpoint_prefix}/onnx/) ===")
    _upload_artifacts_to_s3(s3, cfg.s3.bucket, f"{checkpoint_prefix}/onnx", ONNX_DIR)

    print("\n=== VLA Fine-Tuning: COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GR00T N1.7 fine-tuning + ONNX export")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    args = parser.parse_args()
    run(max_steps=args.max_steps, num_gpus=args.num_gpus, global_batch_size=args.global_batch_size)


if __name__ == "__main__":
    main()
