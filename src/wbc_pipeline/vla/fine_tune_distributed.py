# This project was developed with assistance from AI tools.
"""Distributed VLA fine-tuning: multi-node torchrun with advanced hyperparameters.

Extends the single-node fine_tune.py with c10d rendezvous for multi-node
training and exposes additional GR00T hyperparameters for scaling trials.

Usage (inside VLA container):
    python -m wbc_pipeline.vla.fine_tune_distributed [--num-nodes 2] [--rdzv-endpoint host:29500]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from wbc_pipeline.vla.distributed_config import DistributedTrainingConfig
from wbc_pipeline.vla.fine_tune import (
    DATASET_DIR,
    EXPORT_SCRIPT,
    FINETUNE_SCRIPT,
    GROOT_ROOT,
    MODALITY_CONFIG,
    ONNX_DIR,
    OUTPUT_DIR,
    _download_from_s3,
    _upload_artifacts_to_s3,
)


def run(
    max_steps: int | None = None,
    num_gpus: int | None = None,
    global_batch_size: int | None = None,
    num_nodes: int | None = None,
    rdzv_endpoint: str | None = None,
    learning_rate: float | None = None,
    warmup_ratio: float | None = None,
    state_dropout_prob: float | None = None,
    trial_name: str | None = None,
    skip_export: bool = False,
    rank_zero_only: bool = True,
) -> None:
    """Run distributed GR00T fine-tuning with advanced hyperparameters."""
    cfg = DistributedTrainingConfig()

    max_steps = max_steps if max_steps is not None else cfg.max_steps
    num_gpus = num_gpus if num_gpus is not None else cfg.num_gpus
    global_batch_size = global_batch_size if global_batch_size is not None else cfg.global_batch_size
    num_nodes = num_nodes if num_nodes is not None else cfg.num_nodes
    rdzv_endpoint = rdzv_endpoint if rdzv_endpoint is not None else cfg.rdzv_endpoint
    learning_rate = learning_rate if learning_rate is not None else cfg.learning_rate
    warmup_ratio = warmup_ratio if warmup_ratio is not None else cfg.warmup_ratio
    state_dropout_prob = state_dropout_prob if state_dropout_prob is not None else cfg.state_dropout_prob
    trial_name = trial_name if trial_name is not None else cfg.trial_name

    is_rank_zero = True
    if num_nodes > 1 and rank_zero_only:
        node_rank = os.environ.get("NODE_RANK", "0")
        is_rank_zero = node_rank == "0"

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured. Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    s3 = cfg.s3.create_client()

    # Only rank 0 downloads data (other ranks wait for torchrun sync)
    base_model_dir = Path("/tmp/base-model")
    if is_rank_zero:
        print(f"\n=== Downloading base model from S3 ({cfg.s3.model_prefix}) ===")
        _download_from_s3(s3, cfg.s3.bucket, cfg.s3.model_prefix, base_model_dir)

        print(f"\n=== Downloading dataset from S3 ({cfg.s3.dataset_prefix}) ===")
        _download_from_s3(s3, cfg.s3.bucket, cfg.s3.dataset_prefix, DATASET_DIR)
    else:
        print("\n=== Worker node — skipping S3 download (rank-zero-only) ===")
        base_model_dir.mkdir(parents=True, exist_ok=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATASET_DIR
    task_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()]
    if task_dirs:
        dataset_path = task_dirs[0]
        print(f"Using task subdirectory: {dataset_path}")
    elif not (DATASET_DIR / "meta" / "info.json").exists() and is_rank_zero:
        print("ERROR: No meta/info.json found in dataset directory or subdirectories.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    if cfg.mlflow.enabled:
        env["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
        env["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
        if cfg.mlflow.insecure_tls:
            env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

    total_gpus = num_gpus * num_nodes
    print("\n=== Distributed VLA Fine-Tuning ===")
    print(f"Base model: {base_model_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Embodiment: {cfg.embodiment_tag}")
    print(f"Nodes: {num_nodes} x {num_gpus} GPUs = {total_gpus} total")
    print(f"Max steps: {max_steps}")
    print(f"Global batch size: {global_batch_size} ({global_batch_size // total_gpus}/GPU)")
    print(f"Learning rate: {learning_rate}")
    print(f"Warmup ratio: {warmup_ratio}")
    print(f"State dropout: {state_dropout_prob}")
    print(f"Trial: {trial_name or '(default)'}")
    print(f"MLflow: {'enabled' if cfg.mlflow.enabled else 'disabled'}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build torchrun command
    if num_nodes > 1:
        if not rdzv_endpoint:
            print("ERROR: --rdzv-endpoint required for multi-node training.", file=sys.stderr)
            sys.exit(1)
        torchrun_args = [
            "torchrun",
            f"--nnodes={num_nodes}",
            f"--nproc_per_node={num_gpus}",
            "--rdzv-backend=c10d",
            f"--rdzv-endpoint={rdzv_endpoint}",
        ]
    else:
        torchrun_args = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
        ]

    train_cmd = [
        *torchrun_args,
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
        "--learning-rate",
        str(learning_rate),
        "--warmup-ratio",
        str(warmup_ratio),
        "--state-dropout-prob",
        str(state_dropout_prob),
        "--dataloader-num-workers",
        str(cfg.dataloader_num_workers),
    ]

    print(f"Command: {' '.join(train_cmd)}\n")

    t0 = time.monotonic()
    subprocess.run(train_cmd, env=env, cwd=str(GROOT_ROOT), stdin=subprocess.DEVNULL, check=True)
    elapsed = time.monotonic() - t0

    steps_per_sec = max_steps / elapsed if elapsed > 0 else 0
    print(f"\nTraining wall-clock: {elapsed:.1f}s ({steps_per_sec:.3f} steps/s)")

    if skip_export:
        print("\n=== Skipping ONNX export (--skip-export) ===")
    elif is_rank_zero:
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

        checkpoint_prefix = cfg.s3.checkpoint_prefix
        if trial_name:
            s3_onnx_prefix = f"{checkpoint_prefix}/{trial_name}/onnx"
        else:
            s3_onnx_prefix = f"{checkpoint_prefix}/onnx"
        print(f"\n=== Uploading ONNX to S3 ({s3_onnx_prefix}) ===")
        _upload_artifacts_to_s3(s3, cfg.s3.bucket, s3_onnx_prefix, ONNX_DIR)

    print("\n=== Distributed VLA Fine-Tuning: COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributed GR00T N1.7 fine-tuning")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--rdzv-endpoint", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--state-dropout-prob", type=float, default=None)
    parser.add_argument("--trial-name", type=str, default=None)
    parser.add_argument("--skip-export", action="store_true", help="Skip ONNX export (throughput trials)")
    parser.add_argument("--rank-zero-only", type=str, default="true", help="Only rank 0 does S3/ONNX")
    args = parser.parse_args()

    rank_zero_only = args.rank_zero_only.lower() in ("true", "1", "yes")

    run(
        max_steps=args.max_steps,
        num_gpus=args.num_gpus,
        global_batch_size=args.global_batch_size,
        num_nodes=args.num_nodes,
        rdzv_endpoint=args.rdzv_endpoint,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        state_dropout_prob=args.state_dropout_prob,
        trial_name=args.trial_name,
        skip_export=args.skip_export,
        rank_zero_only=rank_zero_only,
    )


if __name__ == "__main__":
    main()
