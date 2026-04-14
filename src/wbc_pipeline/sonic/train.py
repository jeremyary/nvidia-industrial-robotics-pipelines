# This project was developed with assistance from AI tools.
"""SONIC training entrypoint: run Accelerate training and upload checkpoints to S3.

Usage (inside SONIC container):
    python -m wbc_pipeline.sonic.train [--num_gpus N] [--num_envs N] [--max_iterations N]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from wbc_pipeline.sonic.config import SonicTrainingConfig

SONIC_ROOT = Path("/workspace/sonic")
TRAIN_SCRIPT = SONIC_ROOT / "gear_sonic" / "train_agent_trl.py"
PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"


def _upload_checkpoints_to_s3(s3, bucket: str, prefix: str, experiment_dir: Path) -> None:
    """Upload training checkpoints and logs from experiment dir to S3."""
    if not experiment_dir.exists():
        print(f"WARNING: Experiment dir {experiment_dir} not found, skipping checkpoint upload.")
        return

    print(f"Uploading checkpoints from {experiment_dir} to s3://{bucket}/{prefix}/...")
    for local_path in experiment_dir.rglob("*"):
        if local_path.is_file():
            rel_path = local_path.relative_to(experiment_dir)
            s3_key = f"{prefix}/{rel_path}"
            print(f"  {rel_path} -> s3://{bucket}/{s3_key}")
            s3.upload_file(str(local_path), bucket, s3_key)
    print("Checkpoint upload complete.")


def _find_experiment_dir(base_dir: Path) -> Path | None:
    """Find the most recently created experiment directory under logs_rl/."""
    logs_dir = base_dir / "logs_rl"
    if not logs_dir.exists():
        return None
    exp_dirs = sorted(logs_dir.rglob("last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if exp_dirs:
        return exp_dirs[0].parent
    return None


def run(
    num_gpus: int | None = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    hydra_experiment: str | None = None,
) -> None:
    """Run SONIC training with Accelerate."""
    cfg = SonicTrainingConfig()

    num_gpus = num_gpus if num_gpus is not None else cfg.num_gpus
    num_envs = num_envs if num_envs is not None else cfg.num_envs
    max_iterations = max_iterations if max_iterations is not None else cfg.max_iterations
    hydra_experiment = hydra_experiment if hydra_experiment is not None else cfg.hydra_experiment

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured. Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    s3 = cfg.s3.create_client()

    # Configure MLflow via env vars (HuggingFace reads these automatically with report_to="mlflow")
    env = os.environ.copy()
    if cfg.mlflow.enabled:
        env["MLFLOW_TRACKING_URI"] = cfg.mlflow.tracking_uri
        env["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
        if cfg.mlflow.insecure_tls:
            env["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

    print("\n=== SONIC Training ===")
    print(f"GPUs: {num_gpus}")
    print(f"Envs: {num_envs}")
    print(f"Max iterations: {max_iterations}")
    print(f"Hydra experiment: {hydra_experiment}")
    print(f"MLflow: {'enabled' if cfg.mlflow.enabled else 'disabled'}")

    cmd = [
        PYTHON,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_gpus),
        str(TRAIN_SCRIPT),
        f"+exp=manager/universal_token/all_modes/{hydra_experiment}",
        f"num_envs={num_envs}",
        "headless=True",
        f"max_iterations={max_iterations}",
    ]

    if cfg.mlflow.enabled:
        cmd.append("report_to=mlflow")

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env, cwd=str(SONIC_ROOT))
    if result.returncode != 0:
        print(f"ERROR: Training failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # Upload checkpoints to S3
    experiment_dir = _find_experiment_dir(SONIC_ROOT)
    if experiment_dir and cfg.s3.enabled:
        _upload_checkpoints_to_s3(s3, cfg.s3.bucket, cfg.s3.checkpoint_prefix, experiment_dir)

    print("\n=== SONIC Training: COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SONIC training with Accelerate")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs (default: from env/config)")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum training iterations")
    parser.add_argument("--hydra_experiment", type=str, default=None, help="Hydra experiment config name")
    args = parser.parse_args()
    run(
        num_gpus=args.num_gpus,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        hydra_experiment=args.hydra_experiment,
    )


if __name__ == "__main__":
    main()
