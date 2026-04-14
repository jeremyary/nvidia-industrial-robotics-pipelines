# This project was developed with assistance from AI tools.
"""SONIC ONNX export: load trained checkpoint, export encoder/decoder ONNX models, upload to S3.

Uses SONIC's eval_agent_trl.py with export_onnx_only=True to run the export.

Usage (inside SONIC container):
    python -m wbc_pipeline.sonic.export_onnx [--checkpoint_prefix PREFIX]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wbc_pipeline.sonic.config import SonicTrainingConfig

SONIC_ROOT = Path("/workspace/sonic")
EVAL_SCRIPT = SONIC_ROOT / "gear_sonic" / "eval_agent_trl.py"
PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"
ONNX_EXTENSIONS = {".onnx", ".yaml"}


def _download_checkpoint(s3, bucket: str, prefix: str, local_dir: Path) -> Path:
    """Download trained checkpoint from S3."""
    print(f"Downloading checkpoint from s3://{bucket}/{prefix}/...")
    local_dir.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix) :].lstrip("/")
            if not rel_path:
                continue
            local_path = local_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))

    checkpoint_path = local_dir / "last.pt"
    if not checkpoint_path.exists():
        pt_files = list(local_dir.rglob("*.pt"))
        if pt_files:
            checkpoint_path = pt_files[0]
        else:
            print(f"ERROR: No .pt checkpoint found in s3://{bucket}/{prefix}/", file=sys.stderr)
            sys.exit(1)

    print(f"Checkpoint downloaded: {checkpoint_path}")
    return checkpoint_path


def _run_export(checkpoint_dir: Path, output_dir: Path, hydra_experiment: str) -> list[Path]:
    """Run SONIC's eval script with export_onnx_only=True."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX from {checkpoint_dir}...")
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        f"+exp=manager/universal_token/all_modes/{hydra_experiment}",
        "headless=True",
        "num_envs=1",
        "export_onnx_only=True",
        f"checkpoint={checkpoint_dir}",
        f"export_dir={output_dir}",
    ]

    result = subprocess.run(cmd, cwd=str(SONIC_ROOT))
    if result.returncode != 0:
        print(f"ERROR: ONNX export failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    exported = [p for p in output_dir.rglob("*") if p.is_file() and p.suffix in ONNX_EXTENSIONS]
    print(f"Exported {len(exported)} file(s): {[p.name for p in exported]}")
    return exported


def _upload_onnx_to_s3(s3, bucket: str, prefix: str, output_dir: Path, files: list[Path]) -> list[str]:
    """Upload exported ONNX files to S3."""
    s3_keys = []
    for f in files:
        rel = f.relative_to(output_dir)
        key = f"{prefix}/{rel}"
        print(f"Uploading {rel} -> s3://{bucket}/{key}")
        s3.upload_file(str(f), bucket, key)
        s3_keys.append(key)
    return s3_keys


def run(checkpoint_prefix: str | None = None, hydra_experiment: str | None = None) -> list[str]:
    """Export SONIC model to ONNX and upload to S3. Returns list of S3 keys."""
    cfg = SonicTrainingConfig()

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured.", file=sys.stderr)
        sys.exit(1)

    checkpoint_prefix = checkpoint_prefix or cfg.s3.checkpoint_prefix
    hydra_experiment = hydra_experiment or cfg.hydra_experiment
    s3 = cfg.s3.create_client()

    import tempfile

    with tempfile.TemporaryDirectory(prefix="sonic-export-") as tmpdir:
        tmppath = Path(tmpdir)
        ckpt_dir = tmppath / "checkpoint"
        onnx_dir = tmppath / "onnx"

        _download_checkpoint(s3, cfg.s3.bucket, checkpoint_prefix, ckpt_dir)
        exported = _run_export(ckpt_dir, onnx_dir, hydra_experiment)

        if not exported:
            print("ERROR: No ONNX files exported.", file=sys.stderr)
            sys.exit(1)

        onnx_prefix = f"{checkpoint_prefix}/onnx"
        s3_keys = _upload_onnx_to_s3(s3, cfg.s3.bucket, onnx_prefix, onnx_dir, exported)

    print(f"\n=== SONIC ONNX Export: COMPLETE ({len(s3_keys)} files) ===")
    return s3_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SONIC model to ONNX")
    parser.add_argument("--checkpoint_prefix", type=str, default=None, help="S3 prefix for trained checkpoint")
    parser.add_argument("--hydra_experiment", type=str, default=None, help="Hydra experiment config name")
    args = parser.parse_args()
    run(checkpoint_prefix=args.checkpoint_prefix, hydra_experiment=args.hydra_experiment)


if __name__ == "__main__":
    main()
