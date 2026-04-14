# This project was developed with assistance from AI tools.
"""BONES-SEED data preparation: HuggingFace download, CSV-to-PKL conversion, S3 upload.

Checks S3 for a manifest before doing work — subsequent runs complete in seconds.

Usage (inside SONIC container):
    python -m wbc_pipeline.sonic.data_prep [--force]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from wbc_pipeline.sonic.config import SonicTrainingConfig

BONES_SEED_HF_REPO = "bones-studio/seed"
MANIFEST_KEY = "manifest.json"
SONIC_ROOT = Path("/workspace/sonic")
CONVERTER_SCRIPT = SONIC_ROOT / "gear_sonic" / "data_process" / "convert_soma_csv_to_motion_lib.py"
PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"


def _manifest_exists(s3, bucket: str, prefix: str) -> dict | None:
    """Check if processed data manifest exists in S3. Returns manifest dict or None."""
    key = f"{prefix}/{MANIFEST_KEY}"
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None


def _download_from_hf(output_dir: Path, hf_token: str) -> None:
    """Download BONES-SEED dataset from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    print(f"Downloading BONES-SEED from {BONES_SEED_HF_REPO}...")
    snapshot_download(
        repo_id=BONES_SEED_HF_REPO,
        repo_type="dataset",
        local_dir=str(output_dir),
        token=hf_token or None,
    )
    print(f"Download complete: {output_dir}")


def _convert_csv_to_pkl(input_dir: Path, output_dir: Path) -> list[str]:
    """Convert SOMA CSV files to motion library PKL format."""
    print(f"Converting CSVs from {input_dir} to PKL...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "motion_lib.pkl"

    cmd = [
        PYTHON,
        str(CONVERTER_SCRIPT),
        "--input",
        str(input_dir),
        "--output",
        str(output_file),
        "--fps",
        "30",
        "--num_workers",
        "8",
    ]
    result = subprocess.run(cmd, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        print(f"ERROR: CSV-to-PKL conversion failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    pkl_files = [str(p.relative_to(output_dir)) for p in output_dir.glob("**/*.pkl")]
    print(f"Conversion complete: {len(pkl_files)} PKL file(s)")
    return pkl_files


def _upload_to_s3(s3, bucket: str, prefix: str, local_dir: Path, pkl_files: list[str]) -> None:
    """Upload PKL files and manifest to S3."""
    for pkl in pkl_files:
        local_path = local_dir / pkl
        s3_key = f"{prefix}/{pkl}"
        print(f"Uploading {pkl} -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_path), bucket, s3_key)


def _write_manifest(s3, bucket: str, prefix: str, pkl_files: list[str], hf_repo: str) -> dict:
    """Write manifest.json to S3 recording what was processed."""
    manifest = {
        "hf_repo": hf_repo,
        "pkl_files": pkl_files,
        "prefix": prefix,
    }
    key = f"{prefix}/{MANIFEST_KEY}"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(manifest, indent=2))
    print(f"Manifest written to s3://{bucket}/{key}")
    return manifest


def run(force: bool = False) -> str:
    """Run data prep. Returns S3 prefix where processed data lives."""
    cfg = SonicTrainingConfig()

    if not cfg.s3.enabled:
        print("ERROR: S3 not configured. Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    s3 = cfg.s3.create_client()
    prefix = cfg.s3.data_prefix

    if not force:
        manifest = _manifest_exists(s3, cfg.s3.bucket, prefix)
        if manifest:
            print(f"Data already prepared at s3://{cfg.s3.bucket}/{prefix} ({len(manifest['pkl_files'])} PKL files)")
            print("Use --force to re-download and re-process.")
            return prefix

    with tempfile.TemporaryDirectory(prefix="bones-seed-") as tmpdir:
        download_dir = Path(tmpdir) / "raw"
        pkl_dir = Path(tmpdir) / "processed"

        _download_from_hf(download_dir, cfg.hf_token)
        pkl_files = _convert_csv_to_pkl(download_dir, pkl_dir)
        if not pkl_files:
            print("ERROR: Conversion produced no PKL files.", file=sys.stderr)
            sys.exit(1)
        _upload_to_s3(s3, cfg.s3.bucket, prefix, pkl_dir, pkl_files)
        _write_manifest(s3, cfg.s3.bucket, prefix, pkl_files, BONES_SEED_HF_REPO)

    print(f"\nData prep complete: s3://{cfg.s3.bucket}/{prefix}")
    return prefix


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BONES-SEED data for SONIC training")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-process even if data exists")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
