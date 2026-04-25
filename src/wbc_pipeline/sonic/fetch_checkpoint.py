# This project was developed with assistance from AI tools.
"""Download GEAR-SONIC ONNX checkpoint from HuggingFace and upload to S3.

Caches by checking if the S3 prefix already contains ONNX files.
Use --force to re-download regardless.

Usage:
    python -m wbc_pipeline.sonic.fetch_checkpoint [--repo-id REPO] [--s3-prefix PREFIX] [--force]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "nvidia/GEAR-SONIC"
ALLOW_PATTERNS = ["*.onnx", "observation_config.yaml"]
DEFAULT_S3_PREFIX = "gear-sonic"


def _s3_has_onnx(s3, bucket: str, prefix: str) -> bool:
    """Check if ONNX files already exist under the S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/onnx/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".onnx"):
                return True
    return False


def fetch_and_upload(
    repo_id: str = REPO_ID,
    s3_prefix: str = DEFAULT_S3_PREFIX,
    force: bool = False,
    revision: str | None = None,
) -> list[str]:
    """Download ONNX models from HuggingFace Hub and upload to S3."""
    from huggingface_hub import snapshot_download

    from wbc_pipeline.sonic.config import SonicTrainingConfig

    cfg = SonicTrainingConfig()
    if not cfg.s3.enabled:
        print("ERROR: S3 not configured. Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    s3 = cfg.s3.create_client()
    bucket = cfg.s3.bucket
    onnx_prefix = f"{s3_prefix}/onnx"

    if not force and _s3_has_onnx(s3, bucket, s3_prefix):
        print(f"SONIC checkpoint already cached at s3://{bucket}/{onnx_prefix}/")
        print("Skipping download. Use --force to re-download.")
        return []

    hf_token = os.environ.get("HF_TOKEN")
    print(f"Downloading from HuggingFace: {repo_id}")
    print(f"  Patterns: {ALLOW_PATTERNS}")

    download_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=ALLOW_PATTERNS,
        token=hf_token,
        revision=revision,
    )

    uploaded: list[str] = []
    download_path = Path(download_dir)
    for local_file in sorted(download_path.rglob("*")):
        if not local_file.is_file():
            continue
        if local_file.suffix not in {".onnx", ".yaml"}:
            continue
        s3_key = f"{onnx_prefix}/{local_file.name}"
        size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_file.name} ({size_mb:.1f} MB) -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_file), bucket, s3_key)
        uploaded.append(s3_key)

    print(f"\nUploaded {len(uploaded)} files to s3://{bucket}/{onnx_prefix}/")
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GEAR-SONIC checkpoint from HuggingFace")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    parser.add_argument("--revision", default=None, help="HuggingFace commit SHA or tag for reproducibility")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached in S3")
    args = parser.parse_args()

    result = fetch_and_upload(repo_id=args.repo_id, s3_prefix=args.s3_prefix, force=args.force, revision=args.revision)
    if not result:
        print("No files uploaded (already cached or no files found).")
    sys.exit(0)


if __name__ == "__main__":
    main()
