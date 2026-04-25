# This project was developed with assistance from AI tools.
"""Download GR00T N1.7-3B base model and training dataset from HuggingFace, cache in S3.

Checks S3 for existing files before downloading.
Use --force to re-download regardless.

Usage:
    python -m wbc_pipeline.vla.data_prep [--force]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _s3_has_files(s3, bucket: str, prefix: str, extensions: tuple[str, ...]) -> bool:
    """Check if files with given extensions exist under the S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            if any(obj["Key"].endswith(ext) for ext in extensions):
                return True
    return False


def _download_and_upload(s3, bucket: str, repo_id: str, s3_prefix: str, revision: str | None) -> list[str]:
    """Download a HuggingFace repo and upload all files to S3."""
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    print(f"Downloading from HuggingFace: {repo_id}")

    download_dir = snapshot_download(
        repo_id=repo_id,
        token=hf_token,
        revision=revision,
        repo_type="dataset" if "/" in repo_id and "GR00T-N1" not in repo_id else "model",
    )

    uploaded: list[str] = []
    download_path = Path(download_dir)
    for local_file in sorted(download_path.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(download_path)
        s3_key = f"{s3_prefix}/{relative}"
        size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"  Uploading {relative} ({size_mb:.1f} MB) -> s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_file), bucket, s3_key)
        uploaded.append(s3_key)

    print(f"\nUploaded {len(uploaded)} files to s3://{bucket}/{s3_prefix}/")
    return uploaded


def fetch_and_upload(
    force: bool = False,
    revision: str | None = None,
) -> None:
    """Download base model and dataset from HuggingFace Hub and upload to S3."""
    from wbc_pipeline.vla.config import VlaTrainingConfig

    cfg = VlaTrainingConfig()
    if not cfg.s3.enabled:
        print("ERROR: S3 not configured. Set S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.", file=sys.stderr)
        sys.exit(1)

    s3 = cfg.s3.create_client()
    bucket = cfg.s3.bucket

    # Download base model
    model_prefix = cfg.s3.model_prefix
    if not force and _s3_has_files(s3, bucket, model_prefix, (".safetensors", "config.json")):
        print(f"Base model already cached at s3://{bucket}/{model_prefix}/")
    else:
        print(f"\n=== Downloading base model: {cfg.base_model_repo} ===")
        _download_and_upload(s3, bucket, cfg.base_model_repo, model_prefix, revision)

    # Download dataset
    dataset_prefix = cfg.s3.dataset_prefix
    if not force and _s3_has_files(s3, bucket, dataset_prefix, (".parquet", ".mp4", ".json")):
        print(f"Dataset already cached at s3://{bucket}/{dataset_prefix}/")
    else:
        print(f"\n=== Downloading dataset: {cfg.dataset_repo} ===")
        _download_and_upload(s3, bucket, cfg.dataset_repo, dataset_prefix, revision)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GR00T base model and dataset from HuggingFace")
    parser.add_argument("--revision", default=None, help="HuggingFace commit SHA or tag for reproducibility")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached in S3")
    args = parser.parse_args()

    fetch_and_upload(force=args.force, revision=args.revision)


if __name__ == "__main__":
    main()
