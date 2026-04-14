# This project was developed with assistance from AI tools.
"""Validation utilities for WBC pipeline infrastructure (S3, MLflow)."""

from __future__ import annotations

import argparse
import os


def check_s3() -> None:
    """Verify S3 checkpoints exist in the configured bucket."""
    from wbc_pipeline.config import S3Config

    cfg = S3Config()
    s3 = cfg.create_client()
    bucket = cfg.bucket
    prefix = cfg.prefix
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/")
    files = [obj["Key"] for obj in resp.get("Contents", [])]
    print(f"S3 checkpoints: {files}")
    assert any(f.endswith(".pt") for f in files), "No .pt checkpoint found in S3"
    print("S3 checkpoint verification PASSED")


def check_mlflow() -> None:
    """Verify MLflow experiment and runs exist."""
    os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")

    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

    import mlflow
    from mlflow.tracking.request_header.abstract_request_header_provider import (
        RequestHeaderProvider,
    )
    from mlflow.tracking.request_header.registry import (
        _request_header_provider_registry,
    )

    ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(ns_path):
        with open(ns_path) as f:
            workspace = f.read().strip()

        class _WorkspaceHeaderProvider(RequestHeaderProvider):
            def in_context(self):
                return True

            def request_headers(self):
                return {"X-MLFLOW-WORKSPACE": workspace}

        _request_header_provider_registry.register(_WorkspaceHeaderProvider)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.get_experiment_by_name("g1-29dof-locomotion")
    assert experiment is not None, "MLflow experiment not found"
    print(f"MLflow experiment: {experiment.name} (id={experiment.experiment_id})")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0, "No MLflow runs found"
    print(f"MLflow runs: {len(runs)}")

    cols = [c for c in runs.columns if "Mean_reward" in c or "Total_timesteps" in c]
    print(f"Logged metrics include: {cols}")
    print("MLflow verification PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WBC pipeline infrastructure.")
    parser.add_argument("check", choices=["s3", "mlflow", "all"], help="Which check to run")
    args = parser.parse_args()

    if args.check in ("s3", "all"):
        check_s3()
    if args.check in ("mlflow", "all"):
        check_mlflow()


if __name__ == "__main__":
    main()
