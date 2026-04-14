# This project was developed with assistance from AI tools.
"""RHOAI Model Registry integration for WBC training pipelines.

Registers trained ONNX models with the Kubeflow Model Registry,
enabling versioning, lineage tracking, and serving from the RHOAI dashboard.

Usage:
    python -m wbc_pipeline.registry --name MODEL_NAME --uri S3_URI --version VERSION
"""

from __future__ import annotations

import argparse
import os

from wbc_pipeline.constants import MODEL_REGISTRY_ADDRESS as _DEFAULT_REGISTRY_ADDRESS


def _read_sa_token() -> str | None:
    """Read Kubernetes service account token if available."""
    token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return None


def register_model(
    name: str,
    uri: str,
    version: str,
    model_format_name: str = "onnx",
    model_format_version: str = "1",
    description: str | None = None,
    author: str = "wbc-pipeline",
    metadata: dict | None = None,
) -> str:
    """Register a model with the RHOAI Model Registry. Returns the model version ID."""
    from model_registry import ModelRegistry

    server_address = os.environ.get("MODEL_REGISTRY_ADDRESS", _DEFAULT_REGISTRY_ADDRESS)
    is_secure = server_address.startswith("https")
    token = _read_sa_token() if is_secure else None

    registry = ModelRegistry(
        server_address=server_address,
        author=author,
        is_secure=is_secure,
        user_token=token,
    )

    print(f"Registering model: {name} v{version}")
    print(f"  URI: {uri}")
    print(f"  Format: {model_format_name} v{model_format_version}")

    registered_model = registry.register_model(
        name=name,
        uri=uri,
        version=version,
        model_format_name=model_format_name,
        model_format_version=model_format_version,
        version_description=description,
        metadata=metadata or {},
    )

    print(f"  Registered: {registered_model.name} (id={registered_model.id})")
    return registered_model.id


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a model with RHOAI Model Registry")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--uri", required=True, help="Model artifact URI (e.g. s3://bucket/path)")
    parser.add_argument("--version", required=True, help="Model version")
    parser.add_argument("--format-name", default="onnx", help="Model format name")
    parser.add_argument("--format-version", default="1", help="Model format version")
    parser.add_argument("--description", default=None, help="Model description")
    args = parser.parse_args()

    register_model(
        name=args.name,
        uri=args.uri,
        version=args.version,
        model_format_name=args.format_name,
        model_format_version=args.format_version,
        description=args.description,
    )


if __name__ == "__main__":
    main()
