# This project was developed with assistance from AI tools.
"""Custom OnPolicyRunner with MLflow logging, S3 checkpointing, and SIGTERM handling."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
from typing import TYPE_CHECKING

from rsl_rl.runners import OnPolicyRunner

if TYPE_CHECKING:
    from wbc_pipeline.config import TrainingConfig


class _MlflowTensorboardWriter:
    """Wraps a TensorBoard SummaryWriter to also log metrics to MLflow."""

    def __init__(self, tb_writer, experiment_name: str, tracking_uri: str, insecure_tls: bool = False):
        if insecure_tls:
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        # Inject SA token for Kubernetes-auth MLflow servers (RHOAI)
        sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        if os.path.exists(sa_token_path):
            with open(sa_token_path) as f:
                os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

        import mlflow

        # Register workspace header provider for RHOAI MLflow
        sa_ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        if os.path.exists(sa_ns_path):
            with open(sa_ns_path) as f:
                workspace = f.read().strip()
            from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
            from mlflow.tracking.request_header.registry import _request_header_provider_registry

            class _WorkspaceHeaderProvider(RequestHeaderProvider):
                def in_context(self):
                    return True

                def request_headers(self):
                    return {"X-MLFLOW-WORKSPACE": workspace}

            _request_header_provider_registry.register(_WorkspaceHeaderProvider)

        self._tb = tb_writer
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()
        self._mlflow = mlflow

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        self._tb.add_scalar(tag, scalar_value, global_step, **kwargs)
        step = int(global_step) if global_step is not None else 0
        try:
            self._mlflow.log_metric(tag.replace("/", "_"), float(scalar_value), step=step)
        except Exception:
            pass

    def stop(self):
        self._tb.stop() if hasattr(self._tb, "stop") else None
        self._mlflow.end_run()

    def __getattr__(self, name):
        return getattr(self._tb, name)


class WBCRunner(OnPolicyRunner):
    """OnPolicyRunner with MLflow metrics, S3 checkpoint upload, and SIGTERM handling."""

    def __init__(self, env, cfg: dict, log_dir: str, device: str, training_cfg: TrainingConfig):
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self._training_cfg = training_cfg
        self._s3_client = None
        self._shutting_down = False

        if training_cfg.s3.enabled:
            self._init_s3()

        if training_cfg.checkpoint_interval:
            self.save_interval = training_cfg.checkpoint_interval

        signal.signal(signal.SIGTERM, self._sigterm_handler)

    def _init_s3(self):
        import boto3

        s3_cfg = self._training_cfg.s3
        self._s3_client = boto3.client(
            "s3",
            endpoint_url=s3_cfg.endpoint,
            aws_access_key_id=s3_cfg.access_key,
            aws_secret_access_key=s3_cfg.secret_key,
        )
        print(f"[WBCRunner] S3 checkpointing enabled: s3://{s3_cfg.bucket}/{s3_cfg.prefix}")

    def _prepare_logging_writer(self):
        """Create tensorboard writer, then wrap with MLflow if configured."""
        super()._prepare_logging_writer()
        if self._training_cfg.mlflow.enabled:
            mlflow_cfg = self._training_cfg.mlflow
            try:
                self.writer = _MlflowTensorboardWriter(
                    self.writer,
                    experiment_name=mlflow_cfg.experiment_name,
                    tracking_uri=mlflow_cfg.tracking_uri,
                    insecure_tls=mlflow_cfg.insecure_tls,
                )
                print(f"[WBCRunner] MLflow tracking enabled: {mlflow_cfg.tracking_uri}")
            except Exception as e:
                print(f"[WBCRunner] WARNING: MLflow init failed, continuing without it: {e}")

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        if self._s3_client is not None:
            self._upload_to_s3(path)

    def _upload_to_s3(self, local_path: str):
        s3_cfg = self._training_cfg.s3
        filename = os.path.basename(local_path)
        s3_key = f"{s3_cfg.prefix}/{filename}"
        try:
            self._s3_client.upload_file(local_path, s3_cfg.bucket, s3_key)
            print(f"[WBCRunner] Uploaded checkpoint to s3://{s3_cfg.bucket}/{s3_key}")
        except Exception as e:
            print(f"[WBCRunner] WARNING: S3 upload failed for {filename}: {e}", file=sys.stderr)

    def _sigterm_handler(self, signum, frame):
        if self._shutting_down:
            return
        self._shutting_down = True
        print("[WBCRunner] SIGTERM received — saving final checkpoint...")
        path = os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
        self.save(path)
        if isinstance(self.writer, _MlflowTensorboardWriter):
            self.writer.stop()
        print("[WBCRunner] Graceful shutdown complete.")
        sys.exit(0)

    def resume_from_s3(self, s3_key: str) -> dict:
        """Download a checkpoint from S3 and load it."""
        s3_cfg = self._training_cfg.s3
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            local_path = f.name
        try:
            self._s3_client.download_file(s3_cfg.bucket, s3_key, local_path)
            print(f"[WBCRunner] Downloaded checkpoint from s3://{s3_cfg.bucket}/{s3_key}")
            return self.load(local_path)
        finally:
            os.unlink(local_path)

    def resume_latest_from_s3(self) -> dict | None:
        """Find and load the latest checkpoint from S3."""
        if self._s3_client is None:
            return None
        s3_cfg = self._training_cfg.s3
        try:
            response = self._s3_client.list_objects_v2(Bucket=s3_cfg.bucket, Prefix=s3_cfg.prefix + "/")
            if "Contents" not in response:
                print("[WBCRunner] No checkpoints found in S3.")
                return None
            model_files = [obj for obj in response["Contents"] if obj["Key"].endswith(".pt")]
            if not model_files:
                print("[WBCRunner] No .pt checkpoints found in S3.")
                return None
            latest = max(model_files, key=lambda obj: obj["LastModified"])
            return self.resume_from_s3(latest["Key"])
        except Exception as e:
            print(f"[WBCRunner] WARNING: Failed to list S3 checkpoints: {e}", file=sys.stderr)
            return None
