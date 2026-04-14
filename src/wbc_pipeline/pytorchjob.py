# This project was developed with assistance from AI tools.
"""PyTorchJob launcher: build, submit, and wait for multi-node distributed training.

Backend-agnostic — works with any container image (RSL-RL, SONIC, etc.).
Designed to run inside a KFP pipeline step as a CPU-only launcher pod.

Usage (inside KFP container step):
    python -m wbc_pipeline.pytorchjob \
        --name sonic-train-001 \
        --namespace wbc-training \
        --image quay.io/jary/isaaclab-g1-sonic:latest \
        --command '/bin/bash -c "accelerate launch ..."' \
        --num-workers 2 \
        --gpus-per-worker 1 \
        --timeout 43200
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

JOB_KIND = "PyTorchJob"
DEFAULT_TIMEOUT = 43200  # 12 hours
DEFAULT_POLL_INTERVAL = 60  # seconds


def build_pytorchjob(
    name: str,
    namespace: str,
    image: str,
    command: list[str],
    args: list[str] | None = None,
    num_workers: int = 1,
    gpus_per_worker: int = 1,
    cpu_per_worker: str = "8",
    memory_per_worker: str = "64Gi",
    env_vars: dict[str, str] | None = None,
    secret_refs: dict[str, dict[str, str]] | None = None,
    queue_name: str | None = None,
    labels: dict[str, str] | None = None,
    service_account_name: str | None = None,
    run_as_user: int | None = None,
):
    """Build a PyTorchJob CR for multi-node distributed training."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")

    from kubeflow.training.models import (
        KubeflowOrgV1PyTorchJob,
        KubeflowOrgV1PyTorchJobSpec,
        KubeflowOrgV1ReplicaSpec,
        KubeflowOrgV1RunPolicy,
    )
    from kubernetes.client import (
        V1Container,
        V1EnvVar,
        V1EnvVarSource,
        V1ObjectMeta,
        V1PodSpec,
        V1PodTemplateSpec,
        V1ResourceRequirements,
        V1SecretKeySelector,
        V1SecurityContext,
        V1Toleration,
    )

    env_list = []
    for k, v in (env_vars or {}).items():
        env_list.append(V1EnvVar(name=k, value=v))
    for secret_name, key_map in (secret_refs or {}).items():
        for secret_key, env_name in key_map.items():
            env_list.append(
                V1EnvVar(
                    name=env_name,
                    value_from=V1EnvVarSource(
                        secret_key_ref=V1SecretKeySelector(name=secret_name, key=secret_key),
                    ),
                )
            )

    security_context = V1SecurityContext(run_as_user=run_as_user) if run_as_user is not None else None

    container = V1Container(
        name="pytorch",
        image=image,
        image_pull_policy="Always",
        command=command,
        args=args,
        env=env_list or None,
        resources=V1ResourceRequirements(
            requests={"cpu": cpu_per_worker, "memory": memory_per_worker, "nvidia.com/gpu": str(gpus_per_worker)},
            limits={"nvidia.com/gpu": str(gpus_per_worker)},
        ),
        security_context=security_context,
    )

    tolerations = [V1Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")]

    pod_spec = V1PodSpec(
        containers=[container],
        tolerations=tolerations,
        restart_policy="Never",
        service_account_name=service_account_name,
    )
    template = V1PodTemplateSpec(spec=pod_spec)

    # Master gets 1 replica; workers get num_workers - 1 (master is also a worker)
    replica_specs = {
        "Master": KubeflowOrgV1ReplicaSpec(replicas=1, restart_policy="Never", template=template),
    }
    if num_workers > 1:
        replica_specs["Worker"] = KubeflowOrgV1ReplicaSpec(
            replicas=num_workers - 1,
            restart_policy="Never",
            template=template,
        )

    job_labels = {"app": "wbc-training", "training-backend": "pytorchjob"}
    if labels:
        job_labels.update(labels)

    metadata = V1ObjectMeta(name=name, namespace=namespace, labels=job_labels)
    if queue_name:
        metadata.labels["kueue.x-k8s.io/queue-name"] = queue_name

    spec = KubeflowOrgV1PyTorchJobSpec(
        pytorch_replica_specs=replica_specs,
        run_policy=KubeflowOrgV1RunPolicy(clean_pod_policy="None"),
    )

    return KubeflowOrgV1PyTorchJob(
        api_version="kubeflow.org/v1",
        kind=JOB_KIND,
        metadata=metadata,
        spec=spec,
    )


def submit_and_wait(
    job,
    namespace: str,
    wait_timeout: int = DEFAULT_TIMEOUT,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> bool:
    """Submit a PyTorchJob and poll until Succeeded or Failed."""
    from kubeflow.training import TrainingClient

    client = TrainingClient()
    job_name = job.metadata.name

    # Register cleanup handler for graceful cancellation
    def _cleanup(signum, frame):
        print(f"[PyTorchJob] Signal {signum} received — deleting job {job_name}...")
        try:
            client.delete_job(job_name, namespace=namespace, job_kind=JOB_KIND)
            print(f"[PyTorchJob] Job {job_name} deleted.")
        except Exception as e:
            print(f"[PyTorchJob] WARNING: Failed to delete job: {e}", file=sys.stderr)
        sys.exit(128 + signum)

    print(f"[PyTorchJob] Submitting {job_name} in namespace {namespace}...")
    client.create_job(job=job, namespace=namespace)
    signal.signal(signal.SIGTERM, _cleanup)
    print(f"[PyTorchJob] Job submitted. Polling every {poll_interval}s (timeout: {wait_timeout}s)...")

    try:
        client.wait_for_job_conditions(
            job_name,
            namespace=namespace,
            job_kind=JOB_KIND,
            expected_conditions={"Succeeded", "Failed"},
            wait_timeout=wait_timeout,
            polling_interval=poll_interval,
            timeout=poll_interval + 30,
        )
    except Exception as e:
        print(f"[PyTorchJob] ERROR: Wait failed: {e}", file=sys.stderr)
        return False

    if client.is_job_succeeded(job_name, namespace=namespace, job_kind=JOB_KIND):
        print(f"[PyTorchJob] Job {job_name} SUCCEEDED.")
        return True

    print(f"[PyTorchJob] Job {job_name} FAILED.", file=sys.stderr)
    conditions = client.get_job_conditions(job_name, namespace=namespace, job_kind=JOB_KIND)
    for c in conditions:
        print(f"  Condition: type={c.type}, status={c.status}, reason={c.reason}", file=sys.stderr)
    return False


def run() -> None:
    """CLI entry point — parse args, build job, submit, wait."""
    parser = argparse.ArgumentParser(description="Submit and wait for a PyTorchJob")
    parser.add_argument("--name", required=True, help="Job name")
    parser.add_argument("--namespace", default=os.environ.get("POD_NAMESPACE", "wbc-training"))
    parser.add_argument("--image", required=True, help="Training container image")
    parser.add_argument("--command", required=True, help="Shell command string for training")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpus-per-worker", type=int, default=1)
    parser.add_argument("--cpu-per-worker", default="8")
    parser.add_argument("--memory-per-worker", default="64Gi")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--queue-name", default=None, help="Kueue LocalQueue name")
    parser.add_argument("--service-account-name", default=None, help="K8s ServiceAccount for training pods")
    parser.add_argument("--run-as-user", type=int, default=None, help="UID for container security context")
    args = parser.parse_args()

    # Env vars to propagate to training pods (read from launcher env)
    env_vars = {}
    for key in [
        "S3_ENDPOINT",
        "S3_BUCKET",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_INSECURE_TLS",
        "ACCEPT_EULA",
        "PYTHONUNBUFFERED",
        "S3_PREFIX",
        "S3_DATA_PREFIX",
        "S3_CHECKPOINT_PREFIX",
        "SONIC_NUM_GPUS",
        "SONIC_NUM_ENVS",
        "SONIC_MAX_ITERATIONS",
        "SONIC_HYDRA_EXPERIMENT",
        "CHECKPOINT_INTERVAL",
        "TASK",
        "NUM_ENVS",
        "MAX_ITERS",
        "ONNX_DIR",
        "RESUME",
    ]:
        val = os.environ.get(key)
        if val is not None:
            env_vars[key] = val

    secret_refs = {
        "minio-credentials": {"MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID", "MINIO_ROOT_PASSWORD": "AWS_SECRET_ACCESS_KEY"},
    }
    # Only mount HF credentials if present in launcher env
    if os.environ.get("HF_TOKEN"):
        secret_refs["hf-credentials"] = {"HF_TOKEN": "HF_TOKEN"}

    job = build_pytorchjob(
        name=args.name,
        namespace=args.namespace,
        image=args.image,
        command=["/bin/bash", "-c", args.command],
        num_workers=args.num_workers,
        gpus_per_worker=args.gpus_per_worker,
        cpu_per_worker=args.cpu_per_worker,
        memory_per_worker=args.memory_per_worker,
        env_vars=env_vars,
        secret_refs=secret_refs,
        queue_name=args.queue_name,
        service_account_name=args.service_account_name,
        run_as_user=args.run_as_user,
    )

    succeeded = submit_and_wait(
        job,
        namespace=args.namespace,
        wait_timeout=args.timeout,
        poll_interval=args.poll_interval,
    )

    if not succeeded:
        sys.exit(1)


if __name__ == "__main__":
    run()
