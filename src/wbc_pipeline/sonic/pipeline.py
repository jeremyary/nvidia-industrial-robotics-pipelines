# This project was developed with assistance from AI tools.
"""KFP v2 pipeline: SONIC motion-tracking training for G1 29-DOF humanoid.

Steps: prepare_data → train → export_onnx → validate_onnx → register_model
"""

from kfp import dsl, kubernetes

from wbc_pipeline.constants import MLFLOW_TRACKING_URI, MODEL_REGISTRY_ADDRESS, S3_ENDPOINT

SONIC_IMAGE = "quay.io/jary/isaaclab-g1-sonic:latest"

# Compile-time GPU allocation for training step.
# K8s resource requests must be concrete at compile time.
# The num_gpus pipeline parameter controls accelerate --num_processes inside the container;
# change this constant and recompile if you need a different GPU allocation.
TRAIN_GPU_COUNT = 4


def _configure_gpu_step(task: dsl.PipelineTask, num_gpus: int = 1) -> None:
    """Apply GPU resources, tolerations, secrets, env vars, and pull policy."""
    task.set_accelerator_type("nvidia.com/gpu")
    task.set_accelerator_limit(num_gpus)
    task.set_cpu_request(str(8 * num_gpus))
    task.set_memory_request(f"{64 * num_gpus}Gi")
    task.set_env_variable("ACCEPT_EULA", "Y")
    task.set_env_variable("PYTHONUNBUFFERED", "1")
    task.set_env_variable("S3_ENDPOINT", S3_ENDPOINT)
    task.set_env_variable("S3_BUCKET", "wbc-training")
    task.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    task.set_env_variable("MLFLOW_TRACKING_INSECURE_TLS", "true")
    kubernetes.add_toleration(task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.use_secret_as_env(
        task,
        secret_name="minio-credentials",
        secret_key_to_env={
            "MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID",
            "MINIO_ROOT_PASSWORD": "AWS_SECRET_ACCESS_KEY",
        },
    )
    kubernetes.set_image_pull_policy(task, "Always")


def _configure_cpu_step(task: dsl.PipelineTask) -> None:
    """Apply secrets, env vars, and pull policy to a non-GPU pipeline task."""
    task.set_cpu_request("4")
    task.set_memory_request("16Gi")
    task.set_env_variable("PYTHONUNBUFFERED", "1")
    task.set_env_variable("S3_ENDPOINT", S3_ENDPOINT)
    task.set_env_variable("S3_BUCKET", "wbc-training")
    kubernetes.use_secret_as_env(
        task,
        secret_name="minio-credentials",
        secret_key_to_env={
            "MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID",
            "MINIO_ROOT_PASSWORD": "AWS_SECRET_ACCESS_KEY",
        },
    )
    kubernetes.set_image_pull_policy(task, "Always")


@dsl.container_component
def sonic_prepare_data_op(
    s3_data_prefix: str,
    data_prefix_out: dsl.OutputPath(str),
):
    """Download BONES-SEED from HuggingFace, convert CSV to PKL, upload to S3 (cached)."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=s3_data_prefix, $2=output_file
            "set -euo pipefail\n"
            'S3_DATA_PREFIX="$1"; OUTPUT_FILE="$2"\n'
            "\n"
            'echo "=== SONIC Data Preparation ==="\n'
            'echo "S3 data prefix: ${S3_DATA_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            'export S3_DATA_PREFIX="${S3_DATA_PREFIX}"\n'
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.sonic.data_prep\n"
            "\n"
            'printf "%s" "${S3_DATA_PREFIX}" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC Data Preparation: COMPLETE ==="',
            "--",
            s3_data_prefix,
            data_prefix_out,
        ],
    )


@dsl.container_component
def sonic_train_op(
    data_prefix: str,
    num_gpus: int,
    num_envs: int,
    max_iterations: int,
    hydra_experiment: str,
    checkpoint_prefix: str,
    checkpoint_prefix_out: dsl.OutputPath(str),
):
    """Run SONIC multi-GPU training with Accelerate."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=data_prefix, $2=num_gpus, $3=num_envs,
            # $4=max_iterations, $5=hydra_experiment, $6=checkpoint_prefix, $7=output_file
            "set -euo pipefail\n"
            'S3_DATA_PREFIX="$1"; NUM_GPUS="$2"; NUM_ENVS="$3"\n'
            'MAX_ITERS="$4"; HYDRA_EXP="$5"; S3_CHECKPOINT_PREFIX="$6"\n'
            'OUTPUT_FILE="$7"\n'
            "\n"
            'echo "=== SONIC Training ==="\n'
            'echo "GPUs: ${NUM_GPUS}"\n'
            'echo "Envs: ${NUM_ENVS}"\n'
            'echo "Max iterations: ${MAX_ITERS}"\n'
            'echo "Hydra experiment: ${HYDRA_EXP}"\n'
            'echo "Checkpoint prefix: ${S3_CHECKPOINT_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "export S3_DATA_PREFIX S3_CHECKPOINT_PREFIX\n"
            'export SONIC_NUM_GPUS="${NUM_GPUS}"\n'
            'export SONIC_NUM_ENVS="${NUM_ENVS}"\n'
            'export SONIC_MAX_ITERATIONS="${MAX_ITERS}"\n'
            'export SONIC_HYDRA_EXPERIMENT="${HYDRA_EXP}"\n'
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.sonic.train\n"
            "\n"
            'printf "%s" "${S3_CHECKPOINT_PREFIX}" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC Training: COMPLETE ==="',
            "--",
            data_prefix,
            num_gpus,
            num_envs,
            max_iterations,
            hydra_experiment,
            checkpoint_prefix,
            checkpoint_prefix_out,
        ],
    )


@dsl.container_component
def sonic_export_onnx_op(
    checkpoint_prefix: str,
    hydra_experiment: str,
    onnx_prefix_out: dsl.OutputPath(str),
):
    """Export SONIC encoder/decoder ONNX models from trained checkpoint."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=checkpoint_prefix, $2=hydra_experiment, $3=output_file
            "set -euo pipefail\n"
            'S3_CHECKPOINT_PREFIX="$1"; HYDRA_EXP="$2"; OUTPUT_FILE="$3"\n'
            "\n"
            'echo "=== SONIC ONNX Export ==="\n'
            'echo "Checkpoint prefix: ${S3_CHECKPOINT_PREFIX}"\n'
            'echo "Hydra experiment: ${HYDRA_EXP}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "export S3_CHECKPOINT_PREFIX\n"
            'export SONIC_HYDRA_EXPERIMENT="${HYDRA_EXP}"\n'
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.sonic.export_onnx\n"
            "\n"
            'printf "%s" "${S3_CHECKPOINT_PREFIX}/onnx" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC ONNX Export: COMPLETE ==="',
            "--",
            checkpoint_prefix,
            hydra_experiment,
            onnx_prefix_out,
        ],
    )


@dsl.container_component
def sonic_validate_onnx_op(
    checkpoint_prefix: str,
    validation_result: dsl.OutputPath(str),
):
    """Validate SONIC ONNX models: shapes, inference, determinism."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=checkpoint_prefix, $2=output_file
            "set -euo pipefail\n"
            'S3_CHECKPOINT_PREFIX="$1"; OUTPUT_FILE="$2"\n'
            "\n"
            'echo "=== SONIC ONNX Validation ==="\n'
            'echo "Checkpoint prefix: ${S3_CHECKPOINT_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "export S3_CHECKPOINT_PREFIX\n"
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.sonic.validate\n"
            "\n"
            'echo "PASSED" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC ONNX Validation: COMPLETE ==="',
            "--",
            checkpoint_prefix,
            validation_result,
        ],
    )


@dsl.container_component
def sonic_register_model_op(
    model_name: str,
    model_version: str,
    checkpoint_prefix: str,
    s3_bucket: str,
    registration_result: dsl.OutputPath(str),
):
    """Register validated ONNX model with RHOAI Model Registry."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=model_name, $2=model_version, $3=checkpoint_prefix,
            # $4=s3_bucket, $5=output_file
            "set -euo pipefail\n"
            'MODEL_NAME="$1"; MODEL_VERSION="$2"; CKPT_PREFIX="$3"\n'
            'S3_BUCKET="$4"; OUTPUT_FILE="$5"\n'
            'MODEL_URI="s3://${S3_BUCKET}/${CKPT_PREFIX}/onnx"\n'
            "\n"
            'echo "=== SONIC Model Registration ==="\n'
            'echo "Model: ${MODEL_NAME} v${MODEL_VERSION}"\n'
            'echo "URI: ${MODEL_URI}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.registry \\\n"
            '  --name "${MODEL_NAME}" \\\n'
            '  --uri "${MODEL_URI}" \\\n'
            '  --version "${MODEL_VERSION}" \\\n'
            '  --description "SONIC motion-tracking policy for G1 29-DOF"\n'
            "\n"
            'echo "REGISTERED" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC Model Registration: COMPLETE ==="',
            "--",
            model_name,
            model_version,
            checkpoint_prefix,
            s3_bucket,
            registration_result,
        ],
    )


@dsl.container_component
def sonic_train_pytorchjob_op(
    job_name: str,
    data_prefix: str,
    num_workers: int,
    gpus_per_worker: int,
    num_envs: int,
    max_iterations: int,
    hydra_experiment: str,
    checkpoint_prefix: str,
    queue_name: str,
    checkpoint_prefix_out: dsl.OutputPath(str),
):
    """Launch SONIC training as a multi-node PyTorchJob via Training Operator."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=job_name, $2=data_prefix, $3=num_workers,
            # $4=gpus_per_worker, $5=num_envs, $6=max_iterations,
            # $7=hydra_experiment, $8=checkpoint_prefix, $9=queue_name, ${10}=output_file
            "set -euo pipefail\n"
            'JOB_NAME="$1"; S3_DATA_PREFIX="$2"; NUM_WORKERS="$3"\n'
            'GPUS_PER_WORKER="$4"; NUM_ENVS="$5"; MAX_ITERS="$6"\n'
            'HYDRA_EXP="$7"; S3_CHECKPOINT_PREFIX="$8"; QUEUE_NAME="$9"\n'
            'OUTPUT_FILE="${10}"\n'
            "NAMESPACE=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)\n"
            "\n"
            'echo "=== SONIC PyTorchJob Launcher ==="\n'
            'echo "Job: ${JOB_NAME}"\n'
            'echo "Workers: ${NUM_WORKERS} x ${GPUS_PER_WORKER} GPUs"\n'
            'echo "Envs: ${NUM_ENVS}, Iterations: ${MAX_ITERS}"\n'
            'echo "Namespace: ${NAMESPACE}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "export S3_DATA_PREFIX S3_CHECKPOINT_PREFIX\n"
            'export SONIC_NUM_GPUS="${GPUS_PER_WORKER}"\n'
            'export SONIC_NUM_ENVS="${NUM_ENVS}"\n'
            'export SONIC_MAX_ITERATIONS="${MAX_ITERS}"\n'
            'export SONIC_HYDRA_EXPERIMENT="${HYDRA_EXP}"\n'
            "\n"
            'TRAIN_CMD="/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.sonic.train"\n'
            "\n"
            "QUEUE_ARGS=''\n"
            'if [ -n "${QUEUE_NAME}" ]; then\n'
            '    QUEUE_ARGS="--queue-name ${QUEUE_NAME}"\n'
            "fi\n"
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.pytorchjob \\\n"
            '  --name "${JOB_NAME}" \\\n'
            '  --namespace "${NAMESPACE}" \\\n'
            '  --image "' + SONIC_IMAGE + '" \\\n'
            '  --command "${TRAIN_CMD}" \\\n'
            '  --num-workers "${NUM_WORKERS}" \\\n'
            '  --gpus-per-worker "${GPUS_PER_WORKER}" \\\n'
            "  --service-account-name isaaclab-gpu \\\n"
            "  --run-as-user 0 \\\n"
            "  ${QUEUE_ARGS}\n"
            "\n"
            'printf "%s" "${S3_CHECKPOINT_PREFIX}" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC PyTorchJob Launcher: COMPLETE ==="',
            "--",
            job_name,
            data_prefix,
            num_workers,
            gpus_per_worker,
            num_envs,
            max_iterations,
            hydra_experiment,
            checkpoint_prefix,
            queue_name,
            checkpoint_prefix_out,
        ],
    )


def _wire_post_training_steps(
    train_task,
    hydra_experiment,
    model_name,
    model_version,
    s3_bucket,
):
    """Wire export → validate → register steps after the training step."""
    # Step 3: Export ONNX
    export_task = sonic_export_onnx_op(
        checkpoint_prefix=train_task.outputs["checkpoint_prefix_out"],
        hydra_experiment=hydra_experiment,
    )
    _configure_gpu_step(export_task, num_gpus=1)
    export_task.set_caching_options(False)

    # Step 4: Validate ONNX
    validate_task = sonic_validate_onnx_op(
        checkpoint_prefix=train_task.outputs["checkpoint_prefix_out"],
    )
    validate_task.after(export_task)
    _configure_cpu_step(validate_task)
    validate_task.set_caching_options(False)

    # Step 5: Register model
    register_task = sonic_register_model_op(
        model_name=model_name,
        model_version=model_version,
        checkpoint_prefix=train_task.outputs["checkpoint_prefix_out"],
        s3_bucket=s3_bucket,
    )
    register_task.after(validate_task)
    _configure_cpu_step(register_task)
    register_task.set_cpu_request("1")
    register_task.set_memory_request("2Gi")
    register_task.set_env_variable("MODEL_REGISTRY_ADDRESS", MODEL_REGISTRY_ADDRESS)
    register_task.set_caching_options(False)


@dsl.pipeline(
    name="wbc-sonic-training",
    description="G1 29-DOF SONIC motion-tracking training pipeline",
)
def sonic_training_pipeline(
    num_gpus: int = 4,
    num_envs: int = 4096,
    max_iterations: int = 10000,
    s3_data_prefix: str = "bones-seed/processed",
    s3_checkpoint_prefix: str = "sonic-checkpoints",
    s3_bucket: str = "wbc-training",
    hydra_experiment: str = "sonic_release",
    model_name: str = "g1-sonic-locomotion",
    model_version: str = "v1",
):
    # Step 1: Prepare data (cached — skips if manifest exists)
    data_task = sonic_prepare_data_op(s3_data_prefix=s3_data_prefix)
    _configure_cpu_step(data_task)
    kubernetes.use_secret_as_env(
        data_task,
        secret_name="hf-credentials",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )
    data_task.set_caching_options(False)

    # Step 2: Train with multi-GPU (Tier 1 — single pod)
    train_task = sonic_train_op(
        data_prefix=data_task.outputs["data_prefix_out"],
        num_gpus=num_gpus,
        num_envs=num_envs,
        max_iterations=max_iterations,
        hydra_experiment=hydra_experiment,
        checkpoint_prefix=s3_checkpoint_prefix,
    )
    _configure_gpu_step(train_task, num_gpus=TRAIN_GPU_COUNT)
    kubernetes.set_timeout(train_task, 43200)
    train_task.set_caching_options(False)

    # Steps 3-5: Export → Validate → Register
    _wire_post_training_steps(train_task, hydra_experiment, model_name, model_version, s3_bucket)


@dsl.pipeline(
    name="wbc-sonic-training-distributed",
    description="G1 29-DOF SONIC training via PyTorchJob (multi-node)",
)
def sonic_training_pytorchjob_pipeline(
    job_name: str = "sonic-train",
    num_workers: int = 2,
    gpus_per_worker: int = 1,
    num_envs: int = 4096,
    max_iterations: int = 10000,
    s3_data_prefix: str = "bones-seed/processed",
    s3_checkpoint_prefix: str = "sonic-checkpoints",
    s3_bucket: str = "wbc-training",
    hydra_experiment: str = "sonic_release",
    model_name: str = "g1-sonic-locomotion",
    model_version: str = "v1",
    queue_name: str = "",
):
    # Step 1: Prepare data (cached — skips if manifest exists)
    data_task = sonic_prepare_data_op(s3_data_prefix=s3_data_prefix)
    _configure_cpu_step(data_task)
    kubernetes.use_secret_as_env(
        data_task,
        secret_name="hf-credentials",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )
    data_task.set_caching_options(False)

    # Step 2: Train via PyTorchJob (Tier 2 — multi-node)
    train_task = sonic_train_pytorchjob_op(
        job_name=job_name,
        data_prefix=data_task.outputs["data_prefix_out"],
        num_workers=num_workers,
        gpus_per_worker=gpus_per_worker,
        num_envs=num_envs,
        max_iterations=max_iterations,
        hydra_experiment=hydra_experiment,
        checkpoint_prefix=s3_checkpoint_prefix,
        queue_name=queue_name,
    )
    # Launcher is CPU-only — the PyTorchJob pods get GPUs via Training Operator
    _configure_cpu_step(train_task)
    # Launcher must tolerate GPU taint to schedule on GPU nodes
    kubernetes.add_toleration(train_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    train_task.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    train_task.set_env_variable("MLFLOW_TRACKING_INSECURE_TLS", "true")
    train_task.set_env_variable("ACCEPT_EULA", "Y")
    kubernetes.set_timeout(train_task, 43200)
    train_task.set_caching_options(False)

    # Steps 3-5: Export → Validate → Register
    _wire_post_training_steps(train_task, hydra_experiment, model_name, model_version, s3_bucket)


if __name__ == "__main__":
    import sys

    from kfp import compiler

    output = sys.argv[1] if len(sys.argv) > 1 else "sonic_training_pipeline.yaml"
    compiler.Compiler().compile(sonic_training_pipeline, output)
    print(f"Pipeline compiled to {output}")
