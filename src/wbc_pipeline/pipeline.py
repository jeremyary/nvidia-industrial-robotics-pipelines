# This project was developed with assistance from AI tools.
"""KFP v2 pipeline: train G1 29-DOF locomotion policy, export ONNX, validate, register."""

from kfp import dsl, kubernetes

from wbc_pipeline.constants import MLFLOW_TRACKING_URI, MODEL_REGISTRY_ADDRESS, S3_ENDPOINT

DEFAULT_IMAGE = "quay.io/jary/isaaclab-g1-train:latest"


def _configure_gpu_step(task: dsl.PipelineTask) -> None:
    """Apply GPU resources, tolerations, secrets, env vars, and pull policy to a pipeline task."""
    task.set_accelerator_type("nvidia.com/gpu")
    task.set_accelerator_limit(1)
    task.set_cpu_request("8")
    task.set_memory_request("64Gi")
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
        secret_key_to_env={"MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID", "MINIO_ROOT_PASSWORD": "AWS_SECRET_ACCESS_KEY"},
    )
    kubernetes.set_image_pull_policy(task, "Always")


def _configure_cpu_step(task: dsl.PipelineTask) -> None:
    """Apply secrets, env vars, and pull policy to a non-GPU pipeline task."""
    task.set_cpu_request("2")
    task.set_memory_request("8Gi")
    task.set_env_variable("PYTHONUNBUFFERED", "1")
    task.set_env_variable("S3_ENDPOINT", S3_ENDPOINT)
    kubernetes.use_secret_as_env(
        task,
        secret_name="minio-credentials",
        secret_key_to_env={"MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID", "MINIO_ROOT_PASSWORD": "AWS_SECRET_ACCESS_KEY"},
    )
    kubernetes.set_image_pull_policy(task, "Always")


@dsl.container_component
def train_and_export_op(
    task: str,
    num_envs: int,
    max_iterations: int,
    checkpoint_interval: int,
    s3_prefix: str,
    onnx_s3_key: dsl.OutputPath(str),
):
    """Train G1 locomotion policy and export ONNX in a single Isaac Sim session."""
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=task, $2=num_envs, $3=max_iterations,
            # $4=checkpoint_interval, $5=s3_prefix, $6=output_file
            "set -euo pipefail\n"
            'export TASK="$1" NUM_ENVS="$2" MAX_ITERS="$3"\n'
            'export CHECKPOINT_INTERVAL="$4" S3_PREFIX="$5"\n'
            'OUTPUT_FILE="$6"\n'
            'export ONNX_DIR="/tmp/onnx_export"\n'
            'export RESUME="s3"\n'
            'mkdir -p "${ONNX_DIR}"\n'
            "\n"
            'echo "=== WBC Training Pipeline ==="\n'
            'echo "Task: ${TASK}"\n'
            'echo "Envs: ${NUM_ENVS}"\n'
            'echo "Iterations: ${MAX_ITERS}"\n'
            'echo "S3 prefix: ${S3_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "/workspace/isaaclab/isaaclab.sh -p -m wbc_pipeline.train_and_export\n"
            "\n"
            'printf "%s" "${S3_PREFIX}/policy.onnx" > "${OUTPUT_FILE}"\n'
            'echo ""\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== Training Pipeline: COMPLETE ==="',
            "--",
            task,
            num_envs,
            max_iterations,
            checkpoint_interval,
            s3_prefix,
            onnx_s3_key,
        ],
    )


@dsl.container_component
def validate_onnx_op(
    onnx_s3_key: str,
    expected_obs_dim: int,
    expected_action_dim: int,
    validation_result: dsl.OutputPath(str),
):
    """Download ONNX from S3 and validate shape, inference, and normalizer presence."""
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=onnx_s3_key, $2=obs_dim, $3=action_dim, $4=output_file
            # All user-supplied values are exported as env vars and read with
            # os.environ inside the Python -c string to prevent shell injection.
            "set -euo pipefail\n"
            'ONNX_KEY=$(echo -n "$1"); OBS_DIM="$2"; ACT_DIM="$3"; OUTPUT_FILE="$4"\n'
            'ONNX_PATH="/tmp/policy.onnx"\n'
            "export ONNX_KEY OBS_DIM ACT_DIM ONNX_PATH\n"
            "\n"
            'echo "=== ONNX Validation ==="\n'
            'echo "S3 key: ${ONNX_KEY}"\n'
            'echo "Expected obs_dim: ${OBS_DIM}, action_dim: ${ACT_DIM}"\n'
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -c '\n"
            "import boto3, os, sys\n"
            "import numpy as np\n"
            'onnx_key = os.environ["ONNX_KEY"]\n'
            'obs_dim = int(os.environ["OBS_DIM"])\n'
            'act_dim = int(os.environ["ACT_DIM"])\n'
            'onnx_path = os.environ["ONNX_PATH"]\n'
            "\n"
            's3 = boto3.client("s3",\n'
            '    endpoint_url=os.environ["S3_ENDPOINT"],\n'
            '    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],\n'
            '    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])\n'
            's3.download_file("wbc-training", onnx_key, onnx_path)\n'
            'print(f"Downloaded s3://wbc-training/{onnx_key}")\n'
            "\n"
            "import onnx\n"
            "import onnxruntime as ort\n"
            "\n"
            "model = onnx.load(onnx_path)\n"
            "onnx.checker.check_model(model)\n"
            "\n"
            "inp = model.graph.input[0]\n"
            "out = model.graph.output[0]\n"
            "in_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]\n"
            "out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]\n"
            'print(f"Input: name={inp.name!r}, shape={in_shape}")\n'
            'print(f"Output: name={out.name!r}, shape={out_shape}")\n'
            "\n"
            'assert in_shape == [1, obs_dim], f"Expected input [1, {obs_dim}], got {in_shape}"\n'
            'assert out_shape == [1, act_dim], f"Expected output [1, {act_dim}], got {out_shape}"\n'
            "\n"
            "session = ort.InferenceSession(onnx_path)\n"
            "obs = np.random.randn(1, obs_dim).astype(np.float32)\n"
            "actions = session.run(None, {inp.name: obs})[0]\n"
            'assert actions.shape == (1, act_dim), f"ORT shape mismatch: {actions.shape}"\n'
            'assert np.all(np.isfinite(actions)), "Non-finite outputs detected"\n'
            "\n"
            "actions2 = session.run(None, {inp.name: obs})[0]\n"
            'assert np.array_equal(actions, actions2), "Non-deterministic outputs"\n'
            "\n"
            "op_types = {n.op_type for n in model.graph.node}\n"
            'has_norm = bool(op_types & {"Sub", "BatchNormalization", "Div"})\n'
            'print(f"Normalizer ops present: {has_norm}")\n'
            "if not has_norm:\n"
            '    print("WARNING: No normalization ops found in ONNX graph")\n'
            "\n"
            "print()\n"
            'print("=== ONNX Validation PASSED ===")\n'
            'print(f"  Input: {in_shape}, Output: {out_shape}")\n'
            'print(f"  Deterministic: yes")\n'
            'norm_str = "baked in" if has_norm else "MISSING"\n'
            'print(f"  Normalizer: {norm_str}")\n'
            "'\n"
            "\n"
            'echo "PASSED" > "${OUTPUT_FILE}"\n'
            'echo "=== ONNX Validation COMPLETE ==="',
            "--",
            onnx_s3_key,
            expected_obs_dim,
            expected_action_dim,
            validation_result,
        ],
    )


@dsl.container_component
def register_model_op(
    model_name: str,
    model_version: str,
    onnx_s3_key: str,
    s3_bucket: str,
    registration_result: dsl.OutputPath(str),
):
    """Register validated ONNX model with RHOAI Model Registry."""
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=model_name, $2=model_version, $3=onnx_s3_key,
            # $4=s3_bucket, $5=output_file
            "set -euo pipefail\n"
            'MODEL_NAME="$1"; MODEL_VERSION="$2"; ONNX_KEY="$3"\n'
            'S3_BUCKET="$4"; OUTPUT_FILE="$5"\n'
            'MODEL_URI="s3://${S3_BUCKET}/${ONNX_KEY}"\n'
            "\n"
            'echo "=== Model Registration ==="\n'
            'echo "Model: ${MODEL_NAME} v${MODEL_VERSION}"\n'
            'echo "URI: ${MODEL_URI}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.registry \\\n"
            '  --name "${MODEL_NAME}" \\\n'
            '  --uri "${MODEL_URI}" \\\n'
            '  --version "${MODEL_VERSION}" \\\n'
            '  --description "RSL-RL velocity-tracking policy for G1 29-DOF"\n'
            "\n"
            'echo "REGISTERED" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== Model Registration: COMPLETE ==="',
            "--",
            model_name,
            model_version,
            onnx_s3_key,
            s3_bucket,
            registration_result,
        ],
    )


@dsl.pipeline(name="wbc-training", description="G1 29-DOF WBC locomotion training pipeline")
def wbc_training_pipeline(
    task: str = "WBC-Velocity-Flat-G1-29DOF-v0",
    num_envs: int = 4096,
    max_iterations: int = 6000,
    checkpoint_interval: int = 100,
    s3_prefix: str = "checkpoints",
    s3_bucket: str = "wbc-training",
    expected_obs_dim: int = 103,
    expected_action_dim: int = 29,
    model_name: str = "g1-wbc-flat-policy",
    model_version: str = "v1",
):
    train_task = train_and_export_op(
        task=task,
        num_envs=num_envs,
        max_iterations=max_iterations,
        checkpoint_interval=checkpoint_interval,
        s3_prefix=s3_prefix,
    )
    _configure_gpu_step(train_task)
    kubernetes.set_timeout(train_task, 28800)
    train_task.set_caching_options(False)

    validate_task = validate_onnx_op(
        onnx_s3_key=train_task.outputs["onnx_s3_key"],
        expected_obs_dim=expected_obs_dim,
        expected_action_dim=expected_action_dim,
    )
    _configure_cpu_step(validate_task)
    validate_task.set_caching_options(False)

    register_task = register_model_op(
        model_name=model_name,
        model_version=model_version,
        onnx_s3_key=train_task.outputs["onnx_s3_key"],
        s3_bucket=s3_bucket,
    )
    register_task.after(validate_task)
    _configure_cpu_step(register_task)
    register_task.set_cpu_request("1")
    register_task.set_memory_request("2Gi")
    register_task.set_env_variable("MODEL_REGISTRY_ADDRESS", MODEL_REGISTRY_ADDRESS)
    register_task.set_caching_options(False)


@dsl.container_component
def train_and_export_pytorchjob_op(
    job_name: str,
    task: str,
    num_envs: int,
    max_iterations: int,
    checkpoint_interval: int,
    s3_prefix: str,
    queue_name: str,
    onnx_s3_key: dsl.OutputPath(str),
):
    """Launch RSL-RL training + ONNX export as a PyTorchJob via Training Operator."""
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=job_name, $2=task, $3=num_envs, $4=max_iterations,
            # $5=checkpoint_interval, $6=s3_prefix, $7=queue_name, $8=output_file
            "set -euo pipefail\n"
            'JOB_NAME="$1"; export TASK="$2" NUM_ENVS="$3" MAX_ITERS="$4"\n'
            'export CHECKPOINT_INTERVAL="$5" S3_PREFIX="$6"\n'
            'QUEUE_NAME="$7"; OUTPUT_FILE="$8"\n'
            'export ONNX_DIR="/tmp/onnx_export" RESUME="s3"\n'
            "NAMESPACE=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)\n"
            "\n"
            'echo "=== WBC PyTorchJob Launcher ==="\n'
            'echo "Job: ${JOB_NAME}"\n'
            'echo "Task: ${TASK}"\n'
            'echo "Envs: ${NUM_ENVS}, Iterations: ${MAX_ITERS}"\n'
            'echo "Namespace: ${NAMESPACE}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            'TRAIN_CMD="/workspace/isaaclab/isaaclab.sh -p -m wbc_pipeline.train_and_export"\n'
            "\n"
            "QUEUE_ARGS=''\n"
            'if [ -n "${QUEUE_NAME}" ]; then\n'
            '    QUEUE_ARGS="--queue-name ${QUEUE_NAME}"\n'
            "fi\n"
            "\n"
            "/workspace/isaaclab/_isaac_sim/python.sh -m wbc_pipeline.pytorchjob \\\n"
            '  --name "${JOB_NAME}" \\\n'
            '  --namespace "${NAMESPACE}" \\\n'
            '  --image "' + DEFAULT_IMAGE + '" \\\n'
            '  --command "${TRAIN_CMD}" \\\n'
            "  --num-workers 1 \\\n"
            "  --gpus-per-worker 1 \\\n"
            "  --service-account-name isaaclab-gpu \\\n"
            "  --run-as-user 0 \\\n"
            "  ${QUEUE_ARGS}\n"
            "\n"
            'printf "%s" "${S3_PREFIX}/policy.onnx" > "${OUTPUT_FILE}"\n'
            'echo ""\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== WBC PyTorchJob Launcher: COMPLETE ==="',
            "--",
            job_name,
            task,
            num_envs,
            max_iterations,
            checkpoint_interval,
            s3_prefix,
            queue_name,
            onnx_s3_key,
        ],
    )


@dsl.pipeline(
    name="wbc-training-distributed",
    description="G1 29-DOF WBC training via PyTorchJob (infra validation)",
)
def wbc_training_pytorchjob_pipeline(
    job_name: str = "wbc-train",
    task: str = "WBC-Velocity-Flat-G1-29DOF-v0",
    num_envs: int = 4096,
    max_iterations: int = 6000,
    checkpoint_interval: int = 100,
    s3_prefix: str = "checkpoints",
    s3_bucket: str = "wbc-training",
    expected_obs_dim: int = 103,
    expected_action_dim: int = 29,
    model_name: str = "g1-wbc-flat-policy",
    model_version: str = "v1",
    queue_name: str = "",
):
    # Step 1: Train + Export via PyTorchJob (launcher is CPU-only)
    train_task = train_and_export_pytorchjob_op(
        job_name=job_name,
        task=task,
        num_envs=num_envs,
        max_iterations=max_iterations,
        checkpoint_interval=checkpoint_interval,
        s3_prefix=s3_prefix,
        queue_name=queue_name,
    )
    _configure_cpu_step(train_task)
    # Launcher is CPU-only but must tolerate GPU taint to schedule on GPU nodes
    kubernetes.add_toleration(train_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    train_task.set_env_variable("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    train_task.set_env_variable("MLFLOW_TRACKING_INSECURE_TLS", "true")
    train_task.set_env_variable("ACCEPT_EULA", "Y")
    kubernetes.set_timeout(train_task, 28800)
    train_task.set_caching_options(False)

    # Step 2: Validate ONNX
    validate_task = validate_onnx_op(
        onnx_s3_key=train_task.outputs["onnx_s3_key"],
        expected_obs_dim=expected_obs_dim,
        expected_action_dim=expected_action_dim,
    )
    _configure_cpu_step(validate_task)
    validate_task.set_caching_options(False)

    # Step 3: Register model
    register_task = register_model_op(
        model_name=model_name,
        model_version=model_version,
        onnx_s3_key=train_task.outputs["onnx_s3_key"],
        s3_bucket=s3_bucket,
    )
    register_task.after(validate_task)
    _configure_cpu_step(register_task)
    register_task.set_cpu_request("1")
    register_task.set_memory_request("2Gi")
    register_task.set_env_variable("MODEL_REGISTRY_ADDRESS", MODEL_REGISTRY_ADDRESS)
    register_task.set_caching_options(False)


if __name__ == "__main__":
    import sys

    from kfp import compiler

    output = sys.argv[1] if len(sys.argv) > 1 else "wbc_training_pipeline.yaml"
    compiler.Compiler().compile(wbc_training_pipeline, output)
    print(f"Pipeline compiled to {output}")
