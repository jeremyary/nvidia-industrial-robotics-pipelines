# This project was developed with assistance from AI tools.
"""KFP v2 pipeline: train G1 29-DOF locomotion policy, export ONNX, validate."""

from kfp import dsl, kubernetes

DEFAULT_IMAGE = "quay.io/jary/isaaclab-g1-train:latest"
S3_ENDPOINT = "http://minio.wbc-training.svc.cluster.local:9000"
MLFLOW_TRACKING_URI = "https://mlflow.redhat-ods-applications.svc:8443"


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
    kubernetes.set_image_pull_policy(task, "IfNotPresent")


def _configure_cpu_step(task: dsl.PipelineTask) -> None:
    """Apply secrets, env vars, and pull policy to a non-GPU pipeline task."""
    task.set_cpu_request("2")
    task.set_memory_request("8Gi")
    task.set_env_variable("PYTHONUNBUFFERED", "1")
    task.set_env_variable("S3_ENDPOINT", S3_ENDPOINT)
    kubernetes.add_toleration(task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    kubernetes.use_secret_as_env(
        task,
        secret_name="minio-credentials",
        secret_key_to_env={"MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID", "MINIO_ROOT_PASSWORD": "AWS_SECRET_ACCESS_KEY"},
    )
    kubernetes.set_image_pull_policy(task, "IfNotPresent")


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
            # Parameters arrive as positional args: $1=task, $2=num_envs,
            # $3=max_iterations, $4=checkpoint_interval, $5=s3_prefix, $6=output_file
            "set -euo pipefail\n"
            'TASK="$1"; NUM_ENVS="$2"; MAX_ITERS="$3"; CKPT_INTERVAL="$4"\n'
            'S3_PREFIX="$5"; OUTPUT_FILE="$6"\n'
            'ONNX_DIR="/tmp/onnx_export"\n'
            'mkdir -p "${ONNX_DIR}"\n'
            "\n"
            'echo "=== WBC Training Pipeline ==="\n'
            'echo "Task: ${TASK}"\n'
            'echo "Envs: ${NUM_ENVS}"\n'
            'echo "Iterations: ${MAX_ITERS}"\n'
            'echo "S3 prefix: ${S3_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            'export S3_PREFIX CHECKPOINT_INTERVAL="${CKPT_INTERVAL}"\n'
            "\n"
            '/workspace/isaaclab/isaaclab.sh -p -c "\n'
            "import sys, os\n"
            "sys.argv = ['train', '--task', '${TASK}', '--headless',\n"
            "            '--num_envs', '${NUM_ENVS}', '--max_iterations', '${MAX_ITERS}',\n"
            "            '--resume', 's3']\n"
            "\n"
            "from isaaclab.app import AppLauncher\n"
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--task', type=str)\n"
            "parser.add_argument('--num_envs', type=int)\n"
            "parser.add_argument('--max_iterations', type=int)\n"
            "parser.add_argument('--resume', type=str, default=None)\n"
            "AppLauncher.add_app_launcher_args(parser)\n"
            "args = parser.parse_args()\n"
            "app_launcher = AppLauncher(args)\n"
            "\n"
            "import gymnasium as gym\n"
            "from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper\n"
            "from isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx\n"
            "import wbc_pipeline.envs\n"
            "from wbc_pipeline.config import TrainingConfig\n"
            "from wbc_pipeline.runner import WBCRunner\n"
            "\n"
            "def _resolve(ep):\n"
            "    if isinstance(ep, str):\n"
            "        import importlib\n"
            "        mod, cls = ep.rsplit(':', 1)\n"
            "        return getattr(importlib.import_module(mod), cls)()\n"
            "    return ep()\n"
            "\n"
            "spec = gym.spec(args.task)\n"
            "env_cfg = _resolve(spec.kwargs['env_cfg_entry_point'])\n"
            "agent_cfg = _resolve(spec.kwargs['rsl_rl_cfg_entry_point'])\n"
            "env_cfg.scene.num_envs = args.num_envs\n"
            "agent_cfg.max_iterations = args.max_iterations\n"
            "\n"
            "env = gym.make(args.task, cfg=env_cfg)\n"
            "env = RslRlVecEnvWrapper(env)\n"
            "\n"
            "training_cfg = TrainingConfig()\n"
            "log_dir = os.path.join('/workspace/isaaclab/logs/rsl_rl', agent_cfg.experiment_name)\n"
            "os.makedirs(log_dir, exist_ok=True)\n"
            "runner = WBCRunner(env, agent_cfg.to_dict(), log_dir=log_dir,\n"
            "                  device=agent_cfg.device, training_cfg=training_cfg)\n"
            "\n"
            "if args.resume == 's3':\n"
            "    runner.resume_latest_from_s3()\n"
            "\n"
            "runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)\n"
            "\n"
            "print()\n"
            "print('=== Exporting ONNX policy ===')\n"
            "policy = runner.alg.policy\n"
            "normalizer = policy.actor_obs_normalizer\n"
            "export_policy_as_onnx(policy, path='${ONNX_DIR}', normalizer=normalizer, filename='policy.onnx')\n"
            "\n"
            "import onnx\n"
            "model = onnx.load('${ONNX_DIR}/policy.onnx')\n"
            "in_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]\n"
            "out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]\n"
            "print(f'ONNX exported: input {in_dims}, output {out_dims}')\n"
            "\n"
            "onnx_s3_key = f'{training_cfg.s3.prefix}/policy.onnx'\n"
            "if runner._s3_client is not None:\n"
            "    runner._s3_client.upload_file('${ONNX_DIR}/policy.onnx',\n"
            "        training_cfg.s3.bucket, onnx_s3_key)\n"
            "    print(f'ONNX uploaded to s3://{training_cfg.s3.bucket}/{onnx_s3_key}')\n"
            "\n"
            "env.close()\n"
            "app_launcher.app.close()\n"
            '"\n'
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
            "set -euo pipefail\n"
            'ONNX_KEY=$(echo -n "$1"); OBS_DIM="$2"; ACT_DIM="$3"; OUTPUT_FILE="$4"\n'
            'ONNX_PATH="/tmp/policy.onnx"\n'
            "\n"
            'echo "=== ONNX Validation ==="\n'
            'echo "S3 key: ${ONNX_KEY}"\n'
            'echo "Expected obs_dim: ${OBS_DIM}, action_dim: ${ACT_DIM}"\n'
            "\n"
            '/workspace/isaaclab/_isaac_sim/python.sh -c "\n'
            "import boto3, os, sys\n"
            "s3 = boto3.client('s3',\n"
            "    endpoint_url=os.environ['S3_ENDPOINT'],\n"
            "    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],\n"
            "    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])\n"
            "s3.download_file('wbc-training', '${ONNX_KEY}', '${ONNX_PATH}')\n"
            "print(f'Downloaded s3://wbc-training/${ONNX_KEY}')\n"
            "\n"
            "import onnx\n"
            "import onnxruntime as ort\n"
            "import numpy as np\n"
            "\n"
            "model = onnx.load('${ONNX_PATH}')\n"
            "onnx.checker.check_model(model)\n"
            "\n"
            "inp = model.graph.input[0]\n"
            "out = model.graph.output[0]\n"
            "in_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]\n"
            "out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]\n"
            "print(f'Input: name={inp.name!r}, shape={in_shape}')\n"
            "print(f'Output: name={out.name!r}, shape={out_shape}')\n"
            "\n"
            "assert in_shape == [1, ${OBS_DIM}], f'Expected input [1, ${OBS_DIM}], got {in_shape}'\n"
            "assert out_shape == [1, ${ACT_DIM}], f'Expected output [1, ${ACT_DIM}], got {out_shape}'\n"
            "\n"
            "session = ort.InferenceSession('${ONNX_PATH}')\n"
            "obs = np.random.randn(1, ${OBS_DIM}).astype(np.float32)\n"
            "actions = session.run(None, {inp.name: obs})[0]\n"
            "assert actions.shape == (1, ${ACT_DIM}), f'ORT shape mismatch: {actions.shape}'\n"
            "assert np.all(np.isfinite(actions)), 'Non-finite outputs detected'\n"
            "\n"
            "actions2 = session.run(None, {inp.name: obs})[0]\n"
            "assert np.array_equal(actions, actions2), 'Non-deterministic outputs'\n"
            "\n"
            "op_types = {n.op_type for n in model.graph.node}\n"
            "has_norm = bool(op_types & {'Sub', 'BatchNormalization', 'Div'})\n"
            "print(f'Normalizer ops present: {has_norm}')\n"
            "if not has_norm:\n"
            "    print('WARNING: No normalization ops found in ONNX graph')\n"
            "\n"
            "print()\n"
            "print('=== ONNX Validation PASSED ===')\n"
            "print(f'  Input: {in_shape}, Output: {out_shape}')\n"
            "print(f'  Deterministic: yes')\n"
            "norm_str = 'baked in' if has_norm else 'MISSING'\n"
            "print(f'  Normalizer: {norm_str}')\n"
            '"\n'
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


@dsl.pipeline(name="wbc-training", description="G1 29-DOF WBC locomotion training pipeline")
def wbc_training_pipeline(
    task: str = "WBC-Velocity-Flat-G1-29DOF-v0",
    num_envs: int = 4096,
    max_iterations: int = 6000,
    checkpoint_interval: int = 100,
    s3_prefix: str = "checkpoints",
    expected_obs_dim: int = 103,
    expected_action_dim: int = 29,
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


if __name__ == "__main__":
    import sys

    from kfp import compiler

    output = sys.argv[1] if len(sys.argv) > 1 else "wbc_training_pipeline.yaml"
    compiler.Compiler().compile(wbc_training_pipeline, output)
    print(f"Pipeline compiled to {output}")
