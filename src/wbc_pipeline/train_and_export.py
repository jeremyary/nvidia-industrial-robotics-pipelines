# This project was developed with assistance from AI tools.
"""Combined train + ONNX export entrypoint for RSL-RL pipeline.

Reads TASK, NUM_ENVS, MAX_ITERS, ONNX_DIR from environment variables.
AppLauncher must be created before importing Isaac Lab modules.

Usage (inside Isaac Lab container):
    TASK=WBC-Velocity-Flat-G1-29DOF-v0 NUM_ENVS=4096 MAX_ITERS=6000 ONNX_DIR=/tmp/onnx \
        /workspace/isaaclab/isaaclab.sh -p -m wbc_pipeline.train_and_export
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

task = os.environ["TASK"]
num_envs = int(os.environ["NUM_ENVS"])
max_iters = int(os.environ["MAX_ITERS"])
onnx_dir = os.environ["ONNX_DIR"]
resume = os.environ.get("RESUME", "")

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=task)
parser.add_argument("--num_envs", type=int, default=num_envs)
parser.add_argument("--max_iterations", type=int, default=max_iters)
parser.add_argument("--resume", type=str, default=resume or None)
AppLauncher.add_app_launcher_args(parser)
sys.argv = [
    "train_and_export",
    "--task",
    task,
    "--headless",
    "--num_envs",
    str(num_envs),
    "--max_iterations",
    str(max_iters),
]
if resume:
    sys.argv.extend(["--resume", resume])
args = parser.parse_args()
app_launcher = AppLauncher(args)

import gymnasium as gym  # noqa: E402
import onnx  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx  # noqa: E402

import wbc_pipeline.envs  # noqa: E402, F401
from wbc_pipeline.config import TrainingConfig  # noqa: E402
from wbc_pipeline.runner import WBCRunner  # noqa: E402


def _resolve(entry_point):
    """Resolve a string entry point to a config instance."""
    if isinstance(entry_point, str):
        import importlib

        mod, cls = entry_point.rsplit(":", 1)
        return getattr(importlib.import_module(mod), cls)()
    return entry_point()


def main():
    spec = gym.spec(args.task)
    env_cfg = _resolve(spec.kwargs["env_cfg_entry_point"])
    agent_cfg = _resolve(spec.kwargs["rsl_rl_cfg_entry_point"])
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.max_iterations = args.max_iterations

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    training_cfg = TrainingConfig()
    log_dir = os.path.join("/workspace/isaaclab/logs/rsl_rl", agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    runner = WBCRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, training_cfg=training_cfg)

    if args.resume == "s3":
        runner.resume_latest_from_s3()

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Export ONNX
    print("\n=== Exporting ONNX policy ===")
    policy = runner.alg.policy
    normalizer = policy.actor_obs_normalizer
    os.makedirs(onnx_dir, exist_ok=True)
    export_policy_as_onnx(policy, path=onnx_dir, normalizer=normalizer, filename="policy.onnx")

    onnx_path = os.path.join(onnx_dir, "policy.onnx")
    model = onnx.load(onnx_path)
    in_dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    print(f"ONNX exported: input {in_dims}, output {out_dims}")

    # Upload ONNX to S3
    onnx_s3_key = f"{training_cfg.s3.prefix}/policy.onnx"
    if runner._s3_client is not None:
        runner._s3_client.upload_file(onnx_path, training_cfg.s3.bucket, onnx_s3_key)
        print(f"ONNX uploaded to s3://{training_cfg.s3.bucket}/{onnx_s3_key}")

    env.close()
    app_launcher.app.close()


if __name__ == "__main__":
    main()
