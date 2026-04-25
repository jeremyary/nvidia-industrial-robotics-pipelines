# WBC Pipeline — G1 Humanoid Robot Deployment

> [!NOTE]
> This project was developed with assistance from AI tools.

End-to-end ML pipeline for the Unitree G1 humanoid robot, covering three tiers of the robot's control stack:

```
Mission Planner (Nemotron)  →  VLA (GR00T N1.7)  →  WBC (GEAR-SONIC)
  high-level goals             camera + text →        motion commands →
                               motion commands         joint torques
```

The project includes three KFP v2 pipelines on RHOAI:

1. **RSL-RL training** — trains velocity-tracking WBC policies from scratch using Isaac Lab + RSL-RL PPO
2. **SONIC import** — fetches pre-trained GEAR-SONIC whole-body controller ONNX models from HuggingFace, validates, and registers
3. **VLA fine-tuning** — fine-tunes GR00T N1.7-3B vision-language-action model for G1 navigation

All pipelines produce ONNX models registered in RHOAI Model Registry, ready for deployment.

## RSL-RL Training

RL training pipeline using Isaac Lab v2.3.2 + RSL-RL PPO. Trains walking policies and exports them as ONNX models with observation normalization baked in.

Isaac Lab ships G1 environments for the 37-DOF variant (body + fingers). The downstream inference stack expects 29-DOF policies with a specific joint ordering, default positions, and observation vector. This pipeline bridges that gap — custom environments that produce ONNX models matching the inference contract exactly.

Currently targets the 29-DOF body-only variant (no fingers). The architecture supports variable DOF configurations via the `JointPreset` system — adding support for the full 37-DOF G1 (locomotion + manipulation) or reduced-DOF subsets (legs-only for faster iteration) means defining a new preset, not rewriting env configs. Each env variant has its own `@configclass` config file following Isaac Lab's pattern (separate classes, not parameterized inheritance).

## Environments

Four registered gymnasium environments, all sharing the same action space:

| Task ID | Obs Dim | Terrain | Sensors | Preset |
|---------|---------|---------|---------|--------|
| `WBC-Velocity-Flat-G1-29DOF-v0` | 103 | Flat plane | Contact | Operator |
| `WBC-Velocity-Rough-G1-29DOF-v0` | 290 | Procedural rough | Contact + height scan (187 rays) | Operator |
| `WBC-Velocity-Warehouse-G1-29DOF-v0` | 466 | USD warehouse scene | Contact + lidar (121 rays x 3) | Operator |
| `WBC-Velocity-IsaacLab-Flat-G1-29DOF-v0` | 103 | Flat plane | Contact | Isaac Lab stock |

**Observations** (base vector for N joints): linear velocity (3) + angular velocity (3) + projected gravity (3) + velocity commands (3) + joint positions (N) + joint velocities (N) + last actions (N) + phase oscillator (4). With the current 29-DOF preset, that's 103 dims. Rough and warehouse envs append sensor data.

**Joint presets** (`joint_presets.py`) define the joint ordering, default positions, action scale, and phase oscillator settings. The "operator" preset matches the inference stack's 29-DOF convention. The "Isaac Lab stock" preset uses upstream defaults (shallower knee bend, `action_scale=0.5`, no phase oscillator). Adding a new preset means adding a `JointPreset` dataclass instance — no config classes need to change.

## ONNX Contract

Exported policies must satisfy:

- Input: `"obs"` tensor, shape `[1, obs_dim]`, float32
- Output: `"actions"` tensor, shape `[1, N_joints]`, float32
- Observation normalization baked into the ONNX graph (not applied at inference time)
- Joint ordering matches the active preset's `joint_order`
- Action = position offset scaled by `action_scale` (0.25 for operator, 0.5 for Isaac Lab stock)

The `export_onnx.py` script derives dimensions from the live environment and validates the exported model against `EXPECTED_OBS_DIM` / `EXPECTED_ACTION_DIM` declared on each env config. This catches dim mismatches before training hours are wasted.

## Usage

Three container images, each sized for its workload:

| Image | Base | Size | Purpose |
|-------|------|------|---------|
| `quay.io/jary/isaaclab-g1-train` | Isaac Sim 5.1.0 + Isaac Lab 2.3.2 | ~15GB | RSL-RL training + ONNX export |
| `quay.io/jary/wbc-sonic` | `python:3.12-slim` | ~200MB | SONIC checkpoint import (CPU-only) |
| `quay.io/jary/wbc-vla` | `nvidia/cuda:12.8.0-devel` + Isaac-GR00T | ~8-10GB | VLA fine-tuning + ONNX export |

The RSL-RL container uses a two-stage Containerfile: `isaaclab-base` (cached, ~25 min build) and `runtime` (rebuilds in seconds when only `src/` changes). Requires NGC API key in `.env` (see `.env.example`).

```bash
make build              # build RSL-RL container (requires NGC login)
make push               # push to quay.io
make local-smoke-test   # local GPU test via Podman + CDI
```

Training and ONNX export run inside the container, either locally or via OpenShift jobs:

```bash
python -m wbc_pipeline.train \
  --task WBC-Velocity-Flat-G1-29DOF-v0 \
  --headless --num_envs 4096 --max_iterations 6000

python -m wbc_pipeline.export_onnx \
  --task WBC-Velocity-Flat-G1-29DOF-v0 \
  --headless --checkpoint /path/to/model_6000.pt --output_dir /tmp/onnx
```

S3 checkpointing (`S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), MLflow tracking (`MLFLOW_TRACKING_URI`), and SIGTERM checkpoint-on-preemption are all opt-in via env vars.

## KFP Pipelines (RHOAI DSPA)

Three KFP v2 pipelines run on RHOAI Data Science Pipelines Application:

```bash
make pipeline-compile                  # RSL-RL training (single-pod GPU)
make pipeline-compile-distributed      # RSL-RL training (PyTorchJob, multi-node)
make sonic-pipeline-compile            # SONIC import (CPU-only)
make vla-pipeline-compile              # VLA fine-tuning (single GPU)
```

RSL-RL supports two tiers: **Tier 1** (single-pod, default) runs training in one container; **Tier 2** (PyTorchJob) uses a CPU launcher pod to create a PyTorchJob CR via the Training Operator, with Kueue managing GPU quota for multi-node DDP.

All pipelines share the same post-processing pattern: ONNX validation → Model Registry registration. Submit compiled pipeline YAML through the RHOAI dashboard or DSPA API.

### Model Registry

Trained ONNX models are registered with RHOAI Model Registry for versioning and lineage tracking. The registry runs in `rhoai-model-registries` namespace with a Postgres backend.

```bash
make deploy-model-registry             # create DB secret + deploy Postgres + ModelRegistry CR
```

Requires `MODEL_REGISTRY_DB_PASSWORD` in `.env` (see `.env.example`).

### Kueue GPU Quota

PyTorchJob workloads are managed by Kueue for GPU quota enforcement. The ClusterQueue is scoped to the training namespace.

```bash
make deploy-pytorchjob-infra           # deploy RBAC + Kueue ClusterQueue/LocalQueue
```

## OpenShift Jobs

Manifests in `deploy/` are split into `infra/` (namespace, SCC, MinIO, MLflow) and `jobs/` (GPU workloads). Jobs are managed through a parametric Makefile registry — no per-job targets, just `make job-deploy JOB=<name>`.

```bash
make deploy-infra                      # one-time cluster setup (namespace, SCC, MinIO, MLflow)

make job-deploy JOB=smoke-test         # quick tests (10 iters, 64 envs)
make job-deploy JOB=test-flat
make job-deploy JOB=test-rough
make job-deploy JOB=test-warehouse
make job-deploy JOB=test-preset

make job-deploy JOB=training-flat-6k   # full training (6000 iters, 4096 envs + S3 + ONNX)
make job-deploy JOB=training-rough-6k
make job-deploy JOB=training-warehouse-6k
make job-deploy JOB=training-preset-6k

make job-logs JOB=<name>               # tail logs
make job-clean JOB=<name>              # delete job
make job-list                          # show all available jobs
```

### Adding a new job

Add 3 lines to the Makefile job registry and create a YAML manifest in `deploy/jobs/`:

```makefile
JOB_FILE_my-new-job          = deploy/jobs/my-new-job.yaml
JOB_NAME_my-new-job          = my-new-job
JOB_NEEDS_INFRA_my-new-job   = true    # false if no S3/MLflow needed
```

Jobs with `JOB_NEEDS_INFRA=true` automatically run `make deploy-infra` before deploying. Jobs with `false` only set up the namespace and GPU SCC.

## GEAR-SONIC Import

Imports NVIDIA's pre-trained GEAR-SONIC whole-body controller into the pipeline. SONIC produces three ONNX models (encoder, decoder, planner) that convert motion commands into joint torques for the G1 29-DOF robot.

The import pipeline fetches pre-trained ONNX models from HuggingFace (`nvidia/GEAR-SONIC`), caches them in S3, validates shape/inference/determinism, and registers in Model Registry. No GPU required — runs entirely on CPU.

```bash
make build-sonic                       # build lightweight python:3.12-slim container
make push-sonic                        # push to quay.io
make sonic-pipeline-compile            # compile to sonic_pipeline.yaml
```

Pipeline steps: `fetch_checkpoint` → `validate_onnx` → `register_model`

## VLA Fine-Tuning (GR00T N1.7-3B)

Fine-tunes NVIDIA's GR00T N1.7-3B vision-language-action model for G1 navigation. GR00T N1.7 is a 3.14B-parameter model (Apache 2.0) that converts camera images + language instructions into robot motion commands.

The fine-tuning pipeline downloads the base model from HuggingFace, fine-tunes with `torchrun` using the built-in `UNITREE_G1` embodiment tag, exports to ONNX, validates, and registers. Default configuration targets a proof-of-concept run: 2K steps, 1 GPU (L40S 48GB), ~30-60 minutes.

```bash
make build-vla                         # build CUDA 12.8 + Isaac-GR00T container
make push-vla                          # push to quay.io
make vla-pipeline-compile              # compile to vla_finetune_pipeline.yaml
```

Pipeline steps: `data_prep` (CPU) → `fine_tune_and_export` (GPU) → `validate_onnx` (CPU) → `register_model` (CPU)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_model_repo` | `nvidia/GR00T-N1.7-3B` | HuggingFace model ID |
| `embodiment_tag` | `UNITREE_G1` | Built-in G1 config |
| `dataset_name` | `robot_sim.PickNPlace` | Demo data bundled in container |
| `max_steps` | 2000 | Increase for production runs |
| `global_batch_size` | 32 | |

## B200 Deployment

The pipeline is validated on L40S (48GB VRAM). Moving to B200 (192GB VRAM, 8 GPUs per HGX node) requires:

**GPU Driver**: DGX/HGX B200 nodes use the pre-installed datacenter driver. The GPU Operator cannot deploy driver containers on these platforms — verify the host driver is installed before scheduling workloads.

**Node Configuration**:
- Label GPU nodes: `nvidia.com/gpu.present=true` (required by Kueue ResourceFlavor)
- Taint GPU nodes: `nvidia.com/gpu:NoSchedule` (training pods tolerate this; prevents non-GPU workloads from landing on expensive nodes)
- Verify NVIDIA device plugin is running and GPUs are advertised as allocatable resources

**Kueue Quotas**: Update `deploy/infra/kueue.yaml` for B200 capacity before deploying:
- `nvidia.com/gpu` nominalQuota: 8 per HGX node (adjust for total cluster GPUs)
- `cpu` / `memory`: scale to match B200 node specs
- Deploy with `make deploy-pytorchjob-infra` after updating

**HyperShift**: GPU passthrough for HyperShift hosted clusters is Tech Preview (OCP 4.17+). For production GPU training, bare metal node pools are preferred. Autoscaling GPU node pools to/from zero (OCP 4.21+) reduces cost for burst training.

**Validation Sequence**:
1. `make job-deploy JOB=smoke-test` — verify container + GPU access
2. `make job-deploy JOB=test-flat` — verify 10-iter training runs
3. Submit Tier 2 pipeline via DSPA — verify PyTorchJob + Kueue flow
4. Submit full training run (6K+ iters, 4096 envs) — verify at production scale

## Development

```bash
pip install -e ".[dev]"     # requires Python >=3.10
ruff check src/ tests/      # lint
ruff format src/ tests/     # format
pytest tests/ -v            # 237 tests (no GPU needed)
```

Pre-commit hooks run ruff lint, ruff format, and pytest on every commit. Tests validate joint presets, observation contracts, ONNX structure, phase oscillator behavior, env registration, PyTorchJob CR construction, pipeline compilation (RSL-RL, SONIC, VLA), Model Registry integration, and VLA config. Tests that require Isaac Lab runtime are auto-skipped outside the container.
