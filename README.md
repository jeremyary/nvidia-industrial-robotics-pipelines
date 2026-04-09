# WBC Pipeline — G1 Locomotion Training

> [!NOTE]
> This project was developed with assistance from AI tools.

RL training pipeline for the Unitree G1 humanoid robot using Isaac Lab v2.3.2 + RSL-RL PPO. Trains walking policies and exports them as ONNX models with observation normalization baked in, ready for real-time inference on the physical robot.

Currently targets the 29-DOF body-only variant (no fingers). The architecture supports variable DOF configurations via the `JointPreset` system — adding support for the full 37-DOF G1 (locomotion + manipulation) or reduced-DOF subsets (legs-only for faster iteration) means defining a new preset, not rewriting env configs. Each env variant has its own `@configclass` config file following Isaac Lab's pattern (separate classes, not parameterized inheritance).

## Why This Exists

Isaac Lab ships G1 environments for the 37-DOF variant (body + fingers). The downstream inference stack expects 29-DOF policies with a specific joint ordering, default positions, and observation vector. This pipeline bridges that gap — custom environments that produce ONNX models matching the inference contract exactly.

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

The container is built on `nvcr.io/nvidia/isaac-sim:5.1.0` with Isaac Lab v2.3.2 installed inside. Two-stage Containerfile: `isaaclab-base` (cached, ~25 min build) and `runtime` (rebuilds in seconds when only `src/` changes). Requires NGC API key in `.env` (see `.env.example`).

```bash
make build              # podman build
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

## Roadmap

Phases 0-3b are complete. Key upcoming work:

- **YAML config & DOF flexibility**: Externalize presets to YAML files, support variable joint counts (37-DOF full G1, reduced-DOF subsets), ConfigMap-based preset injection in containers.
- **KFP pipeline**: Define training as a KFP v2 pipeline on RHOAI DSPA (validate → train → export → validate-onnx → register).
- **Deployment manifests**: DSPA CR, Kueue integration, kustomize overlays.
- **B200 scale testing**: Performance benchmarks at 8k+ envs on B200 GPUs.

## Development

```bash
pip install -e ".[dev]"     # requires Python >=3.10, <3.13 (Isaac Sim constraint)
ruff check src/ tests/      # lint
ruff format src/ tests/     # format
pytest tests/ -v            # 93 tests (no GPU needed)
```

Pre-commit hooks run ruff lint, ruff format, and pytest on every commit. Tests validate joint presets, observation contracts, ONNX structure, phase oscillator behavior, and env registration. Tests that require Isaac Lab runtime are auto-skipped outside the container.
