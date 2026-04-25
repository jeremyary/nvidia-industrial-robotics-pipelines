# WBC Pipeline — G1 Humanoid Robot Deployment

> [!NOTE]
> This project was developed with assistance from AI tools.

ML pipelines for the Unitree G1 humanoid robot, covering two tiers of the robot's control stack:

```
VLA (GR00T N1.7)  →  WBC (GEAR-SONIC)
  camera + text →        motion commands →
  motion commands         joint torques
```

Two KFP v2 pipelines on RHOAI:

1. **SONIC import** — fetches pre-trained GEAR-SONIC whole-body controller ONNX models from HuggingFace, validates, and registers
2. **VLA fine-tuning** — fine-tunes GR00T N1.7-3B vision-language-action model for G1 navigation

Both pipelines produce ONNX models registered in RHOAI Model Registry, ready for deployment.

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

### ONNX Validation

Both pipelines share ONNX validation logic (`onnx_validation.py`) that handles multimodal VLA models gracefully:
- Skips structural checks for models exceeding the 2GiB protobuf limit
- Tolerates TensorRT-specific ops (e.g., `MatMul(13)`, `Trilu(14)`) that standard onnxruntime can't run
- Validates input/output metadata, shapes, and determinism where possible

## Infrastructure

### Container Images

| Image | Base | Purpose |
|-------|------|---------|
| `quay.io/jary/wbc-sonic` | `python:3.12-slim` | SONIC checkpoint import (CPU-only) |
| `quay.io/jary/wbc-vla` | `nvidia/cuda:12.8.0-devel` + Isaac-GR00T | VLA fine-tuning + ONNX export |
| `quay.io/jary/wbc-gallery` | `python:3.12-slim` | Training video gallery |

### OCP Setup

```bash
make deploy-infra                      # namespace, MinIO, MLflow, DSPA, Kueue, Gallery
make deploy-model-registry             # Model Registry (Postgres + CR in rhoai-model-registries)
make pipeline-deploy                   # DSPA + RBAC for pipeline submission
```

Infrastructure includes:
- **MinIO** — S3-compatible object store for checkpoints and training artifacts
- **MLflow** — experiment tracking and metric logging
- **DSPA** — KFP v2 pipeline orchestration via RHOAI
- **Kueue** — GPU quota management for multi-tenant scheduling
- **Model Registry** — ONNX model versioning and lineage (RHOAI Model Registry with Postgres backend)
- **Video Gallery** — S3 video browser serving training progress recordings

### Model Registry

Requires `MODEL_REGISTRY_DB_PASSWORD` in `.env` (see `.env.example`).

```bash
make deploy-model-registry             # create DB secret + deploy Postgres + ModelRegistry CR
```

## Development

```bash
pip install -e ".[dev]"     # requires Python >=3.10
ruff check src/ tests/      # lint
ruff format src/ tests/     # format
pytest tests/ -v            # 106 tests (no GPU needed)
```

Pre-commit hooks run ruff lint, ruff format, and pytest on every commit. Tests validate pipeline compilation (SONIC, VLA), config construction, ONNX validation, Model Registry integration, and VLA config. Tests that require Isaac Lab runtime are auto-skipped outside the container.
