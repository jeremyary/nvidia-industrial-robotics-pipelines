# Project Conventions

## Architecture
- ML pipelines for Unitree G1 29-DOF humanoid robot
- Two pipelines: SONIC deployment (CPU-only import) and VLA fine-tuning (GR00T N1.7-3B)
- Deployment: RHOAI on OpenShift with L40S or B200 GPUs
- Pipeline orchestration: KFP v2 on RHOAI DSPA
- Output: ONNX models registered in RHOAI Model Registry

## Container
- Registry: quay.io/jary/
- Runtime: Podman with CDI (`--device nvidia.com/gpu=all`, NOT `--gpus all`)
- SONIC: python:3.12-slim (CPU-only)
- VLA: nvidia/cuda:12.8.0-devel + Isaac-GR00T

## Testing
- Use `respx` for mocking `httpx` in async tests.
- Bind mock/test servers to `127.0.0.1`, not `0.0.0.0`.
