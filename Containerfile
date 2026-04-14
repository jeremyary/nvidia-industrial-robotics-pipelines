# Isaac Lab training container for G1 29-DOF whole-body control
#
# Multi-stage build: isaaclab-base (cached) + runtime (rebuilds on src changes).
# Based on Isaac Lab's docker/Dockerfile.base pattern.
#
# Build:
#   podman build -t quay.io/jary/isaaclab-g1-train:latest -f Containerfile .
#
# Build base only (for caching/pushing):
#   podman build --target isaaclab-base -t quay.io/jary/isaaclab-base:latest -f Containerfile .
#
# Smoke test (requires NVIDIA GPU with CDI configured):
#   podman run --rm --device nvidia.com/gpu=all --env ACCEPT_EULA=Y \
#     quay.io/jary/isaaclab-g1-train:latest \
#     -m wbc_pipeline.train \
#       --task WBC-Velocity-Flat-G1-29DOF-v0 --headless --num_envs 64 --max_iterations 10

# ==============================================================================
# Stage 1: isaaclab-base — Isaac Sim + Isaac Lab + Python dependencies
# Rebuild only when Isaac Lab version or pip deps change.
# ==============================================================================

ARG ISAACSIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION} AS isaaclab-base

SHELL ["/bin/bash", "-c"]

LABEL description="Isaac Lab G1 WBC RL training container"

# Paths matching Isaac Lab's conventions
ENV ISAACSIM_ROOT_PATH=/isaac-sim
ENV ISAACLAB_PATH=/workspace/isaaclab
ENV ACCEPT_EULA=Y
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

USER root

# System dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Clone Isaac Lab at v2.3.2
RUN git clone --depth 1 --branch v2.3.2 \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

# Symlink Isaac Sim into Isaac Lab
RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim && \
    chmod +x ${ISAACLAB_PATH}/isaaclab.sh

# Install toml (needed by isaaclab.sh) and pre-install flatdict without
# build isolation — flatdict 4.0.1's setup.py imports pkg_resources which
# newer setuptools (82+) drops from pip's isolated build environments
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-build-isolation "flatdict==4.0.1"

# Install apt dependencies declared by Isaac Lab extensions
RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directories
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache \
    /root/.cache/ov /root/.cache/pip \
    /root/.cache/nvidia/GLCache /root/.nv/ComputeCache \
    /root/.nvidia-omniverse/logs /root/.local/share/ov/data \
    /root/Documents

# Install Isaac Lab + RSL-RL only (skip rl_games and skrl to reduce image size)
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install rsl_rl

# Install additional Python dependencies for training pipeline
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
    "mlflow>=2.15.0,<3.0" \
    "boto3>=1.34.0,<2.0" \
    "onnx>=1.16.0,<2.0" \
    "onnxruntime>=1.18.0,<2.0"

# ==============================================================================
# Stage 2: runtime — wbc_pipeline source code
# Rebuilds quickly when only src/ or pyproject.toml change.
# ==============================================================================

FROM isaaclab-base AS runtime

# Copy the wbc_pipeline package
COPY src/ /workspace/wbc_pipeline/src/
COPY pyproject.toml /workspace/wbc_pipeline/

# Install wbc_pipeline with pipeline + registry extras
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e "/workspace/wbc_pipeline[pipeline,registry]"

# Convenience aliases
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> /root/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> /root/.bashrc

WORKDIR ${ISAACLAB_PATH}

# Default entrypoint uses Isaac Sim's Python
ENTRYPOINT ["/workspace/isaaclab/isaaclab.sh", "-p"]
