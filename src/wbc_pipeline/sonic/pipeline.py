# This project was developed with assistance from AI tools.
"""KFP v2 pipeline: import pre-trained GEAR-SONIC WBC into Model Registry.

Steps: fetch_checkpoint → validate_onnx → register_model

No GPU required — pulls pre-trained ONNX models from HuggingFace,
validates shapes/inference/determinism, and registers in RHOAI Model Registry.
"""

from kfp import dsl, kubernetes

from wbc_pipeline.constants import MODEL_REGISTRY_ADDRESS, S3_ENDPOINT

SONIC_IMAGE = "quay.io/jary/wbc-sonic:latest"


def _configure_cpu_step(task: dsl.PipelineTask) -> None:
    """Apply secrets, env vars, and pull policy to a CPU-only pipeline task."""
    task.set_cpu_request("2")
    task.set_memory_request("4Gi")
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
def sonic_fetch_checkpoint_op(
    hf_repo_id: str,
    s3_prefix: str,
    s3_prefix_out: dsl.OutputPath(str),
):
    """Download GEAR-SONIC ONNX models from HuggingFace and upload to S3 (cached)."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=hf_repo_id, $2=s3_prefix, $3=output_file
            "set -euo pipefail\n"
            'HF_REPO_ID="$1"; S3_PREFIX="$2"; OUTPUT_FILE="$3"\n'
            "\n"
            'echo "=== SONIC Checkpoint Fetch ==="\n'
            'echo "HuggingFace repo: ${HF_REPO_ID}"\n'
            'echo "S3 prefix: ${S3_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "python -m wbc_pipeline.sonic.fetch_checkpoint \\\n"
            '  --repo-id "${HF_REPO_ID}" \\\n'
            '  --s3-prefix "${S3_PREFIX}"\n'
            "\n"
            'printf "%s" "${S3_PREFIX}" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC Checkpoint Fetch: COMPLETE ==="',
            "--",
            hf_repo_id,
            s3_prefix,
            s3_prefix_out,
        ],
    )


@dsl.container_component
def sonic_validate_checkpoint_op(
    s3_prefix: str,
    validation_result: dsl.OutputPath(str),
):
    """Validate SONIC ONNX models from S3: shapes, inference, determinism."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=s3_prefix, $2=output_file
            "set -euo pipefail\n"
            'S3_PREFIX="$1"; OUTPUT_FILE="$2"\n'
            "\n"
            'echo "=== SONIC ONNX Validation ==="\n'
            'echo "S3 prefix: ${S3_PREFIX}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "python -m wbc_pipeline.sonic.validate \\\n"
            '  --checkpoint-prefix "${S3_PREFIX}"\n'
            "\n"
            'echo "PASSED" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC ONNX Validation: COMPLETE ==="',
            "--",
            s3_prefix,
            validation_result,
        ],
    )


@dsl.container_component
def sonic_register_checkpoint_op(
    model_name: str,
    model_version: str,
    s3_prefix: str,
    s3_bucket: str,
    registration_result: dsl.OutputPath(str),
):
    """Register validated SONIC ONNX models with RHOAI Model Registry."""
    return dsl.ContainerSpec(
        image=SONIC_IMAGE,
        command=["/bin/bash"],
        args=[
            "-c",
            # Parameters: $1=model_name, $2=model_version, $3=s3_prefix,
            # $4=s3_bucket, $5=output_file
            "set -euo pipefail\n"
            'MODEL_NAME="$1"; MODEL_VERSION="$2"; S3_PREFIX="$3"\n'
            'S3_BUCKET="$4"; OUTPUT_FILE="$5"\n'
            'MODEL_URI="s3://${S3_BUCKET}/${S3_PREFIX}/onnx"\n'
            "\n"
            'echo "=== SONIC Model Registration ==="\n'
            'echo "Model: ${MODEL_NAME} v${MODEL_VERSION}"\n'
            'echo "URI: ${MODEL_URI}"\n'
            'echo "Start time: $(date -u)"\n'
            "\n"
            "python -m wbc_pipeline.registry \\\n"
            '  --name "${MODEL_NAME}" \\\n'
            '  --uri "${MODEL_URI}" \\\n'
            '  --version "${MODEL_VERSION}" \\\n'
            '  --description "Pre-trained GEAR-SONIC whole-body controller for G1 29-DOF"\n'
            "\n"
            'echo "REGISTERED" > "${OUTPUT_FILE}"\n'
            'echo "End time: $(date -u)"\n'
            'echo "=== SONIC Model Registration: COMPLETE ==="',
            "--",
            model_name,
            model_version,
            s3_prefix,
            s3_bucket,
            registration_result,
        ],
    )


@dsl.pipeline(
    name="wbc-sonic-import",
    description="Import pre-trained GEAR-SONIC WBC checkpoint into Model Registry",
)
def sonic_import_pipeline(
    hf_repo_id: str = "nvidia/GEAR-SONIC",
    s3_prefix: str = "gear-sonic",
    s3_bucket: str = "wbc-training",
    model_name: str = "g1-sonic-wbc",
    model_version: str = "v1",
):
    # Step 1: Fetch checkpoint from HuggingFace (cached in S3)
    fetch_task = sonic_fetch_checkpoint_op(
        hf_repo_id=hf_repo_id,
        s3_prefix=s3_prefix,
    )
    _configure_cpu_step(fetch_task)
    fetch_task.set_memory_request("8Gi")
    kubernetes.use_secret_as_env(
        fetch_task,
        secret_name="hf-credentials",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )
    fetch_task.set_caching_options(False)

    # Step 2: Validate ONNX models
    validate_task = sonic_validate_checkpoint_op(
        s3_prefix=fetch_task.outputs["s3_prefix_out"],
    )
    _configure_cpu_step(validate_task)
    kubernetes.set_timeout(validate_task, 1800)
    validate_task.set_caching_options(False)

    # Step 3: Register in Model Registry
    register_task = sonic_register_checkpoint_op(
        model_name=model_name,
        model_version=model_version,
        s3_prefix=fetch_task.outputs["s3_prefix_out"],
        s3_bucket=s3_bucket,
    )
    register_task.after(validate_task)
    _configure_cpu_step(register_task)
    register_task.set_cpu_request("1")
    register_task.set_memory_request("2Gi")
    register_task.set_env_variable("MODEL_REGISTRY_ADDRESS", MODEL_REGISTRY_ADDRESS)
    kubernetes.set_timeout(register_task, 600)
    register_task.set_caching_options(False)


if __name__ == "__main__":
    import sys

    from kfp import compiler

    output = sys.argv[1] if len(sys.argv) > 1 else "sonic_pipeline.yaml"
    compiler.Compiler().compile(sonic_import_pipeline, output)
    print(f"Pipeline compiled to {output}")
