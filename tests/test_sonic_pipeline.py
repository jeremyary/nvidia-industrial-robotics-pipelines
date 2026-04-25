# This project was developed with assistance from AI tools.
"""Tests for SONIC import pipeline compilation and structure."""

from __future__ import annotations

from pipeline_test_utils import compile_pipeline, compile_pipeline_full_yaml

from wbc_pipeline.sonic.pipeline import sonic_import_pipeline


def _compile_pipeline() -> dict:
    return compile_pipeline(sonic_import_pipeline)


def _compile_pipeline_full_yaml() -> str:
    return compile_pipeline_full_yaml(sonic_import_pipeline)


class TestPipelineCompilation:
    """Validate import pipeline compiles and has correct structure."""

    def test_compiles_to_valid_yaml(self):
        """Pipeline compiles without errors."""
        spec = _compile_pipeline()
        assert "root" in spec

    def test_has_three_tasks(self):
        """Pipeline contains fetch, validate, and register tasks."""
        spec = _compile_pipeline()
        tasks = spec["root"]["dag"]["tasks"]
        assert len(tasks) == 3
        assert "sonic-fetch-checkpoint-op" in tasks
        assert "sonic-validate-checkpoint-op" in tasks
        assert "sonic-register-checkpoint-op" in tasks

    def test_validate_depends_on_fetch(self):
        """Validate task depends on fetch output."""
        spec = _compile_pipeline()
        validate = spec["root"]["dag"]["tasks"]["sonic-validate-checkpoint-op"]
        assert "sonic-fetch-checkpoint-op" in validate.get("dependentTasks", [])

    def test_register_depends_on_validate(self):
        """Register task depends on validate task."""
        spec = _compile_pipeline()
        register = spec["root"]["dag"]["tasks"]["sonic-register-checkpoint-op"]
        deps = register.get("dependentTasks", [])
        assert "sonic-validate-checkpoint-op" in deps


class TestPipelineParameters:
    """Validate pipeline parameter defaults."""

    def test_default_hf_repo_id(self):
        """Default HuggingFace repo is nvidia/GEAR-SONIC."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["hf_repo_id"]["defaultValue"] == "nvidia/GEAR-SONIC"

    def test_default_s3_prefix(self):
        """Default S3 prefix is gear-sonic."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["s3_prefix"]["defaultValue"] == "gear-sonic"

    def test_default_model_name(self):
        """Default model name is g1-sonic-wbc."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["model_name"]["defaultValue"] == "g1-sonic-wbc"

    def test_default_model_version(self):
        """Default model version is v1."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["model_version"]["defaultValue"] == "v1"


class TestImportConfiguration:
    """Validate resource and secret configuration in compiled YAML."""

    def test_no_gpu_referenced(self):
        """Import pipeline should not request GPUs."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "nvidia.com/gpu" not in yaml_str

    def test_minio_secret_referenced(self):
        """Compiled YAML references the minio-credentials secret."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "minio-credentials" in yaml_str

    def test_hf_credentials_secret_referenced(self):
        """Compiled YAML references the hf-credentials secret for fetch step."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "hf-credentials" in yaml_str

    def test_model_registry_env_var(self):
        """Compiled YAML contains MODEL_REGISTRY_ADDRESS."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "MODEL_REGISTRY_ADDRESS" in yaml_str

    def test_sonic_image_referenced(self):
        """Compiled YAML uses the SONIC container image."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "wbc-sonic" in yaml_str

    def test_caching_disabled_for_all_steps(self):
        """All steps have caching disabled."""
        spec = _compile_pipeline()
        tasks = spec["root"]["dag"]["tasks"]
        for task_name, task_spec in tasks.items():
            caching = task_spec.get("cachingOptions", {})
            assert caching.get("enableCache") is not True, f"{task_name} has caching enabled"

    def test_uses_python_not_isaac_sim(self):
        """Import pipeline uses plain python, not Isaac Sim python.sh."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "python.sh" not in yaml_str
        assert "python -m wbc_pipeline" in yaml_str
