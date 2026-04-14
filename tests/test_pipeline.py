# This project was developed with assistance from AI tools.
"""Tests for KFP v2 pipeline compilation and structure."""

from __future__ import annotations

from pipeline_test_utils import compile_pipeline, compile_pipeline_full_yaml

from wbc_pipeline.pipeline import wbc_training_pipeline


def _compile_pipeline() -> dict:
    return compile_pipeline(wbc_training_pipeline)


def _compile_pipeline_full_yaml() -> str:
    return compile_pipeline_full_yaml(wbc_training_pipeline)


class TestPipelineCompilation:
    """Validate pipeline compiles and has correct structure."""

    def test_compiles_to_valid_yaml(self):
        """Pipeline compiles without errors."""
        spec = _compile_pipeline()
        assert "root" in spec

    def test_has_three_tasks(self):
        """Pipeline contains train-and-export, validate-onnx, and register-model tasks."""
        spec = _compile_pipeline()
        tasks = spec["root"]["dag"]["tasks"]
        assert len(tasks) == 3
        assert "train-and-export-op" in tasks
        assert "validate-onnx-op" in tasks
        assert "register-model-op" in tasks

    def test_validate_depends_on_train(self):
        """Validate task depends on train task output."""
        spec = _compile_pipeline()
        validate = spec["root"]["dag"]["tasks"]["validate-onnx-op"]
        assert "train-and-export-op" in validate.get("dependentTasks", [])

    def test_register_depends_on_validate(self):
        """Register task depends on validate task."""
        spec = _compile_pipeline()
        register = spec["root"]["dag"]["tasks"]["register-model-op"]
        assert "validate-onnx-op" in register.get("dependentTasks", [])


class TestPipelineParameters:
    """Validate pipeline parameter defaults."""

    def test_default_task(self):
        """Default task is WBC-Velocity-Flat-G1-29DOF-v0."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["task"]["defaultValue"] == "WBC-Velocity-Flat-G1-29DOF-v0"

    def test_default_num_envs(self):
        """Default num_envs is 4096."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["num_envs"]["defaultValue"] == 4096.0

    def test_default_max_iterations(self):
        """Default max_iterations is 6000."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["max_iterations"]["defaultValue"] == 6000.0

    def test_default_obs_dim(self):
        """Default expected_obs_dim is 103."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["expected_obs_dim"]["defaultValue"] == 103.0

    def test_default_action_dim(self):
        """Default expected_action_dim is 29."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["expected_action_dim"]["defaultValue"] == 29.0


class TestGPUConfiguration:
    """Validate GPU and resource configuration in compiled YAML."""

    def test_gpu_in_compiled_yaml(self):
        """Compiled YAML references nvidia.com/gpu."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "nvidia.com/gpu" in yaml_str

    def test_train_task_caching_disabled(self):
        """Train task has caching disabled (empty cachingOptions, no enableCache: true)."""
        spec = _compile_pipeline()
        train = spec["root"]["dag"]["tasks"]["train-and-export-op"]
        caching = train.get("cachingOptions", {})
        assert caching.get("enableCache") is not True

    def test_minio_secret_referenced(self):
        """Compiled YAML references the minio-credentials secret."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "minio-credentials" in yaml_str

    def test_toleration_referenced(self):
        """Compiled YAML contains GPU toleration."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "NoSchedule" in yaml_str

    def test_env_vars_in_compiled_yaml(self):
        """Compiled YAML contains required environment variables."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "ACCEPT_EULA" in yaml_str
        assert "S3_ENDPOINT" in yaml_str
