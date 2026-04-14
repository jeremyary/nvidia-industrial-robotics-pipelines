# This project was developed with assistance from AI tools.
"""Tests for SONIC KFP v2 pipeline compilation and structure."""

from __future__ import annotations

from pipeline_test_utils import compile_pipeline, compile_pipeline_full_yaml

from wbc_pipeline.sonic.pipeline import sonic_training_pipeline


def _compile_pipeline() -> dict:
    return compile_pipeline(sonic_training_pipeline)


def _compile_pipeline_full_yaml() -> str:
    return compile_pipeline_full_yaml(sonic_training_pipeline)


class TestPipelineCompilation:
    """Validate SONIC pipeline compiles and has correct structure."""

    def test_compiles_to_valid_yaml(self):
        """Pipeline compiles without errors."""
        spec = _compile_pipeline()
        assert "root" in spec

    def test_has_five_tasks(self):
        """Pipeline contains data prep, train, export, validate, and register tasks."""
        spec = _compile_pipeline()
        tasks = spec["root"]["dag"]["tasks"]
        assert len(tasks) == 5
        assert "sonic-prepare-data-op" in tasks
        assert "sonic-train-op" in tasks
        assert "sonic-export-onnx-op" in tasks
        assert "sonic-validate-onnx-op" in tasks
        assert "sonic-register-model-op" in tasks

    def test_train_depends_on_data_prep(self):
        """Train task depends on data prep output."""
        spec = _compile_pipeline()
        train = spec["root"]["dag"]["tasks"]["sonic-train-op"]
        assert "sonic-prepare-data-op" in train.get("dependentTasks", [])

    def test_export_depends_on_train(self):
        """Export task depends on train output."""
        spec = _compile_pipeline()
        export = spec["root"]["dag"]["tasks"]["sonic-export-onnx-op"]
        assert "sonic-train-op" in export.get("dependentTasks", [])

    def test_validate_depends_on_export(self):
        """Validate task depends on export task."""
        spec = _compile_pipeline()
        validate = spec["root"]["dag"]["tasks"]["sonic-validate-onnx-op"]
        deps = validate.get("dependentTasks", [])
        assert "sonic-export-onnx-op" in deps

    def test_register_depends_on_validate(self):
        """Register task depends on validate task."""
        spec = _compile_pipeline()
        register = spec["root"]["dag"]["tasks"]["sonic-register-model-op"]
        deps = register.get("dependentTasks", [])
        assert "sonic-validate-onnx-op" in deps


class TestPipelineParameters:
    """Validate pipeline parameter defaults."""

    def test_default_num_gpus(self):
        """Default num_gpus is 4."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["num_gpus"]["defaultValue"] == 4.0

    def test_default_num_envs(self):
        """Default num_envs is 4096."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["num_envs"]["defaultValue"] == 4096.0

    def test_default_max_iterations(self):
        """Default max_iterations is 10000."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["max_iterations"]["defaultValue"] == 10000.0

    def test_default_data_prefix(self):
        """Default s3_data_prefix is bones-seed/processed."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["s3_data_prefix"]["defaultValue"] == "bones-seed/processed"

    def test_default_checkpoint_prefix(self):
        """Default s3_checkpoint_prefix is sonic-checkpoints."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["s3_checkpoint_prefix"]["defaultValue"] == "sonic-checkpoints"

    def test_default_hydra_experiment(self):
        """Default hydra_experiment is sonic_release."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["hydra_experiment"]["defaultValue"] == "sonic_release"


class TestGPUConfiguration:
    """Validate GPU and resource configuration in compiled YAML."""

    def test_gpu_in_compiled_yaml(self):
        """Compiled YAML references nvidia.com/gpu."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "nvidia.com/gpu" in yaml_str

    def test_minio_secret_referenced(self):
        """Compiled YAML references the minio-credentials secret."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "minio-credentials" in yaml_str

    def test_hf_credentials_secret_referenced(self):
        """Compiled YAML references the hf-credentials secret."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "hf-credentials" in yaml_str

    def test_toleration_referenced(self):
        """Compiled YAML contains GPU toleration."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "NoSchedule" in yaml_str

    def test_env_vars_in_compiled_yaml(self):
        """Compiled YAML contains required environment variables."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "ACCEPT_EULA" in yaml_str
        assert "S3_ENDPOINT" in yaml_str
        assert "MLFLOW_TRACKING_URI" in yaml_str

    def test_sonic_image_referenced(self):
        """Compiled YAML uses the SONIC container image."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "isaaclab-g1-sonic" in yaml_str

    def test_caching_disabled_for_train(self):
        """Train task has caching disabled."""
        spec = _compile_pipeline()
        train = spec["root"]["dag"]["tasks"]["sonic-train-op"]
        caching = train.get("cachingOptions", {})
        assert caching.get("enableCache") is not True

    def test_caching_disabled_for_data_prep(self):
        """Data prep task has caching disabled (uses its own S3 manifest caching)."""
        spec = _compile_pipeline()
        data = spec["root"]["dag"]["tasks"]["sonic-prepare-data-op"]
        caching = data.get("cachingOptions", {})
        assert caching.get("enableCache") is not True
