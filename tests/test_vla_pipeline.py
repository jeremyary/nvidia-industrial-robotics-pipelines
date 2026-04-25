# This project was developed with assistance from AI tools.
"""Tests for VLA fine-tuning pipeline compilation and structure."""

from __future__ import annotations

from pipeline_test_utils import compile_pipeline, compile_pipeline_full_yaml

from wbc_pipeline.vla.pipeline import vla_finetune_pipeline


def _compile_pipeline() -> dict:
    return compile_pipeline(vla_finetune_pipeline)


def _compile_pipeline_full_yaml() -> str:
    return compile_pipeline_full_yaml(vla_finetune_pipeline)


class TestPipelineCompilation:
    """Validate VLA pipeline compiles and has correct structure."""

    def test_compiles_to_valid_yaml(self):
        """Pipeline compiles without errors."""
        spec = _compile_pipeline()
        assert "root" in spec

    def test_has_four_tasks(self):
        """Pipeline contains data_prep, fine_tune, validate, and register tasks."""
        spec = _compile_pipeline()
        tasks = spec["root"]["dag"]["tasks"]
        assert len(tasks) == 4
        assert "vla-data-prep-op" in tasks
        assert "vla-fine-tune-and-export-op" in tasks
        assert "vla-validate-onnx-op" in tasks
        assert "vla-register-model-op" in tasks

    def test_fine_tune_depends_on_data_prep(self):
        """Fine-tune task depends on data prep output."""
        spec = _compile_pipeline()
        fine_tune = spec["root"]["dag"]["tasks"]["vla-fine-tune-and-export-op"]
        assert "vla-data-prep-op" in fine_tune.get("dependentTasks", [])

    def test_validate_depends_on_fine_tune(self):
        """Validate task depends on fine-tune output."""
        spec = _compile_pipeline()
        validate = spec["root"]["dag"]["tasks"]["vla-validate-onnx-op"]
        assert "vla-fine-tune-and-export-op" in validate.get("dependentTasks", [])

    def test_register_depends_on_validate(self):
        """Register task depends on validate task."""
        spec = _compile_pipeline()
        register = spec["root"]["dag"]["tasks"]["vla-register-model-op"]
        deps = register.get("dependentTasks", [])
        assert "vla-validate-onnx-op" in deps


class TestPipelineParameters:
    """Validate pipeline parameter defaults."""

    def test_default_base_model_repo(self):
        """Default base model is nvidia/GR00T-N1.7-3B."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["base_model_repo"]["defaultValue"] == "nvidia/GR00T-N1.7-3B"

    def test_default_embodiment_tag(self):
        """Default embodiment tag is UNITREE_G1."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["embodiment_tag"]["defaultValue"] == "UNITREE_G1"

    def test_default_dataset_repo(self):
        """Default dataset is NVIDIA G1 teleoperation data."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["dataset_repo"]["defaultValue"] == "nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1"

    def test_default_max_steps(self):
        """Default max steps is 2000."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["max_steps"]["defaultValue"] == 2000.0

    def test_default_num_gpus(self):
        """Default GPU count matches GPU_LIMIT."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["num_gpus"]["defaultValue"] == 1.0

    def test_default_global_batch_size(self):
        """Default batch size is 64 for B200."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["global_batch_size"]["defaultValue"] == 64.0

    def test_default_model_name(self):
        """Default model name is g1-vla-finetune."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["model_name"]["defaultValue"] == "g1-vla-finetune"

    def test_default_model_version(self):
        """Default model version is v1."""
        spec = _compile_pipeline()
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert params["model_version"]["defaultValue"] == "v1"


class TestVlaConfiguration:
    """Validate resource and secret configuration in compiled YAML."""

    def test_gpu_referenced_for_fine_tune(self):
        """Fine-tune step requests GPU resources."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "nvidia.com/gpu" in yaml_str

    def test_minio_secret_referenced(self):
        """Compiled YAML references the minio-credentials secret."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "minio-credentials" in yaml_str

    def test_hf_credentials_secret_referenced(self):
        """Compiled YAML references the hf-credentials secret for data prep."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "hf-credentials" in yaml_str

    def test_model_registry_env_var(self):
        """Compiled YAML contains MODEL_REGISTRY_ADDRESS."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "MODEL_REGISTRY_ADDRESS" in yaml_str

    def test_vla_image_referenced(self):
        """Compiled YAML uses the VLA container image."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "wbc-vla" in yaml_str

    def test_caching_disabled_for_all_steps(self):
        """All steps have caching disabled."""
        spec = _compile_pipeline()
        tasks = spec["root"]["dag"]["tasks"]
        for task_name, task_spec in tasks.items():
            caching = task_spec.get("cachingOptions", {})
            assert caching.get("enableCache") is not True, f"{task_name} has caching enabled"

    def test_uses_python_not_isaac_sim(self):
        """VLA pipeline uses plain python, not Isaac Sim python.sh."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "python.sh" not in yaml_str
        assert "python -m wbc_pipeline" in yaml_str

    def test_uses_torchrun_not_accelerate(self):
        """Fine-tune step uses torchrun (via fine_tune.py), not accelerate."""
        yaml_str = _compile_pipeline_full_yaml()
        assert "accelerate" not in yaml_str
