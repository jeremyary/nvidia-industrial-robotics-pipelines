# This project was developed with assistance from AI tools.
"""Tests for the backend-agnostic PyTorchJob launcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wbc_pipeline.pytorchjob import build_pytorchjob


class TestBuildPyTorchJob:
    """Validate PyTorchJob CR construction."""

    def _build_default(self, **overrides):
        defaults = dict(
            name="test-job",
            namespace="wbc-training",
            image="quay.io/jary/isaaclab-g1-train:latest",
            command=["/bin/bash", "-c", "echo hello"],
            num_workers=1,
            gpus_per_worker=1,
        )
        defaults.update(overrides)
        return build_pytorchjob(**defaults)

    def test_metadata(self):
        """Job has correct name, namespace, and labels."""
        job = self._build_default()
        assert job.metadata.name == "test-job"
        assert job.metadata.namespace == "wbc-training"
        assert job.metadata.labels["app"] == "wbc-training"
        assert job.metadata.labels["training-backend"] == "pytorchjob"

    def test_api_version_and_kind(self):
        """Job has correct apiVersion and kind."""
        job = self._build_default()
        assert job.api_version == "kubeflow.org/v1"
        assert job.kind == "PyTorchJob"

    def test_single_worker_master_only(self):
        """With num_workers=1, only Master replica is created."""
        job = self._build_default(num_workers=1)
        replicas = job.spec.pytorch_replica_specs
        assert "Master" in replicas
        assert "Worker" not in replicas
        assert replicas["Master"].replicas == 1

    def test_multi_worker_creates_workers(self):
        """With num_workers>1, Worker replicas are created (num_workers - 1)."""
        job = self._build_default(num_workers=3)
        replicas = job.spec.pytorch_replica_specs
        assert "Master" in replicas
        assert replicas["Master"].replicas == 1
        assert "Worker" in replicas
        assert replicas["Worker"].replicas == 2

    def test_gpu_resources(self):
        """Container requests correct GPU count."""
        job = self._build_default(gpus_per_worker=2)
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.resources.requests["nvidia.com/gpu"] == "2"
        assert container.resources.limits["nvidia.com/gpu"] == "2"

    def test_cpu_and_memory_resources(self):
        """Container requests correct CPU and memory."""
        job = self._build_default(cpu_per_worker="16", memory_per_worker="128Gi")
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.resources.requests["cpu"] == "16"
        assert container.resources.requests["memory"] == "128Gi"

    def test_gpu_toleration(self):
        """Pod spec includes GPU toleration."""
        job = self._build_default()
        tolerations = job.spec.pytorch_replica_specs["Master"].template.spec.tolerations
        assert len(tolerations) == 1
        assert tolerations[0].key == "nvidia.com/gpu"
        assert tolerations[0].operator == "Exists"
        assert tolerations[0].effect == "NoSchedule"

    def test_restart_policy_never(self):
        """Pod restart policy is Never."""
        job = self._build_default()
        assert job.spec.pytorch_replica_specs["Master"].template.spec.restart_policy == "Never"
        assert job.spec.pytorch_replica_specs["Master"].restart_policy == "Never"

    def test_image_and_command(self):
        """Container uses specified image and command."""
        job = self._build_default(
            image="quay.io/jary/test:v1",
            command=["/bin/bash", "-c", "train.sh"],
        )
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.image == "quay.io/jary/test:v1"
        assert container.command == ["/bin/bash", "-c", "train.sh"]

    def test_env_vars(self):
        """Environment variables are set on the container."""
        job = self._build_default(env_vars={"TASK": "flat", "NUM_ENVS": "4096"})
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        env_names = {e.name: e.value for e in container.env}
        assert env_names["TASK"] == "flat"
        assert env_names["NUM_ENVS"] == "4096"

    def test_secret_refs(self):
        """Secret key references are mounted as env vars."""
        secret_refs = {
            "minio-credentials": {"MINIO_ROOT_USER": "AWS_ACCESS_KEY_ID"},
        }
        job = self._build_default(secret_refs=secret_refs)
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        secret_envs = [e for e in container.env if e.value_from is not None]
        assert len(secret_envs) == 1
        assert secret_envs[0].name == "AWS_ACCESS_KEY_ID"
        assert secret_envs[0].value_from.secret_key_ref.name == "minio-credentials"
        assert secret_envs[0].value_from.secret_key_ref.key == "MINIO_ROOT_USER"

    def test_no_env_when_none(self):
        """Container env is None when no env vars or secrets provided."""
        job = self._build_default(env_vars=None, secret_refs=None)
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.env is None

    def test_kueue_queue_label(self):
        """Queue name is set as a label for Kueue integration."""
        job = self._build_default(queue_name="wbc-training-queue")
        assert job.metadata.labels["kueue.x-k8s.io/queue-name"] == "wbc-training-queue"

    def test_no_kueue_label_without_queue(self):
        """No Kueue label when queue_name is not provided."""
        job = self._build_default(queue_name=None)
        assert "kueue.x-k8s.io/queue-name" not in job.metadata.labels

    def test_custom_labels(self):
        """Custom labels are merged into metadata."""
        job = self._build_default(labels={"experiment": "sonic-v2"})
        assert job.metadata.labels["experiment"] == "sonic-v2"
        assert job.metadata.labels["app"] == "wbc-training"

    def test_clean_pod_policy(self):
        """Run policy preserves all pods for post-mortem."""
        job = self._build_default()
        assert job.spec.run_policy.clean_pod_policy == "None"

    def test_container_name(self):
        """Container name is 'pytorch'."""
        job = self._build_default()
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.name == "pytorch"

    def test_image_pull_policy(self):
        """Container image pull policy is Always."""
        job = self._build_default()
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.image_pull_policy == "Always"

    def test_worker_shares_master_spec(self):
        """Worker pods use the same container spec as master."""
        job = self._build_default(num_workers=2, gpus_per_worker=2)
        master_container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        worker_container = job.spec.pytorch_replica_specs["Worker"].template.spec.containers[0]
        assert master_container.image == worker_container.image
        assert master_container.resources.requests == worker_container.resources.requests

    def test_service_account_name(self):
        """Pod spec uses specified service account."""
        job = self._build_default(service_account_name="isaaclab-gpu")
        pod_spec = job.spec.pytorch_replica_specs["Master"].template.spec
        assert pod_spec.service_account_name == "isaaclab-gpu"

    def test_no_service_account_by_default(self):
        """Pod spec has no service account when not specified."""
        job = self._build_default()
        pod_spec = job.spec.pytorch_replica_specs["Master"].template.spec
        assert pod_spec.service_account_name is None

    def test_run_as_user(self):
        """Container security context sets runAsUser."""
        job = self._build_default(run_as_user=0)
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.security_context.run_as_user == 0

    def test_no_security_context_by_default(self):
        """Container has no security context when run_as_user not specified."""
        job = self._build_default()
        container = job.spec.pytorch_replica_specs["Master"].template.spec.containers[0]
        assert container.security_context is None

    def test_num_workers_validation(self):
        """Raises ValueError when num_workers < 1."""
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            self._build_default(num_workers=0)


class TestSubmitAndWait:
    """Validate PyTorchJob submission and polling logic."""

    def _make_job(self, name="test-job"):
        job = MagicMock()
        job.metadata.name = name
        return job

    @patch("kubeflow.training.TrainingClient", create=True)
    def test_returns_true_on_success(self, MockClient):
        """Returns True when job succeeds."""
        from wbc_pipeline.pytorchjob import submit_and_wait

        client = MockClient.return_value
        client.is_job_succeeded.return_value = True

        job = self._make_job()
        result = submit_and_wait(job, namespace="wbc-training", poll_interval=1)

        assert result is True
        client.create_job.assert_called_once_with(job=job, namespace="wbc-training")

    @patch("kubeflow.training.TrainingClient", create=True)
    def test_returns_false_on_failure(self, MockClient):
        """Returns False when job fails."""
        from wbc_pipeline.pytorchjob import submit_and_wait

        client = MockClient.return_value
        client.is_job_succeeded.return_value = False
        client.get_job_conditions.return_value = []

        job = self._make_job()
        result = submit_and_wait(job, namespace="wbc-training", poll_interval=1)

        assert result is False

    @patch("kubeflow.training.TrainingClient", create=True)
    def test_returns_false_on_wait_exception(self, MockClient):
        """Returns False when wait_for_job_conditions raises."""
        from wbc_pipeline.pytorchjob import submit_and_wait

        client = MockClient.return_value
        client.wait_for_job_conditions.side_effect = RuntimeError("timeout")

        job = self._make_job()
        result = submit_and_wait(job, namespace="wbc-training", poll_interval=1)

        assert result is False

    @patch("kubeflow.training.TrainingClient", create=True)
    def test_polls_with_correct_conditions(self, MockClient):
        """Waits for Succeeded or Failed conditions."""
        from wbc_pipeline.pytorchjob import submit_and_wait

        client = MockClient.return_value
        client.is_job_succeeded.return_value = True

        job = self._make_job("my-job")
        submit_and_wait(job, namespace="ns", wait_timeout=600, poll_interval=30)

        call_kwargs = client.wait_for_job_conditions.call_args[1]
        assert call_kwargs["expected_conditions"] == {"Succeeded", "Failed"}
        assert call_kwargs["wait_timeout"] == 600
        assert call_kwargs["polling_interval"] == 30


class TestRunCLI:
    """Validate CLI entry point argument parsing and job construction."""

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=True)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_basic_args(self, mock_build, mock_submit):
        """CLI parses required arguments and builds job."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        with patch(
            "sys.argv",
            [
                "pytorchjob",
                "--name",
                "test-run",
                "--image",
                "quay.io/jary/test:v1",
                "--command",
                "echo hello",
            ],
        ):
            run()

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["name"] == "test-run"
        assert call_kwargs["image"] == "quay.io/jary/test:v1"
        assert call_kwargs["command"] == ["/bin/bash", "-c", "echo hello"]

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=True)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_env_var_propagation(self, mock_build, mock_submit):
        """CLI propagates known env vars to the job."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        env_patch = {
            "S3_ENDPOINT": "http://minio:9000",
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "TASK": "flat-v0",
        }
        with (
            patch.dict("os.environ", env_patch, clear=False),
            patch(
                "sys.argv",
                [
                    "pytorchjob",
                    "--name",
                    "test",
                    "--image",
                    "img:latest",
                    "--command",
                    "train",
                ],
            ),
        ):
            run()

        call_kwargs = mock_build.call_args[1]
        env_vars = call_kwargs["env_vars"]
        assert env_vars["S3_ENDPOINT"] == "http://minio:9000"
        assert env_vars["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
        assert env_vars["TASK"] == "flat-v0"

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=True)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_minio_secret_always_mounted(self, mock_build, mock_submit):
        """CLI always mounts minio-credentials secret."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        with patch(
            "sys.argv",
            ["pytorchjob", "--name", "t", "--image", "i", "--command", "c"],
        ):
            run()

        call_kwargs = mock_build.call_args[1]
        assert "minio-credentials" in call_kwargs["secret_refs"]

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=True)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_hf_secret_when_token_present(self, mock_build, mock_submit):
        """CLI mounts hf-credentials when HF_TOKEN is in environment."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        with (
            patch.dict("os.environ", {"HF_TOKEN": "hf_abc123"}, clear=False),
            patch(
                "sys.argv",
                ["pytorchjob", "--name", "t", "--image", "i", "--command", "c"],
            ),
        ):
            run()

        call_kwargs = mock_build.call_args[1]
        assert "hf-credentials" in call_kwargs["secret_refs"]

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=True)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_no_hf_secret_without_token(self, mock_build, mock_submit):
        """CLI does not mount hf-credentials without HF_TOKEN."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        env = {k: v for k, v in __import__("os").environ.items() if k != "HF_TOKEN"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "sys.argv",
                ["pytorchjob", "--name", "t", "--image", "i", "--command", "c"],
            ),
        ):
            run()

        call_kwargs = mock_build.call_args[1]
        assert "hf-credentials" not in call_kwargs["secret_refs"]

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=False)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_exits_nonzero_on_failure(self, mock_build, mock_submit):
        """CLI exits with code 1 when job fails."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        with (
            pytest.raises(SystemExit, match="1"),
            patch(
                "sys.argv",
                ["pytorchjob", "--name", "t", "--image", "i", "--command", "c"],
            ),
        ):
            run()

    @patch("wbc_pipeline.pytorchjob.submit_and_wait", return_value=True)
    @patch("wbc_pipeline.pytorchjob.build_pytorchjob")
    def test_queue_name_passed(self, mock_build, mock_submit):
        """CLI passes queue-name to build_pytorchjob."""
        from wbc_pipeline.pytorchjob import run

        mock_build.return_value = MagicMock()

        with patch(
            "sys.argv",
            [
                "pytorchjob",
                "--name",
                "t",
                "--image",
                "i",
                "--command",
                "c",
                "--queue-name",
                "wbc-training-queue",
            ],
        ):
            run()

        call_kwargs = mock_build.call_args[1]
        assert call_kwargs["queue_name"] == "wbc-training-queue"


class TestPyTorchJobPipelineCompilation:
    """Validate Tier 2 pipeline variants compile correctly."""

    def test_rslrl_pytorchjob_pipeline_compiles(self):
        """RSL-RL PyTorchJob pipeline compiles without errors."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline

        spec = compile_pipeline(wbc_training_pytorchjob_pipeline)
        assert "root" in spec

    def test_rslrl_pytorchjob_pipeline_has_three_tasks(self):
        """RSL-RL PyTorchJob pipeline has launcher, validate, and register tasks."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline

        spec = compile_pipeline(wbc_training_pytorchjob_pipeline)
        tasks = spec["root"]["dag"]["tasks"]
        assert len(tasks) == 3
        assert "train-and-export-pytorchjob-op" in tasks
        assert "validate-onnx-op" in tasks
        assert "register-model-op" in tasks

    def test_rslrl_pytorchjob_pipeline_has_job_name_param(self):
        """RSL-RL PyTorchJob pipeline has job_name parameter."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline

        spec = compile_pipeline(wbc_training_pytorchjob_pipeline)
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert "job_name" in params

    def test_rslrl_pytorchjob_pipeline_has_queue_name_param(self):
        """RSL-RL PyTorchJob pipeline has queue_name parameter."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline

        spec = compile_pipeline(wbc_training_pytorchjob_pipeline)
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert "queue_name" in params

    def test_sonic_pytorchjob_pipeline_compiles(self):
        """SONIC PyTorchJob pipeline compiles without errors."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.sonic.pipeline import sonic_training_pytorchjob_pipeline

        spec = compile_pipeline(sonic_training_pytorchjob_pipeline)
        assert "root" in spec

    def test_sonic_pytorchjob_pipeline_has_six_tasks(self):
        """SONIC PyTorchJob pipeline has data_prep, launcher, export, validate, register tasks."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.sonic.pipeline import sonic_training_pytorchjob_pipeline

        spec = compile_pipeline(sonic_training_pytorchjob_pipeline)
        tasks = spec["root"]["dag"]["tasks"]
        assert len(tasks) == 5
        assert "sonic-prepare-data-op" in tasks
        assert "sonic-train-pytorchjob-op" in tasks

    def test_sonic_pytorchjob_pipeline_has_worker_params(self):
        """SONIC PyTorchJob pipeline has num_workers and gpus_per_worker params."""
        from pipeline_test_utils import compile_pipeline

        from wbc_pipeline.sonic.pipeline import sonic_training_pytorchjob_pipeline

        spec = compile_pipeline(sonic_training_pytorchjob_pipeline)
        params = spec["root"]["inputDefinitions"]["parameters"]
        assert "num_workers" in params
        assert "gpus_per_worker" in params

    def test_rslrl_pytorchjob_references_pytorchjob_module(self):
        """RSL-RL PyTorchJob launcher references wbc_pipeline.pytorchjob module."""
        from pipeline_test_utils import compile_pipeline_full_yaml

        from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline

        yaml_str = compile_pipeline_full_yaml(wbc_training_pytorchjob_pipeline)
        assert "wbc_pipeline.pytorchjob" in yaml_str

    def test_sonic_pytorchjob_references_pytorchjob_module(self):
        """SONIC PyTorchJob launcher references wbc_pipeline.pytorchjob module."""
        from pipeline_test_utils import compile_pipeline_full_yaml

        from wbc_pipeline.sonic.pipeline import sonic_training_pytorchjob_pipeline

        yaml_str = compile_pipeline_full_yaml(sonic_training_pytorchjob_pipeline)
        assert "wbc_pipeline.pytorchjob" in yaml_str
