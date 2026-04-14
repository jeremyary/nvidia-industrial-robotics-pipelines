-include .env
export

IMAGE ?= quay.io/jary/isaaclab-g1-train
SONIC_IMAGE ?= quay.io/jary/isaaclab-g1-sonic
TAG ?= latest
NAMESPACE ?= wbc-training

# Model Registry Postgres credentials (override in .env)
MODEL_REGISTRY_DB_USER ?= mlmduser
MODEL_REGISTRY_DB_PASSWORD ?= mlmdpass
MODEL_REGISTRY_DB_NAME ?= wbc_model_registry

# ── Job registry ────────────────────────────────────────────────────
# Map JOB names to YAML files and K8s job names.
# Add new jobs here — no new targets needed.
#
#   make job-deploy JOB=test-flat
#   make job-logs JOB=test-flat
#   make job-clean JOB=test-flat

JOB_FILE_smoke-test              = deploy/jobs/smoke-test.yaml
JOB_NAME_smoke-test              = smoke-test
JOB_NEEDS_INFRA_smoke-test       = false

JOB_FILE_test-flat               = deploy/jobs/test-flat.yaml
JOB_NAME_test-flat               = test-flat
JOB_NEEDS_INFRA_test-flat        = false

JOB_FILE_test-rough              = deploy/jobs/test-rough.yaml
JOB_NAME_test-rough              = test-rough
JOB_NEEDS_INFRA_test-rough       = false

JOB_FILE_test-warehouse          = deploy/jobs/test-warehouse.yaml
JOB_NAME_test-warehouse          = test-warehouse
JOB_NEEDS_INFRA_test-warehouse   = false

JOB_FILE_test-preset             = deploy/jobs/test-preset.yaml
JOB_NAME_test-preset             = test-preset
JOB_NEEDS_INFRA_test-preset      = false

JOB_FILE_training-flat-6k        = deploy/jobs/training-flat-6k.yaml
JOB_NAME_training-flat-6k        = training-flat-6k
JOB_NEEDS_INFRA_training-flat-6k = true

JOB_FILE_training-rough-6k      = deploy/jobs/training-rough-6k.yaml
JOB_NAME_training-rough-6k      = training-rough-6k
JOB_NEEDS_INFRA_training-rough-6k = true

JOB_FILE_training-warehouse-6k  = deploy/jobs/training-warehouse-6k.yaml
JOB_NAME_training-warehouse-6k  = training-warehouse-6k
JOB_NEEDS_INFRA_training-warehouse-6k = true

JOB_FILE_training-preset-6k     = deploy/jobs/training-preset-6k.yaml
JOB_NAME_training-preset-6k     = training-preset-6k
JOB_NEEDS_INFRA_training-preset-6k = true

JOB_FILE_sonic-data-prep        = deploy/jobs/sonic-data-prep.yaml
JOB_NAME_sonic-data-prep        = sonic-data-prep
JOB_NEEDS_INFRA_sonic-data-prep = true

JOB_FILE_sonic-smoke-test       = deploy/jobs/sonic-smoke-test.yaml
JOB_NAME_sonic-smoke-test       = sonic-smoke-test
JOB_NEEDS_INFRA_sonic-smoke-test = false

JOB_FILE_sonic-test-l40s        = deploy/jobs/sonic-test-l40s.yaml
JOB_NAME_sonic-test-l40s        = sonic-test-l40s
JOB_NEEDS_INFRA_sonic-test-l40s = true

JOB_FILE_sonic-training         = deploy/jobs/sonic-training.yaml
JOB_NAME_sonic-training         = sonic-training
JOB_NEEDS_INFRA_sonic-training  = true

JOB_FILE_sonic-training-l40s    = deploy/jobs/sonic-training-l40s.yaml
JOB_NAME_sonic-training-l40s    = sonic-training-l40s
JOB_NEEDS_INFRA_sonic-training-l40s = true

# Resolve JOB variable to file/name/infra-flag
_JOB_FILE       = $(JOB_FILE_$(JOB))
_JOB_NAME       = $(JOB_NAME_$(JOB))
_JOB_NEEDS_INFRA = $(JOB_NEEDS_INFRA_$(JOB))

.PHONY: build push ngc-login local-smoke-test \
        build-sonic push-sonic \
        deploy-infra deploy-pytorchjob-infra deploy-model-registry \
        job-deploy job-logs job-clean job-list \
        pipeline-compile pipeline-compile-distributed \
        sonic-pipeline-compile sonic-pipeline-compile-distributed \
        pipeline-deploy \
        lint test

# ── Container ────────────────────────────────────────────────────────
ngc-login:
	@echo "$(NGC_API_KEY)" | podman login nvcr.io -u '$$oauthtoken' --password-stdin

build: ngc-login
	podman build --format docker -t $(IMAGE):$(TAG) -f Containerfile .

push: build
	podman push $(IMAGE):$(TAG)

build-sonic: ngc-login
	podman build --format docker -t $(SONIC_IMAGE):$(TAG) -f Containerfile.sonic .

push-sonic: build-sonic
	podman push $(SONIC_IMAGE):$(TAG)

# ── Local GPU smoke test (Podman + CDI) ──────────────────────────────
local-smoke-test:
	podman run --rm --device nvidia.com/gpu=all --env ACCEPT_EULA=Y --env PYTHONUNBUFFERED=1 \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.train \
		--task WBC-Velocity-Flat-G1-29DOF-v0 --headless --num_envs 64 --max_iterations 10

# ── OCP infrastructure ──────────────────────────────────────────────
deploy-infra:
	oc apply -f deploy/infra/namespace.yaml
	oc project $(NAMESPACE)
	oc apply -f deploy/infra/gpu-scc.yaml
	oc apply -f deploy/infra/minio.yaml
	oc apply -f deploy/infra/mlflow.yaml
	oc apply -f deploy/infra/mlflow-rbac.yaml
	oc delete job minio-init -n $(NAMESPACE) --ignore-not-found
	oc apply -f deploy/infra/minio-init.yaml
	oc wait --for=condition=complete job/minio-init -n $(NAMESPACE) --timeout=120s
	oc apply -f deploy/infra/dspa.yaml
	@echo "Waiting for DSPA to be ready..."
	oc wait --for=condition=Ready dspa/dspa -n $(NAMESPACE) --timeout=300s
	oc apply -f deploy/infra/dspa-rbac.yaml
	@echo "DSPA deployed. Verify runner SA: oc get sa -n $(NAMESPACE) | grep dspa"
ifdef HF_TOKEN
	oc create secret generic hf-credentials -n $(NAMESPACE) \
		--from-literal=HF_TOKEN=$(HF_TOKEN) \
		--dry-run=client -o yaml | oc apply -f -
	@echo "HF credentials secret created/updated."
endif

# ── Model Registry infrastructure ──────────────────────────────────
deploy-model-registry:
	oc create secret generic wbc-model-registry-db -n rhoai-model-registries \
		--from-literal=POSTGRES_USER=$(MODEL_REGISTRY_DB_USER) \
		--from-literal=POSTGRES_PASSWORD=$(MODEL_REGISTRY_DB_PASSWORD) \
		--from-literal=POSTGRES_DB=$(MODEL_REGISTRY_DB_NAME) \
		--dry-run=client -o yaml | oc apply -f -
	oc apply -f deploy/infra/model-registry.yaml
	@echo "Model Registry deployed. Secret created from MODEL_REGISTRY_DB_* vars."

# ── PyTorchJob infrastructure ──────────────────────────────────────
deploy-pytorchjob-infra:
	oc apply -f deploy/infra/training-operator-rbac.yaml
	WBC_NAMESPACE=$(NAMESPACE) envsubst < deploy/infra/kueue.yaml | oc apply -f -
	@echo "PyTorchJob RBAC and Kueue resources deployed."

# ── Parametric job management ────────────────────────────────────────
#
# Usage:
#   make job-deploy JOB=training-flat-6k    # deploy infra (if needed) + create job
#   make job-logs JOB=training-flat-6k      # tail logs
#   make job-clean JOB=training-flat-6k     # delete job
#   make job-list                           # show available JOB names

job-deploy:
ifndef JOB
	$(error JOB is required. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_FILE),)
	$(error Unknown JOB '$(JOB)'. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_NEEDS_INFRA),true)
	@$(MAKE) --no-print-directory deploy-infra
else
	oc apply -f deploy/infra/namespace.yaml
	oc project $(NAMESPACE)
	oc apply -f deploy/infra/gpu-scc.yaml
endif
	oc delete job $(_JOB_NAME) -n $(NAMESPACE) --ignore-not-found
	oc apply -f $(_JOB_FILE)

job-logs:
ifndef JOB
	$(error JOB is required. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_NAME),)
	$(error Unknown JOB '$(JOB)'. Run 'make job-list' to see available jobs)
endif
	oc logs -f job/$(_JOB_NAME) -n $(NAMESPACE)

job-clean:
ifndef JOB
	$(error JOB is required. Run 'make job-list' to see available jobs)
endif
ifeq ($(_JOB_NAME),)
	$(error Unknown JOB '$(JOB)'. Run 'make job-list' to see available jobs)
endif
	oc delete job $(_JOB_NAME) -n $(NAMESPACE) --ignore-not-found

job-list:
	@echo "Available jobs (use with JOB=<name>):"
	@echo ""
	@echo "  Quick tests (10 iters, 64 envs):"
	@echo "    smoke-test           - container sanity check"
	@echo "    test-flat            - flat terrain, obs=103"
	@echo "    test-rough           - rough terrain, obs=290"
	@echo "    test-warehouse       - warehouse scene + lidar, obs=466"
	@echo "    test-preset          - Isaac Lab stock preset, action_scale=0.5"
	@echo ""
	@echo "  Training (6000 iters, 4096 envs + S3 + ONNX):"
	@echo "    training-flat-6k     - flat terrain"
	@echo "    training-rough-6k    - rough terrain"
	@echo "    training-warehouse-6k - warehouse scene"
	@echo "    training-preset-6k   - Isaac Lab stock preset"
	@echo ""
	@echo "  SONIC (GEAR-SONIC motion-tracking):"
	@echo "    sonic-smoke-test     - container + import sanity check"
	@echo "    sonic-data-prep      - BONES-SEED download + CSV-to-PKL (one-time)"
	@echo "    sonic-test-l40s      - L40S validation (1 GPU, 512 envs, 100 iters)"
	@echo "    sonic-training       - production (4 GPUs, 4096 envs, 10K iters)"
	@echo "    sonic-training-l40s  - L40S training (1 GPU, 512 envs, 100 iters)"

# ── Pipeline ────────────────────────────────────────────────────────
pipeline-compile:
	python -m wbc_pipeline.pipeline

pipeline-compile-distributed:
	python -c "from kfp import compiler; from wbc_pipeline.pipeline import wbc_training_pytorchjob_pipeline; compiler.Compiler().compile(wbc_training_pytorchjob_pipeline, 'wbc_training_pytorchjob_pipeline.yaml'); print('Pipeline compiled to wbc_training_pytorchjob_pipeline.yaml')"

sonic-pipeline-compile:
	python -m wbc_pipeline.sonic.pipeline

sonic-pipeline-compile-distributed:
	python -c "from kfp import compiler; from wbc_pipeline.sonic.pipeline import sonic_training_pytorchjob_pipeline; compiler.Compiler().compile(sonic_training_pytorchjob_pipeline, 'sonic_training_pytorchjob_pipeline.yaml'); print('Pipeline compiled to sonic_training_pytorchjob_pipeline.yaml')"

pipeline-deploy:
	oc apply -f deploy/infra/dspa.yaml
	@echo "Waiting for DSPA to be ready..."
	oc wait --for=condition=Ready dspa/dspa -n $(NAMESPACE) --timeout=300s
	oc apply -f deploy/infra/dspa-rbac.yaml
	@echo "DSPA deployed. Access pipeline UI via RHOAI dashboard."

# ── Development ──────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v
