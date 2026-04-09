-include .env
export

IMAGE ?= quay.io/jary/isaaclab-g1-train
TAG ?= latest
NAMESPACE ?= wbc-training

# ── Job registry ────────────────────────────────────────────────────
# Map JOB names to YAML files and K8s job names.
# Add new jobs here — no new targets needed.
#
#   make job-deploy JOB=overnight
#   make job-logs JOB=overnight
#   make job-clean JOB=overnight

JOB_FILE_smoke-test           = deploy/jobs/smoke-test.yaml
JOB_NAME_smoke-test           = smoke-test
JOB_NEEDS_INFRA_smoke-test    = false

JOB_FILE_rough-terrain        = deploy/jobs/rough-terrain-test.yaml
JOB_NAME_rough-terrain        = rough-terrain-test
JOB_NEEDS_INFRA_rough-terrain = false

JOB_FILE_isaaclab-preset      = deploy/jobs/isaaclab-preset-test.yaml
JOB_NAME_isaaclab-preset      = isaaclab-preset-test
JOB_NEEDS_INFRA_isaaclab-preset = false

JOB_FILE_training-flat-6k     = deploy/jobs/training-flat-6k.yaml
JOB_NAME_training-flat-6k     = training-flat-6k
JOB_NEEDS_INFRA_training-flat-6k = true

# Resolve JOB variable to file/name/infra-flag
_JOB_FILE       = $(JOB_FILE_$(JOB))
_JOB_NAME       = $(JOB_NAME_$(JOB))
_JOB_NEEDS_INFRA = $(JOB_NEEDS_INFRA_$(JOB))

.PHONY: build push ngc-login smoke-test \
        deploy-namespace deploy-infra \
        job-deploy job-logs job-clean job-list \
        lint test

# ── Container ────────────────────────────────────────────────────────
ngc-login:
	@echo "$(NGC_API_KEY)" | podman login nvcr.io -u '$$oauthtoken' --password-stdin

build: ngc-login
	podman build --format docker -t $(IMAGE):$(TAG) -f Containerfile .

push: build
	podman push $(IMAGE):$(TAG)

# ── Local GPU smoke test (Podman + CDI) ──────────────────────────────
smoke-test:
	podman run --rm --device nvidia.com/gpu=all --env ACCEPT_EULA=Y --env PYTHONUNBUFFERED=1 \
		$(IMAGE):$(TAG) \
		-m wbc_pipeline.train \
		--task WBC-Velocity-Flat-G1-29DOF-v0 --headless --num_envs 64 --max_iterations 10

# ── OCP infrastructure ──────────────────────────────────────────────
deploy-namespace:
	oc apply -f deploy/infra/namespace.yaml
	oc project $(NAMESPACE)

deploy-infra: deploy-namespace
	oc apply -f deploy/infra/gpu-scc.yaml
	oc apply -f deploy/infra/minio.yaml
	oc apply -f deploy/infra/mlflow.yaml
	oc apply -f deploy/infra/mlflow-rbac.yaml
	oc delete job minio-init -n $(NAMESPACE) --ignore-not-found
	oc apply -f deploy/infra/minio-init.yaml
	oc wait --for=condition=complete job/minio-init -n $(NAMESPACE) --timeout=120s

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
	@$(MAKE) --no-print-directory deploy-namespace
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
	@echo "  smoke-test        - 10 iters, 64 envs (no S3/MLflow)"
	@echo "  rough-terrain     - rough terrain validation (10 iters)"
	@echo "  isaaclab-preset   - Isaac Lab preset validation (10 iters)"
	@echo "  training-flat-6k  - 6000 iters, 4096 envs + ONNX + S3"

# ── Development ──────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v
