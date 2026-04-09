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

JOB_FILE_smoke-test           = deploy/smoke-test-job.yaml
JOB_NAME_smoke-test           = isaaclab-smoke-test
JOB_NEEDS_INFRA_smoke-test    = false

JOB_FILE_convergence          = deploy/convergence-test-job.yaml
JOB_NAME_convergence          = convergence-test
JOB_NEEDS_INFRA_convergence   = false

JOB_FILE_phase1               = deploy/phase1-validation-job.yaml
JOB_NAME_phase1               = phase1-validation
JOB_NEEDS_INFRA_phase1        = true

JOB_FILE_phase2               = deploy/phase2-validation-job.yaml
JOB_NAME_phase2               = phase2-validation
JOB_NEEDS_INFRA_phase2        = true

JOB_FILE_phase3               = deploy/phase3-validation-job.yaml
JOB_NAME_phase3               = phase3-validation
JOB_NEEDS_INFRA_phase3        = false

JOB_FILE_phase3b-flat         = deploy/phase3b-validation-jobs.yaml
JOB_NAME_phase3b-flat         = phase3b-flat-regression
JOB_NEEDS_INFRA_phase3b-flat  = false

JOB_FILE_phase3b-rough        = deploy/phase3b-validation-jobs.yaml
JOB_NAME_phase3b-rough        = phase3b-rough-validation
JOB_NEEDS_INFRA_phase3b-rough = false

JOB_FILE_phase3b-isaaclab     = deploy/phase3b-validation-jobs.yaml
JOB_NAME_phase3b-isaaclab     = phase3b-isaaclab-preset
JOB_NEEDS_INFRA_phase3b-isaaclab = false

JOB_FILE_overnight            = deploy/overnight-training-job.yaml
JOB_NAME_overnight            = overnight-extended
JOB_NEEDS_INFRA_overnight     = true

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
	oc apply -f deploy/namespace.yaml
	oc project $(NAMESPACE)

deploy-infra: deploy-namespace
	oc apply -f deploy/gpu-scc.yaml
	oc apply -f deploy/minio.yaml
	oc apply -f deploy/mlflow.yaml
	oc apply -f deploy/mlflow-rbac.yaml
	oc delete job minio-init -n $(NAMESPACE) --ignore-not-found
	oc apply -f deploy/minio-init-job.yaml
	oc wait --for=condition=complete job/minio-init -n $(NAMESPACE) --timeout=120s

# ── Parametric job management ────────────────────────────────────────
#
# Usage:
#   make job-deploy JOB=overnight    # deploy infra (if needed) + create job
#   make job-logs JOB=overnight      # tail logs
#   make job-clean JOB=overnight     # delete job
#   make job-list                    # show available JOB names

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
	oc apply -f deploy/gpu-scc.yaml
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
	@echo "  convergence       - 500 iters, 1024 envs (no S3/MLflow)"
	@echo "  phase1            - 100 iters + ONNX export"
	@echo "  phase2            - 20 iters + S3/MLflow validation"
	@echo "  phase3            - 50 iters + ONNX compat checks"
	@echo "  phase3b-flat      - flat env regression (10 iters)"
	@echo "  phase3b-rough     - rough terrain validation (10 iters)"
	@echo "  phase3b-isaaclab  - Isaac Lab preset validation (10 iters)"
	@echo "  overnight         - 6000 iters, 4096 envs + ONNX + S3"

# ── Development ──────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v
