# This project was developed with assistance from AI tools.
-include .env
export

NAMESPACE ?= wbc-training

VLA_IMAGE ?= quay.io/jary/wbc-vla
VLA_TAG ?= latest
SONIC_IMAGE ?= quay.io/jary/wbc-sonic
SONIC_TAG ?= latest
GALLERY_IMAGE ?= quay.io/jary/wbc-gallery
GALLERY_TAG ?= latest

.PHONY: build-vla push-vla build-sonic push-sonic build-gallery push-gallery \
        deploy-infra deploy-model-registry \
        vla-pipeline-compile sonic-pipeline-compile pipeline-deploy \
        lint test

# ── VLA container (GR00T N1.7 fine-tuning) ──────────────────────────
build-vla:
	podman build --format docker -t $(VLA_IMAGE):$(VLA_TAG) -f Containerfile.vla .

push-vla:
	podman push $(VLA_IMAGE):$(VLA_TAG)

# ── SONIC container (checkpoint import) ─────────────────────────────
build-sonic:
	podman build --format docker -t $(SONIC_IMAGE):$(SONIC_TAG) -f Containerfile.sonic .

push-sonic:
	podman push $(SONIC_IMAGE):$(SONIC_TAG)

# ── Gallery container (training video browser) ──────────────────────
build-gallery:
	podman build --format docker -t $(GALLERY_IMAGE):$(GALLERY_TAG) -f Containerfile.gallery .

push-gallery:
	podman push $(GALLERY_IMAGE):$(GALLERY_TAG)

# ── OCP infrastructure ──────────────────────────────────────────────
deploy-infra:
	oc apply -f deploy/infra/namespace.yaml
	oc project $(NAMESPACE)
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
	oc apply -f deploy/infra/kueue.yaml
	oc apply -f deploy/infra/gallery.yaml
	@$(MAKE) --no-print-directory deploy-model-registry
	@echo "Infrastructure deployed."

deploy-model-registry:
	oc apply -f deploy/infra/model-registry.yaml
	@echo "Model Registry deployed."

# ── Pipeline compilation ────────────────────────────────────────────
vla-pipeline-compile:
	python -m wbc_pipeline.vla.pipeline

sonic-pipeline-compile:
	python -m wbc_pipeline.sonic.pipeline

pipeline-deploy:
	oc apply -f deploy/infra/dspa.yaml
	@echo "Waiting for DSPA to be ready..."
	oc wait --for=condition=Ready dspa/dspa -n $(NAMESPACE) --timeout=300s
	oc apply -f deploy/infra/dspa-rbac.yaml
	@echo "DSPA deployed. Access pipeline UI via RHOAI dashboard."

# ── Development ─────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

test:
	pytest tests/ -v
