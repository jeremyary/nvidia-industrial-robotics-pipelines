-include .env
export

SONIC_IMAGE ?= quay.io/jary/wbc-sonic
GALLERY_IMAGE ?= quay.io/jary/wbc-gallery
VLA_IMAGE ?= quay.io/jary/wbc-vla
TAG ?= latest
NAMESPACE ?= wbc-training

# Model Registry Postgres credentials (set in .env)
MODEL_REGISTRY_DB_USER ?= mlmduser
MODEL_REGISTRY_DB_NAME ?= wbc_model_registry

.PHONY: build-sonic push-sonic \
        build-gallery push-gallery \
        build-vla push-vla \
        deploy-infra deploy-model-registry \
        sonic-pipeline-compile vla-pipeline-compile \
        pipeline-deploy \
        lint test

# ── Container ────────────────────────────────────────────────────────
ngc-login:
	@echo "$(NGC_API_KEY)" | podman login nvcr.io -u '$$oauthtoken' --password-stdin

build-sonic:
	podman build --format docker -t $(SONIC_IMAGE):$(TAG) -f Containerfile.sonic .

push-sonic: build-sonic
	podman push $(SONIC_IMAGE):$(TAG)

build-gallery:
	podman build --format docker -t $(GALLERY_IMAGE):$(TAG) -f Containerfile.gallery .

push-gallery: build-gallery
	podman push $(GALLERY_IMAGE):$(TAG)

build-vla:
	podman build --format docker -t $(VLA_IMAGE):$(TAG) -f Containerfile.vla .

push-vla: build-vla
	podman push $(VLA_IMAGE):$(TAG)

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
	@echo "DSPA deployed. Verify runner SA: oc get sa -n $(NAMESPACE) | grep dspa"
ifdef HF_TOKEN
	oc create secret generic hf-credentials -n $(NAMESPACE) \
		--from-literal=HF_TOKEN=$(HF_TOKEN) \
		--dry-run=client -o yaml | oc apply -f -
	@echo "HF credentials secret created/updated."
endif
	WBC_NAMESPACE=$(NAMESPACE) envsubst < deploy/infra/kueue.yaml | oc apply -f -
	@echo "Kueue GPU quota resources deployed."
	oc apply -f deploy/infra/gallery.yaml
	@echo "Video gallery deployed. Route:"
	@oc get route video-gallery -n $(NAMESPACE) -o jsonpath='https://{.spec.host}{"\n"}' 2>/dev/null || echo "  (route pending)"

# ── Model Registry infrastructure ──────────────────────────────────
deploy-model-registry:
ifndef MODEL_REGISTRY_DB_PASSWORD
	$(error MODEL_REGISTRY_DB_PASSWORD is required. Set it in .env)
endif
	oc create secret generic wbc-model-registry-db -n rhoai-model-registries \
		--from-literal=POSTGRES_USER=$(MODEL_REGISTRY_DB_USER) \
		--from-literal=POSTGRES_PASSWORD=$(MODEL_REGISTRY_DB_PASSWORD) \
		--from-literal=POSTGRES_DB=$(MODEL_REGISTRY_DB_NAME) \
		--dry-run=client -o yaml | oc apply -f -
	oc apply -f deploy/infra/model-registry.yaml
	@echo "Model Registry deployed. Secret created from MODEL_REGISTRY_DB_* vars."

# ── Pipeline ────────────────────────────────────────────────────────
sonic-pipeline-compile:
	python -m wbc_pipeline.sonic.pipeline

vla-pipeline-compile:
	python -m wbc_pipeline.vla.pipeline

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
