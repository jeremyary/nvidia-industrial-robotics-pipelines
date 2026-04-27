#!/bin/bash
set -euo pipefail

NAMESPACE=wbc-training
QUAY_USER="${QUAY_USER:-jary}"

echo "=== Step 1: Create namespace ==="
oc apply -f deploy/infra/namespace.yaml 2>/dev/null || true
oc project "$NAMESPACE"

echo ""
echo "=== Step 2: Grant privileged SCC to builder SA ==="
oc apply -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vla-builder
  namespace: wbc-training
EOF
oc adm policy add-scc-to-user privileged "system:serviceaccount:${NAMESPACE}:vla-builder" 2>/dev/null

echo ""
echo "=== Step 3: Create quay.io push credentials ==="
if [ -z "${QUAY_TOKEN:-}" ]; then
  echo "Set QUAY_TOKEN env var with your quay.io robot token or password."
  echo "  export QUAY_TOKEN=<your-token>"
  echo "  bash build-vla-arm64.sh"
  exit 1
fi
oc create secret docker-registry quay-credentials -n "$NAMESPACE" \
  --docker-server=quay.io \
  --docker-username="$QUAY_USER" \
  --docker-password="$QUAY_TOKEN" \
  --dry-run=client -o yaml | oc apply -f -

echo ""
echo "=== Step 4: Start build Job ==="
oc delete job vla-build-image -n "$NAMESPACE" --ignore-not-found
oc apply -f deploy/jobs/vla/build-image.yaml

echo ""
echo "=== Step 5: Follow build logs ==="
echo "Waiting for build pod to start..."
oc wait --for=condition=Ready pod -l pipeline.wbc/phase=build -n "$NAMESPACE" --timeout=300s 2>/dev/null || sleep 30
BUILD_POD=$(oc get pods -n "$NAMESPACE" -l pipeline.wbc/phase=build -o name | head -1)
echo "Pod: $BUILD_POD"
echo "Streaming logs (this will take 30-60 min)..."
oc logs -f "$BUILD_POD" -n "$NAMESPACE" 2>/dev/null || echo "(logs ended, check: oc logs $BUILD_POD -n $NAMESPACE)"
