#!/bin/bash
# This project was developed with assistance from AI tools.
# VLA scaling trials: throughput benchmarks + convergence runs on GB200 cluster.
# Submits bare K8s Jobs with varying hyperparameters, collects timing results.
#
# Usage: bash deploy/jobs/vla/run-trials.sh [--phase 1|2|all] [--trial T1|C1|...]
set -euo pipefail

NAMESPACE="${NAMESPACE:-wbc-training}"
RESULTS_FILE="trial-results.tsv"
PHASE="${1:---phase}"
PHASE_VAL="${2:-all}"

# Handle --phase flag
if [[ "$PHASE" == "--phase" ]]; then
    PHASE_VAL="${PHASE_VAL}"
elif [[ "$PHASE" == "--trial" ]]; then
    SINGLE_TRIAL="${PHASE_VAL}"
    PHASE_VAL="single"
fi

MANIFEST_DIR="deploy/jobs/vla"
DIST_MANIFEST="${MANIFEST_DIR}/fine-tune-distributed.yaml"
SINGLE_MANIFEST="${MANIFEST_DIR}/fine-tune.yaml"

log() { echo "=== $(date '+%H:%M:%S') $* ==="; }

check_multi_node() {
    local ready_gpu_nodes
    ready_gpu_nodes=$(oc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\t"}{.spec.taints[*].key}{"\n"}{end}' \
        | grep -v 'unreachable\|not-ready\|unschedulable' \
        | awk -F'\t' '$2+0 >= 4 {n++} END {print n+0}')
    echo "${ready_gpu_nodes}"
}

wait_for_pod() {
    local label="$1"
    local job_name="$2"
    log "Waiting for ${job_name} pod to schedule..."
    while [ -z "$(oc get pods -l "${label}" -n "${NAMESPACE}" -o name 2>/dev/null)" ]; do sleep 2; done
    while [ "$(oc get pods -l "job-name=${job_name}" -n "${NAMESPACE}" -o jsonpath='{.items[0].status.phase}')" = "Pending" ]; do sleep 3; done
}

extract_timing() {
    local job_name="$1"
    oc logs "job/${job_name}" -n "${NAMESPACE}" 2>/dev/null | grep "Training wall-clock" | tail -1 | sed 's/.*: \([0-9.]*\)s.*/\1/'
}

extract_steps_per_sec() {
    local job_name="$1"
    oc logs "job/${job_name}" -n "${NAMESPACE}" 2>/dev/null | grep "Training wall-clock" | tail -1 | sed 's/.* (\([0-9.]*\) steps.*/\1/'
}

run_single_node_trial() {
    local trial="$1" gpus="$2" batch="$3" steps="$4" lr="$5" skip_export="$6"

    log "Trial ${trial}: ${gpus} GPUs, batch=${batch}, steps=${steps}, lr=${lr}"

    oc delete job vla-trial -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
    sleep 2

    local cmd_args="--num-gpus ${gpus} --global-batch-size ${batch} --max-steps ${steps} --learning-rate ${lr}"
    if [[ "${skip_export}" == "true" ]]; then
        cmd_args="${cmd_args} --skip-export"
    fi

    cat <<EOF | oc apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: vla-trial
  namespace: ${NAMESPACE}
  labels:
    pipeline.wbc/component: vla
    pipeline.wbc/phase: trial
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 14400
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        pipeline.wbc/component: vla
        pipeline.wbc/phase: trial
    spec:
      restartPolicy: Never
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
      containers:
        - name: fine-tune
          image: quay.io/jary/wbc-vla:v1-arm64
          imagePullPolicy: Always
          command: ["/bin/sh", "-c"]
          args:
            - |
              python -m wbc_pipeline.vla.fine_tune_distributed ${cmd_args} --trial-name ${trial}
          env:
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: S3_ENDPOINT
              value: "http://minio.wbc-training.svc.cluster.local:9000"
            - name: S3_BUCKET
              value: "wbc-training"
            - name: VLA_S3_MODEL_PREFIX
              value: "vla-finetune/base-model"
            - name: VLA_S3_DATASET_PREFIX
              value: "vla-finetune/dataset"
            - name: VLA_S3_CHECKPOINT_PREFIX
              value: "vla-finetune"
            - name: HOME
              value: /tmp
            - name: TRITON_CACHE_DIR
              value: /tmp/.triton
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: MINIO_ROOT_USER
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: MINIO_ROOT_PASSWORD
          envFrom:
            - secretRef:
                name: hf-credentials
          resources:
            requests:
              cpu: "14"
              memory: 64Gi
              nvidia.com/gpu: "${gpus}"
            limits:
              memory: 150Gi
              nvidia.com/gpu: "${gpus}"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
EOF

    wait_for_pod "pipeline.wbc/phase=trial" "vla-trial"
    oc logs -f job/vla-trial -n "${NAMESPACE}" || true
    oc wait --for=condition=complete job/vla-trial -n "${NAMESPACE}" --timeout=14400s

    local wall_time steps_per_sec
    wall_time=$(extract_timing "vla-trial")
    steps_per_sec=$(extract_steps_per_sec "vla-trial")

    echo -e "${trial}\t1\t${gpus}\t${batch}\t${steps}\t${lr}\t${wall_time:-N/A}\t${steps_per_sec:-N/A}" >> "${RESULTS_FILE}"
    log "Trial ${trial} complete: ${wall_time:-N/A}s (${steps_per_sec:-N/A} steps/s)"
}

run_distributed_trial() {
    local trial="$1" gpus_per_node="$2" batch="$3" steps="$4" lr="$5" skip_export="$6"
    local total_gpus=$((gpus_per_node * 2))

    local available_nodes
    available_nodes=$(check_multi_node)
    if [[ "${available_nodes}" -lt 2 ]]; then
        log "SKIP Trial ${trial}: need 2 GPU nodes, only ${available_nodes} available"
        echo -e "${trial}\t2\t${total_gpus}\t${batch}\t${steps}\t${lr}\tSKIPPED\tSKIPPED" >> "${RESULTS_FILE}"
        return 0
    fi

    log "Trial ${trial}: 2 nodes x ${gpus_per_node} GPUs = ${total_gpus} total, batch=${batch}, steps=${steps}, lr=${lr}"

    oc delete job vla-trial-node-0 vla-trial-node-1 -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
    oc delete svc vla-trial-dist -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
    sleep 2

    local cmd_args="--num-gpus ${gpus_per_node} --num-nodes 2 --global-batch-size ${batch} --max-steps ${steps} --learning-rate ${lr} --rdzv-endpoint node-0.vla-trial-dist.${NAMESPACE}.svc.cluster.local:29500"
    if [[ "${skip_export}" == "true" ]]; then
        cmd_args="${cmd_args} --skip-export"
    fi

    cat <<EOF | oc apply -f -
apiVersion: v1
kind: Service
metadata:
  name: vla-trial-dist
  namespace: ${NAMESPACE}
spec:
  clusterIP: None
  selector:
    pipeline.wbc/component: vla
    pipeline.wbc/phase: trial-distributed
  ports:
    - name: rdzv
      port: 29500
    - name: nccl
      port: 29400
---
apiVersion: batch/v1
kind: Job
metadata:
  name: vla-trial-node-0
  namespace: ${NAMESPACE}
  labels:
    pipeline.wbc/component: vla
    pipeline.wbc/phase: trial-distributed
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 14400
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        pipeline.wbc/component: vla
        pipeline.wbc/phase: trial-distributed
    spec:
      hostname: node-0
      subdomain: vla-trial-dist
      restartPolicy: Never
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  pipeline.wbc/phase: trial-distributed
              topologyKey: kubernetes.io/hostname
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
      containers:
        - name: fine-tune
          image: quay.io/jary/wbc-vla:v1-arm64
          imagePullPolicy: Always
          command: ["/bin/sh", "-c"]
          args:
            - |
              python -m wbc_pipeline.vla.fine_tune_distributed ${cmd_args} --rank-zero-only true --trial-name ${trial}
          env:
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: NODE_RANK
              value: "0"
            - name: S3_ENDPOINT
              value: "http://minio.wbc-training.svc.cluster.local:9000"
            - name: S3_BUCKET
              value: "wbc-training"
            - name: VLA_S3_MODEL_PREFIX
              value: "vla-finetune/base-model"
            - name: VLA_S3_DATASET_PREFIX
              value: "vla-finetune/dataset"
            - name: VLA_S3_CHECKPOINT_PREFIX
              value: "vla-finetune"
            - name: HOME
              value: /tmp
            - name: TRITON_CACHE_DIR
              value: /tmp/.triton
            - name: NCCL_DEBUG
              value: INFO
            - name: NCCL_IB_DISABLE
              value: "1"
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: MINIO_ROOT_USER
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: MINIO_ROOT_PASSWORD
          envFrom:
            - secretRef:
                name: hf-credentials
          resources:
            requests:
              cpu: "14"
              memory: 64Gi
              nvidia.com/gpu: "${gpus_per_node}"
            limits:
              memory: 150Gi
              nvidia.com/gpu: "${gpus_per_node}"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
---
apiVersion: batch/v1
kind: Job
metadata:
  name: vla-trial-node-1
  namespace: ${NAMESPACE}
  labels:
    pipeline.wbc/component: vla
    pipeline.wbc/phase: trial-distributed
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 14400
  ttlSecondsAfterFinished: 600
  template:
    metadata:
      labels:
        pipeline.wbc/component: vla
        pipeline.wbc/phase: trial-distributed
    spec:
      hostname: node-1
      subdomain: vla-trial-dist
      restartPolicy: Never
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  pipeline.wbc/phase: trial-distributed
              topologyKey: kubernetes.io/hostname
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi
      containers:
        - name: fine-tune
          image: quay.io/jary/wbc-vla:v1-arm64
          imagePullPolicy: Always
          command: ["/bin/sh", "-c"]
          args:
            - |
              python -m wbc_pipeline.vla.fine_tune_distributed ${cmd_args} --rank-zero-only false --trial-name ${trial}
          env:
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: NODE_RANK
              value: "1"
            - name: S3_ENDPOINT
              value: "http://minio.wbc-training.svc.cluster.local:9000"
            - name: S3_BUCKET
              value: "wbc-training"
            - name: VLA_S3_MODEL_PREFIX
              value: "vla-finetune/base-model"
            - name: VLA_S3_DATASET_PREFIX
              value: "vla-finetune/dataset"
            - name: VLA_S3_CHECKPOINT_PREFIX
              value: "vla-finetune"
            - name: HOME
              value: /tmp
            - name: TRITON_CACHE_DIR
              value: /tmp/.triton
            - name: NCCL_DEBUG
              value: INFO
            - name: NCCL_IB_DISABLE
              value: "1"
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: MINIO_ROOT_USER
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: MINIO_ROOT_PASSWORD
          envFrom:
            - secretRef:
                name: hf-credentials
          resources:
            requests:
              cpu: "14"
              memory: 64Gi
              nvidia.com/gpu: "${gpus_per_node}"
            limits:
              memory: 150Gi
              nvidia.com/gpu: "${gpus_per_node}"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
EOF

    wait_for_pod "pipeline.wbc/phase=trial-distributed" "vla-trial-node-0"
    oc logs -f job/vla-trial-node-0 -n "${NAMESPACE}" || true
    oc wait --for=condition=complete job/vla-trial-node-0 job/vla-trial-node-1 -n "${NAMESPACE}" --timeout=14400s

    local wall_time steps_per_sec
    wall_time=$(extract_timing "vla-trial-node-0")
    steps_per_sec=$(extract_steps_per_sec "vla-trial-node-0")

    echo -e "${trial}\t2\t${total_gpus}\t${batch}\t${steps}\t${lr}\t${wall_time:-N/A}\t${steps_per_sec:-N/A}" >> "${RESULTS_FILE}"
    log "Trial ${trial} complete: ${wall_time:-N/A}s (${steps_per_sec:-N/A} steps/s)"
}

# Initialize results file
if [[ ! -f "${RESULTS_FILE}" ]]; then
    echo -e "trial\tnodes\tgpus\tbatch_size\tsteps\tlr\twall_time_s\tsteps_per_sec" > "${RESULTS_FILE}"
fi

log "VLA Scaling Trials — GB200 cluster (2 nodes x 4 GPUs, 189GB VRAM each)"
echo ""

# ── Phase 1: Throughput benchmarks (200 steps, no ONNX export) ──────────
if [[ "${PHASE_VAL}" == "all" || "${PHASE_VAL}" == "1" ]]; then
    log "Phase 1: Throughput benchmarks"
    #         trial  gpus  batch  steps  lr     skip_export
    run_single_node_trial  T1  4  64    200  1e-4   true
    run_single_node_trial  T2  4  256   200  1e-4   true
    run_single_node_trial  T3  4  512   200  1e-4   true
    #              trial  gpus/node  batch  steps  lr     skip_export
    run_distributed_trial  T4  4  256   200  1e-4   true
    run_distributed_trial  T5  4  512   200  1e-4   true
    run_distributed_trial  T6  4  1024  200  1e-4   true
    run_distributed_trial  T7  4  2048  200  1e-4   true
    echo ""
    log "Phase 1 complete. Results in ${RESULTS_FILE}"
fi

# ── Phase 2: Convergence runs (full training + ONNX export) ─────────────
if [[ "${PHASE_VAL}" == "all" || "${PHASE_VAL}" == "2" ]]; then
    log "Phase 2: Convergence runs"
    #         trial  gpus  batch  steps  lr       skip_export
    run_single_node_trial  C1  4  64    2000   1e-4     false
    #              trial  gpus/node  batch  steps  lr       skip_export
    run_distributed_trial  C2  4  128   2000   1.4e-4   false
    run_distributed_trial  C3  4  512   5000   2.8e-4   false
    run_distributed_trial  C4  4  1024  5000   4e-4     false
    run_distributed_trial  C5  4  512   10000  2.8e-4   false
    echo ""
    log "Phase 2 complete. Results in ${RESULTS_FILE}"
fi

echo ""
log "All trials complete."
echo ""
echo "Results:"
column -t -s$'\t' "${RESULTS_FILE}"
