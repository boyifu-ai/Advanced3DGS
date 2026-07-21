#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_REAL:-0}" != "1" ]]; then
  echo "This script resumes all formal Unified 3DGS experiments and rebuilds CSV summaries."
  echo "It uses the registered method catalog and reuses verified completed stages."
  echo "Mip-NeRF 360 flowers and treehill are intentionally excluded."
  echo
  echo "Run:"
  echo "  RUN_REAL=1 CUDA_VISIBLE_DEVICES=5 bash scripts/resume_all_experiments.sh"
  echo
  echo "Set FORCE=1 only when every train/render/eval stage must run again."
  exit 2
fi

if [[ -z "${METHODS:-}" ]]; then
  METHODS="$(
    python - <<'PY'
from unified3dgs.methods.registry import available_methods
print(" ".join(available_methods()))
PY
  )"
  export METHODS
else
  export METHODS
fi
export DATASET_FAMILIES="${DATASET_FAMILIES:-mip360 tandt deep_blending}"
export VALIDATION_ROOT="${VALIDATION_ROOT:-outputs/validation}"
export ITERATIONS="${ITERATIONS:-30000}"
export RESOLUTION="${RESOLUTION:--1}"
export BENCHMARK_PROTOCOL="${BENCHMARK_PROTOCOL:-1}"
export MIN_FREE_GB="${MIN_FREE_GB:-5}"
export FORCE="${FORCE:-0}"
export AGGREGATE_AFTER_EVAL="${AGGREGATE_AFTER_EVAL:-0}"
export AUTO_PATCH_READERS="${AUTO_PATCH_READERS:-1}"
export CHECK_RUNTIME_DEPS="${CHECK_RUNTIME_DEPS:-1}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be set on the shared server." >&2
  echo "Example: RUN_REAL=1 CUDA_VISIBLE_DEVICES=5 bash scripts/resume_all_experiments.sh" >&2
  exit 2
fi

case "${VALIDATION_ROOT}" in
  outputs/validation)
    ;;
  *)
    if [[ "${ALLOW_CUSTOM_VALIDATION_ROOT:-0}" != "1" ]]; then
      echo "Refusing unexpected resume root: ${VALIDATION_ROOT}" >&2
      echo "The formal resume root must be outputs/validation." >&2
      echo "Set ALLOW_CUSTOM_VALIDATION_ROOT=1 only for an intentional custom root." >&2
      exit 2
    fi
    ;;
esac

echo "Unified 3DGS formal experiment resume"
echo "Methods: ${METHODS}"
echo "Dataset families: ${DATASET_FAMILIES}"
echo "Validation root: ${VALIDATION_ROOT}"
echo "Requested iterations: ${ITERATIONS}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "FORCE=${FORCE}"
echo

bash scripts/run_validation.sh

echo
echo "Rebuilding Scene-Level, Dataset-Level, and Method-Level metric summaries..."
python scripts/aggregate_metrics.py \
  --validation-root "${VALIDATION_ROOT}" \
  --iteration "${ITERATIONS}" \
  --levels scene dataset method

echo
echo "Auditing final benchmark outputs..."
python scripts/audit_benchmark_protocol.py \
  --validation-root "${VALIDATION_ROOT}" \
  --output "${VALIDATION_ROOT}/benchmark_protocol_audit.csv" \
  --methods ${METHODS} \
  --dataset-families mip360 tandt deep_blending

echo
echo "All requested experiments and metric summaries are complete."
echo "Method CSV files:"
for method in ${METHODS}; do
  echo "  ${VALIDATION_ROOT}/${method}/metrics_summary.csv"
done
echo "Dataset CSV files:"
echo "  ${VALIDATION_ROOT}/<method>/<dataset>/metrics_summary.csv"
echo "Scene summaries:"
echo "  ${VALIDATION_ROOT}/<method>/<dataset>/<scene>/metrics_summary.json"
