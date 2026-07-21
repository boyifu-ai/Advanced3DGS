#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_REAL:-0}" != "1" ]]; then
  echo "This script runs the formal 11-scene validation jobs."
  echo "Mip-NeRF 360 uses 7 scenes: bicycle, bonsai, counter, garden, kitchen, room, stump."
  echo "Mip-NeRF 360 flowers and treehill are intentionally excluded from training/statistics."
  echo "Set RUN_REAL=1 to continue."
  echo
  echo "Example for selected methods:"
  echo "  RUN_REAL=1 METHODS=\"vanilla_3dgs 2dgs 3dgs_mcmc 3dhgs sss\" CUDA_VISIBLE_DEVICES=5 \\"
  echo "  bash scripts/run_validation.sh"
  exit 2
fi

if [[ -n "${METHODS:-}" ]]; then
  read -r -a METHODS_TO_RUN <<< "${METHODS}"
else
  mapfile -t METHODS_TO_RUN < <(
    python - <<'PY'
from unified3dgs.methods.registry import available_methods
for method in available_methods():
    print(method)
PY
  )
fi

mapfile -t ALL_DATASET_RECORDS < <(
  python - <<'PY'
from unified3dgs.dataset_config import validation_scene_records

for dataset, label in validation_scene_records():
    print(f"{dataset}|{label}")
PY
)

DATASET_RECORDS=()
if [[ -n "${SCENES:-}" ]]; then
  read -r -a SCENES_TO_RUN <<< "${SCENES}"
  for record in "${ALL_DATASET_RECORDS[@]}"; do
    label="${record#*|}"
    for selected_scene in "${SCENES_TO_RUN[@]}"; do
      if [[ "${label}" == "${selected_scene}" ]]; then
        DATASET_RECORDS+=("${record}")
        break
      fi
    done
  done
elif [[ -n "${DATASET_FAMILIES:-}" ]]; then
  read -r -a DATASET_FAMILIES_TO_RUN <<< "${DATASET_FAMILIES}"
  for record in "${ALL_DATASET_RECORDS[@]}"; do
    label="${record#*|}"
    family="${label%%/*}"
    for selected_family in "${DATASET_FAMILIES_TO_RUN[@]}"; do
      if [[ "${family}" == "${selected_family}" ]]; then
        DATASET_RECORDS+=("${record}")
        break
      fi
    done
  done
else
  DATASET_RECORDS=("${ALL_DATASET_RECORDS[@]}")
fi

if [[ "${#DATASET_RECORDS[@]}" -eq 0 ]]; then
  echo "No scenes matched SCENES=${SCENES:-} DATASET_FAMILIES=${DATASET_FAMILIES:-}" >&2
  exit 2
fi

ITERATIONS="${ITERATIONS:-30000}"
RESOLUTION="${RESOLUTION:--1}"
VALIDATION_ROOT="${VALIDATION_ROOT:-outputs/validation}"
BENCHMARK_PROTOCOL="${BENCHMARK_PROTOCOL:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
FORCE="${FORCE:-0}"
AGGREGATE_AFTER_EVAL="${AGGREGATE_AFTER_EVAL:-1}"
AUTO_PATCH_READERS="${AUTO_PATCH_READERS:-1}"
CHECK_RUNTIME_DEPS="${CHECK_RUNTIME_DEPS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-5}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
CLEAN_FAILED_PAIR="${CLEAN_FAILED_PAIR:-0}"
STAGE_TIMEOUT_SECONDS="${STAGE_TIMEOUT_SECONDS:-0}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-11.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"

RUN_LOG_DIR="${VALIDATION_ROOT}/run_logs/${RUN_ID}"
PROGRESS_DIR="${VALIDATION_ROOT}/progress"
EVENTS_FILE="${PROGRESS_DIR}/full_validation_events.tsv"
STATUS_FILE="${PROGRESS_DIR}/full_validation_status.txt"
FAILURES_FILE="${PROGRESS_DIR}/validation_failures.tsv"
HARDWARE_LIMIT_COUNT=0
PROGRAM_FAILURE_COUNT=0
RUN_FINALIZED=0

mkdir -p "${RUN_LOG_DIR}" "${PROGRESS_DIR}"
ln -sfn "${RUN_LOG_DIR}" "${VALIDATION_ROOT}/run_logs/latest" 2>/dev/null || true

exec > >(tee -a "${RUN_LOG_DIR}/driver.log") 2>&1

config_for_method() {
  case "$1" in
    vanilla_3dgs) echo "configs/methods/vanilla_3dgs.yaml" ;;
    2dgs) echo "configs/methods/2dgs.yaml" ;;
    3dgs_mcmc) echo "configs/methods/3dgs_mcmc.yaml" ;;
    3dhgs) echo "configs/methods/3dhgs.yaml" ;;
    sss) echo "configs/methods/sss.yaml" ;;
    *) echo "configs/methods/catalog_method.yaml" ;;
  esac
}

official_images_for_label() {
  case "$1" in
    mip360/bicycle|mip360/garden|mip360/stump)
      echo "images_4"
      ;;
    mip360/room|mip360/counter|mip360/kitchen|mip360/bonsai)
      echo "images_2"
      ;;
    *)
      echo "images"
      ;;
  esac
}

pair_root_for() {
  local method="$1"
  local label="$2"
  echo "${VALIDATION_ROOT}/${method}/${label}"
}

done_marker_for() {
  local method="$1"
  local label="$2"
  local stage="$3"
  echo "$(pair_root_for "${method}" "${label}")/.${stage}.done"
}

total_stages() {
  echo $((${#METHODS_TO_RUN[@]} * ${#DATASET_RECORDS[@]} * 3))
}

completed_stages() {
  local count=0
  local record data label method stage marker
  for method in "${METHODS_TO_RUN[@]}"; do
    for record in "${DATASET_RECORDS[@]}"; do
      data="${record%%|*}"
      label="${record#*|}"
      for stage in train render eval; do
        marker="$(done_marker_for "${method}" "${label}" "${stage}")"
        if [[ -f "${marker}" ]]; then
          count=$((count + 1))
        fi
      done
    done
  done
  echo "${count}"
}

append_event() {
  local method="$1"
  local label="$2"
  local stage="$3"
  local status="$4"
  local log_path="$5"
  if [[ ! -f "${EVENTS_FILE}" ]]; then
    printf "time\tmethod\tdataset_scene\tstage\tstatus\tlog\n" > "${EVENTS_FILE}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S')" \
    "${method}" \
    "${label}" \
    "${stage}" \
    "${status}" \
    "${log_path}" >> "${EVENTS_FILE}"
}

classify_stage_failure() {
  local output="$1"
  local log_path="$2"
  local report="${output}/unified3dgs_training_report.json"
  local category=""

  if [[ -f "${report}" ]]; then
    category="$(python - "${report}" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    payload = {}
classification = payload.get("failure_classification", {})
if isinstance(classification, dict):
    print(classification.get("category", ""))
PY
)"
  fi
  if [[ "${category}" == "hardware_limit_confirmed" ]]; then
    echo "hardware_limit"
    return 0
  fi
  if [[ -f "${log_path}" ]] && grep -Eqi \
    'torch\.cuda\.OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory' \
    "${log_path}"; then
    echo "hardware_limit"
    return 0
  fi
  echo "program_error"
}

record_stage_failure() {
  local method="$1"
  local label="$2"
  local stage="$3"
  local classification="$4"
  local status="$5"
  local log_path="$6"
  local retained_log="${PROGRESS_DIR}/failure_logs/${method}/${label}/${stage}.log"
  mkdir -p "$(dirname "${retained_log}")"
  if [[ -f "${log_path}" ]]; then
    cp -f "${log_path}" "${retained_log}"
  fi
  if [[ ! -f "${FAILURES_FILE}" ]]; then
    printf "run_id\tmethod\tdataset_scene\tstage\tclassification\texit_code\tlog\n" > "${FAILURES_FILE}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${RUN_ID}" "${method}" "${label}" "${stage}" "${classification}" \
    "${status}" "${retained_log}" >> "${FAILURES_FILE}"
  if [[ "${classification}" == "hardware_limit" ]]; then
    HARDWARE_LIMIT_COUNT=$((HARDWARE_LIMIT_COUNT + 1))
  else
    PROGRAM_FAILURE_COUNT=$((PROGRAM_FAILURE_COUNT + 1))
  fi
}

clean_failed_pair_for_retry() {
  local pair_root="$1"
  if [[ "${CLEAN_FAILED_PAIR}" != "1" || ! -d "${pair_root}" ]]; then
    return 0
  fi
  if ! find "${pair_root}" -maxdepth 1 -type f -name '.*.failed' -print -quit | grep -q .; then
    return 0
  fi

  local resolved_root resolved_pair relative
  resolved_root="$(realpath -m "${VALIDATION_ROOT}")"
  resolved_pair="$(realpath -m "${pair_root}")"
  if [[ "${resolved_pair}" != "${resolved_root}/"* ]]; then
    echo "Refusing unsafe failed-pair cleanup outside validation root: ${resolved_pair}" >&2
    return 2
  fi
  relative="${resolved_pair#${resolved_root}/}"
  if [[ "${relative}" != */*/* ]]; then
    echo "Refusing unsafe failed-pair cleanup target: ${resolved_pair}" >&2
    return 2
  fi
  echo "Cleaning previously failed acceptance pair before retry: ${pair_root}"
  rm -rf -- "${resolved_pair}"
}

write_status() {
  local current="${1:-idle}"
  local done total
  done="$(completed_stages)"
  total="$(total_stages)"
  {
    echo "Unified 3DGS full validation status"
    echo "Updated at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Run id: ${RUN_ID}"
    echo "Validation root: ${VALIDATION_ROOT}"
    echo "Methods: ${METHODS_TO_RUN[*]}"
    echo "Dataset families: ${DATASET_FAMILIES:-all}"
    echo "Selected scenes: ${SCENES:-all}"
    echo "Scenes: ${#DATASET_RECORDS[@]}"
    echo "Stages done: ${done}/${total}"
    echo "Current: ${current}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
    echo "ITERATIONS=${ITERATIONS}"
    echo "RESOLUTION=${RESOLUTION}"
    echo "BENCHMARK_PROTOCOL=${BENCHMARK_PROTOCOL}"
    echo "FORCE=${FORCE}"
    echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
    echo "CLEAN_FAILED_PAIR=${CLEAN_FAILED_PAIR}"
    echo "STAGE_TIMEOUT_SECONDS=${STAGE_TIMEOUT_SECONDS}"
    echo "AUTO_PATCH_READERS=${AUTO_PATCH_READERS}"
    echo "CHECK_RUNTIME_DEPS=${CHECK_RUNTIME_DEPS}"
    echo "MIN_FREE_GB=${MIN_FREE_GB}"
    echo
    echo "Recent events:"
    tail -n 25 "${EVENTS_FILE}" 2>/dev/null || true
  } > "${STATUS_FILE}"
}

run_stage() {
  local method="$1"
  local label="$2"
  local stage="$3"
  local log_path="$4"
  local marker="$5"
  shift 5

  mkdir -p "$(dirname "${log_path}")"

  local available_kb required_kb
  available_kb="$(df -Pk "${VALIDATION_ROOT}" | awk 'NR==2 {print $4}')"
  required_kb=$((MIN_FREE_GB * 1024 * 1024))
  if [[ "${available_kb}" -lt "${required_kb}" ]]; then
    append_event "${method}" "${label}" "${stage}" "blocked_low_disk" "${log_path}"
    write_status "BLOCKED: less than ${MIN_FREE_GB} GB free before ${method}/${label}/${stage}"
    echo "Insufficient free disk space before starting stage."
    echo "Required minimum: ${MIN_FREE_GB} GB"
    df -h "${VALIDATION_ROOT}"
    exit 28
  fi

  if [[ "${FORCE}" != "1" && -f "${marker}" ]]; then
    rm -f "${marker%.done}.running"
    echo
    echo "SKIP: method=${method} scene=${label} stage=${stage}"
    echo "Reason: done marker exists at ${marker}"
    append_event "${method}" "${label}" "${stage}" "skipped_done" "${log_path}"
    write_status "skipped ${method}/${label}/${stage}"
    return 0
  fi

  echo
  echo "============================================================"
  echo "RUNNING: method=${method} scene=${label} stage=${stage}"
  echo "Log: ${log_path}"
  echo "============================================================"
  rm -f "${marker%.done}.running"
  date '+%Y-%m-%d %H:%M:%S' > "${marker%.done}.running"
  append_event "${method}" "${label}" "${stage}" "running" "${log_path}"
  write_status "running ${method}/${label}/${stage}"

  set +e
  local status
  if [[ "${STAGE_TIMEOUT_SECONDS}" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=30s "${STAGE_TIMEOUT_SECONDS}" "$@" 2>&1 | tee "${log_path}"
    status="${PIPESTATUS[0]}"
  else
    "$@" 2>&1 | tee "${log_path}"
    status="${PIPESTATUS[0]}"
  fi
  set -e

  if [[ "${status}" -eq 0 ]]; then
    rm -f "${marker%.done}.failed"
    date '+%Y-%m-%d %H:%M:%S' > "${marker}"
    append_event "${method}" "${label}" "${stage}" "success" "${log_path}"
    write_status "finished ${method}/${label}/${stage}"
  else
    date '+%Y-%m-%d %H:%M:%S' > "${marker%.done}.failed"
    local output classification
    output="$(pair_root_for "${method}" "${label}")/method_outputs"
    classification="$(classify_stage_failure "${output}" "${log_path}")"
    record_stage_failure "${method}" "${label}" "${stage}" "${classification}" "${status}" "${log_path}"
    append_event "${method}" "${label}" "${stage}" "failed_exit_${status}" "${log_path}"
    write_status "FAILED ${method}/${label}/${stage}; see ${log_path}"
    echo
    echo "FAILED: method=${method} scene=${label} stage=${stage} exit=${status}"
    echo "Classification: ${classification}"
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "Continuing with the next method/scene; completed stages remain resumable."
      return "${status}"
    fi
    echo "You can fix the issue and rerun this script; completed stages will be skipped."
    exit "${status}"
  fi
}

verify_stage_output() {
  local method="$1"
  local label="$2"
  local stage="$3"
  local output="$4"
  local marker="$5"
  local log_path="$6"

  local running_marker="${marker%.done}.running"
  local verification_command=(
    python scripts/verify_scene_outputs.py
    --stage "${stage}"
    --output "${output}"
    --iteration "${ITERATIONS}"
  )
  if [[ -f "${running_marker}" ]]; then
    verification_command+=(--newer-than "${running_marker}")
  fi

  if "${verification_command[@]}"; then
    rm -f "${running_marker}"
    return 0
  fi

  rm -f "${marker}"
  date '+%Y-%m-%d %H:%M:%S' > "${marker%.done}.failed"
  append_event "${method}" "${label}" "${stage}" "failed_output_verification" "${log_path}"
  write_status "FAILED output verification for ${method}/${label}/${stage}"
  echo
  echo "FAILED OUTPUT VERIFICATION: method=${method} scene=${label} stage=${stage}"
  echo "Removed invalid done marker: ${marker}"
  record_stage_failure "${method}" "${label}" "${stage}" "program_error" "3" "${log_path}"
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    return 3
  fi
  exit 3
}

recover_done_marker() {
  local method="$1"
  local label="$2"
  local stage="$3"
  local output="$4"
  local marker="$5"
  local log_path="$6"

  if [[ "${FORCE}" == "1" || -f "${marker}" ]]; then
    return 0
  fi
  if python scripts/verify_scene_outputs.py \
    --stage "${stage}" \
    --output "${output}" \
    --iteration "${ITERATIONS}" >/dev/null 2>&1; then
    rm -f "${marker%.done}.running"
    date '+%Y-%m-%d %H:%M:%S' > "${marker}"
    append_event "${method}" "${label}" "${stage}" "recovered_existing_output" "${log_path}"
    write_status "recovered existing ${method}/${label}/${stage}"
    echo "RECOVERED: valid existing output for ${method}/${label}/${stage}"
  fi
}

verify_effective_protocol_stage() {
  local method="$1"
  local label="$2"
  local stage="$3"
  local output="$4"
  local data="$5"
  local marker="$6"
  local log_path="$7"

  if [[ "${BENCHMARK_PROTOCOL}" != "1" ]]; then
    return 0
  fi

  if python scripts/verify_effective_protocol.py \
    --stage "${stage}" \
    --output "${output}" \
    --data "${data}" \
    --family "${label%%/*}" \
    --scene "${label#*/}" \
    --iteration "${ITERATIONS}"; then
    return 0
  fi

  rm -f "${marker}"
  date '+%Y-%m-%d %H:%M:%S' > "${marker%.done}.failed"
  append_event "${method}" "${label}" "${stage}" "failed_effective_protocol" "${log_path}"
  write_status "FAILED effective protocol for ${method}/${label}/${stage}"
  echo
  echo "FAILED EFFECTIVE PROTOCOL: method=${method} scene=${label} stage=${stage}"
  echo "Removed invalid done marker: ${marker}"
  record_stage_failure "${method}" "${label}" "${stage}" "program_error" "4" "${log_path}"
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    return 4
  fi
  exit 4
}

on_exit() {
  local status="$?"
  if [[ "${status}" -ne 0 && "${RUN_FINALIZED}" != "1" ]]; then
    write_status "script exited with status ${status}"
  fi
}
trap on_exit EXIT

echo "Unified 3DGS full validation"
echo "Run id: ${RUN_ID}"
echo "Methods: ${METHODS_TO_RUN[*]}"
echo "Dataset families: ${DATASET_FAMILIES:-all}"
echo "Selected scenes: ${SCENES:-all}"
echo "Scenes: ${#DATASET_RECORDS[@]}"
echo "Total stages: $(total_stages)"
echo "Validation root: ${VALIDATION_ROOT}"
echo "Driver log: ${RUN_LOG_DIR}/driver.log"
echo "Status file: ${STATUS_FILE}"

if [[ "${BENCHMARK_PROTOCOL}" == "1" ]]; then
  if [[ "${ITERATIONS}" != "30000" ]]; then
    echo "Official benchmark requires ITERATIONS=30000, got ${ITERATIONS}." >&2
    echo "Set BENCHMARK_PROTOCOL=0 only for non-benchmark diagnostic runs." >&2
    exit 2
  fi
  if [[ "${RESOLUTION}" != "-1" ]]; then
    echo "Official benchmark requires RESOLUTION=-1, got ${RESOLUTION}." >&2
    echo "Set BENCHMARK_PROTOCOL=0 only for non-benchmark diagnostic runs." >&2
    exit 2
  fi
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Refusing to start validation without CUDA_VISIBLE_DEVICES." >&2
  echo "Select a GPU explicitly on the shared server, for example:" >&2
  echo "  CUDA_VISIBLE_DEVICES=5 RUN_REAL=1 bash scripts/run_validation.sh" >&2
  exit 2
fi

if [[ "${CHECK_RUNTIME_DEPS}" == "1" ]]; then
  echo
  echo "Checking unified runtime dependencies before starting long jobs..."
  python scripts/check_runtime_dependencies.py
  python scripts/check_dataset_write_guard.py
else
  echo
  echo "CHECK_RUNTIME_DEPS=0, skipping runtime dependency check."
fi

if [[ "${AUTO_PATCH_READERS}" == "1" ]]; then
  echo
  echo "Applying robust third-party dataset reader patches for selected methods..."
  for method in "${METHODS_TO_RUN[@]}"; do
    python scripts/patch_third_party_readers.py --method "${method}"
    if [[ "${method}" == "3dhgs" ]]; then
      python scripts/patch_3dhgs_render_gt.py
    fi
  done
else
  echo
  echo "AUTO_PATCH_READERS=0, skipping third-party dataset reader patches."
fi

if [[ " ${METHODS_TO_RUN[*]} " == *" 3dgs_mcmc "* ]]; then
  echo
  echo "Preparing unified 3DGS-MCMC scene configs..."
  python scripts/prepare_mcmc_scene_configs.py \
    --validation-root "${VALIDATION_ROOT}" \
    --iterations "${ITERATIONS}" \
    --resolution "${RESOLUTION}"
fi

if [[ " ${METHODS_TO_RUN[*]} " == *" sss "* ]]; then
  echo
  echo "Preparing unified ${ITERATIONS}-iteration SSS scene configs..."
  python scripts/prepare_sss_scene_configs.py \
    --validation-root "${VALIDATION_ROOT}" \
    --iterations "${ITERATIONS}" \
    --resolution "${RESOLUTION}"
fi

echo
echo "Checking selected methods and scene configs before validation..."
READINESS_METHODS=()
for method in "${METHODS_TO_RUN[@]}"; do
  case "${method}" in
    vanilla_3dgs|2dgs|3dgs_mcmc|3dhgs|sss)
      READINESS_METHODS+=("${method}")
      ;;
  esac
done
if [[ "${#READINESS_METHODS[@]}" -gt 0 ]]; then
  python scripts/check_method_scene_readiness.py \
    --methods "${READINESS_METHODS[@]}" \
    --validation-root "${VALIDATION_ROOT}" \
    --iterations "${ITERATIONS}" \
    --resolution "${RESOLUTION}"
else
  echo "Legacy method/scene readiness check is not applicable to the selected catalog method(s)."
  echo "Catalog methods run their exhaustive preflight inside the train stage."
fi

for record in "${DATASET_RECORDS[@]}"; do
  data="${record%%|*}"
  label="${record#*|}"
  if [[ ! -d "${data}" ]]; then
    echo "Missing dataset directory: ${data}" >&2
    exit 2
  fi
done

write_status "starting"

for method in "${METHODS_TO_RUN[@]}"; do
  config="$(config_for_method "${method}")"
  for record in "${DATASET_RECORDS[@]}"; do
    data="${record%%|*}"
    label="${record#*|}"
    pair_root="$(pair_root_for "${method}" "${label}")"
    output="${pair_root}/method_outputs"
    images="$(official_images_for_label "${label}")"
    clean_failed_pair_for_retry "${pair_root}"
    mkdir -p "${pair_root}" "${output}"

    train_command=(
      python train_all.py
      --method "${method}"
      --config "${config}"
      --data "${data}"
      --output "${output}"
      --set "images=${images}"
      --set "dataset_label=${label}"
      --set "iterations=${ITERATIONS}"
      --set "resolution=${RESOLUTION}"
      --set "test_iterations=-1"
    )
    if [[ "${method}" == "3dgs_mcmc" || "${method}" == "sss" ]]; then
      train_command+=(
        --set "generated_scene_config_dir=${VALIDATION_ROOT}/generated_configs/${method}"
      )
    fi

    recover_done_marker "${method}" "${label}" "train" "${output}" \
      "$(done_marker_for "${method}" "${label}" "train")" "${pair_root}/train.log"

    if ! run_stage "${method}" "${label}" "train" "${pair_root}/train.log" "$(done_marker_for "${method}" "${label}" "train")" \
      "${train_command[@]}"; then
      continue
    fi

    if ! verify_stage_output "${method}" "${label}" "train" "${output}" \
      "$(done_marker_for "${method}" "${label}" "train")" "${pair_root}/train.log"; then
      continue
    fi

    if ! verify_effective_protocol_stage "${method}" "${label}" "train" "${output}" "${data}" \
      "$(done_marker_for "${method}" "${label}" "train")" "${pair_root}/train.log"; then
      continue
    fi

    actual_iteration="${ITERATIONS}"
    echo "Resolved final iteration for ${method}/${label}: ${actual_iteration}"

    recover_done_marker "${method}" "${label}" "render" "${output}" \
      "$(done_marker_for "${method}" "${label}" "render")" "${pair_root}/render.log"

    if ! run_stage "${method}" "${label}" "render" "${pair_root}/render.log" "$(done_marker_for "${method}" "${label}" "render")" \
      python render_all.py \
        --method "${method}" \
        --config "${config}" \
        --data "${data}" \
        --output "${output}" \
        --set "images=${images}" \
        --set "resolution=${RESOLUTION}" \
        --set "render_iteration=${actual_iteration}"; then
      continue
    fi

    if ! verify_stage_output "${method}" "${label}" "render" "${output}" \
      "$(done_marker_for "${method}" "${label}" "render")" "${pair_root}/render.log"; then
      continue
    fi

    if ! verify_effective_protocol_stage "${method}" "${label}" "render" "${output}" "${data}" \
      "$(done_marker_for "${method}" "${label}" "render")" "${pair_root}/render.log"; then
      continue
    fi

    recover_done_marker "${method}" "${label}" "eval" "${output}" \
      "$(done_marker_for "${method}" "${label}" "eval")" "${pair_root}/eval.log"

    if ! run_stage "${method}" "${label}" "eval" "${pair_root}/eval.log" "$(done_marker_for "${method}" "${label}" "eval")" \
      python eval_all.py \
        --method "${method}" \
        --config "${config}" \
        --data "${data}" \
        --output "${output}" \
        --set "render_iteration=${actual_iteration}"; then
      continue
    fi

    if ! verify_stage_output "${method}" "${label}" "eval" "${output}" \
      "$(done_marker_for "${method}" "${label}" "eval")" "${pair_root}/eval.log"; then
      continue
    fi

    if [[ "${AGGREGATE_AFTER_EVAL}" == "1" ]]; then
      python scripts/aggregate_metrics.py \
        --validation-root "${VALIDATION_ROOT}" \
        --iteration "${ITERATIONS}"
      write_status "aggregated metrics after ${method}/${label}"
    fi
  done
done

python scripts/aggregate_metrics.py \
  --validation-root "${VALIDATION_ROOT}" \
  --iteration "${ITERATIONS}"
if [[ "${BENCHMARK_PROTOCOL}" == "1" ]]; then
  audit_command=(
    python scripts/audit_benchmark_protocol.py
    --validation-root "${VALIDATION_ROOT}"
    --output "${VALIDATION_ROOT}/benchmark_protocol_audit.csv"
    --methods "${METHODS_TO_RUN[@]}"
  )
  if [[ -n "${DATASET_FAMILIES:-}" ]]; then
    audit_command+=(--dataset-families "${DATASET_FAMILIES_TO_RUN[@]}")
  fi
  if [[ -n "${SCENES:-}" ]]; then
    audit_command+=(--scenes "${SCENES_TO_RUN[@]}")
  fi
  "${audit_command[@]}"
fi
write_status "complete: hardware_limits=${HARDWARE_LIMIT_COUNT} program_failures=${PROGRAM_FAILURE_COUNT}"

echo
echo "Full validation complete."
echo "Status: ${STATUS_FILE}"
echo "Driver log: ${RUN_LOG_DIR}/driver.log"
echo "Failure log: ${FAILURES_FILE}"
echo "Hardware-limited stages: ${HARDWARE_LIMIT_COUNT}"
echo "Program/environment failures: ${PROGRAM_FAILURE_COUNT}"
RUN_FINALIZED=1
if [[ "${PROGRAM_FAILURE_COUNT}" -gt 0 ]]; then
  exit 2
fi
