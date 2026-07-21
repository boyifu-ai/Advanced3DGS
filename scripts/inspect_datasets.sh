#!/usr/bin/env bash
set -u

if [[ -z "${DATASET_1:-}" || -z "${DATASET_2:-}" || -z "${DATASET_3:-}" ]]; then
  echo "Set DATASET_1, DATASET_2, and DATASET_3 before running this script."
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/outputs/dataset_inspect_$(date +%Y%m%d_%H%M%S).log"
DATASETS=("${DATASET_1}" "${DATASET_2}" "${DATASET_3}")

inspect_one() {
  local dataset="$1"
  local image_count=0
  local has_images=0
  local has_sparse=0
  local has_points=0

  echo
  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "============================================================"

  if [[ ! -d "${dataset}" ]]; then
    echo "MISSING_DIRECTORY"
    return 0
  fi

  echo "[basic]"
  ls -lah "${dataset}" 2>/dev/null || true
  du -sh "${dataset}" 2>/dev/null || true

  echo
  echo "[top-level directories]"
  find "${dataset}" -maxdepth 2 -type d 2>/dev/null | sort | head -120 || true

  echo
  echo "[image counts]"
  for image_dir in \
    "${dataset}/images" \
    "${dataset}/input" \
    "${dataset}/image" \
    "${dataset}/rgb"; do
    if [[ -d "${image_dir}" ]]; then
      count="$(find "${image_dir}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l)"
      echo "${image_dir}: ${count}"
      if [[ "${count}" -gt "${image_count}" ]]; then
        image_count="${count}"
      fi
      if [[ "${count}" -gt 0 ]]; then
        has_images=1
      fi
    fi
  done

  echo
  echo "[colmap files]"
  for sparse_dir in \
    "${dataset}/sparse/0" \
    "${dataset}/sparse" \
    "${dataset}/colmap/sparse/0" \
    "${dataset}/COLMAP/sparse/0"; do
    if [[ -d "${sparse_dir}" ]]; then
      echo "Sparse candidate: ${sparse_dir}"
      find "${sparse_dir}" -maxdepth 1 -type f 2>/dev/null | sort
      if find "${sparse_dir}" -maxdepth 1 -type f \
        \( -name 'cameras.bin' -o -name 'cameras.txt' \) 2>/dev/null | grep -q . \
        && find "${sparse_dir}" -maxdepth 1 -type f \
        \( -name 'images.bin' -o -name 'images.txt' \) 2>/dev/null | grep -q .; then
        has_sparse=1
      fi
      if find "${sparse_dir}" -maxdepth 1 -type f \
        \( -name 'points3D.bin' -o -name 'points3D.txt' \) 2>/dev/null | grep -q .; then
        has_points=1
      fi
    fi
  done

  if find "${dataset}" -maxdepth 3 -type f \
    \( -name 'points3D.ply' -o -name 'point_cloud.ply' -o -name 'input.ply' \) \
    2>/dev/null | grep -q .; then
    has_points=1
  fi

  echo
  echo "[other common metadata]"
  find "${dataset}" -maxdepth 2 -type f \
    \( -name 'poses_bounds.npy' -o -name 'transforms_*.json' -o -name 'transforms.json' -o -name 'cameras.json' -o -name 'intrinsics.json' -o -name 'dataset_train.txt' -o -name 'dataset_test.txt' \) \
    2>/dev/null | sort || true

  echo
  echo "[format hints]"
  echo "image_count=${image_count}"
  echo "has_images=${has_images}"
  echo "has_sparse_cameras=${has_sparse}"
  echo "has_point_cloud=${has_points}"
  if [[ "${has_images}" -eq 1 && "${has_sparse}" -eq 1 && "${has_points}" -eq 1 ]]; then
    echo "READY_FOR_COLMAP_STYLE_3DGS"
  elif [[ "${has_images}" -eq 1 && "${has_sparse}" -eq 1 && "${has_points}" -eq 0 ]]; then
    echo "NOT_READY_MISSING_POINT_CLOUD"
  elif [[ -f "${dataset}/poses_bounds.npy" ]]; then
    echo "LIKELY_LLFF_MIPNERF360_RAW_OR_CONVERTED_REQUIRED"
  elif find "${dataset}" -maxdepth 2 -type f -name 'transforms_*.json' 2>/dev/null | grep -q .; then
    echo "LIKELY_NERF_SYNTHETIC_FORMAT"
  else
    echo "FORMAT_UNKNOWN_OR_NEEDS_METHOD_SPECIFIC_CHECK"
  fi
}

mkdir -p "${PROJECT_ROOT}/outputs"

{
echo "Unified 3DGS dataset inspection"
echo "Generated at: $(date)"
echo "This script only reads dataset directories."
for dataset in "${DATASETS[@]}"; do
  inspect_one "${dataset}"
done
} 2>&1 | tee "${LOG_FILE}"

echo
echo "Saved dataset inspection log to: ${LOG_FILE}"
