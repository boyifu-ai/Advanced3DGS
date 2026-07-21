#!/usr/bin/env bash
set -u

ROOT="${1:-${ADVANCED3DGS_DATA_ROOT:-datasets}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/outputs/colmap_dataset_candidates_$(date +%Y%m%d_%H%M%S).log"

has_image_dir() {
  local dataset="$1"
  for image_dir in "${dataset}/images" "${dataset}/input" "${dataset}/image" "${dataset}/rgb"; do
    if [[ -d "${image_dir}" ]] && find "${image_dir}" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | grep -q .; then
      return 0
    fi
  done
  return 1
}

has_sparse_cameras() {
  local dataset="$1"
  for sparse_dir in "${dataset}/sparse/0" "${dataset}/sparse" "${dataset}/colmap/sparse/0" "${dataset}/COLMAP/sparse/0"; do
    if [[ -d "${sparse_dir}" ]] \
      && find "${sparse_dir}" -maxdepth 1 -type f \( -name 'cameras.bin' -o -name 'cameras.txt' \) 2>/dev/null | grep -q . \
      && find "${sparse_dir}" -maxdepth 1 -type f \( -name 'images.bin' -o -name 'images.txt' \) 2>/dev/null | grep -q .; then
      return 0
    fi
  done
  return 1
}

has_point_cloud() {
  local dataset="$1"
  for sparse_dir in "${dataset}/sparse/0" "${dataset}/sparse" "${dataset}/colmap/sparse/0" "${dataset}/COLMAP/sparse/0"; do
    if [[ -d "${sparse_dir}" ]] \
      && find "${sparse_dir}" -maxdepth 1 -type f \( -name 'points3D.bin' -o -name 'points3D.txt' \) 2>/dev/null | grep -q .; then
      return 0
    fi
  done
  if find "${dataset}" -maxdepth 3 -type f \
    \( -name 'points3D.ply' -o -name 'point_cloud.ply' -o -name 'input.ply' \) \
    2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

inspect_candidate() {
  local dataset="$1"
  if has_image_dir "${dataset}" && has_sparse_cameras "${dataset}" && has_point_cloud "${dataset}"; then
    echo "READY ${dataset}"
  elif has_image_dir "${dataset}" && has_sparse_cameras "${dataset}"; then
    echo "MISSING_POINTS ${dataset}"
  fi
}

mkdir -p "${PROJECT_ROOT}/outputs"

{
echo "COLMAP-style dataset candidate scan"
echo "Generated at: $(date)"
echo "Root: ${ROOT}"
echo "This script only reads dataset directories."
echo

while IFS= read -r image_dir; do
  dataset="$(dirname "${image_dir}")"
  inspect_candidate "${dataset}"
done < <(find "${ROOT}" -maxdepth 4 -type d \( -name images -o -name input -o -name image -o -name rgb \) 2>/dev/null | sort)
} 2>&1 | tee "${LOG_FILE}"

echo
echo "Saved candidate scan to: ${LOG_FILE}"
