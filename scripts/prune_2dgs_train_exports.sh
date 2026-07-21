#!/usr/bin/env bash
set -euo pipefail

VALIDATION_ROOT="${VALIDATION_ROOT:-outputs/validation}"
APPLY="${APPLY:-0}"
ROOT="${VALIDATION_ROOT}/2dgs"

if [[ ! -d "${ROOT}" ]]; then
  echo "2DGS validation directory not found: ${ROOT}"
  exit 0
fi

VALIDATION_ROOT_ABS="$(cd "${VALIDATION_ROOT}" && pwd -P)"
ROOT_ABS="$(cd "${ROOT}" && pwd -P)"
if [[ "${ROOT_ABS}" != "${VALIDATION_ROOT_ABS}/2dgs" ]]; then
  echo "Refusing to inspect an unexpected path: ${ROOT_ABS}" >&2
  exit 2
fi

mapfile -d '' TARGETS < <(
  find "${ROOT_ABS}" -type d \
    \( -path '*/method_outputs/train/ours_*/renders' \
       -o -path '*/method_outputs/train/ours_*/vis' \) \
    -print0
)

if [[ "${#TARGETS[@]}" -eq 0 ]]; then
  echo "No 2DGS training-view render exports found under ${ROOT_ABS}."
  exit 0
fi

echo "2DGS training-view exports:"
du -sh "${TARGETS[@]}" 2>/dev/null || true
echo
echo "These directories contain redundant training-view image/depth exports."
echo "This script does not touch point_cloud, test renders, metrics, or fuse*.ply."

if [[ "${APPLY}" != "1" ]]; then
  echo
  echo "Preview only. Re-run with APPLY=1 to delete the listed exports."
  exit 0
fi

for target in "${TARGETS[@]}"; do
  case "${target}" in
    "${ROOT_ABS}"/*/method_outputs/train/ours_*/renders|\
    "${ROOT_ABS}"/*/method_outputs/train/ours_*/vis)
      echo "Deleting files under: ${target}"
      find "${target}" -type f -delete
      find "${target}" -depth -type d -empty -delete
      ;;
    *)
      echo "Refusing unexpected target: ${target}" >&2
      exit 3
      ;;
  esac
done

echo
df -h "${ROOT_ABS}"
