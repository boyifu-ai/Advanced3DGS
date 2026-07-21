#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/install_method_extensions.sh <vanilla_3dgs|2dgs|3dgs_mcmc|3dhgs|sss>"
  exit 2
fi

METHOD="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${PROJECT_ROOT}/third_party_build/${METHOD}"
TARGET="${BUILD_ROOT}/site-packages"
SRC_ROOT="${BUILD_ROOT}/src_$(date +%Y%m%d_%H%M%S)"

case "${METHOD}" in
  vanilla_3dgs)
    REPO="${PROJECT_ROOT}/third_party/gaussian-splatting"
    EXTENSIONS=(
      "submodules/diff-gaussian-rasterization"
      "submodules/simple-knn"
      "submodules/fused-ssim"
    )
    IMPORT_TEST='import diff_gaussian_rasterization; import simple_knn._C'
    ;;
  2dgs)
    REPO="${PROJECT_ROOT}/third_party/2d-gaussian-splatting"
    EXTENSIONS=(
      "submodules/diff-surfel-rasterization"
      "submodules/simple-knn"
    )
    IMPORT_TEST='import diff_surfel_rasterization; import simple_knn._C'
    ;;
  3dgs_mcmc)
    REPO="${PROJECT_ROOT}/third_party/3dgs-mcmc"
    EXTENSIONS=(
      "submodules/diff-gaussian-rasterization"
      "submodules/simple-knn"
    )
    IMPORT_TEST='from diff_gaussian_rasterization import compute_relocation; import simple_knn._C'
    ;;
  3dhgs)
    REPO="${PROJECT_ROOT}/third_party/3DHGS"
    EXTENSIONS=(
      "submodules/diff-gaussian-rasterization"
      "submodules/simple-knn"
    )
    IMPORT_TEST='import diff_gaussian_rasterization; import simple_knn._C'
    ;;
  sss)
    REPO="${PROJECT_ROOT}/third_party/3D-student-splatting-and-scooping"
    EXTENSIONS=(
      "submodules/diff-t-rasterization"
      "submodules/simple-knn"
    )
    IMPORT_TEST='import diff_t_rasterization; import simple_knn._C'
    ;;
  *)
    echo "Unknown method: ${METHOD}"
    exit 2
    ;;
esac

if [[ ! -d "${REPO}" ]]; then
  echo "Missing repository: ${REPO}"
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"

mkdir -p "${TARGET}" "${SRC_ROOT}" "${PROJECT_ROOT}/outputs"

echo "Method: ${METHOD}"
echo "Repository: ${REPO}"
echo "Target site-packages: ${TARGET}"
echo "Source copy root: ${SRC_ROOT}"
echo "Python: $(which python)"
python --version
echo "CUDA_HOME=${CUDA_HOME}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS=${MAX_JOBS}"

python - <<'PY'
import sys
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is not available.")
PY

for extension in "${EXTENSIONS[@]}"; do
  source_path="${REPO}/${extension}"
  if [[ ! -d "${source_path}" ]]; then
    echo "Missing extension source: ${source_path}"
    exit 1
  fi

  copied_path="${SRC_ROOT}/$(basename "${extension}")"
  echo
  echo "Installing extension: ${extension}"
  echo "Copied source: ${copied_path}"
  cp -a "${source_path}" "${copied_path}"
  find "${copied_path}" -maxdepth 1 \
    \( -name 'build' -o -name 'dist' -o -name '*.egg-info' -o -name '__pycache__' \) \
    -print -exec rm -rf {} +
  python -m pip install \
    --no-deps \
    --no-build-isolation \
    --upgrade \
    --target "${TARGET}" \
    "${copied_path}"
done

echo
echo "Validating imports from method-specific target..."
PYTHONPATH="${TARGET}:${REPO}:${PYTHONPATH:-}" python - <<PY
${IMPORT_TEST}
print("Import validation succeeded for ${METHOD}.")
PY

echo
echo "Installed method extensions for ${METHOD} into ${TARGET}"
