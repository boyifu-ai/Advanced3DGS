#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/hac_plus}"
OFFICIAL_CUDA_HOME="${UNIFIED3DGS_HAC_PLUS_CUDA_HOME:-/usr/local/cuda-11.6}"
BUILD_TARGET="${PROJECT_ROOT}/third_party_build/hac_plus/site-packages"
BUILD_MANIFEST="${BUILD_TARGET}/.unified3dgs_extension_build.json"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

cd "$PROJECT_ROOT"

if [[ ! -d third_party/HAC-plus ]]; then
  echo "Missing third_party/HAC-plus"
  exit 2
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Missing conda executable. Install Conda or set CONDA_EXE to its absolute path."
  exit 2
fi
if [[ ! -x "${OFFICIAL_CUDA_HOME}/bin/nvcc" ]]; then
  echo "Missing official CUDA compiler: ${OFFICIAL_CUDA_HOME}/bin/nvcc"
  exit 2
fi
if ! "${OFFICIAL_CUDA_HOME}/bin/nvcc" --version | grep -q "release 11.6"; then
  echo "HAC+ requires the CUDA 11.6 compiler."
  "${OFFICIAL_CUDA_HOME}/bin/nvcc" --version || true
  exit 2
fi

CUDA_HOME="$OFFICIAL_CUDA_HOME"
export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"

probe_torch() {
  "${ENV_PREFIX}/bin/python" - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert sys.version_info[:3] == (3, 7, 13)
assert torch.__version__.startswith("1.12.1")
assert torch.version.cuda == "11.6"
assert torch.cuda.is_available()
PY
}

echo "Creating HAC+ official backend at: $ENV_PREFIX"
CONDA_ACTION=create
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  CONDA_ACTION=install
  echo "Existing backend found; reconciling it to the official pins."
fi
"$CONDA_BIN" "$CONDA_ACTION" -y -p "$ENV_PREFIX" \
  --override-channels \
  -c pytorch -c pyg -c conda-forge \
  python=3.7.13 pip=22.3.1 \
  pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 \
  cudatoolkit=11.6 pytorch-scatter plyfile=0.8.1 tqdm \
  numpy=1.21.6 "setuptools<68" wheel ninja \
  "mkl<2024.1" "intel-openmp<2024.1"

echo
echo "Verifying the official PyTorch runtime before pip or extension builds..."
if ! probe_torch; then
  echo
  echo "HAC+ backend Torch import failed."
  echo "Relevant Intel runtime packages:"
  "$CONDA_BIN" list -p "$ENV_PREFIX" \
    | grep -E '^(mkl|intel-openmp|pytorch|cudatoolkit)[[:space:]]' || true
  echo
  echo "PyTorch 1.12.1 is incompatible with MKL 2024.1 and newer."
  echo "The setup script requires mkl<2024.1 and intel-openmp<2024.1."
  exit 3
fi

"${ENV_PREFIX}/bin/python" -m pip install \
  einops==0.6.1 wandb==0.15.12 lpips==0.1.4

echo
echo "Rechecking PyTorch/CUDA after Python dependency installation..."
probe_torch

if [[ -d "$BUILD_TARGET" ]] \
  && [[ -n "$(find "$BUILD_TARGET" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  BACKUP="${PROJECT_ROOT}/third_party_build/hac_plus/site-packages_before_official_$(date +%Y%m%d_%H%M%S)"
  echo "Moving the previous incompatible HAC+ extension build to: $BACKUP"
  mv "$BUILD_TARGET" "$BACKUP"
fi

"${ENV_PREFIX}/bin/python" scripts/install_catalog_method_extensions.py \
  --method hac_plus \
  --run-real \
  --timeout-seconds 1800 \
  --min-free-disk-gb 5

echo
echo "Validating the extension build manifest..."
BUILD_MANIFEST="$BUILD_MANIFEST" \
EXPECTED_CUDA_HOME="$OFFICIAL_CUDA_HOME" \
  "${ENV_PREFIX}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["BUILD_MANIFEST"])
payload = json.loads(manifest.read_text(encoding="utf-8"))
signature = payload["signature"]
expected = os.environ["EXPECTED_CUDA_HOME"]
assert signature["cuda_home"] == expected, (
    f"extension CUDA_HOME mismatch: expected {expected}, "
    f"got {signature['cuda_home']}"
)
assert signature["torch_version"].startswith("1.12.1")
assert signature["torch_cuda"] == "11.6"
assert signature["python_version"].startswith("3.7.13")
print("extension manifest:", manifest)
print("extension CUDA_HOME:", signature["cuda_home"])
print("extension torch:", signature["torch_version"])
print("extension torch CUDA:", signature["torch_cuda"])
PY

echo
echo "Validating all HAC+ native extension imports..."
PYTHONPATH="${BUILD_TARGET}:${PROJECT_ROOT}/third_party/HAC-plus:${PYTHONPATH:-}" \
  "${ENV_PREFIX}/bin/python" - <<'PY'
import torch
import arithmetic
import _gridencoder
import diff_gaussian_rasterization
import diff_gaussian_rasterization._C as diff_gaussian_rasterization_C
import simple_knn
import simple_knn._C as simple_knn_C

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("arithmetic:", arithmetic.__file__)
print("_gridencoder:", _gridencoder.__file__)
print("diff_gaussian_rasterization:", diff_gaussian_rasterization.__file__)
print("diff_gaussian_rasterization._C:", diff_gaussian_rasterization_C.__file__)
print("simple_knn._C:", simple_knn_C.__file__)
PY

echo
echo "HAC+ official backend is ready."
echo "Export this before running HAC+:"
echo "export UNIFIED3DGS_HAC_PLUS_PYTHON=${ENV_PREFIX}/bin/python"
