#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_PREFIX="${ENV_PREFIX:-${PROJECT_ROOT}/envs/3dcs}"
REPO="${PROJECT_ROOT}/third_party/convex-splatting"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"

cd "$PROJECT_ROOT"

if [[ ! -d "$REPO" ]]; then
  echo "Missing repository: $REPO"
  echo "Clone it first:"
  echo '  git clone --recursive https://github.com/convexsplatting/convex-splatting.git third_party/convex-splatting'
  exit 2
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "Missing conda executable. Install Conda or set CONDA_EXE to its absolute path."
  exit 2
fi

echo "Creating 3DCS / Convex Splatting official backend at: $ENV_PREFIX"
CONDA_ACTION=create
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  CONDA_ACTION=install
  echo "Existing backend found; reconciling it to the official pins."
fi

"$CONDA_BIN" "$CONDA_ACTION" -y -p "$ENV_PREFIX" \
  --override-channels \
  -c nvidia -c conda-forge \
  python=3.11 pip ninja cmake

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CUDA_HOME="$ENV_PREFIX"
export CUDA_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-8}"

echo
echo "Installing official Python dependencies..."
"${ENV_PREFIX}/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo
echo "Installing PyTorch through conda to avoid duplicating pip nvidia-* CUDA wheels..."
"$CONDA_BIN" install -y -p "$ENV_PREFIX" \
  --override-channels \
  -c pytorch -c nvidia -c conda-forge \
  pytorch=2.4.0 torchvision=0.19.0 pytorch-cuda=12.1 \
  cuda-nvcc=12.1 cuda-cudart-dev=12.1 \
  cuda-libraries=12.1 cuda-libraries-dev=12.1 cuda-cccl=12.1 nccl

if ! "${ENV_PREFIX}/bin/python" - <<'PY'
import sys
import torch

print("conda torch:", torch.__version__, "cuda:", torch.version.cuda)
raise SystemExit(0 if torch.__version__.startswith("2.4.0") and torch.version.cuda == "12.1" else 1)
PY
then
  echo
  echo "Conda resolved a CPU-only torch build; replacing it with the official cu121 wheels without pip CUDA dependencies..."
  "${ENV_PREFIX}/bin/python" -m pip install --no-cache-dir --no-deps --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0
fi

"${ENV_PREFIX}/bin/python" -m pip install --no-cache-dir \
  tqdm plyfile open3d lpips mediapy opencv-python

echo
echo "Verifying PyTorch runtime..."
"${ENV_PREFIX}/bin/python" - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert sys.version_info[:2] == (3, 11)
assert torch.__version__.startswith("2.4.0")
assert torch.version.cuda == "12.1"
assert torch.cuda.is_available()
PY

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "Missing CUDA compiler in official env: ${CUDA_HOME}/bin/nvcc"
  echo "The official backend must provide nvcc for diff-convex-rasterization."
  exit 3
fi

echo
echo "CUDA compiler:"
"${CUDA_HOME}/bin/nvcc" --version

echo
echo "Compiling Convex Splatting native extensions..."
cd "$REPO"
bash compile.sh

# These extension setup files import torch at build time. Build isolation hides the
# already-installed official torch runtime and can produce a false ModuleNotFoundError.
"${ENV_PREFIX}/bin/python" -m pip install --no-build-isolation ./submodules/diff-convex-rasterization
"${ENV_PREFIX}/bin/python" -m pip install --no-build-isolation ./submodules/simple-knn

echo
echo "Validating native extension imports..."
PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${ENV_PREFIX}/bin/python" - <<'PY'
import torch
import diff_convex_rasterization
import simple_knn
import simple_knn._C as simple_knn_C

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("diff_convex_rasterization:", diff_convex_rasterization.__file__)
print("simple_knn:", simple_knn.__file__)
print("simple_knn._C:", simple_knn_C.__file__)
PY

echo
echo "3DCS / Convex Splatting official backend is ready."
echo "Export this before running 3DCS:"
echo "export UNIFIED3DGS_3DCS_PYTHON=${ENV_PREFIX}/bin/python"
