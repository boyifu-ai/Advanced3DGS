#!/usr/bin/env bash
set -u

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${ADVANCED3DGS_DATA_ROOT:-datasets}}"
LOG_FILE="${PROJECT_ROOT}/server_env_report_$(date +%Y%m%d_%H%M%S).log"

{
echo "=============================="
echo "Unified 3DGS Server Environment Report"
echo "Generated at: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Work root: ${WORK_ROOT}"
echo "Dataset root: ${DATA_ROOT}"
echo "=============================="

echo
echo "1. User / Path Information"
whoami 2>/dev/null || true
pwd
id 2>/dev/null || true

echo
echo "2. Work Directory Check"
ls -lah "${WORK_ROOT}" 2>/dev/null || true
df -h "${WORK_ROOT}" 2>/dev/null || true

echo
echo "3. Shared Dataset Directory Check"
ls -lah "${DATA_ROOT}" 2>/dev/null || true
df -h "${DATA_ROOT}" 2>/dev/null || true
du -sh "${DATA_ROOT}"/* 2>/dev/null | head -100 || true
find "${DATA_ROOT}" -maxdepth 2 -type d 2>/dev/null | head -200 || true

echo
echo "4. OS / System Information"
uname -a 2>/dev/null || true
cat /etc/os-release 2>/dev/null || true
ldd --version 2>/dev/null || true

echo
echo "5. CPU / Memory / Disk"
nproc 2>/dev/null || true
lscpu 2>/dev/null || true
free -h 2>/dev/null || true
df -h 2>/dev/null || true
ulimit -a 2>/dev/null || true

echo
echo "6. GPU / NVIDIA Driver"
which nvidia-smi 2>/dev/null || true
nvidia-smi 2>/dev/null || true
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,compute_cap --format=csv 2>/dev/null || true

echo
echo "7. CUDA Toolkit / CUDA Paths"
which nvcc 2>/dev/null || true
nvcc --version 2>/dev/null || true
ls -lah /usr/local/ 2>/dev/null | grep cuda || true
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
echo "PATH=${PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

echo
echo "8. Python / Conda / PyTorch"
which python 2>/dev/null || true
python --version 2>/dev/null || true
which conda 2>/dev/null || true
conda --version 2>/dev/null || true
conda info 2>/dev/null || true
python - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
try:
    import torch
    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
except Exception as exc:
    print("torch check error:", repr(exc))
PY

echo
echo "9. Compiler Toolchain"
which gcc 2>/dev/null || true
gcc --version 2>/dev/null || true
which g++ 2>/dev/null || true
g++ --version 2>/dev/null || true
which cmake 2>/dev/null || true
cmake --version 2>/dev/null || true
which ninja 2>/dev/null || true
ninja --version 2>/dev/null || true

echo
echo "10. GitHub Connectivity Without Cloning"
which git 2>/dev/null || true
git --version 2>/dev/null || true
timeout 15 git ls-remote https://github.com/graphdeco-inria/gaussian-splatting.git HEAD 2>&1 || true
timeout 15 git ls-remote https://github.com/hbb1/2d-gaussian-splatting.git HEAD 2>&1 || true

echo
echo "Report complete: ${LOG_FILE}"
} 2>&1 | tee "${LOG_FILE}"
