#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
LOG_FILE="${PROJECT_ROOT}/outputs/third_party_inspect_$(date +%Y%m%d_%H%M%S).log"

repos=(
  "gaussian-splatting"
  "2d-gaussian-splatting"
  "3dgs-mcmc"
  "3DHGS"
  "3D-student-splatting-and-scooping"
)

print_file_if_exists() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    echo
    echo "----- ${path} -----"
    sed -n '1,220p' "${path}" 2>/dev/null || true
  fi
}

inspect_repo() {
  local repo_name="$1"
  local repo_path="${THIRD_PARTY_ROOT}/${repo_name}"

  echo
  echo "============================================================"
  echo "Repository: ${repo_name}"
  echo "Path: ${repo_path}"
  echo "============================================================"

  if [[ ! -d "${repo_path}" ]]; then
    echo "Missing repository directory."
    return 0
  fi

  (
    cd "${repo_path}" || exit 0

    echo
    echo "[git]"
    git rev-parse --show-toplevel 2>/dev/null || true
    git remote -v 2>/dev/null || true
    git rev-parse HEAD 2>/dev/null || true
    git branch --show-current 2>/dev/null || true
    git status --short 2>/dev/null || true

    echo
    echo "[submodules]"
    git submodule status --recursive 2>/dev/null || true

    echo
    echo "[top-level files]"
    find . -maxdepth 2 -type f \
      \( -iname 'README*' -o -iname 'LICENSE*' -o -iname 'environment*.yml' -o -iname 'requirements*.txt' -o -iname 'setup.py' -o -iname 'pyproject.toml' \) \
      | sort

    print_file_if_exists "environment.yml"
    print_file_if_exists "environment.yaml"
    print_file_if_exists "requirements.txt"
    print_file_if_exists "requirements_dev.txt"
    print_file_if_exists "setup.py"
    print_file_if_exists "pyproject.toml"

    echo
    echo "[candidate extension setup files]"
    find . -path './SIBR_viewers' -prune -o \
      -type f \( -name 'setup.py' -o -name 'pyproject.toml' -o -name 'CMakeLists.txt' \) \
      -print | sort

    echo
    echo "[cuda/cpp source counts outside SIBR_viewers]"
    echo -n ".cu files: "
    find . -path './SIBR_viewers' -prune -o -type f -name '*.cu' -print | wc -l
    echo -n ".cpp files: "
    find . -path './SIBR_viewers' -prune -o -type f -name '*.cpp' -print | wc -l
    echo -n ".h/.hpp files: "
    find . -path './SIBR_viewers' -prune -o -type f \( -name '*.h' -o -name '*.hpp' \) -print | wc -l

    echo
    echo "[likely package/import names]"
    find . -path './SIBR_viewers' -prune -o \
      -type f \( -name '*.py' -o -name 'setup.py' \) \
      -print0 \
      | xargs -0 grep -HnE "diff_gaussian|diff_surfel|diff_t|simple_knn|rasterization|ext_modules|CUDAExtension|CppExtension" 2>/dev/null \
      | head -220 || true
  )
}

mkdir -p "${PROJECT_ROOT}/outputs"

{
echo "Unified 3DGS third-party inspection"
echo "Generated at: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Third-party root: ${THIRD_PARTY_ROOT}"

echo
echo "============================================================"
echo "Environment"
echo "============================================================"
echo "PWD: $(pwd)"
echo "Python: $(which python 2>/dev/null || true)"
python --version 2>/dev/null || true
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "CUDA_PATH=${CUDA_PATH:-}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-}"
echo "MAX_JOBS=${MAX_JOBS:-}"

for repo in "${repos[@]}"; do
  inspect_repo "${repo}"
done

echo
echo "============================================================"
echo "Inspection complete"
echo "============================================================"
} 2>&1 | tee "${LOG_FILE}"

echo
echo "Saved inspection log to: ${LOG_FILE}"
