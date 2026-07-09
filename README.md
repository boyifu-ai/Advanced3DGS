# A Unified 3D Gaussian Splatting Rasterization Framework
This is the official code repository for **"3D Gaussian Splatting Rasterization: A Survey"**. A unified framework for reproducing, benchmarking, and comparing state-of-the-art (SOTA) 3DGS methods under a common training, rendering, and evaluation pipeline.

## TO DO LIST
- [x] Vanilla 3DGS (SIGGRAPH 2023)
- [x] 2DGS (SIGGRAPH 2024)
- [x] 3DGS-MCMC (NeurIPS 2024)
- [x] 3D-HGS (CVPR 2025)
- [x] 3D-SSS (CVPR 2025)
- [x] GES (CVPR 2024)
- [x] BetaSplatting (SIGGRAPH 2025)
- [x] LightGaussian (NeurIPS 2024)
- [x] Mini-Splatting (ECCV 2024)
- [x] Speedy-Splat (CVPR 2025)
- [x] Taming-3DGS (SIGGRAPH Asia 2024)
- [x] GHAP (NeurIPS 2025)
- [x] DashGaussian (CVPR 2025)
- [x] FastGS (CVPR 2026)
- [ ] MMGS (arXiv 2026)
- [x] GOF (SIGGRAPH Asia 2024)
- [x] PGSR (TVCG 2024)
- [x] Mip-Splatting (CVPR 2024)
- [x] Scaffold-GS (CVPR 2024)
- [x] Octree-GS (TPAMI 2025)
- [x] Wavelet-GS (ACMMM 2025)
- [x] Compact-3DGS (CVPR 2024)
- [x] ContextGS (NeurIPS 2024)
- [x] HAC++ (TPAMI 2025)
- [x] 3DCS (CVPR 2025)

## Third-party Source Repositories
| Method | Upstream repository |
| --- | --- |
| Vanilla 3DGS | `https://github.com/graphdeco-inria/gaussian-splatting` |
| 2DGS | `https://github.com/hbb1/2d-gaussian-splatting` |
| 3DGS-MCMC | `https://github.com/ubc-vision/3dgs-mcmc` |
| 3D-HGS | `https://github.com/lihaolin88/3DHGS` |
| 3D-SSS | `https://github.com/realcrane/3D-student-splatting-and-scooping` |
| GES | `https://github.com/ajhamdi/ges-splatting` |
| BetaSplatting | `https://github.com/RongLiu-Leo/Beta-Splatting` |
| LightGaussian | `https://github.com/VITA-Group/LightGaussian` |
| Mini-Splatting | `https://github.com/fatPeter/mini-splatting` |
| Speedy-Splat | `https://github.com/j-alex-hanson/speedy-splat` |
| Taming-3DGS | `https://github.com/humansensinglab/taming-3dgs` |
| GHAP | `https://github.com/taowang0105/GHAP` |
| DashGaussian | `https://github.com/YouyuChen0207/DashGaussian` |
| FastGS | `https://github.com/fastgs/FastGS` |
| GOF / Gaussian Opacity Fields | `https://github.com/autonomousvision/gaussian-opacity-fields` |
| PGSR | `https://github.com/zju3dv/PGSR` |
| Mip-Splatting | `https://github.com/autonomousvision/mip-splatting` |
| Scaffold-GS | `https://github.com/city-super/Scaffold-GS` |
| Octree-GS | `https://github.com/city-super/Octree-GS` |
| Wavelet-GS | `https://github.com/ALEX5874/Wavelet-GS` |
| Compact-3DGS | `https://github.com/maincold2/Compact-3DGS` |
| ContextGS | `https://github.com/wyf0912/ContextGS` |
| HAC++ | `https://github.com/YihangChen-ee/HAC-plus` |
| 3DCS | `https://github.com/convexsplatting/convex-splatting` |

## Supported Datasets (11 scenes)
- [x] Mip-NeRF 360: bicycle, bonsai, counter, garden, kitchen, room, stump.
- [x] Tanks & Temples: train, truck.
- [x] Deep Blending: drjohnson, playroom.

## Metric Aggregation
Supported metrics:

- [x] PSNR ↑
- [x] SSIM ↑
- [x] LPIPS ↓

Hierarchical aggregation levels:

- **Scene-Level**: metrics result of individual scene
- **Dataset-Level**: average over scenes in a dataset
- **Method-Level**: average over all datasets for a method

Metric files output:

```text
outputs/validation/<method>/<dataset>/<scene>/metrics_summary.json
outputs/validation/<method>/<dataset>/<scene>/metrics_summary.md
outputs/validation/<method>/<dataset>/metrics_summary.csv
outputs/validation/<method>/<dataset>/metrics_summary.md
outputs/validation/<method>/metrics_summary.csv
outputs/validation/<method>/metrics_summary.md
```

## Interactive Menu

The menu provides:

- [x] Roman-numeral method selection.
- [x] Dataset path specification.
- [x] Letter-based dataset-family selection.
- [x] GPU status display and explicit GPU selection before training.
- [x] Parameter review and simple `key=value` editing.
- [x] Environment and method self-checks.
- [x] Method repository setup and capability checks.
- [x] Training, rendering, evaluation, resumable progress, and logs.
- [x] Scene-level, dataset-level, and method-level metric aggregation.
- [x] System resource status for GPU, memory, disk, and project directory size.

Each train/render/evaluate stage writes a log beside the scene output:

```text
outputs/validation/<method>/<dataset>/<scene>/train.log
outputs/validation/<method>/<dataset>/<scene>/render.log
outputs/validation/<method>/<dataset>/<scene>/eval.log
```

One Command Run:

```bash
python unified3dgs_menu.py
```

Use this command to list registered CLI names:

```bash
python train_all.py --help
```

Dataset roots are user configuration. The preferred setup path is in the menu, choosing `Dataset paths` and set:

```text
Mip-NeRF 360      /path/to/MIP360
Tanks & Temples   /path/to/tandt
Deep Blending     /path/to/deep_blending
```

The menu stores local overrides in `configs/local_dataset_paths.json`. You may also configure paths through environment variables:

```bash
export ADVANCED3DGS_MIP360_ROOT=/path/to/MIP360
export ADVANCED3DGS_TANDT_ROOT=/path/to/tandt
export ADVANCED3DGS_DEEP_BLENDING_ROOT=/path/to/deep_blending
```

## Setup

Example for creating the shared framework environment:

```bash
git clone https://github.com/3DAgentWorld/Advanced3DGS.git
cd Advanced3DGS
conda env create -f environment.yml
conda activate unified-3dgs

export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8
```

Install Python runtime dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/check_runtime_dependencies.py
```

Clone third-party repositories:

```bash
python scripts/manage_method_repositories.py commands --all
RUN_REAL=1 bash scripts/clone_method_repositories.sh
```

Build native extensions and isolated method dependencies:

```bash
python scripts/prepare_method_acceptance.py --run-real
```

Some methods require official isolated backends. Use the dedicated setup scripts instead of installing those dependencies into the shared environment:

```bash
bash scripts/setup_hac_plus_official_backend.sh
bash scripts/setup_3dcs_official_backend.sh
```

Then export the backend Python variables printed by the setup scripts, for example:

```bash
export UNIFIED3DGS_HAC_PLUS_PYTHON="$PWD/envs/hac_plus/bin/python"
export UNIFIED3DGS_3DCS_PYTHON="$PWD/envs/3dcs/bin/python"
```







