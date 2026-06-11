# Advanced3DGS: A Unified 3DGS Framework
This is the official code repository for **"Advances in Rasterization Rendering Algorithms for 3D Gaussian Splatting: A Comprehensive Survey"**. A unified framework for reproducing, benchmarking, and comparing state-of-the-art (SOTA) 3DGS methods under a common training, rendering, and evaluation pipeline.

## TO DO LIST
- [x] Vanilla 3DGS (SIGGRAPH 2023)
- [x] 2DGS (SIGGRAPH 2024)
- [x] 3DGS-MCMC (NeurIPS 2024)
- [x] 3D-HGS (CVPR 2025)
- [x] 3D-SSS (CVPR 2025)
- [ ] GES (CVPR 2024)
- [ ] BetaSplatting (SIGGRAPH 2025)
- [ ] LightGaussian (NeurIPS 2024)
- [ ] Mini-Splatting (ECCV 2024)
- [ ] Speedy-Splat (CVPR 2025)
- [ ] Taming-3DGS (SIGGRAPH Asia 2024)
- [ ] GHAP (NeurIPS 2025)
- [ ] DashGaussian (CVPR 2025)
- [ ] FastGS (CVPR 2026)
- [ ] MMGS (arXiv 2026)
- [ ] GOF (SIGGRAPH Asia 2024)
- [ ] PGSR (TVCG 2024)
- [ ] Mip-Splatting (CVPR 2024)
- [ ] Scaffold-GS (CVPR 2024)
- [ ] Octree-GS (TPAMI 2025)
- [ ] Wavelet-GS (ACMMM 2025)
- [ ] Compact-3DGS (CVPR 2024)
- [ ] ContextGS (NeurIPS 2024)
- [ ] HAC++ (TPAMI 2025)

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

## Supported Datasets
- [x] Mip-NeRF 360
- [x] Tanks & Temples
- [x] Deep Blending

## Metric Aggregation
Supported metrics:

- PSNR ↑
- SSIM ↑
- LPIPS ↓

Hierarchical aggregation levels:
- **Scene-Level**: metrics result of individual scene
- **Dataset-Level**: average over scenes in a dataset
- **Method-Level**: average over all datasets for a method
