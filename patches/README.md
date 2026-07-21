# Third-Party Patches

Third-party repositories are kept out of git, so local compatibility fixes are
applied by scripts instead of committing modified upstream files.

## Robust PLY Reader

Script:

```bash
python scripts/patch_third_party_readers.py
```

Reason:

Some COLMAP-style datasets include `points3D.ply` files without normal fields
(`nx`, `ny`, `nz`). Several 3DGS readers assume normals exist and then silently
set the point cloud to `None` when loading fails. The patch makes `fetchPly`
tolerate missing normals by filling zeros and missing RGB values by filling
neutral gray.

Scope:

- `third_party/gaussian-splatting/scene/dataset_readers.py`
- `third_party/2d-gaussian-splatting/scene/dataset_readers.py`
- `third_party/3dgs-mcmc/scene/dataset_readers.py`
- `third_party/3DHGS/scene/dataset_readers.py`
- `third_party/3D-student-splatting-and-scooping/scene/dataset_readers.py`

Backups are written next to patched files as:

```text
dataset_readers.py.bak_unified3dgs
```

This is a minimal compatibility patch. It does not modify configured dataset
roots.

For 3DGS-MCMC random initialization, the patch also redirects the temporary
random initialization PLY into the experiment output directory. This preserves
the upstream random initialization strategy without writing `random.ply` into
the shared dataset.

## 3D-HGS Test GT Output

Script:

```bash
python scripts/patch_3dhgs_render_gt.py
```

Reason:

The upstream 3D-HGS renderer writes test renders and reports PSNR, but does not
write paired test GT images. Unified evaluation requires paired `renders/` and
`gt/` directories to compute PSNR, SSIM, and LPIPS through the shared metrics
script.

Scope:

```text
third_party/3DHGS/render.py
```

The backup is written as `render.py.bak_unified3dgs_gt`.

## Patch Verification

Run:

```bash
python scripts/check_method_scene_readiness.py --methods \
  vanilla_3dgs 2dgs 3dgs_mcmc 3dhgs sss
```

The full formal-scene runner applies required reader patches automatically for
selected methods and applies the 3D-HGS GT patch when `3dhgs` is selected.
