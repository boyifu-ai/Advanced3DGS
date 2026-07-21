# Third-Party Repositories

Method source repositories are pinned here as Git submodules. Clone the project
with its exact validated upstream revisions:

```bash
git clone --recursive https://github.com/3DAgentWorld/Advanced3DGS.git
```

For an existing checkout:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Do not replace submodules with copied repositories or commit generated build
products here. CUDA extensions belong under `third_party_build/` and are rebuilt
for the local Python, PyTorch, CUDA, compiler, and GPU architecture.
