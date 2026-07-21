# Third-Party Build Artifacts

Method-specific native extensions and runtime caches are generated here.

Typical generated paths:

```text
third_party_build/<method>/site-packages/
third_party_build/runtime/
```

The framework prepends the selected method's `site-packages` path to
`PYTHONPATH` before launching upstream code. This keeps incompatible CUDA
extension forks isolated.

Do not commit generated contents from this directory.
