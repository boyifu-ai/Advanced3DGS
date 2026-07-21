"""Prevent wrapped third-party methods from mutating the shared dataset root."""

from __future__ import annotations

import builtins
import functools
import io
import os
from typing import Any, Optional


_ROOT_VALUE = os.environ.get("UNIFIED3DGS_READONLY_DATASET_ROOT", "").strip()
_READONLY_ROOTS = tuple(
    os.path.realpath(value)
    for value in _ROOT_VALUE.split(os.pathsep)
    if value.strip()
)
_COLMAP_SPARSE_ZERO_FALLBACK = (
    os.environ.get("UNIFIED3DGS_COLMAP_SPARSE_ZERO_FALLBACK", "").strip() == "1"
)
_COLMAP_SPARSE_FILES = {
    "cameras.bin",
    "cameras.txt",
    "images.bin",
    "images.txt",
    "points3D.bin",
    "points3D.ply",
    "points3D.txt",
}

if (
    os.environ.get("UNIFIED3DGS_PY38_FUNCTOOLS_CACHE", "").strip() == "1"
    and not hasattr(functools, "cache")
):
    functools.cache = functools.lru_cache(maxsize=None)

if os.environ.get("UNIFIED3DGS_NUMPY_LEGACY_ALIASES", "").strip() == "1":
    import numpy as _numpy

    for _alias, _builtin in (
        ("bool", bool),
        ("complex", complex),
        ("float", float),
        ("int", int),
        ("object", object),
        ("str", str),
    ):
        if _alias not in _numpy.__dict__:
            setattr(_numpy, _alias, _builtin)

_ORIGINAL_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_MUTATIONS = {
    name: getattr(os, name)
    for name in (
        "chmod",
        "chown",
        "link",
        "mkdir",
        "mkfifo",
        "mknod",
        "remove",
        "rename",
        "replace",
        "rmdir",
        "symlink",
        "truncate",
        "unlink",
        "utime",
    )
    if hasattr(os, name)
}


def _resolved(path: Any) -> Optional[str]:
    if isinstance(path, int):
        return None
    try:
        return os.path.realpath(os.path.abspath(os.fsdecode(path)))
    except (TypeError, ValueError):
        return None


def _inside_readonly_root(path: Any) -> bool:
    resolved = _resolved(path)
    if not _READONLY_ROOTS or resolved is None:
        return False
    for root in _READONLY_ROOTS:
        try:
            if os.path.commonpath((root, resolved)) == root:
                return True
        except ValueError:
            continue
    return False


def _deny(operation: str, path: Any) -> None:
    raise PermissionError(
        f"Unified 3DGS blocked {operation} under read-only dataset root "
        f"{os.pathsep.join(_READONLY_ROOTS)}: {_resolved(path) or path!r}"
    )


def _read_path(path: Any) -> Any:
    """Resolve a method-specific, read-only COLMAP layout fallback."""
    if not _COLMAP_SPARSE_ZERO_FALLBACK or isinstance(path, int):
        return path
    resolved = _resolved(path)
    if resolved is None or not _inside_readonly_root(resolved):
        return path
    if os.path.exists(resolved):
        return path
    parent, name = os.path.split(resolved)
    if os.path.basename(parent) != "sparse" or name not in _COLMAP_SPARSE_FILES:
        return path
    candidate = os.path.join(parent, "0", name)
    return candidate if os.path.isfile(candidate) else path


def _guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
    if any(flag in mode for flag in ("w", "a", "x", "+")) and _inside_readonly_root(file):
        _deny(f"open(mode={mode!r})", file)
    read_file = file if any(flag in mode for flag in ("w", "a", "x", "+")) else _read_path(file)
    return _ORIGINAL_OPEN(read_file, mode, *args, **kwargs)


def _guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
    if any(flag in mode for flag in ("w", "a", "x", "+")) and _inside_readonly_root(file):
        _deny(f"io.open(mode={mode!r})", file)
    read_file = file if any(flag in mode for flag in ("w", "a", "x", "+")) else _read_path(file)
    return _ORIGINAL_IO_OPEN(read_file, mode, *args, **kwargs)


class _GuardedOsOpen:
    """Callable object avoids Python-function descriptor binding in pathlib."""

    def __call__(self, path: Any, flags: int, *args: Any, **kwargs: Any):
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags and _inside_readonly_root(path):
            _deny(f"os.open(flags={flags})", path)
        read_path = path if flags & write_flags else _read_path(path)
        return _ORIGINAL_OS_OPEN(read_path, flags, *args, **kwargs)


class _GuardedOnePathMutation:
    """Callable object avoids descriptor binding when pathlib snapshots os funcs."""

    def __init__(self, name: str):
        self.name = name
        self.original = _ORIGINAL_MUTATIONS[name]

    def __call__(self, path: Any, *args: Any, **kwargs: Any):
        if _inside_readonly_root(path):
            _deny(f"os.{self.name}", path)
        return self.original(path, *args, **kwargs)


class _GuardedTwoPathMutation:
    """Callable object avoids descriptor binding when pathlib snapshots os funcs."""

    def __init__(self, name: str):
        self.name = name
        self.original = _ORIGINAL_MUTATIONS[name]

    def __call__(self, src: Any, dst: Any, *args: Any, **kwargs: Any):
        if _inside_readonly_root(src):
            _deny(f"os.{self.name}(source)", src)
        if _inside_readonly_root(dst):
            _deny(f"os.{self.name}(destination)", dst)
        return self.original(src, dst, *args, **kwargs)


if _READONLY_ROOTS:
    builtins.open = _guarded_open
    io.open = _guarded_io_open
    os.open = _GuardedOsOpen()
    for _name in (
        "chmod",
        "chown",
        "mkdir",
        "mkfifo",
        "mknod",
        "remove",
        "rmdir",
        "truncate",
        "unlink",
        "utime",
    ):
        if _name not in _ORIGINAL_MUTATIONS:
            continue
        setattr(os, _name, _GuardedOnePathMutation(_name))
    for _name in ("link", "rename", "replace", "symlink"):
        if _name not in _ORIGINAL_MUTATIONS:
            continue
        setattr(os, _name, _GuardedTwoPathMutation(_name))
