from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_REVISION = "catalog-acceptance-dataset-v3"
MAX_ACCEPTANCE_POINTS = 10_000
IMAGE_ALIASES = ("images", "images_2", "images_4", "images_8")
SPARSE_CAMERA_FILES = (
    "cameras.bin",
    "cameras.txt",
    "images.bin",
    "images.txt",
)
REQUIRED_PLY_FIELDS = ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")


def _source_sparse(dataset: Path) -> Path:
    nested = dataset / "sparse" / "0"
    if nested.is_dir():
        return nested
    direct = dataset / "sparse"
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"Dataset has no COLMAP sparse directory: {dataset}")


def _source_ply(dataset: Path) -> Path:
    sparse = _source_sparse(dataset)
    candidates = (sparse / "points3D.ply", dataset / "points3D.ply")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Dataset overlay requires an existing points3D.ply so it can add missing "
        f"normal/color fields without mutating the shared dataset: {dataset}"
    )


def _preferred_images(dataset: Path) -> Path:
    for name in ("images_8", "images_4", "images_2", "images"):
        candidate = dataset / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Dataset has no supported images directory: {dataset}")


def _overlay_signature(
    dataset: Path,
    max_points: int = MAX_ACCEPTANCE_POINTS,
) -> Tuple[str, Dict[str, object]]:
    dataset = dataset.resolve()
    sparse = _source_sparse(dataset)
    ply = _source_ply(dataset)
    images = _preferred_images(dataset)
    inputs: Dict[str, object] = {
        "revision": OVERLAY_REVISION,
        "dataset": str(dataset),
        "sparse": str(sparse.resolve()),
        "point_cloud": str(ply.resolve()),
        "point_cloud_size": ply.stat().st_size,
        "point_cloud_mtime_ns": ply.stat().st_mtime_ns,
        "max_acceptance_points": max_points,
        "images": str(images.resolve()),
    }
    digest = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return digest, inputs


def _link(source: Path, destination: Path) -> None:
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def normalize_point_cloud(
    source: Path,
    destination: Path,
    max_points: int = MAX_ACCEPTANCE_POINTS,
) -> Dict[str, object]:
    try:
        import numpy as np
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise RuntimeError(
            "Preparing the dataset overlay requires numpy and plyfile in the unified "
            "environment."
        ) from exc

    ply = PlyData.read(str(source))
    vertices = ply["vertex"].data
    source_vertex_count = int(len(vertices))
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if source_vertex_count > max_points:
        indices = np.linspace(
            0, source_vertex_count - 1, num=max_points, dtype=np.int64
        )
        vertices = vertices[indices]
    names = tuple(vertices.dtype.names or ())
    missing_positions = [name for name in ("x", "y", "z") if name not in names]
    if missing_positions:
        raise ValueError(
            f"Point cloud lacks required position fields {missing_positions}: {source}"
        )

    additions = []
    for name in ("nx", "ny", "nz"):
        if name not in names:
            additions.append((name, "f4"))
    for name in ("red", "green", "blue"):
        if name not in names:
            additions.append((name, "u1"))

    dtype = list(vertices.dtype.descr) + additions
    normalized = np.empty(vertices.shape, dtype=dtype)
    for name in names:
        normalized[name] = vertices[name]
    for name, _ in additions:
        normalized[name] = 127 if name in {"red", "green", "blue"} else 0.0

    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(normalized, "vertex")], text=False).write(
        str(destination)
    )
    output_names = tuple(PlyData.read(str(destination))["vertex"].data.dtype.names or ())
    missing = [name for name in REQUIRED_PLY_FIELDS if name not in output_names]
    if missing:
        raise ValueError(f"Normalized acceptance point cloud still lacks fields: {missing}")
    return {
        "source_fields": list(names),
        "added_fields": [name for name, _ in additions],
        "output_fields": list(output_names),
        "source_vertex_count": source_vertex_count,
        "vertex_count": int(len(normalized)),
        "max_acceptance_points": max_points,
        "point_cap_applied": source_vertex_count > max_points,
    }


def write_colmap_point_cloud(
    ply_path: Path,
    binary_path: Path,
    text_path: Path,
) -> Dict[str, object]:
    """Write matching reduced COLMAP point files from the normalized PLY."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "Preparing the dataset overlay requires plyfile in the unified environment."
        ) from exc

    vertices = PlyData.read(str(ply_path))["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            f"Cannot write reduced COLMAP points; PLY lacks fields {missing}: {ply_path}"
        )

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    with binary_path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(vertices)))
        for index, vertex in enumerate(vertices, 1):
            handle.write(
                struct.pack(
                    "<QdddBBBdQ",
                    index,
                    float(vertex["x"]),
                    float(vertex["y"]),
                    float(vertex["z"]),
                    int(vertex["red"]),
                    int(vertex["green"]),
                    int(vertex["blue"]),
                    0.0,
                    0,
                )
            )

    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(vertices)}, mean track length: 0",
    ]
    lines.extend(
        (
            f"{index} {float(vertex['x']):.17g} {float(vertex['y']):.17g} "
            f"{float(vertex['z']):.17g} {int(vertex['red'])} "
            f"{int(vertex['green'])} {int(vertex['blue'])} 0"
        )
        for index, vertex in enumerate(vertices, 1)
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return {
        "point_count": int(len(vertices)),
        "binary_size": binary_path.stat().st_size,
        "text_size": text_path.stat().st_size,
    }


def _link_existing_files(
    source_root: Path, destinations: Iterable[Path], names: Iterable[str]
) -> None:
    for name in names:
        source = source_root / name
        if not source.is_file():
            continue
        for destination_root in destinations:
            _link(source, destination_root / name)


def _complete_cached_overlay(target: Path, inputs: Dict[str, object]) -> bool:
    manifest_path = target / "overlay_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("inputs") != inputs:
            return False
        if any(not (target / alias).is_dir() for alias in IMAGE_ALIASES):
            return False
        for sparse in (target / "sparse", target / "sparse" / "0"):
            if any(
                not (sparse / name).is_file()
                for name in (
                    "cameras.bin",
                    "images.bin",
                    "points3D.bin",
                    "points3D.txt",
                )
            ):
                return False
        point_cloud = target / "sparse" / "0" / "points3D.ply"
        from plyfile import PlyData

        fields = set(PlyData.read(str(point_cloud))["vertex"].data.dtype.names or ())
        return set(REQUIRED_PLY_FIELDS).issubset(fields)
    except Exception:
        return False


def prepare_dataset_overlay(
    dataset: Path,
    output_root: Path,
    max_points: int = MAX_ACCEPTANCE_POINTS,
) -> Path:
    dataset = dataset.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    try:
        output_root.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Dataset overlay output root must stay inside the project: {output_root}"
        ) from exc
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    digest, inputs = _overlay_signature(dataset, max_points=max_points)
    overlays = output_root / "runtime" / "dataset_overlays"
    target = overlays / f"{dataset.name}_{digest}"
    if _complete_cached_overlay(target, inputs):
        return target

    overlays.mkdir(parents=True, exist_ok=True)
    staging = overlays / f".{target.name}.{os.getpid()}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        preferred_images = _preferred_images(dataset)
        for alias in IMAGE_ALIASES:
            _link(preferred_images, staging / alias)

        for child in dataset.iterdir():
            if child.name == "sparse" or child.name in IMAGE_ALIASES:
                continue
            _link(child, staging / child.name)

        source_sparse = _source_sparse(dataset)
        sparse_root = staging / "sparse"
        sparse_zero = sparse_root / "0"
        sparse_zero.mkdir(parents=True)
        _link_existing_files(
            source_sparse,
            (sparse_root, sparse_zero),
            SPARSE_CAMERA_FILES,
        )

        point_cloud = sparse_zero / "points3D.ply"
        ply_details = normalize_point_cloud(
            _source_ply(dataset),
            point_cloud,
            max_points=max_points,
        )
        (sparse_root / "points3D.ply").symlink_to(Path("0") / "points3D.ply")
        colmap_point_details = write_colmap_point_cloud(
            point_cloud,
            sparse_zero / "points3D.bin",
            sparse_zero / "points3D.txt",
        )
        (sparse_root / "points3D.bin").symlink_to(Path("0") / "points3D.bin")
        (sparse_root / "points3D.txt").symlink_to(Path("0") / "points3D.txt")

        manifest = {
            "revision": OVERLAY_REVISION,
            "inputs": inputs,
            "overlay": str(target),
            "purpose": "iteration=1 completeness testing only",
            "preferred_images": str(preferred_images.resolve()),
            "ply": ply_details,
            "colmap_points": colmap_point_details,
        }
        (staging / "overlay_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target
