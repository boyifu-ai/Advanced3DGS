from __future__ import annotations

import argparse
from pathlib import Path

from plyfile import PlyData


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect PLY vertex fields.")
    parser.add_argument("ply", type=Path)
    args = parser.parse_args()

    ply_path = args.ply.expanduser().resolve()
    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"]
    names = vertices.data.dtype.names or ()

    print(f"PLY: {ply_path}")
    print(f"Vertex count: {len(vertices.data)}")
    print("Fields:")
    for name in names:
        print(f"- {name}")
    print(f"has xyz: {all(name in names for name in ('x', 'y', 'z'))}")
    print(f"has rgb: {all(name in names for name in ('red', 'green', 'blue'))}")
    print(f"has normals: {all(name in names for name in ('nx', 'ny', 'nz'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
