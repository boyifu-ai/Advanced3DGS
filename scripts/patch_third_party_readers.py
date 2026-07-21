from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPOS = {
    "vanilla_3dgs": "third_party/gaussian-splatting",
    "2dgs": "third_party/2d-gaussian-splatting",
    "3dgs_mcmc": "third_party/3dgs-mcmc",
    "3dhgs": "third_party/3DHGS",
    "3dcs": "third_party/convex-splatting",
    "sss": "third_party/3D-student-splatting-and-scooping",
    "beta_splatting": "third_party/Beta-Splatting",
    "compact_3dgs": "third_party/Compact-3DGS",
    "contextgs": "third_party/ContextGS",
    "dashgaussian": "third_party/DashGaussian",
    "fastgs": "third_party/FastGS",
    "ges": "third_party/ges-splatting",
    "ghap": "third_party/GHAP",
    "gof": "third_party/gaussian-opacity-fields",
    "hac_plus": "third_party/HAC-plus",
    "lightgaussian": "third_party/LightGaussian",
    "mini_splatting": "third_party/mini-splatting",
    "mip_splatting": "third_party/mip-splatting",
    "octree_gs": "third_party/Octree-GS",
    "pgsr": "third_party/PGSR",
    "scaffold_gs": "third_party/Scaffold-GS",
    "speedy_splat": "third_party/speedy-splat",
    "taming_3dgs": "third_party/taming-3dgs",
    "wavelet_gs": "third_party/Wavelet-GS",
}

PATCH_MARKER = "Unified 3DGS robust PLY reader patch"
NO_WRITE_MARKER = "Unified 3DGS no dataset-write point cloud patch"
MCMC_RANDOM_MARKER = "Unified 3DGS no dataset-write random initialization patch v2"
FETCH_PLY_PATTERN = re.compile(r"(?ms)^def fetchPly\(path\):\n.*?(?=^def |\Z)")
POINT_CLOUD_BLOCK_PATTERN = re.compile(
    r"""(?ms)^    ply_path = os\.path\.join\(path, "sparse/0/points3D\.ply"\)\n"""
    r"""    bin_path = os\.path\.join\(path, "sparse/0/points3D\.bin"\)\n"""
    r"""    txt_path = os\.path\.join\(path, "sparse/0/points3D\.txt"\)\n"""
    r"""    if not os\.path\.exists\(ply_path\):\n"""
    r""".*?"""
    r"""^    (?:    )?pcd = fetchPly\(ply_path\)\n"""
    r"""(?:^    except:\n^        pcd = None\n|^    # except:\n^    #     pcd = None\n)?"""
)
MCMC_RANDOM_BLOCK_PATTERN = re.compile(
    r"""(?ms)^    elif init_type == "random":\n"""
    r""".*?"""
    r"""^    else:\n"""
)

ROBUST_FETCH_PLY = f'''def fetchPly(path):
    """Load a PLY point cloud with tolerant field handling.

    {PATCH_MARKER}.
    Some shared datasets provide COLMAP-derived PLY files without normal fields.
    The upstream readers assume nx/ny/nz exist, then silently return None when
    loading fails. For validation, missing normals are filled with zeros and
    missing colors are filled with neutral gray.
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    names = vertices.data.dtype.names or ()

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if all(channel in names for channel in ('red', 'green', 'blue')):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.full_like(positions, 0.5, dtype=np.float32)

    if all(channel in names for channel in ('nx', 'ny', 'nz')):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions, dtype=np.float32)

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


'''

FETCH_COLMAP_POINT_CLOUD = f'''def fetchColmapPointCloud(ply_path, bin_path, txt_path):
    """Load COLMAP point cloud without writing generated files to the dataset.

    {NO_WRITE_MARKER}.
    Upstream readers create points3D.ply next to points3D.bin the first time a
    scene is opened. Shared datasets are read-only for this project, so missing
    PLY files are handled in memory from points3D.bin or points3D.txt.
    """
    if os.path.exists(ply_path):
        return fetchPly(ply_path)

    print("Loading COLMAP point cloud directly from points3D.bin/txt without writing points3D.ply.")
    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except Exception:
        xyz, rgb, _ = read_points3D_text(txt_path)

    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb) / 255.0
    normals = np.zeros_like(xyz, dtype=np.float32)
    return BasicPointCloud(points=xyz, colors=rgb, normals=normals)


'''

NO_WRITE_POINT_CLOUD_BLOCK = '''    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    try:
        pcd = fetchColmapPointCloud(ply_path, bin_path, txt_path)
    except Exception as exc:
        print("Failed to load COLMAP point cloud:", repr(exc))
        pcd = None
'''

MCMC_RANDOM_BLOCK = f'''    elif init_type == "random":
        # {MCMC_RANDOM_MARKER}.
        # Random initialization is preserved, but its temporary PLY belongs to
        # the experiment output rather than the shared read-only dataset.
        random_root = os.environ.get("UNIFIED3DGS_OUTPUT_PATH")
        if not random_root:
            raise RuntimeError(
                "UNIFIED3DGS_OUTPUT_PATH is required for read-only random initialization."
            )
        os.makedirs(random_root, exist_ok=True)
        ply_path = os.path.join(random_root, "random_init.ply")
        num_pts = 100_000
        print(f"Generating random point cloud ({{num_pts}}) at {{ply_path}}...")
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 * 2 - (nerf_normalization["radius"] * 3)
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
'''


def _parsed_module(text: str, path: Path) -> ast.Module:
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"cannot parse dataset reader {path}: {exc}") from exc


def _top_level_functions(tree: ast.Module) -> Dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _callback_names(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "sceneLoadTypeCallbacks" for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            names.update(item.id for item in value.values if isinstance(item, ast.Name))
    return names


def _direct_assignment_name(node: ast.AST) -> str:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return ""
    return target.id if isinstance(target, ast.Name) else ""


def _assigns_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            targets = child.targets
        elif isinstance(child, ast.AnnAssign):
            targets = [child.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return True
    return False


def _nested_statement_lists(statements: List[ast.stmt]):
    yield statements
    for statement in statements:
        for attribute in ("body", "orelse", "finalbody"):
            nested = getattr(statement, attribute, None)
            if isinstance(nested, list) and nested:
                yield from _nested_statement_lists(nested)
        handlers = getattr(statement, "handlers", [])
        for handler in handlers:
            if isinstance(handler, ast.ExceptHandler) and handler.body:
                yield from _nested_statement_lists(handler.body)


def replace_colmap_point_cloud_block(path: Path, text: str) -> Tuple[str, int]:
    """Replace the COLMAP conversion block while preserving upstream paths."""
    tree = _parsed_module(text, path)
    lines = text.splitlines(keepends=True)
    required_names = ("ply_path", "bin_path", "txt_path")

    for function in tree.body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function.name != "readColmapSceneInfo":
            continue

        matched_statements = None
        direct_assignments = {}
        for statements in _nested_statement_lists(function.body):
            candidate = {
                _direct_assignment_name(statement): (index, statement)
                for index, statement in enumerate(statements)
                if _direct_assignment_name(statement)
            }
            if all(name in candidate for name in required_names):
                matched_statements = statements
                direct_assignments = candidate
                break
        if matched_statements is None:
            continue
        start_index = min(direct_assignments[name][0] for name in required_names)
        last_path_index = max(direct_assignments[name][0] for name in required_names)
        end_statement = next(
            (
                statement
                for statement in matched_statements[last_path_index + 1 :]
                if _assigns_name(statement, "pcd")
            ),
            None,
        )
        nested_paths = matched_statements is not function.body
        if end_statement is None and nested_paths:
            end_statement = matched_statements[-1]
        if end_statement is None:
            continue

        start_statement = matched_statements[start_index]
        end_lineno = getattr(end_statement, "end_lineno", None)
        if end_lineno is None:
            raise RuntimeError(f"cannot determine COLMAP point-cloud block in {path}")
        indent = " " * start_statement.col_offset
        assignment_lines: List[str] = []
        for name in required_names:
            statement = direct_assignments[name][1]
            statement_end = getattr(statement, "end_lineno", None)
            if statement_end is None:
                raise RuntimeError(f"cannot determine {name} assignment in {path}")
            source = "".join(lines[statement.lineno - 1 : statement_end])
            assignment_lines.append(source.rstrip("\r\n") + "\n")

        replacement = assignment_lines + [
            f"{indent}try:\n",
            f"{indent}    pcd = fetchColmapPointCloud(ply_path, bin_path, txt_path)\n",
            f"{indent}except Exception as exc:\n",
            f"{indent}    print(\"Failed to load COLMAP point cloud:\", repr(exc))\n",
            f"{indent}    pcd = None\n",
        ]
        if nested_paths:
            container_index = next(
                (
                    index
                    for index, statement in enumerate(function.body)
                    if statement.lineno <= start_statement.lineno
                    and getattr(statement, "end_lineno", statement.lineno) >= end_lineno
                ),
                None,
            )
            trailing_pcd = None
            if container_index is not None:
                container = function.body[container_index]
                if _assigns_name(container, "pcd"):
                    trailing_pcd = next(
                        (
                            statement
                            for statement in function.body[container_index + 1 :]
                            if _assigns_name(statement, "pcd")
                        ),
                        None,
                    )
            if trailing_pcd is None:
                continue
            trailing_end = getattr(trailing_pcd, "end_lineno", None)
            if trailing_end is None:
                raise RuntimeError(f"cannot determine trailing PLY load in {path}")
            del lines[trailing_pcd.lineno - 1 : trailing_end]
        lines[start_statement.lineno - 1 : end_lineno] = replacement
        return "".join(lines), 1
    return text, 0


def _pristine_reader_source(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        repo_root = Path(completed.stdout.strip()).resolve()
        relative = path.resolve().relative_to(repo_root).as_posix()
        source = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if source.returncode == 0 and source.stdout.strip():
            return source.stdout
    backup = path.with_suffix(path.suffix + ".bak_unified3dgs")
    if backup.is_file():
        return backup.read_text(encoding="utf-8")
    raise RuntimeError(
        f"cannot recover pristine reader source for {path}; Git HEAD and backup are unavailable"
    )


def repair_missing_callback_functions(path: Path, text: str) -> Tuple[str, List[str]]:
    patched_tree = _parsed_module(text, path)
    patched_functions = _top_level_functions(patched_tree)
    missing = _callback_names(patched_tree) - set(patched_functions)
    if not missing:
        return text, []

    pristine = _pristine_reader_source(path)
    pristine_tree = _parsed_module(pristine, path)
    pristine_functions = _top_level_functions(pristine_tree)
    pristine_lines = pristine.splitlines(keepends=True)

    required = set(missing)
    queue = list(missing)
    while queue:
        name = queue.pop()
        node = pristine_functions.get(name)
        if node is None:
            continue
        dependencies = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        for dependency in dependencies:
            if (
                dependency in pristine_functions
                and dependency not in patched_functions
                and dependency not in required
            ):
                required.add(dependency)
                queue.append(dependency)

    unavailable = sorted(required - set(pristine_functions))
    if unavailable:
        raise RuntimeError(
            "callback function(s) are missing and unavailable in Git HEAD: "
            + ", ".join(unavailable)
        )

    restored: List[str] = []
    blocks: List[str] = []
    for name, node in pristine_functions.items():
        if name not in required:
            continue
        end_lineno = getattr(node, "end_lineno", None)
        if end_lineno is None:
            raise RuntimeError(f"cannot determine source range for function {name}")
        blocks.append("".join(pristine_lines[node.lineno - 1 : end_lineno]).rstrip() + "\n\n")
        restored.append(name)

    assignment = next(
        (
            node
            for node in patched_tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "sceneLoadTypeCallbacks"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
        ),
        None,
    )
    if assignment is None:
        raise RuntimeError("sceneLoadTypeCallbacks assignment disappeared during reader patching")
    lines = text.splitlines(keepends=True)
    insertion = ["# Unified 3DGS restored callback functions from repository HEAD.\n"] + blocks
    lines[assignment.lineno - 1 : assignment.lineno - 1] = insertion
    repaired = "".join(lines)

    repaired_tree = _parsed_module(repaired, path)
    unresolved = _callback_names(repaired_tree) - set(_top_level_functions(repaired_tree))
    if unresolved:
        raise RuntimeError(
            "reader callback repair left unresolved functions: " + ", ".join(sorted(unresolved))
        )
    return repaired, restored


def patch_file(path: Path) -> str:
    if not path.exists():
        return "missing"

    text = path.read_text(encoding="utf-8")
    statuses = []

    backup_path = path.with_suffix(path.suffix + ".bak_unified3dgs")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    if PATCH_MARKER in text:
        statuses.append("fetchPly_already_patched")
        patched = text
    else:
        match = FETCH_PLY_PATTERN.search(text)
        if not match:
            statuses.append("fetchPly_not_found")
            patched = text
        else:
            patched = text[: match.start()] + ROBUST_FETCH_PLY + text[match.end() :]
            statuses.append("fetchPly_patched")

    if NO_WRITE_MARKER in patched and "def fetchColmapPointCloud(" in patched:
        statuses.append("no_write_already_patched")
    elif "def fetchColmapPointCloud(" not in patched:
        insert_at = patched.find("def storePly(")
        if insert_at == -1:
            statuses.append("storePly_not_found")
        else:
            patched = patched[:insert_at] + FETCH_COLMAP_POINT_CLOUD + patched[insert_at:]
            statuses.append("no_write_helper_added")

    if NO_WRITE_MARKER in patched and "pcd = fetchColmapPointCloud" in patched:
        statuses.append("point_cloud_block_already_patched")
    else:
        patched, count = replace_colmap_point_cloud_block(path, patched)
        if not count:
            patched, count = POINT_CLOUD_BLOCK_PATTERN.subn(
                NO_WRITE_POINT_CLOUD_BLOCK, patched, count=1
            )
        statuses.append("point_cloud_block_patched" if count else "point_cloud_block_not_found")

    path_key = path.as_posix().lower()
    if any(name in path_key for name in ("3dgs-mcmc", "beta-splatting")):
        if MCMC_RANDOM_MARKER in patched:
            statuses.append("mcmc_random_already_patched")
        else:
            patched, count = MCMC_RANDOM_BLOCK_PATTERN.subn(MCMC_RANDOM_BLOCK, patched, count=1)
            statuses.append("mcmc_random_patched" if count else "mcmc_random_block_not_found")

    patched, restored = repair_missing_callback_functions(path, patched)
    if restored:
        statuses.append("callbacks_restored=" + "+".join(restored))
    else:
        statuses.append("callbacks_intact")

    if (
        "def readColmapSceneInfo" in patched
        and "points3D.ply" in patched
        and "pcd = fetchColmapPointCloud" not in patched
    ):
        raise RuntimeError(
            f"no-write COLMAP patch did not match {path}; refusing to leave a partial patch"
        )

    path.write_text(patched, encoding="utf-8")
    return f"{', '.join(statuses)} backup={backup_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch third-party dataset readers for robust PLY loading.")
    parser.add_argument(
        "--project-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Project root containing third_party/.",
    )
    parser.add_argument(
        "--method",
        choices=sorted(REPOS),
        default=None,
        help="Patch only one method. Default patches all known methods.",
    )
    args = parser.parse_args()

    methods = [args.method] if args.method else sorted(REPOS)
    failures = []
    for method in methods:
        reader = args.project_root / REPOS[method] / "scene" / "dataset_readers.py"
        try:
            status = patch_file(reader)
        except Exception as exc:
            failures.append((method, str(exc)))
            print(f"{method}: ERROR: {exc}: {reader}")
        else:
            print(f"{method}: {status}: {reader}")

    if failures:
        print("\nReader patch failures:")
        for method, error in failures:
            print(f"- {method}: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
