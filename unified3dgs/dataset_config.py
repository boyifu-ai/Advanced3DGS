from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets.json"
LOCAL_DATASET_PATHS = PROJECT_ROOT / "configs" / "local_dataset_paths.json"


@dataclass(frozen=True)
class DatasetDefinition:
    key: str
    title: str
    env_var: str
    default_root: Path
    scenes: Tuple[str, ...]
    acceptance_scene: str


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_definitions() -> Tuple[DatasetDefinition, ...]:
    data = _load_json(DATASET_CONFIG_PATH)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Dataset config must be a non-empty list: {DATASET_CONFIG_PATH}")
    definitions: List[DatasetDefinition] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Dataset config item must be an object: {item!r}")
        scenes = item.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"Dataset {item.get('key')!r} must declare scenes")
        definitions.append(
            DatasetDefinition(
                key=str(item["key"]),
                title=str(item["title"]),
                env_var=str(item["env_var"]),
                default_root=Path(str(item["default_root"])),
                scenes=tuple(str(scene) for scene in scenes),
                acceptance_scene=str(item.get("acceptance_scene") or scenes[0]),
            )
        )
    return tuple(definitions)


def load_local_dataset_paths() -> Dict[str, str]:
    if not LOCAL_DATASET_PATHS.is_file():
        return {}
    data = _load_json(LOCAL_DATASET_PATHS)
    if not isinstance(data, dict):
        raise ValueError(f"Local dataset path config must be an object: {LOCAL_DATASET_PATHS}")
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def save_local_dataset_path(key: str, root: Path) -> None:
    paths = load_local_dataset_paths()
    paths[key] = str(root.expanduser())
    LOCAL_DATASET_PATHS.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_DATASET_PATHS.write_text(
        json.dumps(paths, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clear_local_dataset_path(key: str) -> None:
    paths = load_local_dataset_paths()
    paths.pop(key, None)
    if paths:
        LOCAL_DATASET_PATHS.write_text(
            json.dumps(paths, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif LOCAL_DATASET_PATHS.exists():
        LOCAL_DATASET_PATHS.unlink()


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def dataset_root(definition: DatasetDefinition) -> Path:
    local_paths = load_local_dataset_paths()
    if definition.key in local_paths:
        return _resolve_path(Path(local_paths[definition.key]))
    env_value = os.environ.get(definition.env_var, "").strip()
    if env_value:
        return _resolve_path(Path(env_value))
    return _resolve_path(definition.default_root)


def dataset_roots() -> Dict[str, Path]:
    return {
        definition.key: dataset_root(definition)
        for definition in load_dataset_definitions()
    }


def readonly_dataset_roots() -> Tuple[Path, ...]:
    explicit = os.environ.get("UNIFIED3DGS_READONLY_DATASET_ROOT", "").strip()
    if explicit:
        return tuple(
            _resolve_path(Path(part))
            for part in explicit.split(os.pathsep)
            if part.strip()
        )
    return tuple(dataset_roots().values())


def readonly_dataset_root_env() -> str:
    return os.pathsep.join(str(path) for path in readonly_dataset_roots())


def validation_scene_records(
    families: Optional[Iterable[str]] = None,
    scenes: Optional[Iterable[str]] = None,
) -> List[Tuple[Path, str]]:
    selected_families = {family.strip() for family in families or () if family.strip()}
    selected_scenes = {scene.strip() for scene in scenes or () if scene.strip()}
    records: List[Tuple[Path, str]] = []
    for definition in load_dataset_definitions():
        if selected_families and definition.key not in selected_families:
            continue
        root = dataset_root(definition)
        for scene in definition.scenes:
            label = f"{definition.key}/{scene}"
            if selected_scenes and label not in selected_scenes:
                continue
            records.append((root / scene, label))
    return records


def acceptance_dataset() -> Path:
    definitions = load_dataset_definitions()
    if not definitions:
        raise ValueError("No dataset definitions are configured")
    definition = definitions[0]
    return dataset_root(definition) / definition.acceptance_scene


def representative_datasets() -> Tuple[Tuple[str, Path], ...]:
    records: List[Tuple[str, Path]] = []
    for definition in load_dataset_definitions():
        label = f"{definition.key}/{definition.acceptance_scene}"
        records.append((label, dataset_root(definition) / definition.acceptance_scene))
    return tuple(records)
