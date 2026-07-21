from __future__ import annotations

import importlib.util
import sys


REQUIRED_IMPORTS = {
    "torch": "torch",
    "yaml": "PyYAML",
    "numpy": "numpy",
    "scipy": "scipy",
    "PIL": "pillow",
    "plyfile": "plyfile",
    "tqdm": "tqdm",
    "cv2": "opencv-python-headless",
    "matplotlib": "matplotlib",
    "imageio": "imageio",
    "skimage": "scikit-image",
    "lpips": "lpips",
    "mediapy": "mediapy==1.1.2",
    "open3d": "open3d-cpu==0.18.0",
    "trimesh": "trimesh==4.3.2",
}


def main() -> int:
    missing = []
    incompatible = []
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    if sys.version_info[:2] != (3, 8):
        incompatible.append(
            f"Python must be 3.8.x according to environment.yml, got {sys.version_info.major}.{sys.version_info.minor}"
        )
    for module_name, package_name in REQUIRED_IMPORTS.items():
        found = importlib.util.find_spec(module_name) is not None
        print(f"{module_name}: {'ok' if found else 'missing'}")
        if not found:
            missing.append(package_name)

    if importlib.util.find_spec("torch") is not None:
        import torch

        print("torch version:", torch.__version__)
        print("torch CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if not str(torch.__version__).startswith("2.0.0"):
            incompatible.append(f"PyTorch must be 2.0.0, got {torch.__version__}")
        if torch.version.cuda != "11.8":
            incompatible.append(f"PyTorch CUDA must be 11.8, got {torch.version.cuda}")
        if not torch.cuda.is_available():
            incompatible.append("PyTorch cannot access CUDA")

    if missing or incompatible:
        print()
        if missing:
            print("Missing packages:")
            for package_name in missing:
                print(f"- {package_name}")
            print()
            print("Install command:")
            print("python -m pip install -r requirements.txt")
        if incompatible:
            print("Runtime mismatches:")
            for error in incompatible:
                print(f"- {error}")
            print()
            print("Activate the framework environment: conda activate unified-3dgs")
        return 1

    print()
    print("Runtime dependency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
