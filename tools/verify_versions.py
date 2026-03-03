#!/usr/bin/env python3
"""
Run inside Docker to compare package versions against local conda env.
Usage (in container):
    /isaac-sim/python.sh verify_versions.py
"""

# === LOCAL CONDA ENV SNAPSHOT (auto-generated) ===
# Update these when your local env changes.
LOCAL_VERSIONS = {
    "lerobot": "0.4.3",
    "torch": "2.7.1",          # will differ (CUDA vs CPU) — informational
    "numpy": "1.26.4",
    "gymnasium": "1.2.2",
    "huggingface-hub": "0.35.3",
    "transformers": None,      # not in local pip list — lerobot pulls it
    "accelerate": "1.12.0",
    "safetensors": "0.7.0",
    "pillow": "12.0.0",
    "einops": "0.8.1",
    "diffusers": "0.35.2",
    "draccus": "0.10.0",
}

# Packages where version mismatch is expected (different platform)
PLATFORM_DEPENDENT = {"torch", "torchvision", "nvidia-cuda-runtime-cu12"}

import importlib.metadata as meta
import sys


def get_version(pkg: str) -> str | None:
    try:
        return meta.version(pkg)
    except meta.PackageNotFoundError:
        return None


def main():
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    mismatches = []
    missing = []

    for pkg, local_ver in LOCAL_VERSIONS.items():
        docker_ver = get_version(pkg)
        if docker_ver is None:
            missing.append(pkg)
            status = "MISSING"
        elif local_ver is None:
            status = f"docker={docker_ver} (not tracked locally)"
        elif docker_ver == local_ver:
            status = f"{docker_ver} OK"
        elif pkg in PLATFORM_DEPENDENT:
            status = f"local={local_ver} docker={docker_ver} (platform-dependent, OK)"
        else:
            status = f"MISMATCH local={local_ver} docker={docker_ver}"
            mismatches.append((pkg, local_ver, docker_ver))

        print(f"  {pkg:25s} {status}")

    print()
    if mismatches:
        print("MISMATCHES FOUND (may cause different behavior):")
        for pkg, local_v, docker_v in mismatches:
            print(f"  {pkg}: local={local_v}  docker={docker_v}")
        print()
        print("To fix, pin versions in Dockerfile:")
        print('  /isaac-sim/python.sh -m pip install \\')
        for pkg, local_v, _ in mismatches:
            print(f'    "{pkg}=={local_v}" \\')
    else:
        print("No critical mismatches found.")

    if missing:
        print(f"\nMissing in Docker: {', '.join(missing)}")

    # Also check lerobot smolvla sub-package
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        print("\nSmolVLA policy class: importable OK")
    except ImportError as e:
        print(f"\nSmolVLA policy class: IMPORT FAILED — {e}")


if __name__ == "__main__":
    main()
