# NumPy + Pip Fix for Isaac Sim Container

## The Error

Running `save_env_cameras.py` (or any script using tiled cameras / synthetic
data) crashed with:

```
File ".../omni/syntheticdata/scripts/SyntheticData.py", line 1070
    dep_attrib_data.set(dep_data)
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

## Root Cause

The base image (`nvcr.io/nvidia/isaac-lab:2.3.2`) ships **numpy 1.26.0**.
Omniverse C extensions (`omni.syntheticdata`, `omni.replicator.core`) are
compiled against the **numpy 1.x ABI**. NumPy 2.0 changed dtype internals
and breaks these extensions.

Installing `lerobot[smolvla]==0.4.3` normally triggers two cascading problems:

1. **rerun-sdk>=0.24** (lerobot dep) requires **numpy>=2** — every version
   in the range (0.24.0–0.26.2) hard-requires it. This upgrades numpy to 2.x,
   breaking Omniverse.

2. **packaging>=24.2** (lerobot dep) replaces the base image's packaging 23.0.
   This corrupts **pip 24.3.1's vendored** `pip._vendor.packaging._structures`
   module, making pip itself unusable for any follow-up fix.

Reference: https://github.com/isaac-sim/IsaacLab/issues/2235

## The Working Fix (current Dockerfile)

**Skip the problematic deps entirely.** Install lerobot with `--no-deps`, then
install its actual dependencies manually — omitting rerun-sdk (visualization
only, not used in our Isaac Sim pipeline).

```dockerfile
# Install lerobot package without resolving its dependency tree
/isaac-sim/python.sh -m pip install --no-deps "lerobot[smolvla]==0.4.3"

# Install lerobot's deps manually, minus rerun-sdk
/isaac-sim/python.sh -m pip install \
    "datasets>=4.0.0,<4.2.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
    "accelerate>=1.10.0,<2.0.0" \
    "cmake>=3.29.0.1,<4.2.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "pynput>=1.7.7,<1.9.0" \
    "pyserial>=3.5,<4.0" \
    "wandb>=0.24.0,<0.25.0" \
    "torchcodec>=0.2.1,<0.6.0" \
    "draccus==0.10.0" \
    "gymnasium>=1.1.1,<2.0.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "imageio[ffmpeg]>=2.34.0,<3.0.0" \
    "num2words>=0.5.14,<0.6.0" \
    "transformers>=4.57.1,<5.0.0"
```

This way:
- **numpy stays at 1.x** — rerun-sdk is never installed, so nothing pulls numpy>=2
- **packaging stays at 23.0** — lerobot's packaging>=24.2 dep is skipped, pip stays healthy
- **No post-install repair needed** — no get-pip.py, no force-reinstall, no constraints files

## What Doesn't Work at Runtime (and Why That's OK)

- **rerun-sdk**: not installed. It's only used for lerobot's visualization/logging
  features (`rerun` CLI). Our Isaac Sim pipeline doesn't use it.

## Dep List Maintenance

The dependency list in the Dockerfile comes from lerobot 0.4.3's `pyproject.toml`:
https://github.com/huggingface/lerobot/blob/v0.4.3/pyproject.toml

When upgrading lerobot, check the new `pyproject.toml` and update the dep list.
Always omit `rerun-sdk` (and any other dep that requires numpy>=2).

## Approaches That Were Tried and Failed

| Approach | Why it failed |
|---|---|
| `PIP_CONSTRAINT` with `numpy<2` during install | rerun-sdk requires numpy>=2 — `ResolutionImpossible` |
| Pin all dep versions in constraints.txt | Eliminated backtracking but opencv-python-headless 4.12+ also needs numpy>=2 |
| `PIP_CONSTRAINT` with `packaging<24` | lerobot 0.4.3 requires packaging>=24.2 — `ResolutionImpossible` |
| Two constraints files (build + runtime) | Same packaging conflict |
| Install freely, then `pip install numpy==1.26.4` | pip is broken by packaging 25.0 — can't run pip |
| `python -m ensurepip --upgrade` | Says "pip already satisfied" (checks version, not integrity) |
| `get-pip.py --force-reinstall` | Fails uninstalling corrupted pip (ENOENT on `_structures.py`) |
| Nuke pip + get-pip.py + downgrade numpy | Works, but unnecessarily complex — skipping rerun-sdk is simpler |

## References

- https://github.com/isaac-sim/IsaacLab/issues/2235 (numpy<2 requirement)
- https://github.com/pypa/pip/issues/6261 (pip vendored module corruption)
- https://github.com/huggingface/lerobot/blob/v0.4.3/pyproject.toml (dep list source)
