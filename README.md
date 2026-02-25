# Pure Isaac SO-101 + SmolVLA inference

Isaac Lab SO-101 pick-cube (or reach) environment with **SmolVLA** as an inference-only policy. No LeRobot/LeIsaac environment stack; simulation is pure Isaac Lab. End-effector position and deltas are exposed via a thin wrapper.

**SO-101** is provided by the **isaac_so_arm101** extension included in this repo (robot URDF, reach and lift-cube tasks). The run script automatically registers its envs when you run from the Isaac directory.

## Prerequisites

1. **Isaac Sim** and **Isaac Lab** installed (see [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).
2. **This repo** (including the `isaac_so_arm101/` extension):
   - Clone this repo. The extension lives under `isaac_so_arm101/` (SO-100/SO-101 URDFs, tasks, scripts).
   - **Optional**: Install the extension so it’s on `PYTHONPATH` when using Isaac Lab:
     ```bash
     cd /path/to/Isaac
     pip install -e isaac_so_arm101
     ```
     If you don’t install it, the run script adds `isaac_so_arm101/src` to `sys.path` so the envs are still registered.

## Install extra dependencies

From the **Isaac Lab** environment (the one you use with `./isaaclab.sh`), install LeRobot with SmolVLA and this project’s extras:

```bash
pip install -r /path/to/Isaac/requirements.txt
```

Or:

```bash
pip install "lerobot[smolvla]" gymnasium numpy torch huggingface_hub
```

## How to run

Scripts must be run via Isaac Lab’s Python so that `isaaclab` and the SO-101 envs are available. From your **Isaac Lab** repo root:

```bash
./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0
```

**Task IDs** (from the in-repo isaac_so_arm101 extension):

| Task | Description |
|------|-------------|
| `Isaac-SO-ARM101-Lift-Cube-v0` | SO-101 pick cube and lift to target (default) |
| `Isaac-SO-ARM101-Lift-Cube-Play-v0` | Same, smaller scene for play |
| `Isaac-SO-ARM101-Reach-v0` | SO-101 reach target pose |
| `Isaac-SO-ARM101-Reach-Play-v0` | Same, play variant |
| `Isaac-SO-ARM100-Lift-Cube-v0`, `Isaac-SO-ARM100-Reach-v0` | SO-100 (same tasks) |

Options:

- `--task`: Gym task id (default: `Isaac-SO-ARM101-Lift-Cube-v0`).
- `--headless`: Run without GUI.
- `--num_envs`: Number of parallel envs (default: 1).
- `--policy`: HuggingFace policy repo or local path (default: `lerobot/smolvla_base`).
- `--instruction`: Language instruction for the policy (default: `"Pick the cube."`).
- `--max_steps`, `--episodes`: Episode length and number of episodes.
- `--robot_name`: Scene articulation name (default: `robot`, as in isaac_so_arm101).
- `--ee_link_name`: End-effector link for EE pose (default: `gripper_link` for SO-101).
- `--no_ee_in_obs`: Do not add `ee_pos` / `ee_quat` / `ee_pos_delta` into the observation dict (EE still available via `get_ee_state()` and in `info["ee_state"]`).

Example (headless, 2 episodes, reach task):

```bash
./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py --task Isaac-SO-ARM101-Reach-v0 --headless --episodes 2 --instruction "Reach the target."
```

## End-effector position and deltas

- **In the wrapper**: After each `step()` and `reset()`, the wrapper reads the SO-101 articulation’s end-effector link pose from the Isaac Lab scene and computes position deltas from the previous step.
- **API**:
  - **`get_ee_state()`**: Call on the wrapped env to get `ee_pos`, `ee_quat`, `ee_pos_delta`, `ee_quat_delta` (each numpy arrays).
  - **`info["ee_state"]`**: The same dict is written into `info` at every `step()` and `reset()`.
- **Observations**: By default the wrapper adds `ee_pos`, `ee_quat`, and `ee_pos_delta` to the observation dict so the policy (and your code) can use them. Disable with `--no_ee_in_obs` if you only need them from `get_ee_state()` / `info`.

Usage in your code:

```python
import isaac_so_arm101.tasks.reach  # noqa: F401
import isaac_so_arm101.tasks.lift   # noqa: F401
import gymnasium as gym
from env_wrapper import IsaacEEWrapper

env = gym.make("Isaac-SO-ARM101-Lift-Cube-v0", num_envs=1)
env = IsaacEEWrapper(env, robot_name="robot", ee_link_name="gripper_link", add_ee_to_obs=True)
obs, info = env.reset()
# EE state in info
ee = info["ee_state"]  # ee["ee_pos"], ee["ee_quat"], ee["ee_pos_delta"], ee["ee_quat_delta"]
# or on the wrapper
ee = env.get_ee_state()
```

## Alternative checkpoints

- Default: **`lerobot/smolvla_base`** (multi-task, SO-100/SO-101–friendly action space).
- You can pass another HuggingFace repo, e.g. **`kenmacken/smolvla_policy`**, via `--policy`. SO-101 aligns well with SmolVLA’s typical SO-100/SO-101 action space; if the checkpoint uses a different robot, you may need to adjust scaling in `adapters.policy_action_to_env()` or add a small mapping in this repo.

## Project layout

| File / folder | Purpose |
|---------------|--------|
| `run_smolvla_isaac.py` | Entrypoint: registers isaac_so_arm101 tasks, creates SO-101 env, wraps with EE wrapper, loads SmolVLA, runs inference loop. |
| `env_wrapper.py` | Gymnasium wrapper: adds EE pose and deltas, exposes `get_ee_state()`, optionally adds EE to obs. |
| `adapters.py` | Isaac obs → policy frame; policy action → env action (scale/clip). |
| `requirements.txt` | Extra deps (lerobot[smolvla], gymnasium, torch, etc.). |
| `isaac_so_arm101/` | **In-repo extension**: SO-100/SO-101 URDFs (`robots/trs_so101/urdf/so_arm101.urdf`, `trs_so100/urdf/so_arm100.urdf`), reach and lift-cube tasks, train/play scripts. Reused as-is; run script adds `isaac_so_arm101/src` to the path to register envs. |
| `README.md` | This file. |

Isaac Sim and Isaac Lab are **not** listed in `requirements.txt`; they are installed and run separately. The SO-101 robot and tasks come from the bundled **isaac_so_arm101** extension (original repo: [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)).
