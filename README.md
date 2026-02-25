# Pure Isaac SO-101 + SmolVLA inference

Isaac Lab SO-101 pick-cube (or reach) environment with **SmolVLA** as an inference-only policy. No LeRobot/LeIsaac environment stack; simulation is pure Isaac Lab. End-effector position and deltas are exposed via a thin wrapper.

## Prerequisites

1. **Isaac Sim** and **Isaac Lab** installed (see [Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)).
2. **SO-101 in Isaac Lab**: SO-101 is not in upstream Isaac Lab. Use one of:
   - **[isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)** (recommended): SO-ARM100/101 Reach env. Clone the repo, then from the Isaac Lab root:
     ```bash
     cd /path/to/isaac_so_arm101
     python -m pip install -e source/SO_100
     ```
     List envs: `python scripts/list_envs.py` (e.g. `SO-ARM100-Reach-v0`).
   - **[so101_isaac](https://github.com/KyleM73/so101_isaac)** or your own SO-101 task registered with Isaac Lab.
3. **This project**: Clone or copy this folder (containing `run_smolvla_isaac.py`, `env_wrapper.py`, `adapters.py`, `requirements.txt`) so it is available from the machine where you run Isaac Lab.

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

Scripts must be run via Isaac Lab’s Python so that `isaaclab` and the SO-101 task are available. From your **Isaac Lab** repo root:

```bash
./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py --task SO-ARM100-Reach-v0
```

Options:

- `--task`: Gym task id (default: `SO-ARM100-Reach-v0`). Use the task id from the SO-101 project you installed (e.g. after adding a lift-cube task: `SO-101-Lift-Cube-v0`).
- `--headless`: Run without GUI.
- `--num_envs`: Number of parallel envs (default: 1).
- `--policy`: HuggingFace policy repo or local path (default: `lerobot/smolvla_base`).
- `--instruction`: Language instruction for the policy (default: `"Pick the cube."`).
- `--max_steps`, `--episodes`: Episode length and number of episodes.
- `--robot_name`, `--ee_link_name`: If your scene uses a specific articulation/link name, set these for correct EE reading in the wrapper.
- `--no_ee_in_obs`: Do not add `ee_pos` / `ee_quat` / `ee_pos_delta` into the observation dict (EE still available via `get_ee_state()` and in `info["ee_state"]`).

Example (headless, 2 episodes):

```bash
./isaaclab.sh -p /path/to/Isaac/run_smolvla_isaac.py --task SO-ARM100-Reach-v0 --headless --episodes 2 --instruction "Reach the target."
```

## End-effector position and deltas

- **In the wrapper**: After each `step()` and `reset()`, the wrapper reads the SO-101 articulation’s end-effector link pose from the Isaac Lab scene and computes position deltas from the previous step.
- **API**:
  - **`get_ee_state()`**: Call on the wrapped env to get `ee_pos`, `ee_quat`, `ee_pos_delta`, `ee_quat_delta` (each numpy arrays).
  - **`info["ee_state"]`**: The same dict is written into `info` at every `step()` and `reset()`.
- **Observations**: By default the wrapper adds `ee_pos`, `ee_quat`, and `ee_pos_delta` to the observation dict so the policy (and your code) can use them. Disable with `--no_ee_in_obs` if you only need them from `get_ee_state()` / `info`.

Usage in your code:

```python
from env_wrapper import IsaacEEWrapper
env = gym.make("SO-ARM100-Reach-v0", num_envs=1)
env = IsaacEEWrapper(env, add_ee_to_obs=True)
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
| `run_smolvla_isaac.py` | Entrypoint: creates SO-101 env, wraps with EE wrapper, loads SmolVLA, runs inference loop. |
| `env_wrapper.py` | Gymnasium wrapper: adds EE pose and deltas, exposes `get_ee_state()`, optionally adds EE to obs. |
| `adapters.py` | Isaac obs → policy frame; policy action → env action (scale/clip). |
| `requirements.txt` | Extra deps (lerobot[smolvla], gymnasium, torch, etc.). |
| `README.md` | This file. |

Isaac Sim and Isaac Lab are **not** listed in `requirements.txt`; they are installed and run separately. This project only adds the wrapper, adapters, and run script around an existing Isaac Lab + SO-101 setup.
