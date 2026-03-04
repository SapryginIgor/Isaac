# URDF / Xacro (D405 camera)

- **d405.urdf.xacro** – Standalone xacro that expands to a full robot (world link + camera). Use this for conversion.
- **_materials.urdf.xacro**, **_usb_plug.urdf.xacro** – Stubs so conversion works without the full `realsense2_description` package.

## Xacro → URDF

From repo root. **With ROS Jazzy** (preferred when available):

```bash
source /opt/ros/jazzy/setup.bash
./scripts/xacro_to_urdf.sh urdf/d405.urdf.xacro
# → writes urdf/d405.urdf
```

Without ROS: `pip install xacrodoc` (e.g. in a venv), then run the same script.

## Mesh (d405.stl)

The generated URDF references `meshes/d405.stl` relative to the URDF file. Either:

1. Copy the mesh from the [RealSense ROS](https://github.com/IntelRealSense/realsense-ros) package:
   - `realsense2_description/meshes/d405.stl` (scale in URDF is 0.001, so the file is in mm).
2. Or leave `urdf/meshes/` empty; the camera will still have collision/inertial from the box; only the visual will be missing.

## URDF → USD

- **Isaac Sim**: File → Import → select the URDF, then export as USD if needed.
- **Isaac Lab**: `./isaaclab.sh -p scripts/urdf_to_usd.py --input urdf/d405.urdf [--output urdf/d405.usd]`

The lift scene spawns the D405 by default from `urdf/d405.usd`. Generate it once from the project root:

```bash
./isaaclab.sh -p scripts/urdf_to_usd.py --input urdf/d405.urdf --output urdf/d405.usd
```

Override the path with the `D405_USD_PATH` environment variable if needed.

## Camera intrinsics (Isaac)

Intrinsics from your D405 (`rs-enumerate-devices -c`) can be stored in the project file **`datasheet`** at the repo root. The module **`d405_camera_params.py`** reads those values (hardcoded for 1280×720 Depth and Color from that file) and exposes `get_d405_isaac_pinhole_kwargs(stream="depth"|"color", resolution=(w,h))` so you can emulate the D405’s RGB and depth cameras in Isaac (PinholeCameraCfg + clipping_range for depth).
