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
