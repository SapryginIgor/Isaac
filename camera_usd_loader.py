#!/usr/bin/env python3
"""
Load camera_side and camera_up config from a USD file at runtime.
Expects cameras nested under prims named "CameraSideXform" and "CameraUpXform".
Used to override env scene cameras when running with --camera_usd /path/to/scene.usd.
"""

from __future__ import annotations

from pathlib import Path


def _find_camera_under_prim(stage, prim):
    """Return first UsdGeom.Camera under prim (prim or any descendant), or None."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Camera):
        return prim
    for child in prim.GetChildren():
        found = _find_camera_under_prim(stage, child)
        if found is not None:
            return found
    return None


def _get_world_pose(prim):
    """Return (position_xyz, quat_wxyz) in world frame."""
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    world = xform.ComputeLocalToWorldTransform(0)
    pos = world.ExtractTranslation()
    rot = world.ExtractRotation().GetQuat()
    # Gf.Quat: GetReal() = w, GetImaginary() = (x,y,z)
    quat_wxyz = (
        rot.GetReal(),
        rot.GetImaginary()[0],
        rot.GetImaginary()[1],
        rot.GetImaginary()[2],
    )
    return (pos[0], pos[1], pos[2]), quat_wxyz


def _get_camera_intrinsics(prim):
    """Return dict with focal_length, horizontal_aperture, vertical_aperture, clipping_range."""
    from pxr import UsdGeom

    cam = UsdGeom.Camera(prim)
    focal = cam.GetFocalLengthAttr()
    ha = cam.GetHorizontalApertureAttr()
    va = cam.GetVerticalApertureAttr()
    clip = cam.GetClippingRangeAttr()
    return {
        "focal_length": focal.Get() if focal else 24.0,
        "horizontal_aperture": ha.Get() if ha else 20.955,
        "vertical_aperture": va.Get() if va else 15.716,  # 20.955 * 3/4
        "clipping_range": tuple(clip.Get()) if clip else (0.1, 1.0e5),
    }


def load_camera_config_from_usd(
    usd_path: str | Path,
    side_xform_name: str = "CameraSideXform",
    up_xform_name: str = "CameraUpXform",
    width: int = 256,
    height: int = 256,
):
    """
    Open the USD and find cameras under prims named side_xform_name and up_xform_name.
    Returns (camera_side_cfg, camera_up_cfg) as TiledCameraCfg instances, or (None, None) if not found.
    """
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import TiledCameraCfg

    usd_path = Path(usd_path).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"Camera USD not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    def find_camera_by_xform_name(name):
        for prim in stage.Traverse():
            if prim.GetName() == name:
                cam = _find_camera_under_prim(stage, prim)
                if cam is not None:
                    return cam
        return None

    side_cam = find_camera_by_xform_name(side_xform_name)
    up_cam = find_camera_by_xform_name(up_xform_name)

    def make_cfg(cam_prim, env_prim_path_key):
        if cam_prim is None:
            return None
        pos, quat_wxyz = _get_world_pose(cam_prim)
        intrinsics = _get_camera_intrinsics(cam_prim)
        return TiledCameraCfg(
            prim_path=env_prim_path_key,
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos,
                rot=quat_wxyz,
                convention="world",
            ),
            data_types=["rgb"],
            width=width,
            height=height,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=intrinsics["focal_length"],
                focus_distance=400.0,
                horizontal_aperture=intrinsics["horizontal_aperture"],
                clipping_range=intrinsics["clipping_range"],
            ),
        )

    camera_side_cfg = make_cfg(side_cam, "{ENV_REGEX_NS}/CameraSide")
    camera_up_cfg = make_cfg(up_cam, "{ENV_REGEX_NS}/CameraUp")
    return camera_side_cfg, camera_up_cfg


def apply_camera_usd_to_env_cfg(env_cfg, usd_path: str | Path, **kwargs):
    """
    If env_cfg has scene.camera_side and scene.camera_up, override them from the given USD.
    kwargs are passed to load_camera_config_from_usd (e.g. width, height).
    """
    camera_side_cfg, camera_up_cfg = load_camera_config_from_usd(usd_path, **kwargs)
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    if camera_side_cfg is not None and hasattr(scene, "camera_side"):
        scene.camera_side = camera_side_cfg
    if camera_up_cfg is not None and hasattr(scene, "camera_up"):
        scene.camera_up = camera_up_cfg
