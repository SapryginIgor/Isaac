"""
Intel RealSense D405 camera parameters for Isaac Sim/Lab.

Intrinsics are taken from rs-enumerate-devices output (see project file "datasheet").
They are converted to Isaac PinholeCameraCfg form: focal_length (mm), horizontal_aperture,
vertical_aperture, and clipping_range for depth.

Conversion: fx_px = focal_length_mm * (width_px / horizontal_aperture_mm)
=> focal_length_mm = fx_px * horizontal_aperture_mm / width_px.
We use horizontal_aperture = 20.955 mm (Isaac default) so focal_length matches the D405 FOV.
"""

from __future__ import annotations

# Nominal sensor width (mm) used to convert pixel intrinsics to physical camera params.
# Isaac's default 20.955 mm is used so FOV matches the datasheet.
_HORIZONTAL_APERTURE_MM = 20.955

# D405 depth range (meters), from Intel D400 series specs.
D405_DEPTH_NEAR_M = 0.07
D405_DEPTH_FAR_M = 0.50


def _to_isaac_intrinsics(
    width_px: int,
    height_px: int,
    fx_px: float,
    fy_px: float,
    horizontal_aperture_mm: float = _HORIZONTAL_APERTURE_MM,
) -> tuple[float, float, float]:
    """Convert pixel intrinsics (fx, fy, width, height) to Isaac physical params.

    Returns:
        (focal_length_mm, horizontal_aperture_mm, vertical_aperture_mm)
    """
    focal_length_mm = fx_px * horizontal_aperture_mm / width_px
    vertical_aperture_mm = height_px * focal_length_mm / fy_px
    return (focal_length_mm, horizontal_aperture_mm, vertical_aperture_mm)


# --- Depth stream: from "datasheet" Intrinsic of "Depth" / 1280x720 / {Z16} ---
# Width=1280, Height=720, Fx=650.419, Fy=650.419, PPX=640.4, PPY=358.53, FOV 89.07 x 57.93
D405_DEPTH_1280x720 = {
    "width": 1280,
    "height": 720,
    "fx_px": 650.419372558594,
    "fy_px": 650.419372558594,
    "ppx": 640.401245117188,
    "ppy": 358.532196044922,
    "fov_deg": (89.07, 57.93),
}

# --- Color stream: from "datasheet" Intrinsic of "Color" / 1280x720 ---
# Width=1280, Height=720, Fx=655.541, Fy=654.794, PPX=636.02, PPY=358.98, FOV 88.62 x 57.6
D405_COLOR_1280x720 = {
    "width": 1280,
    "height": 720,
    "fx_px": 655.540893554688,
    "fy_px": 654.794006347656,
    "ppx": 636.023254394531,
    "ppy": 358.980407714844,
    "fov_deg": (88.62, 57.6),
}


def get_d405_isaac_pinhole_kwargs(
    stream: str = "depth",
    resolution: tuple[int, int] | None = None,
    horizontal_aperture_mm: float = _HORIZONTAL_APERTURE_MM,
    depth_near: float = D405_DEPTH_NEAR_M,
    depth_far: float = D405_DEPTH_FAR_M,
) -> dict:
    """Return kwargs for Isaac PinholeCameraCfg that match the D405 from the datasheet.

    Args:
        stream: "depth" or "color"
        resolution: (width, height) or None for default 1280x720
        horizontal_aperture_mm: sensor width in mm (default 20.955)
        depth_near, depth_far: only used when stream=="depth" for clipping_range

    Returns:
        Dict with focal_length, horizontal_aperture, vertical_aperture (optional),
        clipping_range (for depth), and width/height for TiledCameraCfg.
    """
    if resolution is None:
        resolution = (1280, 720)
    width_px, height_px = resolution

    if stream == "depth":
        # Use depth intrinsics for 1280x720; for other res use same scale
        ref = D405_DEPTH_1280x720
        fx_px = ref["fx_px"] * (width_px / ref["width"])
        fy_px = ref["fy_px"] * (height_px / ref["height"])
    elif stream == "color":
        ref = D405_COLOR_1280x720
        fx_px = ref["fx_px"] * (width_px / ref["width"])
        fy_px = ref["fy_px"] * (height_px / ref["height"])
    else:
        raise ValueError(f"stream must be 'depth' or 'color', got {stream!r}")

    focal_mm, horiz_mm, vert_mm = _to_isaac_intrinsics(
        width_px, height_px, fx_px, fy_px, horizontal_aperture_mm
    )

    out = {
        "focal_length": focal_mm,
        "horizontal_aperture": horiz_mm,
        "vertical_aperture": vert_mm,
        "width": width_px,
        "height": height_px,
        "focus_distance": 0.4,
        "clipping_range": (0.1, 1.0e5),
    }
    if stream == "depth":
        out["clipping_range"] = (depth_near, depth_far)
    return out


def get_d405_depth_isaac_cfg(
    width: int = 1280,
    height: int = 720,
    **kwargs,
) -> dict:
    """Convenience: PinholeCameraCfg kwargs for D405 depth stream (for spawn=)."""
    return get_d405_isaac_pinhole_kwargs(
        stream="depth",
        resolution=(width, height),
        **kwargs,
    )


def get_d405_color_isaac_cfg(
    width: int = 1280,
    height: int = 720,
    **kwargs,
) -> dict:
    """Convenience: PinholeCameraCfg kwargs for D405 color stream (for spawn=)."""
    return get_d405_isaac_pinhole_kwargs(
        stream="color",
        resolution=(width, height),
        **kwargs,
    )
