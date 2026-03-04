#!/usr/bin/env python3
"""
Convert a URDF file to USD using Isaac Lab's UrdfConverter (when run inside Isaac Sim/Lab).
Usage:
  ./isaaclab.sh -p scripts/urdf_to_usd.py --input path/to/robot.urdf [--output path/to/robot.usd]

If Isaac Lab is not available, use Isaac Sim GUI: File → Import → select URDF and export as USD.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert URDF to USD via Isaac Lab")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input URDF path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output USD path (default: same name as URDF)")
    args = parser.parse_args()

    urdf_path = Path(args.input).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    usd_path = Path(args.output).resolve() if args.output else urdf_path.with_suffix(".usd")

    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    except ImportError:
        print("Isaac Lab not available in this environment.")
        print("Alternative: open Isaac Sim → File → Import → select your URDF, then export as USD.")
        return 1

    cfg = UrdfConverterCfg(asset_path=str(urdf_path))
    converter = UrdfConverter(cfg)
    converter.export_usd(str(usd_path))
    print(f"Exported: {usd_path}")
    return 0


if __name__ == "__main__":
    exit(main())
