#!/usr/bin/env bash
# Convert a .xacro file to .urdf (and optionally to .usd).
# Usage:
#   ./scripts/xacro_to_urdf.sh <input.xacro> [output.urdf]
#   ./scripts/xacro_to_urdf.sh urdf/d405.urdf.xacro
#   ./scripts/xacro_to_urdf.sh urdf/d405.urdf.xacro build/d405.urdf
#
# For URDF → USD: use Isaac Sim (File > Import > URDF) or run with --usd to export via Isaac Lab if available.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT="${1:?Usage: $0 <input.xacro> [output.urdf]}"
OUTPUT="${2:-}"

# Resolve input path
if [[ "$INPUT" != /* ]]; then
  INPUT="$REPO_ROOT/$INPUT"
fi
if [[ ! -f "$INPUT" ]]; then
  echo "Error: input not found: $INPUT" >&2
  exit 1
fi

# Default output: same dir and name as input, extension .urdf
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${INPUT%.xacro}"
  OUTPUT="${OUTPUT%.urdf}.urdf"
fi
if [[ "$OUTPUT" != /* ]]; then
  OUTPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$OUTPUT")"
fi

# Run xacro from the directory of the input so relative includes resolve
INPUT_DIR="$(dirname "$INPUT")"
INPUT_NAME="$(basename "$INPUT")"

echo "Converting $INPUT_NAME -> $(basename "$OUTPUT")"

# Prefer ROS 2 (e.g. Jazzy) when sourced, then xacro in PATH, then xacrodoc
if [[ -n "$ROS_DISTRO" ]]; then
  if [[ "$ROS_DISTRO" =~ ^(humble|iron|jazzy|rolling) ]]; then
    (cd "$INPUT_DIR" && ros2 run xacro xacro "$INPUT_NAME" -o "$OUTPUT")
  else
    (cd "$INPUT_DIR" && rosrun xacro xacro --inorder "$INPUT_NAME" > "$OUTPUT")
  fi
elif command -v xacro &>/dev/null; then
  (cd "$INPUT_DIR" && xacro "$INPUT_NAME" -o "$OUTPUT")
elif command -v xacrodoc &>/dev/null; then
  (cd "$INPUT_DIR" && xacrodoc "$INPUT_NAME" -o "$OUTPUT")
else
  echo "Error: no xacro tool found. With ROS Jazzy: source /opt/ros/jazzy/setup.bash then re-run." >&2
  echo "Other options: pip install xacrodoc, or install ros-<distro>-xacro" >&2
  exit 1
fi

echo "Wrote $OUTPUT"
echo "To convert URDF → USD: open Isaac Sim → File → Import → select this URDF (or use Isaac Lab UrdfConverter)."
