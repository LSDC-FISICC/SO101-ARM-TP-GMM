#!/usr/bin/env bash
# Fetch the official SO-101 MuJoCo model (TheRobotStudio/SO-ARM100, Apache-2.0) into
# third_party/so101_mujoco: so101_new_calib.xml + the STL meshes it references + LICENSE.
# Pinned to the commit used to build and test the simulation.
set -euo pipefail
REV=${REV:-a758567c3978dfeefe282ede0500085a48fe8f78}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/so101_mujoco"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/TheRobotStudio/SO-ARM100.git "$TMP/so_arm"
git -C "$TMP/so_arm" sparse-checkout set --no-cone Simulation/SO101 LICENSE
git -C "$TMP/so_arm" checkout --quiet "$REV"
SRC="$TMP/so_arm/Simulation/SO101"
mkdir -p "$DEST/assets"
cp "$SRC/so101_new_calib.xml" "$DEST/"
cp "$TMP/so_arm/LICENSE" "$DEST/LICENSE"
for m in $(grep -o 'mesh file="[^"]*"' "$SRC/so101_new_calib.xml" | cut -d'"' -f2); do
  cp "$SRC/assets/$m" "$DEST/assets/"
done
echo "SO-101 MuJoCo model at $DEST (rev $REV)"
