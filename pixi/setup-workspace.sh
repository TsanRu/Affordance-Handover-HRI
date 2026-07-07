#!/usr/bin/env bash
# Assembles the colcon workspace for standalone use: symlinks this repo's
# packages into ws/src/, then vcs-imports third-party deps (dependencies.repos)
# such as cartesian_controllers. Mirrors what docker/scripts/initialize-colcon-workspace.sh
# does when this repo is bind-mounted as ws/src/ur_python_utilities.
set -euo pipefail

ROOT="$PIXI_PROJECT_ROOT"
SRC="$ROOT/ws/src"
mkdir -p "$SRC"

for pkg in ur_control ur_control_examples ur_gripper_gz ur_gripper_gz_moveit_config ur_pykdl; do
    if [ ! -e "$SRC/$pkg" ]; then
        ln -s "$ROOT/$pkg" "$SRC/$pkg"
    fi
done

if [ -f "$ROOT/dependencies.repos" ]; then
    vcs import "$SRC" < "$ROOT/dependencies.repos"
fi
