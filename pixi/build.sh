#!/usr/bin/env bash
set -euo pipefail

"$PIXI_PROJECT_ROOT/pixi/setup-workspace.sh"

cd "$PIXI_PROJECT_ROOT/ws"
exec colcon build --symlink-install \
    --packages-skip cartesian_controller_simulation cartesian_controller_tests \
    "$@"
