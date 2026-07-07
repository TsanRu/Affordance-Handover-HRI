#!/usr/bin/env bash
# Overlay the colcon workspace on top of the pixi/RoboStack ROS 2 install, if built.
if [ -f "$PIXI_PROJECT_ROOT/ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$PIXI_PROJECT_ROOT/ws/install/setup.bash"
fi
