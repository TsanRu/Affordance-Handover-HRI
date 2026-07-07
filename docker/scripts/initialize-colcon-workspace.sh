#!/bin/bash
# Import third-party sources, resolve deps, and colcon-build /root/ws.

export DEBIAN_FRONTEND=noninteractive

apt-get update || sudo apt-get update || true
rosdep update --rosdistro "${ROS_DISTRO}"
source /opt/ros/"${ROS_DISTRO}"/setup.bash

cd /root/ws/
git config --global --add safe.directory /root/ws
git config --global --add safe.directory /root/ws/src/ur_python_utilities

REPOS_FILE=src/ur_python_utilities/dependencies.repos
if [ -f "${REPOS_FILE}" ]; then
  vcs import src < "${REPOS_FILE}" --force
  find src -name ".git" -type d 2>/dev/null | while read -r gitdir; do
    repo_path=$(dirname "$gitdir")
    git config --global --add safe.directory "/root/ws/${repo_path#./}"
  done
fi

rosdep install --from-paths src --ignore-src -r -y || true

if [ -n "$(find src -name package.xml 2>/dev/null | head -n1)" ]; then
  colcon build --symlink-install \
    --packages-skip cartesian_controller_simulation cartesian_controller_tests \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
fi

source /root/ws/docker/scripts/fix-permission-issues.sh
