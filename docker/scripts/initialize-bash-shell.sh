#!/bin/bash

################################################################################

alias sh='/bin/bash'

################################################################################

stty -ixon
umask 0002

################################################################################

source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash
[ -f /root/ws/install/setup.bash ] && source /root/ws/install/setup.bash

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

################################################################################

function ur-build-workspace () {
  pushd .
  source /root/ws/docker/scripts/repair-git-paths.sh
  source /root/ws/docker/scripts/fix-permission-issues.sh
  source /root/ws/docker/scripts/initialize-colcon-workspace.sh
  popd
}

function ur-initialize-workspace () {
  ur-build-workspace
}

function ur-fix-permission-issues () {
  pushd .
  source /root/ws/docker/scripts/repair-git-paths.sh
  source /root/ws/docker/scripts/fix-permission-issues.sh
  popd
}

function cc () {
  if [ -d ./src ]; then
    rm -rf build install log && echo "Removed build/ install/ log/ in $(pwd)"
  else
    echo "Not a colcon workspace (no ./src here). Aborting."
  fi
}

################################################################################

alias ws='cd /root/ws'
alias wss='cd /root/ws/src'
alias cb='colcon build --symlink-install'
alias s='source /root/ws/install/setup.bash'
alias upu='cd /root/ws/src/ur_python_utilities'

alias rd='rosdep install -i --from-paths'
alias rqt_plot='ros2 run rqt_plot rqt_plot'
alias rqt_reconfigure='ros2 run rqt_reconfigure rqt_reconfigure'
alias rqt_tf_tree='ros2 run rqt_tf_tree rqt_tf_tree'
alias plotjuggler='ros2 run plotjuggler plotjuggler'

export PROMPT_COMMAND="history -a"
export HISTCONTROL=ignoreboth

cd /root/ws
