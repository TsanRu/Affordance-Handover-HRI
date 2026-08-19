#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SESSION="ur3_handeye_calibration"
ROBOT_IP="${ROBOT_IP:-192.168.86.7}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install it before running this script." >&2
  exit 1
fi

if [[ ! -f "$WS/devel/setup.bash" ]]; then
  echo "Missing $WS/devel/setup.bash. Run catkin_make in $WS first." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux attach-session -t "$SESSION"
  exit 0
fi

SETUP="source /opt/ros/noetic/setup.bash && source '$WS/devel/setup.bash'"

tmux new-session -d -s "$SESSION" -n calibration
P_CAM="$(tmux display-message -t "$SESSION:0.0" -p '#{pane_id}')"
P_ARM="$(tmux split-window -h -t "$P_CAM" -P -F '#{pane_id}')"
P_MOVEIT="$(tmux split-window -v -t "$P_CAM" -P -F '#{pane_id}')"
P_CAL="$(tmux split-window -v -t "$P_ARM" -P -F '#{pane_id}')"

tmux select-layout -t "$SESSION:0" tiled
tmux set-option -g -t "$SESSION" mouse on
tmux set-option -p -t "$P_CAM" @label "Camera"
tmux set-option -p -t "$P_ARM" @label "UR driver"
tmux set-option -p -t "$P_MOVEIT" @label "MoveIt"
tmux set-option -p -t "$P_CAL" @label "Hand-eye calibration"

tmux send-keys -t "$P_CAM" "cd '$WS' && $SETUP && roslaunch realsense2_camera rs_camera.launch align_depth:=true" Enter
tmux send-keys -t "$P_ARM" "cd '$WS' && $SETUP && roslaunch ur_robot_driver ur3_bringup.launch robot_ip:='$ROBOT_IP'" Enter
tmux send-keys -t "$P_MOVEIT" "cd '$WS' && $SETUP && sleep 5 && roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true" Enter
tmux send-keys -t "$P_CAL" "cd '$WS' && $SETUP && sleep 8 && roslaunch ur3_handover ur3_eye_to_hand_calibration.launch" Enter

tmux attach-session -t "$SESSION"

