#!/bin/bash
# 重啟指定的模擬/偵測節點，統一環境設定（source/conda activate），
# 避免每次都在 Bash 指令字串裡直接寫這些，也讓呼叫方式固定成單一指令好放行。
# 用法: bash restart_node.sh <gazebo|moveit|brain|anygrasp|foundationpose>

SERVICE="$1"
LOGDIR=/tmp/claude-1000/-home-rvl-ros-ws-src-ros-ur3-ur-control-scripts/645d110c-a450-4263-bac0-2e11f1590554/scratchpad

source /opt/ros/noetic/setup.bash 2>/dev/null
source /home/rvl/ros_ws/devel/setup.bash 2>/dev/null

case "$SERVICE" in
  gazebo)
    roslaunch ur_gripper_gazebo ur_gripper_85_dual_arm.launch > "$LOGDIR/restart_gazebo.log" 2>&1
    ;;
  moveit)
    roslaunch ur_gripper_85_moveit_config start_sim_dual_ur3e_moveit.launch > "$LOGDIR/restart_moveit.log" 2>&1
    ;;
  brain)
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate lang-sam
    cd ~/ros_ws/src/anygrasp_sdk/grasp_detection
    python3 brain.py > "$LOGDIR/restart_brain.log" 2>&1
    ;;
  anygrasp)
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate anygrasp_ros
    cd ~/ros_ws/src/anygrasp_sdk/grasp_detection
    python3 anygrasp_ros.py > "$LOGDIR/restart_anygrasp.log" 2>&1
    ;;
  foundationpose)
    docker exec -d foundationpose bash -c "cd /home/rvl/ros_ws/src/FoundationPose && /opt/conda/envs/my/bin/python3 foundationpose_node.py > /tmp/foundationpose_node.log 2>&1"
    ;;
  *)
    echo "Usage: bash restart_node.sh <gazebo|moveit|brain|anygrasp|foundationpose>"
    exit 1
    ;;
esac
echo "EXIT_CODE=$?"
