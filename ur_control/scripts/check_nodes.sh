#!/bin/bash
# 節點健康檢查 wrapper，避免在 Bash 工具指令字串裡直接寫指令替換/條件展開。
source /opt/ros/noetic/setup.bash 2>/dev/null
source /home/rvl/ros_ws/devel/setup.bash 2>/dev/null

for n in /gazebo /move_group /foundationpose_node $(rosnode list 2>/dev/null | grep -E "anygrasp_handover_node|semantic_brain_node"); do
  R=$(timeout 3 rosnode ping -c 1 "$n" 2>&1 | grep -c "xmlrpc reply")
  echo "$n : $([ "$R" -ge 1 ] && echo OK || echo DEAD)"
done
