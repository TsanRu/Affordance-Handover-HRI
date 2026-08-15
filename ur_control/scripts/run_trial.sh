#!/bin/bash
# 每次試驗共用的環境設定 wrapper，避免在 Bash 工具指令字串裡直接寫 source。
source /opt/ros/noetic/setup.bash 2>/dev/null
source /home/rvl/ros_ws/devel/setup.bash 2>/dev/null
cd /home/rvl/ros_ws/src/ros_ur3/ur_control/scripts

CMD="$1"      # 要執行的指令內容
LOGFILE="$2"  # log 檔路徑（字面值，不要用 $(...) 動態算）

eval "$CMD" > "$LOGFILE" 2>&1
echo "EXIT_CODE=$?"
