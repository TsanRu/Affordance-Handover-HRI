#!/bin/bash
# 彙整實驗結果 wrapper，避免在 Bash 工具指令字串裡直接寫變數賦值+替換。
# 用法: bash compile_results.sh <object_name> <trials_log_dir>
cd /home/rvl/ros_ws/src/ros_ur3/ur_control/scripts
python3 affordance_experiments/compile_results.py "$1" "$2"
