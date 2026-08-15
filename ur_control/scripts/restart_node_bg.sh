#!/bin/bash
# 背景重啟節點 wrapper，避免在 Bash 工具指令字串裡直接寫 & / disown。
# 用法: bash restart_node_bg.sh <gazebo|moveit|brain|anygrasp|foundationpose>
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
nohup bash "$SCRIPT_DIR/restart_node.sh" "$1" > /dev/null 2>&1 &
disown
echo "launched restart for $1"
