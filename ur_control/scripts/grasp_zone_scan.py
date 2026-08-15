#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grasp_zone_scan.py

快速掃描右臂在桌面夾取高度的 2D 可達範圍。
掃描完成後印出視覺化格點地圖，並輸出建議的 WORKSPACE 邊界。

用法：
  python3 grasp_zone_scan.py
"""

import sys
import copy
import argparse
import rospy
import moveit_commander
import numpy as np
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

# ──────────────────────────────────────────
#  掃描參數（可由命令列覆蓋）
# ──────────────────────────────────────────
X_MIN, X_MAX = 0.50, 1.15
Y_MIN, Y_MAX = -0.35, 0.35
STEP         = 0.05

# 各物件預備接近高度 = TABLE_Z + 物件半高 + 安全餘量(0.15m)
# 由各物件的 TABLE_Z 和實際高度推算
OBJECT_PREGRASP_Z = {
    "hammer":          0.87,   # TABLE_Z=0.70, 扁平，頂部≈0.72, +0.15
    "cracker_box":     1.00,   # TABLE_Z=0.780, 側臥≈16cm高，頂部≈0.86, +0.15
    "sugar_box":       0.93,   # TABLE_Z=0.749, 頂部≈0.779, +0.15
    "spatula":         0.87,   # TABLE_Z=0.685, 頂部≈0.717, +0.15
    "spoon":           0.85,   # TABLE_Z=0.6853, 頂部≈0.706, +0.15
    "tomato_soup_can": 0.93,   # TABLE_Z=0.73, 直立≈10cm高，頂部≈0.78, +0.15
    "banana":          0.86,   # TABLE_Z=0.69, 細長扁平
    "bowl":            0.86,   # TABLE_Z=0.69, 淺碗
    "mug":             0.90,   # TABLE_Z=0.69, 高約11cm，頂部≈0.74, +0.15
    "scissors":        0.86,   # TABLE_Z=0.69, 扁平
}

# 夾爪垂直向下姿態（roll=0, pitch=π, yaw=0）
ORIENTATION_RPY = (0, 3.14159, 0)


def scan(move_group):
    xs = np.arange(X_MIN, X_MAX + STEP * 0.1, STEP).round(3)
    ys = np.arange(Y_MIN, Y_MAX + STEP * 0.1, STEP).round(3)

    quat = quaternion_from_euler(*ORIENTATION_RPY)
    grid = {}

    total = len(xs) * len(ys)
    done  = 0
    for x in xs:
        for y in ys:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = PREGRASP_Z
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            move_group.set_pose_target(pose)
            (plan, fraction) = move_group.compute_cartesian_path(
                [copy.deepcopy(pose)], 0.01, True)
            move_group.clear_pose_targets()

            ok = fraction > 0.95 and len(plan.joint_trajectory.points) > 0
            grid[(x, y)] = ok

            done += 1
            sym = "O" if ok else "."
            print(f"  ({x:.2f},{y:.2f}) {sym}  [{done}/{total}]", flush=True)

    return xs, ys, grid


def print_map(xs, ys, grid):
    print("\n===== 可達地圖（O=可達 .=不可達）=====")
    # 標頭：x 值
    header = "  y\\x  " + "  ".join(f"{x:.2f}" for x in xs)
    print(header)
    for y in reversed(ys):   # 從大到小印，視覺上 y+ 在上面
        row = f" {y:+.2f}  "
        for x in xs:
            row += " O  " if grid[(x, y)] else " .  "
        print(row)
    print()


def summarize(xs, ys, grid):
    reachable_x = [x for x in xs if any(grid[(x, y)] for y in ys)]
    reachable_y = [y for y in ys if any(grid[(x, y)] for x in xs)]

    if not reachable_x or not reachable_y:
        print("⚠️ 掃描範圍內沒有任何可達點，請調整掃描參數")
        return

    x_lo, x_hi = min(reachable_x), max(reachable_x)
    y_lo, y_hi = min(reachable_y), max(reachable_y)

    print("===== 建議 WORKSPACE 邊界（右臂夾取高度）=====")
    print(f"  WORKSPACE_X_MIN = {x_lo:.2f}")
    print(f"  WORKSPACE_X_MAX = {x_hi:.2f}")
    print(f"  WORKSPACE_Y_MIN = {y_lo:.2f}")
    print(f"  WORKSPACE_Y_MAX = {y_hi:.2f}")
    print()
    print("  （填入 random_object_placement.py 後，再減去各物件 half_extent 即為有效放置區間）")

    total_ok = sum(1 for v in grid.values() if v)
    print(f"\n  可達點數: {total_ok} / {len(grid)} "
          f"({100*total_ok/len(grid):.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="右臂夾取區域可達性掃描")
    parser.add_argument("--object", default=None,
                        help="物件名稱，自動選對應 PREGRASP_Z（如 hammer, cracker_box）")
    parser.add_argument("--z", type=float, default=None,
                        help="手動指定 PREGRASP_Z（覆蓋 --object 的預設值）")
    args = parser.parse_args()

    if args.z is not None:
        pregrasp_z = args.z
        label = f"z={pregrasp_z:.2f}（手動指定）"
    elif args.object is not None:
        pregrasp_z = OBJECT_PREGRASP_Z.get(args.object)
        if pregrasp_z is None:
            print(f"❌ 找不到 '{args.object}' 的 PREGRASP_Z，請用 --z 手動指定")
            sys.exit(1)
        label = f"{args.object}（z={pregrasp_z:.2f}）"
    else:
        pregrasp_z = 0.87
        label = f"預設 hammer（z={pregrasp_z:.2f}）"

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("grasp_zone_scan", anonymous=True)

    print(f"初始化 rightarm MoveGroup...")
    group = moveit_commander.MoveGroupCommander("rightarm")
    group.set_planning_time(1.0)
    group.set_num_planning_attempts(2)

    total = int((X_MAX - X_MIN) / STEP + 1) * int((Y_MAX - Y_MIN) / STEP + 1)
    print(f"掃描物件：{label}，共 {total} 格...")

    # 臨時覆蓋全域 PREGRASP_Z
    import builtins
    original_scan = scan
    def scan_with_z(mg):
        xs = np.arange(X_MIN, X_MAX + STEP * 0.1, STEP).round(3)
        ys = np.arange(Y_MIN, Y_MAX + STEP * 0.1, STEP).round(3)
        quat = quaternion_from_euler(*ORIENTATION_RPY)
        grid = {}
        total = len(xs) * len(ys)
        done = 0
        for x in xs:
            for y in ys:
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = pregrasp_z
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                mg.set_pose_target(pose)
                (plan, fraction) = mg.compute_cartesian_path(
                    [copy.deepcopy(pose)], 0.01, True)
                mg.clear_pose_targets()
                ok = fraction > 0.95 and len(plan.joint_trajectory.points) > 0
                grid[(x, y)] = ok
                done += 1
                print(f"  ({x:.2f},{y:.2f}) {'O' if ok else '.'}  [{done}/{total}]",
                      flush=True)
        return xs, ys, grid

    xs, ys, grid = scan_with_z(group)
    print_map(xs, ys, grid)
    summarize(xs, ys, grid)
    moveit_commander.roscpp_shutdown()
