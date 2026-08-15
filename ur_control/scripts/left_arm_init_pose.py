#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
left_arm_init_pose.py

將左手臂移到三個展示初始姿態之一。
系統啟動後、執行 execute_mission() 前呼叫一次。

用法：
  python3 left_arm_init_pose.py          # 隨機選 A/B/C
  python3 left_arm_init_pose.py A        # 指定姿態
  python3 left_arm_init_pose.py B
  python3 left_arm_init_pose.py C
"""

import sys
import random

sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append('/home/rvl/ros_ws/devel/lib/python3/dist-packages')

import rospy
import moveit_commander

POSES = {
    "A": [2.2,  -0.5,  -2.2, -1.0,  1.6476,  0.3   ],
    "B": [2.6,  -1.6,  -0.8, -0.3,  1.3,    -0.0237],
    "C": [0.7,  -0.8,  -2.0, -0.8,  1.8,    -0.3   ],
}

if __name__ == "__main__":
    label = sys.argv[1].upper() if len(sys.argv) > 1 else random.choice(list(POSES))
    if label not in POSES:
        print(f"[error] unknown pose '{label}', choose A/B/C")
        sys.exit(1)

    joints = POSES[label]
    print(f"[left_arm_init] moving to pose {label}: {joints}")

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("left_arm_init_pose", anonymous=True)

    group = moveit_commander.MoveGroupCommander("leftarm")
    group.set_max_velocity_scaling_factor(0.3)
    group.set_joint_value_target(joints)
    plan = group.plan()

    if not plan[0]:
        print(f"[left_arm_init] planning failed for pose {label}")
        sys.exit(1)

    group.execute(plan[1], wait=True)
    group.stop()
    print(f"[left_arm_init] done, pose {label}")

    moveit_commander.roscpp_shutdown()
