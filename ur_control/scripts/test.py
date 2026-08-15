#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試左手三個展示初始姿態是否安全可達，
並驗證從任一初始姿態同步回待命位置的路徑規劃。

用法：
  python3 test.py
"""

import sys
import random
import rospy
import moveit_commander
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import actionlib


# 三個展示初始姿態（僅調整 joint1/joint2，其他與 home 相同）
DEMO_LEFT_INIT_POSES = {
    "A（高舉外展）": [2.2,  -0.5,  -2.2,  -1.0,  1.6476,  0.3],
    "B（低伸外旋）": [2.6,  -1.6,  -0.8,  -0.3,  1.3,    -0.0237],
    "C（內收前伸）": [0.7,  -0.8,  -2.0,  -0.8,  1.8,    -0.3],
}

LEFT_HOME_JOINTS    = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]
RIGHT_HOME_JOINTS   = [1.43,   -1.211, -2.0,     0.0,   1.6476, -0.0237]


class DemoInitPoseTest:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('demo_init_pose_test', anonymous=True)

        self.move_group       = moveit_commander.MoveGroupCommander("rightarm")
        self.left_move_group  = moveit_commander.MoveGroupCommander("leftarm")

        self.right_traj_client = actionlib.SimpleActionClient(
            '/rightarm/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.left_traj_client = actionlib.SimpleActionClient(
            '/leftarm/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)

        rospy.loginfo("等待 trajectory action servers...")
        self.right_traj_client.wait_for_server(timeout=rospy.Duration(5.0))
        self.left_traj_client.wait_for_server(timeout=rospy.Duration(5.0))
        rospy.loginfo("✅ 連線成功")
        rospy.sleep(1.0)

    def move_left_to_joints(self, joints, label=""):
        """左手移到指定關節角（joint-space，序列）"""
        self.left_move_group.set_joint_value_target(joints)
        plan = self.left_move_group.plan()
        self.left_move_group.clear_pose_targets()
        if not plan[0]:
            rospy.logwarn(f"⚠️ [{label}] 規劃失敗")
            return False
        self.left_move_group.execute(plan[1], wait=True)
        self.left_move_group.stop()
        rospy.loginfo(f"✅ [{label}] 到達")
        return True

    def sync_left_home_right_home(self, label=""):
        """同步：左手回待命 + 右手確認在 home（展示同步能力）"""
        rospy.loginfo(f"🔄 [{label}] 同步：左手回待命 + 右手回 home")

        self.left_move_group.set_joint_value_target(LEFT_HOME_JOINTS)
        plan_l = self.left_move_group.plan()
        self.left_move_group.clear_pose_targets()

        self.move_group.set_joint_value_target(RIGHT_HOME_JOINTS)
        plan_r = self.move_group.plan()
        self.move_group.clear_pose_targets()

        if plan_l[0] and plan_r[0]:
            goal_l = FollowJointTrajectoryGoal()
            goal_l.trajectory = plan_l[1].joint_trajectory
            goal_r = FollowJointTrajectoryGoal()
            goal_r.trajectory = plan_r[1].joint_trajectory

            self.left_traj_client.send_goal(goal_l)
            self.right_traj_client.send_goal(goal_r)
            self.left_traj_client.wait_for_result()
            self.right_traj_client.wait_for_result()
            self.left_move_group.stop()
            self.move_group.stop()
            rospy.loginfo(f"✅ [{label}] 同步完成")
            return True
        else:
            rospy.logwarn(f"⚠️ [{label}] 同步規劃失敗 (左={plan_l[0]}, 右={plan_r[0]})")
            return False

    def run(self):
        rospy.loginfo("========== 開始測試三個展示初始姿態 ==========")

        results = {}
        for label, joints in DEMO_LEFT_INIT_POSES.items():
            rospy.loginfo(f"\n---------- 測試姿態 {label} ----------")

            # Step 1：移到初始展示姿態
            ok = self.move_left_to_joints(joints, label)
            if not ok:
                results[label] = "❌ 無法到達"
                # 確保回 home 再試下一個
                self.move_left_to_joints(LEFT_HOME_JOINTS, "home 回復")
                continue

            rospy.sleep(1.5)  # 停留讓你觀察姿態

            # Step 2：同步回待命（展示同步能力）
            sync_ok = self.sync_left_home_right_home(label)
            results[label] = "✅ 成功" if sync_ok else "⚠️ 同步失敗（已序列回復）"

            if not sync_ok:
                self.move_left_to_joints(LEFT_HOME_JOINTS, "home 回復")

            rospy.sleep(1.0)

        rospy.loginfo("\n========== 測試結果摘要 ==========")
        for label, result in results.items():
            rospy.loginfo(f"  {label}：{result}")

        rospy.loginfo("\n========== 單次隨機姿態展示 ==========")
        label, joints = random.choice(list(DEMO_LEFT_INIT_POSES.items()))
        rospy.loginfo(f"隨機選中：{label}")
        self.move_left_to_joints(joints, label)
        rospy.sleep(1.5)
        self.sync_left_home_right_home(label)

        rospy.loginfo("🎉 所有測試完成")


if __name__ == "__main__":
    try:
        test = DemoInitPoseTest()
        test.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
