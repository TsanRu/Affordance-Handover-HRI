#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped

def main():
    """
    使用 P2P 運動，規劃並移動到一個「預抓取」姿態。
    【已修改】：此姿態使用 AnyGrasp 的 XYZ，但強制使用一個「垂直朝下」的旋轉。
    """
    # 1. --- 初始化 MoveIt! Commander 和 ROS 節點 ---
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('p2p_move_to_vertical_pre_grasp_node', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "rightarm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    planning_frame = move_group.get_planning_frame()
    rospy.loginfo(f"Move Group '{group_name}' 初始化完畢。 規劃座標系: {planning_frame}")
    rospy.sleep(2.0)


    # 2. --- 定義並加入碰撞物體 ---
    rospy.loginfo("正在將 Gazebo 物體加入 MoveIt! 規劃場景...")
    
    # (從您的腳本中複製，假設這些值是正確的)
    # scene.remove_world_object("table")
    # scene.remove_world_object("coke_can")
    # rospy.sleep(0.5)

    # --- A. 加入桌子 ---
    # table_height = 0.775
    # table_size = [2.0, 2.0, table_height]
    # table_pose = PoseStamped()
    # table_pose.header.frame_id = planning_frame
    # table_pose.pose.orientation.w = 1.0
    # table_pose.pose.position.x = 1.3 
    # table_pose.pose.position.y = 0.0
    # table_pose.pose.position.z = table_height / 2.0
    # scene.add_box("table", table_pose, size=table_size)
    
    # --- B. 加入可樂罐 ---
    # object_height = 0.12 
    # object_radius = 0.03
    # object_pose = PoseStamped()
    # object_pose.header.frame_id = planning_frame
    # object_pose.pose.orientation.w = 1.0
    # object_pose.pose.position.x = 1.0
    # object_pose.pose.position.y = 0.15
    # object_pose.pose.position.z = table_height + (object_height / 2.0) 
    # scene.add_cylinder("coke_can", object_pose, height=object_height, radius=object_radius)

    rospy.loginfo("桌子和可樂罐已加入場景。等待 MoveIt! 更新...")
    rospy.sleep(1.0)


    # 3. --- ===【⭐️ 關鍵修改：定義「預抓取」姿態】=== ---
    
    # 這是 AnyGrasp + TF 轉換後的位置 (我們只保留這個)
    grasp_x = 0.9195
    grasp_y = 0.2271
    grasp_z = 0.7024# 這是物體上的目標點
    
    grasp_ox = -0.8033
    grasp_oy = 0.5901
    grasp_oz = 0.0802
    grasp_ow = -0.0066

    # 定義一個「預抓取」的偏移量（在物體上方 10 公分）
    pre_grasp_offset = 0.3 # 10 cm (0.1 anygrasp夾爪誤差, 0.1 gazebo夾爪誤差, 0.15 預夾取誤差)

    # 設定目標姿態
    target_pose = Pose()
    
    # --- A. 位置 (Position) ---
    # 我們使用 AnyGrasp 的 X 和 Y，但在 Z 軸上增加偏移量
    target_pose.position.x = grasp_x
    target_pose.position.y = grasp_y
    target_pose.position.z = grasp_z + pre_grasp_offset # 關鍵：移動到物體 *上方*
    print(target_pose.position.z)

    # --- B. 旋轉 (Orientation) ---
    # 我們不再使用 AnyGrasp 的複雜旋轉。
    # 我們手動指定一個「夾爪垂直朝下」的姿態。
    # 這對應於 RPY = [0, pi, 0] (繞 Y 軸旋轉 180 度)
    # Quaternion (x,y,z,w) = [0, 1, 0, 0]
    
    # target_pose.orientation.x = 0.0
    # target_pose.orientation.y = 1.0  # 核心：繞 Y 軸轉 180 度
    # target_pose.orientation.z = 0.0
    # target_pose.orientation.w = 0.0  # 注意 w 是 0
    
    target_pose.orientation.x = grasp_ox
    target_pose.orientation.y = grasp_oy
    target_pose.orientation.z = grasp_oz
    target_pose.orientation.w = grasp_ow

    # 4. --- 執行 P2P 規劃 (現在會考慮碰撞) ---
    rospy.loginfo("="*20 + " 嘗試規劃 P2P 運動到「垂直預抓取」姿態 " + "="*20)
    rospy.loginfo("目標姿態 (in 'world'):\n%s", target_pose)

    move_group.set_pose_target(target_pose)
    
    # .go() 函式會自動規劃並執行
    plan_success = move_group.go(wait=True)

    if plan_success:
        rospy.loginfo("P2P 預抓取運動執行完畢！")
        rospy.loginfo("手臂現在應該在物體上方 10 公分處，且夾爪垂直朝下。")
    else:
        rospy.logerr("P2P 預抓取運動規劃失敗！")
        rospy.logerr("請檢查：")
        rospy.logerr("  1. 預抓取位置是否仍在碰撞中？")
        rospy.logerr("  2. 「垂直朝下」的姿態是否是手臂的奇異點或無法到達？")


    # --- 清理 ---
    move_group.stop()
    move_group.clear_pose_targets()
    
    # (您可以保留碰撞體，或在結束時移除它們)
    # scene.remove_world_object("table")
    # scene.remove_world_object("coke_can")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()