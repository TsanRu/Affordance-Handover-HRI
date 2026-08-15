#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import os
import rospy
import rospkg
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
import numpy as np
import json
import random
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import Bool, String
from copy import deepcopy
import threading

import math
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from tf.transformations import quaternion_from_euler, quaternion_multiply
from tf.transformations import quaternion_matrix, quaternion_from_euler, quaternion_multiply, quaternion_from_matrix
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from gazebo_msgs.srv import GetModelState
from affordance_gt import OBJECT_AFFORDANCE_AXIS, world_point_to_axis_ratio, is_handle_grasp

class SimpleGraspController:
    def __init__(self):
        # 1. --- 初始化 MoveIt! & ROS ---
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('p2p_grasp_controller', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # --- 初始化右手 ---
        self.group_name = "rightarm" 
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.planning_frame = self.move_group.get_planning_frame() 
        rospy.loginfo(f"Move Group '{self.group_name}' 初始化完畢。 規劃座標系: {self.planning_frame}")
        
        # --- 初始化左手 ---
        self.left_group_name = "leftarm"
        self.left_move_group = moveit_commander.MoveGroupCommander(self.left_group_name)
        rospy.loginfo(f"Move Group '{self.left_group_name}' 初始化完畢。")
        
        # 2. --- 初始化 TF2 監聽器 (使用你的邏輯) --                                                                                                                                                                                                                                        -
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # 3. --- 初始化通訊 ---
        self.llm_trigger_pub = rospy.Publisher("/system/trigger_llm", String, queue_size=1)
        self.anygrasp_trigger_pub = rospy.Publisher("/system/trigger_detection", String, queue_size=1)
        # FoundationPose 觸發
        self.fp_trigger_pub = rospy.Publisher(
            "/system/trigger_pose", String, queue_size=1)
        
        #  初始化戰況回報的 Publisher
        self.result_pub = rospy.Publisher("/system/handover_result", String, queue_size=1)
        
        # 4. --- 初始化夾爪 Action Client (右手) ---
        gripper_topic = f'/{self.group_name}/gripper_controller/gripper_cmd'
        self.gripper_client = actionlib.SimpleActionClient(gripper_topic, GripperCommandAction)
        
        rospy.loginfo(f"正在等待右手夾爪伺服器: {gripper_topic} ...")
        if self.gripper_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("右手夾爪伺服器連線成功！")
        else:
            rospy.logwarn("右手夾爪伺服器連線失敗 (Timeout)！")
            self.gripper_client = None

        #  初始化夾爪 Action Client (左手) ---
        left_gripper_topic = f'/{self.left_group_name}/gripper_controller/gripper_cmd'
        self.left_gripper_client = actionlib.SimpleActionClient(left_gripper_topic, GripperCommandAction)
        
        rospy.loginfo(f"正在等待左手夾爪伺服器: {left_gripper_topic} ...")
        if self.left_gripper_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("左手夾爪伺服器連線成功！")
        else:
            rospy.logwarn("左手夾爪伺服器連線失敗 (Timeout)！")
            self.left_gripper_client = None
        
        # 5. --- 初始化 Link Attacher 服務 ---
        rospy.loginfo("正在連接 Link Attacher 服務...")
        self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        try:
            self.attach_srv.wait_for_service(timeout=2.0)
            self.detach_srv.wait_for_service(timeout=2.0)
            rospy.loginfo("Link Attacher 服務連接成功！")
        except rospy.ROSException:
            rospy.logwarn("⚠️ 警告：找不到 /link_attacher_node 服務，物理抓取模擬將無效！")
            
        rospy.sleep(1.0) # 等待連線
        
        # 顏色對照表，方便終端機顯示
        self.COLOR_MAP = {
            0: "綠色 (Rank 1)",
            1: "青色 (Rank 2)",
            2: "黃色 (Rank 3)",
            3: "紫色 (Rank 4)",
            4: "紅色 (Rank 5)",
            5: "藍色 (Rank 6)",
            6: "橘色 (Rank 7)",
            7: "深紫 (Rank 8)",
            8: "深青 (Rank 9)",
            9: "灰色 (Rank 10)"
        }
        
        # 6. --- 設定 MoveIt! 碰撞場景 ---
        self.setup_planning_scene()
        
        self.handover_strategy = "geometric"   # 從 brain 接收
        self.receiver_part = None              # functional_end 時的接取部位名稱
        self.resolved_object_name = None       # brain 意圖解析後的英文物件名
        self.object_points_for_pca = None      # 夾取後暫存物件點雲供 PCA 使用
        self.receiver_centroid_for_pca = None
        self.giver_centroid_for_pca = None
        self.object_centroid_for_pca = None
        self.gc_world_actual = None   # 夾取成功後的實際夾爪位置
        self.object_centroid_offset = None  # 物件中心相對夾爪的偏移
        
        # 待命位置 joint angles
        self.left_standby_joints = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]

        # 直接連到 controller 的 action client（用於同步執行）
        self.right_traj_client = actionlib.SimpleActionClient(
            '/rightarm/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.left_traj_client = actionlib.SimpleActionClient(
            '/leftarm/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        rospy.loginfo("等待 trajectory action servers...")
        self.right_traj_client.wait_for_server(timeout=rospy.Duration(5.0))
        self.left_traj_client.wait_for_server(timeout=rospy.Duration(5.0))
        rospy.loginfo("✅ trajectory action servers 連線成功")
        
        self.latest_points = None
        self.latest_mask = None
        rospy.Subscriber("/anygrasp/object_points", String, self._object_points_cb)

        self.gripper_len = 0.135
        self.right_grasp_center_z = None
        self.operation_start_time = None
        self.latest_fp_pose_cam = None
        self.metric_mission_start  = None
        self.metric_inference_time = None
        self.metric_grasp_time     = None
        self.metric_rotation_angle = None
        self.metric_hoe_value      = None
        self.metric_hoe_label      = None
        self.metric_affordance_ratio = None
        self.metric_affordance_hit   = None

    # --- 設定規劃場景 (加入桌子) ---
    def setup_planning_scene(self):
        """將 Gazebo 中的環境物件 (如桌子) 加入 MoveIt! 以進行碰撞檢測"""
        rospy.loginfo("正在設定 MoveIt! 規劃場景 (加入障礙物)...")
        
        # 移除舊物件 (避免重複疊加)
        self.scene.remove_world_object("table")
        rospy.sleep(0.5)

        # --- 加入桌子 ---
        # 桌面是 ur_base 模型（world 裡 9 個排陣列），top plate center z=0.67501m，
        # 厚度 0.02m → 頂面實際在 z=0.685m。
        # 手臂 base joint 在 z=0.69m（桌面上 5mm 的固定座）。
        # box 設 0.680m（桌面下 5mm）：擋住手臂進入桌體，
        # 同時留足夠間隙讓 wrist 在 z=0.69m 時不誤觸 box 邊界。
        table_height = 0.680
        table_size = [2.0, 2.0, table_height]
        
        table_pose = PoseStamped()
        table_pose.header.frame_id = self.planning_frame
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.x = 1.3 
        table_pose.pose.position.y = 0.0
        # Box 的原點在中心，所以 Z 要設為高度的一半
        table_pose.pose.position.z = table_height / 2.0
        
        self.scene.add_box("table", table_pose, size=table_size)
        
        # 註：不建議在這裡加入 "coke_can" 作為碰撞體，
        # 因為 AnyGrasp 偵測的位置可能跟這裡寫死的不同，會導致 MoveIt 誤判碰撞而無法抓取。
        # 我們只加入靜態環境 (桌子) 即可。

        rospy.loginfo("✅ 桌子已加入場景。")
        rospy.sleep(1.0) # 等待場景更新
        
    # --- 控制夾爪的函式 (提取自 handover.py 並改為 class method) ---
    def control_gripper(self, position, max_effort=1000.0, arm="right"):
        client = self.left_gripper_client if arm == "left" else self.gripper_client
        if client is None:
            rospy.logwarn(f"[{arm}] 夾爪未連線，跳過動作。")
            return

        goal = GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        client.send_goal(goal)
        rospy.loginfo(f"[{arm}] 夾爪指令發送: pos={position}")
        
    # --- Attach 物件的函式 ---
    def attach_object(self, object_name, arm="right"):
        """ 呼叫 Gazebo 服務將物件黏在機械手臂上 """
        rospy.loginfo(f"嘗試將物件 '{object_name}' Attach 到 {arm} 手臂...")
        req = AttachRequest()
        req.model_name_1 = object_name      
        req.link_name_1 = "link"            
        req.model_name_2 = "robot"          
        req.link_name_2 = "leftarm_wrist_3_link" if arm == "left" else "rightarm_wrist_3_link"
        
        try:
            resp = self.attach_srv.call(req)
            if resp.ok:
                rospy.loginfo(f"✅ 成功 Attach: {object_name} -> {req.link_name_2}")
                return True
            else:
                rospy.logerr(f"❌ Attach 失敗 (ok=False): {object_name} -> {req.link_name_2}")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Attach 失敗: {e}")
            return False
        
    def detach_object(self, object_name, arm="right"):
        """發生錯誤要放棄抓取時，呼叫此函式解除 Gazebo 裡的物理綁定"""
        rospy.loginfo(f"嘗試解除物件 '{object_name}' 的 Attach 綁定...")
        req = AttachRequest()
        req.model_name_1 = object_name      
        req.link_name_1 = "link"            
        req.model_name_2 = "robot"          
        req.link_name_2 = "leftarm_wrist_3_link" if arm == "left" else "rightarm_wrist_3_link"
        try:
            self.detach_srv.call(req)
            rospy.loginfo("✅ 成功 Detach 釋放物件。")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Detach 失敗: {e}")
            return False
        
    def safe_retreat(self, pose_wrist_grasp, retreat_dist=0.15, arm="right"):
        """
        沿夾取接近方向反向退出，避免撞到物件
        retreat_dist: 退出距離（公尺）
        """
        group = self.left_move_group if arm == "left" else self.move_group
        
        # 取得當前手臂末端姿態
        current_pose = group.get_current_pose().pose
        
        # 從 pose_wrist_grasp 的姿態算出接近方向（tool Z 軸的反方向）
        q = [
            pose_wrist_grasp.orientation.x,
            pose_wrist_grasp.orientation.y,
            pose_wrist_grasp.orientation.z,
            pose_wrist_grasp.orientation.w
        ]
        rot_matrix = quaternion_matrix(q)
        
        # tool Z 軸的反方向（退出方向）
        retreat_vector = -rot_matrix[:3, 2] * retreat_dist
        
        # 從當前位置往退出方向移動
        pose_retreat = deepcopy(current_pose)
        pose_retreat.position.x += retreat_vector[0]
        pose_retreat.position.y += retreat_vector[1]
        pose_retreat.position.z += retreat_vector[2]
        
        rospy.loginfo(f"🔙 [{arm}] 沿接近軸反向退出 {retreat_dist*100:.0f}cm...")
        (plan, fraction) = group.compute_cartesian_path(
            [pose_retreat], 0.01, True
        )
        
        if fraction > 0.8:
            success = group.execute(plan, wait=True)
            if success:
                rospy.loginfo(f"✅ [{arm}] 安全退出成功")
                return True
        
        rospy.logwarn(f"⚠️ [{arm}] 沿軸退出失敗，嘗試垂直向上退出...")
        # 備案：直接往上退
        pose_up = deepcopy(current_pose)
        pose_up.position.z += retreat_dist
        (plan_up, frac_up) = group.compute_cartesian_path(
            [pose_up], 0.01, True
        )
        if frac_up > 0.8:
            group.execute(plan_up, wait=True)
            return True
        
        return False
    
    # --- 安全撤退函式 ---
    def go_home(self, arm="right", pose_near_object=None):
        rospy.loginfo(f"🔄 啟動安全撤退機制：{arm} 退回預設 Home 位置...")
        if pose_near_object is not None:
            self.safe_retreat(pose_near_object, retreat_dist=0.15, arm=arm)
        group = self.left_move_group if arm == "left" else self.move_group
        if arm == "right":
            safe_joint_angles = [1.43, -1.211, -2.0, 0.0, 1.6476, -0.0237]
        else:
            safe_joint_angles = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]
        try:
            group.set_joint_value_target(safe_joint_angles)
            success = group.go(wait=True)
            group.stop()
            if success: rospy.loginfo(f"✅ {arm} 已安全撤退至 Home 位置！")
            else: rospy.logwarn(f"⚠️ {arm} 撤退路徑規劃失敗。")
        except Exception as e:
            rospy.logerr(f"❌ 撤退時發生異常: {e}")
            
    # 將 JSON 的 dict 轉回 ROS Pose 結構
    def dict_to_pose(self, d):
        p = Pose()
        p.position.x, p.position.y, p.position.z = d['position']['x'], d['position']['y'], d['position']['z']
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = d['orientation']['x'], d['orientation']['y'], d['orientation']['z'], d['orientation']['w']
        return p
        
    # 改為單點轉換 (因為我們現在是從 JSON 裡一個一個把點拿出來)
    def transform_single_pose(self, pose, source_frame="camera_color_optical_frame"):
        target_frame = self.planning_frame
        p_stamped = PoseStamped()
        p_stamped.header.frame_id = source_frame
        p_stamped.header.stamp = rospy.Time(0)
        p_stamped.pose = pose
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
            p_world = tf2_geometry_msgs.do_transform_pose(p_stamped, transform)
            return p_world.pose
        except Exception as e:
            rospy.logerr(f"TF Transform failed: {e}")
            return None
        
    def _object_points_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.object_points_for_pca = np.array(data["points"])
            rc = data.get("receiver_centroid", None)
            gc = data.get("giver_centroid", None)  
            oc = data.get("object_centroid", None)  
            self.receiver_centroid_for_pca = np.array(rc) if rc is not None else None
            self.giver_centroid_for_pca = np.array(gc) if gc is not None else None
            self.object_centroid_for_pca = np.array(oc) if oc is not None else None
        except Exception:
            pass

    # =========================================================================
    # 在世界座標系下，精準推算左手在空中的交接點
    # =========================================================================
    def calculate_air_handover_pose(self, pose_R_table_world, pose_L_table_world, pose_R_air_world):
            """
            純平移計算：假設右手搬運時不旋轉，直接將 XYZ 移動量套用到左手上。
            """
            # 1. 計算右手在 XYZ 空間中的「移動向量」
            dx = pose_R_air_world.position.x - pose_R_table_world.position.x
            dy = pose_R_air_world.position.y - pose_R_table_world.position.y
            dz = pose_R_air_world.position.z - pose_R_table_world.position.z

            # 2. 複製左手原本在桌上的 Pose (包含原本的 Orientation)
            pose_L_air = deepcopy(pose_L_table_world)
            
            # 3. 將移動向量加到左手的位置上
            pose_L_air.position.x += dx
            pose_L_air.position.y += dy
            pose_L_air.position.z += dz

            # 旋轉 (Orientation) 完全不動，維持 AnyGrasp 找出來的最佳姿勢！
            return pose_L_air
    
    def calculate_wrist_pose(self, grasp_pose, offset_distance):
        """ 
        【移入】 輔助函式：根據指尖位置與夾爪長度，回推手腕位置 
        """
        q = [grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z, grasp_pose.orientation.w]
        rot_matrix = quaternion_matrix(q)
        # UR tool0: -Z 是後退方向
        local_offset_vector = np.array([0, 0, -offset_distance, 1])
        global_offset_vector = np.dot(rot_matrix, local_offset_vector)
        
        wrist_pose = deepcopy(grasp_pose)
        wrist_pose.position.x += global_offset_vector[0]
        wrist_pose.position.y += global_offset_vector[1]
        wrist_pose.position.z += global_offset_vector[2]
        return wrist_pose

    def calculate_grasp_targets(self, world_pose):
        """ 
        從 AnyGrasp 的原始 Pose 計算出真正的手腕目標點 
        包含：旋轉修正、Pre-grasp 計算
        """
        # 1. 旋轉修正
        q_orig = [
            world_pose.orientation.x,
            world_pose.orientation.y,
            world_pose.orientation.z,
            world_pose.orientation.w
        ]

        # 依據您之前的代碼進行修正 (抬頭 + 轉手腕)
        q_lift = quaternion_from_euler(0, 1.5708, 0) 
        q_step1 = quaternion_multiply(q_orig, q_lift)
        q_rotate_wrist = quaternion_from_euler(0, 0, -1.5708) 
        q_final = quaternion_multiply(q_step1, q_rotate_wrist)

        # 2. 建立指尖的理想 Pose
        pose_fingertip = deepcopy(world_pose)
        pose_fingertip.orientation.x = q_final[0]
        pose_fingertip.orientation.y = q_final[1]
        pose_fingertip.orientation.z = q_final[2]
        pose_fingertip.orientation.w = q_final[3]
        # pose_fingertip.position.z += 0.01
        
        # 3. 計算手腕位置
        gripper_len = self.gripper_len
        pre_grasp_dist = 0.10   # 預備點距離
        
        # 實際抓取時的手腕位置
        pose_wrist_grasp = self.calculate_wrist_pose(pose_fingertip, gripper_len)
        # 預備點的手腕位置
        pose_wrist_pre_grasp = self.calculate_wrist_pose(pose_fingertip, gripper_len + pre_grasp_dist)
        
        return pose_fingertip, pose_wrist_pre_grasp, pose_wrist_grasp

    def compute_place_wrist_z(self, arm="left"):
        """根據右手原始夾取高度，動態計算放置時左手的目標 wrist Z。"""
        mg = self.left_move_group if arm == "left" else self.move_group
        current_pose = mg.get_current_pose().pose
        q = [current_pose.orientation.x, current_pose.orientation.y,
             current_pose.orientation.z, current_pose.orientation.w]
        rot = quaternion_matrix(q)
        tool_z_world_z = rot[2, 2]  # 夾爪 Z 軸的世界 Z 分量
        current_grasp_z = current_pose.position.z + self.gripper_len * tool_z_world_z
        delta_z = self.right_grasp_center_z - current_grasp_z
        return current_pose.position.z + delta_z

    def execute_air_handover(self, ranked_groups, object_name,
                         handover_pose, pose_right_table, use_direct_pose=False):
        """
        use_direct_pose=True：receiver_only 模式，姿態已是交接區真實位置，不做平移
        use_direct_pose=False：dual 模式初始結果，需要做平移計算
        """
        self.control_gripper(0.0, arm="left")
        rospy.sleep(0.5)

        total_count = 0
        for i, group in enumerate(ranked_groups):
            candidates = group.get('left_candidates', [])
            if not candidates:
                continue

            for l_count, cand in enumerate(candidates):
                total_count += 1
                rospy.loginfo(f"接收臂嘗試備案 #{total_count}")

                pose_L_camera = self.dict_to_pose(cand['pose_L_table'])
                pose_L_world = self.transform_single_pose(pose_L_camera)
                if pose_L_world is None:
                    continue
                
                rospy.loginfo(
                f"接收臂目標: x={pose_L_world.position.x:.2f}, "
                f"y={pose_L_world.position.y:.2f}, "
                f"z={pose_L_world.position.z:.2f}")

                if use_direct_pose:
                    # receiver_only：直接用偵測到的姿態
                    pose_L_target = pose_L_world
                else:
                    # dual 初始結果：做平移計算
                    pose_L_target = self.calculate_air_handover_pose(
                        pose_right_table, pose_L_world, handover_pose)

                pose_l_fingertip, pose_l_wrist_pre, pose_l_wrist_grasp = \
                    self.calculate_grasp_targets(pose_L_target)

                # Pre-grasp
                self.left_move_group.set_pose_target(pose_l_wrist_pre)
                l_plan_result = self.left_move_group.plan()
                self.left_move_group.clear_pose_targets()

                if not l_plan_result[0]:
                    rospy.logwarn("接收臂 Pre-grasp 規劃失敗，換下一個備案")
                    continue

                if not self.left_move_group.execute(l_plan_result[1], wait=True):
                    self.left_move_group.stop()
                    self.safe_retreat(pose_l_wrist_pre, arm="left")
                    self.go_home("left")
                    continue

                self.left_move_group.stop()

                (plan_l, frac_l) = self.left_move_group.compute_cartesian_path(
                    [pose_l_wrist_grasp], 0.01, True)

                if frac_l < 0.5:
                    rospy.logwarn(f"接收臂 Approach 規劃不完整 ({frac_l:.2f})")
                    self.safe_retreat(pose_l_wrist_pre, arm="left")
                    self.go_home("left")
                    continue

                if not self.left_move_group.execute(plan_l, wait=True):
                    self.left_move_group.stop()
                    self.safe_retreat(pose_l_wrist_grasp, arm="left")
                    self.go_home("left")
                    continue

                self.left_move_group.stop()
                # ── 精度診斷 ──
                rospy.sleep(0.3)
                tf_t = None
                try:
                    tw = self.tf_buffer.lookup_transform(
                        "world", "leftarm_wrist_3_link",
                        rospy.Time(0), rospy.Duration(1.0))
                    tf = self.tf_buffer.lookup_transform(
                        "world", "leftarm_robotiq_85_left_finger_link",
                        rospy.Time(0), rospy.Duration(1.0))
                    tw_t, tf_t = tw.transform.translation, tf.transform.translation
                    rospy.loginfo(
                        f"📍 [左手] 實際手腕:  x={tw_t.x:.4f}, y={tw_t.y:.4f}, z={tw_t.z:.4f}")
                    rospy.loginfo(
                        f"📍 [左手] 實際指尖:  x={tf_t.x:.4f}, y={tf_t.y:.4f}, z={tf_t.z:.4f}")
                    rospy.loginfo(
                        f"🎯 [左手] 期望指尖:  x={pose_l_fingertip.position.x:.4f}, "
                        f"y={pose_l_fingertip.position.y:.4f}, z={pose_l_fingertip.position.z:.4f}")
                    rospy.loginfo(
                        f"📏 [左手] 指尖誤差:  dx={tf_t.x-pose_l_fingertip.position.x:.4f}, "
                        f"dy={tf_t.y-pose_l_fingertip.position.y:.4f}, "
                        f"dz={tf_t.z-pose_l_fingertip.position.z:.4f}")
                except Exception as e:
                    rospy.logwarn(f"診斷失敗: {e}")
                # ── 診斷結束 ──

                # ── Affordance GT 指標：左手接觸點是否落在握柄 GT 區間 ──
                # 用 tf_t（左手指尖闔爪前的世界座標，上面精度診斷已經量過）
                # 搭配 Gazebo ground truth 物件姿態換算，避免把 FoundationPose
                # 的估測誤差混進這個指標。只對已標定 affordance 主軸的物件量測。
                if object_name in OBJECT_AFFORDANCE_AXIS and tf_t is not None:
                    try:
                        rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
                        get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                        obj_state = get_model_state(object_name, 'world')
                        obj_pos = (obj_state.pose.position.x,
                                   obj_state.pose.position.y,
                                   obj_state.pose.position.z)
                        obj_quat = (obj_state.pose.orientation.x,
                                    obj_state.pose.orientation.y,
                                    obj_state.pose.orientation.z,
                                    obj_state.pose.orientation.w)
                        point_world = (tf_t.x, tf_t.y, tf_t.z)
                        ratio = world_point_to_axis_ratio(point_world, obj_pos, obj_quat, object_name)
                        hit = is_handle_grasp(object_name, ratio)
                        self.metric_affordance_ratio = ratio
                        self.metric_affordance_hit = hit
                        # 物件此時仍 attach 在右手上，正常應在交接高度附近（>0.4m）。
                        # 若明顯偏低，代表途中可能已經飛出/掉落，這筆數據不可信，
                        # 標記出來以便事後稽核，不用再依賴肉眼即時盯著畫面。
                        suspicious = obj_pos[2] < 0.4
                        flag = "  ⚠️ SUSPICIOUS(物件位置異常，疑似飛出/掉落)" if suspicious else ""
                        rospy.loginfo(
                            f"[Affordance] {object_name}: 左手接觸點主軸位置={ratio * 100:.1f}%  "
                            f"落在握柄GT={'是' if hit else '否'}  "
                            f"obj_pos=({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}){flag}")
                    except Exception as e:
                        rospy.logwarn(f"[Affordance] 量測失敗: {e}")
                # ── Affordance GT 指標結束 ──

                self.control_gripper(0.5, arm="left")
                rospy.sleep(1.0)
                self.attach_object(object_name, arm="left")
                self.detach_object(object_name, arm="right")
                self._remove_collision_mesh(object_name)
                self.control_gripper(0.0, arm="right")

                rospy.loginfo("🎉 交接成功，開始撤退流程...")

                # 左手目標：三個展示姿態隨機選一，拿著物件離開交接區
                left_carry_poses = {
                    "A": [2.2,  -0.5,  -2.2, -1.0,  1.6476,  0.3   ],
                    "B": [2.6,  -1.6,  -0.8, -0.3,  1.3,    -0.0237],
                    "C": [0.7,  -0.8,  -2.0, -0.8,  1.8,    -0.3   ],
                }
                carry_key = random.choice(list(left_carry_poses.keys()))
                carry_joints = left_carry_poses[carry_key]
                rospy.loginfo(f"左手目標姿態: {carry_key}")

                # Step 1（序列）：右手先垂直往上退 15cm，脫離物件空間
                current_r = self.move_group.get_current_pose().pose
                lift_r = deepcopy(current_r)
                lift_r.position.z += 0.15
                (plan_lift, frac_lift) = self.move_group.compute_cartesian_path(
                    [lift_r], 0.01, True)
                if frac_lift >= 0.9:
                    self.move_group.execute(plan_lift, wait=True)
                    self.move_group.stop()
                    rospy.loginfo("✅ 右手垂直上移完成")
                else:
                    rospy.logwarn("⚠️ 右手上移規劃失敗，略過")

                # Step 2：嘗試同步（右手回 home + 左手拿物件移到隨機展示姿態）
                right_home_joints = [1.43, -1.211, -2.0, 0.0, 1.6476, -0.0237]
                self.move_group.set_joint_value_target(right_home_joints)
                plan_right_home = self.move_group.plan()
                self.move_group.clear_pose_targets()

                self.left_move_group.set_joint_value_target(carry_joints)
                plan_left_carry = self.left_move_group.plan()
                self.left_move_group.clear_pose_targets()

                if plan_right_home[0] and plan_left_carry[0]:
                    rospy.loginfo(f"✅ 同步：右手回 home + 左手移到姿態 {carry_key}")
                    goal_r = FollowJointTrajectoryGoal()
                    goal_r.trajectory = plan_right_home[1].joint_trajectory
                    goal_l = FollowJointTrajectoryGoal()
                    goal_l.trajectory = plan_left_carry[1].joint_trajectory
                    self.right_traj_client.send_goal(goal_r)
                    self.left_traj_client.send_goal(goal_l)
                    self.right_traj_client.wait_for_result()
                    self.left_traj_client.wait_for_result()
                    self.move_group.stop()
                    self.left_move_group.stop()
                    rospy.loginfo("✅ 同步移動完成")
                else:
                    rospy.logwarn("⚠️ 同步規劃失敗，改為序列執行")
                    self.go_home("right")
                    self.left_move_group.set_joint_value_target(carry_joints)
                    plan_left_seq = self.left_move_group.plan()
                    self.left_move_group.clear_pose_targets()
                    if plan_left_seq[0]:
                        self.left_move_group.execute(plan_left_seq[1], wait=True)
                        self.left_move_group.stop()
                        rospy.loginfo(f"✅ 序列：左手移到姿態 {carry_key}")
                    else:
                        rospy.logwarn(f"⚠️ 左手姿態 {carry_key} 規劃失敗，維持原位")

                rospy.loginfo("🎉🎉 雙臂空中交接成功！")
                elapsed_total = (
                    (rospy.Time.now() - self.operation_start_time).to_sec()
                    if self.operation_start_time is not None else None
                )
                _sep = "=" * 48
                rospy.loginfo(_sep)
                rospy.loginfo("[EXPERIMENT SUMMARY]")
                if self.metric_inference_time is not None:
                    rospy.loginfo(f"  推論時間  (LLM + AnyGrasp) : {self.metric_inference_time:.2f} s")
                if self.metric_grasp_time is not None:
                    rospy.loginfo(f"  夾取耗時                    : {self.metric_grasp_time:.2f} s")
                if elapsed_total is not None:
                    rospy.loginfo(f"  交接總耗時 (夾取→完成)      : {elapsed_total:.2f} s")
                if self.metric_rotation_angle is not None:
                    rospy.loginfo(f"  旋轉量 (rot)                : {self.metric_rotation_angle:+.1f}°")
                if self.metric_hoe_value is not None:
                    label = self.metric_hoe_label or "HOE"
                    rospy.loginfo(f"  {label:<28}: {self.metric_hoe_value:+.1f}°")
                else:
                    rospy.loginfo("  HOE                         : N/A")
                if self.metric_affordance_ratio is not None:
                    rospy.loginfo(
                        f"  Affordance (握柄GT)         : {self.metric_affordance_ratio * 100:.1f}%  "
                        f"({'HIT' if self.metric_affordance_hit else 'MISS'})")
                else:
                    rospy.loginfo("  Affordance (握柄GT)         : N/A")
                rospy.loginfo(_sep)
                self.result_pub.publish(json.dumps({
                    "status": "success",
                    "method": "air_handover",
                    "affordance_ratio": self.metric_affordance_ratio,
                    "affordance_hit": self.metric_affordance_hit,
                }))
                return True

        rospy.logwarn("所有接收臂空中備案皆失敗")
        if self.operation_start_time is not None:
            elapsed = (rospy.Time.now() - self.operation_start_time).to_sec()
            rospy.loginfo(f"⏱️ 夾取到任務失敗總耗時：{elapsed:.2f} 秒")
        return False


    def execute_left_standalone_grasp(self, left_groups, object_name):
        """
        右手放下物件退開後，左手獨立夾取
        """
        rospy.loginfo("左手開始獨立夾取流程")
        self.control_gripper(0.0, arm="left")
        rospy.sleep(0.5)

        for i, group in enumerate(left_groups):
            rospy.loginfo(f"左手嘗試第 #{i + 1} 組姿態")

            candidates = group.get('left_candidates', [])
            if not candidates:
                continue
            pose_L_camera = self.dict_to_pose(candidates[0]['pose_L_table'])
            pose_L_world = self.transform_single_pose(pose_L_camera)
            if pose_L_world is None:
                continue

            pose_l_fingertip, pose_l_wrist_pre, pose_l_wrist_grasp = \
                self.calculate_grasp_targets(pose_L_world)

            # Pre-grasp
            self.left_move_group.set_pose_target(pose_l_wrist_pre)
            plan_result = self.left_move_group.plan()
            self.left_move_group.clear_pose_targets()

            if not plan_result[0]:
                rospy.logwarn("左手 Pre-grasp 規劃失敗，換下一組")
                continue

            if not self.left_move_group.execute(plan_result[1], wait=True):
                self.left_move_group.stop()
                self.safe_retreat(pose_l_wrist_pre, arm="left")
                self.go_home("left")
                continue

            self.left_move_group.stop()

            # Approach
            (plan_app, frac_app) = self.left_move_group.compute_cartesian_path(
                [pose_l_wrist_grasp], 0.01, True)

            if frac_app < 0.5:
                rospy.logwarn(f"左手 Approach 規劃不完整 ({frac_app:.2f})")
                self.safe_retreat(pose_l_wrist_pre, arm="left")
                self.go_home("left")
                continue

            if not self.left_move_group.execute(plan_app, wait=True):
                self.left_move_group.stop()
                self.safe_retreat(pose_l_wrist_grasp, arm="left")
                self.go_home("left")
                continue

            self.left_move_group.stop()
            self.control_gripper(0.2, arm="left")
            rospy.sleep(1.0)
            self.attach_object(object_name, arm="left")

            rospy.loginfo("🎉 接收臂獨立夾取成功，移動到展示姿態...")

            # 左手移到隨機展示姿態
            left_carry_poses = {
                "A": [2.2,  -0.5,  -2.2, -1.0,  1.6476,  0.3   ],
                "B": [2.6,  -1.6,  -0.8, -0.3,  1.3,    -0.0237],
                "C": [0.7,  -0.8,  -2.0, -0.8,  1.8,    -0.3   ],
            }
            carry_key = random.choice(list(left_carry_poses.keys()))
            carry_joints = left_carry_poses[carry_key]
            rospy.loginfo(f"左手目標姿態: {carry_key}")

            self.left_move_group.set_joint_value_target(carry_joints)
            plan_carry = self.left_move_group.plan()
            self.left_move_group.clear_pose_targets()
            if plan_carry[0]:
                self.left_move_group.execute(plan_carry[1], wait=True)
                self.left_move_group.stop()
                rospy.loginfo(f"✅ 左手移到姿態 {carry_key}")
            else:
                rospy.logwarn(f"⚠️ 左手姿態 {carry_key} 規劃失敗，維持原位")

            # 實驗數據摘要
            elapsed_total = (
                (rospy.Time.now() - self.operation_start_time).to_sec()
                if self.operation_start_time is not None else None
            )
            _sep = "=" * 48
            rospy.loginfo(_sep)
            rospy.loginfo("[EXPERIMENT SUMMARY]")
            if self.metric_inference_time is not None:
                rospy.loginfo(f"  推論時間  (LLM + AnyGrasp) : {self.metric_inference_time:.2f} s")
            if self.metric_grasp_time is not None:
                rospy.loginfo(f"  夾取耗時                    : {self.metric_grasp_time:.2f} s")
            if elapsed_total is not None:
                rospy.loginfo(f"  總耗時 (觸發→完成)          : {elapsed_total:.2f} s")
            rospy.loginfo(_sep)

            self.result_pub.publish(json.dumps({
                "status": "success",
                "method": "receiver_standalone"
            }))
            return True

        rospy.logerr("左手獨立夾取所有方案皆失敗")
        self.result_pub.publish(json.dumps({"status": "fail"}))
        return False
    
    def request_foundationpose(self, object_name, object_centroid_world=None, timeout=15.0):
        """
        觸發 FoundationPose 估測物件 pose
        
        Returns:
            np.ndarray (4, 4) 物件在相機座標的 pose，或 None
        """
        rospy.loginfo(f"🎯 觸發 FoundationPose 估測 {object_name}...")
        
        result_event = threading.Event()
        result_container = [None]
        
        def fp_cb(msg):
            try:
                result_container[0] = json.loads(msg.data)
                result_event.set()
            except Exception:
                pass
        
        fp_sub = rospy.Subscriber(
            "/pose/foundationpose_result", String, fp_cb)
        rospy.sleep(0.2)
        
        payload = json.dumps({
            "object_name": object_name,
            "object_centroid_world": object_centroid_world.tolist()
                if object_centroid_world is not None else None
        })
        self.fp_trigger_pub.publish(payload)
        
        done = result_event.wait(timeout=timeout)
        fp_sub.unregister()
        
        if not done or result_container[0] is None:
            rospy.logerr("⚠️ FoundationPose 逾時")
            return None
        
        result = result_container[0]
        if result.get("status") != "ok":
            rospy.logerr(f"⚠️ FoundationPose 失敗: {result.get('reason')}")
            return None
        
        pose = np.array(result["pose"]).reshape(4, 4)
        rospy.loginfo(
            f"✓ FoundationPose pose 接收 (translation: {pose[:3,3]})")
        self.latest_fp_pose_cam = pose
        return pose
    
    def trigger_full_detection(self, object_name, mode="dual",
                           rotation_angle=0.0, timeout=120.0, user_input=None):
        
        # # ── receiver_only：跳過 LLM，直接觸發 AnyGrasp ──
        # if mode == "receiver_only":
        #     rospy.loginfo("receiver_only 模式：跳過 LLM，直接觸發 AnyGrasp")
            
        #     plan_event = threading.Event()
        #     plan_container = [None]

        #     def plan_cb(msg):
        #         try:
        #             plan_container[0] = json.loads(msg.data)
        #             plan_event.set()
        #         except Exception:
        #             pass

        #     plan_sub = rospy.Subscriber(
        #         "/anygrasp/handover_plan", String, plan_cb)
        #     rospy.sleep(0.2)

        #     anygrasp_payload = json.dumps({
        #         "object_name": object_name,
        #         "mode": "receiver_only",
        #         "receiver_centroid": self.receiver_centroid_for_pca.tolist()
        #             if self.receiver_centroid_for_pca is not None else None,
        #         "object_centroid": self.object_centroid_for_pca.tolist()
        #             if self.object_centroid_for_pca is not None else None,
        #         "rotation_angle": rotation_angle
        #     })
        #     self.anygrasp_trigger_pub.publish(anygrasp_payload)
            
        #     # FoundationPose 觸發
        #     self.fp_trigger_pub = rospy.Publisher(
        #     "/system/trigger_pose", String, queue_size=1)

        #     plan_done = plan_event.wait(timeout=timeout)
        #     plan_sub.unregister()

        #     if not plan_done or plan_container[0] is None:
        #         rospy.logerr("等待 AnyGrasp receiver_only 回傳逾時")
        #         return None

        #     ranked = plan_container[0]
        #     return ranked if ranked else None
        
        
        if mode == "receiver_only":
            rospy.loginfo("receiver_only 模式：觸發 FoundationPose → AnyGrasp")

            # 推算物件在交接區的當前位置作為 FP hint
            oc_handover = None
            if self.object_centroid_offset is not None:
                try:
                    trans_grip_now = self.tf_buffer.lookup_transform(
                        "world", "rightarm_robotiq_85_base_link",
                        rospy.Time(0), rospy.Duration(1.0))
                    t = trans_grip_now.transform.translation
                    gc_now = np.array([t.x, t.y, t.z])
                    oc_handover = gc_now + self.object_centroid_offset
                    rospy.loginfo(
                        f"📍 推算交接區物件中心: {oc_handover.round(3)}")
                except Exception as e:
                    rospy.logwarn(f"⚠️ 無法推算物件中心: {e}")

            fp_pose = self.request_foundationpose(
                object_name,
                object_centroid_world=oc_handover)
            fp_pose_list = fp_pose.tolist() if fp_pose is not None else None

            # === Step 2: 觸發 AnyGrasp ===
            plan_event = threading.Event()
            plan_container = [None]

            def plan_cb(msg):
                try:
                    plan_container[0] = json.loads(msg.data)
                    plan_event.set()
                except Exception:
                    pass

            plan_sub = rospy.Subscriber(
                "/anygrasp/handover_plan", String, plan_cb)
            rospy.sleep(0.2)

            anygrasp_payload = json.dumps({
                "object_name": object_name,
                "mode": "receiver_only",
                "receiver_centroid": self.receiver_centroid_for_pca.tolist()
                    if self.receiver_centroid_for_pca is not None else None,
                "object_centroid": self.object_centroid_for_pca.tolist()
                    if self.object_centroid_for_pca is not None else None,
                "rotation_angle": rotation_angle,
                "object_pose_in_cam": fp_pose_list,  # ⭐ 新增
            })
            self.anygrasp_trigger_pub.publish(anygrasp_payload)

            plan_done = plan_event.wait(timeout=timeout)
            plan_sub.unregister()

            if not plan_done or plan_container[0] is None:
                rospy.logerr("等待 AnyGrasp receiver_only 回傳逾時")
                return None

            ranked = plan_container[0]
            return ranked if ranked else None
    
        rospy.loginfo(f"觸發 LLM 前處理 (物件: {object_name}, 模式: {mode})...")
        if user_input:
            llm_payload = json.dumps({"user_input": user_input, "mode": mode})
        else:
            llm_payload = json.dumps({"object_name": object_name, "mode": mode})

        llm_done_event = threading.Event()
        llm_result_container = [None]

        def llm_done_cb(msg):
            try:
                result = json.loads(msg.data)
                if result.get("status") == "done":
                    llm_result_container[0] = result
                    llm_done_event.set()
            except Exception:
                pass

        llm_sub = rospy.Subscriber("/system/llm_done", String, llm_done_cb)
        rospy.sleep(0.2)
        self.llm_trigger_pub.publish(llm_payload)

        llm_done = llm_done_event.wait(timeout=timeout / 2)
        llm_sub.unregister()

        if not llm_done or llm_result_container[0] is None:
            rospy.logerr("等待 LLM 完成逾時或失敗")
            return None

        rospy.loginfo("LLM 完成，觸發 AnyGrasp...")
        
        llm_data = llm_result_container[0]
        # 如果 brain 回傳了解析後的英文物件名（意圖解析場景），存起來供 execute_mission 使用
        resolved = llm_data.get("object_name", "")
        if resolved and resolved != "unknown":
            self.resolved_object_name = resolved
            object_name = resolved
        if mode == "dual":  # 只有 dual 才更新策略
            self.handover_strategy = llm_data.get("handover_strategy", "geometric")
            self.receiver_part = llm_data.get("receiver_part", None)
            rospy.loginfo(f"📋 交接策略: {self.handover_strategy}, 接取部位: {self.receiver_part}")

        plan_event = threading.Event()
        plan_container = [None]

        def plan_cb(msg):
            try:
                data = json.loads(msg.data)
                plan_container[0] = data
                plan_event.set()
            except Exception:
                pass

        plan_sub = rospy.Subscriber("/anygrasp/handover_plan", String, plan_cb)
        rospy.sleep(0.2)
        anygrasp_payload = json.dumps({
            "object_name": object_name,
            "mode": mode
        })
        self.anygrasp_trigger_pub.publish(anygrasp_payload)

        plan_done = plan_event.wait(timeout=timeout / 2)
        plan_sub.unregister()

        if not plan_done or plan_container[0] is None:
            rospy.logerr("等待 AnyGrasp 回傳逾時")
            return None

        ranked_groups = plan_container[0]
        rospy.loginfo(f"收到 {len(ranked_groups)} 個候選方案")
        return ranked_groups if len(ranked_groups) > 0 else None
    
    def calculate_rotation_angle_from_pointcloud(self, object_points, receiver_arm_pos, object_pos):
        """
        用 PCA 計算需要旋轉多少度讓物件以正確朝向到達交接區
        object_points: 物件點雲 (Nx3 numpy array，世界座標系)
        receiver_arm_pos: 接收臂基座位置 (世界座標系)
        object_pos: 物件中心位置 (世界座標系)
        """
        if object_points is None or len(object_points) < 10:
            rospy.logwarn("⚠️ 點雲不足，跳過旋轉計算，使用 0 度")
            return 0.0

        centroid = np.mean(object_points, axis=0)
        centered = object_points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 目標方向：物件到接收臂的水平方向
        target_dir = receiver_arm_pos - object_pos
        target_dir[2] = 0
        norm = np.linalg.norm(target_dir)
        if norm < 1e-6:
            return 0.0
        target_dir = target_dir / norm

        if self.handover_strategy == "functional_end":
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

            if self.receiver_centroid_for_pca is not None:
                # 把 receiver_centroid 投影到主軸上
                # 正值表示重心在主軸正方向那端，負值表示在負方向那端
                centroid_to_receiver = self.receiver_centroid_for_pca - centroid
                proj = np.dot(centroid_to_receiver, principal_axis)

                # 讓主軸指向 receiver_centroid 所在的那端（接取端）
                if proj < 0:
                    principal_axis = -principal_axis

                rospy.loginfo(
                    f"✅ 用 receiver_centroid 確認接取端方向，投影值: {proj:.3f}")
            else:
                # fallback：選跟接收臂方向較近的那端
                rospy.logwarn("⚠️ 無 receiver_centroid，改用接收臂方向估算")
                if np.dot(principal_axis[:2], target_dir) < 0:
                    principal_axis = -principal_axis

            current_dir = principal_axis.copy()
            current_dir[2] = 0
            if np.linalg.norm(current_dir) < 1e-6:
                return 0.0
            current_dir = current_dir / np.linalg.norm(current_dir)
            effective_target = target_dir  # functional_end：長軸對齊接收臂方向

        else:  # geometric
            # 找夾爪握得住（寬度 < 8.5cm）且接觸面積最大的面
            gripper_max_width = 0.085
            dims = []
            for i in range(3):
                axis = eigenvectors[:, i]
                proj = centered @ axis
                size = proj.max() - proj.min()
                dims.append((size, eigenvectors[:, i]))

            grippable = [(s, ax) for s, ax in dims if s <= gripper_max_width]
            if grippable:
                grippable.sort(key=lambda x: x[0])
                best_normal = grippable[0][1]
            else:
                dims.sort(key=lambda x: x[0])
                best_normal = dims[0][1]

            current_dir = best_normal.copy()
            current_dir[2] = 0
            if np.linalg.norm(current_dir) < 1e-6:
                return 0.0
            current_dir = current_dir / np.linalg.norm(current_dir)

            # 目標：讓夾爪軸垂直於接收臂方向，使最大面朝向接收臂
            # target_dir 的兩個垂直方向各選一個，取旋轉量最小（dot 最大）的那個
            perp1 = np.array([-target_dir[1],  target_dir[0], 0.0])
            perp2 = np.array([ target_dir[1], -target_dir[0], 0.0])
            effective_target = perp1 if np.dot(current_dir, perp1) >= np.dot(current_dir, perp2) else perp2

        cos_a = np.clip(np.dot(current_dir, effective_target), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_a))
        cross = np.cross(current_dir, effective_target)
        cross_z = cross[2] if hasattr(cross, '__len__') else cross
        if cross_z < 0:
            angle_deg = -angle_deg

        rospy.loginfo(f"PCA 計算旋轉角度: {angle_deg:.1f}°")
        return angle_deg
    
    def calculate_handover_position(self, pose_wrist_grasp_final, current_pose=None):
        grasp_z = pose_wrist_grasp_final.position.z
        handover_z = grasp_z + 0.12

        try:
            trans_r = self.tf_buffer.lookup_transform(
                "world", "rightarm_base_link",
                rospy.Time(0), rospy.Duration(1.0))
            trans_l = self.tf_buffer.lookup_transform(
                "world", "leftarm_base_link",
                rospy.Time(0), rospy.Duration(1.0))
            arm_reach = 0.35
            mid_x = trans_r.transform.translation.x - arm_reach
            mid_y = (trans_r.transform.translation.y
                    + trans_l.transform.translation.y) / 2.0
        except Exception as e:
            rospy.logwarn(f"TF 取得失敗，使用預設值: {e}")
            mid_x, mid_y = 0.425, 0.0

        # X 不動：手臂維持在與夾取相同的 x 距離，僅調整 y 使物件中心對齊兩臂中線
        target_x = pose_wrist_grasp_final.position.x

        if self.object_centroid_offset is not None:
            target_y = mid_y - self.object_centroid_offset[1]
            rospy.loginfo(
                f"📍 物件中心偏移(offset): "
                f"({self.object_centroid_offset[0]:.3f}, {self.object_centroid_offset[1]:.3f})")
        elif self.object_centroid_for_pca is not None:
            delta_y = self.object_centroid_for_pca[1] - pose_wrist_grasp_final.position.y
            target_y = mid_y - delta_y
            rospy.loginfo(
                f"📍 物件中心(點雲fallback): delta_y={delta_y:.3f}")
        elif self.giver_centroid_for_pca is not None and self.receiver_centroid_for_pca is not None:
            obj_center_y = (self.giver_centroid_for_pca[1] + self.receiver_centroid_for_pca[1]) / 2.0
            delta_y = obj_center_y - pose_wrist_grasp_final.position.y
            target_y = mid_y - delta_y
            rospy.loginfo(
                f"📍 物件中心(遮罩fallback): obj_center_y={obj_center_y:.2f}")
        else:
            target_y = pose_wrist_grasp_final.position.y
            rospy.loginfo("📍 物件中心(手腕fallback)")

        rospy.loginfo(f"📍 兩臂中點: ({mid_x:.2f}, {mid_y:.2f})")
        rospy.loginfo(f"📍 右手目標: ({target_x:.2f}, {target_y:.2f}, {handover_z:.2f})")

        handover = Pose()
        handover.position.x = target_x
        handover.position.y = target_y
        handover.position.z = handover_z
        handover.orientation = current_pose.orientation if current_pose is not None \
            else pose_wrist_grasp_final.orientation
        return handover
    
    def rotate_wrist(self, angle_deg, arm="right"):
        group = self.left_move_group if arm == "left" else self.move_group
        
        # 取得當前末端位姿
        current_pose = group.get_current_pose().pose
        
        # 在世界座標系下繞 Z 軸旋轉（不改變 XYZ 位置）
        q_current = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        # 繞世界 Z 軸的旋轉 quaternion
        q_rot = quaternion_from_euler(0, 0, np.radians(angle_deg))
        # 左乘 = 在世界座標系下旋轉
        q_new = quaternion_multiply(q_rot, q_current)
        
        target_pose = deepcopy(current_pose)
        target_pose.orientation.x = q_new[0]
        target_pose.orientation.y = q_new[1]
        target_pose.orientation.z = q_new[2]
        target_pose.orientation.w = q_new[3]
        
        # 用 Cartesian path 執行（保持位置不變，只改姿態）
        (plan, fraction) = group.compute_cartesian_path(
            [target_pose], 0.01, True)
        
        if fraction > 0.5:
            success = group.execute(plan, wait=True)
            group.stop()
            if success:
                rospy.loginfo(f"✅ 手腕旋轉 {angle_deg:.1f}° 完成（世界 Z 軸）")
                return True
        
        rospy.logwarn(f"⚠️ 手腕旋轉失敗")
        group.stop()
        return False
        
    DEMO_LEFT_INIT_POSES = {
        "A（高舉外展）": [2.2,  -0.5,  -2.2,  -1.0,  1.6476,  0.3],
        "B（低伸外旋）": [2.6,  -1.6,  -0.8,  -0.3,  1.3,    -0.0237],
        "C（內收前伸）": [0.7,  -0.8,  -2.0,  -0.8,  1.8,    -0.3],
    }

    # 各物件理想交接 yaw（世界座標，handle 朝向左臂 -Y 方向）
    IDEAL_YAW = {
        "hammer":              -21.5,
        "scissors":             16.7,
        "mug":                 -91.8,
        "screwdriver": -117.2,
        "spatula":             109.3,
        "spoon":             -72.6,
    }

    # 碰撞 mesh 路徑（nontextured.stl，幾何乾淨適合碰撞計算）
    # 用 rospkg 動態解析 ur_gripper_gazebo 套件路徑，不寫死使用者/工作區位置
    _GAZEBO_MODELS_DIR = os.path.join(rospkg.RosPack().get_path('ur_gripper_gazebo'), 'models')
    OBJECT_MESH_MAP = {
        "hammer":   f"{_GAZEBO_MODELS_DIR}/048_hammer/google_16k/nontextured.stl",
        "scissors": f"{_GAZEBO_MODELS_DIR}/037_scissors/google_16k/nontextured.stl",
        "mug":                  f"{_GAZEBO_MODELS_DIR}/025_mug/google_16k/nontextured.stl",
        "screwdriver": f"{_GAZEBO_MODELS_DIR}/043_phillips_screwdriver/google_16k/nontextured.stl",
        "sugar_box":   f"{_GAZEBO_MODELS_DIR}/004_sugar_box/google_16k/nontextured.stl",
        "spatula":     f"{_GAZEBO_MODELS_DIR}/033_spatula/google_16k/nontextured.stl",
        "spoon":       f"{_GAZEBO_MODELS_DIR}/031_spoon/google_16k/nontextured.stl",
        "tomato_soup_can": f"{_GAZEBO_MODELS_DIR}/005_tomato_soup_can/google_16k/nontextured.stl",
        "large_clamp": f"{_GAZEBO_MODELS_DIR}/051_large_clamp/google_16k/nontextured.stl",
    }
    COLLISION_MESH_SCALE = 0.85  # 略小於實際，補償 FP 位姿誤差

    # 右手持物：整個 gripper 都接觸物件（夾住）
    TOUCH_LINKS_RIGHT_HOLDING = [
        "rightarm_wrist_3_link",
        "rightarm_robotiq_85_base_link",
        "rightarm_robotiq_85_left_finger_link",
        "rightarm_robotiq_85_left_finger_tip_link",
        "rightarm_robotiq_85_left_inner_knuckle_link",
        "rightarm_robotiq_85_left_knuckle_link",
        "rightarm_robotiq_85_right_finger_link",
        "rightarm_robotiq_85_right_finger_tip_link",
        "rightarm_robotiq_85_right_inner_knuckle_link",
        "rightarm_robotiq_85_right_knuckle_link",
    ]
    # 左手接收：只豁免指尖（讓 base/finger body 仍受碰撞限制，强迫繞路）
    TOUCH_LINKS_LEFT_RECEIVING = [
        "leftarm_robotiq_85_left_finger_tip_link",
        "leftarm_robotiq_85_right_finger_tip_link",
    ]
    TOUCH_LINKS_FOR_HELD_OBJECT = TOUCH_LINKS_RIGHT_HOLDING + TOUCH_LINKS_LEFT_RECEIVING

    def _wait_for_scene_update(self, col_name, expect_attached, timeout=5.0):
        """輪詢 MoveIt 規劃場景直到更新確認，避免非同步延遲導致規劃時場景未更新。"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while rospy.Time.now() < deadline:
            attached = self.scene.get_attached_objects([col_name])
            known   = self.scene.get_known_object_names()
            if expect_attached and col_name in attached:
                return True
            if not expect_attached and col_name not in known and col_name not in attached:
                return True
            rospy.sleep(0.1)
        return False

    def _add_world_collision_mesh(self, object_name):
        """從 Gazebo 取得物件當前位姿，加入 MoveIt 場景作為障礙物（不 attach）。
        用於 pre-grasp 規劃前，讓 MoveIt 繞開物件而不撞倒它。"""
        mesh_path = self.OBJECT_MESH_MAP.get(object_name)
        if not mesh_path or not os.path.exists(mesh_path):
            return False
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
            get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = get_state(object_name, 'world')
            ps = PoseStamped()
            ps.header.frame_id = "world"
            ps.header.stamp = rospy.Time.now()
            ps.pose = resp.pose
            col_name = object_name + "_col"
            s = self.COLLISION_MESH_SCALE
            self.scene.add_mesh(col_name, ps, mesh_path, size=(s, s, s))
            self._wait_for_scene_update(col_name, expect_attached=False)
            rospy.loginfo(f"[collision mesh] ✅ {col_name} 加入場景障礙物（Gazebo 位姿）")
            return True
        except Exception as e:
            rospy.logwarn(f"[collision mesh] 加入障礙物失敗: {e}")
            return False

    def _attach_collision_mesh(self, object_name):
        """把物件 mesh 加入 MoveIt 場景並 attach 到右手 wrist，讓接收臂規劃時知道要閃它。"""
        mesh_path = self.OBJECT_MESH_MAP.get(object_name)
        if not mesh_path or not os.path.exists(mesh_path):
            rospy.logwarn(f"[collision mesh] 找不到 {object_name} 的 mesh，跳過")
            return False
        if self.latest_fp_pose_cam is None:
            rospy.logwarn("[collision mesh] 沒有 FoundationPose 估測結果，跳過")
            return False
        try:
            trans = self.tf_buffer.lookup_transform(
                "world", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(2.0))
            t, q = trans.transform.translation, trans.transform.rotation
            T_wc = quaternion_matrix([q.x, q.y, q.z, q.w])
            T_wc[:3, 3] = [t.x, t.y, t.z]
            T_wo = T_wc @ self.latest_fp_pose_cam

            ps = PoseStamped()
            ps.header.frame_id = "world"
            ps.header.stamp = rospy.Time.now()
            ps.pose.position.x = T_wo[0, 3]
            ps.pose.position.y = T_wo[1, 3]
            ps.pose.position.z = T_wo[2, 3]
            qo = quaternion_from_matrix(T_wo)
            ps.pose.orientation.x = qo[0]
            ps.pose.orientation.y = qo[1]
            ps.pose.orientation.z = qo[2]
            ps.pose.orientation.w = qo[3]

            col_name = object_name + "_col"
            rospy.loginfo(
                f"[collision mesh] 物件世界座標: "
                f"({T_wo[0,3]:.3f}, {T_wo[1,3]:.3f}, {T_wo[2,3]:.3f})")

            s = self.COLLISION_MESH_SCALE
            self.scene.add_mesh(col_name, ps, mesh_path, size=(s, s, s))
            if not self._wait_for_scene_update(col_name, expect_attached=False):
                rospy.logwarn("[collision mesh] add_mesh 場景更新超時")

            self.scene.attach_object(col_name, "rightarm_wrist_3_link",
                                     touch_links=self.TOUCH_LINKS_FOR_HELD_OBJECT)
            if not self._wait_for_scene_update(col_name, expect_attached=True):
                rospy.logwarn("[collision mesh] attach_object 場景更新超時，規劃可能不含物件碰撞")
                return False

            rospy.loginfo(
                f"[collision mesh] ✅ {col_name} 已確認 attach 到右手"
                f"（touch_links: right gripper all + left fingertip only）")
            return True
        except Exception as e:
            rospy.logwarn(f"[collision mesh] attach 失敗: {e}")
            return False

    def _remove_collision_mesh(self, object_name):
        """從 MoveIt 場景移除物件碰撞 mesh。"""
        col_name = object_name + "_col"
        try:
            self.scene.remove_attached_object("rightarm_wrist_3_link", name=col_name)
            self.scene.remove_world_object(col_name)
            self._wait_for_scene_update(col_name, expect_attached=False)
            rospy.loginfo(f"[collision mesh] ✅ {col_name} 已確認從規劃場景移除")
        except Exception as e:
            rospy.logwarn(f"[collision mesh] 移除失敗: {e}")

    def execute_mission(self):
        user_input = input("請描述你想要的物件：").strip()

        # 重置每次實驗的 metrics 與狀態
        self.metric_mission_start  = rospy.Time.now()
        self.metric_inference_time = None
        self.metric_grasp_time     = None
        self.metric_rotation_angle = None
        self.metric_hoe_value      = None
        self.metric_hoe_label      = None
        self.metric_affordance_ratio = None
        self.metric_affordance_hit   = None
        self.latest_fp_pose_cam    = None
        self.resolved_object_name  = None

        # =========================================================
        # 階段一：視覺偵測，取得右手夾取姿態（brain 同時解析意圖）
        # =========================================================
        ranked_groups = self.trigger_full_detection("", mode="dual", user_input=user_input)
        if ranked_groups is None:
            return
        TARGET_OBJECT_NAME = self.resolved_object_name
        if not TARGET_OBJECT_NAME:
            rospy.logerr("❌ 無法取得解析後的物件名稱，任務中止")
            return
        rospy.loginfo(f"🎯 目標物件確定為：{TARGET_OBJECT_NAME}")
        self._remove_collision_mesh(TARGET_OBJECT_NAME)  # 清除上次殘留
        self.metric_inference_time = (rospy.Time.now() - self.metric_mission_start).to_sec()

        # =========================================================
        # 階段二：右手夾取（開環，只試一次右手姿態）
        # 改成只試前 N 組，不是無限試錯
        # =========================================================
        grasp_success = False
        pose_wrist_grasp_final = None
        pose_fingertip_final = None
        pose_wrist_pre_final = None
        pose_right_target_final = None
        pre_grasp_yaw_deg = None

        MAX_RIGHT_ATTEMPTS = 8  # 最多試 3 組右手姿態
        self.operation_start_time = rospy.Time.now()

        for i, group in enumerate(ranked_groups[:MAX_RIGHT_ATTEMPTS]):
            rospy.loginfo(f"右手嘗試第 #{i+1} 組姿態")
            
            pose_R_camera = self.dict_to_pose(group['pose_R_table'])
            pose_right_target = self.transform_single_pose(pose_R_camera)
            if pose_right_target is None:
                continue

            pose_fingertip, pose_wrist_pre, pose_wrist_grasp = \
                self.calculate_grasp_targets(pose_right_target)
            self.right_grasp_center_z = pose_fingertip.position.z

            # 抓取前把物件加入場景，讓 pre-grasp 路徑規劃繞開它而不撞倒
            self._add_world_collision_mesh(TARGET_OBJECT_NAME)

            # Pre-grasp：先規劃，失敗直接換下一組
            self.move_group.set_pose_target(pose_wrist_pre)
            plan_result = self.move_group.plan()
            if not plan_result[0]:
                self.move_group.clear_pose_targets()
                rospy.logwarn("Pre-grasp 規劃失敗，換下一組")
                continue

            # 規劃左手到待命位置，與右手 Pre-grasp 同步執行
            self.left_move_group.set_joint_value_target(self.left_standby_joints)
            plan_standby = self.left_move_group.plan()
            self.left_move_group.clear_pose_targets()

            if plan_standby[0]:
                rospy.loginfo("✅ 同步執行右手 Pre-grasp + 左手待命")
                goal_r = FollowJointTrajectoryGoal()
                goal_r.trajectory = plan_result[1].joint_trajectory
                goal_l = FollowJointTrajectoryGoal()
                goal_l.trajectory = plan_standby[1].joint_trajectory
                self.right_traj_client.send_goal(goal_r)
                self.left_traj_client.send_goal(goal_l)
                self.right_traj_client.wait_for_result()
                self.left_traj_client.wait_for_result()
                self.move_group.stop()
                self.left_move_group.stop()
                rospy.loginfo("✅ 同步移動完成")
            else:
                rospy.logwarn("⚠️ 左手待命規劃失敗，序列執行右手 Pre-grasp")
                if not self.move_group.execute(plan_result[1], wait=True):
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                    rospy.logwarn("Pre-grasp 執行失敗，安全退出")
                    self.safe_retreat(pose_wrist_pre, arm="right")
                    self.go_home("right")
                    continue
                self.move_group.stop()
                self.move_group.clear_pose_targets()

            # Approach 前移除障礙物（終點在物件表面，需允許進入）
            self.scene.remove_world_object(TARGET_OBJECT_NAME + "_col")

            # Approach
            (plan_app, fraction) = self.move_group.compute_cartesian_path(
                [pose_wrist_grasp], 0.01, True)
            if fraction < 0.5 or not self.move_group.execute(plan_app, wait=True):
                self.move_group.stop()
                self.safe_retreat(pose_wrist_grasp, arm="right")
                self.go_home("right")
                continue

            self.move_group.stop()
            
            # ── 精度診斷 ──
            rospy.sleep(0.3)
            try:
                tw = self.tf_buffer.lookup_transform(
                    "world", "rightarm_wrist_3_link",
                    rospy.Time(0), rospy.Duration(1.0))
                tf = self.tf_buffer.lookup_transform(
                    "world", "rightarm_robotiq_85_left_finger_link",
                    rospy.Time(0), rospy.Duration(1.0))
                tw_t, tf_t = tw.transform.translation, tf.transform.translation
                rospy.loginfo(
                    f"📍 [右手] 實際手腕:  x={tw_t.x:.4f}, y={tw_t.y:.4f}, z={tw_t.z:.4f}")
                rospy.loginfo(
                    f"📍 [右手] 實際指尖:  x={tf_t.x:.4f}, y={tf_t.y:.4f}, z={tf_t.z:.4f}")
                rospy.loginfo(
                    f"🎯 [右手] 期望指尖:  x={pose_fingertip.position.x:.4f}, "
                    f"y={pose_fingertip.position.y:.4f}, z={pose_fingertip.position.z:.4f}")
                rospy.loginfo(
                    f"📏 [右手] 指尖誤差:  dx={tf_t.x-pose_fingertip.position.x:.4f}, "
                    f"dy={tf_t.y-pose_fingertip.position.y:.4f}, "
                    f"dz={tf_t.z-pose_fingertip.position.z:.4f}")
            except Exception as e:
                rospy.logwarn(f"診斷失敗: {e}")
            # ── 診斷結束 ──
            
            # 夾取
            self.control_gripper(0.6)
            rospy.sleep(1.0)
            self.attach_object(TARGET_OBJECT_NAME, arm="right")

            # 記錄實際夾爪位置，算物件中心相對夾爪的偏移
            try:
                trans_grip = self.tf_buffer.lookup_transform(
                    "world", "rightarm_robotiq_85_base_link",
                    rospy.Time(0), rospy.Duration(1.0))
                t = trans_grip.transform.translation
                self.gc_world_actual = np.array([t.x, t.y, t.z])
                if self.object_centroid_for_pca is not None:
                    self.object_centroid_offset = (
                        self.object_centroid_for_pca - self.gc_world_actual)
                    rospy.loginfo(
                        f"📍 夾爪位置: {self.gc_world_actual.round(3)}")
                    rospy.loginfo(
                        f"📍 物件中心偏移: {self.object_centroid_offset.round(3)}")
            except Exception as e:
                rospy.logwarn(f"⚠️ 無法取得夾爪 TF: {e}")
                self.gc_world_actual = None
                self.object_centroid_offset = None

            # 舉起
            pose_lift = deepcopy(pose_wrist_grasp)
            pose_lift.position.z += 0.12
            (plan_lift, frac_lift) = self.move_group.compute_cartesian_path(
                [pose_lift], 0.01, True)
            if frac_lift < 0.9 or not self.move_group.execute(plan_lift, wait=True):
                self.move_group.stop()
                self.control_gripper(0.0)
                self.detach_object(TARGET_OBJECT_NAME, arm="right")
                self.safe_retreat(pose_wrist_grasp, arm="right")
                self.go_home("right")
                continue

            self.move_group.stop()
            pose_right_target_final = pose_right_target
            grasp_success = True
            pose_wrist_grasp_final = pose_wrist_grasp
            pose_fingertip_final = pose_fingertip
            pose_wrist_pre_final = pose_wrist_pre
            if self.operation_start_time is not None:
                self.metric_grasp_time = (rospy.Time.now() - self.operation_start_time).to_sec()
            break

        if not grasp_success:
            rospy.logerr("右手夾取失敗，任務中止")
            return

        # =========================================================
        # 階段三：計算旋轉角度並移動到交接區（途中旋轉）
        # =========================================================
        rospy.loginfo("計算物件旋轉角度...")

        # 取得接收臂基座位置
        try:
            trans = self.tf_buffer.lookup_transform(
                "world", "leftarm_base_link", rospy.Time(0), rospy.Duration(1.0))
            receiver_base = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except Exception as e:
            rospy.logwarn(f"無法取得接收臂位置，使用預設值: {e}")
            receiver_base = np.array([0.775, -0.4, 0.69])

        # 物件當前位置（從右手夾取位置估算）
        object_pos = np.array([
            pose_wrist_grasp_final.position.x,
            pose_wrist_grasp_final.position.y,
            pose_wrist_grasp_final.position.z
        ])

        rotation_angle = self.calculate_rotation_angle_from_pointcloud(
            self.object_points_for_pca,
            receiver_base,
            object_pos
        )

        # 記錄旋轉前的 Gazebo yaw（供非功能端量測：基準 = pre_yaw + rotation_angle）
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
            _gs = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            _r = _gs(TARGET_OBJECT_NAME, 'world')
            _q = _r.pose.orientation
            _siny = 2.0 * (_q.w * _q.z + _q.x * _q.y)
            _cosy = 1.0 - 2.0 * (_q.y * _q.y + _q.z * _q.z)
            pre_grasp_yaw_deg = math.degrees(math.atan2(_siny, _cosy))
            self.metric_rotation_angle = rotation_angle
            rospy.loginfo(f"[HOE] pre-grasp yaw={pre_grasp_yaw_deg:.1f}°  rot={rotation_angle:.1f}°")
        except Exception as _e:
            rospy.logwarn(f"[HOE] 無法取得旋轉前 yaw: {_e}")

        # 先旋轉（原地）
        if abs(rotation_angle) > 5.0:
            rospy.loginfo(f"執行手腕旋轉 {rotation_angle:.1f}°...")
            self.rotate_wrist(rotation_angle, arm="right")
            rospy.sleep(0.5)
            # 旋轉後同步更新 offset：物件隨夾爪轉動，世界座標下的偏移向量也跟著旋轉
            if self.object_centroid_offset is not None:
                a = np.radians(rotation_angle)
                Rz = np.array([[np.cos(a), -np.sin(a), 0],
                               [np.sin(a),  np.cos(a), 0],
                               [0,          0,          1]])
                self.object_centroid_offset = Rz @ self.object_centroid_offset
                rospy.loginfo(
                    f"📍 旋轉後更新 offset: {self.object_centroid_offset.round(3)}")

        # 旋轉後再移動到交接區（同時左手移動到待命位置）
        current_pose_after_rotation = self.move_group.get_current_pose().pose
        HANDOVER_POSITION = self.calculate_handover_position(
            pose_wrist_grasp_final, current_pose_after_rotation)

        # 規劃右手軌跡（先試 cartesian path，失敗則 fallback 到 joint space）
        (plan_handover, frac_handover) = self.move_group.compute_cartesian_path(
            [HANDOVER_POSITION], 0.01, True)
        if frac_handover < 0.9:
            rospy.logwarn("⚠️ cartesian path 失敗，改用 joint space 規劃...")
            self.move_group.set_pose_target(HANDOVER_POSITION)
            plan_result = self.move_group.plan()
            self.move_group.clear_pose_targets()
            if plan_result[0]:
                plan_handover = plan_result[1]
                frac_handover = 1.0
                rospy.loginfo("✅ joint space fallback 規劃成功")
            else:
                rospy.logwarn("移動到交接區失敗，放下物件")
                self.control_gripper(0.0)
                self.detach_object(TARGET_OBJECT_NAME, arm="right")
                self.go_home("right")
                return

        # 規劃左手到待命位置的軌跡
        self.left_move_group.set_joint_value_target(self.left_standby_joints)
        plan_standby = self.left_move_group.plan()
        self.left_move_group.clear_pose_targets()

        # 靜態碰撞檢測：確認兩個終點位置不會互相侵入對方工作空間
        right_end_x = HANDOVER_POSITION.position.x
        left_end_joints = self.left_standby_joints
        self.left_move_group.set_joint_value_target(left_end_joints)
        left_end_pose = self.left_move_group.get_current_pose().pose
        self.left_move_group.clear_pose_targets()

        collision_safe = (right_end_x > 0.55) and \
                        plan_standby[0] and \
                        (frac_handover >= 0.9)

        if collision_safe:
            rospy.loginfo("✅ 靜態碰撞檢測通過，同步執行兩臂移動")

            goal_r = FollowJointTrajectoryGoal()
            goal_r.trajectory = plan_handover.joint_trajectory

            goal_l = FollowJointTrajectoryGoal()
            goal_l.trajectory = plan_standby[1].joint_trajectory

            self.right_traj_client.send_goal(goal_r)
            self.left_traj_client.send_goal(goal_l)

            self.right_traj_client.wait_for_result()
            self.left_traj_client.wait_for_result()

            rospy.loginfo(
                f"右手結果: {self.right_traj_client.get_state()}, "
                f"左手結果: {self.left_traj_client.get_state()}")
        else:
            rospy.logwarn("⚠️ 靜態碰撞檢測未通過，改為序列執行")
            if not self.move_group.execute(plan_handover, wait=True):
                self.move_group.stop()
                rospy.logwarn("移動到交接區失敗，放下物件")
                self.control_gripper(0.0)
                self.detach_object(TARGET_OBJECT_NAME, arm="right")
                self.go_home("right")
                return

        self.move_group.stop()
        self.left_move_group.stop()

        # =========================================================
        # 量測：比較實際物件 yaw 與理想 yaw
        # =========================================================
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=2.0)
            get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = get_model_state(TARGET_OBJECT_NAME, 'world')
            q = resp.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            actual_yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))

            if TARGET_OBJECT_NAME in self.IDEAL_YAW:
                ideal_yaw_deg = self.IDEAL_YAW[TARGET_OBJECT_NAME]
                error_deg = (actual_yaw_deg - ideal_yaw_deg + 180.0) % 360.0 - 180.0
                self.metric_hoe_value = error_deg
                self.metric_hoe_label = "HOE_func"
                rospy.loginfo(
                    f"[HOE] {TARGET_OBJECT_NAME} (functional): "
                    f"actual={actual_yaw_deg:.1f}°  ideal={ideal_yaw_deg:.1f}°  "
                    f"HOE_func={error_deg:+.1f}°"
                )
            elif pre_grasp_yaw_deg is not None:
                target_yaw_deg = (pre_grasp_yaw_deg + rotation_angle + 180.0) % 360.0 - 180.0
                error_deg = (actual_yaw_deg - target_yaw_deg + 180.0) % 360.0 - 180.0
                self.metric_hoe_value = error_deg
                self.metric_hoe_label = "HOE_exec"
                rospy.loginfo(
                    f"[HOE] {TARGET_OBJECT_NAME} (non-functional): "
                    f"actual={actual_yaw_deg:.1f}°  target={target_yaw_deg:.1f}°  "
                    f"HOE_exec={error_deg:+.1f}°  "
                    f"(pre={pre_grasp_yaw_deg:.1f}° + rot={rotation_angle:.1f}°)"
                )
            else:
                rospy.loginfo(
                    f"[HOE] {TARGET_OBJECT_NAME}: actual={actual_yaw_deg:.1f}° (no reference available)"
                )
        except Exception as e:
            rospy.logwarn(f"[HOE] 無法查詢 Gazebo 物件姿態: {e}")

        # =========================================================
        # 階段四：觸發 receiver_only 重偵測，確認旋轉後的接取姿態
        # =========================================================
        rospy.loginfo("到達交接區，觸發 receiver_only 重偵測...")
        _t_recv_infer = rospy.Time.now()
        receiver_groups = self.trigger_full_detection(
            TARGET_OBJECT_NAME,
            mode="receiver_only",
            rotation_angle=rotation_angle)
        _recv_infer_elapsed = (rospy.Time.now() - _t_recv_infer).to_sec()
        if self.metric_inference_time is not None:
            self.metric_inference_time += _recv_infer_elapsed
        else:
            self.metric_inference_time = _recv_infer_elapsed

        if receiver_groups is not None and len(receiver_groups) > 0:
            rospy.loginfo("receiver_only 偵測成功，嘗試空中交接")
            self._attach_collision_mesh(TARGET_OBJECT_NAME)
            success = self.execute_air_handover(
                receiver_groups,
                TARGET_OBJECT_NAME,
                HANDOVER_POSITION,
                pose_right_target_final,
                use_direct_pose=True
            )
            if success:
                return
            rospy.logwarn("空中交接失敗，改走放下重抓流程")
        else:
            rospy.logwarn("receiver_only 偵測失敗，改走放下重抓流程")

        # =========================================================
        # 階段五：空中交接失敗或不適合，放下物件讓左手重新夾取
        # =========================================================
        rospy.loginfo("判定：改走放下重抓流程")
        
        # 取得右手當前末端位置
        current_right_pose = self.move_group.get_current_pose().pose

        # 右手放下物件
        TABLE_HEIGHT = 0.68
        OBJECT_HEIGHT_APPROX = 0.05
        pose_put_down = deepcopy(current_right_pose)
        pose_put_down.position.z = TABLE_HEIGHT + OBJECT_HEIGHT_APPROX

        (plan_down, frac_down) = self.move_group.compute_cartesian_path(
            [pose_put_down], 0.01, True)
        
        self._remove_collision_mesh(TARGET_OBJECT_NAME)
        if frac_down > 0.5:
            self.move_group.execute(plan_down, wait=True)
            self.move_group.stop()
            self.control_gripper(0.0, arm="right")
            rospy.sleep(1.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="right")
            self.go_home("right", pose_near_object=pose_put_down)
        else:
            # 規劃失敗代表下降路徑有障礙，原地釋放
            rospy.logwarn("無法垂直下降放物件，原地釋放")
            self.control_gripper(0.0, arm="right")
            rospy.sleep(1.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="right")
            self.go_home("right")  # 原地釋放後直接回 Home

        rospy.loginfo("右手已退開，觸發左手重新偵測")
        
        rospy.loginfo("重新觸發 left_only 偵測，讓左手從桌面重新夾取")
        _t_left_infer = rospy.Time.now()
        left_groups = self.trigger_full_detection(TARGET_OBJECT_NAME, mode="left_only")
        _left_infer_elapsed = (rospy.Time.now() - _t_left_infer).to_sec()
        if self.metric_inference_time is not None:
            self.metric_inference_time += _left_infer_elapsed
        else:
            self.metric_inference_time = _left_infer_elapsed
        if left_groups is None:
            rospy.logerr("接收臂重新偵測失敗，任務中止")
            return
        self.execute_left_standalone_grasp(left_groups, TARGET_OBJECT_NAME)

if __name__ == '__main__':
    try:
        controller = SimpleGraspController()
        controller.execute_mission()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()