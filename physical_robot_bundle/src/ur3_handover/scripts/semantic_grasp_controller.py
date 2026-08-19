#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Grasp Controller (AnyGrasp 6D Pose 接收版)
"""

import json
import select
import os, sys, math, numpy as np
import time
import traceback
import yaml
# 確保從 scripts 目錄直接載入，避免 catkin relay exec() 的 import 問題
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_from_euler, quaternion_from_matrix, euler_from_quaternion
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, roscpp_initialize
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
import tf2_ros, tf2_geometry_msgs

from robotiq_gripper import RobotiqGripper

class SemanticGraspController:
    def __init__(self):
        # ---- 1. 參數 ----
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ws_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        default_config_path = os.path.join(ws_root, "src", "ur3_handover", "config", "handover_params.yaml")

        self.base_frame  = self._get_param("~base_frame", None, "base_link")
        self.move_group  = self._get_param("~move_group", None, "manipulator")
        self.use_grasp_metadata = bool(self._get_param("~use_grasp_metadata", None, True))
        self.prefer_camera_facing_side = bool(
            self._get_param("~prefer_camera_facing_side", None, True))
        self.use_server_depth_for_offset = bool(
            self._get_param("~use_server_depth_for_offset", None, False))
        self.prefer_nearest_ik = bool(self._get_param("~prefer_nearest_ik", None, True))

        self.tcp_offset    = float(self._get_param("~tcp_offset", None, 0.174))
        self.grasp_depth   = float(self._get_param("~grasp_depth", None, 0.09))
        self.approach_dist = float(self._get_param("~approach_dist", None, 0.08))
        self.grasp_candidate_topk = int(self._get_param(
            "~grasp_candidate_topk", "/semantic_handover/grasp_candidate_topk", 5))
        self.pca_rotation_enabled = bool(self._get_param("~pca_rotation_enabled", None, False))
        self.pca_strategy = str(self._get_param("~pca_strategy", None, "functional_end"))
        self.pca_min_angle_deg = float(self._get_param("~pca_min_angle_deg", None, 15.0))
        self._latest_object_points_raw = None
        # 側面抓取自動轉俯抓
        self.force_top_down_grasp       = bool(self._get_param("~force_top_down_grasp", None, False))
        self.top_down_grasp_threshold   = float(self._get_param("~top_down_grasp_threshold", None, 0.5))
        # 交接區規劃失敗時以預設方向 fallback 重試
        self.handover_fallback_orientation = bool(
            self._get_param("~handover_fallback_orientation", None, True))

        self.retreat_up_height = float(self._get_param("~retreat_up_height", None, 0.14))  # 待機點高度

        # 計時紀錄：Phase1（client/server 感知）+ Phase2（抓取執行）
        self._phase1_timing = None       # 來自 client 的 Phase1 計時 dict
        self._phase2_start_time = None   # Phase2 開始時間 (time.perf_counter())
        self._phase2_release_done = False
        self._planning_total_s = None    # Step A 路徑規劃總耗時（不含人為等待）
        self._human_confirm_s = None     # Step A 人為確認等待總時間

        self.eef_step  = float(self._get_param("~eef_step", None, 0.02))
        self.vel_scale = float(self._get_param("~vel_scale", None, 0.10))
        self.acc_scale = float(self._get_param("~acc_scale", None, 0.10))

        legacy_dynamic_handover = bool(self._get_param(
            "~use_dynamic_handover", "/semantic_handover/use_dynamic_handover", True))
        self.enable_handover_fine_adjust = bool(self._get_param(
            "~enable_handover_fine_adjust",
            "/semantic_handover/enable_handover_fine_adjust",
            legacy_dynamic_handover))
        # Backward-compatible alias: old use_dynamic_handover now means optional
        # fine adjustment after reaching the default handover pose.
        self.use_dynamic_handover = self.enable_handover_fine_adjust
        # 固定交接點（物體位置，即指尖/TCP 位置，非 tool0）
        self.handover_object_xyz = self._get_param(
            "~handover_object_xyz", None, [0.1606, 0.2881, 0.1554])
        default_handover_pose = self._get_param(
            "~default_handover_pose",
            "/semantic_handover/default_handover_pose",
            {"xyz": [0.1606, 0.2881, 0.1554], "rpy_deg": [180.0, 0.0, 0.0]})
        self.default_handover_pose_xyz, self.default_handover_pose_rpy_deg = \
            self._parse_pose_config(
                default_handover_pose,
                fallback_xyz=self.handover_object_xyz,
                fallback_rpy=[180.0, 0.0, 0.0])
        self.handover_stop_at_zone_entry = bool(self._get_param(
            "~handover_stop_at_zone_entry",
            "/semantic_handover/handover_stop_at_zone_entry",
            True))
        self.handover_zone_center_xyz = np.array(
            self._get_param(
                "~handover_zone_center_xyz",
                "/semantic_handover/handover_zone_center_xyz",
                self.default_handover_pose_xyz.tolist()),
            dtype=float)
        self.handover_zone_half_extents_xyz = np.array(
            self._get_param(
                "~handover_zone_half_extents_xyz",
                "/semantic_handover/handover_zone_half_extents_xyz",
                [0.15, 0.20, 0.08]),
            dtype=float)
        self.handover_zone_yaw_deg = float(self._get_param(
            "~handover_zone_yaw_deg",
            "/semantic_handover/handover_zone_yaw_deg",
            0.0))
        self.handover_zone_entry_margin = float(self._get_param(
            "~handover_zone_entry_margin",
            "/semantic_handover/handover_zone_entry_margin",
            0.02))
        self._update_handover_zone_rotation()
        self.handover_object_offset_xyz = np.array(
            self._get_param(
                "~handover_object_offset_xyz",
                "/semantic_handover/handover_object_offset_xyz",
                [0.0, 0.0, 0.0]),
            dtype=float)
        self.handover_keepout_radius = float(self._get_param(
            "~handover_keepout_radius",
            "/semantic_handover/handover_keepout_radius",
            0.15))
        self.handover_distance_from_hand = float(self._get_param(
            "~handover_distance_from_hand",
            "/semantic_handover/handover_distance_from_hand",
            0.20))
        legacy_max_adjust = float(self._get_param(
            "~handover_fine_adjust_max_translation",
            "/semantic_handover/handover_fine_adjust_max_translation",
            0.12))
        self.handover_max_adjust = float(self._get_param(
            "~handover_max_adjust",
            "/semantic_handover/handover_max_adjust",
            legacy_max_adjust))
        self.handover_min_hand_distance = float(self._get_param(
            "~handover_min_hand_distance",
            "/semantic_handover/handover_min_hand_distance",
            0.12))
        self.handover_max_hand_distance = float(self._get_param(
            "~handover_max_hand_distance",
            "/semantic_handover/handover_max_hand_distance",
            0.50))
        self.handover_z_min = float(self._get_param(
            "~handover_z_min",
            "/semantic_handover/handover_z_min",
            0.15))
        self.handover_z_max = float(self._get_param(
            "~handover_z_max",
            "/semantic_handover/handover_z_max",
            0.45))
        self.handover_fine_adjust_vel_scale = float(self._get_param(
            "~handover_fine_adjust_vel_scale",
            "/semantic_handover/handover_fine_adjust_vel_scale",
            0.05))
        self.handover_fine_adjust_acc_scale = float(self._get_param(
            "~handover_fine_adjust_acc_scale",
            "/semantic_handover/handover_fine_adjust_acc_scale",
            0.05))
        # 待機點（tool0 位置，固定姿態）
        self.ready_xyz = self._get_param("~ready_xyz", None, [-0.2456, 0.2653, 0.2889])
        self.hand_state_topic = self._get_param(
            "~hand_state_topic", "/semantic_handover/hand_state_topic",
            "/semantic_handover/hand_state")
        self.user_cmd_topic = self._get_param(
            "~user_cmd_topic", "/semantic_handover/user_cmd_topic",
            "/semantic_handover/user_cmd")
        self.safety_marker_topic = self._get_param(
            "~safety_marker_topic",
            "/semantic_handover/safety_marker_topic",
            "/semantic_handover/safety_markers")
        self.safety_marker_enabled = bool(self._get_param(
            "~safety_marker_enabled",
            "/semantic_handover/safety_marker_enabled",
            True))
        self.hand_state_stability_threshold = float(
            self._get_param(
                "~hand_state_stability_threshold",
                "/semantic_handover/hand_state_stability_threshold",
                0.85))
        self.hand_state_max_age = float(self._get_param("~hand_state_max_age", None, 0.50))
        self.handover_hand_timeout = float(self._get_param(
            "~handover_hand_timeout", "/semantic_handover/handover_hand_timeout", 10.0))
        self.handover_timeout = float(self._get_param(
            "~handover_timeout", "/semantic_handover/release_timeout", 300.0))
        self.handover_tracking_enabled = bool(self._get_param(
            "~handover_tracking_enabled",
            "/semantic_handover/handover_tracking_enabled",
            False))
        self.handover_tracking_hz = float(self._get_param(
            "~handover_tracking_hz",
            "/semantic_handover/handover_tracking_hz",
            2.0))
        self.handover_replan_interval = float(self._get_param(
            "~handover_replan_interval",
            "/semantic_handover/handover_replan_interval",
            1.0))
        self.handover_replan_min_delta = float(self._get_param(
            "~handover_replan_min_delta",
            "/semantic_handover/handover_replan_min_delta",
            0.03))
        self.handover_track_max_step = float(self._get_param(
            "~handover_track_max_step",
            "/semantic_handover/handover_track_max_step",
            0.05))
        self.handover_tracking_settle_time = float(self._get_param(
            "~handover_tracking_settle_time",
            "/semantic_handover/handover_tracking_settle_time",
            0.5))
        self.handover_config_path = self._get_param(
            "~handover_config_path", None, default_config_path)
        self._handover_config_mtime = None
        self._latest_hand_state = None
        self._abort_requested = False
        self._release_requested = False
        self._active_handover_orientation = None
        # 交接物件姿態 3D 誤差指標狀態
        self._pca_axis_at_grasp = None            # np (3,) 完整 3D 單位向量, base frame
        self._grasp_orientation = None            # geometry_msgs/Quaternion, PCA 前的抓取方向
        self._observed_handover_axis = None       # np (3,) camera frame, 由 topic 填入
        self._observed_handover_frame = None      # str, 觀測軸的座標系
        self._observed_handover_eig_ratio = None  # float, λ1/λ2 主軸明確度
        self._handover_orientation_error_deg = None
        self.table_collision_enabled = bool(self._get_param("~table_collision_enabled", None, True))
        self.table_collision_name = self._get_param("~table_collision_name", None, "safety_table")
        self.table_collision_size = tuple(float(v) for v in self._get_param(
            "~table_collision_size", None, [1.5, 1.5, 0.01]))
        self.table_collision_z = float(self._get_param("~table_collision_z", None, -0.05))
        self.pause_after_grasp = bool(self._get_param("~pause_after_grasp", None, False))

        # 夾爪
        self.grip_ip    = rospy.get_param("~gripper_ip",   "192.168.86.7")
        self.grip_port  = int(rospy.get_param("~gripper_port", 63352))
        self.grip_speed = 100
        self.grip_force = 100

        # ---- 2. 初始化 ----
        roscpp_initialize([])
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander(self.move_group)
        self.group.set_max_velocity_scaling_factor(self.vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.acc_scale)
        self.group.set_planner_id("RRTConnect")
        self.group.set_num_planning_attempts(1)  # 由 joint_plan_execute 控制多次
        self.group.set_planning_time(5.0)        # 每次 5 秒，跑 3 次取最短

        self.g = RobotiqGripper()
        self.init_gripper()

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tflis = tf2_ros.TransformListener(self.tfbuf)
        self.ik_service_name = self._get_param("~ik_service", None, "/compute_ik")
        self.ik_srv = None
        if self.prefer_nearest_ik:
            try:
                rospy.wait_for_service(self.ik_service_name, timeout=2.0)
                self.ik_srv = rospy.ServiceProxy(self.ik_service_name, GetPositionIK)
                rospy.loginfo("[p2p] 啟用最近 IK 分支規劃: %s", self.ik_service_name)
            except (rospy.ROSException, rospy.ROSInterruptException):
                rospy.logwarn("[p2p] 找不到 IK service %s，退回 pose target 規劃",
                              self.ik_service_name)

        self.target_pose_topic = self._get_param(
            "~target_pose_topic", None, "/anygrasp/target_pose")
        self.target_grasp_topic = self._get_param(
            "~target_grasp_topic", None, "/anygrasp/target_grasp")

        # 訂閱 AnyGrasp 姿態 / 完整 grasp metadata
        if self.use_grasp_metadata:
            self.grasp_sub = rospy.Subscriber(
                self.target_grasp_topic, String, self.cb_anygrasp_grasp, queue_size=1)
            rospy.loginfo("[p2p] 使用完整 grasp metadata: %s", self.target_grasp_topic)
        else:
            self.pose_sub = rospy.Subscriber(
                self.target_pose_topic, PoseStamped, self.cb_anygrasp_pose, queue_size=1)
            rospy.loginfo("[p2p] 使用 legacy PoseStamped: %s", self.target_pose_topic)
        # 隨時回待機點的指令
        rospy.Subscriber("/semantic_grasp/go_home", String, self._cb_go_home)
        rospy.Subscriber(self.hand_state_topic, String, self._hand_state_cb, queue_size=1)
        rospy.Subscriber(self.user_cmd_topic, String, self._user_cmd_cb, queue_size=1)
        rospy.Subscriber('/anygrasp/object_points', String, self._cb_object_points, queue_size=1)
        self.safety_marker_pub = rospy.Publisher(
            self.safety_marker_topic, MarkerArray, queue_size=1, latch=True)

        # 交接物件姿態 3D 誤差指標：到位時請求 server 重觀測物件 PCA 主軸
        self.handover_pca_request_pub = rospy.Publisher(
            '/handover/request_pca', String, queue_size=1)
        rospy.Subscriber('/handover/observed_pca', String,
                         self._cb_observed_handover_pca, queue_size=1)
        self.handover_pca_timeout_s = self._get_param(
            "~handover_pca_timeout_s", None, 5.0)

        rospy.loginfo("[p2p] 啟動完成，等待目標姿態...")
        rospy.loginfo("[p2p] 隨時回待機點: rostopic pub /semantic_grasp/go_home std_msgs/String 'go' -1")
        rospy.loginfo(
            "[p2p] handover fine_adjust=%s, hand_state=%s, user_cmd=%s",
            self.enable_handover_fine_adjust,
            self.hand_state_topic,
            self.user_cmd_topic)
        self._ensure_virtual_table()
        self._reload_handover_config_if_needed(force=True)
        self._publish_handover_safety_markers()

    # ------------------------------------------------------------------ #
    #  ROS callbacks                                                       #
    # ------------------------------------------------------------------ #
    def _cb_object_points(self, msg):
        try:
            data = json.loads(msg.data)
            pts = np.array(data["points"], dtype=float)
            if len(pts) >= 10:
                entry = {
                    "frame_id": data.get("frame_id", "camera_color_optical_frame"),
                    "stamp": float(data.get("stamp", 0.0)),
                    "points": pts,
                    "grip_end_3d": None,
                }
                if data.get("grip_end_3d") is not None:
                    entry["grip_end_3d"] = np.array(data["grip_end_3d"], dtype=float)
                    rospy.loginfo("[pca] 收到握持端 3D (camera): (%.3f, %.3f, %.3f)",
                                  *entry["grip_end_3d"])
                self._latest_object_points_raw = entry
        except Exception as e:
            rospy.logwarn("[pca] object_points 解析失敗: %s", e)

    def _cb_observed_handover_pca(self, msg):
        """server 重觀測回傳的交接時物件主軸（camera frame）。"""
        try:
            data = json.loads(msg.data)
            if not data.get("ok", True):
                rospy.logwarn("[metric] server 重觀測失敗: %s", data.get("error", "unknown"))
                return
            axis = np.array(data["axis"], dtype=float)
            n = np.linalg.norm(axis)
            if n < 1e-9:
                rospy.logwarn("[metric] 觀測主軸為零向量，忽略")
                return
            self._observed_handover_axis = axis / n
            self._observed_handover_frame = data.get("frame_id", "camera_color_optical_frame")
            self._observed_handover_eig_ratio = data.get("eig_ratio")
            rospy.loginfo("[metric] 收到交接觀測主軸 (%s): (%.3f, %.3f, %.3f) n_points=%s eig_ratio=%s",
                          self._observed_handover_frame,
                          self._observed_handover_axis[0],
                          self._observed_handover_axis[1],
                          self._observed_handover_axis[2],
                          data.get("n_points"), data.get("eig_ratio"))
        except Exception as e:
            rospy.logwarn("[metric] observed_pca 解析失敗: %s", e)

    def _transform_points_to_base(self, points_camera, frame_id):
        """N×3 camera frame → N×3 base_frame"""
        try:
            T = self.tfbuf.lookup_transform(
                self.base_frame, frame_id, rospy.Time(0), rospy.Duration(1.0))
            q = T.transform.rotation
            t = T.transform.translation
            M = quaternion_matrix([q.x, q.y, q.z, q.w])
            M[:3, 3] = [t.x, t.y, t.z]
            pts_h = np.hstack([points_camera, np.ones((len(points_camera), 1))])
            return (M @ pts_h.T).T[:, :3]
        except Exception as e:
            rospy.logwarn("[pca] TF 轉換失敗: %s", e)
            return None

    def _calculate_pca_rotation_deg(self, object_points_base, object_pos, target_xyz,
                                    grip_end_3d_base=None):
        """
        PCA 計算需要繞世界 Z 軸旋轉幾度，使握持端朝向 target_xyz。
        functional_end 策略：使用 grip_end_3d_base 確認主軸指向握持端；
        geometric 策略：找夾爪可握的最細面，主軸對齊目標方向。
        回傳旋轉角度（度），正值=逆時針。
        """
        if object_points_base is None or len(object_points_base) < 10:
            rospy.logwarn("[pca] 點雲不足，跳過")
            return 0.0

        target_dir = np.array(target_xyz, dtype=float) - np.array(object_pos, dtype=float)
        target_dir[2] = 0.0
        norm = np.linalg.norm(target_dir)
        if norm < 1e-6:
            return 0.0
        target_dir /= norm

        centroid = np.mean(object_points_base, axis=0)
        centered = object_points_base - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        if self.pca_strategy == "functional_end":
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)].copy()
            if grip_end_3d_base is not None:
                centroid_to_grip = np.array(grip_end_3d_base, dtype=float) - centroid
                if np.dot(centroid_to_grip, principal_axis) < 0:
                    principal_axis = -principal_axis
                rospy.loginfo("[pca] 使用 server grip_end 確認握持端方向")
            else:
                rospy.logwarn("[pca] functional_end 策略無 grip_end_3d，改用幾何估算")
            current_dir = principal_axis
        else:  # geometric
            gripper_max_width = 0.085
            dims = [(abs((centered @ eigenvectors[:, i]).max() -
                         (centered @ eigenvectors[:, i]).min()),
                     eigenvectors[:, i]) for i in range(3)]
            grippable = [(s, ax) for s, ax in dims if s <= gripper_max_width]
            if grippable:
                grippable.sort(key=lambda x: x[0])
                current_dir = grippable[0][1].copy()
            else:
                dims.sort(key=lambda x: x[0])
                current_dir = dims[0][1].copy()

        current_dir[2] = 0.0
        if np.linalg.norm(current_dir) < 1e-6:
            return 0.0
        current_dir /= np.linalg.norm(current_dir)
        if self.pca_strategy != "functional_end" and np.dot(current_dir, target_dir) < 0:
            current_dir = -current_dir

        cos_a = np.clip(np.dot(current_dir, target_dir), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_a))
        cross_z = float(np.cross(current_dir, target_dir)[2])
        if cross_z < 0:
            angle_deg = -angle_deg

        rospy.loginfo("[pca] 計算旋轉角度: %.1f° (strategy=%s, grip_end=%s)",
                      angle_deg, self.pca_strategy,
                      "server" if grip_end_3d_base is not None else "geometric")
        return angle_deg

    @staticmethod
    def _pca_axis_3d(points_base):
        """回傳點雲的完整 3D 第一主軸（單位向量），不壓平 Z。
        供交接姿態誤差指標使用，能反映上下傾斜。points_base: N×3 numpy。"""
        if points_base is None or len(points_base) < 10:
            return None
        c = points_base - points_base.mean(axis=0)
        try:
            evals, evecs = np.linalg.eigh(np.cov(c.T))
        except np.linalg.LinAlgError:
            return None
        axis = evecs[:, int(np.argmax(evals))]
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return None
        return axis / n

    def _select_reachable_candidate(self, candidates, frame_id):
        """IK 可達性篩選：對每個候選算出 pre-grasp pose 並呼叫 IK service，
        回傳第一個 IK 有解的候選；全部無解或 IK service 不可用時回傳 None。"""
        candidates = candidates[:self.grasp_candidate_topk]
        if self.ik_srv is None:
            rospy.logwarn("[p2p] IK service 不可用，跳過候選可達性篩選")
            return None

        try:
            tf_cam_to_base = self.tfbuf.lookup_transform(
                self.base_frame, frame_id, rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("[p2p] 候選可達性篩選：TF 查詢失敗 (%s)，使用最高分候選", e)
            return None

        tr = tf_cam_to_base.transform.translation
        T_cam_to_base = quaternion_matrix([
            tf_cam_to_base.transform.rotation.x,
            tf_cam_to_base.transform.rotation.y,
            tf_cam_to_base.transform.rotation.z,
            tf_cam_to_base.transform.rotation.w,
        ])
        T_cam_to_base[:3, 3] = [tr.x, tr.y, tr.z]

        for i, c in enumerate(candidates):
            rot_cam = np.array(c["rotation"], dtype=float).reshape(3, 3)
            T_grasp_cam = np.eye(4)
            T_grasp_cam[:3, :3] = rot_cam
            T_grasp_cam[:3, 3]  = np.array(c["translation"], dtype=float)

            T_grasp_base = T_cam_to_base @ T_grasp_cam

            q_orig   = quaternion_from_matrix(T_grasp_base)
            q_mapped = quaternion_multiply(q_orig, quaternion_from_euler(0, -math.pi / 2, 0))

            ps_ref = PoseStamped()
            ps_ref.header.frame_id = self.base_frame
            ps_ref.header.stamp    = rospy.Time.now()
            ps_ref.pose.position.x = T_grasp_base[0, 3]
            ps_ref.pose.position.y = T_grasp_base[1, 3]
            ps_ref.pose.position.z = T_grasp_base[2, 3]
            ps_ref.pose.orientation.x = q_mapped[0]
            ps_ref.pose.orientation.y = q_mapped[1]
            ps_ref.pose.orientation.z = q_mapped[2]
            ps_ref.pose.orientation.w = q_mapped[3]

            _, ps_pre, _ = self.build_robot_grasp_poses(ps_ref, self.grasp_depth)
            ik_result = self.solve_nearest_ik(ps_pre)
            if ik_result is not None:
                rospy.loginfo("[p2p] 候選 #%d (score=%.4f) IK 可達，採用", i, float(c["score"]))
                return c
            rospy.loginfo("[p2p] 候選 #%d (score=%.4f) IK 無解，跳過", i, float(c["score"]))

        rospy.logwarn("[p2p] 所有 %d 個候選 IK 均無解，退回最高分候選", len(candidates))
        return None

    def _candidate_to_ps_base(self, candidate, tf_transform, camera_pos, source_frame_id):
        """Convert a raw candidate dict (camera frame) to a fully-processed PoseStamped (base_link).
        Applies the same orientation pipeline as _process_target_pose."""
        tvec = np.array(candidate["translation"], dtype=float)
        rot_mat = np.array(candidate["rotation"], dtype=float).reshape(3, 3)
        T_mat = np.eye(4)
        T_mat[:3, :3] = rot_mat
        q = quaternion_from_matrix(T_mat)

        msg = PoseStamped()
        msg.header.frame_id = source_frame_id
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = tvec
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]

        ps_base = tf2_geometry_msgs.do_transform_pose(msg, tf_transform)

        object_surface = np.array([ps_base.pose.position.x,
                                    ps_base.pose.position.y,
                                    ps_base.pose.position.z], dtype=float)
        q_orig = [ps_base.pose.orientation.x, ps_base.pose.orientation.y,
                  ps_base.pose.orientation.z, ps_base.pose.orientation.w]

        if self.prefer_camera_facing_side:
            q_final, _ = self.select_camera_facing_orientation(q_orig, object_surface, camera_pos)
        else:
            q_final = quaternion_multiply(q_orig, quaternion_from_euler(0, -math.pi / 2, 0))

        ps_base.pose.orientation.x = q_final[0]
        ps_base.pose.orientation.y = q_final[1]
        ps_base.pose.orientation.z = q_final[2]
        ps_base.pose.orientation.w = q_final[3]

        ee_z = self.get_ee_z_axis_in_base(ps_base)
        if ee_z[2] > 0.7:
            q_fix = quaternion_from_euler(math.pi, 0, 0)
            q_safe = quaternion_multiply(q_final, q_fix)
            ps_base.pose.orientation.x = q_safe[0]
            ps_base.pose.orientation.y = q_safe[1]
            ps_base.pose.orientation.z = q_safe[2]
            ps_base.pose.orientation.w = q_safe[3]
            q_final = q_safe

        ee_z_final = self.get_ee_z_axis_in_base(ps_base)
        if self.force_top_down_grasp and abs(ee_z_final[2]) < self.top_down_grasp_threshold:
            _, _, yaw = euler_from_quaternion([
                ps_base.pose.orientation.x, ps_base.pose.orientation.y,
                ps_base.pose.orientation.z, ps_base.pose.orientation.w])
            q_td = quaternion_from_euler(math.pi, 0.0, yaw)
            ps_base.pose.orientation.x = q_td[0]
            ps_base.pose.orientation.y = q_td[1]
            ps_base.pose.orientation.z = q_td[2]
            ps_base.pose.orientation.w = q_td[3]

        return ps_base

    def cb_anygrasp_grasp(self, msg: String):
        try:
            grasp = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logerr("[p2p] grasp metadata JSON 解析失敗: %s", msg.data)
            return

        self._phase1_timing = grasp.get("timing")

        candidates = grasp.get("grasp_candidates", [])
        frame_id   = grasp.get("frame_id", "camera_color_optical_frame")
        topk = candidates[:self.grasp_candidate_topk]
        if topk:
            grasp = {**grasp, **topk[0]}
            grasp["_fallback_candidates"] = topk[1:]
            rospy.loginfo("[p2p] 最高分候選 score=%.4f，備選 %d 個",
                          float(grasp.get("score", 0.0)), len(topk) - 1)

        required_keys = ("translation", "rotation")
        missing = [key for key in required_keys if key not in grasp]
        if missing:
            rospy.logerr("[p2p] grasp metadata 缺少欄位: %s", ", ".join(missing))
            return

        try:
            tvec = np.array(grasp["translation"], dtype=float)
            rot_mat = np.array(grasp["rotation"], dtype=float).reshape(3, 3)
        except (ValueError, TypeError) as e:
            rospy.logerr("[p2p] grasp metadata 數值格式錯誤: %s", e)
            return

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        stamp_sec = float(grasp.get("stamp", 0.0))
        pose_msg.header.stamp = rospy.Time.from_sec(stamp_sec) if stamp_sec > 0.0 else rospy.Time.now()
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = tvec
        T = np.eye(4)
        T[:3, :3] = rot_mat
        q = quaternion_from_matrix(T)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]

        rospy.loginfo(
            "[p2p] 收到完整 grasp metadata！score=%.4f width=%.3f depth=%.3f",
            float(grasp.get("score", 0.0)),
            float(grasp.get("width", 0.0)),
            float(grasp.get("depth", self.grasp_depth)))
        self._process_target_pose(pose_msg, grasp)

    def cb_anygrasp_pose(self, msg: PoseStamped):
        rospy.loginfo(f"[p2p] 收到目標！Frame: {msg.header.frame_id}")
        self._process_target_pose(msg, None)

    def _process_target_pose(self, msg: PoseStamped, grasp_meta=None):
        if grasp_meta is None:
            rospy.loginfo("[p2p] 使用 legacy pose 模式，grasp_depth=%.3f", self.grasp_depth)
        try:
            T = self.tfbuf.lookup_transform(
                self.base_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            ps_base = tf2_geometry_msgs.do_transform_pose(msg, T)
        except Exception as e:
            rospy.logerr(f"TF 失敗: {e}"); return

        camera_pos = np.array([
            T.transform.translation.x,
            T.transform.translation.y,
            T.transform.translation.z,
        ], dtype=float)
        object_surface = np.array([
            ps_base.pose.position.x,
            ps_base.pose.position.y,
            ps_base.pose.position.z,
        ], dtype=float)

        # 座標軸對齊：預設維持舊版 Y 軸 -90°，必要時才啟用相機側選邊
        q_orig = [ps_base.pose.orientation.x, ps_base.pose.orientation.y,
                  ps_base.pose.orientation.z, ps_base.pose.orientation.w]
        if self.prefer_camera_facing_side:
            q_final, choice_info = self.select_camera_facing_orientation(
                q_orig, object_surface, camera_pos)
        else:
            q_final = quaternion_multiply(q_orig, quaternion_from_euler(0, -math.pi/2, 0))
            choice_info = None

        # --- 診斷：印出 TF 後、q_to_ur3 前的原始接近方向（相機座標系 X 軸）---
        from tf.transformations import quaternion_matrix as qmat
        M_orig = qmat([q_orig[0], q_orig[1], q_orig[2], q_orig[3]])
        approach_cam = M_orig[:3, 0]  # AnyGrasp X 軸 = 接近方向（in base_link after TF）
        rospy.loginfo(f"[p2p] AnyGrasp 接近方向(base) = ({approach_cam[0]:.3f}, {approach_cam[1]:.3f}, {approach_cam[2]:.3f})")
        rospy.loginfo(
            "[p2p] 相機位置(base) = (%.3f, %.3f, %.3f)",
            *camera_pos)
        if choice_info is not None:
            rospy.loginfo(
                "[p2p] camera->object = (%.3f, %.3f, %.3f)",
                *choice_info["camera_to_object"])
            rospy.loginfo(
                "[p2p] 映射候選分數: Y-90=%.3f, Y+90=%.3f，採用 %s",
                choice_info["score_neg90"],
                choice_info["score_pos90"],
                choice_info["label"])
        ps_base.pose.orientation.x, ps_base.pose.orientation.y, \
            ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_final

        # 安全檢查：夾爪明顯朝上（> 0.7）才翻轉，避免誤判側面抓取
        ee_z = self.get_ee_z_axis_in_base(ps_base)
        rospy.loginfo(f"[p2p] 接收到的 ee_z = ({ee_z[0]:.3f}, {ee_z[1]:.3f}, {ee_z[2]:.3f})")
        if ee_z[2] > 0.7:
            rospy.logwarn(f"偵測到明顯倒立姿態 (ee_z[2]={ee_z[2]:.3f})，自動修正方向...")
            q_fix = quaternion_from_euler(math.pi, 0, 0)
            q_safe = quaternion_multiply(q_final, q_fix)
            ps_base.pose.orientation.x, ps_base.pose.orientation.y, \
                ps_base.pose.orientation.z, ps_base.pose.orientation.w = q_safe

        # 側面抓取自動轉俯抓：abs(ee_z[2]) 小表示接近方向水平，規劃容易失敗
        ee_z_final = self.get_ee_z_axis_in_base(ps_base)
        rospy.loginfo("[p2p] 最終 ee_z = (%.3f, %.3f, %.3f)  abs_z=%.3f",
                      *ee_z_final, abs(ee_z_final[2]))
        if self.force_top_down_grasp and abs(ee_z_final[2]) < self.top_down_grasp_threshold:
            _, _, yaw = euler_from_quaternion([
                ps_base.pose.orientation.x, ps_base.pose.orientation.y,
                ps_base.pose.orientation.z, ps_base.pose.orientation.w])
            q_topdown = quaternion_from_euler(math.pi, 0.0, yaw)
            ps_base.pose.orientation.x = q_topdown[0]
            ps_base.pose.orientation.y = q_topdown[1]
            ps_base.pose.orientation.z = q_topdown[2]
            ps_base.pose.orientation.w = q_topdown[3]
            rospy.logwarn("[p2p] 側面抓取（abs_z=%.3f < %.3f），自動轉俯抓 yaw=%.2f°",
                          abs(ee_z_final[2]), self.top_down_grasp_threshold, math.degrees(yaw))

        server_depth = None
        if grasp_meta is not None:
            try:
                server_depth = float(grasp_meta.get("depth", self.grasp_depth))
            except (TypeError, ValueError):
                rospy.logwarn("[p2p] grasp metadata depth 非法，忽略 server depth")
                server_depth = None
            if server_depth is not None and server_depth <= 0.0:
                rospy.logwarn("[p2p] grasp metadata depth=%.3f 無效，忽略 server depth",
                              server_depth)
                server_depth = None
            if server_depth is not None:
                rospy.loginfo("[p2p] 收到 server grasp depth = %.3f", server_depth)

        final_insert_depth = self.resolve_final_insert_depth(server_depth)
        if grasp_meta is not None and grasp_meta.get("from_thickened_cloud"):
            final_insert_depth += 0.02
            rospy.loginfo("[p2p] 點雲已加厚，grasp_depth +2cm → %.3f", final_insert_depth)
        fallback_targets = []
        if grasp_meta is not None:
            src_frame = grasp_meta.get("frame_id", "camera_color_optical_frame")
            for fc in grasp_meta.get("_fallback_candidates", []):
                try:
                    fb_ps = self._candidate_to_ps_base(fc, T, camera_pos, src_frame)
                    fallback_targets.append(fb_ps)
                except Exception as e:
                    rospy.logwarn("[p2p] 備選候選 pose 轉換失敗: %s", e)
        self.run_once(
            ps_base,
            final_insert_depth=final_insert_depth,
            server_depth=server_depth,
            fallback_targets=fallback_targets)

    def _cb_go_home(self, msg):
        rospy.loginfo("[p2p] 收到回待機點指令...")
        self.go_to_ready_pose()

    def _get_param(self, private_name, shared_name, default):
        if private_name and rospy.has_param(private_name):
            return rospy.get_param(private_name)
        if shared_name and rospy.has_param(shared_name):
            return rospy.get_param(shared_name)
        return default

    def _parse_pose_config(self, value, fallback_xyz, fallback_rpy):
        xyz = fallback_xyz
        rpy = fallback_rpy
        if isinstance(value, dict):
            xyz = value.get("xyz", value.get("position", xyz))
            rpy = value.get("rpy_deg", value.get("rpy", rpy))
        elif isinstance(value, (list, tuple)):
            if len(value) >= 6:
                xyz = value[:3]
                rpy = value[3:6]
            elif len(value) >= 3:
                xyz = value[:3]

        xyz = np.array(xyz, dtype=float)
        rpy = np.array(rpy, dtype=float)
        if xyz.shape[0] != 3 or rpy.shape[0] != 3:
            raise ValueError("handover pose config must provide xyz and rpy_deg with 3 values each")
        return xyz, rpy

    def _ensure_virtual_table(self):
        if not self.table_collision_enabled:
            rospy.loginfo("[p2p] 已停用虛擬桌面防護")
            return

        rospy.loginfo("[p2p] 建立虛擬桌面防護...")
        rospy.sleep(1.0)
        table_pose = PoseStamped()
        table_pose.header.frame_id = self.base_frame
        table_pose.pose.position.z = self.table_collision_z
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box(
            self.table_collision_name,
            table_pose,
            size=self.table_collision_size)
        rospy.loginfo(
            "[p2p] 虛擬桌面防護已就位 name=%s z=%.3f size=(%.3f, %.3f, %.3f)",
            self.table_collision_name,
            self.table_collision_z,
            self.table_collision_size[0],
            self.table_collision_size[1],
            self.table_collision_size[2])

    def _load_handover_config(self):
        if not self.handover_config_path or not os.path.exists(self.handover_config_path):
            return None
        try:
            with open(self.handover_config_path, "r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "[p2p] 讀取 handover config 失敗: %s", exc)
            return None
        data = payload.get("semantic_handover")
        return data if isinstance(data, dict) else None

    def _reload_handover_config_if_needed(self, force=False):
        if not self.handover_config_path or not os.path.exists(self.handover_config_path):
            return
        try:
            mtime = os.path.getmtime(self.handover_config_path)
        except OSError:
            return
        if not force and self._handover_config_mtime == mtime:
            return

        cfg = self._load_handover_config()
        if cfg is None:
            return

        self.enable_handover_fine_adjust = bool(cfg.get(
            "enable_handover_fine_adjust",
            cfg.get("use_dynamic_handover", self.enable_handover_fine_adjust)))
        self.use_dynamic_handover = self.enable_handover_fine_adjust
        if "default_handover_pose" in cfg:
            self.default_handover_pose_xyz, self.default_handover_pose_rpy_deg = \
                self._parse_pose_config(
                    cfg.get("default_handover_pose"),
                    fallback_xyz=self.default_handover_pose_xyz,
                    fallback_rpy=self.default_handover_pose_rpy_deg)
        self.handover_stop_at_zone_entry = bool(cfg.get(
            "handover_stop_at_zone_entry",
            self.handover_stop_at_zone_entry))
        self.handover_zone_center_xyz = np.array(
            cfg.get("handover_zone_center_xyz", self.handover_zone_center_xyz.tolist()),
            dtype=float)
        self.handover_zone_half_extents_xyz = np.array(
            cfg.get("handover_zone_half_extents_xyz", self.handover_zone_half_extents_xyz.tolist()),
            dtype=float)
        self.handover_zone_yaw_deg = float(cfg.get(
            "handover_zone_yaw_deg",
            self.handover_zone_yaw_deg))
        self.handover_zone_entry_margin = float(cfg.get(
            "handover_zone_entry_margin",
            self.handover_zone_entry_margin))
        self._update_handover_zone_rotation()
        self.handover_object_offset_xyz = np.array(
            cfg.get("handover_object_offset_xyz", self.handover_object_offset_xyz.tolist()),
            dtype=float,
        )
        self.handover_keepout_radius = float(cfg.get(
            "handover_keepout_radius",
            self.handover_keepout_radius))
        self.handover_distance_from_hand = float(cfg.get(
            "handover_distance_from_hand",
            self.handover_distance_from_hand))
        self.handover_max_adjust = float(cfg.get(
            "handover_max_adjust",
            cfg.get("handover_fine_adjust_max_translation", self.handover_max_adjust)))
        self.handover_min_hand_distance = float(cfg.get(
            "handover_min_hand_distance",
            self.handover_min_hand_distance))
        self.handover_max_hand_distance = float(cfg.get(
            "handover_max_hand_distance",
            self.handover_max_hand_distance))
        self.handover_z_min = float(cfg.get("handover_z_min", self.handover_z_min))
        self.handover_z_max = float(cfg.get("handover_z_max", self.handover_z_max))
        self.handover_fine_adjust_vel_scale = float(cfg.get(
            "handover_fine_adjust_vel_scale",
            self.handover_fine_adjust_vel_scale))
        self.handover_fine_adjust_acc_scale = float(cfg.get(
            "handover_fine_adjust_acc_scale",
            self.handover_fine_adjust_acc_scale))
        self.hand_state_stability_threshold = float(
            cfg.get("hand_state_stability_threshold", self.hand_state_stability_threshold))
        self.handover_hand_timeout = float(
            cfg.get("handover_hand_timeout", self.handover_hand_timeout))
        self.handover_timeout = float(
            cfg.get("release_timeout",
                    cfg.get("force_release_timeout", self.handover_timeout)))
        self.handover_fallback_orientation = bool(
            cfg.get("handover_fallback_orientation", self.handover_fallback_orientation))
        self.handover_tracking_enabled = bool(cfg.get(
            "handover_tracking_enabled",
            self.handover_tracking_enabled))
        self.handover_tracking_hz = float(cfg.get(
            "handover_tracking_hz",
            self.handover_tracking_hz))
        self.handover_replan_interval = float(cfg.get(
            "handover_replan_interval",
            self.handover_replan_interval))
        self.handover_replan_min_delta = float(cfg.get(
            "handover_replan_min_delta",
            self.handover_replan_min_delta))
        self.handover_track_max_step = float(cfg.get(
            "handover_track_max_step",
            self.handover_track_max_step))
        self.handover_tracking_settle_time = float(cfg.get(
            "handover_tracking_settle_time",
            self.handover_tracking_settle_time))
        self.safety_marker_enabled = bool(cfg.get(
            "safety_marker_enabled",
            self.safety_marker_enabled))
        self.grasp_depth = float(cfg.get("grasp_depth", self.grasp_depth))
        self.grip_force = int(cfg.get("grip_force", self.grip_force))
        self.grasp_candidate_topk = int(cfg.get("grasp_candidate_topk", self.grasp_candidate_topk))
        self.pca_rotation_enabled = bool(cfg.get("pca_rotation_enabled", self.pca_rotation_enabled))
        self.pca_strategy = str(cfg.get("pca_strategy", self.pca_strategy))
        self.pca_min_angle_deg = float(cfg.get("pca_min_angle_deg", self.pca_min_angle_deg))
        self._handover_config_mtime = mtime
        rospy.loginfo(
            "[p2p] 已重載 handover config: fine_adjust=%s tracking=%s default_object=%s keepout=%.3f dist=%.3f hand_range=[%.3f, %.3f] max_adjust=%.3f z=[%.3f, %.3f] safety_markers=%s timeout=%.1f grasp_depth=%.3f grip_force=%d",
            self.enable_handover_fine_adjust,
            self.handover_tracking_enabled,
            self.default_handover_pose_xyz.tolist(),
            self.handover_keepout_radius,
            self.handover_distance_from_hand,
            self.handover_min_hand_distance,
            self.handover_max_hand_distance,
            self.handover_max_adjust,
            self.handover_z_min,
            self.handover_z_max,
            self.safety_marker_enabled,
            self.handover_timeout,
            self.grasp_depth,
            self.grip_force,
        )
        self._publish_handover_safety_markers()

    def _update_handover_zone_rotation(self):
        yaw = math.radians(float(self.handover_zone_yaw_deg))
        c = math.cos(yaw)
        s = math.sin(yaw)
        self.handover_zone_rot = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        self.handover_zone_rot_inv = self.handover_zone_rot.T

    def _hand_state_cb(self, msg: String):
        try:
            self._latest_hand_state = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn_throttle(2.0, "[p2p] hand_state JSON 解析失敗: %s", msg.data)
            return

        palm_xyz = self._extract_palm_xyz(self._latest_hand_state)
        if palm_xyz is not None:
            self._publish_handover_safety_markers(palm_xyz=palm_xyz)

    def _user_cmd_cb(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn_throttle(2.0, "[p2p] user_cmd JSON 解析失敗: %s", msg.data)
            return

        cmd = str(payload.get("cmd", "")).strip().lower()
        if cmd == "abort":
            self._abort_requested = True
            rospy.logwarn("[p2p] 收到 abort 指令")
        elif cmd == "resume":
            self._abort_requested = False
            rospy.loginfo("[p2p] 收到 resume 指令，已清除 handover command state")
        elif cmd == "release":
            self._release_requested = True
            rospy.loginfo("[p2p] 收到 release 指令，將放開夾爪")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _point_from_xyz(self, xyz):
        point = Point()
        point.x = float(xyz[0])
        point.y = float(xyz[1])
        point.z = float(xyz[2])
        return point

    def _extract_palm_xyz(self, hand_state, log_invalid=False):
        try:
            palm_xyz = np.array(hand_state["palm_center_3d"], dtype=float)
        except (KeyError, TypeError, ValueError):
            if log_invalid:
                rospy.logwarn("[p2p] fine adjust 略過：hand_state palm_center_3d 無效")
            return None
        if palm_xyz.shape != (3,) or not np.all(np.isfinite(palm_xyz)):
            if log_invalid:
                rospy.logwarn("[p2p] fine adjust 略過：hand_state palm_center_3d 數值無效")
            return None
        return palm_xyz

    def _make_safety_marker(self, marker_id, marker_type, stamp, ns="handover_safety"):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.0)
        return marker

    def _set_marker_color(self, marker, rgba):
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])

    def _add_sphere_marker(self, markers, marker_id, xyz, radius, rgba, stamp):
        marker = self._make_safety_marker(marker_id, Marker.SPHERE, stamp)
        marker.pose.position = self._point_from_xyz(xyz)
        diameter = max(float(radius) * 2.0, 0.001)
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        self._set_marker_color(marker, rgba)
        markers.markers.append(marker)

    def _add_cube_marker(self, markers, marker_id, center_xyz, scale_xyz, rgba, stamp):
        marker = self._make_safety_marker(marker_id, Marker.CUBE, stamp)
        marker.pose.position = self._point_from_xyz(center_xyz)
        marker.scale.x = max(float(scale_xyz[0]), 0.001)
        marker.scale.y = max(float(scale_xyz[1]), 0.001)
        marker.scale.z = max(float(scale_xyz[2]), 0.001)
        self._set_marker_color(marker, rgba)
        markers.markers.append(marker)

    def _add_line_marker(self, markers, marker_id, points, rgba, stamp, width=0.01):
        marker = self._make_safety_marker(marker_id, Marker.LINE_STRIP, stamp)
        marker.scale.x = float(width)
        marker.points = [self._point_from_xyz(point) for point in points]
        self._set_marker_color(marker, rgba)
        markers.markers.append(marker)

    def _publish_handover_safety_markers(
            self, palm_xyz=None, safe_object_pos=None, adjusted_tool0_xyz=None):
        if not hasattr(self, "safety_marker_pub"):
            return

        stamp = rospy.Time.now()
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.base_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        if not self.safety_marker_enabled:
            self.safety_marker_pub.publish(markers)
            return

        default_object_pos = np.array(self.default_handover_pose_xyz, dtype=float)
        default_tool0_pos = self._pose_xyz(self.make_default_handover_pose())
        z_mid = 0.5 * (self.handover_z_min + self.handover_z_max)
        z_height = max(self.handover_z_max - self.handover_z_min, 0.001)
        adjust_diameter = max(self.handover_max_adjust * 2.0, 0.001)

        self._add_cube_marker(
            markers,
            1,
            [default_object_pos[0], default_object_pos[1], z_mid],
            [adjust_diameter, adjust_diameter, z_height],
            [0.2, 0.8, 0.9, 0.14],
            stamp)
        self._add_sphere_marker(
            markers,
            2,
            default_object_pos,
            self.handover_max_adjust,
            [0.1, 0.7, 1.0, 0.12],
            stamp)
        self._add_sphere_marker(
            markers,
            3,
            default_object_pos,
            0.018,
            [0.0, 0.45, 1.0, 0.95],
            stamp)
        self._add_sphere_marker(
            markers,
            4,
            default_tool0_pos,
            0.014,
            [0.0, 0.2, 0.8, 0.95],
            stamp)

        if palm_xyz is not None:
            self._add_sphere_marker(
                markers,
                10,
                palm_xyz,
                self.handover_keepout_radius,
                [1.0, 0.25, 0.1, 0.18],
                stamp)
            self._add_sphere_marker(
                markers,
                12,
                palm_xyz,
                0.015,
                [1.0, 0.05, 0.05, 0.95],
                stamp)

        if safe_object_pos is not None:
            self._add_sphere_marker(
                markers,
                20,
                safe_object_pos,
                0.02,
                [0.0, 1.0, 0.35, 0.95],
                stamp)
            if palm_xyz is not None:
                self._add_line_marker(
                    markers,
                    21,
                    [palm_xyz, safe_object_pos],
                    [0.0, 1.0, 0.35, 0.8],
                    stamp)

        if adjusted_tool0_xyz is not None:
            self._add_sphere_marker(
                markers,
                30,
                adjusted_tool0_xyz,
                0.017,
                [0.65, 0.25, 1.0, 0.95],
                stamp)
            if safe_object_pos is not None:
                self._add_line_marker(
                    markers,
                    31,
                    [adjusted_tool0_xyz, safe_object_pos],
                    [0.65, 0.25, 1.0, 0.65],
                    stamp,
                    width=0.006)

        self.safety_marker_pub.publish(markers)

    def _normalize_start_state(self):
        """將超出 [-2π, 2π] 的關節值 wrap 回範圍內，避免 MoveIt 拒絕規劃"""
        state = self.group.get_current_state()
        positions = list(state.joint_state.position)
        changed = False
        for i, val in enumerate(positions):
            if val > math.pi * 2:
                positions[i] = val - math.pi * 2
                changed = True
            elif val < -math.pi * 2:
                positions[i] = val + math.pi * 2
                changed = True
        if changed:
            state.joint_state.position = positions
            self.group.set_start_state(state)
            rospy.logwarn("[p2p] 關節超出界限，已自動 normalize 起始狀態")
        else:
            self.group.set_start_state_to_current_state()

    def plan_execute_cartesian_to(self, target_pose):
        self._normalize_start_state()
        wp = target_pose.pose if isinstance(target_pose, PoseStamped) else target_pose
        plan, fraction = self.group.compute_cartesian_path([wp], self.eef_step, True)
        if fraction < 0.99:
            rospy.logwarn(f"笛卡兒路徑不全（避碰）: {fraction:.2f}，改用 avoid_collisions=False 重試...")
            plan, fraction = self.group.compute_cartesian_path([wp], self.eef_step, False)
        if fraction < 0.99:
            rospy.logwarn(f"笛卡兒路徑不全: {fraction:.2f}")
            return False
        ok = self.group.execute(plan, wait=True)
        if not ok:
            cur = self.group.get_current_pose().pose
            dx = cur.position.x - wp.position.x
            dy = cur.position.y - wp.position.y
            dz = cur.position.z - wp.position.z
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            if dist < 0.01:
                rospy.logwarn("[p2p] Cartesian execute 回傳失敗但末端已到位 (err=%.4f m)，視為成功", dist)
                return True
        return ok

    def _plan_length(self, plan):
        """計算關節空間總位移（越小越短）"""
        pts = plan.joint_trajectory.points
        if len(pts) < 2:
            return float('inf')
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            total += sum(abs(j2 - j1) for j1, j2 in zip(a.positions, b.positions))
        return total

    def solve_nearest_ik(self, ps_target):
        if self.ik_srv is None:
            return None

        req = GetPositionIKRequest()
        req.ik_request.group_name = self.move_group
        req.ik_request.pose_stamped = ps_target
        req.ik_request.robot_state = self.group.get_current_state()
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = rospy.Duration(0.2)

        ee_link = self.group.get_end_effector_link()
        if ee_link:
            req.ik_request.ik_link_name = ee_link

        try:
            resp = self.ik_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn("[p2p] IK service 呼叫失敗，退回 pose target 規劃: %s", e)
            return None

        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            rospy.logwarn("[p2p] IK 求解失敗(error=%s)，退回 pose target 規劃",
                          resp.error_code.val)
            return None

        joint_map = dict(zip(resp.solution.joint_state.name, resp.solution.joint_state.position))
        active_joints = self.group.get_active_joints()
        try:
            joint_target = [joint_map[name] for name in active_joints]
        except KeyError as e:
            rospy.logwarn("[p2p] IK 解缺少關節 %s，退回 pose target 規劃", e)
            return None

        rospy.loginfo("[p2p] 採用最近 IK 分支作為 joint target")
        return joint_target

    def joint_plan_execute(
            self, ps_target, label="目標點", n_attempts=2, execute=True,
            log_failure_as_error=True):
        """多次規劃取最短關節路徑，可選擇只規劃不執行（供安全確認用）"""
        self._normalize_start_state()
        joint_target = self.solve_nearest_ik(ps_target) if self.prefer_nearest_ik else None
        try:
            if joint_target is not None:
                self.group.set_joint_value_target(joint_target)
            else:
                self.group.set_pose_target(ps_target)
        except Exception as e:
            if joint_target is not None:
                rospy.logwarn("[p2p] %s 設定 joint target 失敗，退回 pose target 規劃: %s",
                              label, e)
                self.group.clear_pose_targets()
                self.group.set_pose_target(ps_target)
            else:
                raise
        best_plan, best_len = None, float('inf')
        for i in range(n_attempts):
            success, plan, _, error_code = self.group.plan()
            if not success:
                rospy.logwarn(f"[p2p] {label} 第 {i+1} 次規劃失敗: {error_code}")
                continue
            length = self._plan_length(plan)
            if length < best_len:
                best_len, best_plan = length, plan
        self.group.clear_pose_targets()

        if best_plan is None:
            log_fn = rospy.logerr if log_failure_as_error else rospy.logwarn
            log_fn(f"[p2p] {label} 規劃失敗（{n_attempts} 次均無解）")
            return False, None

        rospy.loginfo(f"[p2p] {label} 最短路徑: {best_len:.3f} rad（{n_attempts} 次中選出）")
        if not execute:
            return True, best_plan

        ok = self.group.execute(best_plan, wait=True)
        self.group.stop()
        if not ok:
            rospy.logerr(f"[p2p] {label} 執行失敗")
        return ok, best_plan

    def get_ee_z_axis_in_base(self, pose_stamped):
        q = pose_stamped.pose.orientation
        M = quaternion_matrix([q.x, q.y, q.z, q.w])
        z_axis = M[0:3, 2]
        return z_axis / np.linalg.norm(z_axis)

    def ee_z_from_quaternion(self, quat_xyzw):
        M = quaternion_matrix(quat_xyzw)
        z_axis = M[0:3, 2]
        return z_axis / np.linalg.norm(z_axis)

    def resolve_final_insert_depth(self, server_depth):
        """決定 grasp reference pose 轉成機器人 final grasp pose 時的前進補償量。"""
        if self.use_server_depth_for_offset and server_depth is not None:
            rospy.loginfo("[p2p] final grasp offset 使用 server depth = %.3f", server_depth)
            return float(server_depth)

        if server_depth is not None:
            rospy.loginfo(
                "[p2p] final grasp offset 使用本地 grasp_depth = %.3f（server depth %.3f 僅記錄）",
                self.grasp_depth, server_depth)
        else:
            rospy.loginfo("[p2p] final grasp offset 使用本地 grasp_depth = %.3f", self.grasp_depth)
        return self.grasp_depth

    def build_robot_grasp_poses(self, grasp_ref_pose, final_insert_depth):
        """
        AnyGrasp 回傳的是 grasp reference pose。
        先轉成 UR3 final grasp pose，再由 final grasp 沿接近軸退回 pre-grasp。
        """
        grasp_ref_xyz = np.array([
            grasp_ref_pose.pose.position.x,
            grasp_ref_pose.pose.position.y,
            grasp_ref_pose.pose.position.z
        ], dtype=float)
        ee_z = self.get_ee_z_axis_in_base(grasp_ref_pose)

        # final grasp pose: 由 grasp reference pose 沿接近軸補償插入量，再扣掉 tool0 到夾爪工作點的固定偏移
        final_grasp_xyz = grasp_ref_xyz + ee_z * final_insert_depth - ee_z * self.tcp_offset
        pregrasp_xyz = final_grasp_xyz - ee_z * self.approach_dist

        ps_grasp = self.make_pose_stamped(final_grasp_xyz, grasp_ref_pose.pose.orientation)
        ps_pre = self.make_pose_stamped(pregrasp_xyz, grasp_ref_pose.pose.orientation)

        return ps_grasp, ps_pre, {
            "grasp_ref_xyz": grasp_ref_xyz,
            "final_grasp_xyz": final_grasp_xyz,
            "pregrasp_xyz": pregrasp_xyz,
            "ee_z": ee_z,
            "final_insert_depth": float(final_insert_depth),
        }

    def select_camera_facing_orientation(self, q_orig, object_surface, camera_pos):
        camera_to_object = object_surface - camera_pos
        norm = np.linalg.norm(camera_to_object)
        if norm < 1e-9:
            rospy.logwarn("[p2p] camera_to_object 長度過小，退回舊版 Y-90 映射")
            q_default = quaternion_multiply(q_orig, quaternion_from_euler(0, -math.pi / 2, 0))
            return q_default, {
                "camera_to_object": np.array([0.0, 0.0, 0.0]),
                "score_neg90": 0.0,
                "score_pos90": float("-inf"),
                "label": "Y-90 (fallback)",
            }

        camera_to_object = camera_to_object / norm
        candidates = []
        for y_deg in (-90.0, 90.0):
            q_map = quaternion_multiply(q_orig, quaternion_from_euler(0, math.radians(y_deg), 0))
            ee_z = self.ee_z_from_quaternion(q_map)
            score = float(np.dot(ee_z, camera_to_object))
            candidates.append({
                "label": f"Y{y_deg:+.0f}",
                "quat": q_map,
                "ee_z": ee_z,
                "score": score,
            })

        best = max(candidates, key=lambda item: item["score"])
        return best["quat"], {
            "camera_to_object": camera_to_object,
            "score_neg90": candidates[0]["score"],
            "score_pos90": candidates[1]["score"],
            "label": best["label"],
        }

    def make_pose_stamped(self, xyz, orientation):
        """從 xyz + 四元數 Quaternion 物件建立 PoseStamped"""
        ps = PoseStamped()
        ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation = orientation
        return ps

    def make_pose_stamped_from_xyz_rpy(self, xyz, rpy_deg):
        qx, qy, qz, qw = quaternion_from_euler(
            math.radians(rpy_deg[0]), math.radians(rpy_deg[1]), math.radians(rpy_deg[2]))
        ps = PoseStamped(); ps.header.frame_id = self.base_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation.x, ps.pose.orientation.y, \
            ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
        return ps

    def make_default_handover_pose(self, orientation=None):
        if orientation is None:
            orientation = self._active_handover_orientation

        if orientation is None:
            object_pose = self.make_pose_stamped_from_xyz_rpy(
                self.default_handover_pose_xyz,
                self.default_handover_pose_rpy_deg)
            # Old handover behavior kept for fallback/reference:
            # object_pose = self.make_pose_stamped_from_xyz_rpy(
            #     self.default_handover_pose_xyz,
            #     self.default_handover_pose_rpy_deg)
        else:
            object_pose = self.make_pose_stamped(self.default_handover_pose_xyz, orientation)
        ee_z = self.get_ee_z_axis_in_base(object_pose)
        tool0_xyz = self.default_handover_pose_xyz - ee_z * self.tcp_offset
        return self.make_pose_stamped(tool0_xyz, object_pose.pose.orientation)

    def make_handover_entry_pose(self, from_tool0_pose, orientation=None):
        default_pose = self.make_default_handover_pose(orientation=orientation)
        if not self.handover_stop_at_zone_entry:
            return default_pose, np.array(self.default_handover_pose_xyz, dtype=float), False

        current_ee_z = self.get_ee_z_axis_in_base(from_tool0_pose)
        start_object_pos = self._pose_xyz(from_tool0_pose) + current_ee_z * self.tcp_offset
        center = np.array(self.handover_zone_center_xyz, dtype=float)
        half = np.maximum(np.array(self.handover_zone_half_extents_xyz, dtype=float), 1e-6)
        start_local = self.handover_zone_rot_inv.dot(start_object_pos - center)

        if np.all(np.abs(start_local) <= half):
            rospy.loginfo("[p2p] 抬升後物體已在 handover zone 內，使用目前位置作為 zone entry")
            entry_object_pos = start_object_pos
        else:
            end_local = self.handover_zone_rot_inv.dot(
                np.array(self.default_handover_pose_xyz, dtype=float) - center)
            direction = end_local - start_local
            t_enter = 0.0
            t_exit = 1.0
            for axis in range(3):
                if abs(direction[axis]) < 1e-9:
                    if start_local[axis] < -half[axis] or start_local[axis] > half[axis]:
                        rospy.logwarn("[p2p] 無法計算 handover zone 入口，回退 default_handover_pose")
                        return default_pose, np.array(self.default_handover_pose_xyz, dtype=float), False
                    continue
                t1 = (-half[axis] - start_local[axis]) / direction[axis]
                t2 = (half[axis] - start_local[axis]) / direction[axis]
                t_near = min(t1, t2)
                t_far = max(t1, t2)
                t_enter = max(t_enter, t_near)
                t_exit = min(t_exit, t_far)

            if t_enter > t_exit or t_exit < 0.0 or t_enter > 1.0:
                rospy.logwarn("[p2p] handover zone 入口線段無交集，回退 default_handover_pose")
                return default_pose, np.array(self.default_handover_pose_xyz, dtype=float), False

            start_to_default = np.array(self.default_handover_pose_xyz, dtype=float) - start_object_pos
            path_len = float(np.linalg.norm(start_to_default))
            margin_t = self.handover_zone_entry_margin / max(path_len, 1e-6)
            entry_t = min(max(t_enter, 0.0) + max(margin_t, 0.0), 1.0)
            entry_object_pos = start_object_pos + start_to_default * entry_t

        default_ee_z = self.get_ee_z_axis_in_base(default_pose)
        entry_tool0_xyz = entry_object_pos - default_ee_z * self.tcp_offset
        entry_pose = self.make_pose_stamped(entry_tool0_xyz, default_pose.pose.orientation)
        rospy.loginfo(
            "[p2p] handover zone entry object=(%.3f, %.3f, %.3f), tool0=(%.3f, %.3f, %.3f), center=(%.3f, %.3f, %.3f)",
            entry_object_pos[0], entry_object_pos[1], entry_object_pos[2],
            entry_tool0_xyz[0], entry_tool0_xyz[1], entry_tool0_xyz[2],
            center[0], center[1], center[2])
        return entry_pose, entry_object_pos, True

    def _pose_xyz(self, pose_stamped):
        return np.array([
            pose_stamped.pose.position.x,
            pose_stamped.pose.position.y,
            pose_stamped.pose.position.z,
        ], dtype=float)

    def _with_pose_xyz(self, pose_stamped, xyz):
        return self.make_pose_stamped(xyz, pose_stamped.pose.orientation)

    def _limit_handover_step(self, current_pose, target_pose):
        current_xyz = self._pose_xyz(current_pose)
        target_xyz = self._pose_xyz(target_pose)
        delta = target_xyz - current_xyz
        dist = float(np.linalg.norm(delta))
        if dist < self.handover_replan_min_delta:
            rospy.loginfo_throttle(
                2.0,
                "[p2p] tracking 目標位移 %.3fm 小於門檻 %.3fm，略過",
                dist,
                self.handover_replan_min_delta)
            return None
        if dist <= self.handover_track_max_step:
            return target_pose

        limited_xyz = current_xyz + delta / max(dist, 1e-6) * self.handover_track_max_step
        rospy.logwarn(
            "[p2p] tracking 單次位移 %.3fm 超過上限 %.3fm，已限制步長",
            dist,
            self.handover_track_max_step)
        return self._with_pose_xyz(target_pose, limited_xyz)

    def compute_safe_adjusted_handover_pose(self, hand_state, current_pose=None, orientation=None):
        default_pose = self.make_default_handover_pose(orientation=orientation)
        if not self._is_hand_state_stable(hand_state):
            return None

        palm_xyz = self._extract_palm_xyz(hand_state, log_invalid=True)
        if palm_xyz is None:
            return None
        self._publish_handover_safety_markers(palm_xyz=palm_xyz)

        default_object_pos = np.array(self.default_handover_pose_xyz, dtype=float)
        reference_pose = current_pose if current_pose is not None else default_pose
        reference_ee_z = self.get_ee_z_axis_in_base(reference_pose)
        reference_object_pos = (
            self._pose_xyz(reference_pose) + reference_ee_z * self.tcp_offset
            if current_pose is not None else default_object_pos)

        object_to_hand = palm_xyz - reference_object_pos
        distance_to_hand_now = float(np.linalg.norm(object_to_hand))
        if distance_to_hand_now < self.handover_min_hand_distance:
            rospy.logwarn(
                "[p2p] fine adjust 略過：手心距目前 object %.3fm 小於下限 %.3fm",
                distance_to_hand_now,
                self.handover_min_hand_distance)
            return None
        if distance_to_hand_now > self.handover_max_hand_distance:
            rospy.logwarn(
                "[p2p] fine adjust 略過：手心距目前 object %.3fm 大於上限 %.3fm",
                distance_to_hand_now,
                self.handover_max_hand_distance)
            return None
        if distance_to_hand_now < 1e-6:
            rospy.logwarn("[p2p] fine adjust 略過：目前 object pose 與 palm 過近，方向無法定義")
            return None
        direction_to_hand = object_to_hand / distance_to_hand_now

        if current_pose is not None:
            safe_distance = self.handover_keepout_radius + 0.005
            if distance_to_hand_now <= safe_distance + self.handover_replan_min_delta:
                rospy.loginfo_throttle(
                    2.0,
                    "[p2p] tracking 已接近 hand keepout 邊界 hand_dist=%.3f keepout=%.3f",
                    distance_to_hand_now,
                    self.handover_keepout_radius)
                return None
            safe_object_pos = palm_xyz - direction_to_hand * safe_distance
        else:
            direction_from_hand_to_default = default_object_pos - palm_xyz
            direction_norm = np.linalg.norm(direction_from_hand_to_default)
            if direction_norm < 1e-6:
                rospy.logwarn("[p2p] fine adjust 略過：default object pose 與 palm 過近，方向無法定義")
                return None
            direction_from_hand_to_default = direction_from_hand_to_default / direction_norm
            safe_object_pos = palm_xyz + direction_from_hand_to_default * self.handover_distance_from_hand
            adjust_vec = safe_object_pos - default_object_pos
            adjust_norm = np.linalg.norm(adjust_vec)
            if adjust_norm > self.handover_max_adjust:
                adjust_vec = adjust_vec / max(adjust_norm, 1e-6) * self.handover_max_adjust
                safe_object_pos = default_object_pos + adjust_vec
                rospy.logwarn(
                    "[p2p] fine adjust 位移 %.3fm 超過上限 %.3fm，已 clamp",
                    adjust_norm,
                    self.handover_max_adjust)

        distance_to_hand = float(np.linalg.norm(safe_object_pos - palm_xyz))
        if distance_to_hand < self.handover_keepout_radius:
            rospy.logwarn(
                "[p2p] fine adjust 略過：safe_object_pos 距手心 %.3fm 小於 keepout %.3fm",
                distance_to_hand,
                self.handover_keepout_radius)
            self._publish_handover_safety_markers(
                palm_xyz=palm_xyz,
                safe_object_pos=safe_object_pos)
            return None

        if safe_object_pos[2] < self.handover_z_min or safe_object_pos[2] > self.handover_z_max:
            rospy.logwarn(
                "[p2p] fine adjust 略過：safe_object_pos z=%.3f 超出 [%.3f, %.3f]",
                safe_object_pos[2],
                self.handover_z_min,
                self.handover_z_max)
            self._publish_handover_safety_markers(
                palm_xyz=palm_xyz,
                safe_object_pos=safe_object_pos)
            return None

        ee_z = self.get_ee_z_axis_in_base(default_pose)
        adjusted_tool0_xyz = safe_object_pos - ee_z * self.tcp_offset
        tool0_distance_to_hand = float(np.linalg.norm(adjusted_tool0_xyz - palm_xyz))
        if tool0_distance_to_hand < self.handover_keepout_radius:
            rospy.logwarn(
                "[p2p] fine adjust 略過：adjusted tool0 距手心 %.3fm 小於 keepout %.3fm",
                tool0_distance_to_hand,
                self.handover_keepout_radius)
            self._publish_handover_safety_markers(
                palm_xyz=palm_xyz,
                safe_object_pos=safe_object_pos,
                adjusted_tool0_xyz=adjusted_tool0_xyz)
            return None
        adjusted_pose = self.make_pose_stamped(
            adjusted_tool0_xyz,
            default_pose.pose.orientation)
        self._publish_handover_safety_markers(
            palm_xyz=palm_xyz,
            safe_object_pos=safe_object_pos,
            adjusted_tool0_xyz=adjusted_tool0_xyz)
        rospy.loginfo("[p2p] fine adjust hand palm = (%.3f, %.3f, %.3f)", *palm_xyz)
        rospy.loginfo(
            "[p2p] fine adjust object = (%.3f, %.3f, %.3f), tool0 = (%.3f, %.3f, %.3f), hand_dist=%.3f, tool0_hand_dist=%.3f, from_current=%.3f, adjust=%.3f",
            safe_object_pos[0], safe_object_pos[1], safe_object_pos[2],
            adjusted_tool0_xyz[0], adjusted_tool0_xyz[1], adjusted_tool0_xyz[2],
            distance_to_hand,
            tool0_distance_to_hand,
            float(np.linalg.norm(safe_object_pos - reference_object_pos)),
            float(np.linalg.norm(safe_object_pos - default_object_pos)))
        return adjusted_pose

    def try_fine_adjust_handover_pose(self, hand_state, current_pose=None, orientation=None):
        adjusted_pose = self.compute_safe_adjusted_handover_pose(
            hand_state,
            current_pose=current_pose,
            orientation=orientation)
        if adjusted_pose is None:
            return None
        if current_pose is not None:
            adjusted_pose = self._limit_handover_step(current_pose, adjusted_pose)
            if adjusted_pose is None:
                return None

        old_vel, old_acc = self.vel_scale, self.acc_scale
        self.group.set_max_velocity_scaling_factor(self.handover_fine_adjust_vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.handover_fine_adjust_acc_scale)
        try:
            ok, plan = self.joint_plan_execute(
                adjusted_pose,
                "交接微調",
                execute=False,
                log_failure_as_error=False)
            if not ok or plan is None:
                rospy.logwarn("[p2p] fine adjust 規劃失敗，維持 default_handover_pose")
                return None
            rospy.loginfo(
                "[p2p] 低速執行交接微調 vel=%.2f acc=%.2f",
                self.handover_fine_adjust_vel_scale,
                self.handover_fine_adjust_acc_scale)
            if not self.group.execute(plan, wait=True):
                rospy.logwarn("[p2p] fine adjust 執行失敗，維持目前姿態等待交接")
                return None
            self.group.stop()
            return adjusted_pose
        finally:
            self.group.set_max_velocity_scaling_factor(old_vel)
            self.group.set_max_acceleration_scaling_factor(old_acc)

    def _reset_handover_command_state(self):
        self._abort_requested = False
        self._release_requested = False
        self._active_handover_orientation = None
        self._phase2_start_time = None
        self._phase2_release_done = False
        # 清空姿態誤差指標
        self._pca_axis_at_grasp = None
        self._grasp_orientation = None
        self._observed_handover_axis = None
        self._observed_handover_frame = None
        self._observed_handover_eig_ratio = None
        self._handover_orientation_error_deg = None

    def _get_recent_hand_state(self):
        state = self._latest_hand_state
        if not isinstance(state, dict):
            return None

        try:
            stamp_sec = float(state.get("stamp", 0.0))
        except (TypeError, ValueError):
            return None
        if stamp_sec <= 0.0:
            return None

        age = rospy.Time.now().to_sec() - stamp_sec
        if age > self.hand_state_max_age:
            return None
        return state

    def _is_hand_state_stable(self, state):
        if not isinstance(state, dict):
            return False
        if not bool(state.get("valid", False)):
            return False
        if not bool(state.get("in_zone", False)):
            return False
        palm_center = state.get("palm_center_3d")
        if not isinstance(palm_center, (list, tuple)) or len(palm_center) != 3:
            return False
        try:
            stability = float(state.get("stability", 0.0))
        except (TypeError, ValueError):
            return False
        return stability >= self.hand_state_stability_threshold

    def _wait_for_stable_hand_sample(self):
        self._reload_handover_config_if_needed()
        rospy.loginfo(
            "[p2p] 等待穩定手位（threshold=%.2f, timeout=%.1fs, topic=%s）...",
            self.hand_state_stability_threshold,
            self.handover_hand_timeout,
            self.hand_state_topic)
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(10)
        while rospy.Time.now().to_sec() - t0 < self.handover_hand_timeout:
            if self._abort_requested:
                rospy.logwarn("[p2p] handover 在取樣手位前被 abort")
                return None, None

            state = self._get_recent_hand_state()
            if self._is_hand_state_stable(state):
                hand_xyz = np.array(state["palm_center_3d"], dtype=float)
                rospy.loginfo(
                    "[p2p] 取得穩定手位 xyz=(%.3f, %.3f, %.3f), stability=%.2f",
                    hand_xyz[0], hand_xyz[1], hand_xyz[2], float(state.get("stability", 0.0)))
                return hand_xyz, state

            rospy.loginfo_throttle(2.0, "[p2p] 等待手進入交接區並穩定...")
            rate.sleep()

        rospy.logwarn("[p2p] 等待穩定手位逾時（%.1fs）", self.handover_hand_timeout)
        return None, None

    def resolve_handover_pose(self, grasp_pose, ee_z):
        if not self.use_dynamic_handover:
            handover_object_xyz = np.array(self.handover_object_xyz, dtype=float)
            handover_tool0_xyz = handover_object_xyz - ee_z * self.tcp_offset
            ps_handover = self.make_pose_stamped(
                handover_tool0_xyz, grasp_pose.pose.orientation)
            rospy.loginfo(
                "[p2p] 使用固定交接點 object=(%.3f, %.3f, %.3f)",
                *handover_object_xyz)
            return ps_handover, handover_object_xyz, None

        palm_xyz, hand_state = self._wait_for_stable_hand_sample()
        if palm_xyz is None:
            return None, None, None

        handover_object_xyz = palm_xyz + self.handover_object_offset_xyz
        handover_tool0_xyz = handover_object_xyz - ee_z * self.tcp_offset
        ps_handover = self.make_pose_stamped(handover_tool0_xyz, grasp_pose.pose.orientation)
        rospy.loginfo("[p2p] hand palm xyz    = (%.3f, %.3f, %.3f)", *palm_xyz)
        rospy.loginfo(
            "[p2p] handover offset  = (%.3f, %.3f, %.3f)",
            *self.handover_object_offset_xyz)
        rospy.loginfo("[p2p] handover object = (%.3f, %.3f, %.3f)", *handover_object_xyz)
        rospy.loginfo("[p2p] handover tool0  = (%.3f, %.3f, %.3f)", *handover_tool0_xyz)
        return ps_handover, handover_object_xyz, hand_state

    def _release_triggered(self):
        """非阻塞檢查：Enter 鍵或 release 指令是否已觸發。"""
        if self._release_requested:
            return True
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return True
        return False

    def _wait_for_handover(self):
        self._reload_handover_config_if_needed()
        rospy.loginfo("[p2p] 等待交接，按 Enter 放開夾爪（timeout=%.0fs）...", self.handover_timeout)
        print("\n>>> 按 Enter 放開夾爪 <<<\n", flush=True)
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(20)
        while rospy.Time.now().to_sec() - t0 < self.handover_timeout:
            if self._abort_requested:
                rospy.logwarn("[p2p] handover 被 abort，保持夾持")
                return False
            if self._release_triggered():
                self._release_requested = False
                rospy.loginfo("[p2p] 收到放開指令，釋放夾爪")
                return True
            rate.sleep()
        rospy.logwarn("[p2p] 交接逾時（%.0fs），保持夾持", self.handover_timeout)
        return False

    def _track_and_wait_for_handover(self, start_pose, handover_orientation=None):
        self._reload_handover_config_if_needed()
        if not self.handover_tracking_enabled:
            return self._wait_for_handover(), start_pose

        rospy.loginfo(
            "[p2p] 進入 handover tracking：hz=%.1f replan_interval=%.1fs min_delta=%.3fm max_step=%.3fm",
            self.handover_tracking_hz,
            self.handover_replan_interval,
            self.handover_replan_min_delta,
            self.handover_track_max_step)
        print("\n>>> 按 Enter 放開夾爪 <<<\n", flush=True)
        current_pose = start_pose
        last_replan_t = 0.0
        t0 = rospy.Time.now().to_sec()
        rate = rospy.Rate(max(0.1, self.handover_tracking_hz))

        while rospy.Time.now().to_sec() - t0 < self.handover_timeout:
            if self._abort_requested:
                rospy.logwarn("[p2p] handover tracking 被 abort，保持夾持")
                return False, current_pose
            if self._release_triggered():
                self._release_requested = False
                rospy.loginfo("[p2p] 收到放開指令，釋放夾爪")
                return True, current_pose

            now = rospy.Time.now().to_sec()
            if (self.enable_handover_fine_adjust and
                    now - last_replan_t >= self.handover_replan_interval):
                last_replan_t = now
                hand_state = self._get_recent_hand_state()
                if self._is_hand_state_stable(hand_state):
                    next_pose = self.try_fine_adjust_handover_pose(
                        hand_state,
                        current_pose=current_pose,
                        orientation=handover_orientation)
                    if next_pose is not None:
                        current_pose = next_pose
                else:
                    rospy.loginfo_throttle(2.0, "[p2p] tracking 等待穩定 hand_state...")

            rate.sleep()

        rospy.logwarn("[p2p] handover tracking 逾時（%.0fs），保持夾持", self.handover_timeout)
        return False, current_pose

    def init_gripper(self):
        try:
            self.g.connect(self.grip_ip, self.grip_port); self.g.activate()
            rospy.sleep(1.0); self.g.move_and_wait_for_pos(0, 100, 80)
        except: self.g = None

    # ------------------------------------------------------------------ #
    #  待機點                                                              #
    # ------------------------------------------------------------------ #
    def go_to_ready_pose(self):
        ready_xyz = list(self.ready_xyz)
        ps_ready = self.make_pose_stamped_from_xyz_rpy(ready_xyz, [180.0, 0.0, 0.0])
        rospy.loginfo(f"[p2p] 前往待機點 {ready_xyz} ...")
        ok, _ = self.joint_plan_execute(ps_ready, "待機點")
        if ok:
            rospy.loginfo("[p2p] ✅ 已到達待機點，等待下一次任務")
            if self._phase2_start_time is not None and self._phase2_release_done:
                phase2_total_s = time.perf_counter() - self._phase2_start_time
                rospy.loginfo(
                    "[p2p] ⏱️ Phase2 總時間（開始抓取→交接→釋放→回待機點): %.2f s",
                    phase2_total_s)
                self._log_combined_timing_summary(phase2_total_s)
                self._phase2_start_time = None
                self._phase2_release_done = False
                self._phase1_timing = None
                self._planning_total_s = None
                self._human_confirm_s = None
        else:
            rospy.logerr("[p2p] 無法到達待機點，請手動確認安全")

    def _request_and_compute_handover_error(self):
        """到達交接位置時，請求 server 重觀測物件主軸，並比對 3D 姿態誤差。"""
        if self._pca_axis_at_grasp is None or self._grasp_orientation is None \
                or self._active_handover_orientation is None:
            rospy.logwarn("[metric] 缺少 baseline 主軸或抓取/交接方向，略過姿態誤差計算")
            return

        # 觸發 server 重觀測（client_camera 端會快照當前影格並走 ZMQ）
        self._observed_handover_axis = None
        self._observed_handover_frame = None
        self._observed_handover_eig_ratio = None
        try:
            self.handover_pca_request_pub.publish(String(data="handover_pca"))
        except Exception as e:
            rospy.logwarn("[metric] 發布 request_pca 失敗: %s", e)
            return

        # 短輪詢等待觀測結果（非阻塞流程，此處只等有限時間）
        deadline = rospy.Time.now() + rospy.Duration(float(self.handover_pca_timeout_s))
        rate = rospy.Rate(20)
        while self._observed_handover_axis is None and rospy.Time.now() < deadline \
                and not rospy.is_shutdown():
            rate.sleep()

        if self._observed_handover_axis is None:
            rospy.logwarn("[metric] %.1fs 內未收到 server 重觀測主軸，姿態誤差 = N/A",
                          float(self.handover_pca_timeout_s))
            return

        self._compute_handover_object_error()

    def _compute_handover_object_error(self):
        """比對『理想旋轉後主軸』與『實際觀測主軸』的 3D 夾角。"""
        # 1) 觀測軸轉到 base（僅取旋轉部分）
        try:
            tf_c2b = self.tfbuf.lookup_transform(
                self.base_frame, self._observed_handover_frame,
                rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("[metric] 觀測軸 TF 轉換失敗: %s", e)
            return
        q = tf_c2b.transform.rotation
        R_c2b = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        a_measured = R_c2b.dot(self._observed_handover_axis)
        a_measured /= (np.linalg.norm(a_measured) + 1e-12)

        # 2) 理想軸 = R_delta · a_table，R_delta = G_handover ⊗ G_grasp⁻¹
        qg = self._grasp_orientation
        qh = self._active_handover_orientation
        q_grasp = [qg.x, qg.y, qg.z, qg.w]
        q_handover = [qh.x, qh.y, qh.z, qh.w]
        from tf.transformations import quaternion_inverse
        q_delta = quaternion_multiply(q_handover, quaternion_inverse(q_grasp))
        R_delta = quaternion_matrix(q_delta)[:3, :3]
        a_ideal = R_delta.dot(self._pca_axis_at_grasp)
        a_ideal /= (np.linalg.norm(a_ideal) + 1e-12)

        # 3) 3D 夾角（主軸無方向性 → 取絕對值 → 0–90°）
        d = min(1.0, abs(float(np.dot(a_ideal, a_measured))))
        err = math.degrees(math.acos(d))
        self._handover_orientation_error_deg = err

        eig = self._observed_handover_eig_ratio
        rospy.loginfo(
            "[metric] 交接物件姿態 3D 誤差: %.2f° "
            "(理想 (%.3f,%.3f,%.3f) vs 實際 (%.3f,%.3f,%.3f), eig_ratio=%s)",
            err, a_ideal[0], a_ideal[1], a_ideal[2],
            a_measured[0], a_measured[1], a_measured[2],
            ("%.1f" % eig) if eig is not None else "N/A")

    def _log_combined_timing_summary(self, phase2_total_s):
        p1 = self._phase1_timing or {}
        client_total_s = p1.get("client_total_s")
        planning_s = self._planning_total_s or 0.0
        confirm_s  = self._human_confirm_s  or 0.0
        rospy.loginfo("======== 任務時間總結 ========")
        if client_total_s is not None:
            inference_total_s = client_total_s + planning_s
            rospy.loginfo("  推論總時間: %.2f s", inference_total_s)
            rospy.loginfo("    - Server 端推論: %.2f s", client_total_s)
            if p1.get("gemini_s") is not None:
                rospy.loginfo("      - Gemini 回應: %.2f s", p1["gemini_s"])
            if p1.get("anygrasp_s") is not None:
                rospy.loginfo("      - AnyGrasp 推論: %.2f s", p1["anygrasp_s"])
            rospy.loginfo("    - 路徑規劃: %.2f s", planning_s)
        else:
            inference_total_s = planning_s
            rospy.loginfo("  推論總時間: %.2f s（無 server 計時）", inference_total_s)
            rospy.loginfo("    - 路徑規劃: %.2f s", planning_s)
        rospy.loginfo("  （人為確認等待: %.2f s，不計入總時間）", confirm_s)
        rospy.loginfo("  Phase2 抓取執行總時間: %.2f s", phase2_total_s)
        rospy.loginfo("  ---------------------------")
        rospy.loginfo("  總計（推論+執行）: %.2f s", inference_total_s + phase2_total_s)
        v = self._handover_orientation_error_deg
        rospy.loginfo("  交接物件姿態 3D 誤差: %s",
                      ("%.2f°" % v) if v is not None else "N/A")
        rospy.loginfo("================================")

    def _prompt_ready(self):
        """任務結束（無論成功/失敗）都詢問是否回待機點"""
        try:
            print("\n==================================================")
            ans = input("[p2p] 任務結束。[Enter] 回待機點 | [n] 不移動：")
            print("==================================================\n")
            if ans.lower() != 'n':
                self.go_to_ready_pose()
        except EOFError:
            self.go_to_ready_pose()

    # ------------------------------------------------------------------ #
    #  核心流程                                                             #
    # ------------------------------------------------------------------ #
    def run_once(self, ps_target, final_insert_depth=None, server_depth=None, fallback_targets=None):
        try:
            self._grasp_and_place(
                ps_target,
                final_insert_depth=final_insert_depth,
                server_depth=server_depth,
                fallback_targets=fallback_targets)
        except Exception as e:
            rospy.logerr("[p2p] 抓取流程未捕捉例外: %s", e)
            rospy.logerr("%s", traceback.format_exc())
        finally:
            self._prompt_ready()

    def _grasp_and_place(self, ps_target, final_insert_depth=None, server_depth=None,
                         fallback_targets=None):
        self._reset_handover_command_state()
        final_insert_depth = (
            self.grasp_depth if final_insert_depth is None else float(final_insert_depth))

        all_targets = [ps_target] + list(fallback_targets or [])

        # ── 動作 A：依序試每個候選的 Pre-Grasp 規劃 ─────────────────────── #
        plan = None
        ps_grasp = None
        grasp_info = None
        self._planning_total_s = 0.0
        self._human_confirm_s = 0.0
        for idx, ps_t in enumerate(all_targets):
            ps_grasp_t, ps_pre_t, grasp_info_t = self.build_robot_grasp_poses(
                ps_t, final_insert_depth)
            if idx == 0:
                if server_depth is not None:
                    rospy.loginfo("[p2p] server_depth    = %.3f", server_depth)
                rospy.loginfo("[p2p] grasp_ref_xyz   = (%.3f, %.3f, %.3f)",
                              *grasp_info_t["grasp_ref_xyz"])
                rospy.loginfo("[p2p] ee_z            = (%.3f, %.3f, %.3f)",
                              *grasp_info_t["ee_z"])
                rospy.loginfo(
                    "[p2p] final_insert_depth = %.3f, tcp_offset = %.3f, approach_dist = %.3f",
                    grasp_info_t["final_insert_depth"], self.tcp_offset, self.approach_dist)
                rospy.loginfo("[p2p] final_grasp_xyz = (%.3f, %.3f, %.3f)",
                              *grasp_info_t["final_grasp_xyz"])
                rospy.loginfo("[p2p] pregrasp_xyz    = (%.3f, %.3f, %.3f)",
                              *grasp_info_t["pregrasp_xyz"])
            else:
                rospy.logwarn("[p2p] Pre-Grasp 嘗試備選候選 #%d "
                              "(%.3f, %.3f, %.3f)...",
                              idx, *grasp_info_t["final_grasp_xyz"])

            total_n = len(all_targets)
            candidate_k = idx + 1
            while True:
                _t0 = time.perf_counter()
                success, plan_t = self.joint_plan_execute(
                    ps_pre_t, f"Pre-Grasp #{idx}", execute=False)
                self._planning_total_s += time.perf_counter() - _t0
                if not success:
                    rospy.logwarn("[p2p] Pre-Grasp 候選 #%d 規劃失敗，嘗試下一個", idx)
                    break  # 下一個候選

                plan_len = self._plan_length(plan_t)
                if plan_len > 15.0:
                    rospy.logwarn("[p2p] 候選 %d-%d 路徑過長 (%.2f rad > 15 rad)，自動重新規劃",
                                  total_n, candidate_k, plan_len)
                    continue

                _t1 = time.perf_counter()
                try:
                    print("\n==================================================")
                    ans = input(f"⚠️ [安全鎖] 候選 {total_n}-{candidate_k} 軌跡已顯示在 RViz！\n"
                                "  [Enter] 執行  |  [r] 重新規劃  |  [n] 取消：")
                    print("==================================================\n")
                except EOFError:
                    ans = ""
                self._human_confirm_s += time.perf_counter() - _t1

                if ans.lower() == 'n':
                    return
                elif ans.lower() == 'r':
                    rospy.loginfo("[p2p] 重新規劃候選 %d-%d...", total_n, candidate_k)
                    continue
                else:
                    rospy.loginfo("[p2p] 候選規劃成功 %d-%d (%.2f rad)",
                                  total_n, candidate_k, plan_len)
                    plan = plan_t
                    ps_grasp = ps_grasp_t
                    grasp_info = grasp_info_t
                    break

            if plan is not None:
                break  # 找到可用候選

        if plan is None:
            rospy.logerr("[p2p] 所有 %d 個候選均規劃失敗", len(all_targets))
            return

        ee_z = grasp_info["ee_z"]
        grasp_xyz = grasp_info["final_grasp_xyz"]
        self._active_handover_orientation = ps_grasp.pose.orientation

        self._phase2_start_time = time.perf_counter()
        if not self.group.execute(plan, wait=True):
            cur = self.group.get_current_pose().pose
            dx = cur.position.x - ps_pre.pose.position.x
            dy = cur.position.y - ps_pre.pose.position.y
            dz = cur.position.z - ps_pre.pose.position.z
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            if dist < 0.01:
                rospy.logwarn("[p2p] Pre-Grasp execute 回傳失敗但末端已到位 (err=%.4f m)，視為成功", dist)
            else:
                rospy.logerr("[p2p] Pre-Grasp 執行失敗"); return
        self.group.stop()

        # ── 動作 B：Cartesian 前進至 Grasp ──────────────────────────────── #
        rospy.loginfo("[p2p] 執行直線抓取...")
        if not self.plan_execute_cartesian_to(ps_grasp): return

        # ── 動作 C：夾緊 ────────────────────────────────────────────────── #
        if self.g: self.g.move_and_wait_for_pos(255, self.grip_speed, self.grip_force)
        rospy.loginfo("[p2p] 夾爪夾緊")

        if self.pause_after_grasp:
            try:
                input("[p2p] 已夾取，按 [Enter] 繼續放置，[n] 取消：")
            except EOFError:
                pass

        # ── 動作 D：Cartesian 垂直抬升 10cm ─────────────────────────────── #
        rospy.loginfo("[p2p] 垂直抬升 10cm...")
        lift_pose = self.make_pose_stamped(
            [grasp_xyz[0], grasp_xyz[1], grasp_xyz[2] + 0.1],
            ps_grasp.pose.orientation)
        if not self.plan_execute_cartesian_to(lift_pose): return

        # ── 動作 D.5：PCA 旋轉（合入交接姿態，Step E 一次到位）────────────── #
        # pca_rotation_enabled=false (預設) → 整段 no-op，不影響現有流程
        _grasp_orientation_pre_pca = self._active_handover_orientation  # 備份：無 PCA 的原始抓取方向
        # 姿態誤差指標 baseline：記錄 PCA 前的抓取方向與物件 3D 主軸（不論 pca_rotation 是否啟用）
        self._grasp_orientation = _grasp_orientation_pre_pca
        if self._latest_object_points_raw is not None:
            _raw_base = self._latest_object_points_raw
            _pts_base_metric = self._transform_points_to_base(
                _raw_base["points"], _raw_base["frame_id"])
            self._pca_axis_at_grasp = self._pca_axis_3d(_pts_base_metric)
            if self._pca_axis_at_grasp is not None:
                rospy.loginfo("[metric] baseline 物件 3D 主軸(base): (%.3f, %.3f, %.3f)",
                              *self._pca_axis_at_grasp)
        if self.pca_rotation_enabled and self._latest_object_points_raw is not None:
            raw = self._latest_object_points_raw
            pts_base = self._transform_points_to_base(raw["points"], raw["frame_id"])
            if pts_base is not None:
                ee_z_lift = self.get_ee_z_axis_in_base(lift_pose)
                tool0_xyz = np.array([lift_pose.pose.position.x,
                                       lift_pose.pose.position.y,
                                       lift_pose.pose.position.z])
                object_pos = tool0_xyz + ee_z_lift * self.tcp_offset

                grip_end_base = None
                if raw.get("grip_end_3d") is not None:
                    ge_cam = raw["grip_end_3d"].reshape(1, 3)
                    ge_base = self._transform_points_to_base(ge_cam, raw["frame_id"])
                    if ge_base is not None:
                        grip_end_base = ge_base[0]

                angle_deg = self._calculate_pca_rotation_deg(
                    pts_base, object_pos, self.handover_zone_center_xyz,
                    grip_end_3d_base=grip_end_base)

                if abs(angle_deg) >= self.pca_min_angle_deg:
                    q_cur = [lift_pose.pose.orientation.x, lift_pose.pose.orientation.y,
                             lift_pose.pose.orientation.z, lift_pose.pose.orientation.w]
                    q_rot_world = quaternion_from_euler(0, 0, math.radians(angle_deg))
                    q_new = quaternion_multiply(q_rot_world, q_cur)  # 左乘=繞世界 Z 軸
                    self._active_handover_orientation = Quaternion(
                        x=q_new[0], y=q_new[1], z=q_new[2], w=q_new[3])
                    rospy.loginfo("[pca] 旋轉 %.1f° 併入交接姿態，Step E 合併執行", angle_deg)
                else:
                    rospy.loginfo("[pca] 角度 %.1f° < 閾值 %.1f°，跳過旋轉",
                                  angle_deg, self.pca_min_angle_deg)
        elif self.pca_rotation_enabled:
            rospy.logwarn("[pca] 已啟用但尚未收到物件點雲，跳過旋轉")

        # ── 動作 E：先移動到交接區入口點，再開始微調/追蹤 ───────────────── #
        # Old behavior used default_handover_pose.rpy_deg here. Keep grasp
        # orientation through the whole handover stage unless unavailable.
        ps_default_handover = self.make_default_handover_pose(
            orientation=self._active_handover_orientation)
        ps_handover_entry, handover_entry_object_xyz, used_zone_entry = \
            self.make_handover_entry_pose(
                lift_pose,
                orientation=self._active_handover_orientation)
        target_label = "handover_zone_entry" if used_zone_entry else "default_handover_pose"
        rospy.loginfo(
            "[p2p] 移動至 %s object=(%.3f, %.3f, %.3f), tool0=(%.3f, %.3f, %.3f) ...",
            target_label,
            handover_entry_object_xyz[0],
            handover_entry_object_xyz[1],
            handover_entry_object_xyz[2],
            ps_handover_entry.pose.position.x,
            ps_handover_entry.pose.position.y,
            ps_handover_entry.pose.position.z)
        ok, _ = self.joint_plan_execute(ps_handover_entry, target_label)

        # ── Fallback 1：去掉 PCA 旋轉，用原始抓取方向重試 ───────────────── #
        if not ok and self.handover_fallback_orientation \
                and self._active_handover_orientation is not None \
                and _grasp_orientation_pre_pca is not self._active_handover_orientation:
            rospy.logwarn("[p2p] %s PCA 旋轉方向規劃失敗，改用原始抓取方向重試...", target_label)
            self._active_handover_orientation = _grasp_orientation_pre_pca
            ps_handover_entry, handover_entry_object_xyz, used_zone_entry = \
                self.make_handover_entry_pose(
                    lift_pose, orientation=self._active_handover_orientation)
            ok, _ = self.joint_plan_execute(ps_handover_entry, target_label + "_no_pca")

        # ── Fallback 2：改垂直俯抓方向，原地重新定向再重試 ──────────────── #
        if not ok and self.handover_fallback_orientation \
                and self._active_handover_orientation is not None:
            rospy.logwarn(
                "[p2p] %s 以抓取姿態規劃失敗，先原地重新定向再前往交接區...", target_label)
            self._active_handover_orientation = None

            # 步驟 1：在目前 lift 位置原地轉成垂直俯抓方向（物體位置不動）
            lift_tool0_xyz = np.array([
                lift_pose.pose.position.x,
                lift_pose.pose.position.y,
                lift_pose.pose.position.z], dtype=float)
            current_ee_z = self.get_ee_z_axis_in_base(lift_pose)
            lift_object_xyz = lift_tool0_xyz + current_ee_z * self.tcp_offset
            reoriented_obj_pose = self.make_pose_stamped_from_xyz_rpy(
                lift_object_xyz.tolist(), self.default_handover_pose_rpy_deg)
            new_ee_z = self.get_ee_z_axis_in_base(reoriented_obj_pose)
            reoriented_tool0_xyz = lift_object_xyz - new_ee_z * self.tcp_offset
            ps_reoriented = self.make_pose_stamped(
                reoriented_tool0_xyz, reoriented_obj_pose.pose.orientation)
            rospy.loginfo(
                "[p2p] 原地重新定向目標: object=(%.3f, %.3f, %.3f) tool0=(%.3f, %.3f, %.3f)",
                *lift_object_xyz, *reoriented_tool0_xyz)
            ok_reorient, _ = self.joint_plan_execute(ps_reoriented, "reorient_at_lift")
            if not ok_reorient:
                rospy.logwarn("[p2p] 原地重新定向規劃失敗，放棄 fallback")
                ok = False
            else:
                lift_pose = ps_reoriented  # 以重新定向後的位置作為起點

                # 步驟 2：用垂直俯抓方向規劃到交接區入口
                ps_handover_entry, handover_entry_object_xyz, used_zone_entry = \
                    self.make_handover_entry_pose(lift_pose, orientation=None)
                rospy.loginfo(
                    "[p2p] fallback entry: object=(%.3f, %.3f, %.3f) tool0=(%.3f, %.3f, %.3f)",
                    handover_entry_object_xyz[0],
                    handover_entry_object_xyz[1],
                    handover_entry_object_xyz[2],
                    ps_handover_entry.pose.position.x,
                    ps_handover_entry.pose.position.y,
                    ps_handover_entry.pose.position.z)
                ok, _ = self.joint_plan_execute(
                    ps_handover_entry, target_label + "_fallback")
        if not ok:
            return

        # ── 動作 F：到達交接區入口後，可選一次性微調或持續追蹤 ─────────── #
        active_handover_pose = ps_handover_entry
        if self.enable_handover_fine_adjust and not self.handover_tracking_enabled:
            _, hand_state = self._wait_for_stable_hand_sample()
            if hand_state is None:
                if self._abort_requested:
                    rospy.logwarn("[p2p] handover 被 abort，保持夾持")
                    return
                rospy.logwarn("[p2p] 沒有穩定 hand_state，維持目前交接等待姿態等待力矩交接")
            else:
                adjusted_pose = self.try_fine_adjust_handover_pose(
                    hand_state,
                    orientation=self._active_handover_orientation)
                if adjusted_pose is not None:
                    active_handover_pose = adjusted_pose
                    rospy.loginfo("[p2p] 已完成 handover fine adjust")
                else:
                    rospy.logwarn("[p2p] fine adjust 不可用，維持目前交接等待姿態等待力矩交接")
        else:
            if self.handover_tracking_enabled:
                rospy.loginfo("[p2p] handover tracking 已啟用，進入等待期間持續低頻微調")
            else:
                rospy.loginfo("[p2p] handover fine adjust 已停用，使用目前交接等待姿態")

        # ── 動作 G：等待力矩觸發交接；tracking 模式下會持續低頻微調 ─────── #
        rospy.loginfo(
            "[p2p] 到達交接等待姿態 tool0=(%.3f, %.3f, %.3f)，等待力矩觸發放手...",
            active_handover_pose.pose.position.x,
            active_handover_pose.pose.position.y,
            active_handover_pose.pose.position.z)
        # 交接姿態誤差指標：請求 server 重觀測物件、比對 3D 主軸
        self._request_and_compute_handover_error()
        ok, active_handover_pose = self._track_and_wait_for_handover(
            active_handover_pose,
            handover_orientation=self._active_handover_orientation)
        if not ok:
            rospy.logwarn("[p2p] 未偵測到有效交接拉力，保持夾持")
            return

        # ── 動作 H：張開夾爪 ────────────────────────────────────────────── #
        if self.g: self.g.move_and_wait_for_pos(0, self.grip_speed, self.grip_force)
        rospy.loginfo("[p2p] 夾爪張開，物體已交接")

        if self._phase2_start_time is not None:
            release_elapsed = time.perf_counter() - self._phase2_start_time
            rospy.loginfo("[p2p] ⏱️ 抓取→交接→釋放 耗時: %.2f s", release_elapsed)
        self._phase2_release_done = True

        rospy.loginfo("[p2p] 🎉 交接完成！")


if __name__ == "__main__":
    rospy.init_node("semantic_grasp_controller")
    app = SemanticGraspController(); rospy.spin()
