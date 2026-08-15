#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import io
import rospkg
import queue
import cv2
import json
import time
import threading
import open3d as o3d

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

try:
    import rospy
    import message_filters
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from geometry_msgs.msg import Pose
    from visualization_msgs.msg import Marker, MarkerArray
    import tf2_ros
    print("✅ 成功跨界連接 ROS Noetic！")
except ImportError:
    print("❌ 找不到 ROS")

import torch
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from gsnet import AnyGrasp


def imgmsg_to_numpy(msg):
    dtype_class = np.uint8
    channels = 1
    if "rgb8" in msg.encoding or "bgr8" in msg.encoding:
        channels = 3
    elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
        dtype_class = np.uint16
    elif "32FC1" in msg.encoding:
        dtype_class = np.float32
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
    else:
        img = img.reshape((msg.height, msg.width))
    return img


class _Open3DViewer:
    """非阻塞 Open3D 視覺化視窗：在獨立 daemon thread 中持續 poll_events。"""

    def __init__(self, window_name, x=0, y=550, width=700, height=520):
        self._window_name = window_name
        self._x, self._y = x, y
        self._width, self._height = width, height
        self._geom_queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=self._window_name,
                width=self._width, height=self._height)
            opt = vis.get_render_option()
            opt.background_color = np.array([0.15, 0.15, 0.15])
            opt.point_size = 2.5
            current_geoms = []
            close_count = 0
            while True:
                try:
                    new_geoms = self._geom_queue.get_nowait()
                    for g in current_geoms:
                        vis.remove_geometry(g, reset_bounding_box=False)
                    current_geoms = list(new_geoms)
                    first = True
                    for g in current_geoms:
                        vis.add_geometry(g, reset_bounding_box=first)
                        first = False
                except queue.Empty:
                    pass
                if not vis.poll_events():
                    close_count += 1
                    if close_count >= 5:
                        break   # 連續 5 次才真的認定視窗被關閉
                else:
                    close_count = 0
                vis.update_renderer()
                time.sleep(0.02)
        except Exception as e:
            try:
                import rospy
                rospy.logwarn(f"[Viewer] {self._window_name} 錯誤: {e}")
            except Exception:
                print(f"[Viewer] {self._window_name} 錯誤: {e}")

    def update(self, geom_list):
        """更新顯示幾何體；佇列已滿時丟棄舊幀。"""
        try:
            self._geom_queue.get_nowait()
        except queue.Empty:
            pass
        self._geom_queue.put(list(geom_list))


class Config:
    def __init__(self):
        self.checkpoint_path = './log/checkpoint_detection.tar'
        self.max_gripper_width = 0.085
        self.gripper_height = 0.04
        self.top_down_grasp = False
        # 視覺化開關：要開啟時改成 True
        self.debug = True


class AnyGraspHandoverNode:
    def __init__(self):
        rospy.init_node('anygrasp_handover_node', anonymous=True)
        self.cfgs = Config()
        self.fx, self.fy = 462.16757, 462.16757
        self.cx, self.cy = 320.5, 240.5
        self.mask_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "my_gazebo_data")

        rospy.loginfo("🦾 正在載入 AnyGrasp...")
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("✅ AnyGrasp 載入完成！")

        self.color_sub = message_filters.Subscriber(
            '/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10, slop=0.5, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/system/trigger_detection", String,
                         self.trigger_callback)

        self.plan_pub = rospy.Publisher(
            "/anygrasp/handover_plan", String, queue_size=1)
        self.grasp_marker_pub = rospy.Publisher(
            "/anygrasp/grasp_markers", MarkerArray, queue_size=1, latch=True)

        self.need_detection = False
        self.target_object = "unknown_object"
        self.mode = "dual"  # "dual" 或 "left_only"
        
        # UR3 手臂連桿 mesh 路徑（動態解析套件位置，不寫死使用者/工作區）
        _rospack = rospkg.RosPack()
        self.arm_mesh_dir = os.path.join(
            _rospack.get_path('ur_description'), "meshes/ur3/collision")

        # 只過濾靠近指尖的連桿（forearm + 三個 wrist）
        self.arm_links = {
            "rightarm_forearm_link":  "forearm.stl",
            "rightarm_wrist_1_link":  "wrist1.stl",
            "rightarm_wrist_2_link":  "wrist2.stl",
            "rightarm_wrist_3_link":  "wrist3.stl",
        }

        # 夾爪 mesh 路徑與 link 對應表
        self.gripper_mesh_dir = os.path.join(
            _rospack.get_path('robotiq_description'), "meshes/collision")
        self.gripper_links = {
            "rightarm_robotiq_85_base_link":
                "robotiq_85_base_link.stl",
            "rightarm_robotiq_85_left_knuckle_link":
                "robotiq_85_knuckle_link.stl",
            "rightarm_robotiq_85_right_knuckle_link":
                "robotiq_85_knuckle_link.stl",
            "rightarm_robotiq_85_left_inner_knuckle_link":
                "robotiq_85_inner_knuckle_link.stl",
            "rightarm_robotiq_85_right_inner_knuckle_link":
                "robotiq_85_inner_knuckle_link.stl",
            "rightarm_robotiq_85_left_finger_link":
                "robotiq_85_finger_link.stl",
            "rightarm_robotiq_85_right_finger_link":
                "robotiq_85_finger_link.stl",
            "rightarm_robotiq_85_left_finger_tip_link":
                "robotiq_85_finger_tip_link.stl",
            "rightarm_robotiq_85_right_finger_tip_link":
                "robotiq_85_finger_tip_link.stl",
        }

        # receiver_only 模式由 trigger 傳入的語意資訊
        self.receiver_centroid_world = None
        self.object_centroid_world   = None
        self.rotation_angle_deg      = 0.0
        # FoundationPose 給的物件 pose (camera frame, 4x4)
        self.object_pose_in_cam = None

        # debug 視覺化：Open3D Visualizer 各自跑在 daemon thread，不阻塞 ROS node
        if self.cfgs.debug:
            self._viewer_giver = _Open3DViewer("AnyGrasp 給予臂",  x=0,   y=550)
            self._viewer_recv  = _Open3DViewer("AnyGrasp 接收臂",  x=720, y=550)

        rospy.loginfo("🤖 AnyGrasp 節點就緒，等待觸發...")

    def _show_as_cv_window(self, geom_list, window_name, **_):
        """將幾何體送給對應的 Open3D 非阻塞視窗。"""
        if not self.cfgs.debug:
            return
        if window_name == "AnyGrasp 給予臂":
            self._viewer_giver.update(geom_list)
        else:
            self._viewer_recv.update(geom_list)

    def trigger_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.target_object = data.get("object_name", "unknown")
            self.mode = data.get("mode", "dual")

            # receiver_only 模式才會有這三個欄位
            rc = data.get("receiver_centroid", None)
            oc = data.get("object_centroid", None)
            self.receiver_centroid_world = (
                np.array(rc) if rc is not None else None)
            self.object_centroid_world = (
                np.array(oc) if oc is not None else None)
            self.rotation_angle_deg = data.get("rotation_angle", 0.0)
            
            # FoundationPose 估測的物件 pose
            op = data.get("object_pose_in_cam", None)
            self.object_pose_in_cam = (
                np.array(op).reshape(4, 4) if op is not None else None)

        except (json.JSONDecodeError, TypeError):
            self.target_object = msg.data
            self.mode = "dual"
            self.object_pose_in_cam = None

        rospy.loginfo(
            f"⚡ 收到 Trigger！物件: {self.target_object}，模式: {self.mode}"
            f"，FP pose: {'有' if self.object_pose_in_cam is not None else '無'}")
        self.need_detection = True

    def to_ros_pose(self, grasp):
        p = Pose()
        p.position.x, p.position.y, p.position.z = grasp.translation
        q = Rotation.from_matrix(grasp.rotation_matrix).as_quat()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        return p

    def publish_grasp_markers(self, arm_grasps, frame_id):
        """將 arm_grasps（get_arm_specific_grasps 輸出）發布為 RViz MarkerArray。
        箭頭 = 接近方向，球 = 夾取中心；按排名從綠（最佳）到紅漸變。"""
        ma = MarkerArray()
        n = len(arm_grasps)
        now = rospy.Time.now()

        # 先清除上一次的 markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        clear.ns = "grasp_candidates"
        ma.markers.append(clear)

        for rank, item in enumerate(arm_grasps):
            g = item['grasp']
            t = g.translation
            R = g.rotation_matrix
            approach = R[:, 0]  # AnyGrasp 接近方向

            ratio = rank / max(n - 1, 1)  # 0=最佳(綠) → 1=最差(紅)
            r_col, g_col = float(ratio), float(1.0 - ratio)

            # 箭頭 marker（接近方向）
            arrow = Marker()
            arrow.header.frame_id = frame_id
            arrow.header.stamp = now
            arrow.ns = "grasp_candidates"
            arrow.id = rank * 2
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = float(t[0])
            arrow.pose.position.y = float(t[1])
            arrow.pose.position.z = float(t[2])
            q = Rotation.from_matrix(R).as_quat()
            arrow.pose.orientation.x = float(q[0])
            arrow.pose.orientation.y = float(q[1])
            arrow.pose.orientation.z = float(q[2])
            arrow.pose.orientation.w = float(q[3])
            arrow.scale.x = 0.06  # 箭頭長
            arrow.scale.y = 0.008
            arrow.scale.z = 0.008
            arrow.color.r = r_col
            arrow.color.g = g_col
            arrow.color.b = 0.0
            arrow.color.a = 0.85
            arrow.lifetime = rospy.Duration(0)
            ma.markers.append(arrow)

            # 球 marker（夾取位置）
            sphere = Marker()
            sphere.header.frame_id = frame_id
            sphere.header.stamp = now
            sphere.ns = "grasp_candidates"
            sphere.id = rank * 2 + 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(t[0])
            sphere.pose.position.y = float(t[1])
            sphere.pose.position.z = float(t[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.015
            sphere.color.r = r_col
            sphere.color.g = g_col
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            sphere.lifetime = rospy.Duration(0)
            ma.markers.append(sphere)

        self.grasp_marker_pub.publish(ma)
        rospy.loginfo(f"[viz] 發布 {n} 個夾取姿態 markers → /anygrasp/grasp_markers")

    def pose_to_dict(self, pose):
        return {
            'position': {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': pose.position.z
            },
            'orientation': {
                'x': pose.orientation.x,
                'y': pose.orientation.y,
                'z': pose.orientation.z,
                'w': pose.orientation.w
            }
        }

    def get_arm_base_position(self, arm_name, ref_frame):
        try:
            base_link_name = f"{arm_name}_base_link"
            trans = self.tf_buffer.lookup_transform(
                ref_frame, base_link_name, rospy.Time(0), rospy.Duration(1.0))
            return np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except Exception:
            if "left" in arm_name:
                return np.array([-0.5, 0.2, 0.0])
            else:
                return np.array([0.5, 0.2, 0.0])

    def filter_outlier_points(self, pts):
        if pts.shape[0] < 20:
            return pts
        q1 = np.percentile(pts[:, 2], 25)
        q3 = np.percentile(pts[:, 2], 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        mask = (pts[:, 2] >= lower_bound) & (pts[:, 2] <= upper_bound)
        clean_pts = pts[mask]
        if clean_pts.shape[0] < 20 or iqr < 0.01:
            z_median = np.median(pts[:, 2])
            mask_fallback = np.abs(pts[:, 2] - z_median) < 0.10
            return pts[mask_fallback]
        return clean_pts

    def get_dynamic_lims(self, points_target, pad=0.0, y_top_pad=0.05):
        if points_target.shape[0] < 20:
            return None
        min_b = points_target.min(axis=0)
        max_b = points_target.max(axis=0)
        return [
            min_b[0] - pad, max_b[0] + pad,
            min_b[1] - y_top_pad, max_b[1] + pad,
            min_b[2] - pad, max_b[2] + pad
        ]
        
    def find_ycb_model_path(self, object_name):
        models_dir = os.path.join(rospkg.RosPack().get_path('ur_gripper_gazebo'), 'models')
        for folder in sorted(os.listdir(models_dir)):
            if object_name.lower() in folder.lower():
                model_path = os.path.join(models_dir, folder, "google_16k", "textured.obj")
                if os.path.exists(model_path):
                    return model_path
        return None

    def refine_points_with_icp(self, pts_observed, model_path):
        try:
            mesh = o3d.io.read_triangle_mesh(model_path)
            if not mesh.has_vertices():
                return pts_observed
            model_pcd = mesh.sample_points_uniformly(number_of_points=5000)

            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(pts_observed)

            obs_centroid = pts_observed.mean(axis=0)
            model_centroid = np.asarray(model_pcd.points).mean(axis=0)

            # 多個初始旋轉角度（繞 Z 軸 0/90/180/270 度）
            best_result = None
            best_fitness = -1.0

            for angle_deg in [0, 90, 180, 270]:
                angle_rad = np.radians(angle_deg)
                Rz = np.array([
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad),  np.cos(angle_rad), 0],
                    [0,                  0,                  1]
                ])
                init_transform = np.eye(4)
                init_transform[:3, :3] = Rz
                # 旋轉後重新對齊重心
                rotated_centroid = Rz @ model_centroid
                init_transform[:3, 3] = obs_centroid - rotated_centroid

                result = o3d.pipelines.registration.registration_icp(
                    source=model_pcd,
                    target=obs_pcd,
                    max_correspondence_distance=0.05,
                    init=init_transform,
                    estimation_method=o3d.pipelines.registration
                        .TransformationEstimationPointToPoint()  # 不用法向量
                )

                if result.fitness > best_fitness:
                    best_fitness = result.fitness
                    best_result = result

            rospy.loginfo(
                f"📐 ICP 完成 | best_fitness={best_fitness:.3f}, "
                f"rmse={best_result.inlier_rmse*1000:.1f}mm")

            if best_fitness < 0.3:
                rospy.logwarn(
                    f"⚠️ ICP fitness 過低 ({best_fitness:.3f})，使用原始點雲")
                return pts_observed

            model_pcd.transform(best_result.transformation)
            return np.asarray(model_pcd.points)

        except Exception as e:
            rospy.logwarn(f"⚠️ ICP 補全失敗: {e}，使用原始點雲")
            return pts_observed
        
    def remove_gripper_points(self, points, colors,
                          camera_frame, threshold=0.012):
        """移除場景點雲中屬於右手夾爪和手臂連桿的點"""
        gripper_only_pts = []   # 夾爪點（精細閾值 0.012m）
        arm_only_pts = []       # 手臂點（寬鬆閾值 0.03m）

        # 夾爪 links
        for link_name, mesh_file in self.gripper_links.items():
            try:
                trans = self.tf_buffer.lookup_transform(
                    camera_frame, link_name,
                    rospy.Time(0), rospy.Duration(1.0))
                t = trans.transform.translation
                q = trans.transform.rotation
                T = np.eye(4)
                T[:3, 3] = [t.x, t.y, t.z]
                T[:3, :3] = Rotation.from_quat(
                    [q.x, q.y, q.z, q.w]).as_matrix()

                mesh_path = os.path.join(self.gripper_mesh_dir, mesh_file)
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                if not mesh.has_vertices():
                    continue

                pcd = mesh.sample_points_uniformly(number_of_points=500)
                pts_local = np.asarray(pcd.points)
                pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1))])
                pts_cam = (T @ pts_h.T).T[:, :3]
                gripper_only_pts.append(pts_cam)

            except Exception as e:
                rospy.logwarn(f"⚠️ 無法處理 {link_name}: {e}")
                continue

        # 手臂 links（採樣更密，閾值更大）
        for link_name, mesh_file in self.arm_links.items():
            try:
                trans = self.tf_buffer.lookup_transform(
                    camera_frame, link_name,
                    rospy.Time(0), rospy.Duration(1.0))
                t = trans.transform.translation
                q = trans.transform.rotation
                T = np.eye(4)
                T[:3, 3] = [t.x, t.y, t.z]
                T[:3, :3] = Rotation.from_quat(
                    [q.x, q.y, q.z, q.w]).as_matrix()

                mesh_path = os.path.join(self.arm_mesh_dir, mesh_file)
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                if not mesh.has_vertices():
                    continue

                pcd = mesh.sample_points_uniformly(number_of_points=2000)  # ← 更密
                pts_local = np.asarray(pcd.points)
                pts_h = np.hstack([pts_local, np.ones((len(pts_local), 1))])
                pts_cam = (T @ pts_h.T).T[:, :3]
                arm_only_pts.append(pts_cam)

            except Exception as e:
                rospy.logwarn(f"⚠️ 無法處理手臂 {link_name}: {e}")
                continue

        # 分層過濾
        mask = np.ones(len(points), dtype=bool)

        if gripper_only_pts:
            gripper_pts = np.vstack(gripper_only_pts).astype(np.float32)
            tree_g = cKDTree(gripper_pts)
            d_g, _ = tree_g.query(points, k=1, workers=-1)
            mask &= (d_g > 0.012)       # 夾爪精細閾值

        if arm_only_pts:
            arm_pts = np.vstack(arm_only_pts).astype(np.float32)
            tree_a = cKDTree(arm_pts)
            d_a, _ = tree_a.query(points, k=1, workers=-1)
            mask &= (d_a > 0.03)        # 手臂寬鬆閾值

        if not gripper_only_pts and not arm_only_pts:
            rospy.logwarn("⚠️ 夾爪和手臂點雲都建立失敗，跳過移除步驟")
            return points, colors

        removed = int(np.sum(~mask))
        rospy.loginfo(
            f"🔧 夾爪+手臂點雲移除: {removed}/{len(points)} 點"
            f" ({removed/len(points)*100:.1f}%)")

        return points[mask], colors[mask]
    
    def extract_object_roi(self, points_clean, camera_frame, radius=0.20):
        """
        用指尖中點 TF 定義 ROI
        只保留物件附近的點，排除大部分背景
        """
        try:
            tip_l = self.tf_buffer.lookup_transform(
                camera_frame,
                "rightarm_robotiq_85_left_finger_tip_link",
                rospy.Time(0), rospy.Duration(1.0))
            tip_r = self.tf_buffer.lookup_transform(
                camera_frame,
                "rightarm_robotiq_85_right_finger_tip_link",
                rospy.Time(0), rospy.Duration(1.0))

            tip_l_pos = np.array([
                tip_l.transform.translation.x,
                tip_l.transform.translation.y,
                tip_l.transform.translation.z])
            tip_r_pos = np.array([
                tip_r.transform.translation.x,
                tip_r.transform.translation.y,
                tip_r.transform.translation.z])

            # 指尖中點 = 夾爪實際夾住物件的位置
            gripper_center = (tip_l_pos + tip_r_pos) / 2.0

            dists = np.linalg.norm(points_clean - gripper_center, axis=1)
            roi_mask = dists < radius
            points_roi = points_clean[roi_mask]

            rospy.loginfo(
                f"📦 ROI 過濾: {len(points_roi)}/{len(points_clean)} 點"
                f" (半徑: {radius}m)")

            if len(points_roi) < 50:
                rospy.logwarn("⚠️ ROI 點數不足，使用完整點雲")
                return points_clean

            return points_roi

        except Exception as e:
            rospy.logwarn(f"⚠️ ROI 過濾失敗: {e}，使用完整點雲")
            return points_clean
    
    # def complete_object_with_foundationpose_pose(self, object_pose_in_cam):
    #     """
    #     用 FoundationPose 給的 pose，把 YCB mesh 直接變換到 camera frame
    #     再 sample 出點雲。不做 ICP 微調，純粹 transform + sample。
        
    #     Args:
    #         object_pose_in_cam: 4x4 numpy array, mesh→camera 的變換
        
    #     Returns:
    #         np.ndarray (N, 3) float32, 或 None（找不到 mesh）
    #     """
    #     model_path = self.find_ycb_model_path(self.target_object)
    #     if not model_path:
    #         rospy.logwarn(
    #             f"⚠️ 找不到 {self.target_object} 的模型，無法用 FP 補全")
    #         return None
        
    #     mesh = o3d.io.read_triangle_mesh(model_path)
    #     if not mesh.has_vertices():
    #         rospy.logwarn(f"⚠️ Mesh 無頂點: {model_path}")
    #         return None
        
    #     mesh.transform(object_pose_in_cam)
    #     pcd = mesh.sample_points_uniformly(number_of_points=5000)
    #     points = np.asarray(pcd.points).astype(np.float32)
        
    #     rospy.loginfo(
    #         f"✓ FoundationPose 補全完成: {len(points)} 點 "
    #         f"(pose translation: {object_pose_in_cam[:3,3]})")
        
    #     return points
    
    def complete_object_with_foundationpose_pose(self, object_pose_in_cam):
        model_path = self.find_ycb_model_path(self.target_object)
        if not model_path:
            rospy.logwarn(
                f"⚠️ 找不到 {self.target_object} 的模型，無法用 FP 補全")
            return None

        mesh = o3d.io.read_triangle_mesh(model_path)
        if not mesh.has_vertices():
            rospy.logwarn(f"⚠️ Mesh 無頂點: {model_path}")
            return None

        mesh.transform(object_pose_in_cam)
        pcd = mesh.sample_points_uniformly(number_of_points=5000)
        points = np.asarray(pcd.points).astype(np.float32)

        rospy.loginfo(
            f"✓ FoundationPose 補全完成: {len(points)} 點 "
            f"(pose translation: {object_pose_in_cam[:3,3]})")

        return points
    
    def complete_object_with_icp_camera_frame(self, pts_clean, camera_frame):
        """扣除夾爪後，用 ICP 將 YCB mesh 對齊殘缺點雲，補全物件"""
        model_path = self.find_ycb_model_path(self.target_object)
        if not model_path:
            rospy.logwarn(
                f"⚠️ 找不到 {self.target_object} 的模型，跳過補全")
            return pts_clean

        try:
            mesh = o3d.io.read_triangle_mesh(model_path)
            if not mesh.has_vertices():
                return pts_clean
            model_pcd = mesh.sample_points_uniformly(number_of_points=5000)

            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(pts_clean)
            # obs_pcd, _ = obs_pcd.remove_statistical_outlier(
            #     nb_neighbors=20, std_ratio=1.5)
            # rospy.loginfo(
            #     f"📦 離群點過濾後: {len(obs_pcd.points)} 點")

            # 指尖中點當位置初始值
            try:
                tip_l = self.tf_buffer.lookup_transform(
                    camera_frame,
                    "rightarm_robotiq_85_left_finger_tip_link",
                    rospy.Time(0), rospy.Duration(1.0))
                tip_r = self.tf_buffer.lookup_transform(
                    camera_frame,
                    "rightarm_robotiq_85_right_finger_tip_link",
                    rospy.Time(0), rospy.Duration(1.0))
                tip_l_pos = np.array([
                    tip_l.transform.translation.x,
                    tip_l.transform.translation.y,
                    tip_l.transform.translation.z])
                tip_r_pos = np.array([
                    tip_r.transform.translation.x,
                    tip_r.transform.translation.y,
                    tip_r.transform.translation.z])
                gripper_center = (tip_l_pos + tip_r_pos) / 2.0
                rospy.loginfo(f"📍 指尖中點: {gripper_center.round(3)}")
            except Exception as e:
                rospy.logwarn(f"⚠️ 指尖 TF 失敗: {e}，用點雲質心")
                gripper_center = pts_clean.mean(axis=0)

            model_centroid = np.asarray(model_pcd.points).mean(axis=0)

            # rotation_angle 換到相機座標系當旋轉初始值
            try:
                trans_cw = self.tf_buffer.lookup_transform(
                    camera_frame, "world",
                    rospy.Time(0), rospy.Duration(1.0))
                q = trans_cw.transform.rotation
                R_cw = Rotation.from_quat(
                    [q.x, q.y, q.z, q.w]).as_matrix()
                a = np.radians(self.rotation_angle_deg)
                c, s = np.cos(a), np.sin(a)
                Rz_world = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                R_init = R_cw @ Rz_world @ R_cw.T
            except Exception:
                R_init = np.eye(3)

            # 以 rotation_angle 為基礎嘗試四個方向（處理模型初始朝向的不確定性）
            best_result, best_fitness = None, -1.0
            for extra_deg in [0, 90, 180, 270]:
                ea = np.radians(extra_deg)
                ce, se = np.cos(ea), np.sin(ea)
                R_extra = np.array([[ce, -se, 0], [se, ce, 0], [0, 0, 1]])
                R_combined = R_init @ R_extra

                init_T = np.eye(4)
                init_T[:3, :3] = R_combined
                init_T[:3, 3] = gripper_center - R_combined @ model_centroid

                result = o3d.pipelines.registration.registration_icp(
                    source=model_pcd, target=obs_pcd,
                    max_correspondence_distance=0.02,
                    init=init_T,
                    estimation_method=o3d.pipelines.registration
                        .TransformationEstimationPointToPoint())

                if result.fitness > best_fitness:
                    best_fitness = result.fitness
                    best_result = result

            rospy.loginfo(
                f"📐 ICP 補全 | fitness={best_fitness:.3f},"
                f" rmse={best_result.inlier_rmse*1000:.1f}mm")

            if best_fitness < 0.2:
                rospy.logwarn(
                    f"⚠️ ICP fitness 過低 ({best_fitness:.3f})，使用殘缺點雲")
                return pts_clean

            model_pcd.transform(best_result.transformation)
            completed = np.asarray(model_pcd.points).astype(np.float32)
            
            if self.cfgs.debug:
                trans_mat = np.array(
                    [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

                obs_vis = o3d.geometry.PointCloud()
                obs_vis.points = o3d.utility.Vector3dVector(pts_clean)
                obs_vis.paint_uniform_color([0.7, 0.7, 0.7])
                obs_vis.transform(trans_mat)

                comp_vis = o3d.geometry.PointCloud()
                comp_vis.points = o3d.utility.Vector3dVector(completed)
                comp_vis.paint_uniform_color([0.0, 0.8, 0.0])
                comp_vis.transform(trans_mat)

                vis_list = [obs_vis, comp_vis]

                recv_result = self.compute_receiver_lims_from_icp(
                    completed, camera_frame)
                if recv_result is not None:
                    _, recv_pts = recv_result
                    if len(recv_pts) > 0:
                        recv_vis = o3d.geometry.PointCloud()
                        recv_vis.points = o3d.utility.Vector3dVector(recv_pts)
                        recv_vis.paint_uniform_color([1, 0, 0])
                        recv_vis.transform(trans_mat)
                        vis_list.append(recv_vis)

                self._show_as_cv_window(vis_list, "AnyGrasp 給予臂")
                rospy.loginfo("👀 [Debug] ICP 補全確認（左下角視窗）")
    
            rospy.loginfo(
                f"✅ 補全完成: {len(pts_clean)} → {len(completed)} 點")
            return completed

        except Exception as e:
            rospy.logwarn(f"⚠️ ICP 補全失敗: {e}")
            return pts_clean
    
    def complete_object_with_ransac_icp_camera_frame(self, pts_clean, camera_frame):
        """FPFH RANSAC 全域配準 + ICP 精化補全物件點雲"""
        model_path = self.find_ycb_model_path(self.target_object)
        if not model_path:
            rospy.logwarn(
                f"⚠️ 找不到 {self.target_object} 的模型，跳過補全")
            return pts_clean

        try:
            mesh = o3d.io.read_triangle_mesh(model_path)
            if not mesh.has_vertices():
                return pts_clean
            model_pcd = mesh.sample_points_uniformly(number_of_points=5000)

            obs_pcd = o3d.geometry.PointCloud()
            obs_pcd.points = o3d.utility.Vector3dVector(pts_clean)
            obs_pcd, _ = obs_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=1.5)
            rospy.loginfo(
                f"📦 離群點過濾後: {len(obs_pcd.points)} 點")

            if len(obs_pcd.points) < 50:
                rospy.logwarn("⚠️ 觀測點雲點數過少，跳過 RANSAC 補全")
                return pts_clean

            # 計算法向量
            obs_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            model_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            # 計算 FPFH 特徵
            obs_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                obs_pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))
            model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                model_pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100))

            # RANSAC 全域配準
            result_ransac = o3d.pipelines.registration\
                .registration_ransac_based_on_feature_matching(
                    source=model_pcd,
                    target=obs_pcd,
                    source_feature=model_fpfh,
                    target_feature=obs_fpfh,
                    mutual_filter=True,
                    max_correspondence_distance=0.03,
                    estimation_method=o3d.pipelines.registration
                        .TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    checkers=[
                        o3d.pipelines.registration
                            .CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        o3d.pipelines.registration
                            .CorrespondenceCheckerBasedOnDistance(0.03)
                    ],
                    criteria=o3d.pipelines.registration
                        .RANSACConvergenceCriteria(100000, 0.999))

            rospy.loginfo(
                f"🔍 RANSAC | fitness={result_ransac.fitness:.3f},"
                f" rmse={result_ransac.inlier_rmse*1000:.1f}mm")

            if result_ransac.fitness < 0.15:
                rospy.logwarn(
                    f"⚠️ RANSAC fitness 過低 ({result_ransac.fitness:.3f})，使用殘缺點雲")
                return pts_clean

            # ICP 精化
            result_icp = o3d.pipelines.registration.registration_icp(
                source=model_pcd,
                target=obs_pcd,
                max_correspondence_distance=0.015,
                init=result_ransac.transformation,
                estimation_method=o3d.pipelines.registration
                    .TransformationEstimationPointToPoint())

            rospy.loginfo(
                f"📐 ICP 精化 | fitness={result_icp.fitness:.3f},"
                f" rmse={result_icp.inlier_rmse*1000:.1f}mm")

            if result_icp.fitness < 0.2:
                rospy.logwarn(
                    f"⚠️ ICP fitness 過低 ({result_icp.fitness:.3f})，使用殘缺點雲")
                return pts_clean

            model_pcd.transform(result_icp.transformation)
            completed = np.asarray(model_pcd.points).astype(np.float32)

            if self.cfgs.debug:
                trans_mat = np.array(
                    [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

                obs_vis = o3d.geometry.PointCloud()
                obs_vis.points = o3d.utility.Vector3dVector(pts_clean)
                obs_vis.paint_uniform_color([0.7, 0.7, 0.7])
                obs_vis.transform(trans_mat)

                comp_vis = o3d.geometry.PointCloud()
                comp_vis.points = o3d.utility.Vector3dVector(completed)
                comp_vis.paint_uniform_color([0.0, 0.8, 0.0])
                comp_vis.transform(trans_mat)

                vis_list = [obs_vis, comp_vis]

                recv_result = self.compute_receiver_lims_from_icp(
                    completed, camera_frame)
                if recv_result is not None:
                    _, recv_pts = recv_result
                    if len(recv_pts) > 0:
                        recv_vis = o3d.geometry.PointCloud()
                        recv_vis.points = o3d.utility.Vector3dVector(recv_pts)
                        recv_vis.paint_uniform_color([1, 0, 0])
                        recv_vis.transform(trans_mat)
                        vis_list.append(recv_vis)

                self._show_as_cv_window(vis_list, "AnyGrasp 給予臂")
                rospy.loginfo("👀 [Debug] RANSAC+ICP 補全確認（左下角視窗）")

            rospy.loginfo(
                f"✅ 補全完成: {len(pts_clean)} → {len(completed)} 點")
            return completed

        except Exception as e:
            rospy.logwarn(f"⚠️ RANSAC+ICP 補全失敗: {e}")
            return pts_clean

    def compute_receiver_lims(self, points, camera_frame, radius=0.12):
        """
        以 receiver_centroid_world 為中心，直接建固定大小的 lims。
        同時回傳落在框內的點雲供視覺化用。
        """
        if self.receiver_centroid_world is None:
            return None

        try:
            trans = self.tf_buffer.lookup_transform(
                camera_frame, "world",
                rospy.Time(0), rospy.Duration(1.0))
            t = trans.transform.translation
            q = trans.transform.rotation
            T_cw = np.eye(4)
            T_cw[:3, 3] = [t.x, t.y, t.z]
            T_cw[:3, :3] = Rotation.from_quat(
                [q.x, q.y, q.z, q.w]).as_matrix()
            rc_cam = (T_cw @ np.array([*self.receiver_centroid_world, 1.0]))[:3]
        except Exception as e:
            rospy.logwarn(f"⚠️ world→camera 轉換失敗: {e}")
            return None

        lims = [
            float(rc_cam[0]) - radius, float(rc_cam[0]) + radius,
            float(rc_cam[1]) - radius, float(rc_cam[1]) + radius,
            float(rc_cam[2]) - radius, float(rc_cam[2]) + radius,
        ]
        rospy.loginfo(f"📍 receiver_centroid (cam): {rc_cam}")
        rospy.loginfo(f"📦 receiver lims: {[f'{v:.3f}' for v in lims]}")

        # 視覺化用：框內的場景點
        mask = ((points[:, 0] >= lims[0]) & (points[:, 0] <= lims[1]) &
                (points[:, 1] >= lims[2]) & (points[:, 1] <= lims[3]) &
                (points[:, 2] >= lims[4]) & (points[:, 2] <= lims[5]))
        recv_points_vis = points[mask]
        rospy.loginfo(f"📐 框內點數: {len(recv_points_vis)}")

        return lims, recv_points_vis

    def compute_receiver_lims_from_icp(self, points_completed,
                                    camera_frame, pad=0.02):
        """
        從 ICP 補全的點雲上，根據語意方向篩選接收區的點
        不依賴 TF 估算物件位置，直接用 ICP 對齊結果
        """
        if (self.receiver_centroid_world is None
                or self.object_centroid_world is None):
            return None

        # Step 1：計算 receiver 相對方向（世界座標），套上旋轉角度
        rel_world = (self.receiver_centroid_world
                    - self.object_centroid_world)
        a = np.radians(self.rotation_angle_deg)
        c, s = np.cos(a), np.sin(a)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rel_rotated_world = Rz @ rel_world

        # Step 2：把方向向量轉到相機座標系（只旋轉，不平移）
        try:
            trans = self.tf_buffer.lookup_transform(
                camera_frame, "world",
                rospy.Time(0), rospy.Duration(1.0))
            t = trans.transform.translation
            q = trans.transform.rotation
            T_cw = np.eye(4)
            T_cw[:3, 3] = [t.x, t.y, t.z]
            T_cw[:3, :3] = Rotation.from_quat(
                [q.x, q.y, q.z, q.w]).as_matrix()
            rel_cam = T_cw[:3, :3] @ rel_rotated_world
            rel_cam_norm = rel_cam / np.linalg.norm(rel_cam)
        except Exception as e:
            rospy.logwarn(f"⚠️ world→camera 轉換失敗: {e}")
            return None
        
        rospy.loginfo(f"📍 receiver_centroid_world: {self.receiver_centroid_world}")
        rospy.loginfo(f"📍 object_centroid_world:   {self.object_centroid_world}")
        rospy.loginfo(f"📍 rel_world:               {rel_world}")
        rospy.loginfo(f"📍 rotation_angle_deg:      {self.rotation_angle_deg}")
        rospy.loginfo(f"📍 rel_rotated_world:       {rel_rotated_world}")
        rospy.loginfo(f"📍 rel_cam_norm:            {rel_cam_norm}")

        # Step 3：ICP 點雲質心 = 物件真實位置（不靠 TF 估算）
        mesh_center = points_completed.mean(axis=0)

        # Step 4：對每個點算與接收方向的相似度，篩出接收區那一側
        vecs = points_completed - mesh_center
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        directions = vecs / norms
        similarity = directions @ rel_cam_norm

        recv_points = points_completed[similarity > 0.3]
        rospy.loginfo(
            f"📐 接收區篩選: {len(recv_points)}/{len(points_completed)} 點")

        # 點數不足時自動降低閾值
        if len(recv_points) < 10:
            rospy.logwarn("⚠️ 接收區點數不足，降低閾值重試")
            recv_points = points_completed[similarity > 0.1]
        if len(recv_points) < 10:
            rospy.logwarn("⚠️ 接收區點數仍不足，使用全部點")
            recv_points = points_completed

        lims = [
            float(recv_points[:, 0].min()) - pad,
            float(recv_points[:, 0].max()) + pad,
            float(recv_points[:, 1].min()) - pad,
            float(recv_points[:, 1].max()) + pad,
            float(recv_points[:, 2].min()) - pad,
            float(recv_points[:, 2].max()) + pad,
        ]
        rospy.loginfo(
            f"📦 receiver lims (ICP-based): {[f'{v:.3f}' for v in lims]}")
        return lims, recv_points

    def get_arm_specific_grasps(self, gg, arm_name, ref_frame):
        arm_base_pos = self.get_arm_base_position(arm_name, ref_frame)
        arm_grasps = []
        for i in range(len(gg)):
            grasp = gg[i]
            grasp_approach = grasp.rotation_matrix[:, 0]

            if "right" in arm_name:
                if grasp_approach[0] > 0.7:
                    continue
                if grasp_approach[1] > 0.8:
                    continue
                # if grasp_approach[1] < -0.6:
                #     continue  # 排除從正上方往下夾
            else:
                if grasp_approach[0] < -0.3:
                    continue
                if grasp_approach[1] > 0.8:
                    continue

            dist = np.linalg.norm(grasp.translation - arm_base_pos)

            if "left" in arm_name:
                side_bonus = grasp_approach[0]
            else:
                side_bonus = -grasp_approach[0]

            top_bonus = -grasp_approach[1] * 0.5
            arm_score = (5.0 + grasp.score
                         + side_bonus * 2.0
                         + top_bonus
                         - dist * 8.0)

            arm_grasps.append({
                'orig_idx': i,
                'grasp': grasp,
                'arm_score': arm_score
            })

        arm_grasps.sort(key=lambda x: x['arm_score'], reverse=True)
        return arm_grasps
    
    def generate_giver_only_plan(self, right_grasps):
        """
        dual 模式只生成右手姿態，左手等 receiver_only 階段再偵測
        """
        if not right_grasps:
            return []
        json_plan = []
        for i, r_item in enumerate(right_grasps):
            pose_R = self.to_ros_pose(r_item['grasp'])
            json_plan.append({
                'orig_idx_R': i,
                'group_score': r_item['arm_score'],
                'pose_R_table': self.pose_to_dict(pose_R),
                'left_candidates': []
            })
        return json_plan

    def generate_handover_plan(self, right_grasps, left_grasps):
        if not right_grasps or not left_grasps:
            return []

        valid_pairs = []
        for r_item in right_grasps:
            for l_item in left_grasps:
                grasp_R = r_item['grasp']
                grasp_L = l_item['grasp']
                pos_R = grasp_R.translation
                pos_L = grasp_L.translation

                dist = np.linalg.norm(pos_R - pos_L)

                vec_R = grasp_R.rotation_matrix[:, 0]
                vec_L = grasp_L.rotation_matrix[:, 0]
                cos_theta = np.clip(np.dot(vec_R, vec_L), -1.0, 1.0)
                angle_rad = np.arccos(cos_theta)
                
                # 夾角小於 60 度表示兩手接近方向太相似，容易互撞
                # if angle_rad < np.radians(60):
                #     continue

                valid_pairs.append({
                    'r_item': r_item,
                    'l_item': l_item,
                    'dist': dist,
                    'angle_rad': angle_rad,
                    'pos_R': pos_R
                })

        if not valid_pairs:
            return []

        dists = np.array([p['dist'] for p in valid_pairs])
        mu_dist = np.mean(dists)
        std_dist = np.std(dists) + 1e-6

        w_dist = 10.0
        w_angle = 3.0
        w_height = 10.0
        w_arm_self = 2.0

        grouped_pairs = {}
        for p in valid_pairs:
            z_dist = (p['dist'] - mu_dist) / std_dist
            score_height = p['pos_R'][2] * w_height
            total_score = (p['r_item']['arm_score']
                           + p['l_item']['arm_score']
                           + w_dist * z_dist
                           + w_angle * p['angle_rad']
                           + score_height)

            r_idx = p['r_item']['orig_idx']
            if r_idx not in grouped_pairs:
                grouped_pairs[r_idx] = {
                    'orig_idx_R': r_idx,
                    'pose_R': self.to_ros_pose(p['r_item']['grasp']),
                    'r_arm_score': p['r_item']['arm_score'],
                    'max_pair_score': -float('inf'),
                    'left_candidates': []
                }

            grouped_pairs[r_idx]['left_candidates'].append({
                'orig_idx_L': p['l_item']['orig_idx'],
                'score': total_score,
                'pose_L': self.to_ros_pose(p['l_item']['grasp'])
            })

            if total_score > grouped_pairs[r_idx]['max_pair_score']:
                grouped_pairs[r_idx]['max_pair_score'] = total_score

        for r_idx in grouped_pairs:
            grouped_pairs[r_idx]['left_candidates'].sort(
                key=lambda x: x['score'], reverse=True)

        ranked_groups = list(grouped_pairs.values())
        for g in ranked_groups:
            g['group_score'] = (g['r_arm_score'] * w_arm_self
                                + g['max_pair_score'])
        ranked_groups.sort(key=lambda x: x['group_score'], reverse=True)

        json_plan = []
        for group in ranked_groups:
            plan_item = {
                'orig_idx_R': group['orig_idx_R'],
                'group_score': group['group_score'],
                'pose_R_table': self.pose_to_dict(group['pose_R']),
                'left_candidates': []
            }
            for cand in group['left_candidates']:
                plan_item['left_candidates'].append({
                    'orig_idx_L': cand['orig_idx_L'],
                    'score': cand['score'],
                    'pose_L_table': self.pose_to_dict(cand['pose_L'])
                })
            json_plan.append(plan_item)

        return json_plan

    def generate_left_only_plan(self, left_grasps):
        """
        receiver_only 模式：操作臂旋轉物件後重偵測接收臂姿態
        格式與 dual 相同，控制端從 left_candidates 讀取接收臂目標姿態
        """
        if not left_grasps:
            return []
        
        rospy.loginfo(f"[DEBUG] 即將發送 {len(left_grasps)} 個 grasp:")
        for i, l_item in enumerate(left_grasps[:3]):
            g = l_item['grasp']
            R = g.rotation_matrix
            rospy.loginfo(
                f"  Grasp #{i}: t={g.translation.round(3)}, "
                f"approach(col0)={R[:, 0].round(3)}, "
                f"close(col1)={R[:, 1].round(3)}, "
                f"binormal(col2)={R[:, 2].round(3)}, "
                f"score={l_item.get('arm_score', 'N/A')}")
            
        json_plan = []
        for i, l_item in enumerate(left_grasps):
            pose_L = self.to_ros_pose(l_item['grasp'])
            plan_item = {
                'orig_idx_R': i,
                'group_score': l_item['arm_score'],
                # 左手獨立夾取時，控制端從 pose_R_table 讀取目標姿態
                'pose_R_table': self.pose_to_dict(pose_L),
                'left_candidates': [{
                    'orig_idx_L': l_item['orig_idx'],
                    'score': l_item['arm_score'],
                    'pose_L_table': self.pose_to_dict(pose_L)
                }]
            }
            json_plan.append(plan_item)

        return json_plan

    def callback(self, color_msg, depth_msg):
        if not self.need_detection:
            return

        try:
            self.need_detection = False
            mode = self.mode
            rospy.loginfo(f"⚙️ 啟動偵測管線，模式: {mode}")

            color_np = imgmsg_to_numpy(color_msg)
            if "bgr8" in color_msg.encoding:
                color_np = color_np[:, :, ::-1]
            colors = color_np.astype(np.float32) / 255.0

            depths = imgmsg_to_numpy(depth_msg).astype(np.float32)
            max_depth_val = np.nanmax(depths) if np.nanmax(depths) > 0 else 0.0
            scale = 1000.0 if max_depth_val > 100 else 1.0

            receiver_mask_path = os.path.join(self.mask_dir, "receiver_mask.png")
            giver_mask_path = os.path.join(self.mask_dir, "giver_mask.png")

            if mode == "dual":
                if not os.path.exists(receiver_mask_path) or \
                        not os.path.exists(giver_mask_path):
                    rospy.logerr("❌ 找不到 receiver_mask 或 giver_mask")
                    return
                
            xmap, ymap = np.meshgrid(
                np.arange(depths.shape[1]), np.arange(depths.shape[0]))
            points_z = depths / scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            valid_depth_mask = (points_z > 0)
            points_full = np.stack(
                [points_x, points_y, points_z],
                axis=-1)[valid_depth_mask].astype(np.float32)
            colors_full = colors[valid_depth_mask].astype(np.float32)
            
            MAX_POINTS = 30000
            if points_full.shape[0] > MAX_POINTS:
                indices = np.random.choice(
                    points_full.shape[0], MAX_POINTS, replace=False)
                points_full = points_full[indices]
                colors_full = colors_full[indices]

            camera_frame = (color_msg.header.frame_id
                            if color_msg.header.frame_id
                            else "camera_color_optical_frame")
                
            if mode == "receiver_only":
                rospy.loginfo("🧠 receiver_only：FP 補全 → ICP 接收區 → AnyGrasp")

                # Step 1：移除夾爪點雲
                points_clean, colors_clean = self.remove_gripper_points(
                    points_full, colors_full, camera_frame)
                if points_clean.shape[0] < 200:
                    rospy.logwarn("⚠️ 移除夾爪後點雲過少，中止")
                    self.plan_pub.publish(json.dumps([]))
                    return

                # Step 2：FP 補全（用 FP pose 把 CAD mesh 採樣為完整物件點雲）
                if self.object_pose_in_cam is not None:
                    points_completed = self.complete_object_with_foundationpose_pose(
                        self.object_pose_in_cam)
                    if points_completed is None or len(points_completed) < 50:
                        rospy.logwarn("⚠️ FP 補全失敗，fallback 用 points_clean")
                        points_completed = points_clean
                else:
                    rospy.logwarn("⚠️ 無 FP pose，fallback 用 points_clean")
                    points_completed = points_clean

                # Step 3：計算 receiver lims（基於補全後的物件點雲）
                recv_points_vis = None
                result = self.compute_receiver_lims_from_icp(
                    points_completed, camera_frame)
                if result is None:
                    rospy.logwarn("⚠️ 無法計算 receiver lims，改用 bounding box")
                    lims_recv_new = self.get_dynamic_lims(
                        self.filter_outlier_points(points_completed))
                else:
                    lims_recv_new, recv_points_vis = result
                if lims_recv_new is None:
                    rospy.logwarn("⚠️ lims 計算失敗，中止")
                    self.plan_pub.publish(json.dumps([]))
                    return

                colors_completed = np.full(
                    (len(points_completed), 3), 0.5, dtype=np.float32)

                rospy.loginfo(
                    f"[DEBUG] receiver_only 餵 AnyGrasp: "
                    f"pts={points_completed.shape[0]}, lims={lims_recv_new}")

                # Step 4：AnyGrasp 偵測
                gg_recv, _ = self.anygrasp.get_grasp(
                    points_completed, colors_completed,
                    lims=lims_recv_new,
                    apply_object_mask=False,
                    dense_grasp=True,
                    collision_detection=True)
                torch.cuda.empty_cache()

                if gg_recv is None or len(gg_recv) == 0:
                    rospy.logwarn("⚠️ receiver_only 找不到姿態")
                    self.plan_pub.publish(json.dumps([]))
                    return

                top_k = 20
                gg_recv = gg_recv.nms().sort_by_score()[:top_k]
                recv_filtered = self.get_arm_specific_grasps(
                    gg_recv, "leftarm", camera_frame)
                self.publish_grasp_markers(recv_filtered, camera_frame)
                json_plan = self.generate_left_only_plan(recv_filtered)

                if json_plan:
                    self.plan_pub.publish(json.dumps(json_plan))
                    rospy.loginfo(
                        f"✅ receiver_only 計畫書發布，共 {len(json_plan)} 個姿態")

                    if self.cfgs.debug:
                        trans_mat = np.array(
                            [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                        cloud = o3d.geometry.PointCloud()
                        cloud.points = o3d.utility.Vector3dVector(points_full)
                        cloud.colors = o3d.utility.Vector3dVector(colors_full)
                        cloud.transform(trans_mat)
                        vis_list = [cloud]
                        # 紅色：receiver 篩選區域
                        if recv_points_vis is not None and len(recv_points_vis) > 0:
                            recv_pcd = o3d.geometry.PointCloud()
                            recv_pcd.points = o3d.utility.Vector3dVector(recv_points_vis)
                            recv_pcd.paint_uniform_color([1, 0, 0])
                            recv_pcd.transform(trans_mat)
                            vis_list.append(recv_pcd)
                        # 藍色：AnyGrasp 生成的姿態
                        for item in recv_filtered:
                            g = item['grasp'].to_open3d_geometry()
                            g.transform(trans_mat)
                            g.paint_uniform_color([0, 0, 1])
                            vis_list.append(g)
                        self._show_as_cv_window(vis_list, "AnyGrasp 接收臂")
                        rospy.loginfo("👀 [Debug] 接收臂夾取姿態")
                        
                else:
                    rospy.logwarn("❌ receiver_only 找不到有效姿態")
                    self.plan_pub.publish(json.dumps([]))
                return

            if mode == "left_only":
                rospy.loginfo("🧠 left_only：用 receiver_mask 直接在桌面偵測左手夾取姿態")

                if not os.path.exists(receiver_mask_path):
                    rospy.logerr("❌ 找不到 receiver_mask（left_only 需要 LLM 先完成）")
                    self.plan_pub.publish(json.dumps([]))
                    return

                mask_recv_img = cv2.imread(receiver_mask_path, cv2.IMREAD_GRAYSCALE)
                mask_recv = cv2.resize(
                    mask_recv_img, (color_np.shape[1], color_np.shape[0])) > 127
                pts_recv = np.stack(
                    [points_x, points_y, points_z],
                    axis=-1)[valid_depth_mask & mask_recv].astype(np.float32)
                pts_recv = self.filter_outlier_points(pts_recv)

                if pts_recv.shape[0] < 50:
                    rospy.logwarn("⚠️ left_only：遮罩內點雲過少，中止")
                    self.plan_pub.publish(json.dumps([]))
                    return

                _thin_pad_lo = 0.03 if self.target_object in {"scissors", "screwdriver", "spatula", "spoon"} else 0.0
                lims_recv = self.get_dynamic_lims(pts_recv, pad=_thin_pad_lo)
                if not lims_recv:
                    rospy.logwarn("⚠️ left_only：無法計算 lims")
                    self.plan_pub.publish(json.dumps([]))
                    return

                rospy.loginfo(
                    f"[DEBUG] left_only 餵 AnyGrasp: "
                    f"pts={pts_recv.shape[0]}, lims={lims_recv}")

                gg_recv, _ = self.anygrasp.get_grasp(
                    points_full, colors_full,
                    lims=lims_recv,
                    apply_object_mask=False,
                    dense_grasp=True,
                    collision_detection=True)
                torch.cuda.empty_cache()

                if gg_recv is None or len(gg_recv) == 0:
                    rospy.logwarn("⚠️ left_only 找不到姿態")
                    self.plan_pub.publish(json.dumps([]))
                    return

                top_k = 20
                gg_recv = gg_recv.nms().sort_by_score()[:top_k]
                recv_filtered = self.get_arm_specific_grasps(
                    gg_recv, "leftarm", camera_frame)
                self.publish_grasp_markers(recv_filtered, camera_frame)
                json_plan = self.generate_left_only_plan(recv_filtered)

                if json_plan:
                    self.plan_pub.publish(json.dumps(json_plan))
                    rospy.loginfo(
                        f"✅ left_only 計畫書發布，共 {len(json_plan)} 個姿態")
                else:
                    rospy.logwarn("❌ left_only 找不到有效姿態")
                    self.plan_pub.publish(json.dumps([]))
                return

            mask_recv_img = cv2.imread(receiver_mask_path, cv2.IMREAD_GRAYSCALE)
            mask_recv = cv2.resize(
                mask_recv_img, (color_np.shape[1], color_np.shape[0])) > 127
            pts_recv = np.stack(
                [points_x, points_y, points_z],
                axis=-1)[valid_depth_mask & mask_recv].astype(np.float32)
            pts_recv = self.filter_outlier_points(pts_recv)
            _thin_objects = {"scissors", "screwdriver", "spatula", "spoon"}
            _thin_pad = 0.03 if self.target_object in _thin_objects else 0.0
            lims_recv = self.get_dynamic_lims(pts_recv, pad=_thin_pad)

            # =========================================================
            # dual 模式：生成左右手姿態並配對
            # =========================================================
            mask_giver_img = cv2.imread(giver_mask_path, cv2.IMREAD_GRAYSCALE)
            mask_giver = cv2.resize(
                mask_giver_img, (color_np.shape[1], color_np.shape[0])) > 127
            pts_giver = np.stack(
                [points_x, points_y, points_z],
                axis=-1)[valid_depth_mask & mask_giver].astype(np.float32)
            pts_giver = self.filter_outlier_points(pts_giver)
            lims_giver = self.get_dynamic_lims(pts_giver, pad=_thin_pad)

            if not lims_giver:
                rospy.logwarn("⚠️ 操作臂遮罩內缺乏有效點雲")
                self.plan_pub.publish(json.dumps([]))
                return

            rospy.loginfo("🧠 dual 模式：生成操作臂姿態...")
            gg_giver, _ = self.anygrasp.get_grasp(
                points_full, colors_full,
                lims=lims_giver,
                apply_object_mask=False,
                dense_grasp=True,
                collision_detection=True)
            torch.cuda.empty_cache()

            if gg_giver is None or len(gg_giver) == 0:
                rospy.logwarn("⚠️ 無法找到操作臂姿態")
                self.plan_pub.publish(json.dumps([]))
                return

            top_k = 20
            gg_giver = gg_giver.nms().sort_by_score()[:top_k]
            giver_filtered = self.get_arm_specific_grasps(gg_giver, "rightarm", camera_frame)
            json_plan = self.generate_giver_only_plan(giver_filtered)

            if json_plan:
                self.plan_pub.publish(json.dumps(json_plan))
                rospy.loginfo(f"✅ dual 計畫書發布，共 {len(json_plan)} 個方案")
                
                try:
                    trans = self.tf_buffer.lookup_transform(
                        "world", camera_frame, rospy.Time(0), rospy.Duration(1.0))
                    t = trans.transform.translation
                    q = trans.transform.rotation
                    T = np.eye(4)
                    T[:3, 3] = [t.x, t.y, t.z]
                    T[:3, :3] = Rotation.from_quat(
                        [q.x, q.y, q.z, q.w]).as_matrix()
 
                    pts_giver_h = np.hstack(
                        [pts_giver, np.ones((len(pts_giver), 1))])
                    pts_giver_world = (T @ pts_giver_h.T).T[:, :3]
 
                    # 合併 giver + recv 點雲作為完整物件點雲供 ICP 使用
                    pts_full_object = np.vstack([pts_giver, pts_recv]) \
                        if pts_recv.shape[0] >= 10 else pts_giver
                    pts_full_object_h = np.hstack(
                        [pts_full_object, np.ones((len(pts_full_object), 1))])
                    pts_full_object_world = (T @ pts_full_object_h.T).T[:, :3]
 
                    if pts_recv.shape[0] >= 10:
                        rc_h = np.array([*np.mean(pts_recv, axis=0), 1.0])
                        rc_world = (T @ rc_h)[:3].tolist()
                    else:
                        rc_world = None
                        
                    gc_h = np.array([*np.mean(pts_giver, axis=0), 1.0])
                    gc_world = (T @ gc_h)[:3].tolist()
                    global_mask_path = os.path.join(self.mask_dir, "sam_global_mask_full.png")
                    if os.path.exists(global_mask_path):
                        mask_full_img = cv2.imread(global_mask_path, cv2.IMREAD_GRAYSCALE)
                        mask_full = cv2.resize(
                            mask_full_img,
                            (color_np.shape[1], color_np.shape[0])) > 127
                        pts_full_obj_cam = np.stack(
                            [points_x, points_y, points_z],
                            axis=-1)[valid_depth_mask & mask_full].astype(np.float32)
                        pts_full_obj_cam = self.filter_outlier_points(pts_full_obj_cam)
                        if len(pts_full_obj_cam) >= 10:
                            pts_full_obj_h = np.hstack(
                                [pts_full_obj_cam, np.ones((len(pts_full_obj_cam), 1))])
                            pts_full_obj_world = (T @ pts_full_obj_h.T).T[:, :3]
                            oc_world = np.mean(pts_full_obj_world, axis=0).tolist()
                        else:
                            # fallback：原本的合併算法
                            oc_world = np.mean(pts_full_object_world, axis=0).tolist()
                    else:
                        oc_world = np.mean(pts_full_object_world, axis=0).tolist()
 
                except Exception as e:
                    rospy.logwarn(f"點雲座標轉換失敗，使用相機座標: {e}")
                    pts_giver_world = pts_giver
                    pts_full_object_world = pts_giver
                    rc_world = np.mean(pts_recv, axis=0).tolist() \
                        if pts_recv.shape[0] >= 10 else None
                    gc_world = np.mean(pts_giver, axis=0).tolist()
                    oc_world = np.mean(pts_full_object_world, axis=0).tolist() \
                        if pts_full_object_world is not None \
                        else np.mean(pts_giver, axis=0).tolist()

                object_points_pub = rospy.Publisher(
                    "/anygrasp/object_points", String, queue_size=1)
                rospy.sleep(0.1)  # 等 publisher 建立
                model_path = self.find_ycb_model_path(self.target_object)
                rospy.loginfo(f"🔍 model_path: {model_path}")
                if model_path:
                    refined_pts = self.refine_points_with_icp(pts_full_object_world, model_path)
                else:
                    refined_pts = pts_giver_world

                object_points_pub.publish(json.dumps({
                    "points": refined_pts.tolist(),
                    "receiver_centroid": rc_world,
                    "giver_centroid": gc_world, 
                    "object_centroid": oc_world 
                }))
                
                if self.cfgs.debug:
                    trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points_full)
                    cloud.colors = o3d.utility.Vector3dVector(colors_full)
                    cloud.transform(trans_mat)
                    vis_list = [cloud]
                    for item in giver_filtered:
                        gripper = item['grasp'].to_open3d_geometry()
                        gripper.transform(trans_mat)
                        gripper.paint_uniform_color([1, 0, 0])
                        vis_list.append(gripper)
                    self._show_as_cv_window(vis_list, "AnyGrasp 給予臂")
                    rospy.loginfo("👀 [Debug] 給予臂夾取姿態（左下角視窗）")
            else:
                rospy.logwarn("❌ dual 模式找不到可行方案")
                self.plan_pub.publish(json.dumps([]))

        except Exception as e:
            rospy.logerr(f"管線發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.plan_pub.publish(json.dumps([]))


if __name__ == '__main__':
    node = AnyGraspHandoverNode()
    rospy.sleep(2)
    rospy.spin()