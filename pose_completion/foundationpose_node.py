#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FoundationPose ROS Node

訂閱:
  /system/trigger_pose    (std_msgs/String, JSON)
    payload 範例: {"object_name": "005_tomato_soup_can"}
  /camera/color/image_raw           (sensor_msgs/Image)
  /camera/aligned_depth_to_color/image_raw  (sensor_msgs/Image)
  /camera/color/camera_info         (sensor_msgs/CameraInfo)
  TF: rightarm_robotiq_85_*_finger_tip_link

發佈:
  /pose/foundationpose_result  (std_msgs/String, JSON)
    payload 範例:
      {"status": "ok",
       "object_name": "005_tomato_soup_can",
       "pose": [[...4x4...]],
       "camera_frame": "camera_color_optical_frame",
       "mesh_path": "/.../textured.obj",
       "stamp": 1234567890.123}

設計:
  - Scorer/Refiner/glctx 啟動時建立一次
  - FoundationPose estimator 對每個物件快取
  - Mask 用 gripper TF + 深度幾何生成，無視覺分割模型依賴
"""

import os
import sys
import json
import time
import threading
import traceback

import numpy as np
import cv2
import trimesh
import open3d as o3d
import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# === 加入 FoundationPose 路徑（如果腳本不在 FoundationPose 根目錄）===
FP_ROOT = "/home/rvl/ros_ws/src/FoundationPose"
if FP_ROOT not in sys.path:
    sys.path.insert(0, FP_ROOT)

# FoundationPose 內部 imports（會把 dr、ScorePredictor 等等全部帶進來）
from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr


# ===================== 設定 =====================
YCB_MODELS_ROOT = "/home/rvl/ros_ws/src/ros_ur3/ur_gripper_gazebo/models"
GRIPPER_MESH_DIR = "/home/rvl/ros_ws/src/robotiq/robotiq_description/meshes/collision"
DEBUG_DIR = "/tmp/fp_debug"
SAVE_DIR = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"

# Topics
TOPIC_RGB = "/camera/color/image_raw"
TOPIC_DEPTH = "/camera/aligned_depth_to_color/image_raw"
TOPIC_CAMERA_INFO = "/camera/color/camera_info"
TOPIC_TRIGGER = "/system/trigger_pose"
TOPIC_RESULT = "/pose/foundationpose_result"

GRIPPER_LINKS = {
    "rightarm_robotiq_85_base_link": "robotiq_85_base_link.stl",
    "rightarm_robotiq_85_left_knuckle_link": "robotiq_85_knuckle_link.stl",
    "rightarm_robotiq_85_right_knuckle_link": "robotiq_85_knuckle_link.stl",
    "rightarm_robotiq_85_left_inner_knuckle_link": "robotiq_85_inner_knuckle_link.stl",
    "rightarm_robotiq_85_right_inner_knuckle_link": "robotiq_85_inner_knuckle_link.stl",
    "rightarm_robotiq_85_left_finger_link": "robotiq_85_finger_link.stl",
    "rightarm_robotiq_85_right_finger_link": "robotiq_85_finger_link.stl",
    "rightarm_robotiq_85_left_finger_tip_link": "robotiq_85_finger_tip_link.stl",
    "rightarm_robotiq_85_right_finger_tip_link": "robotiq_85_finger_tip_link.stl",
}

# Mask 參數
BOX_RADIUS = 0.15
GRIPPER_DILATION = 0.012
DEPTH_MIN = 0.1
DEPTH_MAX = 2.0

# FoundationPose 參數
EST_REFINE_ITER = 5    # 跟 run_demo.py 一致
# ================================================


def imgmsg_to_np(msg):
    """Direct ROS Image -> numpy"""
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.width).copy()
    elif msg.encoding == "16UC1":
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width)
        return arr.astype(np.float32) / 1000.0
    elif msg.encoding in ["rgb8", "bgr8"]:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr.copy()
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def find_ycb_mesh_path(object_name):
    """從 object_name 找到 YCB mesh 路徑，支援部分匹配"""
    # 1. 直接匹配
    for sub in ["google_16k", "tsdf"]:
        p = os.path.join(YCB_MODELS_ROOT, object_name, sub, "textured.obj")
        if os.path.exists(p):
            return p

    # 2. Fuzzy 匹配：找名稱包含 object_name 的資料夾
    if os.path.isdir(YCB_MODELS_ROOT):
        candidates = []
        for folder in os.listdir(YCB_MODELS_ROOT):
            if object_name.lower() in folder.lower():
                candidates.append(folder)
        # 取最短匹配（避免 "can" 同時匹到 "master_chef_can" 和 "tomato_soup_can"）
        candidates.sort(key=len)
        for folder in candidates:
            for sub in ["google_16k", "tsdf"]:
                p = os.path.join(YCB_MODELS_ROOT, folder, sub, "textured.obj")
                if os.path.exists(p):
                    rospy.loginfo(f"  Fuzzy 匹配: '{object_name}' → '{folder}'")
                    return p

    raise FileNotFoundError(
        f"找不到 mesh：{object_name}（試過直接匹配和 fuzzy 匹配）")


class FoundationPoseNode:

    def __init__(self):
        rospy.init_node("foundationpose_node", anonymous=False)
        os.makedirs(DEBUG_DIR, exist_ok=True)
        os.makedirs(SAVE_DIR, exist_ok=True)

        # === FoundationPose 核心元件（一次性載入）===
        rospy.loginfo("=" * 60)
        rospy.loginfo("載入 FoundationPose 模型...")
        rospy.loginfo("=" * 60)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        rospy.loginfo("✓ Scorer / Refiner / glctx 載入完成")

        # 每個物件的 estimator 快取：{object_name: (est, mesh, mesh_path)}
        self.estimator_cache = {}

        # === TF ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # === 相機內參（等第一個 CameraInfo）===
        rospy.loginfo(f"等待 CameraInfo: {TOPIC_CAMERA_INFO}")
        cam_info = rospy.wait_for_message(
            TOPIC_CAMERA_INFO, CameraInfo, timeout=10.0)
        self.K = np.array(cam_info.K).reshape(3, 3)
        rospy.loginfo(f"K =\n{self.K}")

        # === RGB-D buffer ===
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_frame = None
        self.buf_lock = threading.Lock()

        rospy.Subscriber(TOPIC_RGB, Image, self._rgb_callback, queue_size=1)
        rospy.Subscriber(TOPIC_DEPTH, Image, self._depth_callback, queue_size=1)

        # === Trigger / Publisher ===
        rospy.Subscriber(TOPIC_TRIGGER, String, self._trigger_callback,
                         queue_size=1)
        self.result_pub = rospy.Publisher(TOPIC_RESULT, String, queue_size=1)

        # === 處理鎖 ===
        self.need_process = False
        self.is_processing = False
        self.pending_request = None  # 收到的 trigger payload

        rospy.loginfo("=" * 60)
        rospy.loginfo(f"FoundationPose 節點就緒")
        rospy.loginfo(f"訂閱 trigger: {TOPIC_TRIGGER}")
        rospy.loginfo(f"發佈結果: {TOPIC_RESULT}")
        rospy.loginfo("=" * 60)

    # ============== ROS Callbacks ==============

    def _rgb_callback(self, msg):
        with self.buf_lock:
            self.latest_rgb = imgmsg_to_np(msg)
            self.latest_camera_frame = (
                msg.header.frame_id or "camera_color_optical_frame")

    def _depth_callback(self, msg):
        with self.buf_lock:
            self.latest_depth = imgmsg_to_np(msg)

    def _trigger_callback(self, msg):
        if self.is_processing:
            rospy.logwarn("⏳ 正在處理上一個請求，忽略此次 trigger")
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            # 向下相容：純字串視為 object_name
            data = {"object_name": msg.data}
        rospy.loginfo(f"⚡ 收到 trigger: {data}")
        self.pending_request = data
        self.need_process = True

    # ============== Mask 生成 ==============

    def _lookup_tf_pos(self, target_frame, source_frame):
        trans = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(2.0))
        t = trans.transform.translation
        return np.array([t.x, t.y, t.z])

    def _lookup_tf_matrix(self, target_frame, source_frame):
        trans = self.tf_buffer.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(2.0))
        t = trans.transform.translation
        q = trans.transform.rotation
        T = np.eye(4)
        T[:3, 3] = [t.x, t.y, t.z]
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return T

    def _sample_gripper_points(self, camera_frame):
        all_pts = []
        for link_name, mesh_file in GRIPPER_LINKS.items():
            try:
                T = self._lookup_tf_matrix(camera_frame, link_name)
                mesh = o3d.io.read_triangle_mesh(
                    os.path.join(GRIPPER_MESH_DIR, mesh_file))
                if not mesh.has_vertices():
                    continue
                pcd = mesh.sample_points_uniformly(number_of_points=500)
                pts_local = np.asarray(pcd.points)
                pts_h = np.hstack(
                    [pts_local, np.ones((len(pts_local), 1))])
                pts_cam = (T @ pts_h.T).T[:, :3]
                all_pts.append(pts_cam)
            except Exception as e:
                rospy.logwarn(f"⚠️ 載入 {link_name} 失敗: {e}")
        return np.vstack(all_pts) if all_pts else np.empty((0, 3))

    def _generate_mask(self, depth, camera_frame, obj_diameter=None, roi_center_cam=None):
        H, W = depth.shape

        # 指尖中點
        tip_l = self._lookup_tf_pos(
            camera_frame, "rightarm_robotiq_85_left_finger_tip_link")
        tip_r = self._lookup_tf_pos(
            camera_frame, "rightarm_robotiq_85_right_finger_tip_link")
        fingertip_mid = (tip_l + tip_r) / 2
        # ROI 中心：優先用傳入的物件中心，否則 fallback 指尖中點
        roi_center = roi_center_cam if roi_center_cam is not None else fingertip_mid
        rospy.loginfo(
            f"  ROI 中心 ({'物件中心' if roi_center_cam is not None else '指尖中點'}) "
            f"(camera frame): [{roi_center[0]:.3f}, {roi_center[1]:.3f}, {roi_center[2]:.3f}]")
        # 每個 pixel 反投影到 3D
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        z = depth.copy()
        z[~np.isfinite(z)] = 0
        x = (u_grid - cx) * z / fx
        y = (v_grid - cy) * z / fy
        pts_3d = np.stack([x, y, z], axis=-1)
        
        # 動態 BOX_RADIUS：物件直徑一半 + 20% 餘裕
        obj_radius = (obj_diameter / 2.0 * 1.2) if obj_diameter else BOX_RADIUS

        # 3D ROI + 有效深度
        valid_depth = (z > DEPTH_MIN) & (z < DEPTH_MAX)
        in_box = (
            (np.abs(pts_3d[..., 0] - roi_center[0]) < obj_radius) &
            (np.abs(pts_3d[..., 1] - roi_center[1]) < obj_radius) &
            (np.abs(pts_3d[..., 2] - roi_center[2]) < obj_radius) &
            valid_depth
        )
        n_box = int(np.sum(in_box))
        rospy.loginfo(f"  3D ROI 內 pixel: {n_box}")
        if n_box == 0:
            return np.zeros((H, W), dtype=np.uint8)
        
        # 深度過濾：只保留深度接近物件的 pixel
        z_roi = z[in_box]
        if len(z_roi) > 10:
            z_median = np.median(z_roi)
            depth_tolerance = min(obj_diameter * 0.3, 0.05) if obj_diameter else 0.04
            depth_filter = np.abs(z - z_median) < depth_tolerance
            in_box = in_box & depth_filter
            rospy.loginfo(
                f"  深度過濾後 pixel: {int(np.sum(in_box))} "
                f"(物件深度≈{z_median:.3f}m, tolerance={depth_tolerance:.3f}m)")

        # 採樣 gripper 點 + 扣除
        gripper_pts = self._sample_gripper_points(camera_frame)
        in_box_idx = np.where(in_box)
        candidate_pts = pts_3d[in_box]

        if len(gripper_pts) > 0:
            tree = cKDTree(gripper_pts)
            dists, _ = tree.query(candidate_pts, k=1, workers=-1)
            keep = dists > GRIPPER_DILATION
            rospy.loginfo(
                f"  扣 gripper 後保留: {int(np.sum(keep))}/{n_box}")
        else:
            keep = np.ones(n_box, dtype=bool)

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[in_box_idx[0][keep], in_box_idx[1][keep]] = 255

        # 形態學：去雜訊 + 填小洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        rospy.loginfo(f"  最終 mask pixel: {int(np.sum(mask > 0))}")
        return mask

    def _save_overlay(self, rgb, mask, path):
        overlay = rgb.astype(np.float32)
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        alpha = (mask > 0).astype(np.float32) * 0.5
        alpha = alpha[..., None]
        blended = (overlay * (1 - alpha) + green * alpha).astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # ============== FoundationPose ==============

    def _ensure_estimator(self, object_name):
        """快取機制：相同物件第二次 trigger 不重建 estimator"""
        if object_name in self.estimator_cache:
            return self.estimator_cache[object_name]

        rospy.loginfo(f"建立 {object_name} 的 FoundationPose estimator...")
        mesh_path = find_ycb_mesh_path(object_name)
        rospy.loginfo(f"  Mesh: {mesh_path}")
        mesh = trimesh.load(mesh_path)

        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=DEBUG_DIR,
            debug=0,
            glctx=self.glctx,
        )
        rospy.loginfo(f"✓ {object_name} estimator 建立完成")
        self.estimator_cache[object_name] = (est, mesh, mesh_path)
        return self.estimator_cache[object_name]

    def _process_one(self, request):
        object_name = request.get("object_name")
        if not object_name:
            self._publish_fail("missing object_name")
            return
        
        object_centroid_world = request.get("object_centroid_world", None)

        # 取最新 RGB-D
        with self.buf_lock:
            if self.latest_rgb is None or self.latest_depth is None:
                self._publish_fail("no_rgbd_yet")
                return
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy()
            camera_frame = self.latest_camera_frame

        rospy.loginfo(f"[Pose] 處理物件: {object_name}")
        rospy.loginfo(f"  Camera frame: {camera_frame}")
        rospy.loginfo(f"  RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
        rospy.loginfo(
            f"  Depth range: [{np.nanmin(depth):.3f}, "
            f"{np.nanmax(depth):.3f}] m")

        # === 1. 先建立或取出 estimator（取得正確 diameter）===
        try:
            est, mesh, mesh_path = self._ensure_estimator(object_name)
        except FileNotFoundError as e:
            self._publish_fail(f"mesh_not_found: {e}")
            return

        # === 2. 生成 mask（用正確的 diameter）===
        roi_center_cam = None
        if object_centroid_world is not None:
            try:
                oc_w = np.array(object_centroid_world)
                T_cw = self._lookup_tf_matrix(camera_frame, "world")
                oc_cam = (T_cw @ np.append(oc_w, 1.0))[:3]
                roi_center_cam = oc_cam
                rospy.loginfo(
                    f"  物件中心 (camera frame): {oc_cam.round(3)}")
            except Exception as e:
                rospy.logwarn(f"⚠️ 物件中心轉換失敗，fallback 指尖: {e}")

        rospy.loginfo("[Pose] 生成 mask...")
        t0 = time.time()
        mask = self._generate_mask(
            depth, camera_frame,
            obj_diameter=est.diameter,
            roi_center_cam=roi_center_cam)
        rospy.loginfo(f"  mask 生成耗時: {time.time()-t0:.3f}s")

        if int(np.sum(mask > 0)) < 100:
            self._publish_fail("mask_too_small")
            return

        cv2.imwrite(os.path.join(SAVE_DIR, "object_mask.png"), mask)
        self._save_overlay(rgb, mask, os.path.join(SAVE_DIR, "mask_overlay.png"))

        # === 3. 跑 register ===
        rospy.loginfo("[Pose] 跑 FoundationPose.register...")
        t0 = time.time()
        try:
            pose = est.register(
                K=self.K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask.astype(bool),
                iteration=EST_REFINE_ITER,
            )
            elapsed = time.time() - t0
            rospy.loginfo(f"  register 耗時: {elapsed:.3f}s")
            rospy.loginfo(f"  Pose:\n{pose}")
        except Exception as e:
            rospy.logerr(f"FoundationPose register 失敗: {e}")
            rospy.logerr(traceback.format_exc())
            self._publish_fail(f"register_error: {e}")
            return

        # === 4. 存 debug 檔 ===
        np.savetxt(os.path.join(DEBUG_DIR, "latest_pose.txt"), pose)

        # ★ 新增：把 mesh 輪廓疊到 RGB 圖，確認 FP 有沒有對準
        try:
            # 把 mesh 頂點用 pose 轉到相機座標，再投影到 2D
            verts = np.array(mesh.vertices)  # (N, 3)
            verts_h = np.hstack([verts, np.ones((len(verts), 1))])
            verts_cam = (pose @ verts_h.T).T[:, :3]  # (N, 3) camera frame

            # 只保留在相機前方的點
            valid_z = verts_cam[:, 2] > 0
            verts_cam = verts_cam[valid_z]

            # 投影到像素座標
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            u = (verts_cam[:, 0] / verts_cam[:, 2] * fx + cx).astype(int)
            v = (verts_cam[:, 1] / verts_cam[:, 2] * fy + cy).astype(int)

            # 畫到 RGB 圖上（綠點）
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
            h, w = vis.shape[:2]
            for ui, vi in zip(u, v):
                if 0 <= ui < w and 0 <= vi < h:
                    cv2.circle(vis, (ui, vi), 2, (0, 255, 0), -1)

            cv2.imwrite(os.path.join(SAVE_DIR, "rgb_handover.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(SAVE_DIR, "fp_pose_overlay.png"), vis)
            rospy.loginfo(f"✓ FP pose overlay 已存: {SAVE_DIR}/fp_pose_overlay.png")
        except Exception as e:
            rospy.logwarn(f"FP overlay 存檔失敗: {e}")

        # === 5. 發佈結果 ===
        result = {
            "status": "ok",
            "object_name": object_name,
            "pose": pose.tolist(),
            "camera_frame": camera_frame,
            "mesh_path": mesh_path,
            "stamp": rospy.Time.now().to_sec(),
        }
        self.result_pub.publish(json.dumps(result))
        rospy.loginfo(f"✓ Pose 已發佈到 {TOPIC_RESULT}")

    def _publish_fail(self, reason):
        rospy.logerr(f"✗ 失敗: {reason}")
        self.result_pub.publish(json.dumps({
            "status": "fail",
            "reason": reason,
            "stamp": rospy.Time.now().to_sec(),
        }))

    # ============== Main Loop ==============

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.need_process and self.pending_request is not None:
                self.need_process = False
                self.is_processing = True
                try:
                    self._process_one(self.pending_request)
                except Exception as e:
                    rospy.logerr(f"處理過程例外: {e}")
                    rospy.logerr(traceback.format_exc())
                    self._publish_fail(f"unexpected: {e}")
                finally:
                    self.pending_request = None
                    self.is_processing = False
            rate.sleep()


if __name__ == "__main__":
    try:
        node = FoundationPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal: {e}")
        rospy.logerr(traceback.format_exc())
        sys.exit(1)