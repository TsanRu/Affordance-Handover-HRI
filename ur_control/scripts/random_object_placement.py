#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_object_placement.py

用途：
  在 Gazebo 中將目標物件隨機放置於右臂安全夾取區間內。
  - 若物件已存在 Gazebo → 直接移動（SetModelState）
  - 若物件不存在（world 檔未放） → 從 SDF 自動 spawn 後再定位

用法：
  python3 random_object_placement.py hammer        # 隨機放置
  python3 random_object_placement.py hammer --seed 42   # 固定 seed
  python3 random_object_placement.py hammer --calibrate # 讀當前座標
  python3 random_object_placement.py hammer --remove    # 從 Gazebo 移除
"""

import sys
import os
import math
import rospkg
import random
import argparse

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path not in sys.path:
    sys.path.append(ros_path)

import rospy
import actionlib
import moveit_commander
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Quaternion
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from gazebo_ros_link_attacher.srv import Attach, AttachRequest

RIGHT_HOME_JOINTS = [1.43, -1.211, -2.0, 0.0, 1.6476, -0.0237]
LEFT_HOME_JOINTS  = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]

# ══════════════════════════════════════════════════════════════
#  各物件的工作空間邊界（由 grasp_zone_scan.py 實測，依接近高度各別掃描）
#
#  掃描高度（PREGRASP_Z）：
#    hammer/banana/bowl/scissors : z=0.87（扁平物件，頂部約 0.72m）
#    cracker_box/mug             : z=0.90（中高物件，頂部約 0.75m）
#    tomato_soup_can             : z=0.93（直立罐，頂部約 0.78m），工作空間最小
# ══════════════════════════════════════════════════════════════
OBJECT_WORKSPACE = {
    # (x_min, x_max, y_min, y_max)  — 由 grasp_zone_scan.py 實測，各物件接近高度不同
    # 保守取 y_min=-0.05（scan 顯示 y=-0.10 邊界點，但 -0.05 更安全）
    # x_min 限制在 0.70：臂基座位於 x=0.775，低於此值物件落在基座後方的鏡頭盲區
    "hammer":          (0.70, 1.05, -0.05, 0.20),  # z=0.87 scan
    "cracker_box":     (0.70, 1.05, -0.05, 0.20),  # z=0.90 scan
    "tomato_soup_can": (0.70, 1.00,  0.00, 0.20),  # z=0.93 scan，最受限
    "banana":          (0.70, 1.05, -0.05, 0.20),  # z=0.86 scan
    "bowl":            (0.70, 1.05, -0.05, 0.20),  # z=0.86 scan
    "mug":             (0.70, 1.05, -0.05, 0.20),  # z=0.90 scan
    "scissors":        (0.70, 1.05, -0.05, 0.20),  # z=0.86 scan
    "screwdriver":     (0.70, 1.05, -0.05, 0.20),  # 長條物件
    "sugar_box":       (0.70, 1.05, -0.05, 0.20),
    "spatula":         (0.70, 1.05, -0.05, 0.20),
    "spoon":           (0.70, 1.05, -0.05, 0.20),
    "large_clamp":     (0.70, 1.05, -0.05, 0.20),
}
DEFAULT_WORKSPACE = (0.70, 1.05, -0.05, 0.20)

# ══════════════════════════════════════════════════════════════
#  各物件的 TABLE_Z（Gazebo 重力落定後的實測值）
# ══════════════════════════════════════════════════════════════
OBJECT_TABLE_Z = {
    "hammer":          0.70,
    "cracker_box":     0.780,  # 側臥（第二大面朝下）
    "tomato_soup_can": 0.73,
    "banana":          0.69,
    "bowl":            0.69,
    "mug":             0.69,
    "scissors":        0.69,
    "screwdriver":     0.69,
    "sugar_box":       0.749,  # 側臥（第二大面朝下）
    "spatula":         0.6855, # 平躺（實測）
    "spoon":           0.6853, # 平躺（實測）
    "large_clamp":     0.6844, # 實測
}
DEFAULT_TABLE_Z = 0.70

# ══════════════════════════════════════════════════════════════
#  各物件的 half_extent（夾取點到物件中心的最大距離估計）
#
#  注意：這不是物件的半長，而是「LLM 最遠可能選到的夾取點距中心的距離」。
#  hammer：握柄約在物件 1/3 處，距中心約 10cm → 0.10
#  長條物件（banana/scissors）：通常抓中段，估計 8cm → 0.08
#  對稱物件（can/bowl/mug）：直接用物件半徑
#
#  限制：hammer/cracker_box/banana/scissors 物件較長，在 y 方向隨機旋轉時
#  物件末端可能略超出 workspace，此為已知限制，AnyGrasp 內部會過濾
#  不可達的夾取姿態，實際影響有限。
# ══════════════════════════════════════════════════════════════
OBJECT_HALF_EXTENT = {
    "hammer":          0.10,   # 握柄距中心約 10cm
    "cracker_box":     0.10,   # 側臥，最長邊 21cm，半寬 10.5cm
    "tomato_soup_can": 0.06,   # 罐頭半徑 5.1cm
    "banana":          0.08,   # 通常抓中段
    "bowl":            0.09,   # 碗半徑 8.1cm
    "mug":             0.06,   # 馬克杯半徑 4.75cm
    "scissors":        0.08,   # 通常抓中段
    "screwdriver":     0.09,   # 握柄距中心約 9cm
    "sugar_box":       0.09,   # 側臥，長邊 17.6cm 水平，半寬 8.8cm
    "spatula":         0.10,   # 全長 30.6cm，workspace 限制取 10cm（同 hammer）
    "spoon":           0.09,   # 全長 18.6cm，碗端距中心 12cm，取 9cm
    "large_clamp":     0.09,   # 全長 17.1cm，估計夾取點距中心約 9cm
}
DEFAULT_HALF_EXTENT = 0.08

# ══════════════════════════════════════════════════════════════
#  功能端物件的 yaw 限制（確保非 affordance 端落在右臂工作空間內）
#
#  每個物件列出兩個允許的 yaw 區間（度），隨機選一個後在其內抽樣。
#  以 handle 方向為基準：
#    hammer : handle 長 16.6cm，需限制避免 handle 在 y 方向超出工作空間
#             [-58°,16°] handle 朝 -x  |  [122°,196°] handle 朝 +x
#  scissors: handle 只有 8.4cm，任何 yaw 都不超出工作空間，不需限制
#  mug     : 旋轉對稱，握把可見性問題由感知層決定，不在此限制
# ══════════════════════════════════════════════════════════════
OBJECT_YAW_RANGES = {
    # 四個區間均勻覆蓋 360°
    # (-58, 16)  : 鎚頭朝右手方向 (+Y)
    # (30, 100)  : 鎚頭朝 -X（鏡頭方向）
    # (122, 196) : 鎚頭朝左手方向 (-Y)
    # (210, 280) : 鎚頭朝 +X（遠離鏡頭方向）
    "hammer": [(-58, 16), (30, 100), (122, 196), (210, 280)],
    # 鏟面（正 X 端）朝四個方向，迴避 ±82°~±98° 的 workspace 缺口
    # (98, 160)   : 鏟面朝左上方（含理想姿態 109°）
    # (20, 80)    : 鏟面朝右上方
    # (-160, -100): 鏟面朝左下方
    # (-80, -20)  : 鏟面朝右下方
    "spatula": [(98, 160), (20, 80), (-160, -100), (-80, -20)],
}

# ══════════════════════════════════════════════════════════════
#  SDF 檔路徑（物件不存在時自動 spawn 用）
# ══════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(rospkg.RosPack().get_path('ur_gripper_gazebo'), 'models')
OBJECT_SDF = {
    "hammer":          "048_hammer/hammer.sdf",
    "cracker_box":     "003_cracker_box/cracker_box.sdf",
    "tomato_soup_can": "005_tomato_soup_can/tomato_soup_can.sdf",
    "banana":          "011_banana/banana.sdf",
    "bowl":            "024_bowl/bowl.sdf",
    "mug":             "025_mug/mug.sdf",
    "scissors":        "037_scissors/scissors.sdf",
    "screwdriver":     "043_phillips_screwdriver/phillips_screwdriver.sdf",
    "sugar_box":       "004_sugar_box/sugar_box.sdf",
    "spatula":         "033_spatula/spatula.sdf",
    "spoon":           "031_spoon/spoon.sdf",
    "large_clamp":     "051_large_clamp/large_clamp.sdf",
}


# ══════════════════════════════════════════════════════════════
#  工具函式
# ══════════════════════════════════════════════════════════════

# 某些物件需要在隨機 yaw 之外先做初始 roll（繞 X 軸），以改變哪一面朝下。
# 值為角度（度），正值 = 右手定則繞 X 軸旋轉。
OBJECT_INIT_ROLL_DEG = {
    "cracker_box": 90.0,   # 將第二大面朝下（側臥）
    "sugar_box":   90.0,   # 將第二大面朝下（側臥）
}

OBJECT_SPAWN_OFFSET = {
    "tomato_soup_can": 0.02,   # 輕量圓罐，落距太大容易倒
}

def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0,
                      z=math.sin(yaw / 2),
                      w=math.cos(yaw / 2))

_override_roll_deg  = None  # 由命令列 --roll 暫時設定
_override_pitch_deg = None  # 由命令列 --pitch 暫時設定

def get_placement_quaternion(model_name: str, yaw: float) -> Quaternion:
    """回傳考慮初始 roll/pitch 後再疊加 yaw 的四元數。順序：先 pitch(Y)，再 roll(X)，再 yaw(Z)。"""
    roll_deg  = _override_roll_deg  if _override_roll_deg  is not None \
        else OBJECT_INIT_ROLL_DEG.get(model_name, 0.0)
    pitch_deg = _override_pitch_deg if _override_pitch_deg is not None else 0.0

    def axis_quat(angle_deg, axis):
        a = math.radians(angle_deg) / 2.0
        c, s = math.cos(a), math.sin(a)
        if axis == 'x': return (c, s, 0.0, 0.0)
        if axis == 'y': return (c, 0.0, s, 0.0)
        if axis == 'z': return (c, 0.0, 0.0, s)

    def qmul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw)

    q = (1.0, 0.0, 0.0, 0.0)
    if pitch_deg != 0.0:
        q = qmul(q, axis_quat(pitch_deg, 'y'))
    if roll_deg != 0.0:
        q = qmul(q, axis_quat(roll_deg, 'x'))
    q = qmul(axis_quat(math.degrees(yaw), 'z'), q)
    return Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])


def model_exists(model_name: str) -> bool:
    rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
    get_svc = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    resp = get_svc(model_name, "world")
    return resp.success


def spawn_object(model_name: str, x: float, y: float, z: float, yaw: float) -> bool:
    sdf_rel = OBJECT_SDF.get(model_name)
    if sdf_rel is None:
        rospy.logerr(f"❌ 找不到 '{model_name}' 的 SDF 路徑設定")
        return False

    sdf_path = os.path.join(MODELS_DIR, sdf_rel)
    if not os.path.exists(sdf_path):
        rospy.logerr(f"❌ SDF 檔不存在：{sdf_path}")
        return False

    with open(sdf_path, 'r') as f:
        sdf_content = f.read()

    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation = get_placement_quaternion(model_name, yaw)

    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
    spawn_svc = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    resp = spawn_svc(model_name, sdf_content, "", pose, "world")
    return resp.success


def release_from_arm(model_name: str):
    """兩手都 detach，並打開兩側夾爪。上一次跑失敗時哪隻手都可能殘留 attach。"""
    try:
        rospy.wait_for_service('/link_attacher_node/detach', timeout=2.0)
        detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        for arm_link in ("rightarm_wrist_3_link", "leftarm_wrist_3_link"):
            try:
                req = AttachRequest()
                req.model_name_1 = model_name
                req.link_name_1 = "link"
                req.model_name_2 = "robot"
                req.link_name_2 = arm_link
                detach_srv.call(req)
                rospy.loginfo(f"[reset] detach: {model_name} from {arm_link}")
            except Exception:
                pass
    except Exception:
        pass

    for arm, topic in [("右", "/rightarm/gripper_controller/gripper_cmd"),
                       ("左", "/leftarm/gripper_controller/gripper_cmd")]:
        try:
            client = actionlib.SimpleActionClient(topic, GripperCommandAction)
            if client.wait_for_server(timeout=rospy.Duration(2.0)):
                goal = GripperCommandGoal()
                goal.command.position = 0.1
                goal.command.max_effort = 50.0
                client.send_goal(goal)
                client.wait_for_result(rospy.Duration(3.0))
                rospy.loginfo(f"[reset] {arm}手夾爪已打開")
        except Exception:
            pass


def go_home_both_arms():
    """兩隻手臂依序回到 home 姿態。"""
    moveit_commander.roscpp_initialize(sys.argv)
    right = moveit_commander.MoveGroupCommander("rightarm")
    left  = moveit_commander.MoveGroupCommander("leftarm")
    right.set_max_velocity_scaling_factor(0.3)
    left.set_max_velocity_scaling_factor(0.3)

    for group, joints, name in [
        (right, RIGHT_HOME_JOINTS, "右手"),
        (left,  LEFT_HOME_JOINTS,  "左手"),
    ]:
        group.set_joint_value_target(joints)
        plan = group.plan()
        if plan[0]:
            group.execute(plan[1], wait=True)
            group.stop()
            rospy.loginfo(f"[home] {name} 已回到 home")
        else:
            rospy.logwarn(f"[home] {name} 規劃失敗，維持原位")


def move_object(model_name: str, x: float, y: float, z: float, yaw: float) -> bool:
    rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
    set_svc = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    state = ModelState()
    state.model_name = model_name
    state.pose = Pose()
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = z
    state.pose.orientation = get_placement_quaternion(model_name, yaw)
    state.reference_frame = "world"

    resp = set_svc(state)
    return resp.success


def remove_object(model_name: str) -> bool:
    rospy.wait_for_service('/gazebo/delete_model', timeout=5.0)
    del_svc = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    resp = del_svc(model_name)
    return resp.success


# ══════════════════════════════════════════════════════════════
#  公開函式
# ══════════════════════════════════════════════════════════════

def random_place(model_name: str, seed: int = None,
                 settle_wait: float = 1.5,
                 fixed_yaw_deg: float = None,
                 fixed_x: float = None, fixed_y: float = None) -> tuple:
    """
    在安全區間內隨機放置物件。
    - 物件已在 Gazebo：直接移動
    - 物件不在 Gazebo：從 SDF spawn 後放置
    settle_wait：spawn 後等待物理落定的秒數（move 時不需要）
    回傳 (x, y, z, yaw_deg)，失敗回傳 None。
    """
    if seed is not None:
        random.seed(seed)
        rospy.loginfo(f"[rand] fixed seed={seed}")

    half    = OBJECT_HALF_EXTENT.get(model_name, DEFAULT_HALF_EXTENT)
    table_z = OBJECT_TABLE_Z.get(model_name, DEFAULT_TABLE_Z)
    ws      = OBJECT_WORKSPACE.get(model_name, DEFAULT_WORKSPACE)
    wx_min, wx_max, wy_min, wy_max = ws

    x_min = wx_min + half
    x_max = wx_max - half
    y_min = wy_min + half
    y_max = wy_max - half

    if x_min >= x_max or y_min >= y_max:
        rospy.logerr(f"[rand] half_extent={half:.3f} makes effective range empty for {model_name}")
        return None

    if fixed_x is not None and fixed_y is not None:
        x = fixed_x
        y = fixed_y
        if fixed_yaw_deg is not None:
            yaw_deg = fixed_yaw_deg
            yaw = math.radians(yaw_deg)
        else:
            yaw = random.uniform(0, 2 * math.pi)
            yaw_deg = math.degrees(yaw)
    elif fixed_yaw_deg is not None:
        # 固定 yaw 模式（測試用）：位置取 workspace 中心
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        yaw_deg = fixed_yaw_deg
        yaw = math.radians(yaw_deg)
    else:
        x   = random.uniform(x_min, x_max)
        y   = random.uniform(y_min, y_max)
        yaw_ranges = OBJECT_YAW_RANGES.get(model_name)
        if yaw_ranges:
            lo, hi = random.choice(yaw_ranges)
            yaw_deg = random.uniform(lo, hi)
            yaw = math.radians(yaw_deg)
        else:
            yaw = random.uniform(0, 2 * math.pi)
            yaw_deg = math.degrees(yaw)

    rospy.loginfo(f"[rand] placing '{model_name}' at x={x:.3f} y={y:.3f} z={table_z:.3f} yaw={yaw_deg:.1f}deg")
    rospy.loginfo(f"[rand] effective range: x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}]")

    spawn_offset = OBJECT_SPAWN_OFFSET.get(model_name, 0.05)
    exists = model_exists(model_name)
    if exists:
        release_from_arm(model_name)
        go_home_both_arms()
        success = move_object(model_name, x, y, table_z + spawn_offset, yaw)
        if success:
            rospy.sleep(settle_wait)
    else:
        rospy.loginfo(f"[rand] model not in Gazebo, spawning from SDF...")
        success = spawn_object(model_name, x, y, table_z + spawn_offset, yaw)
        if success:
            rospy.sleep(settle_wait)

    if success:
        rospy.loginfo("[rand] placement OK")
        return (x, y, table_z, yaw_deg)
    else:
        rospy.logerr("[rand] placement FAILED")
        return None


def calibrate(model_name: str):
    """印出物件當前座標（供確認 TABLE_Z 用）。"""
    if not model_exists(model_name):
        rospy.logerr(f"[calib] '{model_name}' not in Gazebo")
        return

    rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
    get_svc = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    resp = get_svc(model_name, "world")
    p = resp.pose.position
    q = resp.pose.orientation
    yaw_rad = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))

    half = OBJECT_HALF_EXTENT.get(model_name, DEFAULT_HALF_EXTENT)
    ws   = OBJECT_WORKSPACE.get(model_name, DEFAULT_WORKSPACE)
    wx_min, wx_max, wy_min, wy_max = ws

    print(f"\n{'='*50}")
    print(f"  object : {model_name}")
    print(f"  x      = {p.x:.4f}")
    print(f"  y      = {p.y:.4f}")
    print(f"  z      = {p.z:.4f}  <- TABLE_Z")
    print(f"  yaw    = {math.degrees(yaw_rad):.1f}deg")
    print(f"  workspace  : x=[{wx_min},{wx_max}]  y=[{wy_min},{wy_max}]")
    print(f"  effective  : x=[{wx_min+half:.3f},{wx_max-half:.3f}]  y=[{wy_min+half:.3f},{wy_max-half:.3f}]")
    print(f"{'='*50}\n")


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gazebo 物件隨機放置工具")
    parser.add_argument("model_name", help="物件名稱（如 hammer、cracker_box）")
    parser.add_argument("--calibrate", action="store_true",
                        help="讀取當前座標，不移動物件")
    parser.add_argument("--remove", action="store_true",
                        help="將物件從 Gazebo 移除")
    parser.add_argument("--seed", type=int, default=None,
                        help="固定隨機 seed")
    parser.add_argument("--replace", metavar="OLD_MODEL",
                        help="先移除指定的舊物件，再放置新物件（切換物件時使用）")
    parser.add_argument("--clear-all", action="store_true",
                        help="移除所有已知物件後再放置新物件")
    parser.add_argument("--yaw", type=float, default=None, metavar="DEGREES",
                        help="指定 yaw 角度（度），物件置於 workspace 中心，供測試用")
    parser.add_argument("--x", type=float, default=None, metavar="X",
                        help="指定 x 座標（公尺），需與 --y 一起使用")
    parser.add_argument("--y", type=float, default=None, metavar="Y",
                        help="指定 y 座標（公尺），需與 --x 一起使用")
    parser.add_argument("--roll", type=float, default=None, metavar="DEGREES",
                        help="覆蓋初始 roll（繞 X 軸，度）")
    parser.add_argument("--pitch", type=float, default=None, metavar="DEGREES",
                        help="覆蓋初始 pitch（繞 Y 軸，度）")
    args = parser.parse_args()

    rospy.init_node("random_object_placement", anonymous=True)

    if args.calibrate:
        calibrate(args.model_name)
    elif args.remove:
        ok = remove_object(args.model_name)
        print("✅ 已移除" if ok else "❌ 移除失敗（物件可能不存在）")
    else:
        if args.clear_all:
            for m in OBJECT_SDF:
                if m != args.model_name and model_exists(m):
                    print(f"[clear-all] 移除：{m}")
                    remove_object(m)
        elif args.replace:
            print(f"[replace] 移除舊物件：{args.replace}")
            ok = remove_object(args.replace)
            print("✅ 舊物件已移除" if ok else f"⚠️ 舊物件移除失敗（{args.replace} 可能不存在）")
        if args.roll is not None:
            _override_roll_deg = args.roll
        if args.pitch is not None:
            _override_pitch_deg = args.pitch
        random_place(args.model_name, seed=args.seed, fixed_yaw_deg=args.yaw,
                     fixed_x=args.x, fixed_y=args.y)
