#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
affordance_gt.py

用途：
  功能端物件的「握柄 / 功能端」ground truth，定義在物件座標系（object-centric
  frame）的主軸上，取代容易隨試驗擺放角度失效的影像座標系（SoM 網格）標記法。

  每個物件只需標定一次：
    1. 用 google_16k/nontextured.stl（與 SDF collision/visual 用的
       textured.obj 同一份局部座標系）算出點雲的 PCA 主軸。
    2. 沿主軸算「垂直於主軸的最大延伸寬度」輪廓，找出握柄 <-> 功能端的
       幾何轉折點（自動偵測 + 人工檢查校正），記錄成 boundary（0~1 正規化
       位置）與 handle_side（握柄在轉折點的哪一側）。

  校準時用的分析腳本可參考本檔案下方 __main__。以下資料表是已標定好的結果，
  之後不需再重算，除非物件的 mesh 換掉。

  排除物件：mug 沒有納入——mug 的 PCA 主軸實際上抓到的是「握把凸出方向」而非
  單純的柄/功能端分段，寬度輪廓呈現「窄-寬-窄」對稱形狀，無法用單一 boundary
  乾淨切成兩段，需要另一種標定方式（例如以握把凸出的角度而非軸向位置定義），
  暫不在此表中處理。
"""

import numpy as np
from tf.transformations import quaternion_matrix

MODELS_DIR = "/home/rvl/ros_ws/src/ros_ur3/ur_gripper_gazebo/models"

# 供標定腳本使用的 mesh 路徑（需與 simple_grasp_controller.py 的
# OBJECT_MESH_MAP 保持一致）
OBJECT_MESH_MAP = {
    "hammer":      f"{MODELS_DIR}/048_hammer/google_16k/nontextured.stl",
    "scissors":    f"{MODELS_DIR}/037_scissors/google_16k/nontextured.stl",
    "screwdriver": f"{MODELS_DIR}/043_phillips_screwdriver/google_16k/nontextured.stl",
    "spatula":     f"{MODELS_DIR}/033_spatula/google_16k/nontextured.stl",
    "spoon":       f"{MODELS_DIR}/031_spoon/google_16k/nontextured.stl",
    "large_clamp": f"{MODELS_DIR}/051_large_clamp/google_16k/nontextured.stl",
}

# ══════════════════════════════════════════════════════════════
#  各物件的 affordance 主軸（校準一次，數值凍結）
#
#  origin      : PCA 時用的點雲重心（物件局部座標系）
#  axis        : 主軸單位向量（物件局部座標系）
#  t_min/t_max : 全部頂點投影到主軸後的範圍，用來把投影值正規化到 0~1
#  boundary    : 握柄 / 功能端的分界（正規化位置，0~1）
#  handle_side : 握柄在分界的哪一側，"low"=靠近 0 那側，"high"=靠近 1 那側
# ══════════════════════════════════════════════════════════════
OBJECT_AFFORDANCE_AXIS = {
    "hammer": {
        "origin": (-0.049711, 0.031326, 0.015938),
        "axis":   (-0.367134, 0.930164, 0.002921),
        "t_min": -0.231938, "t_max": 0.098734,
        # boundary 非幾何轉折點（握柄本身寬度是漸變，無銳利轉折可偵測），
        # 是人因判斷：真實交接時不會抓靠近鎚頭的握柄前段（不安全、不好施力），
        # 收窄到握柄較粗、離鎚頭較遠的後段。0~60% 握柄 GT，60~100% 排除區（含鎚頭）。
        "boundary": 0.60, "handle_side": "low",
    },
    "scissors": {
        "origin": (0.019228, 0.012158, 0.007457),
        "axis":   (-0.287547, -0.957752, -0.005212),
        "t_min": -0.129501, "t_max": 0.072805,
        "boundary": 0.60, "handle_side": "high",  # 0~60% 刀刃，60~100% 指環握柄
    },
    "screwdriver": {
        "origin": (-0.019025, 0.011449, 0.015789),
        "axis":   (-0.892849, -0.448381, -0.042132),
        "t_min": -0.080662, "t_max": 0.135559,
        "boundary": 0.53, "handle_side": "low",   # 0~53% 握柄，53~100% 金屬桿+起子頭
    },
    "spatula": {
        "origin": (0.002272, -0.091547, 0.014358),
        "axis":   (-0.965654, 0.259805, 0.003731),
        "t_min": -0.122905, "t_max": 0.188396,
        "boundary": 0.48, "handle_side": "high",  # 0~48% 鏟面，48~100% 握柄
    },
    "spoon": {
        "origin": (-0.051642, 0.002211, 0.012081),
        "axis":   (-0.938136, 0.345710, 0.019615),
        "t_min": -0.124527, "t_max": 0.071106,
        "boundary": 0.65, "handle_side": "low",   # 0~65% 握柄，65~100% 湯匙頭
    },
    "large_clamp": {
        "origin": (-0.005300, -0.011427, 0.018336),
        "axis":   (-0.108705, 0.992741, 0.051474),
        "t_min": -0.083180, "t_max": 0.083546,
        # 跟其他物件不同：寬度輪廓是雙峰中間夾一個谷底（56% 附近最窄），
        # 對應鉗夾的樞軸結構，不是單調漸變或單一轉折。PCA 主軸只抓到其中一隻
        # 握把的方向，另一隻交叉的握把跟鉗頭混在一起，邊界比其他物件更依賴
        # 目視判斷。0~56% 鉗頭/樞軸（功能端），56~100% 直桿握柄。
        "boundary": 0.56, "handle_side": "high",
    },
}


def world_point_to_axis_ratio(point_world, obj_position, obj_quaternion, object_name):
    """
    把世界座標系下的接觸點，換算成該物件主軸上的正規化位置（0~1，會 clip）。

    point_world     : (x, y, z)，世界座標系下的接觸點（例如左手 TCP 位置）
    obj_position    : (x, y, z)，物件當下在世界座標系的位置
                       （建議用 Gazebo /gazebo/get_model_state 的 ground truth，
                       不要用 FoundationPose 的估測姿態，避免把估測誤差混進指標）
    obj_quaternion  : (x, y, z, w)，物件當下在世界座標系的姿態
    object_name     : OBJECT_AFFORDANCE_AXIS 裡的物件名稱

    回傳：float，0=axis 的 t_min 端，1=axis 的 t_max 端
    """
    if object_name not in OBJECT_AFFORDANCE_AXIS:
        raise KeyError(f"'{object_name}' 尚未標定 affordance 主軸")

    cfg = OBJECT_AFFORDANCE_AXIS[object_name]

    p_world = np.array(point_world, dtype=float)
    obj_pos = np.array(obj_position, dtype=float)

    # world -> object local frame
    R = quaternion_matrix(obj_quaternion)[:3, :3]
    p_local = R.T @ (p_world - obj_pos)

    origin = np.array(cfg["origin"], dtype=float)
    axis = np.array(cfg["axis"], dtype=float)
    axis = axis / np.linalg.norm(axis)

    t_raw = float((p_local - origin) @ axis)
    ratio = (t_raw - cfg["t_min"]) / (cfg["t_max"] - cfg["t_min"])
    return float(np.clip(ratio, 0.0, 1.0))


def is_handle_grasp(object_name, ratio):
    """判定正規化位置 ratio 是否落在該物件的握柄 GT 區間內。"""
    cfg = OBJECT_AFFORDANCE_AXIS[object_name]
    boundary = cfg["boundary"]
    if cfg["handle_side"] == "low":
        return ratio <= boundary
    else:
        return ratio >= boundary


if __name__ == "__main__":
    # 重新標定 / 檢查某物件的主軸與 boundary 是否合理時使用：
    #   python3 affordance_gt.py hammer
    import sys
    import trimesh

    if len(sys.argv) < 2 or sys.argv[1] not in OBJECT_MESH_MAP:
        print(f"用法: python3 {sys.argv[0]} <{'|'.join(OBJECT_MESH_MAP)}>")
        sys.exit(1)

    name = sys.argv[1]
    mesh = trimesh.load(OBJECT_MESH_MAP[name])
    pts = mesh.vertices
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argsort(eigvals)[::-1][0]]
    t_raw = centered @ axis
    t_min, t_max = t_raw.min(), t_raw.max()

    n_bins = 40
    radius = np.linalg.norm(centered - np.outer(t_raw, axis), axis=1)
    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(t_raw, bin_edges) - 1, 0, n_bins - 1)
    print(f"=== {name} ===")
    for i in range(n_bins):
        mask = bin_idx == i
        w = radius[mask].max() * 2 * 1000 if mask.any() else float("nan")
        pct = 100.0 * i / (n_bins - 1)
        print(f"  {pct:5.1f}%  width={w:6.1f}mm")
