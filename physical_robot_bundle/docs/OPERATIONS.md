# 實機操作手冊

## 安全檢查

1. 清空 UR3 工作範圍，並確保操作人員可以立即按下急停。
2. 確認機器人、筆電、相機、夾爪與遠端 inference server 使用正確網路。
3. 啟動前確認 `192.168.86.7` 確實是本次操作的 UR3。
4. 確認手眼外參屬於目前安裝的相機與固定位置。
5. 先使用 `handover_params.yaml` 內的低速度、低加速度設定。
6. 每一次實體執行前，先在 RViz 檢查 pre-grasp plan。

## 手眼外參

本 bundle 不包含目前機器的標定數值。外參必須放在：

```text
~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml
```

相機、機器人底座或安裝位置改變後必須重新標定。完成 bundle 建置後可執行：

```bash
ROBOT_IP=192.168.86.7 ./scripts/start_calibration.sh
```

## 啟動順序

以下各項應放在不同 terminal 執行。每個 terminal 先載入環境：

```bash
cd /path/to/physical_robot_bundle
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

依序啟動硬體與 ROS 服務：

```bash
# 1. RGB-D 相機
roslaunch realsense2_camera rs_camera.launch align_depth:=true

# 2. UR3 driver
roslaunch ur_robot_driver ur3_bringup.launch robot_ip:=192.168.86.7

# 3. MoveIt
roslaunch ur3_moveit_config moveit_planning_execution.launch limited:=true

# 4. 發布 camera 到 base 的標定 TF
roslaunch ur3_handover publish_handeye.launch

# 5. 載入交接參數
rosparam load src/ur3_handover/config/handover_params.yaml

# 6. RViz
rosrun rviz rviz -d rviz/anygrasp_debug.rviz
```

再依照各節點所需的 Python 環境啟動應用程式：

```bash
# 7. 本機語意偵測
conda activate grasp-py310
rosrun ur3_handover brain_node.py

# 8. 機械臂與夾爪控制器
conda activate anygrasp
rosrun ur3_handover semantic_grasp_controller.py

# 9. 相機 UI 與遠端 inference client
conda activate anygrasp
rosrun ur3_handover client_camera.py

# 10. 人手感知：使用系統 ROS Python，不使用 conda
conda deactivate
rosrun ur3_handover handover_perception.py
```

## 現場設定位置

- UR3 與 gripper IP：driver 參數及 controller 預設值 `192.168.86.7`。
- Robotiq socket port：controller 預設值 `63352`。
- 遠端 AnyGrasp ZMQ：`client_camera.py` 內的 `self.server_addr`。
- 交接區域、tracking、force-release 門檻、TCP offset 與運動限制：
  `src/ur3_handover/config/handover_params.yaml`。
- API keys：從 `.env.example` 建立的 bundle 根目錄 `.env`。

## 最低限度健康檢查

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
rostopic echo -n 1 /camera/color/camera_info
rosrun tf tf_echo base_link camera_color_optical_frame
rostopic list | grep semantic_handover
```

如果 TF 消失、規劃姿態超出工作區、夾爪 socket 無法連線，或 wrench 數值不合理，
應立即停止 controller 並檢查硬體，不可繼續確認執行。
