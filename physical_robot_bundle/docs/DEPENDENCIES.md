# 依賴套件

此 bundle 只包含專案自有的實機端 runtime。建置前需要另外安裝以下依賴。

## ROS packages

- ROS Noetic 與 catkin
- `ur_robot_driver`
- `ur3_moveit_config`
- `realsense2_camera`
- `easy_handeye`、`easy_handeye_msgs`
- `aruco_ros`、`aruco_msgs`
- MoveIt Python interfaces
- `tf`、`tf2_ros`、`tf2_geometry_msgs`、`message_filters`

bundle 內不含 UR driver 與 UR3 MoveIt 原始碼。連接硬體前應使用以下方式逐一確認：

```bash
rospack find <package_name>
```

## Python 環境

目前系統使用不同環境執行各節點，以隔離 ROS Noetic、MediaPipe 與 ML stack
可能衝突的 Python 套件。

| 節點 | 預期環境 | 主要非 ROS 套件 |
| --- | --- | --- |
| `brain_node.py` | `grasp-py310` conda | OpenCV、PyTorch、Transformers、Pillow、Google GenAI、python-dotenv |
| `client_camera.py` | `anygrasp` conda | OpenCV、NumPy、pyzmq |
| `semantic_grasp_controller.py` | 現場使用的 `anygrasp` conda | NumPy、PyYAML、MoveIt Python |
| `handover_perception.py` | 系統 ROS Python | MediaPipe、OpenCV、NumPy、PyYAML |

不要在 AnyGrasp conda 環境中啟動 `handover_perception.py`。

## 遠端推論伺服器

遠端 server 刻意不放進本 bundle，需另外部署：

- `server_anygrasp.py`
- `anygrasp_sdk`
- `graspnetAPI`
- AnyGrasp checkpoint 與 license
- SAM、OWL、Gemini 相關依賴
- 可由實機端連線的 ZMQ endpoint

ZMQ 位址目前設定於 `client_camera.py` 的 `self.server_addr`。

