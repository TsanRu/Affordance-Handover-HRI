# UR3 實體機器人執行包

這個資料夾整理了 UR3 語意抓取與人機交接系統的實機端程式，並採用可直接執行
`catkin_make` 的工作區結構。所有程式均使用 bundle 內的相對路徑，不依賴原本
`/home/weilun/handeye_ws` 的檔案位置。

## 內容

- UR3 MoveIt 抓取與交接控制器
- Robotiq 夾爪 TCP socket 驅動
- RealSense RGB-D client 與遠端 AnyGrasp 請求流程
- 本機語意物件偵測節點
- MediaPipe 人手交接感知節點
- Handover 參數、手部 landmark 模型與 RViz 設定
- 手眼標定與發布外參的 launch

遠端 AnyGrasp server、第三方 ROS 原始碼、API key 與目前機器的手眼外參不包含
在此資料夾內。

## 建置

先安裝 [依賴套件](docs/DEPENDENCIES.md)，再執行：

```bash
cd /path/to/physical_robot_bundle
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

建立不會被 Git 追蹤的本機金鑰檔：

```bash
cp .env.example .env
```

啟動 `brain_node.py` 前至少要填入 `OPENAI_API_KEY`（目前實際使用的模型；`GOOGLE_API_KEY`
系列只有在切回程式碼裡註解掉的 Gemini 版本時才需要）。

## 執行

實機啟動順序、安全檢查、標定與 ROS 診斷方式請見
[實機操作手冊](docs/OPERATIONS.md)。

主要資料流：

```text
RealSense -> client_camera -> 遠端 AnyGrasp -> grasp topics
                                            -> semantic_grasp_controller
                                            -> UR3 + Robotiq gripper

RealSense -> handover_perception -> hand state -> controller
```

## 重要預設值

- UR3 與夾爪主機：`192.168.86.7`
- Robotiq port：`63352`
- 機器人 frame：`base_link`
- 相機 frame：`camera_color_optical_frame`
- 手眼外參：`~/.ros/easy_handeye/ur3_realsense_handeyecalibration_eye_on_base.yaml`
- AnyGrasp endpoint：目前仍設定在 `client_camera.py` 內

以上都是現場部署設定。啟用機器人運動前，必須逐項確認與實際硬體一致。

