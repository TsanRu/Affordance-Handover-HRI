#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    print("✅ 成功跨界連接 ROS Noetic！")
except ImportError:
    print("❌ 找不到 ROS")

import cv2
import json
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage
from transformers import pipeline, SamModel, SamProcessor
import base64
from io import BytesIO
from openai import OpenAI
import warnings

warnings.filterwarnings('ignore')

VISION_SYSTEM_PROMPT = """
你是一個頂尖的雙臂機器人視覺分析專家，專精於安全、自然的物件交接任務。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示左右機械手臂基座、桌面高度、目標物件位置。
【圖片 2】物件特寫網格圖：目標物件的 5x5 網格（X軸 A-E 左到右，Y軸 1-5 上到下），每個格子中心有白色標籤（如 A1、B3）。

【角色定義】
給予臂 Giver（右手，基座在畫面右側）：從桌面穩定夾取物件，移動到交接區後透過手腕旋轉調整物件朝向，再遞給左手。
接收臂 Receiver（左手，基座在畫面左側）：在交接區等待，以自然人類方式從右手接取物件。

相機從正前方拍攝，網格 A 欄在畫面最左、E 欄在畫面最右。
因此，給予臂自然從物件右側（D、E 欄方向）接近，接收臂自然從物件左側（A、B 欄方向）接近。
若物件較小或整體位置偏移，允許跨越中線，但兩者的相對位置仍應反映各自基座的空間方向。

【第一步：判斷物件交接策略】
在選擇網格之前，先判斷這個物件屬於哪種類型：

類型一（functional_end）：物件有明確的功能性接取部位
範例：鎚子（握柄）、刀子（刀柄）、水壺（把手）、鍋子（把手）
→ 給予臂夾持非接取端，接收臂接取功能端
→ 例如鎚子：給予臂夾鎚頭，接收臂接握柄

類型二（geometric）：物件無明確功能分區，形狀對稱、規則或不規則
範例：餅乾盒、湯罐、香蕉、書本
→ 給予臂夾持物件中段，接收臂接取夾爪能穩定握持的面

【第二步：根據策略選擇網格，並嚴格遵守以下物理約束】

───── 給予臂 Giver（右手）的約束 ─────

約束 R1：高度選擇
Y1 是物件頂部，Y5 是物件底部靠近桌面。
選擇時考量兩件事：
第一，給予臂的夾取位置需要穩定支撐後續的手腕旋轉動作，
中段偏下通常比頂部或底部更穩定。
第二，給予臂夾取的位置應盡量為接收臂保留足夠的接取空間，
避免把整個物件的主要接取面都佔住。
靠近桌面的區域（Y4、Y5）需注意夾爪本體不要撞到桌面，
但 AnyGrasp 本身有碰撞偵測，不需要強制排除這個區域。

約束 R2：物理邊緣必要性
選取的網格必須包含物件真實輪廓（頂部、側面、角落）。
夾爪需要邊緣才能施力，絕不能只選中央平坦區域。

約束 R3：面積與連續性
根據物件實際大小動態決定網格數量（夾爪寬約 7-8cm）：
選取的網格總面積必須足以讓夾爪完整貼合施力，
寧可多選一格也不要因面積不足而夾不穩。

判斷方式：
- 觀察物件在【圖片 2】中佔據的實際比例
- 選取的格子應涵蓋「夾爪兩個指尖都能接觸到物件邊緣」的位置，
  不是覆蓋整個部位的所有格子
- 若某個部位（握柄、側面）在視覺上延伸很長，
  選在該部位能穩定施力的中段即可，不需要選滿整個部位
- 網格必須連續集中，不能分散


───── 接收臂 Receiver（左手）的約束 ─────

約束 L1：自然接取方向
先思考「如果是人類要從機器人手中接過這個物件，會自然從哪裡拿？」

functional_end 物件：
直接選功能接取端（握柄、把手）對應的網格區域。

geometric 物件：
優先選擇夾爪能穩定握持的面。判斷方式：
- 規則方形物件（盒子、書本）：優先選窄面或側面，
  大面通常超過夾爪開口（約 8.5cm）難以握持
- 細長或不規則物件（香蕉、湯匙）：
  若窄面不易施力，從上方或側方施力的較大面也是合理選擇，
  不需要強制選窄面，以夾爪能穩定施力為判斷依據
- 圓柱形物件（湯罐）：任何方向皆可，
  選擇接收臂最容易自然接近的那一側

約束 L2：手腕自然性
選擇接收臂手腕不需要大幅扭轉即可自然貼合的區域。

約束 L3：與給予臂的碰撞判斷
核心問題只有一個：「如果兩個夾爪同時夾在各自選定的位置，
夾爪本體（每個寬約 7-8cm）在三維空間中會不會碰到彼此？」

推理方式：
想像右手夾爪從一個方向夾住 right_grids 的區域，
左手夾爪從另一個方向夾住 left_grids 的區域。
兩個夾爪的進入方向通常不同，
因此即使網格在平面圖上看起來接近或重疊，
只要兩個夾爪的進入方向不衝突，實際上就不會碰撞。

若推理後確認不會碰撞 → 維持選取，不需要強制拉開距離。
若推理後確認會碰撞 → 調整其中一個區域，直到兩者可以同時存在。

不要因為「看起來太近」就強制拉開距離，
也不要因為「網格有重疊」就判定為碰撞，
判斷依據永遠是三維空間中夾爪本體的物理干涉，不是平面網格的重疊。

約束 L4：物理邊緣必要性
同 R2，選取網格必須包含物件真實邊緣。

約束 L5：面積與連續性
同 R3，以能讓夾爪完整貼合為目標。

判斷方式：
不是「這個部位延伸多遠就選多遠」，
而是「夾爪放在這個部位的哪個位置，能讓兩個指尖都接觸到物件邊緣」。
選取範圍只需要涵蓋那個接觸位置，不需要覆蓋整個部位。

───── 共同約束 ─────

約束 G1：確認空間方向合理性
給予臂基座在畫面右側，接近物件的自然方向是從右側（D、E 欄）進入。
接收臂基座在畫面左側，接近物件的自然方向是從左側（A、B 欄）進入。
選完網格後，檢查：給予臂區域是否偏向畫面右側？接收臂區域是否偏向畫面左側？
若物件小或位置特殊導致跨越中線，在 reasoning 中說明原因。

【輸出格式】（純 JSON，不含任何其他文字或 markdown）
{
    "object_name": "物件英文名稱",
    "handover_strategy": "functional_end 或 geometric",
    "receiver_part": "若為 functional_end 填接取部位名稱，若為 geometric 填 null",
    "left_grids": ["網格代號", ...],   
    // 以完整覆蓋接取區域為目標，不要因為「夠了」就停止選取
    "right_grids": ["網格代號", ...],  // 同上，完整覆蓋夾取區域比限制格數更重要
    "reasoning": "1.判斷物件類型的理由 → 2.給予臂選擇這些網格的理由（說明為何偏向畫面右側，或若跨越中線說明原因）→ 3.接收臂選擇這些網格的理由（說明為何偏向畫面左側，或若跨越中線說明原因）→ 4.確認兩區域不碰撞"
}
"""

RECEIVER_ONLY_PROMPT = """
你是一個機器人視覺分析專家，專精於自然的物件交接任務。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示給予臂（右手）夾持物件懸空於交接區域的狀態。
【圖片 2】物件特寫網格圖：目標物件的 5x5 網格（X軸 A-E 左到右，Y軸 1-5 上到下），每個格子中心有白色標籤（如 A1、B3）。

【任務背景】
這是交接任務的第二階段。
給予臂 Giver（右手）已夾持物件移動到交接區域並完成旋轉調整，現在靜止等候。
接收臂 Receiver（左手）需要從空中接取懸空的物件，模擬人類自然接取的動作。
請只為接收臂選擇最適合的接取區域，right_grids 一律回傳空陣列。

【約束】

約束一：迴避給予臂佔據的區域
給予臂已夾持物件中段某個位置，接收臂需要選擇給予臂夾爪不會干涉的區域。
判斷方式：想像接收臂夾爪貼合在你選的位置，與給予臂夾爪會不會碰到對方？
- 不會碰到 → 即使相鄰也沒問題，不需要強制空出一格
- 會碰到 → 才需要移到給予臂未佔據的區域
不要強制規定「一定要選頂部」或「一定要選某個 Y 軸」，
以實際不碰撞為判斷依據，任何位置只要不干涉都是候選。

約束二：自然接取方向
先思考「如果是人類要從機器人手中接過這個物件，會自然從哪裡拿？」
選擇接收臂手腕不需要大幅扭轉即可自然貼合的區域，
從側面、上方、斜向接近都是合理選擇，取決於物件形狀與當前朝向。

約束三：物理邊緣必要性
選取網格必須包含物件真實邊緣（頂部、左側、右側或角落輪廓）。
夾爪需要邊緣才能施力，不能只選中央平坦區域。

約束四：面積與連續性
根據物件實際大小動態決定網格數量（夾爪寬約 7-8cm）：
- 細長型或小物件：2 格通常足夠
- 中型物件：2~3 格
- 大型物件：3~4 格
不要因為輸出範例是兩格就固定選兩格，以物件在圖片中的實際比例判斷。
網格必須連續集中，不能分散。

約束五：左手臂可達性
從【圖片 1】確認物件在交接區域的位置，
選擇左手能自然到達、不需要大幅伸展或扭轉手臂的區域。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "left_grids": ["網格代號", ...],   // 數量依物件大小決定，通常 2~4 格
    "right_grids": [],
    "reasoning": "1.觀察給予臂夾持位置在哪裡 → 2.判斷哪些區域可以不干涉地接取 → 3.從中選擇最符合人類自然接取直覺的位置 → 4.確認面積與邊緣條件"
}
"""

LEFT_ONLY_PROMPT = """
你是一個機器人視覺分析專家，專精於從桌面夾取物件。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示當前桌面狀態，目標物件靜置於桌面上，右手臂已退回待命位置。
【圖片 2】物件特寫網格圖：目標物件的 5x5 網格（X軸 A-E 左到右，Y軸 1-5 上到下），每個格子中心有白色標籤（如 A1、B3）。

【任務背景】
這是重新抓取任務。原本的空中交接未能成功，物件已被放回桌面，右手臂退回待命位置。
左手臂需要獨立從桌面夾取這個物件。

【約束】

約束一：穩定夾取
選擇夾爪能完整貼合並施力的區域，以「能不能夾穩」為判斷依據。
任何能讓夾爪不滑脫的位置都是合理選擇，不需要區分物件類型或特定部位。

約束二：物理邊緣必要性
選取的網格必須包含物件真實邊緣（側面、頂部輪廓、角落）。
夾爪需要邊緣才能施力，不能只選中央平坦區域。

約束三：面積與連續性
根據物件實際大小動態決定網格數量（夾爪寬約 7-8cm）：
- 細長型或小物件：2 格通常足夠
- 中型物件：2~3 格
- 大型物件：3~4 格
網格必須連續集中，不能分散。

約束四：左手臂可達性
從【圖片 1】確認物件在桌面的位置，
選擇左手能自然到達、不需要大幅伸展的區域。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "left_grids": ["網格代號", ...],
    "reasoning": "1.觀察物件當前姿態與桌面位置 → 2.選擇夾爪能穩定施力的區域 → 3.確認邊緣與面積條件"
}
"""

INTENT_UNDERSTANDING_PROMPT = """
你是一個雙臂機器人協作系統的語意理解模組，負責在抓取與交接動作開始之前，
判斷使用者真正想要的物件是什麼。

【角色定義】
給予臂（give arm）：負責從工作區夾取物件，並透過空中交接將物件遞送出去。
接收臂（receive arm）：在交接區接住物件，代表「使用者」這個語意角色——
使用者在此情境中扮演接收方，物件最終會被交付到使用者可以直接取用的位置。

你將收到一張桌面場景圖，圖中包含當前桌面上實際存在的物件。

【使用者輸入】
"{user_input}"

【可用物件清單】
你的回答必須從以下清單中選擇，不得使用清單以外的名稱：
hammer, scissors, spatula, banana, tomato_soup_can, sugar_box, bowl, large_clamp

【任務】
請觀察畫面中的物件，判斷使用者指的是畫面中的哪一個，
並從上方清單中選出完全一致的名稱（例如應輸出 tomato_soup_can，不可輸出 can 或 soup）。

使用者的描述可能不是直接的物件名稱，而是功能性或外觀描述
（例如「可以鎚東西的工具」「尖尖的那個」），請結合畫面內容合理判斷。

【confidence 判斷標準】
- high：畫面中有明確對應物件，描述非常吻合
- medium：有可能對應但存在歧義，或畫面中有多個候選
- low：無法確認，或畫面中找不到符合描述的物件

【輸出格式】（純 JSON，不含任何其他文字或 markdown）
{{
    "object_name": "必須是上方清單中的其中一個名稱，完整複製貼上；若畫面中找不到對應物件則填 unknown",
    "confidence": "high / medium / low",
    "reasoning": "一句話說明如何結合畫面與使用者輸入推斷出這個物件（用中文回覆）"
}}
"""

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def imgmsg_to_numpy(msg):
    dtype_class = np.uint8
    channels = 3 if "rgb8" in msg.encoding or "bgr8" in msg.encoding else 1
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
        if "bgr8" in msg.encoding:
            img = img[:, :, ::-1]
    else:
        img = img.reshape((msg.height, msg.width))
    return img


def draw_som_grid(img_rgb, rows=5, cols=5):
    """
    改良版 Set-of-Mark 網格繪製
    在每個格子中心放代號，用半透明色塊交替標示格子
    回傳：標注後的影像、每個格子的絕對座標字典
    """
    h, w = img_rgb.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    col_labels = [chr(65 + i) for i in range(cols)]

    # 建立半透明覆蓋層
    overlay = img_rgb.copy()
    alpha = 0.15  # 透明度，0=完全透明，1=完全不透明

    # 棋盤式交替顏色（淡藍 / 淡橘），讓格子邊界更清晰
    colors = [
        (173, 216, 230),  # 淡藍
        (255, 200, 150),  # 淡橘
    ]

    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            grid_id = f"{col_labels[c]}{r + 1}"
            grid_dict[grid_id] = (x1, y1, x2, y2)

            # 棋盤交替填色
            color = colors[(r + c) % 2]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # 將半透明覆蓋層疊回原圖
    result = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

    # 畫格子邊線（深灰色，比純黑柔和但清晰）
    for i in range(1, rows):
        y = int(i * cell_h)
        cv2.line(result, (0, y), (w, y), (80, 80, 80), 1)
    for j in range(1, cols):
        x = int(j * cell_w)
        cv2.line(result, (x, 0), (x, h), (80, 80, 80), 1)

    # 外框加粗
    cv2.rectangle(result, (0, 0), (w - 1, h - 1), (80, 80, 80), 2)

    # 在每個格子中心放代號（白字黑邊，任何背景都清晰可見）
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            grid_id = f"{col_labels[c]}{r + 1}"
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(cell_w, cell_h) / 60.0  # 根據格子大小自動縮放
            thickness = max(1, int(font_scale * 2))

            (text_w, text_h), _ = cv2.getTextSize(grid_id, font, font_scale, thickness)
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2

            cv2.putText(result, grid_id, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(result, grid_id, (text_x, text_y),
                        font, font_scale, (255, 255, 255), thickness)

    return result, grid_dict


class SemanticBrainNode:
    def __init__(self):
        rospy.init_node('semantic_brain_node', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        rospy.loginfo(f"📦 正在載入模型 ({self.device})...")
        self.detector = pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=self.device
        )
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        rospy.loginfo("✅ 所有 AI 模型載入完成！")

        self.save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "my_gazebo_data")
        self.target_object = ""
        self.user_input = ""
        self.need_process = False
        self.latest_image = None
        self.is_processing = False

        # 訂閱相機影像（持續接收，存最新一幀）
        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_buffer_callback)

        # 接收來自手臂控制端的 trigger
        rospy.Subscriber("/system/trigger_llm", String, self.trigger_callback)

        # 發布完成訊號給手臂控制端
        self.done_pub = rospy.Publisher("/system/llm_done", String, queue_size=1)

        rospy.loginfo("🧠 大腦節點就緒，等待 trigger_llm 訊號...")

    def image_buffer_callback(self, msg):
        """持續接收最新影像，不做處理"""
        self.latest_image = msg

    def resolve_intent(self, user_input: str, img_pil) -> str:
        """用 GPT 將使用者自然語言描述解析成英文物件名稱"""
        rospy.loginfo(f"🔍 解析使用者意圖：「{user_input}」")
        prompt = INTENT_UNDERSTANDING_PROMPT.format(user_input=user_input)
        response = self.gpt_client.chat.completions.create(
            model="gpt-5.4",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pil_to_base64(img_pil)}"}},
                ]
            }]
        )
        raw = response.choices[0].message.content
        result = json.loads(raw.replace("```json", "").replace("```", "").strip())
        name = result.get("object_name", "unknown")
        rospy.loginfo(f"✅ 意圖解析結果：{name}（信心：{result.get('confidence')}，理由：{result.get('reasoning')}）")
        return name

    def trigger_callback(self, msg):
        """收到手臂控制端的觸發訊號"""
        if self.is_processing:
            rospy.logwarn("⏳ AI 正在處理中，忽略重複的 trigger 訊號...")
            return

        try:
            data = json.loads(msg.data)
            self.user_input = data.get("user_input", "")   # 自然語言輸入（優先）
            self.target_object = data.get("object_name", "unknown")
            self.mode = data.get("mode", "dual")
        except json.JSONDecodeError:
            self.user_input = ""
            self.target_object = msg.data
            self.mode = "dual"

        log_target = f"user_input=「{self.user_input}」" if self.user_input else f"object_name={self.target_object}"
        rospy.loginfo(f"⚡ 收到 trigger_llm，目標: {log_target}，模式: {self.mode}")
        self.need_process = True

    def run(self):
        """主迴圈，檢查是否需要處理"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.need_process and self.latest_image is not None:
                self.need_process = False
                self.is_processing = True # 鎖上
                self.process(self.latest_image, self.target_object, self.mode, self.user_input)
                self.is_processing = False # 處理完解鎖
            rate.sleep()

    def process(self, img_msg, object_name, mode="dual", user_input=""):
        t_start = rospy.Time.now()
        try:
            img_np = imgmsg_to_numpy(img_msg)
            h, w = img_np.shape[:2]
            img_pil = PILImage.fromarray(img_np)

            # 如果收到的是自然語言，先用 GPT 解析成英文物件名
            if user_input:
                object_name = self.resolve_intent(user_input, img_pil)
                if object_name == "unknown":
                    rospy.logerr(f"❌ 無法從畫面中識別使用者描述的物件：「{user_input}」")
                    self.done_pub.publish(json.dumps({"status": "fail", "reason": "intent_not_resolved"}))
                    return

            rospy.loginfo(f"📸 開始處理影像，物件: {object_name}，模式: {mode}")

            # 儲存原始影像
            cv2.imwrite(
                os.path.join(self.save_dir, "original_rgb.png"),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            )

            # OWL-v2 偵測物件
            rospy.loginfo(f"🦉 [1/4] OWL-v2 尋找 '{object_name}'...")
            preds = self.detector(img_pil, candidate_labels=[object_name])
            if not preds:
                rospy.logerr(f"❌ 找不到 '{object_name}'")
                self.done_pub.publish(json.dumps({"status": "fail", "reason": "object_not_found"}))
                return

            best_pred = max(preds, key=lambda x: x['score'])
            box = best_pred['box']
            x_min = int(box['xmin'])
            y_min = int(box['ymin'])
            x_max = int(box['xmax'])
            y_max = int(box['ymax'])
            rospy.loginfo(f"   偵測到 '{object_name}'，信心度: {best_pred['score']:.2f}")

            # 裁切物件區域
            pad = 20
            c_xmin = max(0, x_min - pad)
            c_ymin = max(0, y_min - pad)
            c_xmax = min(w, x_max + pad)
            c_ymax = min(h, y_max + pad)
            cropped_img = img_np[c_ymin:c_ymax, c_xmin:c_xmax].copy()

            cv2.imwrite(
                os.path.join(self.save_dir, "object_crop_raw.png"),
                cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            )

            # 改良版 SoM 網格繪製
            rospy.loginfo("✂️ [2/4] 繪製 Set-of-Mark 網格...")
            grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)

            # 轉換為絕對座標（給 SAM 使用）
            grid_dict_absolute = {}
            for grid_id, (lx1, ly1, lx2, ly2) in grid_dict_local.items():
                grid_dict_absolute[grid_id] = [
                    c_xmin + lx1, c_ymin + ly1,
                    c_xmin + lx2, c_ymin + ly2
                ]

            grid_img_path = os.path.join(self.save_dir, "cropped_grid_for_vlm.png")
            cv2.imwrite(grid_img_path, cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR))

            # GPT 推理
            rospy.loginfo("🧠 [3/4] 呼叫 GPT 分析網格...")
            gpt_local_img = PILImage.open(grid_img_path)
            if mode == "receiver_only":
                prompt = RECEIVER_ONLY_PROMPT
            elif mode == "left_only":
                prompt = LEFT_ONLY_PROMPT
            else:
                prompt = VISION_SYSTEM_PROMPT

            response = self.gpt_client.chat.completions.create(
                model="gpt-5.4",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pil_to_base64(img_pil)}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pil_to_base64(gpt_local_img)}"}},
                    ]
                }]
            )

            clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            vlm_result = json.loads(clean_json)

            receiver_grids = vlm_result.get('left_grids', [])
            giver_grids = vlm_result.get('right_grids', [])
            handover_strategy = vlm_result.get('handover_strategy', 'geometric')
            receiver_part = vlm_result.get('receiver_part', None)
            rospy.loginfo(f"💡 GPT 決定 - receiver: {receiver_grids}, giver: {giver_grids}")
            rospy.loginfo(f"   策略: {handover_strategy}, 接取部位: {receiver_part}")
            rospy.loginfo(f"   理由: {vlm_result.get('reasoning', '')}")

            if not receiver_grids:
                rospy.logerr("❌ GPT 沒有回傳有效的 receiver 網格")
                self.done_pub.publish(json.dumps({"status": "fail", "reason": "no_grids"}))
                return

            if mode == "dual":
                if not receiver_grids:
                    rospy.logerr("❌ dual 模式缺少 receiver 網格")
                    self.done_pub.publish(json.dumps({"status": "fail", "reason": "no_receiver_grids"}))
                    return
                if not giver_grids:
                    rospy.logerr("❌ dual 模式缺少 giver 網格")
                    self.done_pub.publish(json.dumps({"status": "fail", "reason": "no_giver_grids"}))
                    return
            elif mode in ("receiver_only", "left_only"):
                if not receiver_grids:
                    rospy.logerr(f"❌ {mode} 模式缺少 receiver 網格")
                    self.done_pub.publish(json.dumps({"status": "fail", "reason": "no_receiver_grids"}))
                    return
                giver_grids = []
                rospy.loginfo(f"   模式：{mode}，跳過 giver 遮罩")

            # SAM 分割
            rospy.loginfo("✂️ [4/4] SAM 精準切割...")
            inputs = self.sam_processor(
                img_pil,
                input_boxes=[[[x_min, y_min, x_max, y_max]]],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs.original_sizes.cpu(),
                inputs.reshaped_input_sizes.cpu()
            )
            global_mask = masks[0][0][0].numpy()
            torch.cuda.empty_cache()

            cv2.imwrite(
                os.path.join(self.save_dir, "sam_global_mask_full.png"),
                (global_mask * 255).astype(np.uint8)
            )

            def save_final_mask(grid_ids_list, filename):
                final_mask = np.zeros_like(global_mask, dtype=bool)
                for grid_id in grid_ids_list:
                    if grid_id not in grid_dict_absolute:
                        rospy.logwarn(f"⚠️ 找不到網格代號: {grid_id}")
                        continue
                    gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                    final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]
                out_path = os.path.join(self.save_dir, filename)
                cv2.imwrite(out_path, (final_mask * 255).astype(np.uint8))
                rospy.loginfo(f"✅ 遮罩已儲存: {filename}")

            save_final_mask(receiver_grids, "receiver_mask.png")
            if giver_grids:
                save_final_mask(giver_grids, "giver_mask.png")

            rospy.loginfo("🎉 視覺前處理完畢！")

            # 發布完成訊號給手臂控制端
            self.done_pub.publish(json.dumps({
                "status": "done",
                "object_name": object_name,
                "mode": mode,
                "receiver_grids": receiver_grids,
                "giver_grids": giver_grids,
                "handover_strategy": handover_strategy,
                "receiver_part": receiver_part
            }))

        except Exception as e:
            rospy.logerr(f"管線發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.done_pub.publish(json.dumps({"status": "fail", "reason": str(e)}))

        finally:
            elapsed = (rospy.Time.now() - t_start).to_sec()
            rospy.loginfo(f"⏱️ [{mode}] LLM 推論耗時：{elapsed:.2f} 秒")


if __name__ == '__main__':
    node = SemanticBrainNode()
    rospy.sleep(2)
    node.run()