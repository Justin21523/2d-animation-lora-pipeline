這是一份將說明文字翻譯為全中文（保留關鍵技術名詞中英對照）的文檔版本，內容結構與程式碼部分完全保持不變。

-----

# 姿勢/動作 LoRA 深度剖析：技術實作指南 (Pose/Action LoRA Deep Dive)

**目標受眾：** 實作 3D 動畫角色姿勢/動作 LoRA 的機器學習工程師
**難度等級：** ⚠️ **高** (涉及 3D 動畫素體的姿勢估計 + 動作分類)
**先決條件：** 角色 LoRA (Character LoRA) 的理解、基礎電腦視覺、姿勢估計 (Pose Estimation) 概念

-----

## 目錄 (Table of Contents)

1.  [為什麼姿勢 LoRA 具有挑戰性](https://www.google.com/search?q=%231-%E7%82%BA%E4%BB%80%E9%BA%BC%E5%A7%BF%E5%8B%A2-lora-%E5%85%B7%E6%9C%89%E6%8C%91%E6%88%B0%E6%80%A7)
2.  [理論基礎](https://www.google.com/search?q=%232-%E7%90%86%E8%AB%96%E5%9F%BA%E7%A4%8E)
3.  [模型選擇框架](https://www.google.com/search?q=%233-%E6%A8%A1%E5%9E%8B%E9%81%B8%E6%93%87%E6%A1%86%E6%9E%B6)
4.  [數據準備流程](https://www.google.com/search?q=%234-%E6%95%B8%E6%93%9A%E6%BA%96%E5%82%99%E6%B5%81%E7%A8%8B)
5.  [實作細節](https://www.google.com/search?q=%235-%E5%AF%A6%E4%BD%9C%E7%B4%B0%E7%AF%80)
6.  [訓練策略](https://www.google.com/search?q=%236-%E8%A8%93%E7%B7%B4%E7%AD%96%E7%95%A5)
7.  [測試與評估](https://www.google.com/search?q=%237-%E6%B8%AC%E8%A9%A6%E8%88%87%E8%A9%95%E4%BC%B0)
8.  [常見問題](https://www.google.com/search?q=%238-%E5%B8%B8%E8%A6%8B%E5%95%8F%E9%A1%8C)

-----

## 1\. 為什麼姿勢 LoRA 具有挑戰性

### 1.1 核心問題 (The Core Problem)

姿勢/動作 LoRA (Pose/Action LoRA) 的目標是教導模型學習特定的身體姿勢和動作（如站立、行走、奔跑、跳躍、坐著、伸手等），並且要**獨立於角色身分 (independently of character identity)**。這在本質上具有挑戰性，因為：

**對於角色 LoRA (Character LoRA)：**

  - 身分 (Identity) 是一致的（相同的臉、相同的服裝）
  - 具有清晰的視覺標記（面部特徵、衣物）
  - 在正面/肖像視角下效果良好

**對於姿勢 LoRA (Pose LoRA)：**

  - **相同的動作看起來不同**，取決於攝影機角度
  - **遮擋 (Occlusions)**（手臂在背後、腿部重疊）
  - 動態動作中的**動態模糊 (Motion blur)**
  - **局部視圖 (Partial views)**（肢體在畫面邊緣被裁切）
  - 廣角鏡頭中的**透視變形 (Perspective distortion)**
  - 同一姿勢下有多個角色（需要分離處理）

### 1.2 3D 動畫特有的挑戰

| 挑戰 | 真人影片 (Real Human Video) | 3D 動畫角色 (3D Animated Characters) | 影響 |
|-----------|------------------|------------------------|--------|
| **身體比例** | 自然解剖結構 | 風格化 (Stylized)（大頭、細肢） | 姿勢模型可能會失敗 |
| **關節位置** | 真實 | 為了視覺吸引力而誇張化 | 關鍵點 (Keypoint) 檢測錯誤 |
| **動作風格** | 自然物理 | 遵循動畫原則 (Animation principles) | 不同的運動模式 |
| **服裝與配件** | 顯露身形 | 通常寬鬆/過大 | 隱藏肢體位置 |
| **攝影機角度** | 典型的視平線 | 創意視角（低角、高角、荷蘭式傾斜） | 不尋常的透視 |
| **動態模糊** | 自然動態模糊 | 刻意/風格化 | 更難檢測關鍵點 |

**範例：** 皮克斯 (Pixar) 角色的「奔跑」姿勢可能比真人有更誇張的前傾和更大的步伐，這打破了訓練於真實影片的姿勢估計模型之假設。

### 1.3 為什麼我們需要姿勢 LoRA

儘管面臨挑戰，姿勢 LoRA 仍然非常有價值，因為：

1.  **控制生成圖像中的角色動作**
2.  **與角色 LoRA 組合使用**，例如「{角色名稱} 正在奔跑」
3.  **可跨角色重複使用**（例如，「跳躍」動作適用於任何角色）
4.  **啟用動態場景**（動作鏡頭、運動、舞蹈）
5.  **組合式生成 (Compositional generation)**：疊加角色 + 表情 + 姿勢 LoRA

-----

## 2\. 理論基礎

### 2.1 什麼是姿勢 LoRA？

姿勢 LoRA 在潛在空間 (latent space) 中學習**身體姿勢/動作特定的特徵**：

```
基礎模型 Latent → 姿勢 LoRA → 姿勢條件化 Latent
```

**關鍵概念：** 姿勢特徵應該是**與身分無關 (identity-agnostic)** 且**視角不變 (view-invariant)**（在一定程度上）。一個「奔跑」的姿勢應該適用於不同的角色和攝影機角度。

### 2.2 LoRA 數學原理 (快速回顧)

與角色/表情 LoRA 相同：

```
W' = W + ΔW
ΔW = A × B

其中:
- W: 原始權重矩陣 (凍結)
- A: 降維投影矩陣 (d_model × rank)
- B: 升維投影矩陣 (rank × d_model)
- rank << d_model (通常姿勢設為 48-64)
```

### 2.3 姿勢 LoRA vs 其他 LoRA 類型

| 面向 | 角色 LoRA | 表情 LoRA | 姿勢 LoRA |
|--------|---------------|-----------------|-----------|
| **目標** | 身分 (Identity) | 面部情緒 | 身體姿勢/動作 |
| **數據集大小** | 200-500 張 | 每類別 100-300 張 | 每動作 150-400 張 |
| **多樣性需求** | 多樣的姿勢/角度 | 多樣的身分 | 多樣的角度/角色 |
| **標註 (Caption) 重點** | 角色名稱 | 表情關鍵字 | 動作動詞 + 姿勢細節 |
| **訓練輪數 (Epochs)** | 8-12 | 6-10 | 8-12 |
| **Rank (維度)** | 32-64 | 32-48 | 48-64 |
| **視角敏感度** | 低 | 非常低 (以臉部為中心) | **高** (全身) |

**關鍵區別：** 姿勢 LoRA 具有**高度的視角依賴性**。需要在訓練數據中平衡攝影機角度。

-----

## 3\. 模型選擇框架

本節針對：「動作偵測等等 一定有很多相關模型 可是要如何應用就是必須考量到多種因素」

### 3.1 任務拆解

姿勢 LoRA 的數據準備需要 **4 個子任務**：

1.  **姿勢估計 (Pose Estimation)**：從角色實例中提取 2D/3D 關鍵點
2.  **動作分類 (Action Classification)**：將動作類型分類（站立、奔跑等）
3.  **視角分類 (View Classification)**：確定攝影機角度（正面、側面、背面等）
4.  **姿勢標準化 (Pose Normalization)**：移除位置/縮放影響以便進行聚類

每個子任務都有多種模型選擇，各有優缺點。

-----

### 3.2 姿勢估計模型 (Pose Estimation Models)

**目標：** 從角色實例中提取骨架關鍵點 (Skeletal keypoints/joints)

#### 選項 1: RTM-Pose (OpenMMLab) ⭐ **推薦**

**優點：**

  - 在 COCO 關鍵點檢測上具有 **SOTA (最高水準) 準確度**
  - **即時效能** (GPU 上約 30 FPS)
  - 多種模型尺寸 (tiny, small, medium, large)
  - 對風格化角色有**良好的泛化能力**
  - 屬於 MMPose 生態系統的一部分（易於使用）

**缺點：**

  - 需要安裝 PyTorch + MMPose
  - 訓練於真人數據（存在一些領域差距 Domain gap）

**使用時機：** 生產流程的預設選擇

**實作：**

```python
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import merge_data_samples

# 初始化模型
config_file = 'rtmpose-m_8xb256-420e_coco-256x192.py'
checkpoint_file = 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 推論 (Inference)
results = inference_topdown(model, img, bboxes=[bbox])

# 提取關鍵點 (17 個 COCO 關鍵點)
keypoints = results[0].pred_instances.keypoints[0]  # 形狀: (17, 2)
scores = results[0].pred_instances.keypoint_scores[0]  # 形狀: (17,)

# COCO 關鍵點順序:
# 0: 鼻子, 1-2: 眼睛, 3-4: 耳朵, 5-6: 肩膀, 7-8: 手肘,
# 9-10: 手腕, 11-12: 臀部, 13-14: 膝蓋, 15-16: 腳踝
```

**3D 特有的調整：**

```python
# 調整分數閾值以適應風格化的身體
valid_keypoints = keypoints[scores > 0.3]  # 針對 3D 設定較低的閾值
```

-----

#### 選項 2: ViTPose++ (Transformer-based) 🔥 **最佳準確度**

**優點：**

  - 在高難度姿勢上具有 **最高準確度**
  - Transformer 架構能捕捉全局上下文
  - 處理遮擋 (Occlusions) 的能力優於 CNN
  - 多種骨幹網路 (ViT-S, ViT-B, ViT-L, ViT-H)

**缺點：**

  - 比 RTM-Pose **慢** (ViT-B 約 10 FPS)
  - 模型尺寸較大
  - VRAM 需求較高

**使用時機：** 準確度至關重要、高難度姿勢（舞蹈、雜技）

**實作：**

```python
from mmpose.apis import init_model, inference_topdown

config_file = 'vitpose-b_coco-256x192.py'
checkpoint_file = 'vitpose-b.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

results = inference_topdown(model, img, bboxes=[bbox])
keypoints = results[0].pred_instances.keypoints[0]
```

-----

#### 選項 3: MediaPipe Pose (Google) 🚀 **最快** ⭐ **CPU-Only 推薦**

**優點：**

  - **極快** (CPU 上可達 60+ FPS\!)
  - 輕量級，無需 GPU
  - 33 個關鍵點（比 COCO 的 17 點更詳細）
  - 可在移動設備上運行
  - **對風格化角色效果良好** (專為 AR/VR 設計)
  - **與 GPU 訓練完全並行** (零衝突)
  - **高 CPU 核心擴展性** (32 threads 可運行 12-16 並行作業)

**缺點：**

  - 準確度低於 RTM-Pose/ViTPose
  - 僅限單人檢測
  - 在極端角度下較不可靠

**使用時機：** 純 CPU 環境、GPU 正忙於訓練、即時預覽、大規模批次處理

**💡 CPU 核心優化建議 (32 threads 環境):**
```bash
# 32 threads CPU 最佳配置
並行作業數 = 12-16 個 (每個作業使用 2 threads)
或
並行作業數 = 8 個 (每個作業使用 4 threads，處理更複雜的動作分類)

推薦配置: 16 並行作業 × 2 threads = 充分利用 32 threads
```

**實作：**

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,  # 0=lite, 1=full, 2=heavy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if results.pose_landmarks:
    # 33 個地標點 (歸一化 0-1)
    landmarks = results.pose_landmarks.landmark

    # 轉換為像素座標
    h, w = img.shape[:2]
    keypoints = np.array([
        [lm.x * w, lm.y * h, lm.visibility]
        for lm in landmarks
    ])
```

**MediaPipe 關鍵點順序 (33 點):**

```
0-10: 臉部/頭部地標
11-12: 肩膀
13-14: 手肘
15-16: 手腕
17-22: 手部地標
23-24: 臀部
25-26: 膝蓋
27-28: 腳踝
29-32: 足部地標
```

-----

#### 選項 4: DWPose (ControlNet 優化) 🎨 **最適合 ControlNet**

**優點：**

  - **專為 ControlNet 訓練而設計**
  - 結合 OpenPose + 全身姿勢檢測 (Whole-body pose detection)
  - 輸出與 ControlNet 相容的骨架圖像
  - 適合生成姿勢條件化模型的訓練數據

**缺點：**

  - 在原始關鍵點檢測上的準確度不如 RTM-Pose
  - 需要額外的預處理

**使用時機：** 準備 ControlNet 訓練數據、姿勢轉圖像生成

**實作：**

```python
from controlnet_aux import DWposeDetector

dwpose = DWposeDetector()
pose_image = dwpose(img)  # 返回骨架可視化圖像

# 也可以獲取原始關鍵點
pose_dict = dwpose(img, output_type='dict')
body_keypoints = pose_dict['bodies']['candidate']
```

-----

#### 選項 5: OpenPose (經典) 📚 **舊版/傳統**

**優點：**

  - 發展成熟，廣泛使用
  - 良好的文檔和範例
  - 多人姿勢估計

**缺點：**

  - 比現代方法 **慢**
  - 需要編譯 (C++ 依賴)
  - 準確度低於 RTM-Pose/ViTPose

**使用時機：** 舊版相容性、多人場景

-----

### 3.3 動作分類方法 (Action Classification Methods)

**目標：** 將檢測到的姿勢分類為動作類型

#### 方法 1: 基於規則的分類 (簡單) ⭐ **推薦入門**

**優點：**

  - 不需要 ML 模型
  - 規則具可解釋性
  - 快速
  - 對於明顯的動作效果良好

**缺點：**

  - 需要手動設計規則
  - 可能無法處理模糊的情況

**使用時機：** 清晰、獨特的動作（站立 vs 奔跑 vs 跳躍）

**實作：**

```python
def classify_action_from_pose(keypoints, scores):
    """
    使用關鍵點幾何學進行基於規則的動作分類

    參數:
        keypoints: (17, 2) COCO 關鍵點
        scores: (17,) 關鍵點置信度分數

    返回:
        action (str), confidence (float)
    """
    # 提取關鍵關節 (COCO 格式)
    nose = keypoints[0]
    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
    l_elbow, r_elbow = keypoints[7], keypoints[8]
    l_wrist, r_wrist = keypoints[9], keypoints[10]
    l_hip, r_hip = keypoints[11], keypoints[12]
    l_knee, r_knee = keypoints[13], keypoints[14]
    l_ankle, r_ankle = keypoints[15], keypoints[16]

    # 計算中心點
    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2

    # 計算幾何特徵
    torso_angle = np.arctan2(
        shoulder_center[1] - hip_center[1],
        shoulder_center[0] - hip_center[0]
    ) * 180 / np.pi

    # 左/右腿角度
    l_leg_angle = compute_angle(l_hip, l_knee, l_ankle)
    r_leg_angle = compute_angle(r_hip, r_knee, r_ankle)

    # 左/右臂角度
    l_arm_angle = compute_angle(l_shoulder, l_elbow, l_wrist)
    r_arm_angle = compute_angle(r_shoulder, r_elbow, r_wrist)

    # 垂直展開度 (膝蓋到肩膀)
    vertical_spread = np.abs(shoulder_center[1] - hip_center[1])

    # 水平展開度 (左腳踝到右腳踝)
    horizontal_spread = np.abs(l_ankle[0] - r_ankle[0])

    # 基於規則的分類

    # 1. 跳躍 (腳離地，垂直展開度小)
    feet_height = min(l_ankle[1], r_ankle[1])
    hip_height = hip_center[1]
    if feet_height < hip_height * 0.8:  # 腳相對於臀部位置較高
        return "jumping", 0.8

    # 2. 奔跑 (軀幹傾斜，腿部展開大，膝蓋彎曲)
    if (abs(torso_angle) > 10 and  # 前傾
        horizontal_spread > vertical_spread * 0.6 and  # 步伐寬大
        (l_leg_angle < 160 or r_leg_angle < 160)):  # 膝蓋彎曲
        return "running", 0.7

    # 3. 行走 (輕微軀幹傾斜，中等腿部展開)
    if (abs(torso_angle) > 5 and
        horizontal_spread > vertical_spread * 0.3 and
        horizontal_spread < vertical_spread * 0.7):
        return "walking", 0.7

    # 4. 坐著 (臀部位置低，腿部彎曲)
    if (hip_height > shoulder_center[1] * 1.2 and  # 臀部低
        l_leg_angle < 120 and r_leg_angle < 120):  # 腿部彎曲
        return "sitting", 0.8

    # 5. 伸手 (單臂伸展)
    if (l_arm_angle > 150 or r_arm_angle > 150):  # 手臂伸直
        wrist_high = max(l_wrist[1], r_wrist[1])
        if wrist_high < shoulder_center[1]:  # 手腕高於肩膀
            return "reaching", 0.6

    # 6. 站立 (預設值，如果直立且腿部伸直)
    if (abs(torso_angle) < 10 and
        l_leg_angle > 160 and r_leg_angle > 160):
        return "standing", 0.7

    # 未知/模糊
    return "unknown", 0.3


def compute_angle(p1, p2, p3):
    """計算由 p1-p2-p3 形成的 p2 處的角度"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi
    return angle
```

-----

#### 方法 2: 基於機器學習的分類 (ResNet/LSTM 處理關鍵點)

**優點：**

  - 學習複雜的動作模式
  - 在模糊情況下準確度較高
  - 可在自定義的 3D 動畫數據集上訓練

**缺點：**

  - 需要已標註的姿勢數據
  - 流程較複雜

**使用時機：** 數據集大、動作複雜、需要較高準確度

**實作：**

```python
import torch
import torch.nn as nn

class PoseActionClassifier(nn.Module):
    """使用 ResNet 從姿勢關鍵點分類動作"""

    def __init__(self, num_keypoints=17, num_classes=8):
        super().__init__()
        # 輸入: (batch, num_keypoints * 2) 展平的關鍵點
        self.fc1 = nn.Linear(num_keypoints * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 用法
model = PoseActionClassifier(num_keypoints=17, num_classes=8)
model.load_state_dict(torch.load('pose_action_classifier.pth'))
model.eval()

# 歸一化並展平關鍵點
keypoints_flat = keypoints.flatten()  # (34,)
keypoints_norm = (keypoints_flat - mean) / std  # 歸一化
keypoints_tensor = torch.tensor(keypoints_norm, dtype=torch.float32).unsqueeze(0)

# 預測
with torch.no_grad():
    logits = model(keypoints_tensor)
    probs = torch.softmax(logits, dim=1)
    action_idx = torch.argmax(probs, dim=1).item()

actions = ['standing', 'walking', 'running', 'jumping', 'sitting', 'reaching', 'waving', 'unknown']
predicted_action = actions[action_idx]
confidence = probs[0, action_idx].item()
```

-----

#### 方法 3: 時序序列分類 (Temporal Sequence Classification - LSTM/Transformer)

**優點：**

  - 利用時間上下文（多幀）
  - 更適合週期性動作（行走步態、奔跑）
  - 對單幀模糊更有魯棒性

**缺點：**

  - 需要序列數據
  - 更複雜
  - 較慢

**使用時機：** 影片序列、週期性動作、需要時間一致性

**實作：**

```python
class TemporalActionClassifier(nn.Module):
    """使用 LSTM 從姿勢序列分類動作"""

    def __init__(self, num_keypoints=17, hidden_size=128, num_classes=8, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_keypoints * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, num_keypoints * 2)
        lstm_out, _ = self.lstm(x)
        # 取最後一個時間步
        last_output = lstm_out[:, -1, :]
        logits = self.fc(last_output)
        return logits

# 使用 8 幀序列的用法
model = TemporalActionClassifier()
sequence = torch.stack([normalize_keypoints(kp) for kp in keypoint_sequence])  # (8, 34)
sequence = sequence.unsqueeze(0)  # (1, 8, 34)

with torch.no_grad():
    logits = model(sequence)
    predicted_action = torch.argmax(logits, dim=1).item()
```

-----

### 3.4 視角/角度分類 (View/Angle Classification)

**目標：** 確定攝影機角度以進行視角平衡訓練

#### 方法 1: 關鍵點幾何分析 (簡單)

**優點：**

  - 無需額外模型
  - 快速
  - 可解釋

**缺點：**

  - 準確度低於學習模型

**實作：**

```python
def classify_view_from_pose(keypoints, scores):
    """
    從姿勢關鍵點分類攝影機視角

    返回:
        view (str): 'front' (正面), 'back' (背面), 'left_side' (左側), 'right_side' (右側), 'three_quarter' (四分之三側)
    """
    # 提取關鍵關節
    nose = keypoints[0]
    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]
    l_ear, r_ear = keypoints[3], keypoints[4]

    # 肩寬 (圖像中可見的)
    shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)

    # 臀寬
    hip_width = np.linalg.norm(r_hip - l_hip)

    # 耳朵可見性
    l_ear_visible = scores[3] > 0.5
    r_ear_visible = scores[4] > 0.5

    # 鼻子相對於肩膀的位置偏移
    nose_offset = nose[0] - (l_shoulder[0] + r_shoulder[0]) / 2

    # 正面視角：雙肩可見，肩寬大，鼻子居中
    if (scores[5] > 0.5 and scores[6] > 0.5 and
        shoulder_width > hip_width * 0.7 and
        abs(nose_offset) < shoulder_width * 0.2):
        if l_ear_visible and r_ear_visible:
            return "front"
        else:
            return "three_quarter_front"

    # 背面視角：鼻子分數低，雙臀可見
    if scores[0] < 0.3 and scores[11] > 0.5 and scores[12] > 0.5:
        return "back"

    # 側面視角：單肩主導
    if scores[5] > 0.6 and scores[6] < 0.3:
        return "left_side"
    elif scores[6] > 0.6 and scores[5] < 0.3:
        return "right_side"

    # 四分之三視角 (預設)
    if nose_offset < 0:
        return "three_quarter_left"
    else:
        return "three_quarter_right"
```

-----

#### 方法 2: CNN 視角分類器 (學習型)

訓練一個簡單的 CNN 從關鍵點熱圖 (Heatmaps) 或圖像中分類視角。

-----

### 3.5 模型選擇決策樹 (Model Selection Decision Tree)

```
你的優先考量是什麼？

準確度 (Accuracy)
├─ 最佳: ViTPose++ (基于 transformer)
├─ 優異: RTM-Pose Large
└─ 良好: RTM-Pose Medium

速度 (Speed)
├─ 即時 (GPU): RTM-Pose Small/Tiny
├─ 即時 (CPU): MediaPipe Pose
└─ 最快: MediaPipe Lite

資源 (Resources)
├─ 有 GPU: RTM-Pose 或 ViTPose
├─ 僅 CPU: MediaPipe
└─ VRAM 有限: RTM-Pose Tiny 或 MediaPipe

動作分類 (Action Classification)
├─ 簡單動作: 基於規則 (關鍵點幾何學)
├─ 複雜動作: 關鍵點上的 ResNet 分類器
└─ 時序序列: LSTM/Transformer

數據集大小 (Dataset Size)
├─ 大 (>2k 姿勢): 訓練自定義動作分類器
├─ 中 (500-2k): 微調 ResNet 分類器
└─ 小 (<500): 使用基於規則的分類
```

-----

### 3.6 推薦流程 (3D 動畫角色)

**針對皮克斯風格的 3D 角色（如 Luca）：**

```yaml
pose_estimation:
  model: RTM-Pose  # 3D 的良好平衡
  config:
    model_size: medium  # rtmpose-m
    score_threshold: 0.3  # 針對風格化身體調低
    device: cuda

action_classification:
  method: rule_based  # 從簡單開始
  actions:
    - standing (站立)
    - walking (行走)
    - running (奔跑)
    - jumping (跳躍)
    - sitting (坐著)
    - reaching (伸手)
    - waving (揮手)

  # 可選: 如有需要使用 ML 分類器
  ml_fallback:
    enable: false
    model: resnet_keypoints
    confidence_threshold: 0.6

view_classification:
  method: keypoint_geometry
  views:
    - front (正面)
    - back (背面)
    - left_side (左側)
    - right_side (右側)
    - three_quarter (四分之三側)

pose_normalization:
  center: hip_center  # 歸一化到臀部中心
  scale: torso_height  # 按軀幹高度縮放
  remove_rotation: true  # 對齊垂直方向
```

-----

## 4\. 數據準備流程

### 4.1 流程總覽

```
角色實例 (來自 SAM2)
  ↓
姿勢估計 (RTM-Pose)
  ↓
動作分類 (基於規則或 ML)
  ↓
視角分類 (關鍵點幾何學)
  ↓
姿勢標準化 (移除位置/縮放/旋轉)
  ↓
品質過濾 (關鍵點可見性、置信度)
  ↓
按動作 + 視角聚類 (HDBSCAN 或手動分桶)
  ↓
人工審查 & 重新標記
  ↓
標註生成 (以動作為重點)
  ↓
數據集組裝 (Kohya_ss 格式)
```

### 4.2 逐步實作

#### 步驟 1: 姿勢估計 (Pose Estimation)

**目標：** 從所有角色實例中提取骨架關鍵點

```python
from mmpose.apis import init_model, inference_topdown
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import numpy as np

class PoseExtractor:
    """使用 RTM-Pose 從角色實例中提取姿勢"""

    def __init__(self, model_size='medium', device='cuda'):
        # 模型配置
        configs = {
            'tiny': ('rtmpose-t_8xb256-420e_coco-256x192.py', 'rtmpose-t.pth'),
            'small': ('rtmpose-s_8xb256-420e_coco-256x192.py', 'rtmpose-s.pth'),
            'medium': ('rtmpose-m_8xb256-420e_coco-256x192.py', 'rtmpose-m.pth'),
            'large': ('rtmpose-l_8xb256-420e_coco-384x288.py', 'rtmpose-l.pth'),
        }

        config_file, checkpoint_file = configs[model_size]
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.device = device

    def extract_pose(self, img, bbox=None):
        """
        從圖像中提取姿勢關鍵點

        參數:
            img: np.array (BGR)
            bbox: [x1, y1, x2, y2] 或 None (使用整張圖)

        返回:
            {
                'keypoints': (17, 2),
                'scores': (17,),
                'bbox': [x1, y1, x2, y2]
            }
        """
        h, w = img.shape[:2]

        if bbox is None:
            bbox = [0, 0, w, h]

        # 推論
        results = inference_topdown(self.model, img, bboxes=[bbox])

        if len(results) == 0:
            return None

        keypoints = results[0].pred_instances.keypoints[0]  # (17, 2)
        scores = results[0].pred_instances.keypoint_scores[0]  # (17,)

        return {
            'keypoints': keypoints,
            'scores': scores,
            'bbox': bbox,
        }

# 用法
extractor = PoseExtractor(model_size='medium', device='cuda')

instance_dir = Path("/path/to/character_instances")
output_dir = Path("/path/to/poses_extracted")
output_dir.mkdir(exist_ok=True)

pose_metadata = []

for img_path in tqdm(list(instance_dir.glob("*.png"))):
    img = cv2.imread(str(img_path))

    # 提取姿勢 (因為實例已被裁切，故使用整張圖像 bbox)
    pose_result = extractor.extract_pose(img, bbox=None)

    if pose_result is None:
        continue

    # 儲存元數據
    pose_metadata.append({
        'image_path': str(img_path),
        'keypoints': pose_result['keypoints'].tolist(),
        'scores': pose_result['scores'].tolist(),
        'bbox': pose_result['bbox'],
    })

# 儲存元數據
with open(output_dir / "pose_metadata.json", 'w') as f:
    json.dump(pose_metadata, f, indent=2)

print(f"✅ 已從 {len(pose_metadata)} 個實例中提取姿勢")
```

-----

#### 步驟 2: 動作分類 (Action Classification)

```python
def classify_all_actions(pose_metadata):
    """
    為所有姿勢進行動作分類

    參數:
        pose_metadata: 來自步驟 1 的姿勢字典列表

    返回:
        更新後的 pose_metadata (包含動作標籤)
    """
    for pose_meta in tqdm(pose_metadata, desc="正在分類動作"):
        keypoints = np.array(pose_meta['keypoints'])
        scores = np.array(pose_meta['scores'])

        # 使用基於規則的方法分類動作 (來自章節 3.3)
        action, confidence = classify_action_from_pose(keypoints, scores)

        pose_meta['action'] = action
        pose_meta['action_confidence'] = confidence

    # 統計數據
    action_counts = {}
    for pose in pose_metadata:
        action = pose['action']
        action_counts[action] = action_counts.get(action, 0) + 1

    print("動作分佈:")
    for action, count in sorted(action_counts.items()):
        print(f"  {action}: {count}")

    return pose_metadata
```

-----

#### 步驟 3: 視角分類 (View Classification)

```python
def classify_all_views(pose_metadata):
    """
    為所有姿勢分類攝影機視角
    """
    for pose_meta in tqdm(pose_metadata, desc="正在分類視角"):
        keypoints = np.array(pose_meta['keypoints'])
        scores = np.array(pose_meta['scores'])

        # 使用關鍵點幾何學分類視角 (來自章節 3.4)
        view = classify_view_from_pose(keypoints, scores)

        pose_meta['view'] = view

    # 統計數據
    view_counts = {}
    for pose in pose_metadata:
        view = pose['view']
        view_counts[view] = view_counts.get(view, 0) + 1

    print("視角分佈:")
    for view, count in sorted(view_counts.items()):
        print(f"  {view}: {count}")

    return pose_metadata
```

-----

#### 步驟 4: 姿勢標準化 (Pose Normalization)

**為什麼要標準化？** 移除位置、縮放和旋轉的變異，以便直接比較姿勢。

```python
def normalize_pose(keypoints):
    """
    標準化姿勢關鍵點

    步驟:
    1. 居中於臀部中心
    2. 按軀幹高度縮放
    3. 旋轉以垂直對齊軀幹

    參數:
        keypoints: (17, 2) COCO 關鍵點

    返回:
        normalized_keypoints: (17, 2)
    """
    # 提取關鍵關節
    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]

    # 計算中心
    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center = (l_hip + r_hip) / 2

    # 1. 居中於臀部
    centered = keypoints - hip_center

    # 2. 按軀幹高度縮放
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    if torso_height > 0:
        scaled = centered / torso_height
    else:
        scaled = centered

    # 3. 旋轉以垂直對齊軀幹
    torso_vector = shoulder_center - hip_center
    angle = np.arctan2(torso_vector[0], torso_vector[1])  # 與垂直線的夾角

    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    rotated = scaled @ rotation_matrix.T

    return rotated

# 應用於所有姿勢
for pose_meta in pose_metadata:
    keypoints = np.array(pose_meta['keypoints'])
    normalized = normalize_pose(keypoints)
    pose_meta['keypoints_normalized'] = normalized.tolist()
```

-----

#### 步驟 5: 品質過濾 (Quality Filtering)

```python
def filter_pose_quality(pose_metadata, config):
    """
    根據品質標準過濾姿勢

    參數:
        pose_metadata: 姿勢字典列表
        config: 品質過濾配置

    返回:
        過濾後的列表
    """
    filtered = []
    rejection_stats = defaultdict(int)

    for pose in pose_metadata:
        scores = np.array(pose['scores'])

        # 1. 最低關鍵點可見性
        visible_count = np.sum(scores > config['min_score'])
        if visible_count < config['min_visible_keypoints']:
            rejection_stats['too_few_visible_keypoints'] += 1
            continue

        # 2. 關鍵關節必須可見 (肩膀、臀部)
        key_joints = [5, 6, 11, 12]  # 肩膀 + 臀部
        if not all(scores[j] > config['min_key_joint_score'] for j in key_joints):
            rejection_stats['key_joints_not_visible'] += 1
            continue

        # 3. 動作置信度
        if pose['action_confidence'] < config['min_action_confidence']:
            rejection_stats['low_action_confidence'] += 1
            continue

        # 4. 模糊檢查 (針對原始圖像)
        img = cv2.imread(pose['image_path'], cv2.IMREAD_GRAYSCALE)
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        if blur_score < config['min_blur_score']:
            rejection_stats['too_blurry'] += 1
            continue

        pose['blur_score'] = blur_score
        filtered.append(pose)

    print(f"✅ 品質過濾: 保留 {len(filtered)} / {len(pose_metadata)}")
    print("❌ 拒絕原因:")
    for reason, count in sorted(rejection_stats.items()):
        print(f"  - {reason}: {count}")

    return filtered

# 配置
quality_config = {
    'min_score': 0.3,  # 關鍵點置信度閾值
    'min_visible_keypoints': 10,  # 至少 10/17 個關鍵點可見
    'min_key_joint_score': 0.5,  # 肩膀和臀部必須清晰
    'min_action_confidence': 0.5,
    'min_blur_score': 100.0,
}

filtered_poses = filter_pose_quality(pose_metadata, quality_config)
```

-----

#### 步驟 6: 按動作 + 視角整理 (Organize by Action + View)

**策略：** 按照動作類型和攝影機視角建立分桶 (buckets)，以進行平衡訓練。

```python
def organize_by_action_view(filtered_poses, output_dir):
    """
    將姿勢按動作和視角整理到資料夾中

    輸出結構:
        output_dir/
        ├── standing/
        │   ├── front/
        │   ├── side/
        │   └── back/
        ├── running/
        │   ├── front/
        │   └── side/
        └── ...
    """
    output_dir = Path(output_dir)

    # 按動作和視角分組
    groups = defaultdict(lambda: defaultdict(list))

    for pose in filtered_poses:
        action = pose['action']
        view = pose['view']
        groups[action][view].append(pose)

    # 建立資料夾並複製文件
    for action, view_dict in groups.items():
        for view, poses in view_dict.items():
            dest_dir = output_dir / action / view
            dest_dir.mkdir(parents=True, exist_ok=True)

            for pose in poses:
                src_path = Path(pose['image_path'])
                dst_path = dest_dir / src_path.name
                shutil.copy(src_path, dst_path)
                pose['organized_path'] = str(dst_path)

            print(f"📁 {action}/{view}: {len(poses)} images")

    return groups

action_view_groups = organize_by_action_view(
    filtered_poses,
    output_dir="/path/to/poses_organized"
)
```

-----

#### 步驟 7: 平衡數據集 (Balance Dataset)

**策略：** 跨動作**以及**視角進行平衡，以獲得魯棒的 LoRA。

```python
def balance_action_view_dataset(action_view_groups, target_per_bucket=50):
    """
    平衡各動作-視角分桶的數據集

    參數:
        action_view_groups: 嵌套字典 {action: {view: [poses]}}
        target_per_bucket: 每個動作-視角組合的目標圖像數量

    返回:
        平衡後的群組
    """
    balanced = defaultdict(dict)

    for action, view_dict in action_view_groups.items():
        for view, poses in view_dict.items():
            n_poses = len(poses)

            if n_poses >= target_per_bucket:
                # 欠採樣 (Undersample)
                sampled = list(np.random.choice(poses, size=target_per_bucket, replace=False))
                balanced[action][view] = sampled
                print(f"📉 {action}/{view}: {n_poses} → {target_per_bucket}")

            elif n_poses >= target_per_bucket // 2:
                # 過採樣 (Oversample) 至目標值
                repeats = target_per_bucket // n_poses
                remainder = target_per_bucket % n_poses
                balanced_list = poses * repeats
                balanced_list += list(np.random.choice(poses, size=remainder, replace=False))
                balanced[action][view] = balanced_list
                print(f"📈 {action}/{view}: {n_poses} → {target_per_bucket}")

            else:
                # 樣本太少，跳過或警告
                balanced[action][view] = poses
                print(f"⚠️  {action}/{view}: {n_poses} (too few, using all)")

    return balanced

balanced_groups = balance_action_view_dataset(action_view_groups, target_per_bucket=50)
```

-----

#### 步驟 8: 標註生成 (Caption Generation)

**姿勢 LoRA 的標註策略：**

不同於角色 LoRA，姿勢 LoRA 的標註應該：

1.  **強調動作**（主要焦點）
2.  包含姿勢/視角細節（次要）
3.  淡化或中立化身分 (Identify)
4.  保留 3D 風格描述詞

<!-- end list -->

```python
def generate_pose_caption(
    action: str,
    view: str = "front",
    include_3d_style: bool = True,
    character_name: Optional[str] = None
):
    """
    生成姿勢 LoRA 訓練標註

    參數:
        action: 動作名稱 (standing, running 等)
        view: 攝影機視角 (front, side, back 等)
        include_3d_style: 添加 3D 風格標籤
        character_name: 可選的角色名稱 (用於特定角色的姿勢 LoRA)

    返回:
        標註字串
    """
    # 動作描述詞
    action_descriptors = {
        'standing': {
            'base': 'standing upright',
            'details': 'neutral stance, feet shoulder-width apart, arms at sides',
        },
        'walking': {
            'base': 'walking',
            'details': 'mid-stride, one leg forward, natural gait',
        },
        'running': {
            'base': 'running',
            'details': 'dynamic motion, forward lean, arms pumping',
        },
        'jumping': {
            'base': 'jumping',
            'details': 'mid-air, legs bent, arms raised',
        },
        'sitting': {
            'base': 'sitting',
            'details': 'seated pose, legs bent, relaxed posture',
        },
        'reaching': {
            'base': 'reaching out',
            'details': 'extended arm, reaching gesture',
        },
        'waving': {
            'base': 'waving hand',
            'details': 'raised hand, greeting gesture',
        },
    }

    # 視角描述詞
    view_descriptors = {
        'front': 'front view',
        'back': 'back view',
        'left_side': 'left side view, profile',
        'right_side': 'right side view, profile',
        'three_quarter': 'three-quarter view',
        'three_quarter_left': 'three-quarter left view',
        'three_quarter_right': 'three-quarter right view',
    }

    parts = []

    # 角色描述 (除非指定否則通用)
    if character_name:
        parts.append(f"a 3d animated character {character_name}")
    else:
        parts.append("a 3d animated character")

    # 動作 (主要焦點)
    action_info = action_descriptors.get(action, {'base': action, 'details': ''})
    parts.append(action_info['base'])

    # 姿勢細節
    if action_info['details']:
        parts.append(action_info['details'])

    # 視角/角度
    view_desc = view_descriptors.get(view, view)
    parts.append(view_desc)

    # 3D 風格標籤
    if include_3d_style:
        parts.extend([
            "pixar style",
            "smooth shading",
            "detailed anatomy",
            "dynamic pose",
        ])

    caption = ", ".join(parts)
    return caption

# 範例
caption1 = generate_pose_caption('running', view='three_quarter')
# "a 3d animated character, running, dynamic motion, forward lean, arms pumping, three-quarter view, pixar style, smooth shading, detailed anatomy, dynamic pose"

caption2 = generate_pose_caption('standing', view='front', character_name='luca')
# "a 3d animated character luca, standing upright, neutral stance, feet shoulder-width apart, arms at sides, front view, pixar style, smooth shading, detailed anatomy, dynamic pose"
```

-----

#### 步驟 9: 組裝最終數據集 (Assemble Final Dataset)

```python
def assemble_pose_dataset(balanced_groups, output_dir, caption_generator):
    """
    組裝最終的姿勢 LoRA 訓練數據集

    輸出結構:
        output_dir/
        ├── images/
        │   ├── running_front_001.jpg
        │   ├── running_side_001.jpg
        │   ├── standing_front_001.jpg
        │   └── ...
        ├── captions/
        │   ├── running_front_001.txt
        │   └── ...
        └── metadata.json
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = {
        'total_images': 0,
        'action_view_counts': {},
    }

    global_idx = 0

    for action, view_dict in balanced_groups.items():
        for view, poses in view_dict.items():
            action_view_count = 0

            for pose in poses:
                # 複製圖像
                src_path = Path(pose['organized_path'])
                img_filename = f"{action}_{view}_{global_idx:04d}.jpg"
                dst_img_path = images_dir / img_filename
                shutil.copy(src_path, dst_img_path)

                # 生成標註
                caption = caption_generator(action, view=view, include_3d_style=True)

                # 儲存標註
                caption_path = captions_dir / f"{action}_{view}_{global_idx:04d}.txt"
                caption_path.write_text(caption, encoding='utf-8')

                global_idx += 1
                action_view_count += 1

            key = f"{action}/{view}"
            dataset_stats['action_view_counts'][key] = action_view_count
            dataset_stats['total_images'] += action_view_count

    # 儲存元數據
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    print(f"✅ 姿勢 LoRA 數據集已組裝: {dataset_stats['total_images']} 張圖像")
    print("動作-視角分佈:")
    for key, count in sorted(dataset_stats['action_view_counts'].items()):
        print(f"  {key}: {count}")

    return dataset_stats

# 用法
dataset_stats = assemble_pose_dataset(
    balanced_groups=balanced_groups,
    output_dir="/path/to/final_dataset",
    caption_generator=generate_pose_caption
)
```

-----

## 5\. 實作細節

### 5.1 完整流程腳本

```bash
#!/bin/bash
# prepare_pose_lora_data.sh

CHARACTER_INSTANCES="/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2/instances"
OUTPUT_DIR="/mnt/data/ai_data/training_data/3d_poses/luca_poses"

# 步驟 1-9: 全部在一個腳本中
python scripts/generic/training/prepare_pose_lora_data.py \
  --character-instances "$CHARACTER_INSTANCES" \
  --output-dir "$OUTPUT_DIR" \
  --pose-model rtmpose-m \
  --actions standing walking running jumping sitting reaching waving \
  --target-per-bucket 50 \
  --min-visible-keypoints 10 \
  --min-action-confidence 0.5 \
  --device cuda

echo "✅ 姿勢 LoRA 數據準備完成！"
echo "數據集就緒於: $OUTPUT_DIR/final_dataset"
```

-----

## 6\. 訓練策略

### 6.1 姿勢 LoRA 訓練設定

**TOML 設定檔：** `configs/training/pose_lora_sdxl.toml`

```toml
[model]
pretrained_model_name_or_path = "/path/to/sd_xl_base_1.0.safetensors"
vae = "/path/to/sdxl_vae.safetensors"

[dataset]
train_data_dir = "/path/to/final_dataset/images"
caption_extension = ".txt"
resolution = "1024,1024"
batch_size = 4
num_workers = 4

# 3D 不使用增強 (保持姿勢一致性)
color_aug = false
flip_aug = false  # 關鍵：翻轉會破壞左/右姿勢！
random_crop = false

[network]
network_module = "networks.lora"
network_dim = 56            # 姿勢建議使用中等 rank
network_alpha = 28          # alpha = rank/2

# 訓練 UNet 和 Text Encoder (姿勢既是視覺的也是文本的)
network_train_unet_only = false

[training]
output_dir = "/path/to/lora_outputs/pose_lora"
output_name = "running_pose"

learning_rate = 0.0003
unet_lr = 0.0003
text_encoder_lr = 0.0001    # Text Encoder 使用較低 LR

lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

max_train_epochs = 10
save_every_n_epochs = 2

# 混合精度
mixed_precision = "fp16"
full_fp16 = false

[optimizer]
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1"]

[noise]
noise_offset = 0.05

[saving]
save_model_as = "safetensors"
save_precision = "fp16"

[logging]
logging_dir = "/path/to/lora_outputs/pose_lora/logs"
log_with = "tensorboard"

[sample]
sample_every_n_epochs = 2
sample_prompts = "/path/to/prompts/pose_test_prompts.txt"
sample_sampler = "euler_a"
sample_steps = 30
```

### 6.2 姿勢專屬的訓練考量

**中等 Rank (48-64):**

  - 姿勢比表情複雜，但比完整角色簡單
  - 需要足夠的容量來容納不同的動作和視角

**同時訓練 UNet + Text Encoder:**

  - 姿勢既是視覺的（身體配置），也是文本的（動作動詞）
  - Text Encoder 有助於學習動作關鍵字

**禁止水平翻轉 (NO Horizontal Flips):**

  - **關鍵：** 翻轉會破壞姿勢語義（左踢 ≠ 右踢）
  - 不同於角色 LoRA 翻轉是可以接受的

**平衡視角分佈:**

  - 確保訓練數據包含正面、側面、背面視角
  - 防止視角偏差（例如，只在正面視角下有效）

**較長的訓練時間 (10-12 epochs):**

  - 變異性比表情 LoRA 大
  - 需要更多 epochs 來學習動作-視角組合

-----

### 6.3 多動作 vs 單動作 LoRA

與表情 LoRA 類似，有兩種策略：

#### 策略 A: 多動作 LoRA (All-in-One)

**數據集：** 所有動作在一個數據集中
**標註：** 包含動作關鍵字
**優點：** 單個 LoRA，學習共享的身體結構
**缺點：** 數據集較大，可能會混淆相似動作

#### 策略 B: 每個動作獨立 LoRA

**數據集：** 每個動作一個 LoRA (奔跑 LoRA, 跳躍 LoRA)
**標註：** 通用的身體姿勢描述
**優點：** 精細控制，易於除錯
**缺點：** 多個文件

**建議：** 一般用途使用多動作 LoRA，複雜動作（舞蹈、武術）使用獨立 LoRA

-----

## 7\. 測試與評估

### 7.1 測試提示詞 (Test Prompts)

```text
# Running poses (奔跑姿勢)
a 3d animated character running, dynamic motion, forward lean, three-quarter view, pixar style
a young boy running fast, sprinting, mid-stride, side view, 3d render, smooth shading
a character running forward, athletic pose, front view, pixar animation style

# Jumping poses (跳躍姿勢)
a 3d animated character jumping, mid-air, legs bent, arms raised, front view, pixar style
a character jumping high, excited leap, dynamic pose, three-quarter view, 3d render

# Standing poses (站立姿勢)
a 3d animated character standing upright, neutral stance, front view, pixar style, smooth shading
a character standing casually, relaxed pose, hands in pockets, side view, 3d animation

# Sitting poses (坐姿)
a 3d animated character sitting on chair, relaxed posture, side view, pixar style
a character sitting cross-legged, casual pose, front view, 3d render
```

### 7.2 評估指標

1.  **姿勢準確度 (Pose Accuracy):** 在生成的圖像上使用姿勢估計器，與目標動作進行比較
2.  **視角一致性 (View Consistency):** 從不同視角生成相同動作，檢查連貫性
3.  **角色遷移 (Character Transfer):** 使用不同的角色描述測試姿勢

-----

## 8\. 常見問題

### 問題 1: 姿勢 LoRA 沒有效果

**解決方案：**

  - 增加 LoRA 權重 (1.0-1.5)
  - 使用與訓練時完全相同的動作動詞（例如用 "running" 而不是 "jogging"）
  - 檢查基礎模型是否已經知道該姿勢
  - 使用更高的 Rank (64) 重新訓練

### 問題 2: 姿勢僅在單一視角有效，其他視角無效

**原因：** 訓練中的視角分佈不平衡
**解決方案：**

  - 嚴格平衡各視角的數據
  - 收集更多側面/背面視角數據
  - 如果需要，為每個視角訓練單獨的 LoRA

### 問題 3: 姿勢過擬合 (Overfits) 於特定角色

**原因：** 所有訓練數據都來自同一個角色
**解決方案：**

  - 從標註中移除角色名稱
  - 在訓練數據中混合多個角色
  - 降低 Rank 以防止死記硬背

### 問題 4: 左/右混淆

**原因：** 訓練期間使用了水平翻轉
**解決方案：**

  - 在訓練配置中**禁用 flip\_aug**
  - 如果需要，在標註中添加 left/right 關鍵字

-----

## 總結：姿勢 LoRA 實作檢查清單

  - [ ] 選擇姿勢估計模型 (推薦 RTM-Pose)
  - [ ] 從角色實例中提取關鍵點
  - [ ] 分類動作 (基於規則或 ML)
  - [ ] 分類攝影機視角 (關鍵點幾何學)
  - [ ] 標準化姿勢 (居中、縮放、旋轉)
  - [ ] 按品質過濾 (可見性、置信度、模糊度)
  - [ ] 按動作**以及**視角整理
  - [ ] 平衡各分桶 (buckets) 的數據集
  - [ ] 生成以動作為重點的標註
  - [ ] 組裝 Kohya\_ss 數據集
  - [ ] 配置訓練 (Rank 48-64, 禁止翻轉, 同時訓練 UNet+TE)
  - [ ] 使用視角平衡的數據進行訓練
  - [ ] 跨動作、視角、角色進行測試
  - [ ] 評估姿勢準確度和一致性
  - [ ] 對常見問題進行除錯

-----

**下一篇指南：**

  - `BACKGROUND_LORA_DEEP_DIVE.md` - 場景理解與背景 LoRA
  - `STYLE_LORA_DEEP_DIVE.md` - 藝術風格轉移 LoRA

**本指南是 3D 動畫 LoRA 流程專案的一部分。**