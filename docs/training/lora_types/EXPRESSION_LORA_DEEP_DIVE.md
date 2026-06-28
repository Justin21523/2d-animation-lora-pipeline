這是一份將說明文字翻譯為全中文（保留關鍵技術名詞中英對照）的文檔版本，內容結構與程式碼部分完全保持不變。

-----

# 表情 LoRA 深度剖析：技術實作指南 (Expression LoRA Deep Dive)

**目標受眾：** 實作 3D 動畫角色面部表情 LoRA 的機器學習工程師
**難度等級：** ⚠️ **高** (風格化 3D 面孔的情緒檢測)
**先決條件：** 角色 LoRA (Character LoRA) 的理解、人臉檢測基礎、聚類 (Clustering) 知識

-----

## 目錄 (Table of Contents)

1.  [為什麼表情 LoRA 具有挑戰性](https://www.google.com/search?q=%231-%E7%82%BA%E4%BB%80%E9%BA%BC%E8%A1%A8%E6%83%85-lora-%E5%85%B7%E6%9C%89%E6%8C%91%E6%88%B0%E6%80%A7)
2.  [理論基礎](https://www.google.com/search?q=%232-%E7%90%86%E8%AB%96%E5%9F%BA%E7%A4%8E)
3.  [模型選擇框架](https://www.google.com/search?q=%233-%E6%A8%A1%E5%9E%8B%E9%81%B8%E6%93%87%E6%A1%86%E6%9E%B6)
4.  [數據準備流程](https://www.google.com/search?q=%234-%E6%95%B8%E6%93%9A%E6%BA%96%E5%82%99%E6%B5%81%E7%A8%8B)
5.  [實作細節](https://www.google.com/search?q=%235-%E5%AF%A6%E4%BD%9C%E7%B4%B0%E7%AF%80)
6.  [訓練策略](https://www.google.com/search?q=%236-%E8%A8%93%E7%B7%B4%E7%AD%96%E7%95%A5)
7.  [測試與評估](https://www.google.com/search?q=%237-%E6%B8%AC%E8%A9%A6%E8%88%87%E8%A9%95%E4%BC%B0)
8.  [常見問題](https://www.google.com/search?q=%238-%E5%B8%B8%E8%A6%8B%E5%95%8F%E9%A1%8C)

-----

## 1\. 為什麼表情 LoRA 具有挑戰性

### 1.1 核心問題 (The Core Problem)

表情 LoRA 的目標是教導模型學習特定的面部表情（如開心、悲傷、生氣、驚訝、中性等），並且要**獨立於角色身分 (independently of character identity)**。這在本質上比角色 LoRA 更難，因為：

**對於角色 LoRA (Character LoRA)：**

  - 身分在不同幀之間是一致的
  - 人臉嵌入向量 (Face embeddings) 自然聚類
  - 3D 模型具有穩定的幾何結構和紋理
  - 清晰的視覺一致性

**對於表情 LoRA (Expression LoRA)：**

  - **同一個角色在不同幀中可能有截然不同的表情**
  - 表情變化可能是**細微的**（微表情）或**極端的**（卡通誇張化）
  - 3D 動畫表情是**風格化的 (stylized)**，而非寫實的
  - 真人情緒模型與 3D 動畫臉孔之間存在**領域差距 (Domain gap)**
  - 動畫中的**比例誇張**（大眼睛、簡化的特徵）

### 1.2 3D 動畫特有的挑戰

| 挑戰 | 真人臉孔 (Real Faces) | 3D 動畫臉孔 (3D Animated Faces) | 影響 |
|-----------|------------|-------------------|--------|
| **面部幾何** | 自然人類比例 | 風格化 (大眼、小鼻) | 預訓練模型可能會失敗 |
| **表情強度** | 細微的微表情 | 為了可讀性而誇張化 | 需要更廣泛的情緒範圍 |
| **紋理與陰影** | 毛孔、皺紋 | 平滑的 PBR 材質 | 較少的紋理線索 |
| **眼睛大小** | 成比例 | 通常大 2-3 倍 | 基於眼睛的情緒模型不可靠 |
| **動態模糊** | 影片中常見 | 刻意的景深 (DoF) 模糊 | 人臉檢測更困難 |
| **光照風格化** | 自然 | 電影/藝術效果 | 改變表情的視覺呈現 |

**範例：** 皮克斯 (Pixar) 角色的「驚訝」表情，其眼睛可能佔據臉部面積的 40%，這打破了針對真人臉孔訓練的情緒檢測模型之假設。

### 1.3 為什麼我們需要表情 LoRA

儘管面臨挑戰，表情 LoRA 仍然非常有價值，因為：

1.  **控制生成圖像的情緒基調**
2.  **與角色 LoRA 結合使用**，例如「{角色名稱} 帶著開心的表情」
3.  **可跨角色重複使用**（例如，「生氣」表情適用於 Luca、Alberto 等）
4.  **數據集需求較小**，比起訓練包含所有表情的完整角色模型
5.  **組合式生成 (Compositional generation)**：角色 LoRA + 表情 LoRA + 姿勢 LoRA

-----

## 2\. 理論基礎

### 2.1 什麼是表情 LoRA？

表情 LoRA 在潛在空間 (latent space) 中學習**面部表情特定的特徵**：

```
基礎模型 Latent → 表情 LoRA → 表情條件化 Latent
```

**關鍵概念：** 表情特徵應該是**與身分無關 (identity-agnostic)** 的。「微笑」應該能跨越不同的角色進行轉移。

### 2.2 LoRA 數學原理 (快速回顧)

與角色 LoRA 相同：

```
W' = W + ΔW
ΔW = A × B

其中:
- W: 原始權重矩陣 (凍結)
- A: 降維投影矩陣 (d_model × rank)
- B: 升維投影矩陣 (rank × d_model)
- rank << d_model (通常表情設為 32-64)
```

### 2.3 表情 LoRA vs 角色 LoRA

| 面向 | 角色 LoRA | 表情 LoRA |
|--------|---------------|-----------------|
| **目標** | 身分保留 (Identity preservation) | 情緒/表情轉移 |
| **數據集大小** | 200-500 張圖像 | 每個表情類別 100-300 張 |
| **多樣性需求** | 多樣的姿勢/角度 | 同一表情下的多樣身分 |
| **標註 (Caption) 重點** | 角色名稱 + 細節 | 表情關鍵字 + 強度 |
| **訓練輪數 (Epochs)** | 8-12 | 6-10 |
| **Rank (維度)** | 32-64 | 32-48 (較低以避免過擬合) |
| **可組合性** | 可與其他疊加 | **必須** 與角色 LoRA 疊加 |

**關鍵區別：** 表情 LoRA 應該適用於**多個角色**，而不僅僅是一個。

-----

## 3\. 模型選擇框架

本節針對：「有些難度特別高的 情緒動作偵測等等 一定有很多相關模型 可是要如何應用就是必須考量到多種因素」

### 3.1 任務拆解

表情 LoRA 的數據準備需要 **3 個子任務**：

1.  **人臉檢測 (Face Detection)**：在角色實例中定位臉部
2.  **表情分類 (Expression Classification)**：將表情分類（開心、悲傷、生氣等）
3.  **表情強度估計 (Expression Intensity Estimation)**：情緒有多強烈？（可選）

每個子任務都有多種模型選擇，各有優缺點。

-----

### 3.2 人臉檢測模型 (Face Detection Models)

**目標：** 定位 3D 動畫角色的臉部區域

#### 選項 1: RetinaFace (推薦用於 3D)

**優點：**

  - 對臉部尺寸變化有很強的魯棒性
  - 處理卡通/風格化臉孔的效果相當不錯
  - 返回 5 點面部地標（眼睛、鼻子、嘴角）
  - 推論速度快 (GPU 上每張圖約 20ms)

**缺點：**

  - 訓練於真人臉孔（存在一些領域差距）
  - 可能難以處理極端風格化（非常大的眼睛）

**使用時機：** 半寫實比例的皮克斯風格 3D 角色的預設選擇

**實作：**

```python
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

faces = app.get(img)  # 返回邊界框 + 地標
for face in faces:
    bbox = face.bbox  # [x1, y1, x2, y2]
    landmarks = face.kps  # 5 個關鍵點
```

**3D 特有的調整：**

```python
# 調整風格化臉孔的檢測閾值
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)  # 較低的閾值
```

-----

#### 選項 2: YOLOv8-Face (最快)

**優點：**

  - **極快** (GPU 上約 10ms)
  - 對卡通臉孔有良好的泛化能力
  - 可在 3D 動畫臉孔上進行微調

**缺點：**

  - 預設沒有地標 (Landmarks)
  - 準確度略低於 RetinaFace

**使用時機：** 大規模處理 (\>1萬張圖像)，速度優先

**實作：**

```python
from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')
results = model.predict(img, conf=0.25)

for result in results:
    boxes = result.boxes.xyxy  # [x1, y1, x2, y2]
```

-----

#### 選項 3: MediaPipe Face Detection (輕量級)

**優點：**

  - **最輕量** (CPU 友善)
  - 專為多樣化臉型設計
  - 在風格化臉孔上效果出奇地好

**缺點：**

  - 在極端角度下準確度較低
  - 沒有情緒特徵

**使用時機：** 純 CPU 環境、快速原型製作

**實作：**

```python
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 1 = 全範圍 (0 = 短距離)
    min_detection_confidence=0.3
)

results = face_detection.process(img)
for detection in results.detections:
    bbox = detection.location_data.relative_bounding_box
```

-----

#### 選項 4: Anime Face Detector (專用)

**優點：**

  - **專門針對動漫/卡通臉孔訓練**
  - 能處理極端的眼睛大小和風格化
  - 開源 (nagadomi/lbpcascade\_animeface)

**缺點：**

  - 僅限於 2D 動漫風格（可能無法完美遷移到 3D）
  - 沒有地標
  - 較舊的技術 (Haar cascades)

**使用時機：** 類 2D 的 3D 動畫（例如：原神、崩壞：星穹鐵道）

**實作：**

```python
import cv2

cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
```

-----

### 3.3 表情分類模型 (Expression Classification Models)

**目標：** 將面部表情分類為情緒類別

#### 選項 1: FER+ (Emotion Recognition in the Wild) ⭐ 推薦

**優點：**

  - 在 FER2013+ 數據集（真人臉孔）上預訓練
  - 8 種情緒類別：中性、開心、悲傷、生氣、驚訝、恐懼、厭惡、輕蔑
  - 良好的基準準確度（真人臉孔約 70%）
  - 可在 3D 動畫臉孔上進行微調

**缺點：**

  - 領域差距（訓練於真人臉孔）
  - 可能會混淆風格化的表情
  - 需要將臉部裁切為 48×48 灰階圖

**使用時機：** 起點，在您的 3D 數據集上進行微調

**實作：**

```python
import torch
from torchvision import transforms
from fer_plus_model import FERPlusModel  # 自定義封裝

model = FERPlusModel.load_pretrained('fer_plus.pth')
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

face_crop = crop_face(img, bbox)
face_tensor = transform(face_crop).unsqueeze(0)

with torch.no_grad():
    logits = model(face_tensor)
    probs = torch.softmax(logits, dim=1)
    emotion_idx = torch.argmax(probs, dim=1).item()

emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'contempt']
predicted_emotion = emotions[emotion_idx]
confidence = probs[0, emotion_idx].item()
```

-----

#### 選項 2: AffectNet 預訓練模型 (較高準確度)

**優點：**

  - 訓練於更大的 AffectNet 數據集（45 萬張圖像）
  - 7-8 種情緒類別
  - 比 FER+ 更好的泛化能力
  - 多種骨幹網路選擇 (ResNet, EfficientNet, ViT)

**缺點：**

  - 仍然是訓練於真人臉孔
  - 模型尺寸較大
  - 需要更多預處理

**使用時機：** 需要較高準確度，願意犧牲速度

**實作：**

```python
import timm
import torch

# 使用 EfficientNet 骨幹
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=8)
model.load_state_dict(torch.load('affectnet_efficientnet_b0.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_crop = crop_face(img, bbox)
face_tensor = transform(face_crop).unsqueeze(0).to('cuda')

with torch.no_grad():
    logits = model(face_tensor)
    emotion_idx = torch.argmax(logits, dim=1).item()
```

-----

#### 選項 3: Vision Transformer for Emotion (SOTA) ⭐ 最佳準確度

**優點：**

  - 在情緒識別基準測試中達到 **SOTA (最高水準) 準確度**
  - 更好地處理局部遮擋
  - 注意力機制能捕捉細微的表情線索
  - 可處理更高解析度的臉孔 (224×224)

**缺點：**

  - 推論速度比 CNN 慢
  - 模型尺寸較大 (ViT-B 約 85M 參數)
  - 需要更多 VRAM

**使用時機：** 準確度至關重要，GPU 資源充足

**實作：**

```python
from transformers import ViTForImageClassification, ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('trpakov/vit-face-expression')
model = ViTForImageClassification.from_pretrained('trpakov/vit-face-expression')
model.to('cuda').eval()

face_crop = crop_face(img, bbox)
inputs = processor(images=face_crop, return_tensors="pt").to('cuda')

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

# 模型類別: angry, disgust, fear, happy, neutral, sad, surprise
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
predicted_emotion = emotions[predicted_class]
```

-----

#### 選項 4: 基於 CLIP 的情緒分類 (零樣本) 🔥 新穎方法

**優點：**

  - **無需在真人臉孔上訓練** - 使用視覺-語言對齊
  - **開箱即用於 3D 動畫臉孔**
  - 可定義自定義情緒描述
  - 更好地處理領域差距

**缺點：**

  - 準確度低於監督式模型
  - 較慢 (CLIP ViT-L 推論)
  - 需要仔細的提示詞工程 (Prompt engineering)

**使用時機：** 沒有已標註的 3D 情緒數據，偏好零樣本 (zero-shot)

**實作：**

```python
import torch
import clip
from PIL import Image

model, preprocess = clip.load("ViT-L/14", device="cuda")

# 定義情緒描述 (提示詞工程很關鍵!)
emotion_prompts = [
    "a 3d animated character with a happy, smiling expression",
    "a 3d animated character with a sad, frowning expression",
    "a 3d animated character with an angry, upset expression",
    "a 3d animated character with a surprised, shocked expression",
    "a 3d animated character with a neutral, calm expression",
    "a 3d animated character with a fearful, scared expression",
]

text_tokens = clip.tokenize(emotion_prompts).to("cuda")

face_crop = crop_face(img, bbox)
image_input = preprocess(face_crop).unsqueeze(0).to("cuda")

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

    # 歸一化特徵
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 餘弦相似度
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    emotion_idx = similarity.argmax().item()

emotions = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful']
predicted_emotion = emotions[emotion_idx]
confidence = similarity[0, emotion_idx].item()
```

-----

### 3.4 表情強度估計 (可選)

**目標：** 量化情緒的強烈程度（淺笑 vs 大笑）

#### 選項 1: AU (Action Unit) 檢測

**優點：**

  - 基於 FACS (面部動作編碼系統)
  - 客觀、細粒度 (例如 AU6 = 臉頰提升, AU12 = 嘴角拉動)
  - 可從 AU 激活計算情緒強度

**缺點：**

  - 複雜 (46 個 AUs)
  - 大多數模型訓練於真人臉孔
  - 計算開銷大

**使用時機：** 需要精確的表情分解、研究用途

**實作：**

```python
from feat import Detector

detector = Detector(device='cuda')
face_crop = crop_face(img, bbox)

aus = detector.detect_aus(face_crop)
# 返回 AU 激活值 (0-5 強度量表)
# 範例: {'AU6': 3.2, 'AU12': 4.1} → 中度微笑
```

-----

#### 選項 2: 基於分類置信度的啟發式強度 (Heuristic Intensity)

**優點：**

  - 無需額外模型
  - 快速
  - 適用於任何分類器

**缺點：**

  - 準確度低於基於 AU 的方法
  - 置信度 \!= 真實強度

**使用時機：** 簡單的強度估計即足夠

**實作：**

```python
# 使用分類模型的輸出機率
emotion_probs = model(face_tensor).softmax(dim=1)
max_prob = emotion_probs.max().item()

# 將置信度映射到強度
if max_prob > 0.8:
    intensity = "strong"
elif max_prob > 0.5:
    intensity = "moderate"
else:
    intensity = "subtle"
```

-----

### 3.5 模型選擇決策樹 (Model Selection Decision Tree)

```
開始 (START)
│
├─ 你有已標註的 3D 情緒數據嗎？
│  ├─ 是 → 在你的數據上微調 FER+ 或 AffectNet 模型
│  └─ 否 → 使用 CLIP 零樣本方法
│
├─ 你的優先考量是什麼？
│  ├─ 準確度 → Vision Transformer (基於 ViT 的情緒模型)
│  ├─ 速度 → YOLOv8-Face + FER+ (基於 ResNet)
│  └─ 平衡 → RetinaFace + AffectNet (EfficientNet)
│
├─ GPU 還是 CPU？
│  ├─ GPU → 以上任何模型
│  └─ CPU → MediaPipe Face + MobileNet-based 情緒分類器
│
└─ 數據集大小？
   ├─ 大 (>5k 臉孔) → 訓練自定義 3D 情緒分類器
   ├─ 中 (500-5k) → 微調 AffectNet 模型
   └─ 小 (<500) → CLIP 零樣本或直接使用預訓練模型
```

-----

### 3.6 推薦流程 (3D 動畫角色)

**針對皮克斯風格的 3D 角色（如 Luca）：**

```yaml
face_detection:
  model: RetinaFace  # 風格化臉孔的良好平衡
  config:
    det_size: [640, 640]
    det_thresh: 0.3  # 對 3D 臉孔設低一點

expression_classification:
  primary_model: CLIP_zero_shot  # 處理 3D 領域差距
  fallback_model: FER+  # 用於模糊情況

  clip_config:
    model: ViT-L/14
    prompts:
      - "a 3d animated character with a joyful, happy smile"
      - "a 3d animated character with a sad, crying expression"
      - "a 3d animated character with an angry, mad face"
      - "a 3d animated character with a surprised, shocked look"
      - "a 3d animated character with a neutral, calm face"
      - "a 3d animated character looking scared or fearful"
    confidence_threshold: 0.6

  fer_plus_config:
    use_when: clip_confidence < 0.6  # 備援
    model_path: models/fer_plus.pth

intensity_estimation:
  method: confidence_based  # 簡單的啟發式方法
```

-----

## 4\. 數據準備流程

既然我們選擇了模型，現在來構建實際的流程。

### 4.1 流程總覽

```
角色實例 (來自 SAM2)
  ↓
人臉檢測 (RetinaFace)
  ↓
表情分類 (CLIP + FER+ 混合)
  ↓
過濾 (置信度、品質、臉部大小)
  ↓
按表情聚類 (HDBSCAN 或手動分桶)
  ↓
人工審查 & 重新標記 (互動式 UI)
  ↓
標註生成 (以表情為重點)
  ↓
數據集組裝 (Kohya_ss 格式)
```

### 4.2 逐步實作

#### 步驟 1: 人臉檢測與裁切 (Face Detection & Cropping)

**目標：** 從角色實例中提取臉部區域

```python
from insightface.app import FaceAnalysis
from pathlib import Path
import cv2
import json
from tqdm import tqdm

class FaceExtractor:
    def __init__(self, det_thresh=0.3, det_size=(640, 640)):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)

    def extract_faces(self, img_path):
        """
        從圖像中提取所有臉孔

        返回:
            List[Dict]: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'landmarks': [[x,y], ...],  # 5 點
                    'det_score': 0.95,
                    'face_crop': np.array,  # 裁切的臉部區域
                }
            ]
        """
        img = cv2.imread(str(img_path))
        faces = self.app.get(img)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # 增加填充 (對於延伸出臉部框的表情很重要)
            pad_ratio = 0.2
            w, h = x2 - x1, y2 - y1
            pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)

            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(img.shape[1], x2 + pad_w)
            y2 = min(img.shape[0], y2 + pad_h)

            face_crop = img[y1:y2, x1:x2]

            results.append({
                'bbox': [x1, y1, x2, y2],
                'landmarks': face.kps.tolist(),
                'det_score': float(face.det_score),
                'face_crop': face_crop,
            })

        return results

# 用法
extractor = FaceExtractor(det_thresh=0.3)

instance_dir = Path("/path/to/character_instances")
output_dir = Path("/path/to/faces")
output_dir.mkdir(exist_ok=True)

face_metadata = []

for img_path in tqdm(list(instance_dir.glob("*.png"))):
    faces = extractor.extract_faces(img_path)

    for i, face in enumerate(faces):
        # 儲存臉部裁切圖
        face_filename = f"{img_path.stem}_face{i}.jpg"
        face_path = output_dir / face_filename
        cv2.imwrite(str(face_path), face['face_crop'])

        # 儲存元數據
        face_metadata.append({
            'source_image': str(img_path),
            'face_id': i,
            'face_path': str(face_path),
            'bbox': face['bbox'],
            'det_score': face['det_score'],
        })

# 儲存元數據
with open(output_dir / "face_metadata.json", 'w') as f:
    json.dump(face_metadata, f, indent=2)
```

**3D 特有的考量：**

  - **填充比例 0.2-0.3**：誇張的表情可能會超出典型的人臉邊界框
  - **較低的檢測閾值 0.3**：風格化臉孔的分數可能低於真人臉孔
  - **臉部大小過濾**：跳過非常小的臉孔 (\< 64×64)，因為缺乏表情細節

-----

#### 步驟 2: 表情分類 (混合方法)

**策略：** 使用 CLIP 作為主要方法，FER+ 作為低置信度情況下的備援

```python
import torch
import clip
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np

class HybridExpressionClassifier:
    def __init__(self, device='cuda'):
        self.device = device

        # 載入 CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)

        # 定義情緒提示詞 (對 3D 很關鍵!)
        self.emotion_prompts = [
            "a 3d animated character with a happy, joyful, smiling expression, pixar style",
            "a 3d animated character with a sad, unhappy, crying expression, pixar style",
            "a 3d animated character with an angry, mad, upset expression, pixar style",
            "a 3d animated character with a surprised, shocked, amazed expression, pixar style",
            "a 3d animated character with a neutral, calm, relaxed expression, pixar style",
            "a 3d animated character with a fearful, scared, worried expression, pixar style",
        ]

        self.emotion_labels = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'fearful']
        self.text_tokens = clip.tokenize(self.emotion_prompts).to(device)

        # 預先計算文本特徵
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        # 載入 FER+ 作為備援 (可選)
        self.fer_model = None  # 若有需要則初始化
        self.clip_confidence_threshold = 0.6

    def classify_clip(self, face_crop):
        """
        使用 CLIP 零樣本分類表情

        返回:
            emotion (str), confidence (float)
        """
        image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 餘弦相似度
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            confidence, emotion_idx = similarity.max(dim=-1)

        emotion = self.emotion_labels[emotion_idx.item()]
        confidence = confidence.item()

        return emotion, confidence

    def classify(self, face_crop):
        """
        混合分類：CLIP 為主，FER+ 為備援
        """
        emotion, confidence = self.classify_clip(face_crop)

        # 高置信度 → 返回 CLIP 結果
        if confidence >= self.clip_confidence_threshold:
            return {
                'emotion': emotion,
                'confidence': confidence,
                'method': 'clip',
            }

        # 低置信度 → 若可用，備援到 FER+
        if self.fer_model is not None:
            fer_emotion, fer_conf = self.classify_fer(face_crop)
            return {
                'emotion': fer_emotion,
                'confidence': fer_conf,
                'method': 'fer_fallback',
                'clip_prediction': emotion,
                'clip_confidence': confidence,
            }

        # 無備援 → 返回 CLIP 並帶有警告
        return {
            'emotion': emotion,
            'confidence': confidence,
            'method': 'clip_low_confidence',
        }

# 用法
classifier = HybridExpressionClassifier(device='cuda')

for face_meta in face_metadata:
    face_crop = cv2.imread(face_meta['face_path'])
    result = classifier.classify(face_crop)

    face_meta['expression'] = result['emotion']
    face_meta['expression_confidence'] = result['confidence']
    face_meta['classification_method'] = result['method']
```

**針對 3D 的提示詞工程技巧：**

  - 包含 "3d animated character" 以限定領域
  - 添加風格描述詞 ("pixar style", "smooth shading")
  - 使用多個同義詞 ("happy, joyful, smiling")
  - 在樣本圖像上測試提示詞並迭代

-----

#### 步驟 3: 品質過濾 (Quality Filtering)

**表情數據的過濾標準：**

```python
def filter_expression_samples(face_metadata, config):
    """
    根據品質標準過濾臉部樣本

    參數:
        face_metadata: 包含表情預測的臉部字典列表
        config: 品質過濾配置

    返回:
        過濾後的列表
    """
    filtered = []
    rejection_stats = defaultdict(int)

    for face in face_metadata:
        # 1. 檢測置信度
        if face['det_score'] < config['min_detection_score']:
            rejection_stats['low_detection_score'] += 1
            continue

        # 2. 表情分類置信度
        if face['expression_confidence'] < config['min_expression_confidence']:
            rejection_stats['low_expression_confidence'] += 1
            continue

        # 3. 臉部大小 (太小缺乏表情細節)
        bbox = face['bbox']
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]

        if face_width < config['min_face_size'] or face_height < config['min_face_size']:
            rejection_stats['face_too_small'] += 1
            continue

        # 4. 長寬比 (極端比例表示臉部不完整)
        aspect_ratio = face_width / max(face_height, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            rejection_stats['extreme_aspect_ratio'] += 1
            continue

        # 5. 模糊檢測 (可選，重複使用角色 LoRA 的方法)
        face_crop = cv2.imread(face['face_path'], cv2.IMREAD_GRAYSCALE)
        blur_score = cv2.Laplacian(face_crop, cv2.CV_64F).var()
        if blur_score < config['min_blur_score']:
            rejection_stats['too_blurry'] += 1
            continue

        filtered.append(face)

    print(f"✅ Kept: {len(filtered)} / {len(face_metadata)} faces")
    print("❌ Rejections:")
    for reason, count in rejection_stats.items():
        print(f"  - {reason}: {count}")

    return filtered

# 配置
quality_config = {
    'min_detection_score': 0.7,
    'min_expression_confidence': 0.5,  # 比檢測低 (表情更難)
    'min_face_size': 80,  # 像素
    'min_blur_score': 100.0,
}

filtered_faces = filter_expression_samples(face_metadata, quality_config)
```

**3D 特有的閾值：**

  - 表情置信度 0.5-0.6 (身分為 0.7-0.8) - 表情本質上更模糊
  - 較大的最小臉部尺寸 (80-100px) 以捕捉表情細節
  - 更嚴格的長寬比以避免局部遮擋

-----

#### 步驟 4: 按表情聚類 (Clustering by Expression)

**兩種方法：**

##### 方法 A: 直接使用預測標籤 (較簡單)

```python
from collections import defaultdict
from pathlib import Path
import shutil

def organize_by_expression(filtered_faces, output_dir):
    """
    將臉孔按預測的表情整理到資料夾中
    """
    output_dir = Path(output_dir)

    # 按表情分組
    expression_groups = defaultdict(list)
    for face in filtered_faces:
        expression = face['expression']
        expression_groups[expression].append(face)

    # 建立資料夾並複製文件
    for expression, faces in expression_groups.items():
        expr_dir = output_dir / expression
        expr_dir.mkdir(parents=True, exist_ok=True)

        for face in faces:
            src_path = Path(face['face_path'])
            dst_path = expr_dir / src_path.name
            shutil.copy(src_path, dst_path)

        print(f"📁 {expression}: {len(faces)} faces")

    return expression_groups

expression_groups = organize_by_expression(
    filtered_faces,
    output_dir="/path/to/expressions_organized"
)
```

##### 方法 B: 基於嵌入向量的聚類 (更魯棒)

適用於預測標籤雜訊較多的情況：

```python
import umap
import hdbscan
from sklearn.preprocessing import normalize

def cluster_expressions_embedding(filtered_faces, clip_model, device='cuda'):
    """
    按 CLIP 圖像嵌入進行聚類以發現自然的表情分組
    """
    # 提取 CLIP 圖像嵌入
    embeddings = []
    for face in tqdm(filtered_faces, desc="Extracting embeddings"):
        face_crop = cv2.imread(face['face_path'])
        image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        image_input = clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            embeddings.append(image_features.cpu().numpy().flatten())

    embeddings = np.array(embeddings)
    embeddings = normalize(embeddings)

    # UMAP + HDBSCAN
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,  # 表情聚類用較小值
        min_samples=2,
        metric='euclidean',
        cluster_selection_epsilon=0.0
    )
    labels = clusterer.fit_predict(embedding_2d)

    # 分配聚類標籤
    for i, face in enumerate(filtered_faces):
        face['cluster_id'] = int(labels[i])

    # 分析聚類 (每個聚類代表什麼表情？)
    cluster_analysis = defaultdict(lambda: defaultdict(int))
    for face in filtered_faces:
        if face['cluster_id'] >= 0:  # 跳過雜訊 (-1)
            cluster_analysis[face['cluster_id']][face['expression']] += 1

    # 為每個聚類分配主要表情
    cluster_to_expression = {}
    for cluster_id, expr_counts in cluster_analysis.items():
        dominant_expr = max(expr_counts, key=expr_counts.get)
        cluster_to_expression[cluster_id] = dominant_expr
        print(f"Cluster {cluster_id} → {dominant_expr} ({expr_counts})")

    return labels, cluster_to_expression
```

**何時使用哪種方法：**

  - **方法 A (直接標籤)：** 分類置信度高 (\>0.7)，預測乾淨
  - **方法 B (聚類)：** 預測有雜訊，希望發現自然的分組

-----

#### 步驟 5: 人工審查與重新標記 (Manual Review & Relabeling)

**關鍵步驟：** 表情分類並不完美，尤其是在 3D 臉孔上。

**互動式審查 UI 需求：**

  - 顯示按表情分組的臉孔
  - 允許拖放以在表情文件夾之間移動臉孔
  - 能夠建立自定義表情類別（例如：除了基本的 6 種情緒外，還有「傻笑」、「咧嘴笑」）
  - 標記強度（微弱、中等、強烈）
  - 批量重新標記相似的臉孔

**實作** (重複使用/調整角色聚類的互動選擇器)：

```bash
python scripts/generic/clustering/interactive_expression_selector.py \
  --expression-dir /path/to/expressions_organized \
  --output-dir /path/to/expressions_reviewed
```

**所需的 UI 功能：**

  - 所有表情的網格視圖
  - 按置信度過濾（先審查低置信度的）
  - 比較視圖（並排顯示相似的臉孔）
  - 註釋模式（添加強度標籤、備註）

-----

#### 步驟 6: 平衡表情數據集 (Balance Dataset)

**問題：** 來源影片中的表情分佈通常是不平衡的（大量的「中性」，很少的「生氣」）

**解決方案：** 分層採樣 (Stratified sampling) 或數據增強

```python
def balance_expression_dataset(expression_groups, target_per_expression=100):
    """
    平衡各表情的數據集

    策略:
    1. 過採樣少數類別 (重複)
    2. 欠採樣多數類別 (隨機採樣)
    3. 組合
    """
    balanced_groups = {}

    for expression, faces in expression_groups.items():
        n_faces = len(faces)

        if n_faces >= target_per_expression:
            # 欠採樣 (Undersample)
            sampled = np.random.choice(faces, size=target_per_expression, replace=False)
            balanced_groups[expression] = sampled.tolist()
            print(f"📉 {expression}: {n_faces} → {target_per_expression} (undersampled)")

        else:
            # 過採樣 (Oversample) (重複直到達到目標)
            repeats = target_per_expression // n_faces
            remainder = target_per_expression % n_faces

            balanced = faces * repeats
            balanced += np.random.choice(faces, size=remainder, replace=False).tolist()
            balanced_groups[expression] = balanced
            print(f"📈 {expression}: {n_faces} → {target_per_expression} (oversampled {repeats}x)")

    return balanced_groups

balanced_groups = balance_expression_dataset(expression_groups, target_per_expression=150)
```

**替代方案：** 如果願意為每個表情訓練單獨的 LoRA，則無需平衡。

-----

#### 步驟 7: 標註生成 (Caption Generation)

**表情 LoRA 的標註策略：**

不同於角色 LoRA（強調身分），表情 LoRA 的標註應該：

1.  **強調表情**（主要焦點）
2.  淡化或中立化身分細節
3.  包含強度（如果相關）
4.  保留 3D 風格描述詞

**基於模板的標註：**

```python
def generate_expression_caption(
    expression: str,
    intensity: str = "moderate",
    include_3d_style: bool = True
):
    """
    生成表情 LoRA 訓練標註

    參數:
        expression: happy, sad, angry, surprised, neutral, fearful
        intensity: subtle, moderate, strong
        include_3d_style: 添加 3D 風格標籤

    返回:
        標註字串
    """
    # 基礎表情描述
    expression_descriptors = {
        'happy': {
            'subtle': "a slight smile, content expression",
            'moderate': "a happy smile, cheerful expression",
            'strong': "a wide smile, very happy and joyful expression",
        },
        'sad': {
            'subtle': "a slightly sad, downcast expression",
            'moderate': "a sad, unhappy expression with frown",
            'strong': "a very sad, crying expression with tears",
        },
        'angry': {
            'subtle': "a slightly annoyed, frowning expression",
            'moderate': "an angry, upset expression with furrowed brows",
            'strong': "a very angry, furious expression, shouting",
        },
        'surprised': {
            'subtle': "a slightly surprised, curious expression",
            'moderate': "a surprised, shocked expression with wide eyes",
            'strong': "a very surprised, amazed expression with open mouth",
        },
        'neutral': {
            'subtle': "a calm, relaxed expression",
            'moderate': "a neutral, calm face",
            'strong': "a completely neutral expression, no emotion",
        },
        'fearful': {
            'subtle': "a slightly worried, nervous expression",
            'moderate': "a scared, fearful expression with wide eyes",
            'strong': "a very scared, terrified expression, panicked",
        },
    }

    parts = []

    # 角色描述 (通用的，非特定身分)
    parts.append("a 3d animated character")

    # 表情 (主要焦點)
    expr_desc = expression_descriptors[expression][intensity]
    parts.append(expr_desc)

    # 3D 風格標籤
    if include_3d_style:
        parts.extend([
            "pixar style",
            "smooth shading",
            "detailed facial features",
            "expressive animation",
        ])

    caption = ", ".join(parts)
    return caption

# 用法範例
caption_happy = generate_expression_caption('happy', intensity='moderate')
# 輸出: "a 3d animated character, a happy smile, cheerful expression, pixar style, smooth shading, detailed facial features, expressive animation"
```

**基於 VLM 的標註 (可選):**

用於更細緻的描述：

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def generate_vlm_expression_caption(face_crop, expression_hint):
    """
    使用 VLM 生成詳細的表情標註
    """
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = f"""Describe this 3D animated character's facial expression in detail.
Focus on:
1. The primary emotion ({expression_hint})
2. Specific facial features (eyes, mouth, eyebrows)
3. Expression intensity
4. Animation style

Keep description concise (40-60 tokens). Use tags separated by commas.
Do NOT mention character identity or clothing.

Example format: "a 3d animated character, [expression details], pixar style, smooth shading"
"""

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(face_crop)},
            {"type": "text", "text": prompt}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[Image.fromarray(face_crop)],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=80)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return caption
```

**混合方法：**

  - 模板標註用於乾淨、一致的訓練
  - VLM 標註用於驗證集或增加多樣性

-----

#### 步驟 8: 數據集組裝 (Dataset Assembly)

**Kohya\_ss 格式的表情 LoRA：**

```python
def assemble_expression_dataset(
    balanced_groups,
    output_dir,
    caption_generator,
    character_name=None  # 可選: 如果是為特定角色訓練表情
    ):
    """
    組裝最終的表情 LoRA 訓練數據集

    輸出結構:
        output_dir/
        ├── images/
        │   ├── happy_001.jpg
        │   ├── happy_002.jpg
        │   ├── sad_001.jpg
        │   └── ...
        ├── captions/
        │   ├── happy_001.txt
        │   ├── happy_002.txt
        │   └── ...
        └── metadata.json
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = {
        'total_images': 0,
        'expression_counts': {},
        'caption_method': 'template',
    }

    for expression, faces in balanced_groups.items():
        expr_count = 0

        for i, face in enumerate(faces):
            # 複製臉部圖像
            src_path = Path(face['face_path'])
            img_filename = f"{expression}_{i:04d}.jpg"
            dst_img_path = images_dir / img_filename
            shutil.copy(src_path, dst_img_path)

            # 生成標註
            intensity = face.get('intensity', 'moderate')
            caption = caption_generator(expression, intensity=intensity)

            # 可選：添加角色名稱前綴
            if character_name:
                caption = f"{character_name}, {caption}"

            # 儲存標註
            caption_path = captions_dir / f"{expression}_{i:04d}.txt"
            caption_path.write_text(caption, encoding='utf-8')

            expr_count += 1

        dataset_stats['expression_counts'][expression] = expr_count
        dataset_stats['total_images'] += expr_count

    # 儲存元數據
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    print(f"✅ Expression LoRA dataset assembled: {dataset_stats['total_images']} images")
    print(f"   Expression distribution: {dataset_stats['expression_counts']}")

# 用法
assemble_expression_dataset(
    balanced_groups=balanced_groups,
    output_dir="/path/to/lora_training/expression_lora",
    caption_generator=generate_expression_caption,
    character_name=None  # 通用表情 LoRA
)
```

-----

### 4.3 完整流程腳本

整合所有步驟：

```bash
#!/bin/bash
# prepare_expression_lora_data.sh

CHARACTER_INSTANCES="/mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2/instances"
OUTPUT_DIR="/mnt/data/ai_data/training_data/3d_expressions/luca_expressions"

# 步驟 1: 提取臉部
python scripts/generic/training/processors/expression/face_extractor.py \
  --input-dir "$CHARACTER_INSTANCES" \
  --output-dir "$OUTPUT_DIR/faces_extracted" \
  --det-thresh 0.3 \
  --min-face-size 80 \
  --device cuda

# 步驟 2: 分類表情
python scripts/generic/training/processors/expression/expression_classifier.py \
  --input-dir "$OUTPUT_DIR/faces_extracted" \
  --output-dir "$OUTPUT_DIR/expressions_classified" \
  --model clip_hybrid \
  --confidence-threshold 0.5 \
  --device cuda

# 步驟 3: 過濾品質
python scripts/generic/training/processors/expression/quality_filter.py \
  --input-dir "$OUTPUT_DIR/expressions_classified" \
  --output-dir "$OUTPUT_DIR/expressions_filtered" \
  --min-expression-confidence 0.5 \
  --min-blur-score 100

# 步驟 4: 按表情整理
python scripts/generic/training/processors/expression/organizer.py \
  --input-dir "$OUTPUT_DIR/expressions_filtered" \
  --output-dir "$OUTPUT_DIR/expressions_organized" \
  --method direct_labels  # 或 'embedding_clustering'

# 步驟 5: 互動式審查 (手動)
python scripts/generic/clustering/interactive_expression_selector.py \
  --expression-dir "$OUTPUT_DIR/expressions_organized" \
  --output-dir "$OUTPUT_DIR/expressions_reviewed"

# 步驟 6: 平衡數據集
python scripts/generic/training/processors/expression/balancer.py \
  --input-dir "$OUTPUT_DIR/expressions_reviewed" \
  --output-dir "$OUTPUT_DIR/expressions_balanced" \
  --target-per-expression 150

# 步驟 7: 生成標註 & 組裝數據集
python scripts/generic/training/prepare_expression_lora_data.py \
  --input-dir "$OUTPUT_DIR/expressions_balanced" \
  --output-dir "$OUTPUT_DIR/final_dataset" \
  --caption-method template \
  --include-intensity

echo "✅ Expression LoRA data preparation complete!"
echo "Dataset ready: $OUTPUT_DIR/final_dataset"
```

-----

## 5\. 實作細節

### 5.1 模組化程式碼結構

遵循 `MODULAR_ARCHITECTURE.md` 的模組化架構：

```
scripts/generic/training/
├── processors/
│   └── expression/
│       ├── __init__.py
│       ├── face_extractor.py          # RetinaFace 封裝
│       ├── expression_classifier.py   # CLIP + FER+ 混合
│       ├── quality_filter.py          # 按置信度、大小、模糊度過濾
│       ├── organizer.py               # 按表情聚類/整理
│       ├── balancer.py                # 平衡數據集
│       └── caption_generator.py       # 模板 + VLM 標註
│
├── models/
│   ├── face_detection/
│   │   ├── retinaface_backend.py
│   │   ├── yolov8face_backend.py
│   │   └── mediapipe_backend.py
│   │
│   └── expression/
│       ├── clip_classifier.py
│       ├── fer_plus_classifier.py
│       ├── affectnet_classifier.py
│       └── vit_emotion_classifier.py
│
└── prepare_expression_lora_data.py    # 主要 CLI 入口點
```

### 5.2 配置驅動設計 (Configuration-Driven Design)

**設定檔：** `configs/stages/expression/expression_lora.yaml`

```yaml
face_detection:
  backend: retinaface  # retinaface | yolov8face | mediapipe
  det_thresh: 0.3
  det_size: [640, 640]
  min_face_size: 80
  padding_ratio: 0.25

expression_classification:
  backend: clip_hybrid  # clip_hybrid | fer_plus | affectnet | vit_emotion

  clip:
    model: ViT-L/14
    confidence_threshold: 0.6
    prompts:
      happy: "a 3d animated character with a happy, joyful, smiling expression, pixar style"
      sad: "a 3d animated character with a sad, unhappy, crying expression, pixar style"
      angry: "a 3d animated character with an angry, mad, upset expression, pixar style"
      surprised: "a 3d animated character with a surprised, shocked, amazed expression, pixar style"
      neutral: "a 3d animated character with a neutral, calm, relaxed expression, pixar style"
      fearful: "a 3d animated character with a fearful, scared, worried expression, pixar style"

  fer_plus:
    model_path: models/fer_plus.pth
    use_as_fallback: true

quality_filter:
  min_detection_score: 0.7
  min_expression_confidence: 0.5
  min_face_size: 80
  min_blur_score: 100.0
  max_aspect_ratio_deviation: 0.5

clustering:
  method: direct_labels  # direct_labels | embedding_clustering

  embedding_clustering:
    enable: false
    min_cluster_size: 10
    min_samples: 2

balancing:
  enable: true
  target_per_expression: 150
  strategy: hybrid  # oversample_minority | undersample_majority | hybrid

caption:
  method: template  # template | vlm | hybrid
  include_intensity: true
  include_character_name: false

  template:
    include_3d_style: true

  vlm:
    model: Qwen2-VL-7B-Instruct
    max_tokens: 80
```

### 5.3 範例：`expression_classifier.py`

```python
#!/usr/bin/env python3
"""
Expression Classification Module

Supports multiple backends:
- CLIP zero-shot (recommended for 3D)
- FER+ (real-face baseline)
- AffectNet (higher accuracy)
- ViT Emotion (SOTA)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import torch
import clip
import cv2
import json
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

from core.utils.logger import setup_logger


class CLIPExpressionClassifier:
    """基於 CLIP 的零樣本表情分類器"""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.logger = setup_logger(self.__class__.__name__)

        # 載入 CLIP
        self.logger.info(f"Loading CLIP model: {config.clip.model}")
        self.model, self.preprocess = clip.load(config.clip.model, device=device)

        # 準備文本提示詞
        self.emotion_labels = list(config.clip.prompts.keys())
        self.prompts = [config.clip.prompts[label] for label in self.emotion_labels]

        self.logger.info(f"Loaded {len(self.emotion_labels)} emotion classes:")
        for label in self.emotion_labels:
            self.logger.info(f"  - {label}")

        # 編碼文本提示詞
        text_tokens = clip.tokenize(self.prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def classify(self, face_crop):
        """
        分類表情

        參數:
            face_crop: np.array (BGR)

        返回:
            {
                'emotion': str,
                'confidence': float,
                'all_scores': dict
            }
        """
        # 預處理圖像
        image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # 編碼圖像
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # 計算相似度
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            confidence, emotion_idx = similarity.max(dim=-1)

        emotion = self.emotion_labels[emotion_idx.item()]
        confidence = confidence.item()

        # 所有分數
        all_scores = {
            label: float(similarity[0, i])
            for i, label in enumerate(self.emotion_labels)
        }

        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_scores': all_scores,
        }


def main():
    parser = argparse.ArgumentParser(description="Expression classification")
    parser.add_argument('--input-dir', type=str, required=True, help="Face images directory")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    parser.add_argument('--config', type=str, default='configs/stages/expression/expression_lora.yaml')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Initialize classifier
    if config.expression_classification.backend == 'clip_hybrid':
        classifier = CLIPExpressionClassifier(config.expression_classification, device=args.device)
    else:
        raise NotImplementedError(f"Backend {config.expression_classification.backend} not implemented")

    # Process faces
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load face metadata
    metadata_path = input_dir / "face_metadata.json"
    with open(metadata_path, 'r') as f:
        face_metadata = json.load(f)

    # Classify all faces
    logger = setup_logger(__name__)
    logger.info(f"Classifying {len(face_metadata)} faces...")

    for face_meta in tqdm(face_metadata):
        face_path = Path(face_meta['face_path'])
        face_crop = cv2.imread(str(face_path))

        result = classifier.classify(face_crop)

        face_meta['expression'] = result['emotion']
        face_meta['expression_confidence'] = result['confidence']
        face_meta['expression_scores'] = result['all_scores']

    # Save updated metadata
    output_metadata_path = output_dir / "face_metadata.json"
    with open(output_metadata_path, 'w') as f:
        json.dump(face_metadata, f, indent=2)

    logger.info(f"✅ Classification complete! Results saved to {output_metadata_path}")

    # Print distribution
    expression_counts = {}
    for face in face_metadata:
        expr = face['expression']
        expression_counts[expr] = expression_counts.get(expr, 0) + 1

    logger.info("Expression distribution:")
    for expr, count in sorted(expression_counts.items()):
        logger.info(f"  {expr}: {count}")


if __name__ == '__main__':
    main()
```

-----

## 6\. 訓練策略

### 6.1 表情 LoRA 訓練設定

**TOML 設定檔：** `configs/training/expression_lora_sdxl.toml`

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

# 3D 不使用增強 (保持表情一致性)
color_aug = false
flip_aug = false
random_crop = false

[network]
network_module = "networks.lora"
network_dim = 48            # 表情使用較低的 rank (相比角色的 64)
network_alpha = 24          # alpha = rank/2

# 僅訓練 UNet (不訓練 text encoder) 用於表情
network_train_unet_only = true

[training]
output_dir = "/path/to/lora_outputs/expression_lora"
output_name = "happy_expression"

learning_rate = 0.0003
unet_lr = 0.0003
text_encoder_lr = 0.0          # 不訓練 text encoder

lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

max_train_epochs = 8            # Epochs 比角色 LoRA 少
save_every_n_epochs = 2

# 混合精度
mixed_precision = "fp16"
full_fp16 = false

[optimizer]
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1"]

[noise]
noise_offset = 0.05
adaptive_noise_scale = 0.0

[saving]
save_model_as = "safetensors"
save_precision = "fp16"

[logging]
logging_dir = "/path/to/lora_outputs/expression_lora/logs"
log_with = "tensorboard"
log_prefix = "expression_lora"

[sample]
sample_every_n_epochs = 2
sample_prompts = "/path/to/prompts/expression_test_prompts.txt"
sample_sampler = "euler_a"
sample_steps = 30
```

### 6.2 表情專屬的訓練考量

**較低的 Rank (32-48):**

  - 表情是比完整角色身分更簡單的特徵
  - 較低的 rank 可防止過擬合到特定角色的身分
  - 提高跨角色的泛化能力

**僅訓練 UNet (Train UNet Only):**

  - 表情主要是視覺的，而非文本的
  - 訓練 Text encoder 可能導致過擬合到特定的標註措辭
  - 僅訓練 UNet 可保持更好的可組合性

**較短的訓練時間 (6-10 epochs):**

  - 表情比身分更容易過擬合
  - 密切監控驗證損失 (validation loss)
  - 如果驗證指標停滯則提前停止

**無數據增強 (No Data Augmentation):**

  - 水平翻轉可能會反轉表情語義（左邊微笑 → 右邊微笑）
  - 顏色增強會破壞表情線索（例如，在某些語境下紅色=生氣，藍色=悲傷）
  - 裁切可能會排除重要的表情特徵（嘴巴、眼睛）

-----

### 6.3 多表情 vs 單表情 LoRA

**兩種訓練策略：**

#### 策略 A: 多表情 LoRA (All-in-One)

**數據集：** 所有表情在一個數據集中（開心、悲傷、生氣等）
**標註：** 在每個標註中包含表情關鍵字
**優點：**

  - 單個 LoRA 文件
  - 學習共享的表情特徵

**缺點：**

  - 需要較大的數據集
  - 如果表情相似可能會混淆
  - 對個別表情強度的控制較少

**訓練指令：**

```bash
conda run -n ai_env accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file=configs/training/expression_lora_sdxl_multi.toml
```

-----

#### 策略 B: 每個表情獨立 LoRA (模組化)

**數據集：** 每個表情一個數據集（開心 LoRA、悲傷 LoRA 等）
**標註：** 通用標註，不包含情緒關鍵字
**優點：**

  - 細粒度控制（調整每個表情的權重）
  - 可以組合多個表情 LoRA
  - 更容易針對每個表情的問題進行除錯

**缺點：**

  - 需要管理多個 LoRA 文件
  - 需要在推論時進行堆疊

**訓練指令：**

```bash
# 訓練開心 LoRA
conda run -n ai_env accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file=configs/training/expression_lora_sdxl_happy.toml

# 訓練悲傷 LoRA
conda run -n ai_env accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file=configs/training/expression_lora_sdxl_sad.toml

# ... 其他表情依此類推
```

**建議：**

  - 如果數據集較大（總共 \> 600 張圖像），使用 **多表情** 以簡化流程
  - 為了最大程度的控制和可組合性，使用 **獨立 LoRA**

-----

## 7\. 測試與評估

### 7.1 測試提示詞設計

**目標：** 驗證表情 LoRA 在不同情況下是否有效：

1.  角色（如果是通用的）
2.  姿勢
3.  場景/背景
4.  提示詞措辭

**測試提示詞模板：**

```
{character_description}, {expression_keyword}, {pose}, {scene}, {style_tags}
```

**範例提示詞：**

```text
# Happy expression (開心表情)
a 3d animated character, happy smile, cheerful, standing, outdoor sunny park, pixar style, smooth shading
a young boy character, joyful grin, laughing, sitting on bench, bright daylight, 3d render
a female character, smiling warmly, waving hand, indoor cafe, soft lighting, pixar animation style

# Sad expression (悲傷表情)
a 3d animated character, sad frown, crying, sitting alone, rainy street, pixar style, moody lighting
a child character, unhappy, downcast eyes, standing, bedroom, dim light, 3d animated
a character, melancholic expression, tears, close-up portrait, neutral background, smooth shading

# Angry expression (生氣表情)
a 3d animated character, angry, furrowed brows, shouting, clenched fists, dramatic lighting, pixar style
a male character, upset, mad face, pointing finger, argument scene, indoor, 3d render

# Surprised expression (驚訝表情)
a 3d animated character, surprised, wide eyes, open mouth, shocked, sudden event, bright flash, pixar style
a character, amazed, gasping, hands on cheeks, colorful background, 3d animation

# Neutral expression (中性表情)
a 3d animated character, neutral, calm, relaxed, standing, plain background, pixar style, soft lighting
```

儲存到 `prompts/expression_lora_test.txt`

-----

### 7.2 自動化評估

**指標：**

1.  **表情分類準確度：** 使用相同的分類器驗證生成的圖像是否具有預期的表情
2.  **角色多樣性：** 驗證表情是否能轉移到不同的角色描述上
3.  **視覺品質：** CLIP score, FID

**評估腳本：**

```python
#!/usr/bin/env python3
"""
評估表情 LoRA 品質
"""

import torch
from diffusers import StableDiffusionXLPipeline
from pathlib import Path
import json
from tqdm import tqdm

# 載入表情分類器 (重複使用數據準備階段的)
from scripts.generic.training.processors.expression.expression_classifier import CLIPExpressionClassifier

def evaluate_expression_lora(
    lora_path,
    base_model_path,
    test_prompts,
    expression_classifier,
    output_dir,
    device='cuda'
):
    """
    評估表情 LoRA

    指標:
    1. 表情準確度: 生成的圖像是否有預期的表情？
    2. 一致性: 不同提示詞下表情是否相同？
    3. 角色多樣性: 是否適用於不同的角色描述？
    """
    # 載入流程
    pipe = StableDiffusionXLPipeline.from_single_file(
        base_model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    ).to(device)

    # 載入 LoRA
    pipe.load_lora_weights(lora_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, prompt_data in enumerate(tqdm(test_prompts)):
        prompt = prompt_data['prompt']
        expected_expression = prompt_data['expected_expression']

        # 生成圖像
        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.manual_seed(42)
        ).images[0]

        # 儲存圖像
        img_path = output_dir / f"test_{i:03d}_{expected_expression}.png"
        image.save(img_path)

        # 分類生成圖像中的表情
        # (需要先檢測人臉，然後分類)
        # 為了簡化，假設圖像主要是人臉
        import cv2
        import numpy as np

        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 簡單做法：分類整張圖 (不理想，但作為快速測試)
        classification = expression_classifier.classify(img_bgr)

        # 檢查是否匹配預期
        is_correct = classification['emotion'] == expected_expression

        results.append({
            'prompt': prompt,
            'expected_expression': expected_expression,
            'predicted_expression': classification['emotion'],
            'confidence': classification['confidence'],
            'is_correct': is_correct,
            'image_path': str(img_path)
        })

    # 計算指標
    accuracy = sum(r['is_correct'] for r in results) / len(results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)

    metrics = {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'total_tests': len(results),
        'correct_predictions': sum(r['is_correct'] for r in results),
        'results': results
    }

    # 儲存報告
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Evaluation complete!")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Avg Confidence: {avg_confidence:.2f}")
    print(f"   Report: {report_path}")

    return metrics

# 用法
test_prompts = [
    {'prompt': 'a 3d animated character, happy smile, pixar style', 'expected_expression': 'happy'},
    {'prompt': 'a character, sad frown, crying, 3d render', 'expected_expression': 'sad'},
    # ... 更多提示詞
]

config = OmegaConf.load('configs/stages/expression/expression_lora.yaml')
classifier = CLIPExpressionClassifier(config.expression_classification)

evaluate_expression_lora(
    lora_path='/path/to/happy_expression.safetensors',
    base_model_path='/path/to/sd_xl_base_1.0.safetensors',
    test_prompts=test_prompts,
    expression_classifier=classifier,
    output_dir='outputs/expression_lora_eval'
)
```

-----

## 8\. 常見問題與解決方案

### 問題 1: 表情 LoRA 沒有效果

**症狀：**

  - 無論有無表情 LoRA，生成的圖像看起來都一樣
  - 提示詞中的表情關鍵字被忽略

**可能原因：**

1.  **LoRA 權重太低：** 嘗試更高的權重 (1.0-1.5)
2.  **基礎模型已經知道該表情：** SDXL 可能已經將 "happy" 與微笑關聯
3.  **標註關鍵字不在訓練數據中：** 使用了 "joyful" 但訓練時用的是 "happy"
4.  **表情特徵太弱：** Rank 太低，訓練不足

**解決方案：**

```python
# 增加 LoRA 權重
pipe.load_lora_weights(lora_path, adapter_name="expression")
pipe.set_adapters(["expression"], adapter_weights=[1.2])  # 嘗試 1.0-1.5

# 使用準確的訓練標註關鍵字
# 如果訓練時用 "happy smile"，就使用相同的詞組

# 使用更高的 rank 重新訓練
network_dim = 64  # 代替 32

# 訓練更久
max_train_epochs = 12  # 代替 8
```

-----

### 問題 2: 表情過擬合到特定角色

**症狀：**

  - 表情 LoRA 僅對訓練數據中的角色有效（例如：只有 Luca）
  - 無法轉移到新的角色描述上

**可能原因：**

1.  **所有訓練數據都來自同一個角色：** 沒有多樣性
2.  **標註包含角色名稱：** 模型學習到 "Luca + happy" 而不是通用的 "happy"
3.  **Rank 太高：** 記住了角色身分

**解決方案：**

```python
# 解決方案 1: 在多個角色上訓練
# 在同一個表情類別中包含 Luca、Alberto、Giulia 等的臉孔

# 解決方案 2: 從標註中移除角色名稱
# 不好:  "Luca, happy smile, ..."
# 好: "a 3d animated character, happy smile, ..."

# 解決方案 3: 降低 rank
network_dim = 32  # 代替 64
network_alpha = 16

# 解決方案 4: 更強的正則化
optimizer_args = ["weight_decay=0.2"]  # 從 0.1 增加
```

-----

### 問題 3: 表情看起來不自然

**症狀：**

  - 表情在技術上是正確的，但看起來「怪怪的」
  - 恐怖谷效應
  - 誇張或太細微

**可能原因：**

1.  **訓練數據的表情強度不一致**
2.  **低品質的臉部裁切（局部臉孔、遮擋）**
3.  **混合的表情數據**（中性 + 稍微開心被標記為「開心」）

**解決方案：**

```python
# 解決方案 1: 數據準備期間嚴格的品質過濾
quality_config = {
    'min_expression_confidence': 0.7,  # 更嚴格
    'min_face_size': 120,              # 更大的臉孔
    'min_blur_score': 150.0,           # 只要更清晰的
}

# 解決方案 2: 手動審查並清理標籤
# 移除模糊不清的案例

# 解決方案 3: 按強度分開
# 將 "happy_subtle", "happy_moderate", "happy_strong" 訓練為不同的 LoRA

# 解決方案 4: 在標註中添加強度控制
caption = "a 3d character, happy smile (moderate intensity), pixar style"
```

-----

### 問題 4: 表情與角色 LoRA 衝突

**症狀：**

  - 角色 LoRA + 表情 LoRA 一起使用產生奇怪的結果
  - 一個 LoRA 壓倒另一個

**可能原因：**

1.  **兩個 LoRA 都在相同的層上訓練：** 更新衝突
2.  **不相容的標註格式：** 不同的分詞
3.  **權重不平衡**

**解決方案：**

```python
# 解決方案 1: 調整相對權重
pipe.load_lora_weights(character_lora_path, adapter_name="character")
pipe.load_lora_weights(expression_lora_path, adapter_name="expression")

# 嘗試不同的權重組合
pipe.set_adapters(
    ["character", "expression"],
    adapter_weights=[1.0, 0.8]  # 表情稍微低一點
)

# 解決方案 2: 在不同的層上訓練表情 LoRA
# 在訓練配置中:
network_args = ["train_unet_only=True", "unet_lr=0.0003"]
# 角色 LoRA 訓練了 UNet + text encoder
# 表情 LoRA 僅訓練 UNet → 衝突較少

# 解決方案 3: 兩者使用相同的標註風格
# 角色和表情數據集都應該使用:
# "a 3d animated character, {details}, pixar style, smooth shading"
```

-----

### 問題 5: 數據集不平衡影響訓練

**症狀：**

  - 模型偏向於常見表情（中性、開心）
  - 罕見表情（恐懼、厭惡）幾乎無效

**可能原因：**

1.  **來源數據不平衡：** 大多數幀都是中性的
2.  **數據集準備期間沒有平衡**

**解決方案：**

```python
# 解決方案 1: 積極平衡
balanced_groups = balance_expression_dataset(
    expression_groups,
    target_per_expression=200,  # 強制相等大小
)

# 解決方案 2: 訓練中的類別權重 (如果訓練器支持)
# 對少數類別自定義損失權重

# 解決方案 3: 用增強過採樣少數類別
# (小心：增強可能會扭曲表情)

# 解決方案 4: 為罕見表情訓練單獨的 LoRA
# 專用的 "fearful" LoRA，使用所有可用樣本
```

-----

## 總結：表情 LoRA 實作檢查清單

  - [ ] 理解為什麼表情 LoRA 比角色 LoRA 更難（領域差距、風格化）
  - [ ] 選擇人臉檢測模型（推薦用於 3D 的 RetinaFace）
  - [ ] 選擇表情分類模型（用於 3D 的 CLIP 零樣本，FER+ 作為備援）
  - [ ] 從角色實例中提取帶有填充的臉孔
  - [ ] 使用混合方法分類表情
  - [ ] 按品質過濾（置信度、大小、模糊度、長寬比）
  - [ ] 按表情整理臉孔（直接標籤或嵌入聚類）
  - [ ] 手動審查和重新標記（關鍵步驟！）
  - [ ] 平衡各表情的數據集
  - [ ] 生成標註（模板或 VLM，強調表情）
  - [ ] 組裝 Kohya\_ss 格式數據集
  - [ ] 配置訓練（較低 rank 32-48，僅 UNet，6-10 epochs，無增強）
  - [ ] 決定：多表情 vs 獨立 LoRA
  - [ ] 使用適當的超參數進行訓練
  - [ ] 使用多樣化的提示詞進行測試（角色、姿勢、場景）
  - [ ] 評估準確度、一致性、可轉移性
  - [ ] 對常見問題進行除錯（無效果、過擬合、不自然、衝突）
  - [ ] 根據結果進行迭代

-----

**下一篇指南：**

  - `BACKGROUND_LORA_DEEP_DIVE.md` - 場景理解與背景 LoRA
  - `POSE_LORA_DEEP_DIVE.md` - 姿勢估計與動作 LoRA
  - `STYLE_LORA_DEEP_DIVE.md` - 藝術風格轉移 LoRA

**本指南是 3D 動畫 LoRA 流程專案的一部分。**
**如有問題或議題，請參考：** `docs/training/MULTI_LORA_IMPLEMENTATION_GUIDE.md`