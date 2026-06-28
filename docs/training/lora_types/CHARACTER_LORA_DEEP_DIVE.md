# Character LoRA 深度技術指南

**類型**: Identity Preservation LoRA
**目標**: 學習並保持特定角色的外觀特徵
**難度**: ⭐⭐⭐⭐ (中高)

---

## 📚 理論基礎

### 什麼是 Character LoRA？

Character LoRA 訓練模型學習特定角色的**身份特徵 (identity features)**，包括：
- 面部特徵 (facial features)
- 髮型和髮色 (hairstyle & color)
- 眼睛形狀和顏色 (eye shape & color)
- 體型和比例 (body proportions)
- 標誌性服裝 (signature clothing)
- 獨特配飾 (unique accessories)

### 核心原理

**LoRA (Low-Rank Adaptation)** 在 Stable Diffusion 的 U-Net 中注入低秩矩陣：

```
W' = W + ΔW
ΔW = A × B

其中:
W: 原始權重矩陣 (frozen)
A: 下投影矩陣 (rank × d_model)
B: 上投影矩陣 (d_model × rank)
rank << d_model (典型: 32-64 vs 768-2048)
```

**為什麼有效？**
- 只更新 <1% 的參數
- 保留 base model 的泛化能力
- 注入特定角色的特徵分布

### 3D 動畫角色的特殊性

與 2D 動畫/真人相比：
1. **一致性更高** - 3D 模型本身就是固定的
2. **光照變化大** - PBR 材質導致外觀隨光照變化
3. **姿態多樣** - 動畫中姿態變化豐富
4. **需要更少數據** - 200-500 張 vs 2D 的 500-1000 張

---

## 🔧 數據準備詳解

### 步驟 1: Instance Segmentation (SAM2)

**目的**: 從視頻幀中提取純淨的角色 instances

**技術選擇**:
| 方法 | 優點 | 缺點 | 推薦場景 |
|------|------|------|---------|
| **SAM2** | 高精度、實例分離 | 慢 (0.5-1s/幀) | **推薦** - 最終訓練 |
| **U²-Net** | 快速 | 無實例分離 | 快速預覽 |
| **ISNet (RMBG)** | 邊緣質量好 | 無實例分離 | 單角色場景 |

**SAM2 關鍵參數 (3D 優化)**:
```python
# 3D 動畫特定設置
sam2_config = {
    'model_type': 'sam2_hiera_large',
    'min_instance_size': 4096,      # 最小 instance 大小 (64×64)
    'points_per_side': 32,          # 提示點網格密度
    'pred_iou_thresh': 0.88,        # IoU 閾值
    'stability_score_thresh': 0.95, # 穩定性閾值
    'box_nms_thresh': 0.7,          # NMS 閾值
    'crop_n_layers': 1,             # 裁剪層數
}
```

**為什麼這些參數對 3D 重要？**
- `min_instance_size`: 3D 角色通常較大，過濾小噪聲
- `pred_iou_thresh`: 較高閾值確保邊緣質量（3D 有柔和抗鋸齒）
- `stability_score_thresh`: 高穩定性適合一致的 3D 模型

**執行**:
```bash
python scripts/generic/segmentation/instance_segmentation.py \
  --input-dir /path/to/frames \
  --output-dir /path/to/instances \
  --model-type sam2_hiera_large \
  --min-instance-size 4096 \
  --save-masks \
  --context-mode transparent \
  --device cuda
```

**輸出結構**:
```
instances/
├── frame_0001_instance_0.png  # 角色 instance (RGBA)
├── frame_0001_instance_1.png
├── frame_0001_instance_2.png
├── masks/
│   ├── frame_0001_mask_0.png  # 對應的 mask
│   └── ...
└── metadata.json              # instance 信息
```

---

### 步驟 2: Identity Clustering (Face Recognition)

**目的**: 將屬於同一角色的 instances 聚類到一起

**為什麼需要 Face-based Clustering？**
- **問題**: CLIP/視覺相似度會將「相同場景的不同角色」聚在一起
- **解決**: 使用 **ArcFace embeddings** 進行身份識別

**技術方案**:

#### 方案 A: ArcFace + HDBSCAN (推薦)

```python
from insightface.app import FaceAnalysis
import hdbscan
import numpy as np

# 1. 初始化 ArcFace
face_app = FaceAnalysis(
    name='buffalo_l',  # 或 'antelopev2'
    providers=['CUDAExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 2. 提取 face embeddings
face_embeddings = []
valid_instances = []

for instance_path in instance_paths:
    img = cv2.imread(str(instance_path))
    faces = face_app.get(img)

    if len(faces) > 0:
        # 取最大的臉
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        face_embeddings.append(face.embedding)  # 512-dim
        valid_instances.append(instance_path)

face_embeddings = np.array(face_embeddings)

# 3. Normalize embeddings
face_embeddings_norm = face_embeddings / np.linalg.norm(face_embeddings, axis=1, keepdims=True)

# 4. HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=12,    # 3D: 較小值 (一致性高)
    min_samples=2,          # 3D: 較小值
    metric='euclidean',
    cluster_selection_method='eom'
)

labels = clusterer.fit_predict(face_embeddings_norm)
```

**3D 特定調整**:
```python
# 3D 角色聚類參數
clustering_params_3d = {
    'min_cluster_size': 12,   # 2D: 20-25, 3D: 10-15
    'min_samples': 2,         # 2D: 3-5, 3D: 2
    'alpha': 1.0,             # 較低的 alpha 讓聚類更緊密
}
```

#### 方案 B: CLIP + Face Weighting (混合方案)

```python
# 結合 CLIP 視覺特徵和 ArcFace 身份特徵
clip_weight = 0.3
face_weight = 0.7

combined_features = (
    clip_weight * clip_embeddings_norm +
    face_weight * face_embeddings_norm
)
```

**執行**:
```bash
python scripts/generic/clustering/face_identity_clustering.py \
  --input-dir /path/to/instances \
  --output-dir /path/to/character_clusters \
  --model arcface \
  --min-cluster-size 12 \
  --min-samples 2 \
  --use-face-detection
```

**輸出結構**:
```
character_clusters/
├── character_0/           # Luca
│   ├── frame_0001_instance_0.png
│   ├── frame_0055_instance_1.png
│   └── ... (200-500 images)
├── character_1/           # Alberto
├── character_2/           # Giulia
├── noise/                 # 未分類/低置信度
└── cluster_report.json    # 聚類統計
```

---

### 步驟 3: Pose Diversity Subclustering (可選但推薦)

**目的**: 在同一角色內，按姿態/視角進行二次聚類，確保訓練數據多樣性平衡

**為什麼需要？**
- 防止某種姿態/視角過度代表
- 提高 LoRA 泛化能力
- 改善 caption 一致性

**技術實作**:

```python
from mmpose.apis import init_model, inference_topdown
import umap
import hdbscan

# 1. 姿態估計
pose_model = init_model(
    'rtmpose-m_8xb256-420e_coco-256x192.py',
    'rtmpose-m.pth',
    device='cuda'
)

pose_features = []
for img_path in character_images:
    img = cv2.imread(str(img_path))
    results = inference_topdown(pose_model, img)

    if len(results) > 0:
        keypoints = results[0].pred_instances.keypoints[0]  # (17, 2)

        # 2. 歸一化姿態特徵 (移除位置/尺度)
        center = keypoints.mean(axis=0)
        centered = keypoints - center
        scale = np.linalg.norm(centered, axis=1).max()
        normalized = centered / (scale + 1e-8)

        pose_features.append(normalized.flatten())

pose_features = np.array(pose_features)

# 3. UMAP 降維 + HDBSCAN 聚類
reducer = umap.UMAP(n_components=10, metric='euclidean')
pose_embedding = reducer.fit_transform(pose_features)

pose_clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
pose_labels = pose_clusterer.fit_predict(pose_embedding)

# 4. 從每個姿態聚類中均勻採樣
balanced_samples = []
for cluster_id in set(pose_labels):
    if cluster_id == -1:  # noise
        continue
    cluster_indices = np.where(pose_labels == cluster_id)[0]
    # 從該姿態類別採樣 N 張
    sampled = np.random.choice(cluster_indices, min(50, len(cluster_indices)), replace=False)
    balanced_samples.extend(sampled)
```

**執行**:
```bash
python scripts/generic/clustering/pose_subclustering.py \
  --input-dir character_clusters/character_0 \
  --output-dir character_0_pose_balanced \
  --pose-model rtmpose-m \
  --samples-per-pose 50 \
  --device cuda
```

---

### 步驟 4: Quality Filtering

**目的**: 移除低質量圖片，確保訓練數據乾淨

**過濾標準**:

1. **模糊檢測** (Blur Detection)
```python
import cv2

def is_blurry(image, threshold=100):
    """Laplacian variance blur detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# 3D 動畫: threshold = 80-100 (允許輕微景深模糊)
# 2D 動畫: threshold = 120-150 (更嚴格)
```

2. **遮擋檢測** (Occlusion Detection)
```python
def detect_occlusion(instance_mask):
    """檢測角色是否被嚴重遮擋"""
    # 計算 mask 的完整性
    bbox = get_bounding_box(instance_mask)
    bbox_area = bbox.width * bbox.height
    mask_area = np.sum(instance_mask > 0)

    completeness = mask_area / bbox_area
    return completeness < 0.5  # 50% 以下視為嚴重遮擋
```

3. **尺寸過濾**
```python
def is_too_small(instance_mask, min_size=128):
    """角色太小可能細節不足"""
    h, w = np.where(instance_mask > 0)
    height = h.max() - h.min()
    width = w.max() - w.min()
    return min(height, width) < min_size
```

4. **人臉可見性** (Face Visibility)
```python
def has_visible_face(image, face_detector):
    """確保臉部可見（對 character LoRA 重要）"""
    faces = face_detector.detect(image)
    if len(faces) == 0:
        return False

    # 檢查臉部大小
    face_area = (faces[0].bbox[2] - faces[0].bbox[0]) * \
                (faces[0].bbox[3] - faces[0].bbox[1])
    img_area = image.shape[0] * image.shape[1]

    return face_area / img_area > 0.05  # 臉部至少佔 5%
```

**執行**:
```bash
python scripts/generic/preprocessing/quality_filter.py \
  --input-dir character_clusters/character_0 \
  --output-dir character_0_filtered \
  --blur-threshold 80 \
  --min-size 128 \
  --require-face \
  --min-face-ratio 0.05
```

---

### 步驟 5: Background Handling

**關鍵問題**: Character LoRA 應該使用什麼背景？

**選項比較**:

| 背景類型 | 優點 | 缺點 | 推薦場景 |
|---------|------|------|---------|
| **透明背景** | 最純粹、無污染 | 可能導致邊緣問題 | **推薦** - SD1.5/SDXL |
| **純色背景** | 簡單、安全 | 可能影響光照學習 | 快速實驗 |
| **模糊背景** | 保留上下文 | 可能有場景泄漏 | 特定風格 |
| **隨機背景** | 強泛化性 | 增加訓練難度 | 高級用途 |

**最佳實踐 (3D 角色)**:

```python
def prepare_character_image(instance_rgba, background_mode='transparent'):
    """
    準備 character 訓練圖片

    Args:
        instance_rgba: RGBA instance image
        background_mode: 'transparent', 'white', 'black', 'blur', 'random'
    """
    if background_mode == 'transparent':
        # 保持 RGBA，訓練時 SD 會處理
        return instance_rgba

    elif background_mode == 'white':
        # 白色背景（安全選擇）
        rgb = instance_rgba[:, :, :3]
        alpha = instance_rgba[:, :, 3:] / 255.0
        white_bg = np.ones_like(rgb) * 255
        return (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

    elif background_mode == 'random':
        # 隨機純色背景（增強泛化）
        rgb = instance_rgba[:, :, :3]
        alpha = instance_rgba[:, :, 3:] / 255.0
        random_color = np.random.randint(0, 256, 3)
        random_bg = np.full_like(rgb, random_color)
        return (rgb * alpha + random_bg * (1 - alpha)).astype(np.uint8)
```

**推薦**:
- SD1.5: 透明背景 (PNG with alpha)
- SDXL: 透明背景或白色背景
- 3D 動畫: 透明背景 + 少量白色背景混合 (9:1)

---

### 步驟 6: Caption Generation

**目的**: 為每張圖片生成描述性 caption

**Caption 結構** (Character LoRA):

```
{trigger_word}, {character_description}, {pose_description}, {style_tags}

範例:
"luca, a 3d animated boy, brown curly hair, green eyes, blue striped shirt,
standing pose, neutral expression, pixar style, smooth shading, high quality"
```

**方法比較**:

#### 方法 A: Template-based (快速、一致)

```python
def generate_character_caption_template(
    character_name: str,
    character_info: dict,
    pose_type: str = "neutral"
):
    """基於模板生成 caption"""
    parts = []

    # 1. Trigger word
    parts.append(character_name)

    # 2. Character type
    parts.append(f"a 3d animated {character_info['age_group']}")

    # 3. Physical features
    if character_info.get('hair'):
        parts.append(character_info['hair'])
    if character_info.get('eyes'):
        parts.append(character_info['eyes'])

    # 4. Clothing (top 2)
    parts.extend(character_info.get('clothing', [])[:2])

    # 5. Pose
    parts.append(f"{pose_type} pose")

    # 6. Style tags
    parts.extend([
        "pixar style",
        "3d animation",
        "smooth shading",
        "detailed character",
        "high quality"
    ])

    return ", ".join(parts)
```

#### 方法 B: VLM-based (靈活、詳細)

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class VLMCaptionGenerator:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def generate_caption(self, image, character_name):
        """使用 Qwen2-VL 生成 caption"""
        prompt = f"""Describe this 3D animated character named {character_name}.

Focus on:
1. Physical appearance (hair, eyes, skin tone)
2. Clothing and accessories
3. Current pose and expression
4. Art style (mention "pixar style", "3d animation", "smooth shading")

Format: Comma-separated tags, starting with the character name.
Keep it concise (60-80 tokens)."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda")

        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        caption = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return caption
```

#### 方法 C: Hybrid (推薦)

```python
def generate_caption_hybrid(image, character_name, character_info):
    """混合方法：模板作為基礎，VLM 補充細節"""

    # 1. Template baseline
    base_caption = generate_character_caption_template(
        character_name, character_info
    )

    # 2. VLM 補充動態信息（pose, expression）
    vlm_prompt = f"""Given the base description: "{base_caption}"

    Add specific details about:
    - Current pose (e.g., "standing", "walking", "sitting")
    - Facial expression (e.g., "smiling", "neutral", "surprised")
    - Any unique details in this specific image

    Keep the original tags and just append new specific details."""

    vlm_additions = vlm_generator.generate(image, vlm_prompt)

    # 3. Combine
    final_caption = base_caption + ", " + vlm_additions

    return final_caption
```

**執行**:
```bash
# Template-based (快速)
python scripts/generic/training/generate_captions.py \
  --input-dir character_0_filtered \
  --output-dir character_0_captioned \
  --method template \
  --character-name luca \
  --character-info data/films/luca/characters/luca.yaml

# VLM-based (高質量)
python scripts/generic/training/qwen_caption_generator_robust.py \
  --input-dir character_0_filtered \
  --output-dir character_0_captioned \
  --model qwen2_vl \
  --character-name luca \
  --batch-size 4
```

---

### 步驟 7: Dataset Assembly

**最終訓練數據結構** (Kohya_ss 格式):

```
training_data/luca_character/
├── images/
│   ├── luca_001.png
│   ├── luca_002.png
│   └── ... (200-500 images)
│
├── captions/  (可選，如果使用 txt 文件)
│   ├── luca_001.txt
│   ├── luca_002.txt
│   └── ...
│
└── metadata.json
    {
      "character_name": "luca",
      "total_images": 350,
      "source_film": "Luca (2021)",
      "resolution": "512x512",
      "background_type": "transparent",
      "caption_method": "hybrid",
      "quality_filters_applied": [
        "blur_detection",
        "occlusion_filter",
        "face_visibility"
      ]
    }
```

**或使用 Kohya metadata 格式** (推薦):

```json
// metadata.json
{
  "images": [
    {
      "file": "luca_001.png",
      "caption": "luca, a 3d animated boy, brown curly hair, green eyes, standing pose, pixar style",
      "width": 512,
      "height": 512,
      "tags": ["luca", "boy", "brown_hair", "green_eyes", "standing"]
    },
    ...
  ]
}
```

**數據增強** (可選):

```python
# Character LoRA 推薦的增強
augmentation_config = {
    'horizontal_flip': False,     # ❌ 3D 角色常有不對稱特徵
    'vertical_flip': False,        # ❌ 從不使用
    'rotation': False,             # ❌ 角色方向很重要
    'color_jitter': False,         # ❌ 破壞 PBR 材質
    'random_crop': True,           # ✅ 輕微裁剪增加多樣性
    'crop_ratio': (0.9, 1.0),      # 只裁剪 0-10%
    'brightness': (0.95, 1.05),    # ✅ 輕微亮度調整
    'contrast': (0.95, 1.05),      # ✅ 輕微對比度調整
}
```

---

## 🎓 訓練策略

### 超參數設定 (SDXL, 3D Character)

```toml
# configs/training/luca_character_sdxl.toml

[model]
pretrained_model_name_or_path = "/path/to/sd_xl_base_1.0.safetensors"
vae = "/path/to/sdxl_vae.safetensors"  # 可選

[dataset]
resolution = 1024  # SDXL native
batch_size = 4
num_repeats = 1

[network]
network_module = "networks.lora"
network_dim = 64           # Rank (3D 推薦 32-64)
network_alpha = 32         # Alpha (通常 = rank/2)
network_train_unet_only = true
network_train_text_encoder_only = false

[training]
learning_rate = 0.0003     # 3D 推薦 2e-4 到 4e-4
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.01"]

max_train_epochs = 10      # 3D: 8-12 epochs
save_every_n_epochs = 2    # 每 2 epoch 保存

mixed_precision = "fp16"   # 或 "bf16"
gradient_checkpointing = true
gradient_accumulation_steps = 1

[noise]
noise_offset = 0.05        # 輕微 noise offset 提升對比度
adaptive_noise_scale = 0.00357  # SDXL 推薦值

[advanced]
clip_skip = 2              # SDXL 通常用 2
min_snr_gamma = 5.0        # Min-SNR weighting
scale_v_pred_loss_like_noise_pred = true  # SDXL v-prediction

[output]
output_name = "luca_character"
output_dir = "/path/to/output"
logging_dir = "/path/to/logs"
```

### Learning Rate Scheduling

**推薦**: Cosine with Restarts

```python
# 為什麼？
# - 前期快速學習 (warm-up)
# - 中期穩定優化
# - 後期精細調整 (cosine decay)
# - Restarts 避免局部最優

lr_schedule = {
    'scheduler': 'cosine_with_restarts',
    'lr_warmup_steps': 100,          # 5% of total steps
    'lr_scheduler_num_cycles': 3,    # 3 個 cosine cycles
}
```

**替代方案**:
- `constant`: 固定 LR（簡單但不推薦）
- `cosine`: 單一 cosine decay（穩定但可能欠擬合）
- `polynomial`: 多項式 decay（通用選擇）

### Rank (network_dim) 選擇

**經驗法則**:

| 數據量 | Rank | 原因 |
|--------|------|------|
| < 200 | 16-24 | 小 rank 防止過擬合 |
| 200-400 | 32-48 | 平衡容量和泛化 |
| 400-600 | 48-64 | 充分學習複雜特徵 |
| > 600 | 64-96 | 大數據需要大容量 |

**3D 角色推薦**: 32-64 (即使數據較多也不需要過大)

### Batch Size vs Learning Rate

**關係**:
```
effective_lr = base_lr * sqrt(batch_size)

範例:
batch_size=1, lr=0.0001  →  effective_lr = 0.0001
batch_size=4, lr=0.0002  →  effective_lr = 0.0004
batch_size=8, lr=0.0003  →  effective_lr = 0.00085
```

**推薦組合** (SDXL, 24GB VRAM):
- `batch_size=2, lr=0.00015` (保守)
- `batch_size=4, lr=0.0003` (推薦)
- `batch_size=8, lr=0.0004` (激進，需要 >32GB)

---

## 🔍 訓練監控

### 關鍵指標

1. **Loss Curve**
```python
# 理想的 loss curve
# Epoch 1-2: 快速下降 (0.1 → 0.05)
# Epoch 3-6: 緩慢下降 (0.05 → 0.03)
# Epoch 7-10: 平穩 (0.03 → 0.025)
# Epoch 10+: 可能過擬合 (持續下降但測試效果變差)
```

2. **Checkpoint 測試**
```bash
# 每 2 epoch 測試一次
for epoch in 2 4 6 8 10; do
    python scripts/evaluation/evaluate_single_checkpoint.py \
        --lora checkpoints/luca_character_epoch${epoch}.safetensors \
        --prompts test_prompts.json \
        --output-dir test_results/epoch${epoch}
done
```

3. **過擬合檢測**
```python
# Signs of overfitting:
# - Loss 持續下降但生成質量變差
# - 只能生成訓練集中的姿態
# - 對 prompt 變化不敏感
# - 出現訓練數據的 artifacts

# 解決方法:
# - 減少 epochs (10 → 8)
# - 降低 network_dim (64 → 48)
# - 增加 noise_offset (0.05 → 0.1)
# - 添加更多數據
```

---

## 🧪 測試和評估

### Test Prompts 設計

```json
{
  "identity_preservation": [
    "luca, standing pose, neutral expression, white background, pixar style",
    "luca, close-up portrait, smiling, pixar 3d animation",
    "luca, full body, casual pose, simple background"
  ],
  "pose_variety": [
    "luca, walking pose, dynamic movement, pixar style",
    "luca, sitting on chair, relaxed pose, 3d animation",
    "luca, jumping in air, excited expression, pixar rendering"
  ],
  "novel_scenes": [
    "luca in a forest, surrounded by trees, pixar style",
    "luca at night, under starry sky, moonlight, 3d animation",
    "luca in winter, snow background, wearing coat, pixar"
  ],
  "extreme_tests": [
    "luca as a superhero, action pose, dramatic lighting",
    "luca in cyberpunk city, neon lights, futuristic style",
    "luca portrait, oil painting style, classical art"
  ]
}
```

### 評估維度

| 維度 | 評估方法 | 目標 |
|------|---------|------|
| **Identity** | 人工檢查 + Face similarity | >90% 識別率 |
| **Consistency** | 多次生成同一 prompt | 高度一致 |
| **Pose Control** | 不同姿態 prompts | 響應 prompt 變化 |
| **Style Transfer** | 不同風格 prompts | 保持身份特徵 |
| **Novel Scenes** | 未見過的場景 | 泛化能力 |

### 自動化評估

```python
from deepface import DeepFace
import numpy as np

def evaluate_identity_consistency(
    reference_image: str,
    generated_images: List[str]
) -> float:
    """評估身份一致性"""
    similarities = []

    for gen_img in generated_images:
        result = DeepFace.verify(
            reference_image,
            gen_img,
            model_name='ArcFace',
            enforce_detection=False
        )
        similarities.append(result['distance'])

    # ArcFace distance: 越小越相似
    avg_similarity = 1.0 - np.mean(similarities)
    return avg_similarity

# 使用
ref_img = "data/films/luca/reference/luca_front.jpg"
gen_imgs = list(Path("test_results/epoch8").glob("*.png"))
score = evaluate_identity_consistency(ref_img, gen_imgs)
print(f"Identity Score: {score:.2%}")  # 目標: >85%
```

---

## ⚠️ 常見問題和解決方案

### 問題 1: LoRA 無效果

**症狀**: 生成的圖片看不出角色特徵

**可能原因**:
1. LoRA 權重過低
2. Trigger word 未使用
3. 訓練數據不足
4. Network dim 過小

**解決方案**:
```python
# 1. 提高 LoRA 權重
pipe.set_adapters(["character"], adapter_weights=[1.2])  # 試試 >1.0

# 2. Prompt 中明確使用 trigger word
prompt = "luca, ..."  # 確保 "luca" 在最前面

# 3. 檢查訓練數據量
# 最少 150 張，推薦 250-400 張

# 4. 增加 network_dim
network_dim = 64  # 從 32 增加到 64
```

### 問題 2: 過度擬合特定姿態

**症狀**: 只能生成訓練集中的姿態

**解決方案**:
1. 使用 Pose Subclustering 平衡姿態分布
2. 減少訓練 epochs
3. 增加數據多樣性

### 問題 3: 背景泄漏

**症狀**: 總是生成訓練集中的背景

**解決方案**:
```bash
# 重新準備數據，使用透明背景
python scripts/generic/training/prepare_training_data.py \
  --background-mode transparent \
  ...

# 或在 prompt 中明確指定背景
prompt = "luca, ..., simple white background"
```

### 問題 4: 服裝固定

**症狀**: 無法改變角色服裝

**解決方案**:
1. 訓練數據包含不同服裝
2. Caption 中分離身份和服裝
```
"luca boy with brown hair and green eyes, wearing blue striped shirt, ..."
      ↑ 身份特徵                           ↑ 可變服裝
```

---

## 📈 最佳實踐總結

### ✅ Do's

1. ✅ 使用 Face-based clustering (ArcFace + HDBSCAN)
2. ✅ Pose subclustering 確保多樣性
3. ✅ 透明背景或純色背景
4. ✅ 質量過濾 (blur, occlusion, face visibility)
5. ✅ 混合 caption 方法 (template + VLM)
6. ✅ 定期測試 checkpoints
7. ✅ 監控 loss curve
8. ✅ 3D 特定參數 (較小 min_cluster_size, 較少數據)

### ❌ Don'ts

1. ❌ 使用純 CLIP clustering (會混淆角色)
2. ❌ 數據增強 (flip, color jitter) 對 3D 角色
3. ❌ 過大的 network_dim (>96)
4. ❌ 過長訓練 (>15 epochs)
5. ❌ 忽略質量過濾
6. ❌ 訓練數據包含原始背景
7. ❌ Caption 中混淆角色和場景描述

---

## 🔗 相關資源

**工具腳本**:
- `scripts/generic/segmentation/instance_segmentation.py`
- `scripts/generic/clustering/face_identity_clustering.py`
- `scripts/generic/clustering/pose_subclustering.py`
- `scripts/generic/training/prepare_training_data.py`

**配置範例**:
- `configs/training/luca_character_sdxl.toml`
- `configs/characters/luca.yaml`

**測試工具**:
- `scripts/evaluation/test_lora_checkpoints.py`
- `scripts/evaluation/evaluate_single_checkpoint.py`

---

**最後更新**: 2025-11-16
**作者**: LLMProvider Tooling
**適用版本**: Kohya_ss sd-scripts, SDXL/SD1.5
