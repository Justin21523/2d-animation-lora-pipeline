# 多類型 LoRA 訓練系統 - 模組化架構設計

**版本**: 2.0
**設計理念**: 模組化、可擴展、多演算法選擇
**狀態**: 架構規劃

---

## 🎯 設計目標

1. **模組化**：每個功能獨立成模組，易於維護和擴展
2. **可選擇**：提供多種演算法/模型選擇，適應不同場景
3. **SOTA 整合**：使用最新的 AI 模型和技術
4. **配置驅動**：通過配置文件選擇算法和參數
5. **可測試**：每個模組可獨立測試

---

## 📁 新的目錄結構

```
scripts/generic/training/
├── lora_data_preparers/           # LoRA 數據準備器 (主入口)
│   ├── __init__.py
│   ├── background_preparer.py     # Background LoRA 主控制器
│   ├── pose_preparer.py           # Pose LoRA 主控制器
│   ├── expression_preparer.py     # Expression LoRA 主控制器
│   ├── style_preparer.py          # Style LoRA 主控制器
│   └── lighting_preparer.py       # Lighting LoRA 主控制器
│
├── processors/                    # 核心處理器模組
│   ├── __init__.py
│   ├── background/                # 背景處理
│   │   ├── __init__.py
│   │   ├── scene_classifier.py   # 場景分類器
│   │   ├── scene_clusterer.py    # 場景聚類器
│   │   └── quality_filter.py     # 品質過濾
│   │
│   ├── pose/                      # 姿態處理
│   │   ├── __init__.py
│   │   ├── pose_estimator.py     # 姿態估計 (RTM-Pose, MediaPipe)
│   │   ├── action_classifier.py  # 動作分類
│   │   └── pose_clusterer.py     # 姿態聚類
│   │
│   ├── expression/                # 表情處理
│   │   ├── __init__.py
│   │   ├── face_detector.py      # 人臉檢測
│   │   ├── expression_classifier.py  # 表情分類
│   │   └── face_cropper.py       # 人臉裁剪
│   │
│   ├── style/                     # 風格處理
│   │   ├── __init__.py
│   │   ├── style_analyzer.py     # 風格分析
│   │   ├── transition_filter.py  # 轉場過濾
│   │   └── consistency_checker.py # 一致性檢查
│   │
│   └── common/                    # 共用處理器
│       ├── __init__.py
│       ├── image_loader.py       # 圖片載入
│       ├── feature_extractor.py  # 特徵提取
│       └── quality_assessor.py   # 品質評估
│
├── models/                        # AI 模型後端
│   ├── __init__.py
│   ├── segmentation/              # 分割模型
│   │   ├── sam2_backend.py       # SAM2
│   │   ├── u2net_backend.py      # U²-Net
│   │   └── isnet_backend.py      # ISNet
│   │
│   ├── pose_estimation/           # 姿態估計
│   │   ├── rtmpose_backend.py    # RTM-Pose (SOTA)
│   │   ├── mediapipe_backend.py  # MediaPipe
│   │   ├── openpose_backend.py   # OpenPose
│   │   └── vitpose_backend.py    # ViTPose++
│   │
│   ├── face_detection/            # 人臉檢測
│   │   ├── retinaface_backend.py # RetinaFace
│   │   ├── yolov8face_backend.py # YOLOv8-Face
│   │   ├── scrfd_backend.py      # SCRFD
│   │   └── mediapipe_face.py     # MediaPipe Face
│   │
│   ├── expression/                # 表情識別
│   │   ├── fer_plus_backend.py   # FER+
│   │   ├── emotion_vit_backend.py # Vision Transformer
│   │   └── affectnet_backend.py  # AffectNet 預訓練
│   │
│   ├── scene_understanding/       # 場景理解
│   │   ├── clip_backend.py       # CLIP
│   │   ├── siglip_backend.py     # SigLIP (更強的視覺編碼)
│   │   ├── eva_clip_backend.py   # EVA-CLIP (SOTA)
│   │   └── blip2_backend.py      # BLIP2
│   │
│   ├── inpainting/                # 修復模型
│   │   ├── lama_backend.py       # LaMa
│   │   ├── mat_backend.py        # MAT
│   │   ├── powerpaint_backend.py # PowerPaint (ECCV 2024)
│   │   └── propainter_backend.py # ProPainter (視頻修復)
│   │
│   └── caption_generation/        # Caption 生成
│       ├── qwen2_vl_backend.py   # Qwen2.5-VL (SOTA)
│       ├── internvl2_backend.py  # InternVL2
│       ├── llava_backend.py      # LLaVA-NeXT
│       └── blip2_backend.py      # BLIP2
│
├── clustering/                    # 聚類演算法模組
│   ├── __init__.py
│   ├── hdbscan_clusterer.py      # HDBSCAN (當前主力)
│   ├── kmeans_clusterer.py       # KMeans
│   ├── spectral_clusterer.py     # Spectral Clustering
│   ├── agglomerative_clusterer.py # 層次聚類
│   └── dbscan_clusterer.py       # DBSCAN
│
├── feature_extractors/            # 特徵提取器
│   ├── __init__.py
│   ├── clip_features.py          # CLIP embeddings
│   ├── dinov2_features.py        # DINOv2 (自監督學習 SOTA)
│   ├── mae_features.py           # Masked Autoencoder
│   ├── color_histogram.py        # 顏色直方圖
│   ├── spatial_features.py       # 空間特徵
│   └── pose_features.py          # 姿態特徵
│
├── caption_engines/               # Caption 生成引擎
│   ├── __init__.py
│   ├── template_based.py         # 模板式 caption
│   ├── vlm_based.py              # VLM 自動生成
│   ├── hybrid_caption.py         # 混合式 (模板 + VLM)
│   └── schema_guided.py          # Schema 引導式
│
└── utils/                         # 工具函數
    ├── __init__.py
    ├── config_loader.py          # 配置載入
    ├── dataset_builder.py        # 數據集建構
    ├── augmentation.py           # 數據增強
    └── visualization.py          # 可視化工具
```

---

## 🔧 模組化設計範例

### 1. Background LoRA Preparer (主控制器)

```python
# scripts/generic/training/lora_data_preparers/background_preparer.py

from pathlib import Path
from typing import Optional, Dict, Any

# Import processors
from processors.background.scene_clusterer import SceneClusterer
from processors.background.quality_filter import BackgroundQualityFilter
from processors.common.feature_extractor import FeatureExtractor

# Import models
from models.scene_understanding.eva_clip_backend import EVACLIPBackend
from models.inpainting.lama_backend import LaMaBackend

# Import clustering
from clustering.hdbscan_clusterer import HDBSCANClusterer

# Import caption engines
from caption_engines.vlm_based import VLMCaptionEngine


class BackgroundLoRAPreparer:
    """
    Background LoRA 數據準備主控制器

    支持配置不同的：
    - 特徵提取模型 (CLIP, SigLIP, EVA-CLIP)
    - 聚類演算法 (HDBSCAN, KMeans, Spectral)
    - Caption 生成方法 (Template, VLM, Hybrid)
    """

    def __init__(
        self,
        sam2_backgrounds_dir: Path,
        lama_backgrounds_dir: Optional[Path] = None,
        output_dir: Path = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or self._default_config()

        # Initialize feature extractor based on config
        self.feature_extractor = self._init_feature_extractor()

        # Initialize clusterer based on config
        self.clusterer = self._init_clusterer()

        # Initialize caption engine based on config
        self.caption_engine = self._init_caption_engine()

        # Initialize quality filter
        self.quality_filter = BackgroundQualityFilter(
            min_resolution=self.config['quality']['min_resolution'],
            blur_threshold=self.config['quality']['blur_threshold']
        )

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'feature_extractor': {
                'model': 'eva_clip',  # Options: clip, siglip, eva_clip, dinov2
                'model_size': 'large',
                'device': 'cuda'
            },
            'clusterer': {
                'algorithm': 'hdbscan',  # Options: hdbscan, kmeans, spectral
                'min_cluster_size': 15,
                'min_samples': 2,
                'similarity_threshold': 0.75
            },
            'caption_engine': {
                'method': 'vlm',  # Options: template, vlm, hybrid
                'model': 'qwen2_vl',  # For VLM: qwen2_vl, internvl2, llava
                'device': 'cuda'
            },
            'quality': {
                'min_resolution': (512, 512),
                'blur_threshold': 100
            }
        }

    def _init_feature_extractor(self):
        """Initialize feature extractor based on config"""
        model_name = self.config['feature_extractor']['model']

        if model_name == 'eva_clip':
            from models.scene_understanding.eva_clip_backend import EVACLIPBackend
            return EVACLIPBackend(
                model_size=self.config['feature_extractor']['model_size'],
                device=self.config['feature_extractor']['device']
            )
        elif model_name == 'siglip':
            from models.scene_understanding.siglip_backend import SigLIPBackend
            return SigLIPBackend(...)
        elif model_name == 'clip':
            from models.scene_understanding.clip_backend import CLIPBackend
            return CLIPBackend(...)
        else:
            raise ValueError(f"Unknown feature extractor: {model_name}")

    def _init_clusterer(self):
        """Initialize clusterer based on config"""
        algorithm = self.config['clusterer']['algorithm']

        if algorithm == 'hdbscan':
            from clustering.hdbscan_clusterer import HDBSCANClusterer
            return HDBSCANClusterer(
                min_cluster_size=self.config['clusterer']['min_cluster_size'],
                min_samples=self.config['clusterer']['min_samples']
            )
        elif algorithm == 'kmeans':
            from clustering.kmeans_clusterer import KMeansClusterer
            return KMeansClusterer(...)
        # ... more options

    def _init_caption_engine(self):
        """Initialize caption engine based on config"""
        method = self.config['caption_engine']['method']

        if method == 'vlm':
            from caption_engines.vlm_based import VLMCaptionEngine
            return VLMCaptionEngine(
                model_name=self.config['caption_engine']['model'],
                device=self.config['caption_engine']['device']
            )
        elif method == 'template':
            from caption_engines.template_based import TemplateCaptionEngine
            return TemplateCaptionEngine()
        # ... more options

    def prepare(self):
        """Execute full preparation pipeline"""
        # 1. Scan backgrounds
        backgrounds = self.scan_backgrounds()

        # 2. Quality filtering
        filtered = self.quality_filter.filter(backgrounds)

        # 3. Feature extraction
        features = self.feature_extractor.extract_batch(filtered)

        # 4. Clustering
        clusters = self.clusterer.fit_predict(features)

        # 5. Caption generation
        captions = self.caption_engine.generate_batch(filtered, clusters)

        # 6. Save training dataset
        self.save_dataset(filtered, captions, clusters)
```

---

## 🎯 SOTA 模型整合建議

### Pose Estimation (姿態估計)

| 模型 | 年份 | 優點 | 推薦場景 |
|------|------|------|---------|
| **RTM-Pose** | 2023 | SOTA、實時、高精度 | **推薦作為主力** |
| **ViTPose++** | 2023 | Transformer-based、高精度 | 複雜姿態 |
| **MediaPipe Pose** | 2022 | 輕量、快速 | 預覽/快速過濾 |
| **DWPose (Controlnet)** | 2023 | 專為 SD 優化 | ControlNet 整合 |

**實作範例**:
```python
# scripts/generic/training/models/pose_estimation/rtmpose_backend.py

from mmpose.apis import inference_topdown, init_model
import numpy as np

class RTMPoseBackend:
    """RTM-Pose backend (SOTA pose estimation)"""

    def __init__(
        self,
        model_config: str = 'rtmpose-m_8xb256-420e_coco-256x192',
        checkpoint: str = None,
        device: str = 'cuda'
    ):
        self.model = init_model(
            model_config,
            checkpoint,
            device=device
        )

    def estimate_pose(self, image):
        """Estimate 2D pose keypoints"""
        results = inference_topdown(self.model, image)
        keypoints = results[0].pred_instances.keypoints[0]  # (17, 2)
        scores = results[0].pred_instances.keypoint_scores[0]  # (17,)
        return keypoints, scores

    def extract_pose_features(self, keypoints):
        """Extract normalized pose features for clustering"""
        # Normalize keypoints (remove position/scale)
        center = keypoints.mean(axis=0)
        centered = keypoints - center
        scale = np.linalg.norm(centered, axis=1).max()
        normalized = centered / (scale + 1e-8)
        return normalized.flatten()
```

### Scene Understanding (場景理解)

| 模型 | 優點 | 推薦用途 |
|------|------|---------|
| **EVA-CLIP** | SOTA、5B 參數 | **推薦作為主力** |
| **SigLIP** | Google 最新、性能強 | 場景聚類 |
| **CLIP ViT-L/14** | 成熟穩定 | 基準對比 |
| **DINOv2** | 自監督學習 SOTA | 無標籤場景特徵 |

**實作範例**:
```python
# scripts/generic/training/models/scene_understanding/eva_clip_backend.py

import torch
from transformers import AutoModel, AutoProcessor

class EVACLIPBackend:
    """EVA-CLIP backend (SOTA vision-language model)"""

    def __init__(
        self,
        model_size: str = 'large',  # 'base', 'large', 'giant'
        device: str = 'cuda'
    ):
        model_name = f"BAAI/EVA-CLIP-{model_size}"
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def extract_image_features(self, images):
        """Extract image features"""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy()
```

### Expression Recognition (表情識別)

| 模型 | 優點 | 推薦用途 |
|------|------|---------|
| **Vision Transformer (ViT) for Emotion** | Transformer-based | **推薦** |
| **FER+ / AffectNet 預訓練** | 大規模標註數據 | 7種基本表情 |
| **Multi-modal Emotion Recognition** | 結合音訊和視覺 | 多模態分析 |

### Caption Generation (Caption 生成)

| 模型 | 參數量 | 優點 | 推薦用途 |
|------|---------|------|---------|
| **Qwen2.5-VL** | 7B-72B | SOTA、支持長視頻 | **推薦主力** |
| **InternVL2** | 1B-76B | 多解析度、高性能 | 備選方案 |
| **LLaVA-NeXT** | 7B-34B | 開源、社區活躍 | 研究用途 |

---

## 📝 配置文件範例

### background_lora_config.yaml

```yaml
# Background LoRA 訓練數據準備配置

input:
  sam2_backgrounds: /path/to/sam2/backgrounds
  lama_backgrounds: /path/to/lama/cleaned

output:
  dir: /path/to/training_data/portorosso_background
  scene_name: portorosso

# Feature Extraction
feature_extractor:
  model: eva_clip  # Options: clip, siglip, eva_clip, dinov2
  model_size: large
  device: cuda
  batch_size: 16

# Clustering
clustering:
  algorithm: hdbscan  # Options: hdbscan, kmeans, spectral, agglomerative
  min_cluster_size: 15
  min_samples: 2
  similarity_threshold: 0.75
  max_clusters: 10

# Quality Filtering
quality:
  enable: true
  min_resolution: [512, 512]
  blur_threshold: 100
  remove_duplicates: true
  dedup_threshold: 0.95

# Caption Generation
caption:
  method: vlm  # Options: template, vlm, hybrid
  model: qwen2_vl  # For VLM: qwen2_vl, internvl2, llava
  device: cuda
  batch_size: 4
  schema_guided: true
  schema:
    - scene_type
    - location
    - weather
    - time_of_day
    - colors
    - atmosphere

# Dataset
dataset:
  target_size: 300
  format: kohya_ss
  image_format: jpg
  caption_format: txt
```

### pose_lora_config.yaml

```yaml
# Pose LoRA 訓練數據準備配置

input:
  character_instances: /path/to/instances

output:
  dir: /path/to/training_data/running_pose
  action_name: running

# Pose Estimation
pose_estimation:
  model: rtmpose  # Options: rtmpose, vitpose, mediapipe, dwpose
  model_size: m   # s, m, l, x
  device: cuda
  confidence_threshold: 0.3

# Action Classification
action_classification:
  method: rule_based  # Options: rule_based, ml_classifier, vlm
  action_keywords:
    - run
    - running
    - sprint
    - motion

# Pose Clustering (optional)
pose_clustering:
  enable: true
  algorithm: kmeans  # Pose features work well with kmeans
  n_clusters: 5
  normalize_keypoints: true

# Caption Generation
caption:
  method: hybrid  # Combine template + VLM for pose details
  include_pose_details: true
  character_name: luca

# Dataset
dataset:
  target_size: 200
  keep_transparency: true
  image_format: png
```

---

## 🚀 使用範例

### 1. Background LoRA (配置驅動)

```bash
# 使用配置文件執行
python scripts/generic/training/prepare_lora_data.py \
  --config configs/lora/background_lora_config.yaml \
  --type background

# 或使用命令列參數覆蓋配置
python scripts/generic/training/prepare_lora_data.py \
  --config configs/lora/background_lora_config.yaml \
  --type background \
  --feature-extractor siglip \
  --clustering-algorithm spectral \
  --caption-model internvl2
```

### 2. Pose LoRA (模組化選擇)

```bash
# 使用 RTM-Pose + HDBSCAN + VLM caption
python scripts/generic/training/prepare_lora_data.py \
  --config configs/lora/pose_lora_config.yaml \
  --type pose \
  --pose-model rtmpose \
  --clustering hdbscan \
  --caption-method vlm

# 使用 MediaPipe + KMeans + Template caption (更快)
python scripts/generic/training/prepare_lora_data.py \
  --config configs/lora/pose_lora_config.yaml \
  --type pose \
  --pose-model mediapipe \
  --clustering kmeans \
  --caption-method template
```

---

## 🔬 模組化優勢

### 1. 易於實驗和比較

```bash
# 測試不同特徵提取器的聚類效果
for model in clip siglip eva_clip dinov2; do
    python prepare_lora_data.py \
        --type background \
        --feature-extractor $model \
        --output-suffix _${model}
done

# 比較聚類結果
python scripts/evaluation/compare_clustering_results.py \
    outputs/background_lora_clip/ \
    outputs/background_lora_siglip/ \
    outputs/background_lora_eva_clip/ \
    outputs/background_lora_dinov2/
```

### 2. 靈活組合不同模組

```python
# 自定義 pipeline 組合
preparer = BackgroundLoRAPreparer(
    feature_extractor=EVACLIPBackend(model_size='giant'),
    clusterer=SpectralClusterer(n_clusters=8),
    caption_engine=HybridCaptionEngine(
        template_engine=TemplateCaptionEngine(),
        vlm_engine=Qwen2VLEngine(model_size='72B')
    )
)
```

### 3. 易於添加新模型

```python
# 添加新的姿態估計後端只需實現標準接口
class NewPoseBackend(BasePoseEstimator):
    def estimate_pose(self, image):
        # Implement pose estimation
        return keypoints, scores

    def extract_pose_features(self, keypoints):
        # Implement feature extraction
        return features
```

---

## 📊 實作優先級

### Phase 1: 核心重構 (1-2 天)
- [ ] 創建模組化目錄結構
- [ ] 實作 Base 抽象類
- [ ] 重構現有 Background 和 Pose preparer
- [ ] 添加配置文件支持

### Phase 2: SOTA 模型整合 (2-3 天)
- [ ] EVA-CLIP backend
- [ ] RTM-Pose backend
- [ ] Qwen2.5-VL caption engine
- [ ] DINOv2 feature extractor

### Phase 3: 完善其他 LoRA 類型 (3-4 天)
- [ ] Expression LoRA preparer
- [ ] Style LoRA preparer
- [ ] Lighting LoRA preparer (optional)

### Phase 4: 測試和文檔 (1-2 天)
- [ ] 單元測試
- [ ] 整合測試
- [ ] 使用文檔
- [ ] 最佳實踐指南

---

## ✅ 總結

**模組化架構優勢**:
1. ✅ **可維護性**: 每個模組職責單一，易於理解和修改
2. ✅ **可擴展性**: 新增模型/演算法只需實作標準接口
3. ✅ **可測試性**: 每個模組可獨立測試
4. ✅ **靈活性**: 配置驅動，無需修改代碼即可切換模型
5. ✅ **性能**: 支持不同模型以適應不同性能需求

**下一步行動**:
1. 等 SAM2 完成後，先用現有腳本準備 Background 和 Pose 數據
2. 並行進行模組化重構
3. 逐步遷移到新架構
4. 添加更多 SOTA 模型選項

---

**最後更新**: 2025-11-16
