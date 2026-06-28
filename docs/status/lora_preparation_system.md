# LoRA Data Preparation System - Complete Implementation

**Created**: 2025-11-19
**Status**: ✅ Operational
**Purpose**: CPU-based multi-type LoRA training data preparation

---

## 系統概述

已完整實作一個 **CPU 密集型** LoRA 訓練資料準備系統，可與 GPU SAM2 處理並行執行。

### 支援的 LoRA 類型

| LoRA 類型 | 目的 | 資料來源 | 處理時間/電影 |
|-----------|------|----------|--------------|
| **Background** | 場景/環境 | Inpainted backgrounds | ~30-45分鐘 |
| **Action/Pose** | 動作/姿態 | Character instances | ~2-3小時 |
| **Expression** | 面部表情 | Character instances | ~1-2小時 |
| **Style** | 視覺風格/渲染 | Original frames | ~20-30分鐘 |

---

## 已實作的組件

### 1. 核心處理腳本 (`scripts/generic/`)

#### Pose Estimation
**檔案**: `scripts/generic/pose/pose_estimation.py`
- RTM-Pose 骨架關鍵點提取
- CPU/GPU 支援
- JSON keypoints 輸出
- 可視化選項

#### Action Clustering
**檔案**: `scripts/generic/clustering/action_clustering.py`
- CLIP 視覺特徵 + HDBSCAN 聚類
- 無需姿態估計即可使用
- 自動分組動作類型
- 支援 visual 和 keypoints 模式

#### Expression Classification
**檔案**: `scripts/generic/face/expression_classification.py`
- 臉部檢測 + CLIP 聚類
- 表情分組
- 支援 VLM, landmarks, CLIP 模式
- 當前使用 CLIP (最穩定)

#### Style Frame Selector
**檔案**: `scripts/generic/quality/style_frame_selector.py`
- CLIP 風格一致性分析
- 轉場檢測與移除
- 模糊度/亮度過濾
- 多樣性採樣 (K-means)

#### Quality Assessment
**檔案**: `scripts/generic/quality/quality_assessment.py`
- 模糊檢測 (Laplacian variance)
- 解析度過濾
- 重複檢測 (pHash + SSIM)
- 亮度/對比度分析

### 2. 批次處理系統 (`scripts/batch/`)

#### 主腳本
**檔案**: `scripts/batch/batch_lora_data_preparation.sh`

**功能**:
- 彈性處理任意電影組合
- 支援選擇性 LoRA 類型
- 並行處理 (可設定並發數)
- 完整日誌記錄
- 自動錯誤處理

**用法**:
```bash
# 處理所有電影，所有 LoRA 類型
bash scripts/batch/batch_lora_data_preparation.sh all

# 處理特定電影
bash scripts/batch/batch_lora_data_preparation.sh luca onward

# 只處理特定 LoRA 類型
bash scripts/batch/batch_lora_data_preparation.sh all --lora-types background,style

# 使用 GPU (SAM2 完成後)
bash scripts/batch/batch_lora_data_preparation.sh all --device cuda
```

#### 監控腳本
**檔案**: `scripts/batch/monitor_lora_preparation.sh`

**功能**:
- 即時進度顯示
- 活動進程列表 (PID, CPU, MEM)
- GPU 狀態
- 各電影完成情況
- 最新日誌路徑

**用法**:
```bash
# 手動執行
bash scripts/batch/monitor_lora_preparation.sh

# 持續監控 (每10秒更新)
watch -n 10 'bash scripts/batch/monitor_lora_preparation.sh'
```

### 3. 文檔

- **使用指南**: `docs/guides/CPU_LORA_DATA_PREPARATION.md`
- **系統狀態**: `docs/status/lora_preparation_system.md` (本文件)

---

## 當前執行狀態 (2025-11-19 04:44)

### 批次處理

**主進程**: PID 108322
**啟動時間**: 04:44:14
**處理電影**: all (coco, elio, luca, onward, turning-red)
**LoRA 類型**: background, action, expression, style

### 活動任務

**Background Clustering** (Phase 1 - 進行中):
- coco: PID 108402 (CPU 176%)
- elio: PID 108454 (CPU 264%)
- onward: PID 108494 (CPU 24%)
- turning-red: 待啟動
- luca: 待啟動

### 資源使用

- **CPU**: 3個並行任務，總 CPU ~460%
- **GPU**: 100% (SAM2 處理 up 電影，無干擾)
- **記憶體**: 正常

---

## 輸出結構

處理完成後，每部電影會有以下結構：

```
/mnt/data/ai_data/datasets/3d-anime/{film}/lora_data/
├── background_clusters/           # Background LoRA 訓練資料
│   ├── character_0/   (~500 images)
│   ├── character_1/   (~400 images)
│   ├── character_2/   (~300 images)
│   └── noise/
│
├── action_clusters/                # Pose/Action LoRA 訓練資料
│   ├── action_000/    (~200 images)
│   ├── action_001/    (~150 images)
│   ├── action_002/    (~100 images)
│   └── noise/
│
├── expression_clusters/            # Expression LoRA 訓練資料
│   ├── expression_000/  (~100 images)
│   ├── expression_001/  (~80 images)
│   ├── expression_002/  (~60 images)
│   └── unclassified/
│
└── style_frames/                   # Style LoRA 訓練資料
    ├── frame_00001.jpg
    ├── frame_00123.jpg
    ├── ...  (400 total)
    └── style_selection.json
```

---

## 預估完成時間

### Per Film (Sequential)
- Background: 30-45 min
- Action: 2-3 hours
- Expression: 1-2 hours
- Style: 20-30 min
- **Total**: ~4-6 hours/film

### All 5 Films (Parallel, 3 concurrent jobs)
- Background: ~45 min
- Action: ~3 hours
- Expression: ~2 hours
- Style: ~30 min
- **Total**: ~4-5 hours

**當前批次預計完成時間**: 明早 ~09:00-10:00

---

## 下一步流程

### 1. 人工檢視 (批次完成後)

```bash
# 瀏覽結果
ls -lh /mnt/data/ai_data/datasets/3d-anime/*/lora_data/

# 視覺檢查
nautilus /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/background_clusters/
```

### 2. 重新命名 Clusters

```bash
# 為 clusters 命名
cd /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/background_clusters/
mv character_0 portorosso_town
mv character_1 underwater_scenes
mv character_2 luca_house_interior
```

### 3. Caption 生成 (使用 LLMProvider API)

```bash
# Background LoRA
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir .../background_clusters/portorosso_town \
  --output-dir .../training_data/luca/background_portorosso \
  --lora-type background \
  --scene-info "Italian coastal town, colorful buildings, Mediterranean" \
  --film-info "Luca (2021), Pixar, warm sunny colors" \
  --api-provider llm_vendor

# Action LoRA
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir .../action_clusters/action_000 \
  --output-dir .../training_data/luca/pose_running \
  --lora-type pose \
  --pose-info "running pose, forward lean, arms swinging" \
  --api-provider llm_vendor

# Expression LoRA
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir .../expression_clusters/expression_000 \
  --output-dir .../training_data/luca/expression_happy \
  --lora-type expression \
  --expression-info "happy expression, wide smile, bright eyes" \
  --api-provider llm_vendor

# Style LoRA
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir .../style_frames \
  --output-dir .../training_data/luca/style_pixar \
  --lora-type style \
  --style-info "Pixar 3D animation, smooth shading, vibrant colors" \
  --api-provider llm_vendor
```

### 4. LoRA 訓練

使用準備好的訓練資料訓練各類型 LoRA。

---

## 成本估算 (LLMProvider API Captions)

基於 LLMProvider 3 Haiku 定價 ($0.25/1M input, $1.25/1M output):

| LoRA 類型 | 圖片數/電影 | 成本/電影 | 5部電影總計 |
|-----------|-------------|-----------|-------------|
| Background | 500 | $0.60 | $3.00 |
| Pose | 200 | $0.24 | $1.20 |
| Expression | 100 | $0.12 | $0.60 |
| Style | 400 | $0.48 | $2.40 |
| **總計** | **1,200** | **$1.44** | **$7.20** |

非常便宜，考慮到節省的時間和品質提升！

---

## 技術特點

### CPU/GPU 資源隔離
✅ 所有任務使用 `--device cpu`
✅ CLIP 模型在 CPU 上運行
✅ 完全不影響 GPU 的 SAM2 處理
✅ 可並行處理，最大化資源利用

### 彈性與可擴展性
✅ 支援任意電影組合
✅ 可選擇性處理 LoRA 類型
✅ 可調整並發數 (PARALLEL_JOBS)
✅ 完整錯誤處理與日誌

### 自動化程度
✅ 一鍵批次處理
✅ 自動資料驗證
✅ 自動目錄創建
✅ 自動生成總結報告

### 監控與除錯
✅ 即時進度監控
✅ 詳細日誌記錄
✅ 錯誤追蹤
✅ 資源使用顯示

---

## 依賴項

### Python 套件
- ✅ `open_clip_torch` - CLIP embeddings
- ✅ `pillow` - 圖片處理
- ✅ `opencv-python` - 品質檢測
- ✅ `scikit-learn` - 聚類算法
- ✅ `umap-learn` - 降維
- ✅ `imagehash` - 重複檢測 (選用)

### 安裝
```bash
conda activate ai_env
pip install open_clip_torch pillow opencv-python scikit-learn umap-learn imagehash
```

---

## 故障排除

### 問題: CLIP 模型下載失敗
```bash
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"
```

### 問題: 記憶體不足
降低並行數:
```bash
# 在 batch_lora_data_preparation.sh 中
PARALLEL_JOBS=2  # 從 3 改為 2
```

### 問題: 找不到 clusters
降低最小 cluster 大小:
```bash
--min-cluster-size 10  # 從 15-20 降到 10
```

---

## 相關文檔

- `docs/guides/CPU_LORA_DATA_PREPARATION.md` - 詳細使用指南
- `docs/training/multi-type-lora-system.md` - LoRA 系統概述
- `docs/training/lora_types/*.md` - 各類型 LoRA 詳細說明
- `docs/films/FILM_CHARACTER_CONTEXT.yaml` - Caption 生成 context

---

## 總結

✅ **完整實作**: 5種 LoRA 類型的資料準備流程
✅ **批次處理**: 自動化處理多部電影
✅ **監控工具**: 即時進度追蹤
✅ **完整文檔**: 使用指南與技術說明
✅ **成本效益**: API caption 僅 $7.20 for 5 films
✅ **並行執行**: 與 GPU SAM2 零衝突

**系統狀態**: ✅ **全面運行中**
**預計完成**: 2025-11-19 09:00-10:00
**後續任務**: 人工檢視 → 命名 → Caption 生成 → LoRA 訓練
