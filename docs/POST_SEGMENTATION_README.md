# Post-Segmentation Pipeline - Quick Reference

**當 SAM2 分割和 inpainting 完成後，使用此 pipeline 建立訓練資料集**

---

## 📚 文檔導覽

### 1. **POST_SEGMENTATION_PIPELINE_PLAN.md** (完整計畫)
   - 所有 7 個階段的詳細設計
   - 每個階段的程式碼實作範例
   - 配置文件說明
   - 使用範例

### 2. **IMPLEMENTATION_STATUS.md** (實作進度追蹤)
   - 每個階段的完成度
   - 待辦事項清單
   - 測試計畫
   - 已知問題

### 3. **配置文件**
   - `configs/stages/clustering/face_identity_clustering.yaml` - 身份聚類配置
   - `configs/stages/captioning/vlm_3d_character_captioning.yaml` - VLM 字幕配置

---

## 🚀 快速開始

### 前置條件

SAM2 instance segmentation 已完成:
```
segmented/
└── characters/
    ├── frame_0001_instance_0.png
    ├── frame_0001_instance_1.png
    ├── frame_0002_instance_0.png
    └── ...
```

### 完整 Pipeline 執行

```bash
# Step 1: 實例預過濾 (移除 80-90% 背景)
python scripts/generic/clustering/instance_prefilter.py \
  --input-dir /path/to/segmented/characters \
  --output-dir /path/to/filtered_instances \
  --mode balanced \
  --enable-semantic

# Step 2: 人臉身份聚類 (依角色分組)
python scripts/generic/clustering/face_identity_clustering.py \
  --input-dir /path/to/filtered_instances \
  --output-dir /path/to/identity_clusters \
  --config configs/stages/clustering/face_identity_clustering.yaml

# Step 3: 互動式審查 (可選但建議)
python scripts/generic/clustering/web_cluster_review.py \
  --cluster-dir /path/to/identity_clusters \
  --port 5000
# 開啟瀏覽器 http://localhost:5000 進行審查

# Step 4: VLM 字幕生成
python scripts/generic/training/vlm_caption_generator.py \
  --cluster-dir /path/to/identity_clusters/character_0 \
  --character-name "luca" \
  --model qwen2_vl \
  --output-format both

# Step 5: 資料集組裝
python scripts/generic/training/dataset_assembler.py \
  --input-dir /path/to/identity_clusters/character_0 \
  --output-dir /path/to/training_data/luca \
  --character-name "luca" \
  --augment \
  --target-size 400
```

### 自動化 Pipeline (推薦)

```bash
# 一鍵執行所有階段
python scripts/pipelines/post_segmentation_pipeline.py \
  --input-dir /mnt/data/datasets/3d-anime/luca/segmented/characters \
  --output-dir /mnt/data/datasets/3d-anime/luca/training_data \
  --project-name luca \
  --enable-pose-subclustering
```

---

## 📊 Pipeline 階段概覽

| 階段 | 輸入 | 輸出 | 耗時 (估計) | 狀態 |
|------|------|------|------------|------|
| 0. SAM2 分割 | Video frames | Character instances | 30-60 min | ✅ 完成 |
| 1. 實例預過濾 | Character instances | Filtered instances | 1-2 min | ✅ 完成 |
| 2. 身份聚類 | Filtered instances | Identity clusters | 5-10 min | 🚧 進行中 |
| 3. 互動審查 | Identity clusters | Refined clusters | 10-30 min | 🚧 進行中 |
| 4. 姿態子聚類 | Identity clusters | Pose buckets | 5-10 min | ⏳ 計畫中 |
| 5. VLM 字幕 | Clusters | Captioned images | 30-60 min | ⏳ 計畫中 |
| 6. 資料集組裝 | Captioned images | Training dataset | 5-10 min | ⏳ 計畫中 |

**總耗時:** ~2-3 小時 (1000 instances, 包含人工審查)

---

## 💡 關鍵概念

### 1. **身份聚類 vs. 視覺聚類**

**❌ 舊方法 (CLIP 視覺相似度):**
- 依視覺相似度分組
- 問題: 同場景的不同角色會被分在一起
- 問題: 相似服裝的不同角色會混淆

**✅ 新方法 (人臉身份辨識):**
- 依「誰是誰」分組 (identity)
- 使用 ArcFace embeddings (512-D)
- 同一角色的不同表情/角度會正確分組

### 2. **3D 動畫專用參數**

**與 2D anime 的差異:**
- `alpha_threshold`: 0.15 (vs. 0.25) - 3D 有更柔和的抗鋸齒邊緣
- `blur_threshold`: 80 (vs. 100) - 3D 常有景深模糊效果
- `min_cluster_size`: 12 (vs. 20-25) - 3D 角色模型一致性高
- `dataset_size`: 200-500 (vs. 500-1000) - 3D 需要較少訓練樣本

### 3. **Schema-Guided VLM Captioning**

**不只是生成文字描述，而是結構化提取:**
```json
{
  "character": "luca",
  "materials": {
    "skin": "smooth 3d skin shader with subsurface scattering",
    "clothing": "cotton t-shirt, PBR fabric material"
  },
  "lighting": {
    "key": "warm sunlight from upper right",
    "fill": "soft ambient fill light",
    "rim": "blue rim light from behind"
  },
  "camera": {
    "shot_type": "medium shot",
    "angle": "eye-level",
    "depth_of_field": "shallow DoF with bokeh"
  }
}
```

**好處:**
- 確保字幕包含 3D 渲染特徵 (材質、光照、相機)
- 一致性高 (同角色的字幕風格統一)
- 可程式化處理 (自動檢查、補充)

---

## 🛠️ 當前實作狀態

### ✅ 可立即使用
1. SAM2 instance segmentation
2. Instance pre-filtering

### 🚧 正在實作 (本週完成)
3. Face identity clustering
4. Web-based cluster review UI

### ⏳ 下週開始
5. VLM captioning (Qwen2-VL integration)
6. Dataset assembly

---

## 📝 實作優先順序

**Phase 1 (本週):**
1. 完成 `face_identity_clustering.py` 實作
2. 建立 web cluster review UI (HTML/CSS/JS + Flask backend)
3. 在 Super Wings 資料集上測試 (8 個角色的已知 ground truth)

**Phase 2 (下週):**
4. 整合 Qwen2-VL 模型
5. 實作 schema-guided caption generation
6. 測試字幕品質 (CLIP score, consistency)

**Phase 3 (第三週):**
7. 實作 dataset assembler (quality filtering, augmentation, train/val split)
8. 建立 pipeline orchestrator
9. End-to-end 測試

---

## 🧪 測試資料集

**Super Wings (理想測試集):**
- ✅ 8 個已知角色 (ground truth)
- ✅ 多角色場景 (測試身份聚類)
- ✅ 各種角度和姿勢
- ✅ 已完成 SAM2 分割

**測試指令:**
```bash
# 在 Super Wings 上測試完整 pipeline
python scripts/pipelines/post_segmentation_pipeline.py \
  --input-dir /mnt/data/datasets/general/super-wings/segmented/characters \
  --output-dir /mnt/data/datasets/general/super-wings/training_data \
  --project-name super_wings

# 驗證結果
python scripts/evaluation/validate_training_dataset.py \
  --dataset-dir /mnt/data/datasets/general/super-wings/training_data \
  --ground-truth configs/characters/super_wings_characters.json
```

**預期結果:**
- 8 個角色正確分離
- 每個角色 150-250 張圖片
- 字幕包含正確的顏色和特徵 (特別是 Flip 的藍色帽子)

---

## 📦 依賴安裝

```bash
# Face detection & recognition
pip install retinaface-pytorch facenet-pytorch mtcnn

# Clustering
pip install hdbscan umap-learn

# VLM
pip install transformers>=4.35.0 accelerate bitsandbytes

# Web UI
pip install flask flask-cors

# Image processing
pip install imagehash opencv-python pillow
```

**或使用模組化 requirements:**
```bash
pip install -r requirements/clustering.txt
pip install -r requirements/training.txt
```

---

## 🔍 常見問題

### Q1: 為什麼需要人臉身份辨識？CLIP 視覺聚類不夠嗎？

**A:** CLIP 會依視覺相似度分組，在多角色場景中會失敗。例如:
- 同場景的 Luca 和 Alberto 會被分在一起 (因為背景和光照相同)
- 穿相似服裝的不同角色會混淆

人臉身份辨識使用 ArcFace embeddings，專門訓練來區分「誰是誰」，不受表情、光照、角度影響。

### Q2: 為什麼需要互動審查？自動聚類不準確嗎？

**A:** HDBSCAN 聚類通常 90%+ 準確，但仍需人工審查:
- 合併錯誤分開的同一角色
- 分離混在一起的不同角色
- 為角色命名 (character_0 → "luca")
- 移除 noise 實例

互動審查只需 10-30 分鐘，但能大幅提升最終資料集品質。

### Q3: VLM 字幕生成很慢怎麼辦？

**A:** 優化策略:
1. 使用 4-bit 量化 (load_in_4bit=True)
2. 減少 batch size (2-4)
3. 只對每個 cluster 的代表性圖片生成字幕，其餘複用
4. 使用更小的模型 (Qwen2-VL-2B instead of 7B)

### Q4: 如何確保字幕品質？

**A:** 多層品質控制:
1. **Schema enforcement:** JSON schema 強制結構化輸出
2. **Consistency checks:** 同角色的字幕應該相似 (CLIP score)
3. **Hallucination detection:** 拒絕包含虛構細節的字幕
4. **Token length constraints:** 確保 40-77 tokens (SD 訓練需求)
5. **Human review:** 抽樣檢查並修正

---

## 📖 參考資料

**相關文檔:**
- `docs/guides/MULTI_CHARACTER_CLUSTERING.md` - 多角色聚類完整方法論
- `docs/3d_anime_specific/PROCESSING_GUIDE.md` - 3D 動畫處理指南
- `docs/3d_anime_specific/PARAMETERS_COMPARISON.md` - 2D vs 3D 參數對照表

**相關配置:**
- `configs/stages/clustering/` - 聚類配置文件
- `configs/stages/captioning/` - 字幕配置文件
- `configs/characters/` - 角色定義文件

**腳本位置:**
- `scripts/generic/clustering/` - 聚類相關腳本
- `scripts/generic/training/` - 訓練資料準備腳本
- `scripts/pipelines/` - Pipeline orchestrators

---

## ✅ Next Steps

1. **閱讀完整計畫:** `POST_SEGMENTATION_PIPELINE_PLAN.md`
2. **檢查實作狀態:** `IMPLEMENTATION_STATUS.md`
3. **開始實作:** 從 `face_identity_clustering.py` 開始
4. **測試:** 在 Super Wings 資料集上驗證

**有任何問題請參考相關文檔或更新 IMPLEMENTATION_STATUS.md** 🚀
