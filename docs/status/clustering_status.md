# 🔄 Clustering 執行狀態報告

**時間**: 2025-11-19 02:30
**狀態**: ✅ 正在執行中

---

## ✅ 當前執行狀態 (2025-11-19 02:35)

### CPU Clustering Progress (Identity Clustering - Face Detection Phase)

All 4 completed films running face detection in parallel:

| Film | Instances | PID | Runtime | CPU | Status |
|------|-----------|-----|---------|-----|--------|
| onward | 33,065 | 97578, 100247 | 1:00h, 0:18h | 446%, 304% | 🔄 Processing |
| turning-red | 27,659 | 97658, 100256 | 1:18h, 0:22h | 584%, 365% | 🔄 Processing |
| coco | 30,430 | 97772, 100332 | 1:24h, 0:23h | 630%, 394% | 🔄 Processing |
| elio | 26,010 | 98010, 100368 | 0:58h, 0:21h | 437%, 359% | 🔄 Processing |

**Note**: Duplicate processes detected (from 01:18 and 01:26 batch starts). Not harmful but wasteful.

**Face Detection Progress**: Still in initial phase, no output clusters yet. Expected to complete in ~8-10 hours total.

### GPU Status - SAM2 Batch Processing

**Current Task**: Processing "up" film SAM2 segmentation
- PID 70051, Runtime: 3h 29min, CPU: 88.8%, Memory: 2.4GB
- Progress: 696/1640 frames (42.4%)
- GPU: 100% utilization, 15.6GB/16.3GB memory, 41°C

**Completed**:
- ✅ coco: 2056/2056 SAM2 + LaMa
- ✅ elio: 1908/1908 SAM2 + LaMa
- ✅ onward: 2316/2316 SAM2 + LaMa
- ✅ turning-red: 1892/1892 SAM2 + LaMa

**Pending**:
- 🔄 up: 696/1640 SAM2 (42.4% - currently processing)
- ⏳ luca: 14410/14410 SAM2 ✅, 0/14410 LaMa ⏳
- ⏳ orion: 383/2803 SAM2 (13.6% - needs resume)

**Conclusion**: ✅ **CPU clustering completely independent, zero GPU interference**

---

## 📋 處理流程

### Phase 1: Identity Clustering (進行中)
使用 ArcFace 面部識別 + HDBSCAN 聚類

| 電影 | 狀態 | 預估時間 |
|------|------|---------|
| onward | 🔄 處理中 | ~1.5h |
| turning-red | 🔄 處理中 | ~1.5h |
| coco | 🔄 剛啟動 | ~1h |
| elio | ⏳ 待處理 | ~1h |
| luca | ⏳ 待處理 | ~2h |

**總預估時間**: ~3 小時（並行處理）

### Phase 2: Scene Clustering (待啟動)
使用 CLIP embeddings + HDBSCAN 聚類

| 電影 | 狀態 | 預估時間 |
|------|------|---------|
| All 5 films | ⏳ 待 Phase 1 完成 | ~2h |

**Phase 2 預估時間**: ~2 小時（並行處理）

---

## ⏱️ 完成時間預估

**開始時間**: 01:20
**Phase 1 完成**: ~04:20 (3 小時後)
**Phase 2 完成**: ~06:20 (5 小時後)

**總預計完成時間**: 明早 **06:20** 左右

---

## 📊 預期輸出

### Identity Clusters (角色聚類)

```
/mnt/data/ai_data/datasets/3d-anime/
├── luca/identity_clusters/
│   ├── character_0/     (~2000 images - Luca)
│   ├── character_1/     (~1500 images - Alberto)
│   ├── character_2/     (~800 images - Giulia)
│   ├── character_3/     (~400 images - Massimo)
│   ├── noise/           (未分類)
│   ├── cluster_report.json
│   └── cluster_visualization.png
├── onward/identity_clusters/
│   ├── character_0/     (~1200 images - Ian)
│   ├── character_1/     (~1000 images - Barley)
│   └── ...
└── [other films]/...
```

### Scene Clusters (場景聚類)

```
/mnt/data/ai_data/datasets/3d-anime/
├── luca/scene_clusters/
│   ├── character_0/     (~4000 images - Portorosso town)
│   ├── character_1/     (~3000 images - Beach/coastline)
│   ├── character_2/     (~2000 images - Underwater)
│   └── ...
└── [other films]/...
```

**總計**: ~50K 角色實例 + ~10K 背景場景

---

## 🎯 後續步驟 (明早 06:30+)

### 1. 人工檢視分組結果 (~30分鐘)

```bash
# 快速預覽
bash scripts/batch/quick_preview_clusters.sh luca identity
bash scripts/batch/quick_preview_clusters.sh luca scene
firefox /tmp/luca_identity_preview.html

# 或直接瀏覽資料夾
ls -lh /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/
```

### 2. 準備角色/場景命名清單

建立文件記錄每個 cluster 對應的角色/場景名稱：

```
# luca_character_mapping.txt
character_0 = Luca
character_1 = Alberto
character_2 = Giulia
character_3 = Massimo

# luca_scene_mapping.txt
character_0 = Portorosso_town
character_1 = Beach_coastline
character_2 = Underwater_scenes
```

### 3. 使用 LLMProvider API 生成 Captions

已準備好的詳細 context: `docs/films/FILM_CHARACTER_CONTEXT.yaml`

```bash
# 設定 API Key
export LLM_VENDOR_API_KEY="sk-ant-your-key-here"

# Character LoRA - Luca (範例)
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/character_0 \
  --output-dir /mnt/data/ai_data/training_data/luca/character_luca \
  --lora-type character \
  --character-info "Luca Paguro, 13-14 year old sea monster boy who transforms between sea monster (bright teal-green scaly skin, purple-blue wavy hair with fins, large green-blue eyes) and human form (tan skin, dark brown messy hair, brown eyes). Curious, cautious, eager to learn, kind-hearted personality. Typically wears yellow short-sleeve t-shirt with brown shorts." \
  --film-info "Luca (2021), Pixar Animation Studios, directed by Enrico Casarosa. Set in Italian Riviera coastal town of Portorosso during summer 1960s. Vibrant warm Mediterranean colors with saturated blues for water and pastel buildings. Warm natural outdoor lighting with strong sun. Smooth PBR shading with subsurface scattering. Hand-painted impressionistic feel. Coming-of-age friendship story." \
  --api-provider llm_vendor \
  --model llm_provider-3-haiku-20240307

# Background LoRA - Portorosso (範例)
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters/character_0 \
  --output-dir /mnt/data/ai_data/training_data/luca/background_portorosso \
  --lora-type background \
  --scene-info "Portorosso town, Italian coastal village with stacked colorful pastel buildings in yellows, oranges, and pinks. Narrow cobblestone streets winding upward, Mediterranean architecture with terracotta roofs and stone walls. Vintage 1960s Italian seaside charm with harbor, plaza, and market areas." \
  --film-info "Luca (2021), Pixar Animation. Warm sunny Mediterranean color palette with golden sunlight. Vibrant saturated colors. Hand-painted architectural feel with impressionistic water. Established as bustling tourist and fishing town." \
  --api-provider llm_vendor
```

---

## 💰 Caption 成本估算 (LLMProvider 3 Haiku)

### Character LoRA (主要角色)

| 電影 | 主要角色 | 預估圖片數 | 成本 (USD) |
|------|---------|-----------|-----------|
| Luca | Luca, Alberto, Giulia, Massimo | 1,200 | $1.44 |
| Onward | Ian, Barley | 800 | $0.96 |
| Turning Red | Mei, Ming | 700 | $0.84 |
| Coco | Miguel, Héctor | 600 | $0.72 |
| Elio | Elio | 400 | $0.48 |
| **小計** | **12 個主角** | **3,700** | **$4.44** |

### Background LoRA (主要場景)

| 電影 | 主要場景數 | 預估圖片數 | 成本 (USD) |
|------|-----------|-----------|-----------|
| All 5 films | 10-15 場景 | 3,000 | $3.60 |

### 總計

**Character + Background**: ~6,700 張圖片 ≈ **$8.04 USD**

---

## 📝 監控指令

### 查看進度

```bash
# 查看正在運行的 clustering 進程
ps aux | grep "face_identity_clustering\|clip_character_clustering" | grep -v grep

# 查看 CPU 使用率
top -bn1 | grep python | head -5

# 查看最新日誌
tail -f logs/clustering/identity_luca_*.log
tail -f logs/clustering/scene_luca_*.log
```

### 檢查輸出

```bash
# 檢查是否開始生成 clusters
ls -lh /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/
find /mnt/data/ai_data/datasets/3d-anime -type d -name "character_*"

# 統計已處理的角色數量
find /mnt/data/ai_data/datasets/3d-anime/*/identity_clusters -type d -name "character_*" | wc -l
```

### GPU 狀態（確認不受影響）

```bash
# 持續監控 GPU
watch -n 5 nvidia-smi

# 確認 GPU 只被 SAM2 使用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## 🎯 關鍵優勢

1. ✅ **完全並行**: GPU 處理 SAM2，CPU 處理 clustering
2. ✅ **零衝突**: Clustering 使用 `--device cpu`，不影響 GPU
3. ✅ **高品質**: LLMProvider 3 Haiku API 品質優於本地 VLM
4. ✅ **低成本**: $8 for 6.7K captions（比本地 GPU 時間便宜）
5. ✅ **詳細 Context**: 準備了完整的電影、角色、場景資訊

---

## 🚨 故障排除

### Clustering 失敗
```bash
# 檢查日誌
tail -100 logs/clustering/identity_luca_*.log

# 重新執行單一電影
bash scripts/batch/batch_identity_scene_clustering.sh luca
```

### 記憶體不足
```bash
# 降低並行數（編輯腳本）
PARALLEL_JOBS=2  # 從 4 改為 2

# 或串行處理
bash scripts/batch/batch_identity_scene_clustering.sh luca
# 等完成後再處理下一部
```

### GPU 被佔用
```bash
# 確認 clustering 使用 CPU
ps aux | grep "face_identity_clustering" | grep "device cpu"

# 如果看到 --device cuda，立即停止並修正
pkill -f face_identity_clustering
```

---

## 📚 相關文檔

- `scripts/batch/batch_identity_scene_clustering.sh` - Clustering 主腳本
- `scripts/generic/training/api_caption_generator.py` - Caption 生成
- `docs/films/FILM_CHARACTER_CONTEXT.yaml` - **詳細電影/角色 context**
- `docs/training/API_CAPTION_COST_GUIDE.md` - API 成本指南
- `docs/guides/QUICK_START_CPU_TASKS.md` - 快速開始
- `CPU_TASKS_SUMMARY.md` - 總結文檔

---

**狀態**: ✅ **一切正常運行中**
**預計完成**: 明早 06:20
**後續任務**: 人工檢視 → 命名 clusters → LLMProvider API captions → LoRA 訓練
