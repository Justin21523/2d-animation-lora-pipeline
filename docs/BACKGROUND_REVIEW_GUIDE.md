# 背景場景審查與處理流程

## 概述

這個系統提供完整的背景場景處理 pipeline：從 clustering 到互動式審查，最後到 caption 生成。

## 完整流程

```
Background Inpainting
  ↓
Giant Clustering (自動分群)
  ↓
Web-based Interactive Review (手動審查)
  ↓
Export Kept Scenes
  ↓
Caption Generation (保留場景的 caption)
```

---

## 📋 使用步驟

### Step 1: 背景 Inpainting (如果還沒做)

```bash
# 例如：Orion 背景 inpainting
conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \
  --frames-dir /mnt/data/ai_data/datasets/3d-anime/orion/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/orion/backgrounds_lama_v2 \
  --device cuda
```

### Step 2: Scene Clustering (Giant)

處理所有電影的背景：

```bash
bash scripts/batch/process_backgrounds_pipeline.sh all --stage 1
```

或指定特定電影：

```bash
bash scripts/batch/process_backgrounds_pipeline.sh luca elio coco onward --stage 1
```

**輸出：** 每個電影會產生 `lora_data/scene_clusters_giant/` 目錄

---

### Step 3: 互動式背景審查 (Web UI)

#### 啟動審查介面

對每個電影分別啟動：

```bash
# 例如：Luca
python scripts/generic/clustering/interactive_background_reviewer.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/scene_clusters_giant \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/backgrounds_reviewed \
  --port 5000

# 然後打開瀏覽器訪問: http://localhost:5000
```

#### 審查功能

**Web 介面提供：**
- ✅ 網格式顯示所有 scene clusters
- 🖼️ 點擊預覽整個 cluster 的所有圖片
- ✓ 快速標記「保留」或「刪除」
- 🔍 過濾器：全部 / 待審查 / 已保留 / 已刪除
- 💾 自動保存進度 (儲存在 `review_state.json`)
- 📤 一鍵匯出所有保留的背景

**鍵盤快捷鍵：**
- `K` - 標記為保留 (Keep)
- `D` - 標記為刪除 (Discard)
- `Space` - 預覽當前 cluster
- `Esc` - 關閉預覽

#### 審查建議

1. **快速瀏覽** - 先過一遍所有 clusters，標記明顯要刪除的（模糊、重複、無用）
2. **詳細檢查** - 對不確定的 clusters 點擊預覽，查看所有圖片
3. **保留標準**：
   - ✅ 清晰、完整的場景背景
   - ✅ 有代表性的場景（室內、室外、特定地點）
   - ✅ 光線、色調合理
   - ❌ 模糊、過曝、過暗
   - ❌ 角色殘影（inpainting 失敗）
   - ❌ 過於相似的重複場景

4. **匯出結果** - 審查完成後點擊「匯出保留的背景」按鈕

**輸出：** `backgrounds_reviewed/` 目錄 + `export_manifest.json`

---

### Step 4: Caption Generation

為所有保留的背景生成 captions：

```bash
bash scripts/batch/process_backgrounds_pipeline.sh all --stage 3
```

這會對每個電影的 `backgrounds_reviewed/` 中的場景生成 captions。

---

## 🎯 實際操作範例

### 處理 Luca, Elio, Coco 三部電影

```bash
# 1. Clustering (如果背景已經 inpainted)
bash scripts/batch/process_backgrounds_pipeline.sh luca elio coco --stage 1

# 2. 等待 clustering 完成後，分別審查每部電影

# 審查 Luca (Port 5000)
python scripts/generic/clustering/interactive_background_reviewer.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/scene_clusters_giant \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/backgrounds_reviewed \
  --port 5000 &

# 審查 Elio (Port 5001)
python scripts/generic/clustering/interactive_background_reviewer.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/elio/lora_data/scene_clusters_giant \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/elio/lora_data/backgrounds_reviewed \
  --port 5001 &

# 審查 Coco (Port 5002)
python scripts/generic/clustering/interactive_background_reviewer.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/coco/lora_data/scene_clusters_giant \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/coco/lora_data/backgrounds_reviewed \
  --port 5002 &

# 打開三個瀏覽器標籤：
#   http://localhost:5000 (Luca)
#   http://localhost:5001 (Elio)
#   http://localhost:5002 (Coco)

# 3. 完成審查並匯出後，生成 captions
bash scripts/batch/process_backgrounds_pipeline.sh luca elio coco --stage 3
```

---

## 📊 輸出結構

```
<movie>/lora_data/
├── scene_clusters_giant/        # Stage 1: Clustering 結果
│   ├── scene_0/
│   ├── scene_1/
│   ├── ...
│   ├── scene_clustering_report.json
│   └── review_state.json        # 審查進度 (自動生成)
│
├── backgrounds_reviewed/        # Stage 3: 保留的背景
│   ├── scene_0/                 # 只包含保留的 clusters
│   ├── scene_3/
│   ├── ...
│   └── export_manifest.json     # 匯出清單
│
└── backgrounds_captioned/       # Stage 4: Captions (可選)
    ├── scene_0/
    │   ├── image1.png
    │   └── image1.txt           # Caption
    └── ...
```

---

## 🚀 進階技巧

### 批次處理所有電影

```bash
# 一次處理所有電影的 clustering
bash scripts/batch/process_backgrounds_pipeline.sh all --stage 1

# 等待完成後，啟動多個審查介面（不同 port）
for i in {0..6}; do
  movie=$(ls -d /mnt/data/ai_data/datasets/3d-anime/*/ | sed -n "$((i+1))p" | xargs basename)
  port=$((5000 + i))
  python scripts/generic/clustering/interactive_background_reviewer.py \
    --cluster-dir "/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/scene_clusters_giant" \
    --output-dir "/mnt/data/ai_data/datasets/3d-anime/$movie/lora_data/backgrounds_reviewed" \
    --port $port &
  echo "Started $movie on port $port"
done
```

### 恢復中斷的審查

如果審查過程中關閉了瀏覽器，重新啟動同樣的命令即可恢復進度（`review_state.json` 會自動載入）。

### 重新審查特定電影

```bash
# 刪除審查狀態重新開始
rm /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/scene_clusters_giant/review_state.json

# 重新啟動審查
python scripts/generic/clustering/interactive_background_reviewer.py \
  --cluster-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/scene_clusters_giant \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/backgrounds_reviewed \
  --port 5000
```

---

## ❓ 常見問題

**Q: Clustering 需要多久？**
A: 取決於背景數量，通常每個電影 10-30 分鐘（GPU 加速）。

**Q: 審查介面卡住怎麼辦？**
A: 刷新瀏覽器頁面，進度會自動恢復。

**Q: 如何修改 cluster 的最小大小？**
A: 編輯 `process_backgrounds_pipeline.sh` 中的 `--min-cluster-size` 參數（預設 8）。

**Q: Caption 生成失敗怎麼辦？**
A: 檢查 logs/background_pipeline/caption_* 日誌檔案，可能是 API 配額或網路問題。

---

## 📝 技術細節

### Clustering 演算法

- **Embeddings**: CLIP ViT-L/14
- **Dimensionality Reduction**: UMAP
- **Clustering**: HDBSCAN (auto-k, noise filtering)
- **Mode**: `giant` (適合大量場景，較大的 cluster sizes)

### Web 介面技術

- **Backend**: Flask (輕量級 Python web framework)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **圖片服務**: Flask send_from_directory
- **狀態管理**: JSON 檔案 (filesystem-based)

### Caption 生成 (待實作)

目前 Stage 3 的 caption 生成腳本 `generate_scene_captions.py` 還需要實作，可參考現有的：
- `scripts/generic/training/llm_provider_character_caption.py` (角色 caption)
- `scripts/generic/training/generate_expression_captions.py` (表情 caption)

---

## 🎓 最佳實踐

1. **先小規模測試** - 在一部電影上完整跑一遍流程，確認沒問題後再批次處理
2. **定期備份** - `review_state.json` 和 `export_manifest.json` 很重要
3. **並行審查** - 使用不同 port 同時審查多部電影，提高效率
4. **保留標準一致** - 建立明確的保留/刪除標準，確保所有電影審查品質一致
5. **記錄決策** - 對於特殊情況（如邊界案例），在 notes 欄位記錄原因

---

## 🔗 相關腳本

- `scripts/generic/clustering/scene_clustering.py` - CLIP-based clustering
- `scripts/generic/clustering/interactive_background_reviewer.py` - Web UI
- `scripts/batch/process_backgrounds_pipeline.sh` - 完整 pipeline
- `scripts/batch/monitor_inpainting.sh` - 監控 inpainting 進度

---

**Author**: LLMProvider + User
**Last Updated**: 2025-11-21
**Version**: 1.0
