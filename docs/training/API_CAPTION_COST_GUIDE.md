# API Caption Generation 成本與使用指南

## 📊 API 提供商比較（2024年11月價格）

### 支援的 API 提供商

| 提供商 | 推薦模型 | 每張圖片成本 | 速度 | 質量 | 推薦度 |
|--------|---------|-------------|------|------|--------|
| **LLMVendor LLMProvider** | `llm_provider-3-haiku-20240307` | ~$0.0012 | 快 | 極好 | ⭐⭐⭐⭐⭐ 最推薦 |
| **Google Gemini** | `gemini-1.5-flash` | ~$0.0003 | 最快 | 良好 | ⭐⭐⭐⭐ 性價比最高 |
| **OpenAI GPT-4** | `gpt-4o` | ~$0.004 | 中等 | 極好 | ⭐⭐⭐ 較貴 |

### 詳細定價（截至 2024-11-18）

#### LLMVendor LLMProvider 3 Haiku
- **輸入**: $0.25 / 1M tokens
- **輸出**: $1.25 / 1M tokens
- **圖片處理**: 每張圖片 ~1.6K tokens
- **預估**: $0.0012/圖片
- **特點**: 最佳品質/成本比，支援繁體中文指令

#### Google Gemini 1.5 Flash
- **輸入**: $0.075 / 1M tokens (≤128K context)
- **輸出**: $0.30 / 1M tokens
- **圖片處理**: 每張圖片 ~258 tokens
- **預估**: $0.0003/圖片
- **特點**: 最便宜，速度最快，免費額度高

#### OpenAI GPT-4o
- **輸入**: $2.50 / 1M tokens
- **輸出**: $10.00 / 1M tokens
- **圖片處理**: 每張圖片 ~85 tokens (low detail)
- **預估**: $0.004/圖片
- **特點**: 品質極好但較貴

---

## 💰 總成本估算

### 典型專案規模

| 專案規模 | 圖片數量 | LLMProvider Haiku | Gemini Flash | GPT-4o |
|---------|---------|-------------|--------------|--------|
| **小型專案** (1個角色) | 400 | $0.48 | $0.12 | $1.60 |
| **中型專案** (1部電影) | 2,000 | $2.40 | $0.60 | $8.00 |
| **大型專案** (5部電影) | 10,000 | $12.00 | $3.00 | $40.00 |
| **超大型專案** (全部數據) | 25,000 | $30.00 | $7.50 | $100.00 |

### 我們的專案（5部已完成 SAM2 的電影）

**預估數據量**:
- 總實例數: ~50,000 個角色實例
- 總背景數: ~10,000 張背景
- 總需要 caption: ~60,000 張圖片

**總成本估算**:
- **LLMProvider 3 Haiku**: $72 USD ⭐ 推薦
- **Gemini 1.5 Flash**: $18 USD ⭐⭐ 最便宜
- **GPT-4o**: $240 USD

**分批處理策略**:
1. 先處理 Luca (最完整) → ~$15 (LLMProvider) / $4 (Gemini)
2. 評估品質，決定是否繼續
3. 批次處理剩餘 4 部電影

---

## 🚀 使用方式

### 1. 設定 API Key

```bash
# LLMVendor LLMProvider (推薦)
export LLM_VENDOR_API_KEY="sk-ant-..."

# Google Gemini (最便宜)
export GEMINI_API_KEY="AIza..."

# OpenAI GPT-4
export OPENAI_API_KEY="sk-proj-..."
```

### 2. Character LoRA Caption 生成

```bash
# 使用 LLMProvider 3 Haiku (推薦)
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/character_0 \
  --output-dir /mnt/data/ai_data/training_data/luca/character_luca/captions \
  --lora-type character \
  --character-info "Luca Paguro, young sea monster boy, can transform between sea monster and human form, green scaly skin in sea form, brown hair and tan skin in human form, curious and adventurous personality, often wears yellow t-shirt and shorts" \
  --film-info "Luca (2021), Pixar Animation Studios, set in 1960s Italian Riviera coastal town of Portorosso, coming-of-age story about friendship and self-discovery" \
  --api-provider llm_vendor \
  --model llm_provider-3-haiku-20240307

# 或使用 Gemini Flash (最便宜)
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/identity_clusters/character_0 \
  --output-dir /mnt/data/ai_data/training_data/luca/character_luca/captions \
  --lora-type character \
  --character-info "Luca Paguro, young sea monster boy, green skin, curious personality" \
  --film-info "Luca (2021), Pixar Animation, Italian Riviera setting" \
  --api-provider gemini
```

### 3. Background LoRA Caption 生成

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters/scene_portorosso \
  --output-dir /mnt/data/ai_data/training_data/luca/background_portorosso/captions \
  --lora-type background \
  --scene-info "Portorosso town square, Italian coastal village with colorful pastel buildings, cobblestone streets, Mediterranean architecture, seaside atmosphere, vintage 1960s charm" \
  --film-info "Luca (2021), Pixar Animation, warm sunny Mediterranean color palette" \
  --api-provider llm_vendor
```

### 4. Pose LoRA Caption 生成

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/pose_clusters/running \
  --output-dir /mnt/data/ai_data/training_data/luca/pose_running/captions \
  --lora-type pose \
  --pose-info "running forward, dynamic running pose, legs mid-stride, arms swinging" \
  --character-info "Luca Paguro, young sea monster boy" \
  --film-info "Luca (2021), Pixar Animation" \
  --api-provider llm_vendor
```

### 5. Expression LoRA Caption 生成

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/expression_clusters/happy \
  --output-dir /mnt/data/ai_data/training_data/luca/expression_happy/captions \
  --lora-type expression \
  --expression-info "happy expression, wide genuine smile, bright eyes, joyful face" \
  --character-info "Luca Paguro" \
  --film-info "Luca (2021), Pixar Animation" \
  --api-provider llm_vendor
```

---

## 📝 Prompt 範例輸出

### Character LoRA Caption 範例
```
a character named Luca, young boy with green scaly skin and purple hair in sea monster form, wearing yellow t-shirt, curious expression, three-quarter view, 3d animated character, pixar style, smooth shading, warm outdoor lighting, Italian coastal setting, full body shot
```

### Background LoRA Caption 範例
```
Portorosso town square, Italian coastal village with pastel colored buildings, cobblestone streets, Mediterranean architecture, warm afternoon sunlight, blue sky, vintage lampposts, 3d animated environment, pixar style, vibrant warm colors, establishing shot, cinematic composition
```

### Pose LoRA Caption 範例
```
Luca running forward, dynamic running pose, arms swinging, legs mid-stride, front three-quarter view, energetic movement, 3d animated character, pixar style, motion blur on limbs, outdoor beach setting, full body action shot
```

---

## ⚡ 最佳實踐

### 1. 分批處理策略

```bash
# 先處理主要角色（優先級最高）
python api_caption_generator.py --input-dir .../character_luca --lora-type character

# 再處理配角
python api_caption_generator.py --input-dir .../character_alberto --lora-type character

# 最後處理背景和其他
python api_caption_generator.py --input-dir .../scene_portorosso --lora-type background
```

### 2. 成本控制

```bash
# 測試階段：先處理少量圖片驗證品質
python api_caption_generator.py \
  --input-dir .../character_luca \
  --output-dir /tmp/test_captions \
  --lora-type character \
  ... \
  --overwrite

# 檢查 /tmp/test_captions/*.txt 確認品質
# 滿意後再處理全部數據
```

### 3. 並行處理（不同 cluster 使用不同終端）

```bash
# Terminal 1: Luca
python api_caption_generator.py ... --input-dir .../character_luca &

# Terminal 2: Alberto
python api_caption_generator.py ... --input-dir .../character_alberto &

# Terminal 3: Background
python api_caption_generator.py ... --input-dir .../scene_portorosso &
```

**注意**: API 有 rate limit，並行處理可能觸發限制。建議串行處理或調整 `request_delay`。

### 4. 錯誤處理與恢復

腳本會自動：
- ✅ 跳過已存在的 caption（除非 `--overwrite`）
- ✅ 記錄失敗的圖片到 `caption_generation_stats.json`
- ✅ 保存成本估算

如果中斷：
```bash
# 重新運行相同命令，會自動續傳
python api_caption_generator.py ... # 自動跳過已完成的
```

---

## 📊 成本追蹤

每次運行會生成統計文件：

```json
{
  "total": 400,
  "generated": 395,
  "skipped": 0,
  "failed": 5,
  "total_cost_usd": 0.474,
  "failed_images": [
    ["image_001.png", "API timeout"],
    ...
  ],
  "timestamp": "2024-11-18T20:30:00",
  "api_provider": "llm_vendor",
  "model": "llm_provider-3-haiku-20240307",
  "lora_type": "character"
}
```

### 彙總所有成本

```bash
# 查看所有 caption 生成成本
find /mnt/data/ai_data/training_data -name "caption_generation_stats.json" -exec jq '.total_cost_usd' {} \; | paste -sd+ | bc
```

---

## 🎯 推薦方案

### 方案 A: 最佳品質（LLMProvider 3 Haiku）
- **成本**: $72 USD (60,000 張圖片)
- **品質**: ⭐⭐⭐⭐⭐
- **速度**: 快
- **適合**: 需要最佳 caption 品質，預算充足

### 方案 B: 性價比之王（Gemini 1.5 Flash）⭐ 推薦
- **成本**: $18 USD (60,000 張圖片)
- **品質**: ⭐⭐⭐⭐
- **速度**: 最快
- **適合**: 大量數據，預算有限

### 方案 C: 混合策略
- **主要角色**: LLMProvider Haiku ($10)
- **配角 + 背景**: Gemini Flash ($8)
- **總成本**: ~$18 USD
- **品質**: 平衡

---

## 🔧 進階配置

### 自訂 Prompt Template

編輯 `api_caption_generator.py` 中的 `PROMPTS` 字典：

```python
PROMPTS = {
    "character": """你的自訂 prompt...""",
    ...
}
```

### 調整 Rate Limiting

```python
# 在 APICaptionGenerator.__init__
self.request_delay = 0.5  # 改為 0.2 加快速度（注意 rate limit）
```

### 批次大小控制

```bash
# 分批處理大型資料夾
ls /path/to/images | split -l 500 - batch_

# 處理每個批次
for batch in batch_*; do
    mkdir batch_input
    cat $batch | xargs -I {} cp /path/to/images/{} batch_input/
    python api_caption_generator.py --input-dir batch_input ...
    rm -rf batch_input
done
```

---

## ❓ 常見問題

**Q: 哪個 API 最推薦？**
A: **Gemini 1.5 Flash** 性價比最高（$18 vs $72），品質足夠好。如果追求極致品質，用 LLMProvider 3 Haiku。

**Q: 可以混用多個 API 嗎？**
A: 可以！不同 cluster 用不同 API，或先用 Gemini 測試，重要角色用 LLMProvider。

**Q: 如何降低成本？**
A:
1. 使用 Gemini Flash（最便宜）
2. 先處理 top-k 優質圖片（用 CLIP score 篩選）
3. 只為主要角色生成 caption，配角用模板

**Q: Caption 品質如何？**
A: 三個 API 都很好，LLMProvider 和 GPT-4 略勝，但 Gemini 也完全足夠訓練使用。

**Q: 需要多久時間？**
A:
- Gemini Flash: ~8 小時 (60,000 張)
- LLMProvider Haiku: ~10 小時
- GPT-4o: ~12 小時

**Q: 有免費額度嗎？**
A:
- Gemini: 1500 requests/day 免費（可處理 ~1500 張/天）
- LLMProvider: 新用戶 $5 credit
- OpenAI: 新用戶 $5 credit

---

## 📚 相關文檔

- `api_caption_generator.py` - Caption 生成腳本
- `batch_dataset_preparation.sh` - 批次數據準備
- `docs/training/lora-composition.md` - Multi-LoRA 系統
- `LLM_PROVIDER.md` - 專案總覽

---

**最後更新**: 2024-11-18
**推薦方案**: Gemini 1.5 Flash（$18 處理全部 60K 圖片）
