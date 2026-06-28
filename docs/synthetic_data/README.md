# Synthetic Data Generation System

**完整的合成數據生成系統，用於訓練 Pose/Expression/Action LoRAs**

## 📚 文檔索引

### 核心文檔
- **[配置檢查清單 (CONFIGURATION_CHECKLIST.md)](./CONFIGURATION_CHECKLIST.md)** - 完整的配置驗證清單，包含最新優化參數
- **[原始進度追蹤](../../SYNTHETIC_DATA_GENERATION_PROGRESS.md)** - 專案進度和問題追蹤

### 相關配置文件
- **[Pipeline 配置](../../configs/batch/synthetic_data_generation.yaml)** - 主要配置文件
- **[角色定義](../../configs/characters/)** - 14 個角色的配置

### 核心腳本
- **[Orchestrator](../../scripts/batch/synthetic_data_orchestrator.py)** - Pipeline 協調器
- **[Vocabulary Generator](../../scripts/generic/training/orchestration/vocabulary_generator.py)** - Prompt 生成器
- **[Batch Image Generator](../../scripts/generic/training/batch_image_generator.py)** - 圖片生成器

---

## 🎯 系統概述

### 目標
使用現有的 Identity LoRAs 自動生成大量訓練圖片，用於訓練：
- **Pose LoRAs** - 姿勢/視角控制
- **Expression LoRAs** - 表情控制
- **Action LoRAs** - 動作控制

### 規模
- **14 個角色** (Pixar + DreamWorks)
- **3 種 LoRA 類型** (pose, expression, action)
- **每種類型 50 個 prompts**
- **每個 prompt 10 張圖片**
- **總計: 21,000 張圖片**

---

## 🚀 快速開始

### 1. 啟動 Pipeline

```bash
cd /mnt/c/ai_projects/3d-animation-lora-pipeline && \
nohup /home/justin/miniconda3/envs/ai_env/bin/python \
  scripts/batch/synthetic_data_orchestrator.py \
  --config configs/batch/synthetic_data_generation.yaml \
  > /mnt/data/ai_data/synthetic_lora_data/logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 2. 監控進度

```bash
tail -f /mnt/data/ai_data/synthetic_lora_data/logs/pipeline_*.log
```

### 3. 檢查狀態

```bash
# 查看 checkpoint
cat /mnt/data/ai_data/synthetic_lora_data/checkpoints/pipeline_progress.json

# 查看生成的圖片數量
find /mnt/data/ai_data/synthetic_lora_data/generated_data -name "*.png" | wc -l
```

---

## 🏗️ 系統架構

### Phase 1: Vocabulary Generation (Prompt 生成)

**腳本**: `vocabulary_generator.py`

**輸入**:
- Character name (觸發詞)
- Character description
- LoRA type (pose/expression/action)
- Number of prompts

**輸出**:
- `{character}/{lora_type}/prompts.json`

**特點**:
- 使用模板化生成
- 14 種風格變體 (50% 包含完整材質描述)
- 自動包含 Pixar 風格和技術參數

### Phase 2: Image Generation (圖片生成)

**腳本**: `batch_image_generator.py`

**輸入**:
- Prompts JSON
- Identity LoRA path
- SDXL base model

**輸出**:
- `{character}/{lora_type}/generated/*.png`

**特點**:
- 批次處理，支援 GPU 優化
- 隨機種子確保多樣性
- Comprehensive negative prompt 避免瑕疵

---

## ⚙️ 核心配置

### 優化後的參數 (v2.1)

| 參數 | 值 | 說明 |
|------|-----|------|
| `num_inference_steps` | **40** | 基於測試優化，提供最佳細節 |
| `guidance_scale` | **7.5** | 平衡的 prompt 遵循度 |
| `lora_scale` | **1.0** | 完整角色特徵 |
| `use_random_seeds` | **true** | 生成多樣化圖片 |
| `negative_prompt` | **939 字元** | Comprehensive 版本 |

### Comprehensive Negative Prompt

```
photograph, photo, realistic, photorealistic, real person, real life, live action,
hyperrealistic, real human, candid photo, portrait photo, stock photo,
multiple people, two people, group, crowd, duplicate character, duplicate person,
clone, second person, extra character, adult, elderly, old person, mature, grown-up,
old man, baby, toddler, infant, young child, girl, female, woman, extra limbs,
extra arms, extra legs, extra hands, extra fingers, missing limbs, missing arms,
missing legs, missing hands, missing fingers, deformed, disfigured, distorted,
malformed, mutated, mutation, bad anatomy, wrong anatomy, anatomically incorrect,
blurry, out of focus, unfocused, fuzzy, hazy, soft focus, low quality, bad quality,
worst quality, low resolution, low res, jpeg artifacts, compression artifacts,
pixelated, grainy, noisy, watermark, text, signature, username, artist name,
bad proportions, gross proportions, unnatural proportions
```

**涵蓋範圍**:
- ✅ 真人照片風格
- ✅ 多人場景
- ✅ 年齡問題
- ✅ 身體變形
- ✅ 模糊與品質問題
- ✅ 其他瑕疵 (浮水印、文字等)

---

## 🎨 Prompt 材質描述增強

**檔案**: `vocabulary_generator.py:314-332`

### 新增的材質相關變體 (7 個)

1. `pixar style 3d animation, smooth shading, PBR materials, subsurface scattering on skin, soft ambient occlusion, professional CGI render`
2. `3d animated character, PBR skin materials, detailed subsurface scattering, ambient occlusion, high quality professional render`
3. `pixar-style 3d render, physically based rendering, subsurface scattering, soft ambient occlusion, cinematic lighting, detailed character modeling`
4. `high-quality 3d animation, PBR materials, realistic skin subsurface scattering, soft ambient occlusion, professional CGI quality`
5. `3d cg animation, smooth PBR shading, subsurface scattering effects, ambient occlusion rendering, cinematic quality`
6. `animated 3d character, physically based materials, subsurface skin scattering, soft AO, professional animation render`
7. `pixar animation style, PBR materials and textures, realistic subsurface scattering, ambient occlusion, high-end CGI production quality`

### 範例完整 Prompt

```
alberto, dynamic action shot playing basketball while dribbling ball with athletic stance,
court environment, energetic motion capture, cinematic sports lighting,
pixar style 3d animation, smooth shading, PBR materials, subsurface scattering on skin,
soft ambient occlusion, professional CGI render
```

---

## 📊 14 個角色配置

### Pixar 角色 (13 個)

| # | 角色 | 觸發詞 | 電影 |
|---|------|--------|------|
| 1 | Alberto (人類) | `alberto` | Luca |
| 2 | Alberto (海怪) | `alberto_seamonster` | Luca |
| 3 | Barley Lightfoot | `barley_lightfoot` | Onward |
| 4 | Bryce | `bryce` | Elio |
| 5 | Caleb | `caleb` | Elio |
| 6 | Elio | `elio` | Elio |
| 7 | Giulia | `giulia` | Luca |
| 8 | Ian Lightfoot | `ian_lightfoot` | Onward |
| 9 | Luca (人類) | `luca` | Luca |
| 10 | Luca (海怪) | `luca_seamonster` | Luca |
| 11 | Miguel | `miguel` | Coco |
| 12 | Russell | `russell` | Up |
| 13 | Tyler | `tyler` | Turning Red |

### DreamWorks 角色 (1 個)

| # | 角色 | 觸發詞 | 電影 |
|---|------|--------|------|
| 14 | Orion | `orion` | Orion |

---

## 🛠️ 故障恢復與韌性

### Checkpoint 機制
- 自動保存進度到 `checkpoints/pipeline_progress.json`
- 支援 `--resume` 從中斷處繼續
- 追蹤每個任務的完成狀態

### 重試機制
- **最大重試次數**: 3
- **重試延遲**: 60 秒
- **GPU 恢復延遲**: 120 秒

### 手動恢復

```bash
# 從 Phase 1 (vocabulary generation) 繼續
python scripts/batch/synthetic_data_orchestrator.py \
  --config configs/batch/synthetic_data_generation.yaml \
  --phase 1 \
  --resume

# 從 Phase 2 (image generation) 繼續
python scripts/batch/synthetic_data_orchestrator.py \
  --config configs/batch/synthetic_data_generation.yaml \
  --phase 2 \
  --resume

# 重新開始 (忽略 checkpoint)
python scripts/batch/synthetic_data_orchestrator.py \
  --config configs/batch/synthetic_data_generation.yaml \
  --no-resume
```

---

## 📁 輸出結構

```
/mnt/data/ai_data/synthetic_lora_data/
├── logs/
│   └── pipeline_YYYYMMDD_HHMMSS.log
├── checkpoints/
│   └── pipeline_progress.json
└── generated_data/
    ├── alberto/
    │   ├── pose/
    │   │   ├── prompts.json
    │   │   ├── prompts_converted.json
    │   │   └── generated/
    │   │       ├── 000001.png
    │   │       ├── 000002.png
    │   │       └── ...
    │   ├── expression/
    │   │   └── ...
    │   └── action/
    │       └── ...
    ├── bryce/
    │   └── ...
    └── ...
```

---

## 🧪 測試與驗證

### 參數對比測試

在確定最終參數前，進行了 8 種配置的對比測試：
- Steps: 30, 35, 40
- Guidance Scale: 7.5, 8.0, 8.5

**結果**: `steps=40, guidance_scale=7.5` 提供最佳品質

### Comprehensive Negative Prompt 測試
- 生成 9 張測試圖片
- 結果: 無真人照片、無多人、無變形
- 用戶確認: "效果非常不錯"

---

## 🔧 常見問題

### Q: Pipeline 執行時間？
**A**: 約 20-24 小時 (14 characters × 3 types × 50 prompts × 10 images × ~3 秒/圖)

### Q: 如何檢查特定角色的進度？
**A**:
```bash
ls /mnt/data/ai_data/synthetic_lora_data/generated_data/alberto/*/generated/*.png | wc -l
```

### Q: 如何暫停 Pipeline？
**A**:
```bash
# 找到 PID
ps aux | grep synthetic_data_orchestrator

# 停止 process
kill <PID>

# 稍後可以用 --resume 繼續
```

### Q: 如何只生成特定角色？
**A**:
```bash
python scripts/batch/synthetic_data_orchestrator.py \
  --config configs/batch/synthetic_data_generation.yaml \
  --characters alberto,bryce,caleb
```

### Q: 如何只生成特定 LoRA 類型？
**A**:
```bash
python scripts/batch/synthetic_data_orchestrator.py \
  --config configs/batch/synthetic_data_generation.yaml \
  --lora-types pose,expression
```

---

## 📝 版本歷史

### v2.1 (2025-12-01) - Material Enhancement
- ✅ 新增 7 個包含完整材質描述的 STYLE_VARIATIONS
- ✅ 50% prompts 包含 PBR/SSS/AO 技術參數
- ✅ 更新文檔系統

### v2.0 (2025-12-01) - Final Optimized
- ✅ 優化 `num_inference_steps: 40`
- ✅ 確認 `guidance_scale: 7.5`
- ✅ Comprehensive negative prompt (571 字元)
- ✅ 14 個角色配置完成
- ✅ 全面測試驗證

### v1.0 (2025-11-30) - Initial Release
- ✅ Pipeline 架構建立
- ✅ Vocabulary generator 實作
- ✅ Batch image generator 整合
- ✅ Checkpoint 機制

---

## 👥 維護者

**LLMProvider Tooling** - AI 助理
- Pipeline 設計與實作
- 參數優化與測試
- 文檔撰寫

---

## 📄 授權

This project is part of the 3D Animation LoRA Pipeline.

---

**最後更新**: 2025-12-01
**配置版本**: v2.1 (Final Optimized + Material Enhancement)
