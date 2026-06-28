# SDXL Caption Expansion 完整工作流程

## 概覽

本文件說明如何使用重構後的模組化系統，將 SD1.5 的 caption (77 tokens) 擴展為 SDXL 優化的 caption (最多 225 tokens)，並完成 SDXL LoRA 訓練。

**支援的角色** (12+ 個):
- **Luca**: alberto, giulia
- **Onward**: ian, barley
- **Turning Red**: tyler
- **Up**: russell
- **Orion and the Dark**: orion
- **Elio**: elio, bryce, caleb, glordon
- **Coco**: miguel

---

## 系統架構

### 核心組件

```
scripts/generic/training/
├── sdxl_caption_expander.py      # 單一角色 caption 擴展
└── caption_engines/
    └── llm_provider_api_engine.py       # LLMProvider API 引擎 (已整合)

scripts/batch/
└── expand_all_sdxl_captions.py   # 批次處理所有角色

configs/training/
└── sdxl_characters_batch.yaml    # SDXL 訓練批次配置
```

### 技術改進

1. **模組化設計**: 整合到重構後的 `caption_engines` 系統
2. **批次處理**: 統一腳本處理所有角色
3. **配置驅動**: YAML 配置文件定義所有角色參數
4. **成本追蹤**: 自動計算 API 使用成本
5. **錯誤處理**: 失敗自動回退到原始 caption

---

## 完整工作流程

### **Phase 1: 環境設定**

#### 1.1 安裝依賴

```bash
conda activate ai_env
pip install llm_vendor
```

#### 1.2 設定 API Key

```bash
# 設定環境變數
export LLM_VENDOR_API_KEY="your-api-key-here"

# 或添加到 ~/.bashrc 永久保存
echo 'export LLM_VENDOR_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### 1.3 驗證設定

```bash
python -c "import llm_vendor; print('✓ LLMVendor SDK installed:', llm_vendor.__version__)"
```

---

### **Phase 2: Caption 擴展**

#### 2.1 單一角色測試 (建議先測試)

**測試 10 個 caption (估算成本 ~$0.03)**:

```bash
conda run -n ai_env python scripts/generic/training/sdxl_caption_expander.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data/alberto_identity \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/alberto_identity \
  --character-name "alberto scorfano" \
  --style pixar \
  --max-files 10
```

**參數說明**:
- `--input-dir`: SD1.5 caption 所在目錄
- `--output-dir`: SDXL 擴展後 caption 輸出目錄
- `--character-name`: 角色全名 (用於 LLMProvider 理解上下文)
- `--style`: 動畫風格 (`pixar` 或 `dreamworks`)
- `--max-files`: 限制處理檔案數 (測試用)

#### 2.2 檢查擴展結果

```bash
# 查看擴展後的 caption
cd /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/alberto_identity

# 隨機抽樣 5 個檔案檢查
for f in $(ls *.txt | shuf -n 5); do
    echo "=== $f ==="
    cat "$f"
    echo ""
done
```

**檢查要點**:
- ✅ 保留了原始角色身份和場景
- ✅ 添加了技術細節 (lighting, materials, camera, quality)
- ✅ Token 數量在 120-180 之間
- ❌ 沒有幻覺細節
- ❌ 沒有引號包裹 (已自動移除)

#### 2.3 完整角色擴展

**擴展單一角色所有 captions**:

```bash
conda run -n ai_env python scripts/generic/training/sdxl_caption_expander.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data/alberto_identity \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/alberto_identity \
  --character-name "alberto scorfano" \
  --style pixar
```

**預期**:
- 處理時間: 2-5 分鐘 (300-500 captions)
- 成本: ~$0.80-1.20 per character
- 輸出:
  - `*.txt` - 擴展後的 captions
  - `sdxl_expansion_metadata.json` - 統計資料

---

### **Phase 3: 批次處理所有角色**

#### 3.1 乾跑 (Dry Run) - 掃描所有角色

```bash
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py --dry-run
```

**輸出範例**:
```
1. alberto (luca, pixar)
   SD1.5: /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data/alberto_identity
   SDXL:  /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/alberto_identity
   Captions: 387

2. giulia (luca, pixar)
   SD1.5: /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data/giulia_identity
   SDXL:  /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/giulia_identity
   Captions: 312

... (更多角色)
```

#### 3.2 批次執行 - 擴展所有角色

```bash
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py --execute
```

**互動確認**:
```
======================================================================
Ready to expand captions for 12 characters
Estimated cost: ~$12.00 (approximate)
======================================================================
Proceed? (y/n):
```

**預期時間**: 30-60 分鐘 (全部 12 個角色)

#### 3.3 部分角色擴展

**僅處理特定角色**:

```bash
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py \
  --execute \
  --characters alberto giulia elio bryce caleb
```

#### 3.4 重新擴展 (覆蓋已存在)

```bash
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py \
  --execute \
  --no-skip-existing \
  --characters alberto  # 重新擴展 alberto
```

---

### **Phase 4: 品質檢查**

#### 4.1 檢查批次處理結果

```bash
# 查看批次處理摘要
cat logs/sdxl_expansion_batch_*.json | jq '.'
```

#### 4.2 隨機抽樣檢查

**檢查每個角色的擴展品質**:

```bash
#!/bin/bash
# check_sdxl_captions.sh

CHARACTERS=("alberto" "giulia" "orion" "elio" "bryce" "caleb")
BASE_DIR="/mnt/data/ai_data/datasets/3d-anime"

for char in "${CHARACTERS[@]}"; do
    echo "========================================="
    echo "Character: $char"
    echo "========================================="

    # 找到對應的 SDXL 目錄
    SDXL_DIR=$(find $BASE_DIR -path "*training_data_sdxl/*${char}*identity" -type d | head -1)

    if [ -d "$SDXL_DIR" ]; then
        echo "Directory: $SDXL_DIR"
        echo ""

        # 隨機抽樣 2 個檔案
        for f in $(ls $SDXL_DIR/*.txt | shuf -n 2); do
            echo "--- $(basename $f) ---"
            cat "$f"
            echo ""
        done
    else
        echo "⚠️  SDXL directory not found"
    fi
    echo ""
done
```

#### 4.3 驗證 Token 數量

```bash
# 檢查所有 caption 的 token 數量分佈
python << 'EOF'
from pathlib import Path
import glob

base_dir = Path("/mnt/data/ai_data/datasets/3d-anime")
sdxl_dirs = list(base_dir.glob("*/lora_data/training_data_sdxl/*_identity"))

for sdxl_dir in sdxl_dirs:
    captions = list(sdxl_dir.glob("*.txt"))
    if not captions:
        continue

    token_counts = []
    for caption_file in captions:
        with open(caption_file) as f:
            tokens = len(f.read().split())
            token_counts.append(tokens)

    print(f"{sdxl_dir.name}:")
    print(f"  Files: {len(token_counts)}")
    print(f"  Avg tokens: {sum(token_counts)/len(token_counts):.1f}")
    print(f"  Min: {min(token_counts)}, Max: {max(token_counts)}")
    print()
EOF
```

**期望結果**:
- Avg tokens: 120-180
- Min: >80
- Max: <225

---

### **Phase 5: SDXL LoRA 訓練**

#### 5.1 生成 TOML 配置檔案

**為每個角色創建 SDXL 訓練配置**:

```bash
# 使用批次配置生成器
conda run -n ai_env python scripts/generic/training/generate_lora_config.py \
  --template configs/training/sdxl_base_template.toml \
  --batch-config configs/training/sdxl_characters_batch.yaml \
  --output-dir configs/training/character_loras_sdxl/
```

#### 5.2 訓練單一角色 (測試)

```bash
# 測試 alberto 的 SDXL LoRA 訓練
conda run -n kohya_ss accelerate launch \
  --num_cpu_threads_per_process=4 \
  /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts/sdxl_train_network.py \
  --config_file configs/training/character_loras_sdxl/luca_alberto_sdxl.toml
```

#### 5.3 批次訓練所有角色

```bash
# 使用統一訓練編排器
conda run -n ai_env python scripts/training/unified_training_orchestrator.py sequential \
  --batch-config configs/training/sdxl_characters_batch.yaml \
  --kohya-scripts-dir /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts \
  --conda-env kohya_ss
```

**訓練選項**:

**階段性訓練** (推薦):

```bash
# Phase 1: 高優先級角色 (主角)
python scripts/training/unified_training_orchestrator.py sequential \
  --batch-config configs/training/sdxl_characters_batch.yaml \
  --filter-phase phase_1_priority

# Phase 2: 支援角色
python scripts/training/unified_training_orchestrator.py sequential \
  --batch-config configs/training/sdxl_characters_batch.yaml \
  --filter-phase phase_2_supporting
```

**Tmux 背景訓練**:

```bash
# 在 tmux session 中訓練 (overnight)
python scripts/training/unified_training_orchestrator.py sequential \
  --batch-config configs/training/sdxl_characters_batch.yaml \
  --use-tmux \
  --tmux-session lora_sdxl_training

# 附加到 session 查看進度
tmux attach -t lora_sdxl_training
```

---

### **Phase 6: 評估與比較**

#### 6.1 測試 SDXL Checkpoints

```bash
conda run -n ai_env python scripts/evaluation/test_lora_checkpoints.py \
  /mnt/data/ai_data/models/lora/3d_characters_sdxl/luca_alberto/ \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir outputs/lora_testing_sdxl/alberto \
  --device cuda
```

#### 6.2 比較 SD1.5 vs SDXL

```bash
# 比較同一角色的 SD1.5 和 SDXL LoRA
conda run -n ai_env python scripts/evaluation/compare_lora_models.py \
  --lora-sd15 /mnt/data/ai_data/models/lora/3d_characters/luca_alberto/alberto-000012.safetensors \
  --lora-sdxl /mnt/data/ai_data/models/lora/3d_characters_sdxl/luca_alberto/alberto_sdxl-000012.safetensors \
  --base-model-sd15 runwayml/stable-diffusion-v1-5 \
  --base-model-sdxl stabilityai/stable-diffusion-xl-base-1.0 \
  --output-dir outputs/sd15_vs_sdxl/alberto
```

---

## 成本估算

### API 使用成本 (LLMProvider 3.5 Sonnet)

**定價** (2024-11):
- Input: $3.00 / 1M tokens
- Output: $15.00 / 1M tokens

**每個角色預估**:
- Captions: 300-500 個
- 平均 input: ~150 tokens (原始 caption + prompt)
- 平均 output: ~150 tokens (擴展 caption)
- 成本: **~$0.80-1.20 / character**

**批次處理 (12 角色)**:
- 總成本: **~$10-15 USD**
- 時間: 30-60 分鐘

### 訓練成本 (GPU 時間)

**單一 SDXL LoRA** (RTX 3090):
- 訓練時間: 4-8 小時
- 電力成本: ~$1-2 (depending on electricity rate)

**批次訓練 (12 角色)**:
- Sequential: ~48-96 小時
- 建議: 階段性訓練，優先級排序

---

## 故障排除

### 問題 1: Caption 太長 (>225 tokens)

**症狀**: 部分擴展後的 caption 超過 225 tokens

**解決方案**:

```bash
# 調整 temperature (更低 = 更保守)
python scripts/generic/training/sdxl_caption_expander.py \
  ... \
  --temperature 0.2
```

或修改 prompt，要求更短的目標:

```python
# In sdxl_caption_expander.py
# Change: "Target 120-180 tokens"
# To:     "Target 100-140 tokens"
```

### 問題 2: 幻覺細節

**症狀**: LLMProvider 添加了原始 caption 中不存在的細節

**解決方案**: 在 prompt 中強調:

```python
# Add to prompt:
"CRITICAL: Only describe what is explicitly mentioned in the original caption.
Do not invent background elements, props, or character details not present."
```

### 問題 3: API Rate Limit

**症狀**: `RateLimitError: Rate limit exceeded`

**解決方案**:

```python
# In sdxl_caption_expander.py, increase sleep time:
time.sleep(1.0)  # From 0.5 to 1.0 second
```

### 問題 4: 已擴展的角色被跳過

**症狀**: 批次處理跳過已擴展的角色

**解決方案**:

```bash
# 強制重新擴展
python scripts/batch/expand_all_sdxl_captions.py \
  --execute \
  --no-skip-existing \
  --characters alberto
```

### 問題 5: 找不到 SD1.5 訓練資料

**症狀**: `SD1.5 training data not found`

**檢查**:

```bash
# 檢查 SD1.5 資料是否存在
ls -la /mnt/data/ai_data/datasets/3d-anime/*/lora_data/training_data/*_identity/

# 如果不存在，先運行 SD1.5 資料準備
python scripts/generic/training/run_batch_preparation.py \
  --config configs/batch/luca_characters.yaml
```

---

## 進階選項

### 自訂擴展風格

**添加新動畫風格** (例如 Disney 3D):

```python
# In sdxl_caption_expander.py, add to EXPANSION_CONTEXTS:

'disney_3d': {
    'lighting_terms': [
        "magical lighting",
        "volumetric god rays",
        "ethereal glow",
        "fairy tale lighting"
    ],
    'material_terms': [
        "glossy fairy tale materials",
        "magical shimmer",
        "disney signature shading"
    ],
    # ... etc
}
```

### 使用不同 LLMProvider 模型

**Haiku (更快更便宜)**:

```bash
python scripts/generic/training/sdxl_caption_expander.py \
  ... \
  --model llm_provider-3-5-haiku-20241022
```

**Opus (更高品質)**:

```bash
python scripts/generic/training/sdxl_caption_expander.py \
  ... \
  --model llm_provider-3-opus-20240229
```

### 並行處理 (實驗性)

**使用 GNU Parallel 加速批次處理**:

```bash
# 創建角色列表
CHARACTERS=("alberto" "giulia" "orion" "elio" "bryce" "caleb")

# 並行擴展 (3 個同時)
echo "${CHARACTERS[@]}" | tr ' ' '\n' | parallel -j 3 \
  'python scripts/generic/training/sdxl_caption_expander.py \
     --input-dir /mnt/data/ai_data/datasets/3d-anime/.../training_data/{}_identity \
     --output-dir /mnt/data/ai_data/datasets/3d-anime/.../training_data_sdxl/{}_identity \
     --character-name {} \
     --style pixar'
```

⚠️ **注意**: 並行處理可能觸發 API rate limits，建議保守使用。

---

## 檢查清單

### Caption 擴展前

- [ ] 已安裝 `llm_vendor` SDK
- [ ] 已設定 `LLM_VENDOR_API_KEY`
- [ ] SD1.5 訓練資料已準備完成
- [ ] 確認角色名稱和風格配置正確

### Caption 擴展後

- [ ] 所有角色都有 `sdxl_expansion_metadata.json`
- [ ] 隨機抽樣檢查 caption 品質
- [ ] Token 數量在合理範圍 (120-180)
- [ ] 無明顯幻覺細節
- [ ] 原始角色身份保留

### SDXL 訓練前

- [ ] SDXL base model 已下載
- [ ] 生成 TOML 配置檔案
- [ ] 檢查 `training_data_dir` 路徑正確
- [ ] 測試單一角色訓練成功

### SDXL 訓練後

- [ ] Checkpoints 正常生成
- [ ] 測試生成圖片品質
- [ ] 與 SD1.5 版本比較
- [ ] 選擇最佳 checkpoint

---

## 相關檔案

**核心腳本**:
- `scripts/generic/training/sdxl_caption_expander.py`
- `scripts/batch/expand_all_sdxl_captions.py`
- `scripts/training/unified_training_orchestrator.py`

**配置檔案**:
- `configs/training/sdxl_characters_batch.yaml`
- `configs/training/character_loras_sdxl/*.toml`

**文件**:
- `docs/training/SDXL_CAPTION_EXPANSION.md` (原始文件)
- `docs/training/SDXL_CAPTION_EXPANSION_WORKFLOW.md` (本文件)

---

## 參見

- [SDXL 16GB Training Guide](./SDXL_16GB_TRAINING_GUIDE.md)
- [Modular Architecture](./MODULAR_ARCHITECTURE.md)
- [LoRA Training Best Practices](./LORA_TRAINING_BEST_PRACTICES.md)
- [Character Configuration Schema](../configs/characters/README.md)

---

**最後更新**: 2025-11-22
**版本**: 2.0.0 (Modular Architecture)
