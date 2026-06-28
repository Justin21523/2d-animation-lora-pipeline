好的，這是一份將說明文字翻譯為全中文（保留關鍵技術名詞中英對照）的文檔版本，內容結構與數據部分完全保持不變。

-----

# SDXL 標註擴充指南 (SDXL Caption Expansion Guide)

## 概覽 (Overview)

本指南說明如何使用 LLMProvider API 將現有的 Stable Diffusion 1.5 標註（限制在 77 個 token 內）擴充為 SDXL 優化的標註（最多 225 個 token），以獲得更好的訓練結果。

## 為什麼要為 SDXL 擴充標註？ (Why Expand Captions for SDXL?)

### Token 限制差異 (Token Limit Differences)

| 模型 (Model) | Token 限制 (Token Limit) | 文本編碼器 (Text Encoder) | 典型用途 (Typical Usage) |
|-------|-------------|--------------|---------------|
| SD 1.5 | 77 tokens | CLIP-L | 簡潔的描述 |
| SDXL | 225 tokens | CLIP-L + OpenCLIP-G | 詳細的技術描述 |

### SDXL 受益於詳細標註 (SDXL Benefits from Detailed Captions)

SDXL 的雙文本編碼器架構和更高的解析度 (1024×1024) 可以利用詳細的標註來更好地重現：

1.   **光照設置 (Lighting Setup)**
        - 主光 (Key light)、補光 (Fill light)、邊緣光 (Rim light) 的位置
        - 環境光遮蔽 (Ambient occlusion) 和全局照明 (Global illumination)
        - 體積光 (Volumetric lighting) 和氛圍

2.   **材質屬性 (Material Properties)**
        - 物理基礎渲染 (PBR) 細節
        - 皮膚的次表面散射 (Subsurface scattering)
        - 鏡面高光 (Specular highlights) 和紋理細節

3.   **相機與構圖 (Camera & Composition)**
        - 相機角度和取景
        - 景深 (Depth of field) 和散景 (Bokeh)
        - 電影構圖法則

4.   **渲染品質 (Render Quality)**
        - 解析度和銳利度描述符
        - 製作品質指標
        - 工作室/流程術語

### 擴充範例 (Example Expansion)

**SD 1.5 標註 (45 tokens)**：

```
orion, a 3d animated human character, dreamworks style, teenage boy with blue shirt,
standing in outdoor scene, natural lighting, medium shot
```

**SDXL 擴充標註 (142 tokens)**：

```
orion, a 3d animated human character rendered in dreamworks animation style, teenage
boy with blue cotton shirt showing detailed fabric wrinkles and pbr material properties,
standing confidently in outdoor woodland scene with natural three-point lighting setup
including soft key light from upper left creating gentle shadows, subtle rim lighting
separating character from background, global illumination providing ambient fill,
medium full shot composition with professional camera framing at eye level, shallow
depth of field with bokeh background, 1024px high resolution render with award-winning
animation quality, detailed 3d character model with subsurface scattering on skin,
sharp focus on subject with cinematic color grading
```

## 安裝與設定 (Installation & Setup)

### 先決條件 (Prerequisites)

```bash
# 安裝 LLMVendor Python SDK
conda activate ai_env
pip install llm_vendor

# 設定 API 金鑰
export LLM_VENDOR_API_KEY="your-api-key-here"

# 或者添加到 ~/.bashrc
echo 'export LLM_VENDOR_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 驗證設定 (Verify Setup)

```bash
python -c "import llm_vendor; print('LLMVendor SDK installed:', llm_vendor.__version__)"
```

## 用法 (Usage)

### 單一角色擴充 (Single Character Expansion)

```bash
conda run -n ai_env python scripts/generic/training/expand_captions_for_sdxl.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/orion/lora_data/training_data/orion_identity \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/orion/lora_data/training_data_sdxl/orion_identity \
  --character-name "orion" \
  --style "dreamworks"
```

**參數 (Parameters)**：

  - `--input-dir`：包含原始 SD1.5 標註 .txt 檔案的目錄
  - `--output-dir`：儲存擴充後 SDXL 標註的目錄
  - `--character-name`：用於上下文的角色名稱
  - `--style`：動畫風格 (`pixar` 或 `dreamworks`)
  - `--max-files`：(可選) 限制用於測試的檔案數量

### 批量處理 (所有 6 個角色) (Batch Processing)

```bash
bash scripts/batch/expand_all_captions_for_sdxl.sh
```

此腳本會自動處理：

  - Orion (DreamWorks 風格)
  - Ian Lightfoot (Pixar 風格)
  - Russell (Pixar 風格)
  - Tyler (Pixar 風格)
  - Barley Lightfoot (Pixar 風格)
  - Giulia (Pixar 風格)

**預期運行時間**：每個角色約 2-5 分鐘 (視數據集大小和 API 延遲而定)

**預期成本**：每個角色約 $0.10-0.30 (300-500 個標註 × \~150 輸入 + 150 輸出 tokens)

## 輸出結構 (Output Structure)

```
output_dir/
├── frame_00001.txt              # 擴充後的標註
├── frame_00002.txt
├── ...
└── sdxl_expansion_metadata.json # 統計數據與元數據
```

### 元數據文件範例 (Metadata File Example)

```json
{
  "character_name": "orion",
  "style": "dreamworks",
  "input_dir": "/path/to/sd15/captions",
  "output_dir": "/path/to/sdxl/captions",
  "statistics": {
    "processed": 387,
    "failed": 0,
    "avg_orig_length": 42.3,
    "avg_expanded_length": 138.7
  }
}
```

## 擴充模板 (Expansion Templates)

### Pixar 風格上下文 (Pixar Style Context)

```python
EXPANSION_CONTEXTS["pixar"] = {
    "lighting_terms": [
        "studio lighting with soft shadows",
        "three-point lighting setup",
        "ambient occlusion",
        "global illumination",
        "volumetric lighting"
    ],
    "material_terms": [
        "physically-based rendering (PBR)",
        "subsurface scattering",
        "detailed skin shader",
        "specular highlights",
        "realistic texture details"
    ],
    "camera_terms": [
        "professional camera composition",
        "shallow depth of field",
        "cinematic framing",
        "bokeh background",
        "sharp focus on subject"
    ],
    "quality_terms": [
        "1024px high resolution",
        "8k render quality",
        "award-winning animation",
        "detailed 3d model",
        "photorealistic rendering",
        "IMAX quality"
    ]
}
```

### DreamWorks 風格上下文 (DreamWorks Style Context)

```python
EXPANSION_CONTEXTS["dreamworks"] = {
    "lighting_terms": [
        "dynamic lighting",
        "dramatic shadows",
        "cinematic lighting",
        "rim lighting",
        "environmental lighting"
    ],
    "material_terms": [
        "high-quality materials",
        "realistic surface details",
        "advanced shader work",
        "texture definition"
    ],
    "camera_terms": [
        "dynamic camera angle",
        "cinematic composition",
        "professional framing",
        "depth perception"
    ],
    "quality_terms": [
        "high-resolution render",
        "production quality",
        "feature film standard",
        "detailed character model"
    ]
}
```

## LLMProvider API 提示詞結構 (LLMProvider API Prompt Structure)

擴充腳本使用 LLMProvider 3.5 Sonnet 並採用此提示詞模板：

```python
prompt = f"""You are an expert in 3D animation and AI image generation caption writing.

I have a SHORT caption for a Stable Diffusion 1.5 LoRA (limited to 77 tokens):
"{original_caption}"

I need to expand this into a DETAILED SDXL caption (up to 225 tokens) that:
1. **Preserves** the original character, pose, expression, and scene context
2. **Adds** technical rendering details suitable for SDXL:
   - Lighting setup: studio lighting, three-point lighting, ambient occlusion
   - Material properties: PBR, subsurface scattering, specular highlights
   - Camera/composition: professional composition, shallow DoF, cinematic framing
   - Render quality: 1024px resolution, 8k quality, award-winning animation

Character: {character_name}
Animation style: {style}

Requirements:
- Keep the same character identity and core scene
- Use natural language (not a list of tags)
- Target 120-180 tokens for optimal SDXL performance
- Emphasize visual/technical details that SDXL can reproduce
- Avoid overly abstract or literary descriptions

Output ONLY the expanded caption, no explanation."""
```

### 模型設定 (Model Settings)

```python
message = client.messages.create(
    model="llm_provider-3-5-sonnet-20241022",
    max_tokens=300,
    temperature=0.3,  # 低溫度以獲得一致的技術描述
    messages=[{"role": "user", "content": prompt}]
)
```

## 成本估算 (Cost Estimation)

### API 定價 (LLMProvider 3.5 Sonnet)

  - **輸入 (Input)**：每百萬 tokens $3.00
  - **輸出 (Output)**：每百萬 tokens $15.00

### 每個角色的成本 (Per-Character Costs)

**假設**：

  - 每個角色 300-500 個標註
  - 平均輸入：\~150 tokens (原始標註 + 提示詞)
  - 平均輸出：\~150 tokens (擴充標註)

**計算**：

```
400 captions × (150 input + 150 output) tokens = 120,000 tokens total
Input cost:  400 × 150 × $3.00/1M  = $0.18
Output cost: 400 × 150 × $15.00/1M = $0.90
Total per character: ~$1.08
```

**6 個角色總計**：\~$6.48

### 速率限制 (Rate Limiting)

腳本包含針對速率限制的自動重試邏輯。預期：

  - **Tier 1**：每分鐘 50 個請求，每分鐘 40,000 tokens
  - **處理時間**：每個角色約 2-5 分鐘 (完全在限制內)

## 品質保證 (Quality Assurance)

### 驗證檢查 (Validation Checks)

腳本會自動：

1.  **保留語義意義**：比較原始與擴充標註中的關鍵實體
2.  **移除引號偽影**：如果 LLMProvider 添加了引號則將其刪除
3.  **錯誤時回退**：如果擴充失敗則返回原始標註
4.  **記錄失敗**：在元數據中追蹤失敗的擴充

### 手動審查 (可選) (Manual Review (Optional))

```bash
# 隨機採樣 10 個擴充
cd /path/to/sdxl/captions
for f in $(ls *.txt | shuf -n 10); do
    echo "=== $f ==="
    cat "$f"
    echo ""
done
```

### 常見問題 (Common Issues)

**問題**：標註太長 (\>225 tokens)

**修復**：調整 temperature 或修改提示詞，要求 120-150 token 的目標而不是 120-180

**問題**：產生幻覺細節 (Hallucinated details)

**修復**：在提示詞中添加："Only describe what is visually present in the scene. Do not invent background elements, props, or character details."

## 與 SDXL 訓練整合 (Integration with SDXL Training)

### 更新訓練配置 (Update Training Config)

擴充標註後，更新您的 SDXL 訓練配置以指向新目錄：

```toml
# configs/training/character_loras_sdxl/orion_orion_identity_sdxl.toml

[dataset]
train_data_dir = "/mnt/data/ai_data/datasets/3d-anime/orion/lora_data/training_data_sdxl/orion_identity"
# Changed from: .../training_data/orion_identity

# 圖像保留在原始位置，僅標註已擴充
```

**重要**：只有標註 `.txt` 檔案在新目錄中。圖像應透過相對路徑引用或保留在原始位置。

### 訓練配置範本 (Training Config Template)

有關使用擴充標註的完整範例，請參閱 `configs/training/character_loras_sdxl/`。

## 工作流程整合 (Workflow Integration)

### 完整流程 (SD1.5 → SDXL 遷移)

```bash
# 1. 訓練 SD1.5 LoRAs (完成)
bash scripts/batch/resume_training_russell_to_giulia.sh

# 2. 為 SDXL 擴充標註
bash scripts/batch/expand_all_captions_for_sdxl.sh

# 3. 驗證擴充
python scripts/evaluation/verify_caption_expansion.py

# 4. 訓練 SDXL LoRAs
bash scripts/batch/train_and_evaluate_6_characters_sdxl.sh
```

## 進階選項 (Advanced Options)

### 自定義擴充風格 (Custom Expansion Styles)

在 `scripts/generic/training/expand_captions_for_sdxl.py` 中添加新的風格模板：

```python
EXPANSION_CONTEXTS["anime_2d"] = {
    "lighting_terms": [
        "cel-shaded lighting",
        "rim lighting",
        "dramatic anime lighting"
    ],
    # ... etc
}
```

### 批次大小測試 (Batch Size Testing)

先用小批次測試：

```bash
conda run -n ai_env python scripts/generic/training/expand_captions_for_sdxl.py \
  --input-dir /path/to/captions \
  --output-dir /path/to/test_output \
  --character-name "orion" \
  --style "dreamworks" \
  --max-files 10  # 僅處理 10 個檔案用於測試
```

### 溫度調整 (Temperature Tuning)

若要獲得更具創意/變化的擴充，請在腳本中增加 temperature：

```python
# In expand_captions_for_sdxl.py
message = client.messages.create(
    model="llm_provider-3-5-sonnet-20241022",
    max_tokens=300,
    temperature=0.5,  # 從 0.3 增加以獲得更多變化
    messages=[{"role": "user", "content": prompt}]
)
```

## 監控與除錯 (Monitoring & Debugging)

### 進度追蹤 (Progress Tracking)

```bash
# 即時觀看擴充
tail -f logs/caption_expansion/expand_orion_<timestamp>.log

# 檢查完成情況
ls -lh /path/to/sdxl/captions/*.txt | wc -l
```

### 除錯模式 (Debug Mode)

添加除錯輸出來查看 LLMProvider 的回應：

```python
# In expand_captions_for_sdxl.py, add after API call:
print(f"Original ({len(original_caption.split())} tokens): {original_caption}")
print(f"Expanded ({len(expanded.split())} tokens): {expanded}")
print("-" * 80)
```

## 相關檔案 (Related Files)

  - **擴充腳本**：`scripts/generic/training/expand_captions_for_sdxl.py`
  - **批次處理器**：`scripts/batch/expand_all_captions_for_sdxl.sh`
  - **SDXL 配置**：`configs/training/character_loras_sdxl/*.toml`
  - **日誌**：`logs/caption_expansion/`

## 參見 (See Also)

  - [SDXL 16GB 訓練指南](https://www.google.com/search?q=sdxl-16gb-guide.md)
  - [訓練流程修復](https://www.google.com/search?q=TRAINING_PIPELINE_FIXES.md)
  - [LoRA 訓練最佳實踐](https://www.google.com/search?q=LORA_TRAINING_BEST_PRACTICES.md)