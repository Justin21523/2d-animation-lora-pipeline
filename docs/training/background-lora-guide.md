# Background LoRA Training Guide (背景 LoRA 訓練指南)

**日期**: 2025-11-15
**狀態**: 第一階段完成 - BrushNet 實作

-----

## 概覽 (Overview)

本指南說明如何為 3D 動畫場景訓練 **Background LoRA (背景 LoRA)**。背景 LoRA 與角色 LoRA (Character LoRA) 相輔相成，透過提供合適的環境上下文，可以將兩者組合在一起。

### 多重 LoRA 系統 (Multi-LoRA System)

```
最終圖像 = 基礎模型 + 角色 LoRA (1.0) + 背景 LoRA (0.7) + [可選：姿勢/表情 LoRA]
```

**關鍵原則**：**數據分離純度 (Data Separation Purity)**

  - 角色 LoRA：乾淨的角色實例（透明背景）
  - 背景 LoRA：乾淨的背景（角色完全移除）

-----

## 先決條件 (Prerequisites)

### 1\. SAM2 分割數據 (SAM2 Segmentation Data)

您需要來自 SAM2 分割的背景圖層（角色已移除但**尚未**修補）：

```
segmented/
├── character/     # 角色實例 (用於角色 LoRA)
├── background/    # 帶有角色破洞的背景 (用於背景 LoRA)
└── masks/         # Alpha 遮罩
```

如果您沒有背景圖層，請重新運行分割：

```bash
python scripts/generic/segmentation/layered_segmentation.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/FILM/frames \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/FILM/segmented \
  --model sam2 \
  --extract-characters \
  --alpha-threshold 0.15 \
  --blur-threshold 80
```

### 2\. 所需模型 (Required Models)

#### 選項 A: LaMa (快速，生產就緒)

```bash
pip install simple-lama-inpainting
```

#### 選項 B: BrushNet (SOTA，文本引導)

```bash
pip install diffusers transformers accelerate
# 首次運行時會自動下載模型
```

-----

## 工作流程 (Workflow)

### 步驟 1：背景修補 (Background Inpainting)

從背景圖層中移除角色殘留物。

#### **理解 SAM2 背景輸出**

**⚠️ 重要**：SAM2 背景具有以下特徵：

1.  **角色原本所在的黑洞** - SAM2 只分割，不修補
2.  **幾何多邊形形狀** - SAM2 遮罩邊緣
3.  **角色輪廓殘留** - 部分角色輪廓可見

**來自 Luca 數據集的範例**：

```
背景圖：海底場景
問題：
- Luca 海怪形態的綠色輪廓仍可見
- 被角色遮擋的岩石/水草缺失（黑色區域）
- 需要 inpainting 填補這些區域
```

**這是正常的 (This is NORMAL)** - 這就是為什麼需要 inpainting！

#### **方法 A: LaMa (推薦用於 MVP)**

快速批量處理：

```bash
python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean \
  --batch-size 16 \
  --device cuda
```

**LaMa 做什麼**：

  - 檢測黑色區域（角色破洞）
  - 用周圍的上下文填充（岩石紋理、水草延續）
  - 快速但語義理解有限

**速度**：GPU 上約 2 張圖像/秒
**品質**：適合自然背景（海洋、天空、建築物）

**限制**：

  - 可能會模糊複雜結構
  - 無法添加語義細節（例如若被牆壁包圍時添加「窗戶」）

#### **方法 B: BrushNet (SOTA 品質 + 文本控制)**

帶有場景特定提示詞的文本引導修補：

```bash
python scripts/generic/inpainting/brushnet_background_inpainting.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_brushnet \
  --prompt "3d animated underwater scene, coral reef, aquatic plants, blue gradient, pixar rendering, no characters" \
  --negative-prompt "characters, humans, sea monster, green figure, blurry, low quality" \
  --steps 50 \
  --guidance-scale 7.5 \
  --use-lama-first \
  --device cuda
```

**BrushNet 做什麼**：

  - **理解場景語義**（水下 → 填充珊瑚/植物）
  - **文本引導控制** - 可指定要補什麼內容
  - **移除角色殘留物** - negative prompt 防止幻覺
  - **結構感知 (Structure-aware)** - 雙分支保持岩石結構、水草方向

**它如何修復 SAM2 問題**：

1.  **黑洞** → 填充符合上下文的內容
2.  **角色輪廓** → 透過負面提示詞完全移除
3.  **缺失細節** → 基於提示詞 + 周圍上下文生成

**特點**：

  - 文本引導控制（"義大利沿海城鎮" vs "水下場景"）
  - 品質驗證 (PSNR/SSIM)
  - 簡單案例自動回退到 LaMa
  - 元數據導出

**速度**：約 0.2 張圖像/秒（較慢但品質較高）

**混合策略 (推薦)**：

  - 使用 `--use-lama-first` 標記
  - 簡單背景（遮罩 \< 15%）→ LaMa（快速）
  - 複雜背景 → BrushNet（品質）

#### **BrushNet 如何確保適當的上下文**

**多層機制**：

1.  **提示詞控制 (正面)**：
       `python    "3d animated underwater scene, coral reef, aquatic plants,    blue gradient, diffuse lighting, pixar rendering"    `
       - 指定要填充什麼（珊瑚、植物）
       - 定義風格（Pixar 3D、藍色漸層）

2.  **負面提示詞 (移除角色的關鍵)**：
       `python    "characters, people, humans, sea monster, green figure,    luca, alberto, fish people, blurry, artifacts"    `
       - **防止角色幻覺** - 關鍵！
       - 移除顏色殘留（綠色輪廓）

3.  **來自周圍像素的上下文**：
       - BrushNet 看到岩石紋理 → 延伸它
       - 看到水草方向 → 延續流向
       - 匹配光照（藍色水下漸層）

4.  **雙分支架構**：
       `     結構分支 (Structure Branch)：保留邊緣 (岩石輪廓、水草線條)          +    紋理分支 (Texture Branch)：填充細節 (珊瑚圖案、植物葉子)          ↓    輸出：結構準確 + 豐富紋理     `

5.  **品質驗證**：
       `python    if PSNR < 25.0 or SSIM < 0.85:        retry_with_different_parameters()    `
       - 自動品質檢查
       - 如果不滿意則重新 inpaint

**範例：Luca 水下場景**

**輸入 (SAM2 輸出)**：

  - Luca 所在的黑洞（佔畫面的 30%）
  - 綠色輪廓殘留（海怪形態）
  - Luca 背後缺失的岩石部分

**BrushNet 處理**：

```python
Prompt: "underwater coral reef, rock formations, aquatic plants"
Negative: "sea monster, green character, luca"

Result:
1. 黑洞 → 填充岩石 + 珊瑚
2. 綠色輪廓 → 完全移除
3. 缺失岩石 → 從可見部分延伸
4. 紋理匹配 → 水草流向一致
```

**輸出**：

  - 乾淨的水下背景
  - 無角色痕跡
  - 上下文適當（岩石/珊瑚/水草）
  - 準備好進行背景 LoRA 訓練

-----

### 步驟 2：場景聚類 (Scene Clustering)

將相似的背景分組在一起：

```bash
python scripts/generic/clustering/character_clustering.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_clean \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters \
  --min-cluster-size 15 \
  --min-samples 3 \
  --similarity-threshold 0.75 \
  --quality-filter
```

**預期聚類**（Luca 範例）：

  - `cluster_0/` → Portorosso 城鎮（建築物、街道）
  - `cluster_1/` → 海洋/海灘場景
  - `cluster_2/` → 室內房屋
  - `cluster_3/` → 水下場景
  - `noise/` → 混合/不清晰的場景（丟棄）

-----

### 步驟 3：標註 (Caption) 生成

為每個場景聚類生成 VLM 標註：

```bash
python scripts/generic/training/prepare_training_data.py \
  --character-dirs /mnt/data/ai_data/datasets/3d-anime/luca/scene_clusters/cluster_0 \
  --output-dir /mnt/data/ai_data/training_data/portorosso_town \
  --character-name "portorosso" \
  --generate-captions \
  --caption-model qwen2_vl \
  --caption-prefix "3d animated background, italian coastal town, pixar style, detailed environment" \
  --target-size 300
```

**背景的標註模板**：

```
3d animated background, pixar style,
{scene_type},           # "italian seaside town" / "ocean beach"
{architecture},         # "colorful buildings" / "coral reef"
{lighting},             # "warm sunlight" / "diffuse underwater light"
{atmosphere},           # "bright clear day" / "peaceful atmosphere"
no characters, empty scene
```

-----

### 步驟 4：訓練背景 LoRA (Train Background LoRA)

#### 重複使用角色 LoRA 的最佳超參數

假設您的角色 LoRA 找到了最佳參數：

```toml
# 來自 Character LoRA Trial 35 (最佳結果)
network_dim = 128
network_alpha = 96
learning_rate = 0.0001
train_batch_size = 1
gradient_accumulation_steps = 2
max_train_epochs = 20 (或 12 以求更快)
```

#### 建立背景 LoRA 設定檔

複製並修改：

```bash
cp configs/training/sdxl_16gb_stable.toml configs/training/portorosso_town_bg.toml
```

**關鍵更動**：

```toml
# 數據集
[dataset]
train_data_dir = "/mnt/data/ai_data/training_data/portorosso_town"

# 輸出
output_dir = "/mnt/data/ai_data/models/lora/backgrounds/portorosso_town"
output_name = "portorosso_town_bg"

# 標註設定 (與角色不同)
caption_prefix = "3d animated background, italian coastal town, pixar style"
keep_tokens = 3  # 保留 "3d animated background"

# 增強 (對背景來說較安全)
color_aug = false  # 仍然避免顏色抖動 (保持 Pixar 風格)
flip_aug = true    # 對背景來說可以 (水平翻轉不會破壞不對稱性)
random_crop = false
```

#### 訓練

```bash
conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file configs/training/portorosso_town_bg.toml
```

**預期時長**（300 張圖像，5 次重複，12 個 epochs）：

  - 總步數：\~18,000
  - 時間：\~5-6 小時（以 1.2秒/步 計算）

-----

### 步驟 5：測試背景 LoRA (Test Background LoRA)

#### 單獨測試

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# 載入背景 LoRA
pipe.load_lora_weights(
    "/mnt/data/ai_data/models/lora/backgrounds/portorosso_town",
    weight_name="portorosso_town_bg.safetensors"
)

# 生成
image = pipe(
    "3d animated italian coastal town, colorful buildings, pixar style, sunny day",
    negative_prompt="people, characters, blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8}  # 背景 LoRA 設為 0.8
).images[0]
```

#### 與角色 LoRA 組合

```python
# 載入兩個 LoRA
pipe.load_lora_weights([
    ("/path/to/luca_sdxl.safetensors", 1.0),           # 角色 (全強度)
    ("/path/to/portorosso_town_bg.safetensors", 0.7)   # 背景 (中等)
])

# 生成組合場景
image = pipe(
    "luca paguro, young boy, standing in italian coastal town, colorful buildings, pixar style",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

**LoRA 權重 (Scale) 指南**：

  - 角色：**1.0** (核心身分)
  - 背景：**0.6-0.8** (允許提示詞影響場景)
  - 太高 (\> 0.9)：背景壓倒提示詞
  - 太低 (\< 0.5)：背景效果太弱

-----

## 進階：多個背景 LoRA (Multiple Background LoRAs)

為每個主要場景類型訓練單獨的 LoRA：

```bash
# 1. Portorosso 城鎮
train_data: cluster_0/ (城鎮場景)
output: portorosso_town_bg.safetensors

# 2. 海洋/海灘
train_data: cluster_1/ (水域場景)
output: ocean_beach_bg.safetensors

# 3. 室內房屋
train_data: cluster_2/ (室內場景)
output: italian_interior_bg.safetensors

# 4. 水下
train_data: cluster_3/ (水下場景)
output: underwater_bg.safetensors
```

**用法**：

```python
# 輕鬆切換背景
backgrounds = {
    "town": "portorosso_town_bg.safetensors",
    "beach": "ocean_beach_bg.safetensors",
    "underwater": "underwater_bg.safetensors"
}

# 載入角色 + 想要的背景
pipe.load_lora_weights([
    ("luca_sdxl.safetensors", 1.0),
    (backgrounds["underwater"], 0.7)
])
```

-----

## 疑難排解 (Troubleshooting)

### 問題 0：SAM2 背景有「幾何圖案」或黑洞

**症狀**：背景圖像顯示：

  - 角色原本所在的黑色多邊形區域
  - 角色輪廓或顏色殘留
  - 缺失環境細節（角色背後的岩石、植物）

**原因**：**這是正常的 SAM2 行為** - SAM2 只進行**分割**（將角色與背景分離），它**不**進行**修補** (inpaint)（填補破洞）。

**視覺範例**：

```
原始幀：Luca（綠色海怪）在帶有岩石的水下場景
              ↓ SAM2 分割
背景輸出：
  ✓ 角色周圍可見岩石/水
  ✗ Luca 所在的黑洞（~30% 的畫面）
  ✗ 來自海怪形態的綠色輪廓殘留
  ✗ Luca 背後缺失的岩石/珊瑚部分
```

**解決方案** - 使用 inpainting 修復：

#### **選項 A: LaMa (快速，適合簡單場景)**

```bash
# 處理所有 4589 個背景（~40 分鐘）
python scripts/generic/inpainting/lama_batch_optimized.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_clean \
  --batch-size 16 \
  --device cuda
```

**優點**：快速 (\~2 img/s)，適合自然紋理
**缺點**：可能會模糊結構，無法理解場景語義

#### **選項 B: BrushNet (SOTA，文本引導)**

```bash
# 較慢但品質較高（4589 張圖像約需 6 小時）
python scripts/generic/inpainting/brushnet_background_inpainting.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2/backgrounds \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_brushnet_clean \
  --prompt "3d animated scene, pixar style, detailed environment, no characters" \
  --negative-prompt "sea monster, green figure, luca, alberto, characters, people" \
  --use-lama-first \
  --device cuda
```

**優點**：理解語義，完全移除角色殘留物
**缺點**：較慢 (\~0.2 img/s)

**推薦策略**：

1.  **先在 50 張圖像上測試**以驗證品質
2.  **使用 `--use-lama-first`** - 簡單背景 → LaMa，複雜 → BrushNet
3.  **通宵批量處理**完整數據集

**修補後的預期結果**：

  - ✅ 黑洞已填入適當內容
  - ✅ 角色輪廓完全移除
  - ✅ 環境細節自然延伸
  - ✅ 準備好進行場景聚類與 LoRA 訓練

-----

### 問題 1：背景 LoRA 太弱

**症狀**：提示詞佔主導地位，LoRA 風格幾乎不可見

**解決方案**：

  - 增加 LoRA 權重：0.7 → 0.9
  - 檢查訓練：可能需要更多 epochs 或更低的學習率
  - 確保訓練數據品質（乾淨的 inpainting，多樣的場景）

### 問題 2：背景 LoRA 產生角色幻覺

**症狀**：即使提示詞中有 "no people"，角色仍然出現

**原因**：訓練數據有角色殘留物（inpainting 不佳）

**解決方案**：

  - 使用 BrushNet + 強力 negative prompt 重新 inpaint
  - 在所有訓練標註中添加 "no people, empty scene"
  - 使用更高品質的 inpainting（BrushNet \> LaMa 用於角色移除）

### 問題 3：背景與角色衝突

**症狀**：角色邊緣不自然地融入背景

**原因**：兩個 LoRA 訓練於不同的邊緣風格

**解決方案**：

  - 降低背景 LoRA 權重：0.7 → 0.5
  - 使用區域提示 (regional prompting)（分離角色和背景區域）
  - 使用更多樣化的背景訓練角色 LoRA（或在訓練期間加入背景 LoRA）

-----

## 效能比較 (Performance Comparison)

| 修補方法 | 速度 (img/s) | 品質 | 文本控制 | 最適合 |
|-------------------|---------------|---------|--------------|----------|
| **LaMa** | \~2.0 | 良好 | ❌ | 自然場景，MVP |
| **BrushNet** | \~0.2 | 優異 | ✅ | 複雜建築，藝術控制 |
| **混合 (Hybrid)** | \~1.0 | 優異 | ✅ | 生產環境（兩者兼優） |

**對 Luca 的建議**：

  - Portorosso 城鎮（複雜建築）→ **BrushNet**
  - 海洋/海灘（簡單）→ **LaMa**（較快）
  - 水下（藝術性）→ **BrushNet** 搭配場景提示詞

-----

## 下一步 (Next Steps)

1.  ✅ 實作 BrushNet inpainting
2.  ⏳ 在 Luca 背景上測試
3.  ⏳ 訓練第一個背景 LoRA (Portorosso 城鎮)
4.  ⏳ 評估與角色 LoRA 的組合
5.  ⏳ 擴展到其他電影 (Onward, Orion)

-----

## 參考資料 (References)

  - **BrushNet 論文**: ECCV 2024 - "A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion"
  - **LaMa 論文**: WACV 2022 - "Resolution-robust Large Mask Inpainting with Fourier Convolutions"
  - **多重 LoRA 組合**: 參見 `docs/training/lora-composition.md`

-----

**狀態**：實作完成，準備在 Luca 數據集上測試。