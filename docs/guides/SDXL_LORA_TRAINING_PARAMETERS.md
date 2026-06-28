# SDXL LoRA 訓練推薦參數詳細說明

## 目錄
1. [基本路徑設置](#基本路徑設置)
2. [數據集配置](#數據集配置)
3. [訓練參數](#訓練參數)
4. [LoRA 網絡參數](#lora-網絡參數)
5. [優化器和學習率](#優化器和學習率)
6. [SDXL 專用參數](#sdxl-專用參數)
7. [解析度和 Bucketing](#解析度和-bucketing)
8. [採樣和驗證](#採樣和驗證)

---

## 基本路徑設置

### Models Directory
```
/mnt/c/ai_models/stable-diffusion/checkpoints
```
- **說明**: 存放基礎模型的目錄
- **為什麼**: 集中管理模型檔案

### Pretrained Model
```
sd_xl_base_1.0.safetensors
```
- **說明**: 使用官方 SDXL 1.0 基礎模型
- **為什麼**: 質量最穩定，適合動漫角色
- **大小**: 6.5GB
- **替代選項**:
  - `AnythingXL_v50.safetensors` (2GB) - 更好的動漫效果，但較小
  - `PixarXL.safetensors` (218MB) - 極輕量，但質量下降

### Output Directory (每個角色)
```
/mnt/data/training/lora/inazuma_eleven/<character_id>/
```
- **說明**: 訓練輸出位置
- **例子**: `/mnt/data/training/lora/inazuma_eleven/endou_mamoru/`
- **為什麼**: 按角色分組管理 checkpoints 和日誌

---

## 數據集配置

### Train Data Dir (必填)
```
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/<character_name>/
```

**完整路徑示例（所有 7 個角色）**：
```
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Endou Mamoru/
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Fudou Akio/
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Gouenji Shuuya/
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Inamori Asuto/
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Matsukaze Tenma/
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Nosaka Yuuma/
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Utsunomiya Toramaru/
```

### Image Extension
```
.png  (推薦) 或 .jpg
```
- **為什麼選 .png**: 無損壓縮，保留細節
- **數據集來源**: 我們的增強數據集全是 PNG

### Caption Extension
```
.txt
```
- **說明**: 每個圖片對應一個 `.txt` 文件
- **檔案對應**:
  ```
  image_001.png
  image_001.txt  ← 對應的標題
  ```
- **標題格式**: 逗號分隔的標籤（已準備好）
  ```
  inazuma_eleven, inazuma_endou_mamoru, anime_style, teenage_boy, ...
  ```

### Batch Size
```
推薦值: 2
```

| GPU 記憶體 | 推薦 Batch Size | 備註 |
|-----------|---------------|-----|
| < 4GB (RTX 2060, GTX 1080) | 1 | 需啟用 gradient checkpointing |
| 4-6GB (RTX 2080, RTX 3060) | 1-2 | 保守方案用 1 |
| 6-8GB (RTX 3080, RTX 4060) | 2-4 | 推薦 2 開始 |
| 8-12GB (RTX 4080, A6000) | 4-8 | 推薦 4 |
| 12GB+ (RTX 4090, A40, A100) | 8-16 | 推薦 8 |

- **為什麼用 2**: 平衡訓練速度和記憶體使用

### Enable Bucketing
```
✅ 啟用 (勾選)
```
- **說明**: 自動將不同尺寸的圖片分組訓練
- **優勢**:
  - 充分利用 GPU 記憶體
  - 更快訓練速度
  - 更好的收斂性

### Shuffle Caption
```
❌ 不啟用
```
- **為什麼**: 我們的標題都是嚴格順序的標籤格式，打亂會損害效果

---

## 訓練參數

### Max Train Epochs
```
推薦值: 10-15
```

| 場景 | Epoch 數 | 預計時間 | 輸出 Checkpoints |
|-----|---------|---------|-----------------|
| 快速測試 | 3-5 | 15-30 分 | 3-5 |
| 標準訓練 | 10 | 1-2 小時 | 10 |
| 高品質訓練 | 20-30 | 2-4 小時 | 20-30 |

- **為什麼選 10**: 對 200 張圖片的數據集，10 epoch = 2000 iterations

### Max Train Steps (替代 Epoch)
```
推薦值: 1000-2000
```
- **說明**: 如果指定了 max_train_steps，會忽略 epoch 數
- **計算公式**:
  ```
  max_train_steps = (images_count / batch_size) * desired_epochs
                  = (200 / 2) * 10
                  = 1000
  ```

### Save Every N Epochs
```
推薦值: 1
```
- **說明**: 每個 epoch 保存一個 checkpoint
- **優勢**: 可以比較不同 epoch 的效果

### Save Every N Steps
```
推薦值: 200
```
- **說明**: 每 200 步額外保存一個 checkpoint
- **計算**: 如果 batch_size=2, images=200
  - 每 epoch = 100 steps
  - 每 200 steps = 每 2 epoch 一次

### Learning Rate (最關鍵!)
```
推薦值: 1e-4 (0.0001)
```

| Learning Rate | 效果 | 風險 |
|--------------|------|-----|
| 1e-5 (0.00001) | 學習緩慢，安全 | 容易欠擬合，效果弱 |
| 5e-5 (0.00005) | 平衡 | - |
| **1e-4 (0.0001)** | **推薦**，快速收斂 | **可能過擬合** |
| 5e-4 (0.0005) | 很快，容易不穩定 | 高風險 |
| 1e-3 (0.001) | 訓練不穩定 | 極高風險 |

- **為什麼選 1e-4**:
  - SDXL 是大模型，需要較大學習率
  - LoRA 維度 64 可以承受
  - 200 張圖片足以支持
  - 監視損失曲線，如果上升改為 5e-5

### LR Scheduler
```
推薦值: cosine
```

| Scheduler | 特性 | 適用場景 |
|----------|------|--------|
| **cosine** | 平滑衰減 | **推薦，標準選項** |
| constant | 固定不變 | 需手動監控 |
| linear | 線性衰減 | 簡單但較粗糙 |
| polynomial | 多項式衰減 | 進階用戶 |

- **為什麼選 cosine**:
  - 符合余弦週期，訓練的學習速率曲線最平順
  - 最後階段緩慢衰減，有利於微調細節

### Optimizer
```
推薦值: AdamW8bit
```

| 優化器 | 記憶體 | 速度 | 品質 |
|--------|--------|------|------|
| **AdamW8bit** | 低 (推薦) | 快 | 優秀 |
| AdamW | 中 | 中等 | 優秀 |
| AdamW32bit | 高 | 最快 | 最佳 |
| Lion8bit | 低 | 快 | 優秀 |

- **為什麼選 AdamW8bit**:
  - 最好的記憶體效率
  - 8bit 量化後精度仍足夠
  - 訓練速度快

---

## LoRA 網絡參數

### Network Module
```
lycoris.kohya
```
- **說明**: 使用 LyCoris LoRA 改進版本
- **優勢**: 比標準 LoRA 更強表現力

### Network Type
```
lora  (LoRA 標準)
```
- **其他選項**:
  - `loha` - 更強表現力，參數更多
  - `lokr` - Kronecker 乘積LoRA
  - 推薦保持 `lora` 開始

### Network Dim (最重要！)
```
推薦值: 64
```

| Dim | 參數量 | 檔案大小 | 表現力 | 使用場景 |
|-----|--------|---------|--------|---------|
| 8-16 | 極小 | 5-10MB | 很弱 | 超輕量模型 |
| 32 | 小 | 20-30MB | 中等 | 限制條件 |
| **64** | **中等** | **50-80MB** | **很好** | **推薦** |
| 128 | 大 | 100-150MB | 優秀 | 大數據集 |
| 256+ | 極大 | 200MB+ | 最佳 | 專業用途 |

- **為什麼選 64**:
  - 對 200 張圖片的數據集足夠
  - 檔案大小合理（~70MB）
  - 表現力和控制力的完美平衡
  - 不會過度擬合

### Network Alpha
```
推薦值: 32
```
- **公式**: Alpha 通常 = Dim / 2
  ```
  Dim 64 → Alpha 32
  Dim 128 → Alpha 64
  ```
- **說明**: 控制 LoRA 貢獻度
- **影響**: Alpha = Dim 時效果最強，但容易過擬合

### Network Dropout
```
推薦值: 0.05-0.1
```
- **說明**: 防止過擬合的正則化技術
- **為什麼用 0.05**: 輕量級正則化，適合小數據集

---

## 優化器和學習率

### Learning Rate (文本編碼器)
```
推薦值: 1e-5 (learning_rate / 10)
```
- **主要 UNet LR**: 1e-4
- **文本編碼器 LR**: 1e-5
- **比例**: 1:10

**為什麼分開設置?**
- UNet 需要改變生成內容（較大 LR）
- 文本編碼器只需微調詞彙理解（較小 LR）
- 分開設置能避免文本編碼器過度改變

### Max Grad Norm
```
推薦值: 1.0
```
- **說明**: 梯度裁剪，防止爆炸
- **保持默認**: 1.0 對大多數情況足夠

### Gradient Checkpointing
```
根據 GPU 決定
- 記憶體 < 6GB: ✅ 啟用
- 記憶體 >= 8GB: ❌ 不需要
```
- **作用**: 用計算換記憶體（訓練略慢但省記憶體）

---

## SDXL 專用參數

### SDXL Cache Text Encoder Outputs
```
✅ 啟用
```
- **說明**: 預先計算並快取文本編碼器輸出
- **優勢**: 加速訓練 20-30%
- **需求**: 額外 2-3GB 記憶體

### No Half VAE
```
✅ 啟用
```
- **說明**: 以全精度運行 VAE
- **為什麼**: 某些 SDXL VAE 在半精度下質量下降
- **代價**: 記憶體增加，但值得

### Clip Skip
```
推薦值: 2
```
- **說明**: 跳過文本編碼器的最後 N 層
- **Clip Skip 1**: 使用最後層（更詳細的語義）
- **Clip Skip 2**: 跳過最後層（更抽象，更靈活）- **推薦**

---

## 解析度和 Bucketing

### Resolution
```
推薦值: 1024,1024
```

| 解析度 | 記憶體 | 訓練速度 | 細節 | 推薦 |
|--------|-------|---------|-----|-----|
| 512,512 | 少 | 快 | 低 | 不推薦 (太小) |
| 768,768 | 中 | 中等 | 中等 | 可考慮 |
| **1024,1024** | **較高** | **標準** | **高** | **推薦** |
| 1024,1024+ | 很高 | 慢 | 最高 | 高端 GPU |

- **為什麼選 1024x1024**:
  - SDXL 設計解析度
  - 足以捕捉細節
  - 記憶體消耗合理

### Min Bucket Reso
```
推薦值: 256
```
- **說明**: 最小分桶解析度
- **作用**: 允許訓練非常小的圖片

### Max Bucket Reso
```
推薦值: 2048
```
- **說明**: 最大分桶解析度
- **作用**: 允許訓練很大的圖片

### Bucket Resolution Steps
```
推薦值: 64
```
- **說明**: 分桶解析度的步長
- **例子**: 256, 320, 384, 448, 512, ...
- **為什麼 64**: 64 的倍數對神經網絡友好

### Bucket No Upscale
```
✅ 啟用
```
- **說明**: 不放大小於目標解析度的圖片
- **優勢**: 保持原始質量，不引入上采樣失真

---

## 採樣和驗證

### Sample Every N Epochs
```
推薦值: 1
```
- **說明**: 每個 epoch 生成一次採樣圖片
- **用途**: 實時查看訓練效果進展

### Sample Prompts
```
推薦提示詞格式（以 Endou Mamoru 為例）:

1. "a boy, inazuma_endou_mamoru, anime_style, orange_headband,
    goalkeeper, determined expression, looking_at_viewer, masterpiece"

2. "1boy, inazuma_endou_mamoru, soccer uniform, confident pose,
    dynamic composition, professional quality"

3. "teenage boy, inazuma_endou_mamoru, anime style, cute face,
    white blue jersey, energetic, colorful background"
```

**標題詞組成要素** (必須包含):
1. 人物數量: `a boy`, `1boy`
2. **角色令牌**: `inazuma_endou_mamoru` (必須!)
3. 風格: `anime_style`
4. 特徵: `orange_headband`, `goalkeeper`
5. 表情/姿態: `determined expression`, `confident pose`
6. 品質: `masterpiece`, `professional quality`

### Negative Prompts
```
推薦值:
"bad quality, low quality, blurry, distorted, ugly, malformed,
 missing limbs, extra limbs, deformed face"
```
- **說明**: 告訴模型避免的東西
- **為什麼重要**: SDXL 在沒有負提示時容易生成低質量

### Sample Sampler
```
推薦值: euler_a
```
- **其他選項**: ddim, pndm, heun, dpm_2
- **為什麼選 euler_a**: 速度快，質量好，穩定

---

## 完整參數配置表 (一覽)

### 稲妻11 SDXL LoRA 標準配置

```
========== 基本設置 ==========
Base Model: sd_xl_base_1.0.safetensors
Output Precision: bf16

========== 數據集 ==========
Batch Size: 2
Enable Bucketing: ✅
Caption Extension: .txt
Shuffle Caption: ❌

========== 訓練參數 ==========
Max Epochs: 10
Max Steps: 1000 (或根據 batch 自動計算)
Learning Rate: 1e-4 (0.0001)
LR Scheduler: cosine
Optimizer: AdamW8bit
Max Grad Norm: 1.0

========== LoRA 參數 ==========
Network Dim: 64
Network Alpha: 32
Network Module: lycoris.kohya
Network Type: lora
Network Dropout: 0.05

========== SDXL 專用 ==========
Cache Text Encoder: ✅
No Half VAE: ✅
Clip Skip: 2

========== 解析度 ==========
Resolution: 1024,1024
Min Bucket: 256
Max Bucket: 2048
Bucket Steps: 64
No Upscale: ✅

========== 採樣 ==========
Sample Every N Epochs: 1
Sample Sampler: euler_a
Sample Prompts: (見上方)
```

---

## 根據 GPU 調整的預設

### RTX 3060 (12GB VRAM)
```
Batch Size: 2
Full BF16: ❌
Gradient Checkpointing: ❌
Cache Text Encoder: ❌ (記憶體吃緊)
Network Dim: 64
```

### RTX 4090 (24GB VRAM)
```
Batch Size: 4-8
Full BF16: ✅
Gradient Checkpointing: ❌
Cache Text Encoder: ✅
Network Dim: 64 或 128
```

### A100 (40GB VRAM)
```
Batch Size: 16+
Full BF16: ✅
Gradient Checkpointing: ❌
Cache Text Encoder: ✅
Network Dim: 128+
```

---

## 訓練過程中的監控

### 損失值 (Loss)
- **正常**: 逐漸下降，最後平穩
- **異常上升**: 學習率太大，改為 5e-5
- **下降緩慢**: 學習率太小，改為 5e-4

### GPU 記憶體
```bash
# 監控命令
watch -n 1 nvidia-smi
```
- 目標: 80-95% 使用率
- 過低: 增加 batch size
- 過高: 減小 batch size 或啟用 gradient checkpointing

### 訓練時間預估
```
每個 epoch 時間 ≈ (200 / 2) * (平均 step 時間)
               ≈ 100 steps * 0.5 秒/step
               ≈ 50 秒/epoch

10 epoch ≈ 8-10 分鐘
（視 GPU 和其他設置而異）
```

---

## 故障排除參數調整

### 如果 GPU 記憶體不足

**逐步降級** (按優先順序):
1. Batch Size: 2 → 1
2. Disable Cache Text Encoder
3. Network Dim: 64 → 32
4. Enable Gradient Checkpointing
5. Resolution: 1024 → 768

### 如果訓練很慢

**優化方案**:
1. Enable Full BF16
2. Enable Cache Text Encoder
3. Increase Batch Size
4. Increase Max Bucket Resolution

### 如果生成效果差

**調整訓練參數**:
1. 增加 Max Epochs: 10 → 20-30
2. 減小 Learning Rate: 1e-4 → 5e-5
3. 檢查數據質量

---

這就是全部參數說明！現在你可以按照上方的 GUI 指南直接在 Kohya_ss GUI 中輸入這些值。
