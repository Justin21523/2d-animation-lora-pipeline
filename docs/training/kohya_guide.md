好的，我已經將這兩份文件與您要求的第一份文件進行了整合，創建了一份全新的、互補合併的完整文件。這份文件保留了關鍵技術名詞的中英對照，並統一了結構，使其更具參考價值。

-----

# LoRA 訓練完整指南與疑難排解手冊 (LoRA Training Complete Guide & Troubleshooting Handbook)

**最後更新 (Last Updated)**：2025-11-22
**環境 (Environment)**：RTX 5080 16GB, PyTorch 2.7.1+cu128, Kohya SS sd-scripts (最新版)
**整合版本 (Integrated Version)**：包含訓練指南、疑難排解記錄及錯誤修復

-----

## 目錄 (Table of Contents)

1.  [環境設定 (Environment Setup)](https://www.google.com/search?q=%231-%E7%92%B0%E5%A2%83%E8%A8%AD%E5%AE%9A-environment-setup)
2.  [設定檔與 TOML 結構 (Configuration & TOML Structure)](https://www.google.com/search?q=%232-%E8%A8%AD%E5%AE%9A%E6%AA%94%E8%88%87-toml-%E7%B5%90%E6%A7%8B-configuration--toml-structure)
3.  [訓練方法 (Training Methods)](https://www.google.com/search?q=%233-%E8%A8%93%E7%B7%B4%E6%96%B9%E6%B3%95-training-methods)
4.  [RTX 5080 的最佳參數與限制 (Optimal Parameters & Constraints for RTX 5080)](https://www.google.com/search?q=%234-rtx-5080-%E7%9A%84%E6%9C%80%E4%BD%B3%E5%8F%83%E6%95%B8%E8%88%87%E9%99%90%E5%88%B6-optimal-parameters--constraints-for-rtx-5080)
5.  [訓練啟動與監控 (Training Launch & Monitoring)](https://www.google.com/search?q=%235-%E8%A8%93%E7%B7%B4%E5%95%9F%E5%8B%95%E8%88%87%E7%9B%A3%E6%8E%A7-training-launch--monitoring)
6.  [訓練前檢查清單 (Pre-Training Checklist)](https://www.google.com/search?q=%236-%E8%A8%93%E7%B7%B4%E5%89%8D%E6%AA%A2%E6%9F%A5%E6%B8%85%E5%96%AE-pre-training-checklist)
7.  [疑難排解與錯誤修復 (Troubleshooting & Error Fixes)](https://www.google.com/search?q=%237-%E7%96%91%E9%9B%A3%E6%8E%92%E8%A7%A3%E8%88%87%E9%8C%AF%E8%AA%A4%E4%BF%AE%E5%BE%A9-troubleshooting--error-fixes)
8.  [參考指令 (Reference Commands)](https://www.google.com/search?q=%238-%E5%8F%83%E8%80%83%E6%8C%87%E4%BB%A4-reference-commands)

-----

## 1\. 環境設定 (Environment Setup)

### 專用 Conda 環境：`kohya_ss`

我們維護一個**專用的 conda 環境**專門用於 Kohya LoRA 訓練，以確保相容性並防止版本衝突。

**請注意：**

  * `ai_env` - 數據預處理 (分割、聚類、標註)
  * `kohya_ss` - **僅用於 LoRA 訓練**

#### 快速設定

```bash
# 執行自動化設定腳本
bash /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/setup_kohya_env.sh
```

#### 手動設定

```bash
# 1. 建立環境
conda create -n kohya_ss python=3.10 -y

# 2. 安裝 PyTorch (支援 CUDA 12.8)
conda run -n kohya_ss pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安裝 Kohya 相依套件
conda run -n kohya_ss pip install \
    accelerate==0.30.0 \
    transformers==4.44.0 \
    diffusers[torch]==0.25.0 \
    bitsandbytes>=0.45.0 \
    safetensors==0.4.2 \
    toml==0.10.2 \
    # ... (完整列表請參閱 setup_kohya_env.sh)
```

#### 驗證

```bash
conda run -n kohya_ss python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

import bitsandbytes as bnb
print(f'bitsandbytes: {bnb.__version__}')

# 測試 AdamW8bit
param = torch.nn.Parameter(torch.randn(10, 10).cuda())
optimizer = bnb.optim.AdamW8bit([param], lr=1e-3)
print('✓ AdamW8bit works!')
"
```

**預期輸出：**

```
PyTorch: 2.7.1+cu128
CUDA: True
bitsandbytes: 0.48.2
✓ AdamW8bit works!
```

-----

## 2\. 設定檔與 TOML 結構 (Configuration & TOML Structure)

### ⚠️ 重要：Kohya 配置系統說明

Kohya SS 實際上只**官方支援一種配置方式**：**`--dataset_config` + CLI 參數**。

  * **`--dataset_config` (TOML)**：僅用於配置數據集（路徑、解析度、批次大小等）。
  * **`--config_file` (TOML)**：用於配置訓練參數（學習率、優化器、epochs 等），這是我們**推薦**用於保存完整訓練設定的方式，以確保可重現性。

### 錯誤的 TOML 結構 (❌ 切勿使用)

Kohya sd-scripts **無法正確解析**嵌套的 TOML 區段（除了 `[metadata]`）。

**❌ 錯誤示範:**

```toml
[model]  # 錯誤：不應有嵌套區段
pretrained_model_name_or_path = "..."
network_dim = 64

[training] # 錯誤：不應有嵌套區段
train_batch_size = 8
learning_rate = 6e-5
```

### 正確的 TOML 結構 (✅ 務必使用)

使用**扁平的 TOML 結構**來設定所有配置參數。

**✅ 正確範本 (角色身分 LoRA):**

```toml
# Miguel Rivera (Coco) - 身分 LoRA 訓練配置
# 基於 Trial 3.6 最終版 + RTX 5080 優化

# 模型參數 (Model Arguments) - 直接寫在頂層
pretrained_model_name_or_path = "/path/to/v1-5-pruned-emaonly.safetensors"
output_name = "character_identity_lora"
output_dir = "/path/to/output"
save_model_as = "safetensors"
save_precision = "fp16"

# 網路架構 (Network Architecture)
network_module = "networks.lora"
network_dim = 64
network_alpha = 32
network_dropout = 0.1

# 數據集配置 (Dataset Configuration)
# 注意：train_data_dir 必須指向包含 "N_classname" 子目錄的父目錄
train_data_dir = "/path/to/training_data/miguel_identity"
caption_extension = ".txt"  # ⚠️ 必須明確指定
resolution = "512,512"
enable_bucket = true
min_bucket_reso = 256
max_bucket_reso = 1024
bucket_reso_steps = 64
shuffle_caption = false
keep_tokens = 0

# 訓練參數 (Training Parameters)
max_train_epochs = 14        # 視數據集大小而定 (14-18)
train_batch_size = 8         # RTX 5080 優化 (從 4 增加)
gradient_accumulation_steps = 2
save_every_n_epochs = 2
mixed_precision = "fp16"
gradient_checkpointing = true
max_data_loader_n_workers = 4
persistent_data_loader_workers = true

# 優化器 (Optimizer)
optimizer_type = "AdamW8bit"
learning_rate = 6e-5
unet_lr = 6e-5
text_encoder_lr = 3e-5
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 2
lr_warmup_steps = 100

# 正則化 (Regularization)
min_snr_gamma = 5.0
noise_offset = 0.05
max_grad_norm = 1.0

# 數據增強 (Data Augmentation)
color_aug = false           # ❌ 3D PBR 材質禁用
flip_aug = false            # ❌ 不對稱角色禁用
random_crop = true          # ✅ 3D 可用 (模擬構圖)
# cache_latents = false     # ❌ 使用 random_crop 時禁用 cache_latents

# RTX 5080 關鍵設定
xformers = false            # ❌ RTX 5080 不相容
sdpa = true                 # ✅ RTX 5080 使用原生 SDPA

# 記錄 (Logging)
logging_dir = "/path/to/logs"
log_prefix = "character_identity"
log_with = "tensorboard"

# 樣本生成 (Sample Generation)
sample_prompts = "prompt1\nprompt2\nprompt3"
sample_every_n_epochs = 2
sample_sampler = "euler_a"

# 訓練種子 (Training Seed)
seed = 42

# Metadata (僅此區段允許嵌套，用於文檔)
[metadata]
character_name = "Miguel Rivera"
dataset_size = 449
lora_type = "character_identity"
notes = "Training notes here"
```

-----

## 3\. 訓練方法 (Training Methods)

### 方法 1：TOML 配置 (推薦)

**優點：**

  - ✅ 可重現 (保存確切的訓練設定)
  - ✅ 適合版本控制
  - ✅ 易於分享和重複使用
  - ✅ 比長長的 CLI 指令更不易出錯

**用法：**

```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file configs/your_character/training_config.toml
```

### 方法 2：CLI 參數 (不推薦)

**缺點：**

  - ❌ 難以重現
  - ❌ 容易出錯 (拼寫錯誤、遺漏標誌)
  - ❌ 難以追蹤變更

**僅用於快速測試：**

```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --dataset_config /path/to/dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    # ... (50+ 個更多參數)
```

-----

## 4\. RTX 5080 的最佳參數與限制 (Optimal Parameters & Constraints for RTX 5080)

### 硬體規格

  - **GPU:** NVIDIA GeForce RTX 5080
  - **VRAM:** 16GB
  - **CUDA:** 12.8
  - **架構:** Blackwell / Ada Lovelace

### 關鍵限制 (Critical Constraints)

#### ❌ 絕對禁止 (DO NOT USE):

1.  `--xformers` 標誌：硬體不相容，會導致 `ImportError`。
2.  `flip_aug` 或 `color_aug`：會破壞 3D 角色的一致性與材質。
3.  同時啟用 `cache_latents` 與 `random_crop`：兩者不相容。

#### ✅ 務必使用 (MUST USE):

1.  `sdpa = true`：使用 PyTorch 原生 SDPA 加速。
2.  `optimizer_type = "AdamW8bit"`：在 `kohya_ss` 環境中運作良好，節省 VRAM。
3.  `gradient_checkpointing = true`：節省 VRAM。

### 批次大小優化 (Batch Size Optimization)

觀察顯示 `batch_size = 4` 時 GPU 利用率僅 38%，VRAM 僅用 36.5%。

**優化設定：**

  * **Train Batch Size**: `8` (從 4 增加)
  * **Gradient Accumulation**: `2`
  * **Effective Batch Size**: 16 (8 \* 2)
  * **預期 VRAM 使用**: \~6000 MB / 16303 MB (仍有餘裕，可嘗試 12)

### 推薦訓練參數 (依類型)

#### 用於角色 LoRA (200-400 張圖像)：

```toml
learning_rate = 0.0001 (或 6e-5)
unet_lr = 0.0001 (或 6e-5)
text_encoder_lr = 0.00005 (或 3e-5)
max_train_epochs = 12-15
network_dim = 64
network_alpha = 32
resolution = "512,512"
train_batch_size = 8
```

#### 用於風格 LoRA (500-1000 張圖像)：

```toml
learning_rate = 0.00005 (較低)
max_train_epochs = 8-10
network_dim = 128 (較高容量)
network_alpha = 64
train_batch_size = 8
```

-----

## 5\. 訓練啟動與監控 (Training Launch & Monitoring)

### 標準啟動指令

```bash
cd /path/to/kohya_ss/sd-scripts
conda run -n kohya_ss \
  accelerate launch --num_cpu_threads_per_process=4 \
  train_network.py \
  --config_file=/path/to/your_config.toml
```

### 背景訓練 (帶日誌)

```bash
cd /path/to/kohya_ss/sd-scripts
conda run -n kohya_ss \
  accelerate launch --num_cpu_threads_per_process=4 \
  train_network.py \
  --config_file=/path/to/your_config.toml \
  2>&1 | tee /path/to/logs/training_$(date +%Y%m%d_%H%M%S).log &
```

### 監控訓練

```bash
# 檢查 GPU 使用率
nvidia-smi

# 檢查訓練日誌
tail -f /path/to/logs/training_*.log

# 檢查 TensorBoard
tensorboard --logdir=/path/to/logs --port 6006
```

### 預期訓練行為

  * **初始化**: \~30-60 秒 (載入模型、準備 buckets)。
  * **訓練速度**: RTX 5080 約 1-1.5 秒/步。
  * **總時長**: 典型角色 LoRA (約 7800 步) 需 3-4 小時。
  * **檢查點**: 每 2 個 epochs 儲存一次。

-----

## 6\. 訓練前檢查清單 (Pre-Training Checklist)

在開始任何 LoRA 訓練之前，請務必驗證：

### 1\. 配置 (Configuration)

  - [ ] **扁平的 TOML 結構** (除 `[metadata]` 外無嵌套區段)
  - [ ] **caption\_extension = ".txt"** 明確設定
  - [ ] **xformers = false, sdpa = true** (針對 RTX 5080)
  - [ ] 使用 `random_crop` 時 **無 cache\_latents**
  - [ ] 對於 3D 角色 **color\_aug = false, flip\_aug = false**

### 2\. 目錄結構 (Directory Structure)

  - [ ] `train_data_dir` 指向包含 `N_classname/` 子目錄的**父目錄**
  - [ ] 文件夾名稱包含重複計數 (10\_miguel, 20\_miguel 等)
  - [ ] 圖像和標註在同一個文件夾中
  - [ ] 標註文件與圖像名稱匹配，副檔名為 `.txt`

### 3\. 環境 (Environment)

  - [ ] 使用 `kohya_ss` conda 環境
  - [ ] 基礎模型路徑正確且可訪問

### 4\. 數據集驗證 (Dataset Verification)

  - [ ] 所有圖像都有對應的標註文件
  - [ ] 數據集大小適當 (角色身分為 200-500)
  - [ ] 無損壞的圖像

-----

## 7\. 疑難排解與錯誤修復 (Troubleshooting & Error Fixes)

### 常見錯誤對照表

| 錯誤訊息 / 症狀 | 原因 | 解決方案 |
| :--- | :--- | :--- |
| **ImportError: No xformers** | RTX 5080 未安裝 xformers | 設定 `xformers = false`, `sdpa = true` |
| **No caption file found** | 標註副檔名不符 | 在配置中設定 `caption_extension = ".txt"` |
| **No data found** | `train_data_dir` 路徑錯誤 | 指向包含 `10_class` 子文件夾的**父目錄**，而非圖像目錄本身 |
| **AssertionError: cache\_latents...** | 同時啟用快取與增強 | 設定 `cache_latents = false` (若使用 `random_crop`) |
| **ModuleNotFoundError: bitsandbytes** | 環境錯誤 | 確保使用 `kohya_ss` 環境，而非 `ai_env` |
| **CUDA out of memory** | 批次大小過大 | 降低 `train_batch_size`，增加 `gradient_accumulation_steps` |
| **配置參數被忽略** | 使用了嵌套 TOML 區段 | 移除 `[model]`, `[training]` 等標頭，使用扁平結構 |

### 詳細錯誤案例

#### 錯誤 1：無效的 TOML 區段結構

**問題**：使用了 Kohya 無法解析的 `[model]`、`[training]` 嵌套區段。
**症狀**：設定被忽略，回退到預設值（例如標註副檔名變回 `.caption`）。
**修復**：移除所有區段標頭，將參數直接寫在檔案頂層。

#### 錯誤 2：cache\_latents 與 random\_crop 不相容

**問題**：同時啟用兩者導致錯誤。
**解釋**：快取的 latents 是固定裁切的，無法進行運行時增強。
**修復**：對於 3D 角色，優先選擇 `random_crop = true`，並設定 `cache_latents = false`（或直接不設定）。

#### 錯誤 3：訓練卡住/掛起

**症狀**：GPU 使用率 0%，無進度。
**修復**：

1.  檢查數據集路徑和標註。
2.  設定 `max_data_loader_n_workers = 0` 進行除錯。
3.  檢查是否使用了最小配置測試。

-----

## 8\. 參考指令 (Reference Commands)

### 環境管理

```bash
# 啟用環境
conda activate kohya_ss

# 更新套件
conda run -n kohya_ss pip install --upgrade bitsandbytes transformers

# 重建環境 (如果損壞)
conda remove -n kohya_ss --all -y
bash setup_kohya_env.sh
```

### 檔案操作

```bash
# 列出訓練好的檢查點
ls -lh /path/to/output/*.safetensors

# 檢查檢查點大小
du -h /path/to/output/character_lora.safetensors

# 比較兩個配置檔
diff configs/v1/config.toml configs/v2/config.toml
```

-----

**相關文件 (Related Documentation):**

  * `configs/templates/lora_training_template.toml` - 推薦的主配置範本
  * `docs/GPU_OPTIMIZATION_GUIDE.md` - RTX 5080 進階優化
  * `setup_kohya_env.sh` - 環境設定腳本

**版本歷史 (Version History):**

  * **2025-11-11**: 初始指南發布 (v1.0)
  * **2025-11-21**: 基於 Miguel LoRA 訓練經驗更新，新增 RTX 5080 特定修復與疑難排解 (v2.0)
  * **2025-11-22**: 整合所有文檔為單一完整指南 (v2.1)