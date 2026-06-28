好的，這是一份將說明文字翻譯為全中文（保留關鍵技術名詞中英對照）的文檔版本，內容結構與數據部分完全保持不變。

-----

# Kohya LoRA 訓練完整指南 (Kohya LoRA Training Complete Guide)

**最後更新**：2025-11-11
**環境**：RTX 5080 16GB, PyTorch 2.7.1+cu128, Kohya SS sd-scripts (最新版)

-----

## 目錄 (Table of Contents)

1.  [環境設定](https://www.google.com/search?q=%23%E7%92%B0%E5%A2%83%E8%A8%AD%E5%AE%9A)
2.  [設定檔](https://www.google.com/search?q=%23%E8%A8%AD%E5%AE%9A%E6%AA%94)
3.  [訓練方法](https://www.google.com/search?q=%23%E8%A8%93%E7%B7%B4%E6%96%B9%E6%B3%95)
4.  [RTX 5080 的最佳參數](https://www.google.com/search?q=%23rtx-5080-%E7%9A%84%E6%9C%80%E4%BD%B3%E5%8F%83%E6%95%B8)
5.  [疑難排解](https://www.google.com/search?q=%23%E7%96%91%E9%9B%A3%E6%8E%92%E8%A7%A3)
6.  [參考指令](https://www.google.com/search?q=%23%E5%8F%83%E8%80%83%E6%8C%87%E4%BB%A4)

-----

## 環境設定 (Environment Setup)

### 專用 Conda 環境：`kohya_ss`

我們維護一個**專用的 conda 環境**專門用於 Kohya LoRA 訓練，以確保相容性並防止版本衝突。

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

## 設定檔 (Configuration Files)

### ⚠️ 重要：Kohya 配置系統說明 (Kohya Configuration System Explained)

**請務必理解 Kohya 的配置系統，避免之前遇到的問題！**

Kohya SS 實際上只**官方支援一種配置方式**：

#### `--dataset_config`（唯一官方支援的 TOML 配置）

```bash
python train_network.py \
    --dataset_config dataset.toml \
    --pretrained_model_name_or_path /path/to/model \
    --output_dir /path/to/output \
    --learning_rate 0.0001 \
    # ... 其他參數都用 CLI
```

**特點：**

  - ✅ **唯一官方文檔化的 TOML 配置**
  - ⚠️  **只能配置數據集**（image\_dir, batch\_size, resolution 等）
  - ❌ **不能配置訓練參數**（learning\_rate, optimizer, epochs 等）
  - ❌ 其他參數必須通過 CLI 傳遞

**詳細解釋請參考：** `docs/TOML_CONFIG_EXPLAINED.md`

### 為什麼之前 TOML 配置出問題？

**問題原因：**

1.  我們混淆了 `--dataset_config` 和 `--config_file` 兩種方式
2.  創建的範本使用了 `[model_arguments]`、`[training_arguments]` 等區段
3.  但 Kohya 的 `--dataset_config` 只能解析數據集配置
4.  導致訓練參數無法被讀取，必須改用 CLI

**當前解決方案：**

  - 使用 `--dataset_config` + CLI 參數混合方式
  - 穩定可靠，Kohya 官方支援
  - 我們的迭代訓練系統已採用此方式

### TOML 配置結構 (TOML Configuration Structure)

Kohya 支援的配置方式：

1.  **dataset\_config + CLI** (官方支援，當前使用)
2.  **Pure CLI** (完全支援，但命令行過長)
3.  **config\_file** (未文檔化，不推薦依賴)

我們使用 **方法 1** (dataset\_config + CLI) 以確保可靠性。

### 範本檔案

```
configs/
├── templates/
│   ├── lora_training_template.toml      # 主訓練配置範本
│   └── dataset_config_template.toml     # 數據集配置範本
└── your_character/
    ├── training_config.toml              # 您的自定義配置
    └── dataset_config.toml               # 您的數據集配置
```

### 建立新配置

```bash
# 1. 複製範本
mkdir -p configs/your_character
cp configs/templates/lora_training_template.toml configs/your_character/training_config.toml
cp configs/templates/dataset_config_template.toml configs/your_character/dataset_config.toml

# 2. 編輯配置 (替換 YOUR_CHARACTER 佔位符)
nano configs/your_character/training_config.toml
nano configs/your_character/dataset_config.toml

# 3. 測試配置
bash test_toml_training.sh  # 驗證配置結構
```

### 關鍵配置區段

#### 1\. 模型參數 (Model Arguments)

```toml
[model_arguments]
pretrained_model_name_or_path = "/path/to/stable-diffusion-v1-5"
output_dir = "/path/to/output"
output_name = "character_lora_v1"
save_model_as = "safetensors"
save_precision = "fp16"
```

#### 2\. 訓練參數 (Training Arguments)

```toml
[training_arguments]
learning_rate = 0.0001
unet_lr = 0.0001
text_encoder_lr = 0.00005
lr_scheduler = "cosine_with_restarts"
optimizer_type = "AdamW8bit"  # 或 "AdamW"
max_train_epochs = 15
mixed_precision = "fp16"
gradient_checkpointing = true
gradient_accumulation_steps = 2
```

#### 3\. 網路參數 (Network Arguments) (LoRA)

```toml
[network_arguments]
network_module = "networks.lora"
network_dim = 64      # LoRA rank
network_alpha = 32    # 通常為 rank/2
```

#### 4\. 數據集配置 (Dataset Configuration)

```toml
[dataset_arguments]
dataset_config = "/path/to/dataset_config.toml"
```

**在 `dataset_config.toml` 中：**

```toml
[[datasets]]
resolution = 512
batch_size = 10

  [[datasets.subsets]]
  image_dir = "/path/to/images"
  class_tokens = "character_name trigger_word"
  num_repeats = 1
  caption_extension = ".txt"
```

-----

## 訓練方法 (Training Methods)

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

## RTX 5080 的最佳參數 (Optimal Parameters for RTX 5080)

### 硬體規格

  - **GPU:** NVIDIA GeForce RTX 5080
  - **VRAM:** 16GB
  - **CUDA:** 12.8
  - **架構:** Blackwell

### 關鍵限制

#### ❌ 請勿使用：

  - `--xformers` 標誌 (硬體不相容)
  - `flip_aug` 或 `color_aug` (破壞 3D 角色一致性)
  - `optimizer_type = "AdamW8bit"` **未經測試** (需要適當的 bitsandbytes)

#### ✅ 推薦：

```toml
[training_arguments]
optimizer_type = "AdamW8bit"  # 在 kohya_ss 環境中運作
# 或
optimizer_type = "AdamW"      # 安全備案，VRAM 稍微多一點

[caching_arguments]
cache_latents = true
cache_latents_to_disk = true

[training_arguments]
gradient_checkpointing = true
gradient_accumulation_steps = 2
```

### 按解析度的最佳設定

| 解析度 (Resolution) | 批次大小 (Batch Size) | Network Dim | VRAM 使用量 | 訓練速度 |
|------------|------------|-------------|------------|----------------|
| 512x512    | 10         | 64          | \~15GB      | \~30s/step      |
| 512x512    | 8          | 64          | \~13GB      | \~24s/step      |
| 768x768    | 4-6        | 64          | \~14GB      | \~40s/step      |
| 1024x1024  | 2-4        | 128         | \~15GB      | \~60s/step      |

### 推薦訓練參數

#### 用於角色 LoRA (200-400 張圖像)：

```toml
[training_arguments]
learning_rate = 0.0001
unet_lr = 0.0001
text_encoder_lr = 0.00005
max_train_epochs = 12-15
lr_scheduler = "cosine_with_restarts"

[network_arguments]
network_dim = 64
network_alpha = 32

[[datasets]]
resolution = 512
batch_size = 10
num_repeats = 1  # 如果圖像少於 150 張，增加到 2
```

#### 用於風格 LoRA (500-1000 張圖像)：

```toml
[training_arguments]
learning_rate = 0.00005  # 風格使用較低學習率
max_train_epochs = 8-10

[network_arguments]
network_dim = 128  # 風格使用較高容量
network_alpha = 64

[[datasets]]
batch_size = 8
num_repeats = 1
```

-----

## 疑難排解 (Troubleshooting)

### 常見問題與解決方案

#### 1\. bitsandbytes 匯入錯誤

**錯誤：**

```
ModuleNotFoundError: No module named 'bitsandbytes'
ImportError: libbitsandbytes_cuda128.so not found
```

**解決方案：**

```bash
# 使用 kohya_ss 環境
conda activate kohya_ss

# 或重建環境
bash setup_kohya_env.sh
```

#### 2\. xformers 錯誤

**錯誤：**

```
ImportError: No module named 'xformers'
```

**解決方案：**
❌ **請勿在 RTX 5080 上安裝 xformers**。它不相容。

從所有訓練指令和配置中移除 `--xformers` 標誌。

#### 3\. TOML 配置無法運作

**錯誤：**

```
KeyError: 'model_arguments'
ValueError: Unknown argument: pretrained_model_name_or_path
```

**解決方案：**
Kohya 期望特定的配置結構。使用我們的範本：

  - `configs/templates/lora_training_template.toml`
  - `configs/templates/dataset_config_template.toml`

**測試配置有效性：**

```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file your_config.toml \
    --help
```

#### 4\. 記憶體不足 (OOM)

**錯誤：**

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解決方案：**
逐步減少記憶體使用：

1.  **降低批次大小：**
       `toml    batch_size = 8  # 或 6, 4    `

2.  **增加梯度累積：**
       `toml    gradient_accumulation_steps = 4  # 實際上使批次大小加倍    `

3.  **啟用更多快取：**
       `toml    cache_latents_to_disk = true    cache_text_encoder_outputs_to_disk = true  # 用於 SDXL    `

4.  **降低網路容量：**
       `toml    network_dim = 32  # 代替 64    `

#### 5\. 訓練卡住/掛起

**症狀：**

  - 過程開始但沒有訓練進度
  - GPU 使用率 0%
  - 無錯誤訊息

**解決方案：**

**a) 檢查數據集路徑：**

```bash
# 驗證圖像存在
ls -lh /path/to/your/dataset/images/*.png | head

# 驗證標註存在
ls -lh /path/to/your/dataset/images/*.txt | head
```

**b) 使用最小配置測試：**

```toml
max_train_epochs = 1
save_every_n_epochs = 1
[[datasets.subsets]]
num_repeats = 1
```

**c) 檢查數據加載器工作線程：**

```toml
max_data_loader_n_workers = 0  # 禁用多進程以進行除錯
```

#### 6\. 訓練結果不佳

**症狀：**

  - LoRA 未捕捉到角色特徵
  - 過擬合 (Overfitting)（死記硬背確切的訓練圖像）
  - 欠擬合 (Underfitting)（幾乎不影響輸出）

**解決方案：**

**針對過擬合：**

  - 減少 epochs：`max_train_epochs = 10` (代替 20)
  - 增加數據集大小或重複次數
  - 降低學習率：`learning_rate = 0.00005`
  - 增加標註 dropout：`caption_dropout_rate = 0.1`

**針對欠擬合：**

  - 增加 epochs：`max_train_epochs = 20`
  - 增加網路容量：`network_dim = 128`
  - 提高學習率：`learning_rate = 0.0002`
  - 檢查標註品質 (通用 vs. 特異)

-----

## 參考指令 (Reference Commands)

### 環境管理

```bash
# 啟用 kohya 環境
conda activate kohya_ss

# 停用
conda deactivate

# 列出環境
conda env list

# 更新套件
conda run -n kohya_ss pip install --upgrade bitsandbytes transformers

# 移除並重建環境
conda remove -n kohya_ss --all -y
bash setup_kohya_env.sh
```

### 訓練指令

**基本訓練：**

```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file configs/my_character/training_config.toml
```

**帶有輸出重定向的訓練：**

```bash
conda run -n kohya_ss python /path/to/sd-scripts/train_network.py \
    --config_file configs/my_character/training_config.toml \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
```

**在 tmux 中訓練 (用於長時間會話)：**

```bash
tmux new -s training
conda activate kohya_ss
python /path/to/sd-scripts/train_network.py --config_file configs/my_config.toml

# 分離 (Detach): Ctrl+B, D
# 重新連接 (Reattach): tmux attach -t training
```

### 監控

```bash
# GPU 使用率
nvidia-smi

# 持續監控
watch -n 1 nvidia-smi

# TensorBoard 日誌
tensorboard --logdir /path/to/logs --port 6006

# 訓練進度 (如果使用 tee)
tail -f logs/training_20251111_120000.log
```

### 檔案操作

```bash
# 列出訓練好的檢查點
ls -lh /path/to/output/*.safetensors

# 檢查檢查點大小
du -h /path/to/output/character_lora.safetensors

# 比較配置
diff configs/version1/training_config.toml configs/version2/training_config.toml

# 尋找所有 TOML 配置
find configs/ -name "*.toml" -type f
```

-----

## 額外資源 (Additional Resources)

### 內部文件

  - `configs/templates/lora_training_template.toml` - 主配置範本 (參考用)
  - `configs/templates/dataset_config_template.toml` - 數據集配置範本 (用於 --dataset\_config)
  - `docs/GPU_OPTIMIZATION_GUIDE.md` - RTX 5080 優化指南
  - `docs/TOML_CONFIG_EXPLAINED.md` - **為什麼 TOML 有問題以及如何正確使用**
  - `setup_kohya_env.sh` - 環境設定腳本
  - `test_toml_training.sh` - 配置驗證腳本

### 外部資源

  - [Kohya SS GitHub](https://github.com/kohya-ss/sd-scripts)
  - [LoRA Training Guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-en.md)
  - [Stable Diffusion Training Tips](https://rentry.org/2chAI_LoRA_Dreambooth_guide_english)

-----

## 版本歷史 (Version History)

| 日期       | 版本 | 變更                                          |
|------------|---------|--------------------------------------------------|
| 2025-11-11 | 1.0     | 初始指南，包含 kohya\_ss 環境設定    |
| 2025-11-11 | 1.1     | 新增 TOML 範本和 RTX 5080 優化  |

-----

**如有問題或議題，請參考：**

  - 專案文件：`/docs/`
  - 環境設定日誌：`/tmp/kohya_env_setup.log`
  - 訓練日誌：`/mnt/data/ai_data/models/lora/*/logs/`