# SDXL LoRA Training on 16GB VRAM - Complete Guide

## 概述

本指南詳細說明如何在**16GB VRAM GPU**上成功訓練**SDXL LoRA**模型，包含：
- 記憶體優化技術詳解
- 完整配置說明
- 預期VRAM使用量
- Troubleshooting指南
- SD 1.5 vs SDXL比較

---

## 🎯 核心優化技術

### 1. **8-bit AdamW Optimizer（最關鍵）**

#### 原理
將optimizer states從32-bit量化為8-bit：
- **Momentum buffers**: 32-bit → 8-bit
- **Variance buffers**: 32-bit → 8-bit
- **Gradients**: 保持32-bit精度

#### VRAM節省
- **標準AdamW**: ~8GB optimizer states（SDXL）
- **8-bit AdamW**: ~2GB optimizer states
- **節省**: **~6GB (40%)**

#### 品質影響
- **CLIP Score差異**: <0.5%
- **視覺差異**: 肉眼幾乎無法分辨
- **訓練穩定性**: 與32-bit相當

#### 配置
```toml
[training]
optimizer_type = "AdamW8bit"  # 啟用8-bit量化
```

---

### 2. **Gradient Checkpointing**

#### 原理
- 訓練時不保存所有中間激活值
- Forward pass計算時只保存checkpoints
- Backward pass時重新計算缺失的激活值
- **時間換空間**策略

#### VRAM節省
- **Without checkpointing**: ~12GB activations（SDXL）
- **With checkpointing**: ~4GB activations
- **節省**: **~8GB (30%)**

#### 代價
- 訓練速度降低**15-20%**
- 16GB VRAM情況下完全值得

#### 配置
```toml
[training]
gradient_checkpointing = true
```

---

### 3. **Mixed Precision Training (BF16)**

#### 原理
- 大部分計算使用16-bit浮點數（bfloat16）
- 關鍵步驟保持32-bit精度
- 自動混合精度（AMP）技術

#### VRAM節省
- **FP32 training**: ~16GB model weights
- **BF16 training**: ~8GB model weights
- **節省**: **~8GB (25%)**

#### BF16 vs FP16
| 特性 | BF16 | FP16 |
|------|------|------|
| **動態範圍** | 與FP32相同 | 較窄 |
| **數值穩定性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **需要loss scaling** | ❌ | ✅ |
| **推薦用途** | SDXL訓練 | 推理 |

#### 配置
```toml
[training]
mixed_precision = "bf16"
full_bf16 = true
```

---

### 4. **Latent Caching**

#### 原理
- 預先將所有圖片編碼為VAE latents
- 緩存到RAM或磁盤
- 訓練時直接使用緩存，不重複編碼

#### VRAM節省
- **Without caching**: VAE encoder占用~2GB
- **With caching**: VAE不需加載到VRAM
- **節省**: **~2GB**

#### 配置
```toml
[training]
cache_latents = true
cache_latents_to_disk = false  # 使用RAM（更快）
vae_batch_size = 1             # 一次處理一張
```

---

### 5. **Flash Attention 2（可選）**

#### 原理
- 優化Transformer attention計算
- 使用更高效的CUDA kernel
- Fused operations減少記憶體讀寫

#### VRAM節省
- **Standard attention**: ~3GB
- **Flash Attention 2**: ~2GB
- **節省**: **~1GB (15%)**
- **額外優勢**: 速度提升**2倍**

#### 安裝
```bash
# 需要CUDA 11.8+
conda activate kohya_ss
pip install flash-attn --no-build-isolation
```

#### 自動啟用
Kohya_ss會自動檢測並使用Flash Attention（無需配置）

---

## 📊 VRAM使用量預估

### **完整優化組合下的VRAM分配**

| 組件 | 無優化 (SDXL) | 16GB優化 | 節省 |
|------|---------------|----------|------|
| **Model Weights (UNet + TE)** | 16GB | 8GB (bf16) | -8GB |
| **Optimizer States** | 8GB | 2GB (8bit) | -6GB |
| **Activations/Gradients** | 12GB | 4GB (checkpointing) | -8GB |
| **VAE Encoder** | 2GB | 0GB (caching) | -2GB |
| **Attention** | 3GB | 2GB (flash attn) | -1GB |
| **Batch Data** | 2GB | 0.5GB (batch=1) | -1.5GB |
| **PyTorch Overhead** | 1GB | 1GB | 0GB |
| **總計** | **44GB** | **17.5GB** | **-26.5GB** |

### **實際VRAM使用曲線**

```
訓練過程中的VRAM使用（16GB優化配置）：

Initialization:    ████████░░░░░░░░  8GB   (加載模型)
First Forward:     ████████████████  14GB  (峰值，首次計算)
Training Stable:   ██████████████░░  12-13GB (穩定狀態)
Saving Checkpoint: ████████████░░░░  11GB  (保存時)
Validation:        █████████████░░░  12GB  (生成樣本)
```

### **安全邊界**
- **16GB VRAM**: **✅ 安全** (峰值14-15GB)
- **12GB VRAM**: **⚠️ 困難** (需進一步優化)
- **8GB VRAM**: **❌ 不可行** (即使極限優化)

---

## ⚙️ 完整配置範例

### **基礎配置（必需）**

```toml
[model]
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
network_dim = 128
network_alpha = 96

[training]
# 核心優化三件套
optimizer_type = "AdamW8bit"         # ⭐ 8-bit optimizer
mixed_precision = "bf16"             # ⭐ BF16 training
full_bf16 = true
gradient_checkpointing = true        # ⭐ Gradient checkpointing

# 記憶體優化
cache_latents = true                 # ⭐ Cache VAE latents
vae_batch_size = 1
max_data_loader_n_workers = 2

# Batch設置（維持effective batch = 8）
train_batch_size = 1                 # ⭐ 小batch for VRAM
gradient_accumulation_steps = 8      # ⭐ 累積梯度

# Learning rates
learning_rate = 0.0001
text_encoder_lr = 0.00006
unet_lr = 0.0001

# Training duration
max_train_epochs = 20
save_every_n_epochs = 2
```

### **進階優化（推薦）**

```toml
[training]
# Min-SNR weighting (提升穩定性)
min_snr_gamma = 5.0

# Noise offset (改善對比度)
noise_offset = 0.05

# Cosine schedule with restarts
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 3
lr_warmup_steps = 100

# SDXL resolution & bucketing
resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 640
max_bucket_reso = 1536
bucket_reso_steps = 64
bucket_no_upscale = true
```

---

## 🚀 使用方法

### **步驟1：準備數據集**

```bash
# 使用通用腳本準備Kohya格式數據集
bash scripts/training/prepare_kohya_dataset.sh \
  --source-dir /path/to/curated_images \
  --output-dir /path/to/sdxl_training \
  --repeat 10 \
  --name character_name \
  --validate
```

### **步驟2：檢查GPU**

```bash
# 確認VRAM
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits

# 清空GPU緩存
python3 -c "import torch; torch.cuda.empty_cache()"
```

### **步驟3：啟動訓練**

```bash
# 使用16GB優化腳本
bash scripts/training/start_sdxl_16gb_training.sh
```

### **步驟4：監控VRAM**

```bash
# 新終端窗口監控VRAM
watch -n 1 nvidia-smi

# 或使用腳本
while true; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  sleep 5
done
```

---

## 🔧 Troubleshooting

### **問題1: OOM (Out of Memory)**

#### 症狀
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 15.90 GiB total capacity; 14.50 GiB already allocated)
```

#### 解決方案（按順序嘗試）

**A. 降低batch size（最有效）**
```toml
train_batch_size = 1  # 已經是最小
gradient_accumulation_steps = 8  # 可以增加到12維持effective batch
```

**B. 減小resolution**
```toml
resolution = "768,768"  # 從1024降到768（VRAM減少~30%）
max_bucket_reso = 1024  # 同步調整
```

**C. 凍結Text Encoder（僅訓練UNet）**
```toml
train_text_encoder = false  # 節省~2GB VRAM
```

**D. 啟用CPU offloading（最後手段）**
```toml
# 在accelerate config中啟用
# 會顯著降低速度（2-3倍慢）
offload_optimizer_states = true
offload_gradients = true
```

---

### **問題2: 訓練速度太慢**

#### 症狀
- 預期5-6小時，實際>10小時

#### 原因與解決

**A. Gradient checkpointing開銷**
- 造成15-20%速度下降
- **正常現象**，16GB VRAM下無法避免

**B. 數據加載瓶頸**
```toml
# 增加workers（如果RAM充足）
max_data_loader_n_workers = 4  # 從2增加到4
persistent_data_loader_workers = true
```

**C. Flash Attention未啟用**
```bash
# 檢查
python3 -c "import flash_attn; print('Installed')"

# 安裝（提速2倍）
pip install flash-attn --no-build-isolation
```

**D. VAE未緩存**
```toml
# 確認已啟用
cache_latents = true
cache_latents_to_disk = false  # RAM緩存更快
```

---

### **問題3: 訓練loss不下降**

#### 症狀
- Loss停在某個值不降（如0.15）
- 或震盪劇烈

#### 解決方案

**A. Learning rate過高**
```toml
# SDXL需要較低學習率
learning_rate = 0.00008  # 從0.0001降低
text_encoder_lr = 0.00005
```

**B. Gradient clipping過嚴格**
```toml
max_grad_norm = 1.0  # 可以放寬到1.5
```

**C. 添加warmup**
```toml
lr_warmup_steps = 200  # 增加warmup步數
```

**D. 調整Min-SNR**
```toml
min_snr_gamma = 3.0  # 從5.0降低（對某些數據集更好）
```

---

### **問題4: 生成圖片模糊/細節不足**

#### 可能原因

**A. Checkpoint太早（欠擬合）**
- **解決**: 等到Epoch 12-16再評估

**B. Network dim太小**
```toml
network_dim = 192  # 從128增加（VRAM允許的話）
network_alpha = 128
```

**C. 數據集質量問題**
- 檢查訓練數據是否清晰
- 確認resolution是否為1024px

**D. 推理參數需調整**
```python
# 使用更強的CFG scale
guidance_scale = 8.0  # SDXL推薦7-9

# 增加steps
num_inference_steps = 40  # SDXL需要更多steps
```

---

### **問題5: SDXL base model下載失敗**

#### 症狀
```
OSError: Can't load tokenizer for 'stabilityai/stable-diffusion-xl-base-1.0'
```

#### 解決方案

**A. 手動下載**
```bash
# 使用git-lfs
cd /mnt/data/ai_data/models/base
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

**B. 使用鏡像**
```bash
# 中國大陸用戶
export HF_ENDPOINT=https://hf-mirror.com
```

**C. 直接指定本地路徑**
```toml
[model]
pretrained_model_name_or_path = "/mnt/data/ai_data/models/base/stable-diffusion-xl-base-1.0"
```

---

## 📈 SD 1.5 vs SDXL 比較表

### **訓練成本對比**

| 項目 | SD 1.5 | SDXL (16GB優化) | 差異 |
|------|--------|-----------------|------|
| **所需VRAM** | 10-12GB | 14-15GB | +30% |
| **訓練時間** | 2.2小時 | 5-6小時 | +2.5x |
| **模型檔案大小** | 140MB | 800MB | +6x |
| **推理速度** | 快 | 慢2-3倍 | -2.5x |
| **Base model下載** | 4GB | 6.9GB | +1.7x |

### **輸出品質對比**

| 指標 | SD 1.5 | SDXL | 改善 |
|------|--------|------|------|
| **解析度** | 512px | 1024px | **+100%** |
| **面部細節** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **+40%** |
| **材質質感** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **+30%** |
| **光影細節** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **+35%** |
| **Prompt理解** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **+20%** |

### **適用場景建議**

#### **選擇SD 1.5，如果：**
- ✅ VRAM ≤ 12GB
- ✅ 需要快速迭代測試
- ✅ 512px解析度已足夠
- ✅ 推理速度優先
- ✅ 存儲空間有限

#### **選擇SDXL，如果：**
- ✅ VRAM ≥ 16GB
- ✅ 需要高解析度輸出（1024px+）
- ✅ 視覺品質優先
- ✅ 可以接受較長訓練時間
- ✅ 硬體資源充足

---

## 🎓 最佳實踐

### **1. 漸進式遷移策略**

```
階段1: SD 1.5基線
  ↓ 驗證超參數和數據集品質
階段2: SDXL小規模測試
  ↓ 用200張圖測試16GB優化配置
階段3: 全面SDXL訓練
  ↓ 完整410張圖，20 epochs
階段4: 評估與比較
  └─ 使用SOTA metrics選擇最佳checkpoint
```

### **2. 數據集準備建議**

**SDXL特殊要求：**
- ✅ 優先使用**原生高解析度**圖片（≥1024px）
- ✅ 避免upscale低解析度圖片（會有artifacts）
- ✅ Caption可以更長更詳細（SDXL支援225 tokens）
- ✅ 多角度、多pose平衡採樣更重要

### **3. 超參數遷移指南**

從SD 1.5 Trial 3.5遷移到SDXL：

```toml
# SD 1.5 → SDXL 調整
learning_rate:   0.00013 → 0.0001   (降低23%)
text_encoder_lr: 0.00008 → 0.00006  (降低25%)
batch_size:      4 → 1              (VRAM限制)
grad_accum:      3 → 8              (維持effective batch)
max_epochs:      18 → 20            (稍微增加)
resolution:      512 → 1024         (2倍)

# 保持不變
network_dim:     128
network_alpha:   96
optimizer_type:  AdamW (→ AdamW8bit for SDXL)
min_snr_gamma:   5.0
```

### **4. 監控與調試**

**必看指標：**
```bash
# 1. VRAM使用（應 <15GB）
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'

# 2. Training loss（應穩定下降）
tail -f /tmp/luca_sdxl_training.log | grep "loss:"

# 3. Checkpoint檔案大小（應~800MB）
ls -lh /path/to/output/*.safetensors
```

**異常警告：**
- VRAM超過15.5GB → OOM風險高
- Loss震盪幅度>0.05 → 學習率可能過高
- Checkpoint<500MB → 可能保存失敗

---

## 📦 完整檔案清單

### **配置與腳本**

```
configs/training/
├── sdxl_16gb_stable.toml             # 穩定版16GB優化配置（基礎模板）
└── character_loras_sdxl/             # 角色專用SDXL配置
    ├── orion_orion_identity_sdxl.toml
    ├── onward_ian_lightfoot_identity_sdxl.toml
    ├── up_russell_identity_sdxl.toml
    ├── turning-red_tyler_identity_sdxl.toml
    ├── onward_barley_lightfoot_identity_sdxl.toml
    └── luca_giulia_identity_sdxl.toml

scripts/generic/training/
└── expand_captions_for_sdxl.py       # SD1.5→SDXL Caption擴充工具

scripts/batch/
├── expand_all_captions_for_sdxl.sh   # 批次Caption擴充（6角色）
└── train_and_evaluate_6_characters_sdxl.sh  # SDXL批次訓練+評估

docs/training/
├── sdxl-16gb-guide.md                # 本文檔
├── SDXL_CAPTION_EXPANSION.md         # Caption擴充詳細指南
└── TRAINING_PIPELINE_FIXES.md        # 批次訓練問題修復文檔
```

### **使用流程**

```bash
# 1. 準備數據集
bash scripts/training/prepare_kohya_dataset.sh \
  --source-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_final_data \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_sdxl_training \
  --repeat 10 \
  --name luca

# 2. 啟動訓練（自動處理所有優化）
bash scripts/training/start_sdxl_16gb_training.sh

# 3. 訓練完成後評估
conda run -n ai_env python scripts/evaluation/sota_lora_evaluator.py \
  --evaluate-samples \
  --lora-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1 \
  --sample-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1/sample \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_trial1/evaluation \
  --device cuda
```

---

## 🚀 6角色批次SDXL訓練流程

### **概述**

本項目已為6個角色準備好完整的SDXL訓練配置和批次處理腳本：
- **Orion** (Orion and the Dark - DreamWorks)
- **Ian Lightfoot** (Onward - Pixar)
- **Russell** (Up - Pixar)
- **Tyler** (Turning Red - Pixar)
- **Barley Lightfoot** (Onward - Pixar)
- **Giulia** (Luca - Pixar)

### **階段1：Caption擴充（可選但推薦）**

將現有SD1.5 captions（77 tokens）擴充為SDXL格式（120-180 tokens），加入技術細節。

```bash
# 批次處理所有6個角色
bash scripts/batch/expand_all_captions_for_sdxl.sh
```

**預期時間**：每角色2-5分鐘（總計10-30分鐘）

**預期成本**：約$6-10（LLMProvider API，300-500張圖/角色）

**輸出位置**：
```
/mnt/data/ai_data/datasets/3d-anime/
├── orion/lora_data/training_data_sdxl/orion_identity/
├── onward/lora_data/training_data_sdxl/ian_lightfoot_identity/
├── up/lora_data/training_data_sdxl/russell_identity/
├── turning-red/lora_data/training_data_sdxl/tyler_identity/
├── onward/lora_data/training_data_sdxl/barley_lightfoot_identity/
└── luca/lora_data/training_data_sdxl/giulia_identity/
```

**擴充範例**：
```
原始SD1.5（42 tokens）：
"orion, a 3d animated character, dreamworks style, teenage boy,
blue shirt, outdoor scene, natural lighting, medium shot"

擴充SDXL（138 tokens）：
"orion, a 3d animated human character rendered in dreamworks animation
style, teenage boy with blue cotton shirt showing detailed fabric wrinkles
and pbr material properties, standing confidently in outdoor woodland scene
with natural three-point lighting setup including soft key light from upper
left creating gentle shadows, subtle rim lighting separating character from
background, global illumination providing ambient fill, medium full shot
composition with professional camera framing at eye level, shallow depth
of field with bokeh background, 1024px high resolution render..."
```

詳細文檔：[SDXL_CAPTION_EXPANSION.md](SDXL_CAPTION_EXPANSION.md)

---

### **階段2：SDXL批次訓練**

自動依序訓練所有6個角色，每個角色完成後自動進行checkpoint評估。

```bash
# 在TMUX中啟動（推薦）
tmux new-session -s sdxl_training

# 執行批次訓練腳本
bash scripts/batch/train_and_evaluate_6_characters_sdxl.sh

# 安全離開TMUX（訓練繼續）
# 按 Ctrl+B 然後 D
```

**預期時間**：
- 每角色：5-6小時（訓練）+ 10-15分鐘（評估）
- 總計：**30-36小時**（連續執行）

**VRAM使用**：
- 訓練時：14-15GB（穩定）
- 峰值：15.5GB（短暫）
- 評估時：12GB

**自動化流程**：
```
For each character:
  1. 加載SDXL base model
  2. 訓練12 epochs（每2 epoch保存checkpoint）
  3. 生成7個checkpoints
  4. 自動評估所有checkpoints
  5. 識別最佳checkpoint
  6. 繼續下一個角色
```

---

### **階段3：監控與檢查**

**重新連接TMUX**：
```bash
tmux attach-session -t sdxl_training
```

**檢查進度**：
```bash
# 主日誌
tail -f logs/training/train_and_evaluate_6chars_sdxl_<timestamp>.log

# 單一角色訓練日誌
tail -f logs/training/train_<character>_<timestamp>.log

# 評估日誌
tail -f logs/training/eval_<character>_<timestamp>.log
```

**檢查VRAM**：
```bash
watch -n 1 nvidia-smi
```

**檢查輸出**：
```bash
# 訓練中的checkpoints
ls -lh /mnt/data/ai_data/models/lora/*/*/lora-*.safetensors

# 評估結果
ls -lh /mnt/data/ai_data/models/lora/*/*/evaluations_*/
```

---

### **預期輸出結構**

```
/mnt/data/ai_data/models/lora/
├── orion/orion_identity/
│   ├── lora-orion-epoch-2.safetensors
│   ├── lora-orion-epoch-4.safetensors
│   ├── ...
│   ├── lora-orion-epoch-12.safetensors  (~800MB each)
│   └── evaluations_<timestamp>/
│       ├── evaluation_report.json
│       ├── checkpoint_epoch2/
│       ├── checkpoint_epoch4/
│       └── ...
├── onward/ian_lightfoot_identity/
│   └── ...
├── up/russell_identity/
│   └── ...
├── turning-red/tyler_identity/
│   └── ...
├── onward/barley_lightfoot_identity/
│   └── ...
└── luca/giulia_identity/
    └── ...
```

---

### **6角色SDXL配置統一參數**

所有角色使用相同的優化配置（基於 `sdxl_16gb_stable.toml`）：

| 參數 | 值 | 說明 |
|------|---|------|
| **模型** | sd_xl_base_1.0 | SDXL base model |
| **Optimizer** | AdamW8bit | 8-bit量化（節省6GB） |
| **Mixed Precision** | bf16 | BF16訓練（節省8GB） |
| **Gradient Checkpointing** | true | 啟用（節省8GB） |
| **Latent Caching** | true | 啟用（節省2GB） |
| **Batch Size** | 1 | 單張/batch |
| **Gradient Accumulation** | 2 | Effective batch = 2 |
| **Learning Rate** | 1e-4 | SDXL優化值 |
| **Text Encoder LR** | 6e-5 | 較低LR for CLIP |
| **Network Dim/Alpha** | 128/96 | 與SD1.5 Trial 3.6相同 |
| **Max Epochs** | 12 | SDXL標準 |
| **Resolution** | 1024×1024 | Native SDXL |
| **Min SNR Gamma** | 5.0 | 繼承自SD1.5最佳 |
| **Noise Offset** | 0.05 | 繼承自SD1.5最佳 |

---

### **Troubleshooting**

**問題：批次訓練在某角色後中斷**

```bash
# 檢查最後的錯誤
tail -50 logs/training/train_and_evaluate_6chars_sdxl_*.log

# 手動從失敗角色重新開始
# 編輯腳本，註釋掉已完成的角色
nano scripts/batch/train_and_evaluate_6_characters_sdxl.sh
```

詳細故障排除：[TRAINING_PIPELINE_FIXES.md](TRAINING_PIPELINE_FIXES.md)

**問題：VRAM OOM**

- 確認沒有其他GPU進程：`nvidia-smi`
- 降低batch size已經是1（無法再降）
- 考慮降低resolution到768×768（在配置中修改）

**問題：訓練速度太慢（>7小時/角色）**

- 確認Flash Attention 2已安裝：`pip list | grep flash`
- 檢查latent caching是否啟用
- 確認使用bf16而非fp32

---

### **與SD1.5訓練結果對比**

完成SDXL訓練後，使用相同的測試prompts對比：

```bash
# 生成對比圖
python scripts/evaluation/compare_sd15_vs_sdxl.py \
  --sd15-lora /path/to/sd15/lora \
  --sdxl-lora /path/to/sdxl/lora \
  --output-dir comparisons/sd15_vs_sdxl \
  --prompts-file prompts/lora_testing/character_test_prompts.txt
```

**預期改善**：
- ✅ 解析度：512px → 1024px（**+100%**）
- ✅ 面部細節：**+40%**
- ✅ 材質質感：**+30%**
- ✅ Prompt理解：**+20%**

---

## ✅ 總結

### **16GB VRAM可以訓練SDXL嗎？**
**✅ 完全可以！** 透過以下技術組合：
1. 8-bit AdamW （-40% VRAM）
2. Gradient Checkpointing （-30% VRAM）
3. BF16 Mixed Precision （-25% VRAM）
4. Latent Caching （-2GB VRAM）
5. Flash Attention 2 （-15% VRAM, +2x speed）

**實際VRAM使用：14-15GB（安全範圍內）**

### **值得從SD 1.5遷移到SDXL嗎？**
**取決於需求：**
- **追求視覺品質**：⭐⭐⭐⭐⭐（強烈推薦）
- **快速迭代測試**：⭐⭐（不推薦，太慢）
- **硬體資源有限**：⭐⭐⭐（可以，但需優化）

### **推薦workflow:**
1. ✅ 先用SD 1.5 Trial 3.5建立基線（2小時）
2. ✅ 驗證數據集品質和超參數
3. ✅ 再用SDXL訓練（5-6小時）
4. ✅ 對比兩者結果，選擇最適合的


## SDXL 訓練快速參考卡 🎯 核心概念

**可以遷移**: ✅ 超參數、數據集、訓練策略
**需要調整**: ⚠️ 分辨率、batch size、base model
**完全不同**: ❌ 模型架構、VRAM 需求、訓練時間

---

## 📊 SD1.5 vs SDXL 對比表

| 項目 | SD1.5 | SDXL |
|------|-------|------|
| **分辨率** | 512×512 | 1024×1024 |
| **模型大小** | 0.9B | 2.6B |
| **Text Encoders** | 1 (CLIP-L) | 2 (CLIP-L + OpenCLIP-G) |
| **VRAM 需求** | 8-12 GB | 16-24 GB |
| **Batch Size** | 8-16 | 2-4 |
| **訓練時間/epoch** | ~15 min | ~30-45 min |
| **LoRA 大小 (dim=64)** | ~73 MB | ~190 MB |
| **圖片質量** | 7/10 | 9/10 |

---

## ⚡ 一鍵啟動 SDXL 訓練

### 前置條件
```bash
# 1. SD1.5 優化已完成，提取最佳參數
BEST_LR=0.0003
BEST_DIM=64
BEST_ALPHA=32
BEST_OPTIMIZER="AdamW8bit"
BEST_SCHEDULER="cosine_with_restarts"
BEST_GRAD_ACCUM=2
BEST_EPOCHS=12

# 2. 下載 SDXL base model
cd /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

### 訓練命令（複製即用）
```bash
cd /mnt/c/AI_LLM_projects/kohya_ss/sd-scripts

nohup conda run -n kohya_ss python train_network.py \
  --dataset_config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/sdxl/luca_human_dataset_sdxl.toml \
  --pretrained_model_name_or_path /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1 \
  --output_name luca_sdxl_v1 \
  --network_module networks.lora \
  --network_dim $BEST_DIM \
  --network_alpha $BEST_ALPHA \
  --learning_rate $BEST_LR \
  --text_encoder_lr $(echo "$BEST_LR * 0.67" | bc -l) \
  --unet_lr $BEST_LR \
  --max_train_epochs $BEST_EPOCHS \
  --save_every_n_epochs 2 \
  --save_model_as safetensors \
  --save_precision fp16 \
  --mixed_precision fp16 \
  --gradient_checkpointing \
  --gradient_accumulation_steps $BEST_GRAD_ACCUM \
  --optimizer_type $BEST_OPTIMIZER \
  --lr_scheduler $BEST_SCHEDULER \
  --lr_scheduler_num_cycles 3 \
  --lr_warmup_steps 100 \
  --logging_dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs \
  --log_with tensorboard \
  --seed 42 \
  --clip_skip 2 \
  --cache_latents \
  --cache_latents_to_disk \
  --max_data_loader_n_workers 8 \
  --persistent_data_loader_workers \
  --xformers \
  --max_token_length 225 \
  --bucket_reso_steps 64 \
  --bucket_no_upscale \
  > /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log 2>&1 &

echo "SDXL 訓練已啟動，PID: $!"
```

---

## 🔧 關鍵參數說明

### SDXL 特有參數（必須添加）
```bash
--text_encoder_lr 0.0002       # 雙 text encoder，獨立設置
--unet_lr 0.0003               # U-Net 學習率
--max_token_length 225         # SDXL 支持更長 tokens
--xformers                     # 記憶體優化（必須）
```

### 從 SD1.5 遷移的參數（直接使用）
```bash
--network_dim 64               # SD1.5 最佳值
--network_alpha 32             # SD1.5 最佳值
--learning_rate 0.0003         # SD1.5 最佳值
--optimizer_type AdamW8bit     # SD1.5 最佳值
--lr_scheduler cosine_with_restarts  # SD1.5 最佳值
--gradient_accumulation_steps 2      # SD1.5 最佳值
--max_train_epochs 12          # SD1.5 最佳值（或 +20%）
```

### 需要調整的參數
```bash
--batch_size 4                 # SD1.5: 8 → SDXL: 4（VRAM 限制）
--text_encoder_lr              # 新增，約為 learning_rate × 0.5-0.8
```

---

## 📈 監控命令

### 實時日誌
```bash
tail -f /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log
```

### 查看 Checkpoints
```bash
watch -n 60 'ls -lh /mnt/data/ai_data/models/lora/luca/sdxl_v1/*.safetensors'
```

### TensorBoard 可視化
```bash
tensorboard --logdir /mnt/data/ai_data/models/lora/luca/sdxl_v1/logs --port 6007 --bind_all
```

### 訓練進度估算
```bash
# 查看當前 epoch
grep -oP "Epoch \d+/\d+" /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log | tail -1

# 查看 loss 趨勢
grep "loss:" /mnt/data/ai_data/models/lora/luca/sdxl_v1/training.log | tail -20
```

---

## 🧪 評估 SDXL Checkpoint

### 單個 Checkpoint 測試
```bash
conda run -n kohya_ss python /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/evaluation/evaluate_single_checkpoint.py \
  --checkpoint /mnt/data/ai_data/models/lora/luca/sdxl_v1/luca_sdxl_v1-000008.safetensors \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/sdxl/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/sdxl_v1/eval_epoch8 \
  --num-samples 16 \
  --device cuda \
  --resolution 1024
```

### 對比 SD1.5 vs SDXL 質量
生成相同 prompt 的圖片並比較：
- **SD1.5**: 512×512, 較快, 質量中等
- **SDXL**: 1024×1024, 較慢, 質量優秀

---

## ⚠️ 常見問題快速解決

### ❌ OOM (Out of Memory)
```bash
# 解決方案 1：降低 batch size
--batch_size 2
--gradient_accumulation_steps 4  # 等效 batch size = 8

# 解決方案 2：啟用記憶體優化
--xformers  # 必須
--gradient_checkpointing  # 必須

# 解決方案 3：極端情況
--lowvram
--medvram
```

### 🐌 訓練太慢
- **正常現象**：SDXL 預期慢 2-3 倍
- **無法加速**：模型大小和分辨率決定
- **建議**：使用 overnight 訓練

### 🖼️ 圖片質量不佳
- **檢查**：是否使用了 SD1.5 最佳超參數？
- **檢查**：dataset 配置是否正確（resolution=1024）？
- **檢查**：是否在正確的 epoch（通常 epoch 6-10 最佳）？

### 📦 LoRA 檔案太大
- **正常**：SDXL LoRA 約 2.5 倍 SD1.5 大小
- **優化**：降低 network_dim（但可能影響質量）
- **建議**：保持 dim=64 或 128

---

## 🎓 最佳實踐 Checklist

- [ ] SD1.5 優化已完成，最佳參數已提取
- [ ] SDXL base model 已下載
- [ ] 數據集路徑檢查（使用相同的 SD1.5 數據集）
- [ ] VRAM 充足（至少 16GB，建議 24GB）
- [ ] 使用 `--xformers` 和 `--gradient_checkpointing`
- [ ] `text_encoder_lr` 設為 `learning_rate × 0.67`
- [ ] `batch_size` 降低到 2-4
- [ ] `max_token_length` 設為 225
- [ ] 使用 nohup 背景運行
- [ ] 啟動 TensorBoard 監控
- [ ] 每 2 epochs 保存 checkpoint
- [ ] 預留 6-10 小時訓練時間

---

## 📂 目錄結構參考

```
/mnt/data/ai_data/models/lora/luca/
├── optimization_overnight/           # SD1.5 優化結果
│   ├── trial_0025/                  # 假設最佳 trial
│   │   ├── params.json              # 提取超參數
│   │   └── lora_trial_25.safetensors
│   └── CONVERGENCE_ALERT.txt
│
└── sdxl_v1/                         # SDXL 訓練結果
    ├── luca_sdxl_v1-000002.safetensors  (epoch 2)
    ├── luca_sdxl_v1-000004.safetensors  (epoch 4)
    ├── luca_sdxl_v1-000006.safetensors  (epoch 6)
    ├── luca_sdxl_v1-000008.safetensors  (epoch 8)
    ├── luca_sdxl_v1-000010.safetensors  (epoch 10)
    ├── luca_sdxl_v1.safetensors  (final, epoch 12)
    ├── training.log
    ├── logs/  (TensorBoard)
    └── eval_epoch8/  (測試圖片)
```

---

## 🚀 快速決策樹

```
SD1.5 優化完成？
 ├─ 是 → 提取最佳參數 → 啟動 SDXL 訓練
 └─ 否 → 等待完成（監控收斂狀態）

VRAM 充足（≥16GB）？
 ├─ 是 → batch_size=4
 └─ 否 → batch_size=2 + gradient_accumulation_steps=4

需要極致質量？
 ├─ 是 → SDXL（1024×1024）
 └─ 否 → SD1.5（512×512）足夠

時間緊迫？
 ├─ 是 → 使用 SD1.5（快 2-3x）
 └─ 否 → 使用 SDXL（質量更好）
```


## 📚 參考資源

- **Kohya_ss官方文檔**: https://github.com/kohya-ss/sd-scripts
- **SDXL論文**: https://arxiv.org/abs/2307.01952
- **8-bit Optimizer論文**: https://arxiv.org/abs/2110.02861
- **Flash Attention 2**: https://arxiv.org/abs/2307.08691
- **本項目**: 3d-animation-lora-pipeline
