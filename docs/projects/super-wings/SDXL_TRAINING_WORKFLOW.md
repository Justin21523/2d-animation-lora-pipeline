# Super Wings SDXL LoRA Training - Complete Workflow

**專案概述**: 為9個Super Wings擬人化飛機/直升機角色訓練高品質的SDXL LoRA adapters

**角色類型**: 擬人化交通工具角色 (anthropomorphic planes/helicopters)
- 具有獨特的顏色設計、外型特徵
- 保留飛機/直升機的機械結構特徵
- 融合卡通化的表情和個性

**資料集**: 1,242張角色圖片，分佈於9個角色
- 已完成SAM2 instance segmentation
- 已完成CLIP clustering分類
- 位置: `/mnt/data/datasets/general/super-wings/lora_data/characters/`

---

## Pipeline Overview

```
原始圖片 (1,242張)
    ↓
[1] 智能升級 (RealESRGAN 2x/4x)
    ↓
[2] 品質增強 (CodeFormer + CLAHE + Denoising)
    ↓
[3] Letterbox Padding (1024x1024, 黑邊)
    ↓
[4] Data Augmentation (確保每角色≤300張)
    ↓
[5] LLMProvider API Caption生成 (Haiku 3.5)
    ↓
[6] SDXL LoRA訓練 (Kohya_ss)
    ↓
[7] Checkpoint評估與選擇
    ↓
最終LoRA模型 (9個角色)
```

---

## Prerequisites

### 1. Environment Setup

```bash
# 確認conda環境
conda activate ai_env

# 確認Python版本
python --version  # Should be 3.10+

# 確認CUDA可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### 2. Required Models

```bash
# SDXL base model
ls /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors

# SDXL VAE
ls /mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors

# Kohya_ss sd-scripts
ls ~/sd-scripts/train_network.py
```

### 3. API Keys

```bash
# 設定LLMProvider API key (用於caption生成)
export LLM_VENDOR_API_KEY="sk-ant-..."

# 驗證
python -c "import os; print('✓ API key set' if os.getenv('LLM_VENDOR_API_KEY') else '✗ API key missing')"
```

---

## Step-by-Step Workflow

### Stage 1-5: Preprocessing Pipeline

**腳本**: `prepare_super_wings_sdxl_pipeline.py`

這個master orchestrator會執行所有preprocessing階段:

```bash
cd /mnt/c/ai_projects/3d-animation-lora-pipeline

# 執行完整pipeline (所有9個角色)
conda run -n ai_env python scripts/batch/prepare_super_wings_sdxl_pipeline.py
```

**Pipeline細節**:

1. **Upscaling** (7-10分鐘)
   - RealESRGAN_x4plus模型
   - Auto-decision: 2x或4x (基於原始尺寸)
   - 目標: ~1024px shortest edge

2. **Enhancement** (14-20分鐘)
   - CodeFormer: 3D角色臉部修復
   - CLAHE: 適應性對比增強
   - CNN denoising: 去除壓縮噪點
   - Sharpening: 溫和銳化

3. **Letterbox Padding** (1-2分鐘)
   - 目標: 精確1024x1024
   - 保留原始比例
   - 黑色邊框填充

4. **Data Augmentation** (2-3分鐘)
   - 每角色最多300張圖片
   - 對<300張的角色進行augmentation:
     * 水平翻轉 (30%機率，保護非對稱設計)
     * 小角度旋轉 (±5度)
     * 亮度調整 (0.9-1.1x)
   - **不使用**: 色彩抖動、透視變形 (會破壞3D材質)

5. **Caption Generation** (2.5-4分鐘, ~$0.20成本)
   - Model: LLMProvider 3.5 Haiku (成本效益最高)
   - 使用角色特定prompts (從`docs/projects/super-wings/character/`讀取)
   - 輸出格式: Kohya format (圖片+配對的.txt文件)
   - 強調3D動畫風格特徵 (materials, lighting, camera)

**預期輸出**:

```
/mnt/data/datasets/general/super-wings/training_data/
├── jet/
│   └── 10_jet/              # Kohya format: repeat_concept
│       ├── frame_0001.png
│       ├── frame_0001.txt   # Caption file
│       ├── frame_0002.png
│       └── frame_0002.txt
├── jerome/
│   └── 10_jerome/
│       └── ...
└── ... (7 more characters)
```

---

### Stage 6: Generate Training Configs

**腳本**: `generate_super_wings_sdxl_configs.py`

為每個角色生成最佳化的SDXL training configs:

```bash
# 自動生成所有角色的TOML configs
conda run -n ai_env python scripts/batch/generate_super_wings_sdxl_configs.py
```

**Config特點** (基於成功的Pixar角色LoRA結果):

- **Best Epoch**: 1-2 (不是5!)
- **Network Dimensions**: dim=64, alpha=32
- **Optimizer**: AdamW8bit (節省~6GB VRAM)
- **Mixed Precision**: BF16 (節省~8GB VRAM)
- **Gradient Checkpointing**: enabled (節省~8GB VRAM)
- **CLIP Skip**: 2 (對3D角色LoRA關鍵)
- **RTX 5080最佳化**: xformers=false, sdpa=true
- **No Augmentations**: color_aug=false, flip_aug=false

**自動調整參數** (基於dataset size):

```python
# <200 images
max_train_epochs = 3
learning_rate = 0.00012

# 200-300 images
max_train_epochs = 2
learning_rate = 0.0001
```

**輸出位置**:

```
/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras/
├── super-wings-jet-sdxl.toml
├── super-wings-jerome-sdxl.toml
├── ... (7 more configs)
└── config_generation_report.json
```

---

### Stage 7: Train All LoRAs

**腳本**: `train_super_wings_sdxl_loras.py`

按順序訓練所有9個角色的LoRAs (避免GPU memory衝突):

```bash
# Sequential training (一次訓練一個角色)
conda run -n ai_env python scripts/batch/train_super_wings_sdxl_loras.py
```

**Training時間估計**:

- 每個角色: 30-60分鐘 (200-300張圖片, 2 epochs)
- 總時間: ~4.5-9小時 (9個角色)

**即時監控**:

```bash
# 查看training logs
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/super_wings_training/*.log

# 查看TensorBoard (如果啟用)
tensorboard --logdir /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/super_wings_training/
```

**輸出Checkpoints**:

```
/mnt/c/ai_models/diffusion/lora/sdxl/super-wings/
├── jet/
│   ├── super-wings-jet-sdxl-000001.safetensors  # Epoch 1
│   └── super-wings-jet-sdxl-000002.safetensors  # Epoch 2
├── jerome/
│   ├── super-wings-jerome-sdxl-000001.safetensors
│   └── super-wings-jerome-sdxl-000002.safetensors
└── ... (7 more characters)
```

---

### Stage 8: Evaluate Checkpoints

**腳本**: `evaluate_super_wings_loras.py`

為所有角色的checkpoints生成測試圖片:

```bash
# 自動評估所有LoRAs
conda run -n ai_env python scripts/batch/evaluate_super_wings_loras.py
```

**測試設定**:

- **Prompts**: 5個測試prompts per角色
  - 基本身份測試 (front view, neutral)
  - 3/4視角測試
  - 飛行動作測試
  - 特寫細節測試
  - 場景情境測試

- **LoRA Scales**: [0.7, 0.8, 0.9, 1.0]
- **Seeds**: [42, 123, 456] (測試一致性)
- **Total Images**: 每checkpoint ~60張測試圖 (5 prompts × 4 scales × 3 seeds)

**參考角色文檔**:

所有prompts都基於 `docs/projects/super-wings/character/*.md` 中的角色描述:

- Jett: "Red jet plane with yellow propeller, main protagonist"
- Jerome: "White helicopter with blue rotor, delivery supervisor"
- Flip: "Green stunt plane, acrobatic expert"
- ... (參見 `configs/training/super_wings_caption_config.yaml`)

**評估輸出**:

```
/mnt/data/outputs/lora_evaluation/super_wings/
├── jet/
│   ├── super-wings-jet-sdxl-000001/
│   │   ├── prompt_01/
│   │   │   ├── scale0.7_seed42.png
│   │   │   ├── scale0.8_seed42.png
│   │   │   └── ...
│   │   └── prompt_02/
│   ├── super-wings-jet-sdxl-000002/
│   └── evaluation_report.json
└── evaluation_summary.json
```

---

## Quality Checks

### 1. 驗證Preprocessing輸出

```bash
# 檢查所有圖片都是1024x1024
find /mnt/data/datasets/general/super-wings/processed/padded/ -name "*.png" -exec identify -format "%f: %wx%h\n" {} \; | grep -v "1024x1024" || echo "✓ All verified"

# 檢查caption文件存在
cd /mnt/data/datasets/general/super-wings/training_data
for char in beard bello flip jerone jet paul shark tank tony; do
  img_count=$(find $char -name "*.png" | wc -l)
  txt_count=$(find $char -name "*.txt" | wc -l)
  echo "$char: $img_count images, $txt_count captions"
done
```

### 2. 驗證Training Configs

```bash
# 檢查所有configs生成
ls -lh /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras/

# 檢查config內容
head -50 /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/super_wings_loras/super-wings-jet-sdxl.toml
```

### 3. 監控Training Progress

```bash
# 檢查訓練是否在進行
ps aux | grep train_network.py

# 檢查GPU使用率
nvidia-smi -l 5

# 查看最新log
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/super_wings_training/*.log
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

```bash
# 減少batch size (編輯TOML config)
train_batch_size = 1  # Already minimal

# 增加gradient accumulation
gradient_accumulation_steps = 4  # From 2 to 4

# 確認CUDA memory config設定
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### LLMProvider API Rate Limits

```bash
# 如果遇到rate limiting，在pipeline script中增加batch-size間隔
# 編輯 prepare_super_wings_sdxl_pipeline.py:
# cmd.extend(["--batch-size", "5"])  # 從10降到5
# cmd.extend(["--delay", "2.0"])     # 新增2秒延遲
```

### Training太慢

```bash
# 檢查是否使用了正確的優化
# 1. 確認xformers disabled (RTX 5080)
# 2. 確認SDPA enabled
# 3. 確認latent caching enabled

# 編輯TOML config:
cache_latents = true
cache_latents_to_disk = true
sdpa = true
xformers = false
```

---

## Expected Results

基於成功的Pixar角色LoRA訓練經驗:

✅ **Best Checkpoint**: Epoch 1 或 Epoch 2
✅ **Character Identity**: 精準重現角色特徵 (顏色、設計、材質)
✅ **Flexibility**: 可生成不同視角、姿勢、光照
✅ **Composition**: 可與其他LoRAs組合使用
✅ **Prompt Sensitivity**: 對camera angle, lighting等prompt敏感

**測試Prompt範例** (成功訓練的LoRA應能生成):

```
a 3d animated character, Jett, red jet plane with yellow propeller,
Pixar-style rendering, smooth shading, glossy metallic surface,
three-quarter view, flying pose, blue sky background,
cinematic lighting, high quality 3d render
```

---

## Next Steps After Training

### 1. Select Best Checkpoints

```bash
# 視覺比較evaluation輸出
ls /mnt/data/outputs/lora_evaluation/super_wings/jet/

# 通常Epoch 1是最佳的
```

### 2. Deploy to Production

```bash
# 複製最佳checkpoints到production目錄
mkdir -p /mnt/c/ai_models/diffusion/lora/sdxl/production/super_wings

for char in beard bello flip jerone jet paul shark tank tony; do
  # 假設Epoch 1是最佳的
  cp "/mnt/c/ai_models/diffusion/lora/sdxl/super-wings/$char/super-wings-${char}-sdxl-000001.safetensors" \
     "/mnt/c/ai_models/diffusion/lora/sdxl/production/super_wings/"
done
```

### 3. Test Multi-LoRA Composition

```bash
# 測試2-3個角色組合
# 範例: Jett + Jerome一起飛行的場景
```

---

## Cost Analysis

| 項目 | 成本 |
|------|------|
| GPU Time (RTX 5080) | $0 (自有硬體) |
| LLMProvider API (1,242張 × Haiku 3.5) | ~$0.12-0.20 |
| Storage (~20GB) | $0 (本地儲存) |
| **總成本** | **~$0.20** |

**時間投入**:

| 階段 | 時間 |
|------|------|
| Preprocessing (Stage 1-5) | 30-40分鐘 |
| Config生成 | 1-2分鐘 |
| Training (9角色) | 4.5-9小時 |
| Evaluation | 1-2小時 |
| **總時間** | **~6-12小時** |

---

## References

- **Character Documentation**: `docs/projects/super-wings/character/*.md`
- **Caption Config**: `configs/training/super_wings_caption_config.yaml`
- **Training Template**: `configs/training/character_lora_sdxl_template.toml`
- **Successful Results**: `docs/progress/sdxl_lora_results.md`
- **16GB VRAM Guide**: `docs/training/sdxl-16gb-guide.md`

---

## Quick Start (一鍵執行)

```bash
# 1. 設定API key
export LLM_VENDOR_API_KEY="sk-ant-..."

# 2. 執行preprocessing
conda run -n ai_env python scripts/batch/prepare_super_wings_sdxl_pipeline.py

# 3. 生成training configs
conda run -n ai_env python scripts/batch/generate_super_wings_sdxl_configs.py

# 4. 開始訓練 (overnight job)
nohup conda run -n ai_env python scripts/batch/train_super_wings_sdxl_loras.py > /tmp/super_wings_training.log 2>&1 &

# 5. 次日評估結果
conda run -n ai_env python scripts/batch/evaluate_super_wings_loras.py
```

---

**Last Updated**: 2025-12-13
**Status**: Ready for execution
