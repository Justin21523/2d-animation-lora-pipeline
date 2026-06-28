# Character LoRA Training Pipeline

## Overview

完整的 6 角色 Identity LoRA 訓練與評估流程。每個角色完成訓練+評估後才進入下一個，確保最佳 checkpoint 選擇。

## Pipeline Architecture

```
訓練準備 → 訓練執行 → Checkpoint 評估 → 最佳選擇 → 下一個角色
```

### 三階段處理流程

每個角色經過以下三個階段：

**階段 1: LoRA 訓練**
- Kohya SS 訓練引擎
- 每 2 個 epoch 自動生成樣本圖片（sample_prompts）
- 完整訓練日誌記錄
- 輸出：多個 checkpoint (.safetensors)

**階段 2: Checkpoint 全面評估**
- 使用 `test_lora_checkpoints.py` 評估所有 checkpoints
- 測試配置：4 變體 × 6 prompts = 24 張圖片/checkpoint
- 生成評估報告 JSON
- 輸出：視覺化測試結果 + 量化指標

**階段 3: 最佳 Checkpoint 選擇**
- 自動從評估報告提取最佳 checkpoint
- 顯示 checkpoint 名稱、epoch、評分
- 記錄到總結報告

## Character List

| 順序 | 角色 | 電影 | 圖片數 | 訓練步數/epoch | Epochs | 預估時間 |
|------|------|------|--------|---------------|--------|---------|
| 1 | Orion | Orion and the Dark (2024) | 261 | 3,915 (15 repeats) | 12 | ~3-4h |
| 2 | Ian Lightfoot | Onward (2020) | 359 | 3,590 (10 repeats) | 14 | ~3-4h |
| 3 | Russell | Up (2009) | 243 | 3,645 (15 repeats) | 16 | ~3-4h |
| 4 | Tyler | Turning Red (2022) | 276 | 4,140 (15 repeats) | 14 | ~3-4h |
| 5 | Barley Lightfoot | Onward (2020) | 258 | 2,580 (10 repeats) | 14 | ~2-3h |
| 6 | Giulia | Luca (2021) | 273 | 2,730 (10 repeats) | 14 | ~2-3h |

**總預估時間**: 18-24 小時（依序執行，避免 GPU 衝突）

## Quick Start

### 1. 訓練準備確認

```bash
# 確認訓練目錄已建立
ls -la /mnt/data/ai_data/datasets/3d-anime/*/lora_data/training_data/*/

# 確認配置文件
ls -la configs/training/character_loras/*.toml

# 確認測試 prompts
ls -la prompts/lora_testing/*_identity_prompts.json
```

### 2. 啟動完整訓練流程

```bash
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline
bash scripts/batch/train_and_evaluate_6_characters.sh
```

### 3. 監控進度

**主日誌**（總體進度）:
```bash
tail -f logs/training/train_evaluate_6chars_<timestamp>.log
```

**當前角色訓練日誌**:
```bash
tail -f logs/training/train_<character>_<timestamp>.log
```

**當前角色評估日誌**:
```bash
tail -f logs/training/eval_<character>_<timestamp>.log
```

## Training Configurations

### Directory Structure

**訓練數據** (Kohya 格式):
```
/mnt/data/ai_data/datasets/3d-anime/<movie>/lora_data/training_data/
└── <character>_identity/
    └── <repeats>_<class_name>/
        ├── image_001.png
        ├── image_001.txt
        ├── image_002.png
        ├── image_002.txt
        └── ...
```

**配置文件**:
```
configs/training/character_loras/
├── orion_orion_identity.toml
├── onward_ian_lightfoot_identity.toml
├── up_russell_identity.toml
├── turning-red_tyler_identity.toml
├── onward_barley_lightfoot_identity.toml
└── luca_giulia_identity.toml
```

**測試 Prompts**:
```
prompts/lora_testing/
├── orion_identity_prompts.json
├── ian_lightfoot_identity_prompts.json
├── russell_identity_prompts.json
├── tyler_identity_prompts.json
├── barley_lightfoot_identity_prompts.json
└── giulia_identity_prompts.json
```

### Training Parameters (Trial 3.6 Proven)

所有角色使用統一的最佳訓練參數：

```toml
# Network Architecture
network_dim = 64
network_alpha = 32
network_dropout = 0.1

# Optimizer
optimizer_type = "AdamW8bit"
learning_rate = 6e-5
unet_lr = 6e-5
text_encoder_lr = 3e-5
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 2

# Regularization
min_snr_gamma = 5.0
noise_offset = 0.05
max_grad_norm = 1.0

# Data Augmentation (3D Conservative)
color_aug = false
flip_aug = false
random_crop = false

# Sample Generation
sample_every_n_epochs = 2
save_every_n_epochs = 2
```

## Output Structure

### Per-Character Outputs

每個角色訓練完成後的輸出結構：

```
/mnt/data/ai_data/models/lora/<movie>/<character>_identity/
├── <character>_identity_lora_epoch_002.safetensors
├── <character>_identity_lora_epoch_004.safetensors
├── <character>_identity_lora_epoch_006.safetensors
├── ...
├── <character>_identity_lora.safetensors  # Final checkpoint
│
├── evaluations_<timestamp>/
│   ├── checkpoint_epoch_002/
│   │   ├── prompt_1_portrait_var_1.png
│   │   ├── prompt_1_portrait_var_2.png
│   │   ├── prompt_1_portrait_var_3.png
│   │   ├── prompt_1_portrait_var_4.png
│   │   ├── prompt_2_fullbody_var_1.png
│   │   └── ...  # 6 prompts × 4 variations = 24 images
│   ├── checkpoint_epoch_004/
│   ├── checkpoint_epoch_006/
│   ├── ...
│   ├── evaluation_report.json  # ← 最佳 checkpoint 資訊
│   └── comparison_grid.png     # 視覺化比較
│
└── logs/
    └── <timestamp>/
        ├── events.out.tfevents.*  # TensorBoard logs
        └── train.log
```

### Evaluation Report Format

`evaluation_report.json` 範例：

```json
{
  "best_checkpoint": {
    "name": "orion_identity_lora_epoch_008.safetensors",
    "epoch": 8,
    "score": 0.87,
    "metrics": {
      "clip_score": 0.89,
      "consistency": 0.85,
      "quality": 0.88
    }
  },
  "checkpoints": [
    {
      "name": "orion_identity_lora_epoch_002.safetensors",
      "epoch": 2,
      "score": 0.72,
      "path": ".../epoch_002.safetensors"
    },
    // ... 其他 checkpoints
  ],
  "test_prompts": [ /* ... */ ],
  "timestamp": "2025-11-21T09:30:00"
}
```

## Checkpoint Testing Details

### Test Prompts Per Character

每個角色有 6 個測試 prompts，涵蓋不同視角和場景：

1. **Portrait** - 中性表情，工作室燈光
2. **Full Body** - 站立姿態，自然光
3. **Close-up Face** - 微笑，柔和燈光
4. **Three-quarter View** - 戲劇性燈光
5. **Side Profile** - 電影級燈光
6. **Outdoor Scene** - 黃金時段燈光

### Evaluation Metrics

自動評估指標：

- **CLIP Score**: Prompt-image 相似度
- **Consistency**: 不同 seed 間的一致性
- **Quality**: 圖片質量評分
- **Character Fidelity**: 角色特徵保留度

## Monitoring & Logs

### Real-time Monitoring

**主日誌文件**:
```
logs/training/train_evaluate_6chars_<timestamp>.log
```

包含：
- Pipeline 整體進度
- 每個角色的開始/完成時間
- Checkpoint 數量統計
- 最佳 checkpoint 資訊
- 錯誤和警告訊息

**個別角色日誌**:
```
logs/training/train_<character>_<timestamp>.log  # 訓練日誌
logs/training/eval_<character>_<timestamp>.log   # 評估日誌
```

### TensorBoard Visualization

啟動 TensorBoard 查看訓練曲線：

```bash
# 單一角色
tensorboard --logdir /mnt/data/ai_data/models/lora/<movie>/<character>_identity/logs

# 所有角色比較
tensorboard --logdir /mnt/data/ai_data/models/lora/
```

## Troubleshooting

### Common Issues

**1. GPU 記憶體不足 (OOM)**

症狀: `CUDA out of memory` 錯誤

解決:
```bash
# 檢查 GPU 使用情況
nvidia-smi

# 清理殘留進程
pkill -f train_network
```

調整 batch_size (在 TOML 配置中):
```toml
train_batch_size = 4  # 降低從 6 到 4
```

**2. 訓練卡住或速度極慢**

症狀: Loss 不更新，進度條停滯

檢查:
```bash
# 查看進程狀態
ps aux | grep train_network

# 查看最新日誌
tail -50 logs/training/train_<character>_<timestamp>.log
```

**3. Checkpoint 評估失敗**

症狀: 評估階段錯誤

檢查:
```bash
# 確認 checkpoint 存在
ls -lh /mnt/data/ai_data/models/lora/<movie>/<character>_identity/*.safetensors

# 確認 prompts 文件
cat prompts/lora_testing/<character>_identity_prompts.json
```

### Recovery Procedures

**從失敗的角色繼續**:

編輯腳本中的 `CHARACTERS` 陣列，移除已完成的角色：

```bash
# 修改 scripts/batch/train_and_evaluate_6_characters.sh
# 將已完成的角色從陣列中刪除
declare -a CHARACTERS=(
    # "orion:..."  # 已完成，註解掉
    "ian:configs/training/character_loras/onward_ian_lightfoot_identity.toml:..."
    # ... 剩餘角色
)
```

**手動評估單一 Checkpoint**:

```bash
conda run -n ai_env python scripts/evaluation/test_lora_checkpoints.py \
    /path/to/lora/directory \
    --base-model /path/to/base_model.safetensors \
    --output-dir /path/to/output \
    --prompts-file prompts/lora_testing/<character>_prompts.json \
    --device cuda
```

## Best Practices

### Before Training

1. ✅ **驗證數據完整性**
   - 確認所有圖片和 caption 配對完整
   - 檢查訓練目錄權限

2. ✅ **GPU 預熱**
   - 先運行小測試確認環境正常
   - 確保足夠的磁碟空間 (每個角色 ~5-10GB)

3. ✅ **Backup 配置**
   - 保存當前 TOML 配置的副本
   - 記錄開始訓練的時間戳

### During Training

1. 📊 **定期檢查進度**
   - 每 2 小時查看一次日誌
   - 確認樣本圖片生成正常

2. 🔍 **監控 Loss 曲線**
   - TensorBoard 實時查看
   - Loss 應該穩定下降

3. 💾 **驗證 Checkpoint 保存**
   - 確認每 2 epoch 都有新的 .safetensors 文件

### After Training

1. 📋 **檢查評估報告**
   - 查看 `evaluation_report.json`
   - 比較不同 checkpoint 的視覺效果

2. 🎯 **選擇最佳模型**
   - 不一定是最後的 checkpoint
   - 考慮 overfitting 問題

3. 🧪 **實際測試**
   - 用自訂 prompts 測試最佳 checkpoint
   - 驗證角色一致性和風格保留

## Next Steps

訓練完成後的後續步驟：

1. **LoRA Composition Testing**
   ```bash
   python scripts/evaluation/test_lora_composition.py \
       --loras lora1.safetensors lora2.safetensors \
       --weights 0.8 0.5
   ```

2. **Expression/Pose LoRA Training**
   - 使用 identity LoRA 作為基礎
   - 訓練額外的表情和姿態 LoRA

3. **Scene LoRA Training**
   - 場景和背景風格 LoRA
   - 多角色組合場景

4. **Production Deployment**
   - 整合到推理 pipeline
   - 建立 LoRA 管理系統

## References

- **Training Configs**: `configs/training/character_loras/`
- **Evaluation Scripts**: `scripts/evaluation/`
- **Test Prompts**: `prompts/lora_testing/`
- **Pipeline Script**: `scripts/batch/train_and_evaluate_6_characters.sh`
- **Previous Successful Training**: Alberto (Luca) Trial 3.6

## Version History

- **v1.0** (2025-11-21): 初始版本，6 角色自動化訓練+評估 pipeline
  - Characters: Orion, Ian, Russell, Tyler, Barley, Giulia
  - Trial 3.6 proven parameters
  - Comprehensive checkpoint evaluation
