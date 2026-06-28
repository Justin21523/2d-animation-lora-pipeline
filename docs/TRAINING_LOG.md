# LoRA Training Progress Log

## Batch 2 Retraining (2025-11-30)

### Issue Identified
Batch 2 characters were under-trained due to overly conservative parameters:
- Learning rate was too low (0.00005-0.00007 vs optimal 0.0001)
- Unnecessary dropout was added (0.03-0.05)
- Results: Under-trained models, especially Russell

### Solution Applied
Reverted all 4 Batch 2 characters to Batch 1 successful parameters:
- **Learning Rate**: 0.0001 (UNet & base)
- **Text Encoder LR**: 6e-05
- **Network Dropout**: 0.0 (removed)
- **Epochs**: 2
- **Other settings**: Same as Batch 1 (min_snr_gamma=5.0, noise_offset=0.05)

### Characters Being Retrained

#### 1. Barley Lightfoot (Onward) ✅ COMPLETED
- **Status**: Training completed at 08:44
- **Duration**: 31 minutes 14 seconds
- **Final Loss**: 0.104
- **Steps**: 1016 steps/epoch × 2 epochs = 2032 total steps
- **Dataset**: 254 images × 4 repeats = 1016 steps/epoch
- **Checkpoints**:
  - `/mnt/c/ai_models/diffusion/lora/sdxl/onward/barley_lightfoot_identity/barley_lightfoot_lora_sdxl-000001.safetensors` (Epoch 1)
  - `/mnt/c/ai_models/diffusion/lora/sdxl/onward/barley_lightfoot_identity/barley_lightfoot_lora_sdxl.safetensors` (Epoch 2, final)

#### 2. Alberto - Human Form (Luca) 🔄 IN PROGRESS
- **Status**: Training started after Barley
- **Dataset**: 509 images × 2 repeats = 1018 steps/epoch
- **Target**: 2 epochs (2036 total steps)
- **Expected Duration**: ~30 minutes
- **Output**: `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/`

#### 3. Giulia (Luca) ⏳ PENDING
- **Dataset**: 273 images × 4 repeats = 1092 steps/epoch
- **Target**: 2 epochs (2184 total steps)
- **Output**: `/mnt/c/ai_models/diffusion/lora/sdxl/luca/giulia_identity/`

#### 4. Russell (Up) ⏳ PENDING
- **Dataset**: 227 images × 4 repeats = 908 steps/epoch
- **Target**: 2 epochs (1816 total steps)
- **Output**: `/mnt/c/ai_models/diffusion/lora/sdxl/up/russell_identity/`

### Backup Information
All previous checkpoints backed up to:
- `/mnt/c/ai_models/diffusion/lora/sdxl/backup_batch2_undertrained_[timestamp]/`

### Training Script Location
- `/tmp/retrain_batch2_correct_params.sh`

### Monitoring
```bash
# Check training progress
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/batch_retrain_*.log

# Check background process status
ps aux | grep sdxl_train_network
```

### Training Parameters (Corrected)

```toml
[training]
optimizer_type = "AdamW8bit"
mixed_precision = "bf16"
full_bf16 = true
gradient_checkpointing = true

train_batch_size = 1
gradient_accumulation_steps = 2

# REVERTED to Batch 1 Learning Rates
learning_rate = 0.0001
text_encoder_lr = 6e-05
unet_lr = 0.0001

lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 1
lr_warmup_steps = 50-100 (varies by dataset size)

min_snr_gamma = 5.0
noise_offset = 0.05

# NO DROPOUT
network_dropout = 0.0

# EARLY STOPPING at 2 epochs
max_train_epochs = 2
save_every_n_epochs = 1
```

### Expected Completion
- **Total estimated time**: ~2 hours for all 4 characters
- **Started**: 2025-11-30 08:13
- **Expected completion**: 2025-11-30 10:15

### Next Steps After Training
1. Generate test images for all 4 retrained characters
2. Compare with previous under-trained versions
3. Verify improved quality before proceeding to synthetic data generation

---

## Training History Summary

### Batch 1 (Successful - Parameters Used as Reference)
- **Characters**: Orion, Bryce, Caleb
- **Learning Rate**: 0.0001
- **Dropout**: 0.0
- **Epochs**: 2
- **Result**: High quality, properly trained LoRAs

### Batch 2 (First Attempt - Failed)
- **Characters**: Barley, Alberto, Giulia, Russell
- **Learning Rate**: 0.00005-0.00007 (TOO LOW)
- **Dropout**: 0.03-0.05 (UNNECESSARY)
- **Epochs**: 2
- **Result**: Under-trained, especially Russell

### Batch 2 Retrain (Current)
- **Characters**: Same 4 characters
- **Learning Rate**: 0.0001 (CORRECTED)
- **Dropout**: 0.0 (REMOVED)
- **Epochs**: 2
- **Status**: IN PROGRESS
- **Expected Result**: Properly trained LoRAs matching Batch 1 quality

---

## Lessons Learned

1. **Learning Rate Sensitivity**: Halving the learning rate to 0.00005 significantly degrades training quality
2. **Dropout Not Needed**: For 2-epoch training with high-quality datasets, dropout causes instability
3. **Consistency is Key**: Using proven parameters (Batch 1) ensures reproducible results
4. **Early Detection**: Test images should be generated after each batch to catch under-training early

---

## Configuration Files Modified

1. `/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/onward_barley_lightfoot_sdxl.toml`
2. `/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_alberto_human_sdxl.toml`
3. `/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_giulia_sdxl.toml`
4. `/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/up_russell_sdxl.toml`

All configs updated with:
- `learning_rate = 0.0001`
- `text_encoder_lr = 6e-05`
- `unet_lr = 0.0001`
- `network_dropout = 0.0`

---

*Last Updated: 2025-11-30 09:00*
