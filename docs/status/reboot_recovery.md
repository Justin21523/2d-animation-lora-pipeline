# Reboot Recovery Plan - 2025-11-28

## Current Status Before Reboot

### Completed
✅ **Alberto Sea Monster LoRA Training** - Finished (Epoch 8)
   - Location: `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_seamonster_identity/`
   - Checkpoints: epoch 4, 6, 8 (final)

✅ **System Memory Protection** - Installed & Enabled for auto-start
   - Service: `~/.config/systemd/user/system-protection.service`
   - Status: Enabled (will auto-start on boot)
   - Verify after reboot: `systemctl --user status system-protection`

### In Progress (Will be interrupted)
⚠️ **Luca Sea Monster LoRA Training** - Started but no checkpoints yet
   - Training was running when reboot initiated
   - No checkpoints saved yet (saves every 2 epochs)
   - Will need to restart from beginning

### Failed
❌ **Alberto Sea Monster Testing** - Failed due to CUDA driver mismatch
   - Will retry after reboot with fixed drivers

---

## After Reboot - Recovery Steps

### 1. Verify NVIDIA Driver Fixed ✓
```bash
# Check driver version is consistent
nvidia-smi
cat /proc/driver/nvidia/version

# Should both show 580.95 or both show 575.64 (consistent)
```

### 2. Verify Memory Protection Auto-Started ✓
```bash
# Check systemd service
systemctl --user status system-protection

# Check watchdog processes
pgrep -af watchdog

# Manual start if needed
system-protection-start
```

### 3. Resume Luca Sea Monster Training
```bash
cd /mnt/c/ai_projects/kohya_ss/sd-scripts

conda run -n kohya_ss accelerate launch \
  --num_cpu_threads_per_process=2 \
  sdxl_train_network.py \
  --config_file /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_luca_seamonster_sdxl.toml
```

### 4. Test Alberto Sea Monster LoRA
```bash
cd /mnt/c/ai_projects/3d-animation-lora-pipeline

conda run -n ai_env python scripts/evaluation/sdxl_multi_lora_compositor.py \
    --loras /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_seamonster_identity/alberto_seamonster_lora_sdxl.safetensors \
    --lora-names alberto_seamonster \
    --weight-combos "0.8" "1.0" \
    --prompts-file prompts/single_character/alberto_seamonster_prompts.txt \
    --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
    --output-dir /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_seamonster_identity/test_results \
    --num-samples 2 \
    --steps 30 \
    --guidance-scale 7.5 \
    --seed-start 100 \
    --width 1024 \
    --height 1024
```

---

## Quick Reference Commands

### Check Training Status
```bash
# List training processes
ps aux | grep train_network

# Check GPU usage (after driver fix)
nvidia-smi

# Monitor system resources
system-protection-monitor
```

### Training Data Locations
- **Luca Sea Monster**: `/mnt/data/datasets/general/luca/sdxl_seamonster_training/luca_seamonster_prepared/` (255 images)
- **Alberto Sea Monster**: Training complete, checkpoints ready

### Output Locations
- **Luca LoRA**: `/mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_seamonster_identity/`
- **Alberto LoRA**: `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_seamonster_identity/`
- **Test Results**: Will be in respective `test_results/` subdirectories

---

## Important Notes

1. **Memory Protection** is now permanent - will auto-start on every boot via systemd
2. **Luca training** needs to restart from scratch (no checkpoints were saved)
3. **Alberto testing** can be done immediately after reboot
4. **Training config** is at: `configs/training/character_loras_sdxl/luca_luca_seamonster_sdxl.toml`

---

Generated: 2025-11-28 21:05
