# Alberto LoRA Anti-Blur Training Guide

## Problem Diagnosis

**Root Cause:** Training dataset contains images with significant motion blur and color bleeding, causing the model to learn incorrect texture patterns that result in watercolor-like artifacts.

**Evidence:**
- Original images (489x976) show heavy motion blur and chromatic aberration
- When upscaled to 1024x1024, these artifacts are amplified
- Orion training succeeded with similar captions but cleaner source images

## Solution: Optimized Training Parameters

### Parameter Adjustments (vs. Original)

| Parameter | Original | Optimized | Reason |
|-----------|----------|-----------|---------|
| `network_dim` | 64 | **32** | Lower capacity reduces overfitting to blur |
| `network_alpha` | 32 | **16** | Proportional reduction |
| `learning_rate` | 0.0001 | **0.00004** | Slower learning prevents memorizing blur |
| `text_encoder_lr` | 0.00006 | **0.00002** | Conservative text encoding |
| `unet_lr` | 0.0001 | **0.00004** | Reduced U-Net learning rate |
| `min_snr_gamma` | 5.0 | **8.0** | Stronger noise filtering for blurry data |
| `noise_offset` | 0.05 | **0.15** | More noise diversity combats artifacts |
| `max_train_epochs` | 4 | **6** | Longer but slower training |
| `save_every_n_epochs` | 2 | **1** | Save all epochs for early stopping |
| `sample_every_n_epochs` | 2 | **1** | Monitor quality every epoch |

### Key Strategies

1. **Lower Learning Rate (0.00004)**
   - Prevents aggressive learning of blur patterns
   - Allows gradual refinement of features
   - Reduces risk of catastrophic overfitting

2. **Higher SNR Gamma (8.0)**
   - Filters out low signal-to-noise ratio samples
   - More robust to blurry/noisy training data
   - Focuses learning on cleaner parts of images

3. **Increased Noise Offset (0.15)**
   - Adds more variation to training noise
   - Helps model generalize beyond blurry inputs
   - Improves robustness to artifacts

4. **Reduced Network Capacity (dim 32)**
   - Smaller network can't memorize all blur patterns
   - Forces learning of essential character features
   - Natural regularization effect

## Training Commands

### 1. Start Training

```bash
# Make sure old checkpoints are backed up or removed
mkdir -p /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/backup_old
mv /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/*.safetensors \
   /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/backup_old/ 2>/dev/null || true

# Start training
conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=2 \
  sd-scripts/sdxl_train_network.py \
  --config_file=/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/luca_alberto_identity_sdxl.toml
```

### 2. Monitor Training (Separate Terminal)

```bash
# Watch training progress
watch -n 10 'ls -lht /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/*.safetensors | head -5'

# Check sample images
ls -lt /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/sample/*.png | head -10
```

### 3. Monitor GPU Usage

```bash
watch -n 2 nvidia-smi
```

## Evaluation Strategy

### Critical Checkpoints to Test

Test **EVERY** epoch checkpoint (1-6) because:
- Early epochs (1-2) may be cleaner before learning blur
- Middle epochs (3-4) may hit sweet spot
- Later epochs (5-6) may overfit to blur patterns

### Testing Command

```bash
conda run -n ai_env python scripts/evaluation/sdxl_lora_evaluator.py \
  --lora-dir /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity \
  --base-model /home/b0979/models/sdxl/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/ai_data/lora_evaluation/alberto_anti_blur \
  --prompts /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/prompts/lora_testing/alberto_identity_test.txt \
  --device cuda \
  --batch-size 4
```

### Quality Checklist

For each checkpoint, check:
- ✅ **No watercolor effect**: Edges should be clean, not blurred/smeared
- ✅ **No color bleeding**: Colors stay within boundaries
- ✅ **Clear features**: Hair, eyes, clothing should be distinct
- ✅ **Proper style**: Pixar 3D look, not painterly
- ✅ **Character accuracy**: Recognizable as Alberto

### Expected Results

| Epoch | Expected Quality | Notes |
|-------|------------------|-------|
| 1 | Basic features | May lack detail but should be clean |
| 2 | Improved accuracy | Sweet spot candidate #1 |
| 3 | Peak quality | Sweet spot candidate #2 |
| 4 | Slight overfit risk | Monitor for blur creeping in |
| 5 | Overfit warning | May show watercolor signs |
| 6 | Likely overfit | Compare carefully with epoch 3 |

**Recommendation:** Expect best results around **epoch 2-3**.

## Early Stopping

Stop training manually if you observe:
1. Sample images start showing blur/watercolor effect
2. Quality degrades after a good epoch
3. Color bleeding appears in generated images

To stop training:
```bash
# Find training process
ps aux | grep sdxl_train_network

# Kill training
pkill -f sdxl_train_network
```

## Troubleshooting

### If still too blurry after epoch 3:
- Further reduce learning rate to 0.00002
- Increase min_snr_gamma to 10.0
- Consider filtering dataset (remove blurriest images)

### If training is too slow:
- Current settings prioritize quality over speed
- Do NOT increase learning rate
- Be patient - better slow than blurry

### If not learning character features:
- May need to increase to epoch 4-5
- Check if learning rate is too conservative
- Verify dataset has enough variety

## Post-Training

### 1. Compare All Checkpoints

```bash
# Generate comparison grid
ls /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/*-0000*.safetensors
```

### 2. Select Best Checkpoint

Based on visual quality, not loss metrics!

### 3. Rename Best Checkpoint

```bash
cp /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/alberto_identity_lora_sdxl-000003.safetensors \
   /mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/alberto_identity_lora_sdxl_BEST.safetensors
```

## Backup

Original config backed up to:
```
configs/training/character_loras_sdxl/luca_alberto_identity_sdxl.toml.backup
```

To restore:
```bash
cp configs/training/character_loras_sdxl/luca_alberto_identity_sdxl.toml.backup \
   configs/training/character_loras_sdxl/luca_alberto_identity_sdxl.toml
```

## Summary

✅ **Config updated** with anti-blur optimizations
✅ **Training strategy** adjusted for blurry source data
✅ **Monitoring plan** to catch optimal checkpoint
✅ **Evaluation criteria** focused on avoiding watercolor artifacts

**Next Steps:**
1. Start training with optimized config
2. Monitor samples every epoch
3. Test all checkpoints (1-6)
4. Select cleanest result (likely epoch 2-3)
5. Report results for further refinement if needed
