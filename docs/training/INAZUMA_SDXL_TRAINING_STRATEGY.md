# Inazuma Eleven SDXL LoRA Training Strategy

**Date**: 2025-12-19
**Version**: v1.0 - 2D Anime Character Adaptation

---

## Executive Summary

**Objective**: Train high-quality SDXL character identity LoRAs for Inazuma Eleven characters, adapted from proven 3D character training configuration.

**Key Differences from 3D Training**:
- **2D anime** (cel-shading, line art) vs 3D rendered (smooth PBR materials)
- **Higher learning rate** for sharper line features
- **Stronger regularization** to prevent overfitting on cel-shaded consistency
- **Timeline-aware captions** to enable conditional generation

---

## Dataset Analysis

### Current State
```
Base Path: /mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/

Characters (7 total):
├── 1_endou_mamoru      (200 images)
├── 1_fudou_akio        (200 images)
├── 1_gouenji_shuuya    (200 images)
├── 1_inamori_asuto     (200 images)
├── 1_matsukaze_tenma   (200 images)
├── 1_nosaka_yuuma      (200 images)
└── 1_utsunomiya_toramaru (200 images)
```

### Caption Structure (Canonical)
All images share a single canonical caption per character:
```
inazuma_eleven, inazuma_endou_mamoru, anime_style, teenage_boy,
timeline_original, timeline_go, timeline_ares, timeline_orion,
role_goalkeeper, element_mountain, orange_headband,
spiky_brown_hair, big_round_brown_eyes, determined_expression,
soccer_uniform, gloves, daytime, training, confident_pose,
white_and_blue_jersey, muscular, energetic, focused,
goalkeeper_gloves, sports_environment
```

**Key Features**:
- ✅ Multi-timeline tags (original, go, ares, orion)
- ✅ Character identity markers
- ✅ Visual attributes (hair, eyes, outfit)
- ✅ Contextual tags (sport, environment)

### Training Strategy Decision

**Option A**: Split by timeline → 4 LoRAs per character × 50 images each
- ❌ Insufficient data per LoRA (50 images)
- ❌ 4× training time
- ❌ Complex inference (need to select correct LoRA)

**Option B**: **Unified LoRA with timeline conditioning** (SELECTED)
- ✅ Full 200 images per LoRA
- ✅ Timeline tags enable conditional generation
- ✅ Single LoRA handles all character variants
- ✅ 1/4 training time

**Rationale**:
2D anime characters maintain high visual consistency across timelines (similar proportions, line style, cel-shading). Main differences (age, outfit) can be controlled via prompts. Timeline tags in captions allow model to learn conditional representations.

---

## Configuration Adaptation: 3D → 2D Anime

### Core Parameter Adjustments

| Parameter | 3D Value | 2D Anime Value | Rationale |
|-----------|----------|----------------|-----------|
| **Learning Rate** | 1e-4 | **8e-5** | 2D line art needs slightly lower LR to avoid overshooting sharp features |
| **Text Encoder LR** | 5e-5 | **4e-5** | Proportional reduction |
| **Network Dim** | 32 | **32** | Identity focus, keep moderate capacity |
| **Network Alpha** | 16 | **16** | Maintain stable training |
| **Optimizer** | AdamW8bit | **AdamW8bit** | Memory efficiency preserved |
| **LR Scheduler** | cosine | **cosine_with_restarts** | Better for anime feature learning |
| **Num Restarts** | - | **2** | Helps escape local minima in line art space |
| **Resolution** | 1024 | **1024** | SDXL native resolution |
| **Batch Size** | 1 | **1** | Memory safety |
| **Gradient Accum** | 2 | **2** | Effective batch = 2 |
| **Max Epochs** | 10 | **12** | Anime needs slightly more epochs for line consistency |
| **Repeats** | 5-12 (adaptive) | **10** | 200 images × 10 = 2000 steps/epoch |

### Regularization Strategy

**3D Configuration**:
- Light regularization (PBR materials are self-consistent)
- No color jitter (breaks PBR)
- No flips (breaks asymmetry)

**2D Anime Adaptation**:
- **Caption dropout: 10%** (force visual learning, not just text)
- **Flip augmentation: DISABLED** (Inazuma characters have asymmetric designs)
- **Color jitter: DISABLED** (cel-shading has specific color palettes)
- **Keep token: 2** (always keep character name + series)

### Checkpoint Strategy

```toml
save_every_n_epochs = 2
save_last_n_epochs_state = 3
save_model_as = "safetensors"
```

**Checkpoints**: Epoch 2, 4, 6, 8, 10, 12
**Evaluation**: Test with timeline-specific prompts at each checkpoint

---

## Training Configuration Template

### File Naming Convention
```
configs/training/character_loras_sdxl/inazuma_{character_id}_sdxl.toml
```

Example: `inazuma_endou_mamoru_sdxl.toml`

### TOML Structure (Annotated)

```toml
[general]
output_name = "inazuma_endou_mamoru_lora_sdxl"
output_dir = "/mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/endou_mamoru_identity"
pretrained_model_name_or_path = "/mnt/data/ai_data/models/sdxl/stable-diffusion-xl-base-1.0"
vae = "/mnt/data/ai_data/models/sdxl/sdxl_vae.safetensors"

[dataset]
train_data_dir = "/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/1_endou_mamoru"
resolution = "1024,1024"
enable_bucket = true
bucket_reso_steps = 64
bucket_no_upscale = true
min_bucket_reso = 768
max_bucket_reso = 1280

[training]
# 2D Anime optimized learning rates
learning_rate = 8e-5
unet_lr = 8e-5
text_encoder_lr = 4e-5
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 2
lr_warmup_steps = 100

# Network architecture
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

# Training dynamics
optimizer_type = "AdamW8bit"
train_batch_size = 1
gradient_accumulation_steps = 2
max_train_epochs = 12
dataset_repeats = 10  # 200 images × 10 = 2000 steps/epoch

# Regularization (2D anime specific)
caption_dropout_rate = 0.1
caption_dropout_every_n_epochs = 1
keep_tokens = 2  # Keep "inazuma_eleven, inazuma_endou_mamoru"

# Data loading
persistent_data_loader_workers = true
max_data_loader_n_workers = 4
vae_batch_size = 2

# Memory optimization
mixed_precision = "fp16"
cache_latents = true
cache_latents_to_disk = false  # RAM cache for speed (64GB available)
gradient_checkpointing = true
xformers_memory_efficient_attention = true

# Checkpointing
save_every_n_epochs = 2
save_last_n_epochs_state = 3
save_model_as = "safetensors"
save_precision = "fp16"

# Logging
logging_dir = "/mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/endou_mamoru_identity/logs"
log_with = "tensorboard"
log_prefix = "inazuma_endou_mamoru"

# Reproducibility
seed = 42

# Quality settings
clip_skip = 2
min_snr_gamma = 5.0
noise_offset = 0.05
```

---

## Evaluation Protocol

### Test Prompts (Per Character)

**Timeline-Specific Tests**:
```python
test_prompts = {
    "original_neutral": "inazuma_endou_mamoru, timeline_original, soccer uniform, standing, neutral expression, white background",
    "original_action": "inazuma_endou_mamoru, timeline_original, goalkeeper dive, intense focus, stadium background",
    "go_adult": "inazuma_endou_mamoru, timeline_go, adult, coach outfit, confident smile, training ground",
    "ares_match": "inazuma_endou_mamoru, timeline_ares, match scene, orange headband visible, determined face",
    "orion_captain": "inazuma_endou_mamoru, timeline_orion, captain armband, team huddle, leadership pose",
    "style_test": "inazuma_endou_mamoru, anime style, close-up portrait, detailed eyes, professional anime art"
}
```

### Quality Metrics
1. **Identity Consistency**: Character recognizable across timelines
2. **Timeline Accuracy**: Visual differences between original/go/ares/orion respected
3. **Style Fidelity**: Clean anime line art, proper cel-shading
4. **Prompt Responsiveness**: Non-character prompts don't bleed character features

---

## Batch Training Plan

### Sequential Training Order
```
1. endou_mamoru     (protagonist, test pipeline)
2. gouenji_shuuya   (main deuteragonist)
3. fudou_akio       (complex design test)
4. matsukaze_tenma  (GO protagonist)
5. inamori_asuto    (Ares protagonist)
6. nosaka_yuuma     (Orion key character)
7. utsunomiya_toramaru (Original series character)
```

### Time Estimates (Per Character)
- **Epochs**: 12
- **Steps/epoch**: ~2000 (200 images × 10 repeats)
- **Total steps**: ~24,000
- **Time/epoch**: ~40-50 minutes (estimated from 3D config)
- **Total time**: ~8-10 hours per character

**Total Pipeline**: 7 characters × 9 hours = **~63 hours (2.6 days)**

---

## Success Criteria

### Minimum Acceptable Quality
- ✅ Character identity recognizable at 95%+ confidence
- ✅ Timeline conditioning works (different prompts → appropriate age/outfit)
- ✅ No prompt bleeding (non-character prompts don't trigger character)
- ✅ Clean anime line art (no blurry 3D-style rendering)
- ✅ Consistent across checkpoints epoch 8-12

### Optimization Targets
- 🎯 LoRA weight size: <100MB (network_dim=32 ensures this)
- 🎯 Inference speed: <2s per image (SDXL baseline)
- 🎯 Compatibility: Works with standard SDXL pipelines

---

## Risk Mitigation

### Potential Issues

**1. Timeline Confusion**
- **Risk**: Model mixes timeline features (adult face on child body)
- **Mitigation**:
  - Strong timeline tags in all captions
  - Test with timeline-specific prompts at epoch 2
  - If confusion occurs, consider splitting original vs go timelines

**2. Overfitting**
- **Risk**: 200 images may cause memorization
- **Mitigation**:
  - Caption dropout 10%
  - Monitor validation loss (if available)
  - Stop at epoch 10 if quality plateaus

**3. Identity Drift**
- **Risk**: Character becomes generic anime boy
- **Mitigation**:
  - Keep character name tokens (keep_tokens=2)
  - High min_snr_gamma (5.0) for identity stability
  - Test with minimal prompts ("just character name")

### Rollback Plan
- Original 3D config available as fallback
- Checkpoints every 2 epochs allow A/B testing
- Can adjust LR/epochs mid-training if needed

---

## Next Steps

1. ✅ Generate 7 TOML configs (one per character)
2. ✅ Create batch training script with tmux support
3. ✅ Prepare evaluation script with timeline-specific prompts
4. 🚀 Launch training (start with endou_mamoru as validation)
5. 📊 Monitor epoch 2 results, adjust if needed
6. 🎯 Complete batch training
7. 📈 Comparative evaluation of all 7 LoRAs

---

**Configuration Status**: Ready for generation
**Training Environment**: Verified (64GB RAM, 16GB VRAM, CUDA available)
**Estimated Completion**: 2025-12-22 (assuming immediate start)
