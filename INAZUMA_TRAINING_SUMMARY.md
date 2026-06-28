# Inazuma Eleven SDXL LoRA Training Summary

**Date**: 2025-12-19
**Status**: Ready to Train
**Pipeline**: 3D → 2D Anime Character LoRA Adaptation

---

## Overview

This document summarizes the complete SDXL LoRA training setup for Inazuma Eleven characters, adapted from the proven 3D character pipeline to 2D anime optimization.

---

## Characters & Datasets

| # | Character | Japanese | Dataset | Images | Trigger Word |
|---|-----------|----------|---------|--------|--------------|
| 1 | Endou Mamoru | 円堂守 | `1_endou_mamoru` | 200 | `inazuma_endou_mamoru` |
| 2 | Gouenji Shuuya | 豪炎寺修也 | `1_gouenji_shuuya` | 200 | `inazuma_gouenji_shuuya` |
| 3 | Fudou Akio | 不動明王 | `1_fudou_akio` | 200 | `inazuma_fudou_akio` |
| 4 | Matsukaze Tenma | 松風天馬 | `1_matsukaze_tenma` | 200 | `inazuma_matsukaze_tenma` |
| 5 | Inamori Asuto | 稲森明日人 | `1_inamori_asuto` | 200 | `inazuma_inamori_asuto` |
| 6 | Nosaka Yuuma | 野坂悠馬 | `1_nosaka_yuuma` | 200 | `inazuma_nosaka_yuuma` |
| 7 | Utsunomiya Toramaru | 宇都宮虎丸 | `1_utsunomiya_toramaru` | 200 | `inazuma_utsunomiya_toramaru` |

**Total**: 7 characters, 1400 images

---

## Key Configuration Parameters

### Adapted from 3D to 2D Anime

| Parameter | 3D Value | 2D Anime | Rationale |
|-----------|----------|----------|-----------|
| **Learning Rate** | 1e-4 | **8e-5** | Lower for sharp line art features |
| **Text Encoder LR** | 5e-5 | **4e-5** | Proportional reduction |
| **LR Scheduler** | cosine | **cosine_with_restarts** | Better for anime features (2 cycles) |
| **Epochs** | 10 | **12** | More epochs for line consistency |
| **Repeats** | 5-12 | **10** | 200 × 10 = 2000 steps/epoch |
| **Caption Dropout** | 0% | **10%** | Force visual learning |
| **Network Dim** | 32 | **32** | Identity focus maintained |
| **Batch Size** | 1 | **1** | Memory safety |
| **Gradient Accum** | 2 | **2** | Effective batch = 2 |

### Unchanged (Proven)
- ✅ **Optimizer**: AdamW8bit
- ✅ **Resolution**: 1024×1024 (SDXL native)
- ✅ **Mixed Precision**: FP16
- ✅ **Cache Latents**: True (RAM, not disk)
- ✅ **Min SNR Gamma**: 5.0
- ✅ **Noise Offset**: 0.05
- ✅ **Keep Tokens**: 2

---

## Training Strategy

### Unified LoRA with Timeline Conditioning

**Decision**: Train ONE LoRA per character that handles all timelines (original, go, ares, orion) via caption conditioning.

**Why NOT split by timeline?**
- ❌ Splitting → 4 LoRAs × 50 images each (insufficient data)
- ❌ 4× training time
- ❌ Complex inference (need to select correct LoRA)

**Unified LoRA Advantages**:
- ✅ Full 200 images per LoRA
- ✅ Timeline tags enable conditional generation
- ✅ Single LoRA handles all character variants
- ✅ 1/4 training time

**Example Caption Structure**:
```
inazuma_eleven, inazuma_endou_mamoru, anime_style, teenage_boy,
timeline_original, timeline_go, timeline_ares, timeline_orion,
role_goalkeeper, element_mountain, orange_headband,
spiky_brown_hair, big_round_brown_eyes, determined_expression,
soccer_uniform, gloves, daytime, training, confident_pose
```

**Usage at Inference**:
```python
# Original timeline
"inazuma_endou_mamoru, timeline_original, soccer_uniform, stadium"

# GO timeline (adult)
"inazuma_endou_mamoru, timeline_go, adult, coach_outfit, confident_smile"

# Ares timeline
"inazuma_endou_mamoru, timeline_ares, match_scene, diving_save"
```

---

## Training Schedule

### Per Character
- **Epochs**: 12
- **Steps/Epoch**: ~2,000 (200 images × 10 repeats ÷ 2 batch)
- **Total Steps**: ~24,000
- **Checkpoints**: Epoch 2, 4, 6, 8, 10, 12 (6 checkpoints)
- **Estimated Time**: 8-10 hours

### Batch Training (All 7 Characters)
- **Sequential**: One character at a time
- **Total Time**: 7 × 9h = **~63 hours (2.6 days)**
- **Start**: 2025-12-19
- **Estimated Completion**: 2025-12-22

---

## File Locations

### Configurations
```
configs/training/character_loras_sdxl/
├── inazuma_endou_mamoru_sdxl.toml
├── inazuma_gouenji_shuuya_sdxl.toml
├── inazuma_fudou_akio_sdxl.toml
├── inazuma_matsukaze_tenma_sdxl.toml
├── inazuma_inamori_asuto_sdxl.toml
├── inazuma_nosaka_yuuma_sdxl.toml
└── inazuma_utsunomiya_toramaru_sdxl.toml
```

### Datasets
```
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/
├── 1_endou_mamoru/          (200 images + captions)
├── 1_gouenji_shuuya/        (200 images + captions)
├── 1_fudou_akio/            (200 images + captions)
├── 1_matsukaze_tenma/       (200 images + captions)
├── 1_inamori_asuto/         (200 images + captions)
├── 1_nosaka_yuuma/          (200 images + captions)
└── 1_utsunomiya_toramaru/   (200 images + captions)
```

### Output (LoRA Models)
```
/mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/
├── endou_mamoru_identity/
│   ├── inazuma_endou_mamoru_lora_sdxl-000002.safetensors
│   ├── inazuma_endou_mamoru_lora_sdxl-000004.safetensors
│   ├── ...
│   ├── inazuma_endou_mamoru_lora_sdxl-000012.safetensors
│   └── logs/
├── gouenji_shuuya_identity/
│   └── ...
└── ...
```

### Logs
```
logs/
├── inazuma_batch_training_20251219_HHMMSS.log
├── inazuma_endou_mamoru_sdxl_training_20251219_HHMMSS.log
├── inazuma_gouenji_shuuya_sdxl_training_20251219_HHMMSS.log
└── ...
```

---

## Execution Commands

### 1. Generate Configurations (Completed)
```bash
python scripts/batch/generate_inazuma_sdxl_configs.py
```
✅ **Status**: Completed - 7 TOML files generated

### 2. Start Batch Training
```bash
# In tmux session (recommended)
tmux new-session -s inazuma_training
bash scripts/batch/train_inazuma_sdxl_loras.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t inazuma_training
```

### 3. Monitor Progress
```bash
# Watch batch log
tail -f logs/inazuma_batch_training_*.log

# Check GPU
watch -n 5 nvidia-smi

# Check individual character log
tail -f logs/inazuma_endou_mamoru_sdxl_training_*.log
```

### 4. Evaluate Checkpoints
```bash
# Evaluate all characters, final checkpoints (epoch 12)
python scripts/batch/evaluate_inazuma_loras.py \
  --checkpoint-epoch 12

# Evaluate specific character
python scripts/batch/evaluate_inazuma_loras.py \
  --character endou_mamoru

# Evaluate all checkpoints (2, 4, 6, 8, 10, 12)
python scripts/batch/evaluate_inazuma_loras.py
```

---

## Expected Outputs

### Per Character (Example: Endou Mamoru)

**Checkpoints** (6 files):
```
inazuma_endou_mamoru_lora_sdxl-000002.safetensors  (~90MB)
inazuma_endou_mamoru_lora_sdxl-000004.safetensors
inazuma_endou_mamoru_lora_sdxl-000006.safetensors
inazuma_endou_mamoru_lora_sdxl-000008.safetensors
inazuma_endou_mamoru_lora_sdxl-000010.safetensors
inazuma_endou_mamoru_lora_sdxl-000012.safetensors
```

**TensorBoard Logs**:
- Loss curves
- Learning rate schedule
- Training metrics

**Evaluation Images** (per checkpoint):
- Timeline-specific tests
- Style consistency tests
- Negative tests (prompt bleeding check)

---

## Quality Criteria

### Minimum Acceptable
- ✅ Character identity recognizable (95%+ confidence)
- ✅ Timeline conditioning works (prompt → appropriate age/outfit)
- ✅ No prompt bleeding (generic prompts don't trigger character)
- ✅ Clean anime line art (no 3D-style blurriness)
- ✅ Consistent across checkpoints epoch 8-12

### Optimization Targets
- 🎯 LoRA size: <100MB per character
- 🎯 Inference: <2s per image (SDXL baseline)
- 🎯 Compatible with standard SDXL pipelines
- 🎯 Works with LoRA scale 0.6-1.0

---

## Timeline-Specific Test Prompts

### Endou Mamoru
```
timeline_original: "inazuma_endou_mamoru, timeline_original, goalkeeper_uniform, orange_headband, stadium"
timeline_go: "inazuma_endou_mamoru, timeline_go, adult, coach_outfit, confident_smile"
timeline_ares: "inazuma_endou_mamoru, timeline_ares, match_scene, diving_save"
timeline_orion: "inazuma_endou_mamoru, timeline_orion, captain_armband, leadership_pose"
```

### Gouenji Shuuya
```
timeline_original: "inazuma_gouenji_shuuya, timeline_original, forward, white_headband, cool_expression"
timeline_go: "inazuma_gouenji_shuuya, timeline_go, masked_identity, ishido_shuuji, authority_figure"
timeline_ares: "inazuma_gouenji_shuuya, timeline_ares, kidokawa_seishuu_uniform, ace_striker"
```

---

## Risk Mitigation

### Potential Issues & Solutions

**1. Timeline Confusion**
- **Risk**: Model mixes timeline features
- **Mitigation**: Strong timeline tags + epoch 2 testing
- **Fallback**: Consider splitting original vs go if confusion persists

**2. Overfitting**
- **Risk**: 200 images may cause memorization
- **Mitigation**: Caption dropout 10%, monitor loss curves
- **Action**: Stop at epoch 10 if quality plateaus

**3. Identity Drift**
- **Risk**: Character becomes generic anime boy
- **Mitigation**: keep_tokens=2, min_snr_gamma=5.0
- **Test**: Generate with minimal prompt (just character name)

---

## Next Steps (Post-Training)

1. **Checkpoint Selection**
   - Compare epochs 8, 10, 12 for each character
   - Select best checkpoint based on:
     * Identity consistency
     * Timeline conditioning accuracy
     * Prompt responsiveness
     * Style quality

2. **Comparative Evaluation**
   - Test all 7 characters side-by-side
   - Verify no cross-character contamination
   - Document best practices per character

3. **Integration Testing**
   - Test with ComfyUI / A1111
   - Multi-LoRA composition (if applicable)
   - Different base models (if needed)

4. **Documentation**
   - Create usage guide with example prompts
   - Document optimal LoRA scales per character
   - Share findings for future 2D anime LoRA training

---

## References

- **3D Config Source**: `docs/SDXL_TRAINING_CONFIG_FINAL.md`
- **Training Strategy**: `docs/training/INAZUMA_SDXL_TRAINING_STRATEGY.md`
- **Character Profiles**: `docs/projects/inazuma-eleven/characters/*.md`
- **Series Overview**: `docs/projects/inazuma-eleven/inazuma_eleven_series.md`

---

## Status Tracker

- [x] Read 3D SDXL training config
- [x] Analyze dataset structure
- [x] Adapt config for 2D anime
- [x] Generate 7 TOML configs
- [x] Create batch training script
- [x] Create evaluation script
- [ ] **Execute training** ← NEXT STEP
- [ ] Evaluate checkpoints
- [ ] Select best checkpoints
- [ ] Document results

---

**Ready to Launch**: All configurations and scripts are in place. Execute training with:

```bash
tmux new-session -s inazuma_training
bash scripts/batch/train_inazuma_sdxl_loras.sh
```

**Estimated Completion**: 2025-12-22 (~2.6 days from start)
