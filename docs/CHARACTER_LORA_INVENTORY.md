# Character LoRA Inventory

**Total Character LoRAs**: 15
**Base Model**: SDXL (Stable Diffusion XL)
**Training Method**: Kohya_ss sd-scripts
**Network Type**: LoRA (Low-Rank Adaptation)

---

## Complete Character List

### 1. Elio (Movie: Elio) - 4 Characters

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Elio | `elio` | `/mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/` | ✅ Trained |
| Glordon | `glordon` | `/mnt/c/ai_models/diffusion/lora/sdxl/elio/glordon_identity/` | ✅ Trained |
| Bryce | `bryce` | `/mnt/c/ai_models/diffusion/lora/sdxl/elio/bryce_identity/` | ✅ Trained (Batch 1) |
| Caleb | `caleb` | `/mnt/c/ai_models/diffusion/lora/sdxl/elio/caleb_identity/` | ✅ Trained (Batch 1) |

### 2. Luca (Movie: Luca) - 5 Characters

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Luca (Human) | `luca` | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_identity/` | ✅ Trained |
| Luca (Sea Monster) | `luca_seamonster` | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_seamonster_identity/` | ✅ Trained |
| Alberto (Human) | `alberto` | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/` | 🔄 Retraining (Batch 2) |
| Alberto (Sea Monster) | `alberto_seamonster` | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_seamonster_identity/` | ✅ Trained |
| Giulia | `giulia` | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/giulia_identity/` | 🔄 Retraining (Batch 2) |

### 3. Coco (Movie: Coco) - 1 Character

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Miguel | `miguel` | `/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/` | ✅ Trained |

### 4. Up (Movie: Up) - 1 Character

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Russell | `russell` | `/mnt/c/ai_models/diffusion/lora/sdxl/up/russell_identity/` | 🔄 Retraining (Batch 2) |

### 5. Turning Red (Movie: Turning Red) - 1 Character

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Tyler | `tyler` | `/mnt/c/ai_models/diffusion/lora/sdxl/turning-red/tyler_identity/` | ✅ Trained |

### 6. Onward (Movie: Onward) - 2 Characters

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Ian Lightfoot | `ian_lightfoot` | `/mnt/c/ai_models/diffusion/lora/sdxl/onward/ian_lightfoot_identity/` | ✅ Trained |
| Barley Lightfoot | `barley_lightfoot` | `/mnt/c/ai_models/diffusion/lora/sdxl/onward/barley_lightfoot_identity/` | 🔄 Retraining (Batch 2) |

### 7. Orion and the Dark (Movie: Orion) - 1 Character

| Character | Token | Directory | Status |
|-----------|-------|-----------|--------|
| Orion | `orion` | `/mnt/c/ai_models/diffusion/lora/sdxl/orion/orion_identity/` | ✅ Trained (Batch 1) |

---

## Status Summary

- ✅ **Fully Trained & Ready**: 11 characters
- 🔄 **Currently Retraining (Batch 2)**: 4 characters
  - Barley Lightfoot (Completed ✓)
  - Alberto (Human) (In Progress)
  - Giulia (Pending)
  - Russell (Pending)

---

## Training Configuration Reference

### Batch 1 (Successful Parameters)
- **Characters**: Orion, Bryce, Caleb
- **Learning Rate**: 0.0001
- **Text Encoder LR**: 6e-05
- **Network Dropout**: 0.0
- **Epochs**: 2
- **Optimizer**: AdamW8bit
- **Result**: ✅ High quality

### Batch 2 Retrain (Corrected Parameters)
- **Characters**: Barley, Alberto, Giulia, Russell
- **Learning Rate**: 0.0001 (corrected from 0.00005-0.00007)
- **Text Encoder LR**: 6e-05
- **Network Dropout**: 0.0 (removed)
- **Epochs**: 2
- **Status**: 🔄 In Progress

---

## Checkpoint File Naming Convention

Each character has 2-3 checkpoint files:

1. **Epoch 1**: `{character_name}_lora_sdxl-000001.safetensors`
2. **Epoch 2 (Final)**: `{character_name}_lora_sdxl.safetensors` or `-000002.safetensors`

Example for Bryce:
```
/mnt/c/ai_models/diffusion/lora/sdxl/elio/bryce_identity/
├── bryce_lora_sdxl-000001.safetensors  # Epoch 1
└── bryce_lora_sdxl.safetensors         # Epoch 2 (final)
```

---

## Usage in Prompts

### Basic Format
```
{character_token}, {description}, pixar style 3d animation, high quality
```

### Examples
```
# Bryce
bryce, happy expression, front view, medium shot, pixar style 3d animation, high quality

# Luca (Sea Monster)
luca_seamonster, a blue and purple sea monster boy with green eyes and scales,
full body, pixar style 3d animation, high quality, detailed

# Miguel
miguel, playing guitar, three-quarter view, full body, pixar style 3d animation
```

---

## Character Trigger Tokens

| Character | Trigger Token | Notes |
|-----------|---------------|-------|
| Elio | `elio` | Main protagonist |
| Glordon | `glordon` | Alien character |
| Bryce | `bryce` | Teen character, blue hoodie |
| Caleb | `caleb` | Teen character |
| Luca (Human) | `luca` | Italian boy, human form |
| Luca (Sea Monster) | `luca_seamonster` | Blue/purple sea monster |
| Alberto (Human) | `alberto` | Curly hair, tan skin |
| Alberto (Sea Monster) | `alberto_seamonster` | Green/purple sea monster |
| Giulia | `giulia` | Red hair, Italian girl |
| Miguel | `miguel` | Mexican boy, guitarist |
| Russell | `russell` | Young wilderness explorer |
| Tyler | `tyler` | Teenage boy |
| Ian Lightfoot | `ian_lightfoot` | Elf, blue shirt |
| Barley Lightfoot | `barley_lightfoot` | Elf, larger build, beard |
| Orion | `orion` | Young boy, anxious character |

---

## Dataset Sizes (Approximate)

| Character | Images | Repeats | Steps/Epoch | Notes |
|-----------|--------|---------|-------------|-------|
| Barley Lightfoot | 254 | 4 | 1016 | Retrained |
| Alberto (Human) | 509 | 2 | 1018 | Retraining |
| Giulia | 273 | 4 | 1092 | Retraining |
| Russell | 227 | 4 | 908 | Retraining |
| Bryce | ~500 | varies | ~1000 | Batch 1 |
| Caleb | ~500 | varies | ~1000 | Batch 1 |
| Orion | ~400 | varies | ~800 | Batch 1 |

---

## Quality Recommendations

### Best Performing LoRAs (Batch 1)
1. **Orion** - Excellent consistency, good expression range
2. **Bryce** - High quality, diverse poses
3. **Caleb** - Solid performance across prompts

### Retraining in Progress (Batch 2)
- Barley Lightfoot ✓ (Completed with correct parameters)
- Alberto, Giulia, Russell (In progress)

### Expected Quality After Retrain
All Batch 2 characters should match Batch 1 quality after retraining with corrected parameters.

---

## Storage Locations

### LoRA Models
```
/mnt/c/ai_models/diffusion/lora/sdxl/{movie}/{character}_identity/
```

### Training Data (SDXL)
```
/mnt/data/datasets/general/{movie}/lora_data/training_data_sdxl/{character}_identity/
```

### Training Configs
```
/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/
```

---

## Character Demographics

### Age Groups
- **Young Children**: Luca, Alberto, Miguel, Russell, Orion
- **Teenagers**: Bryce, Caleb, Elio, Tyler, Giulia
- **Young Adults**: Ian Lightfoot, Barley Lightfoot

### Species/Types
- **Humans**: 11 characters
- **Elves**: 2 (Ian, Barley)
- **Sea Monsters**: 2 (Luca, Alberto - monster forms)
- **Alien**: 1 (Glordon)

### Gender Distribution
- **Male**: 12 characters
- **Female**: 1 (Giulia)
- **Other/Alien**: 2

---

## Planned Use Cases

### 1. Expression LoRA Training
- Use all 15 character LoRAs
- Generate ~1,500 diverse expression images per character
- Focus on facial variations

### 2. Pose LoRA Training
- Use all 15 character LoRAs
- Generate ~2,000 diverse pose images per character
- Cover all camera angles and body positions

### 3. Action LoRA Training
- Use all 15 character LoRAs
- Generate ~2,000 diverse action images per character
- Include movement, gestures, interactions

---

*Last Updated: 2025-11-30 09:20*
*Document maintained by: AI Assistant*
