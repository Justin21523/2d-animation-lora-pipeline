# LoRA Model Inventory - 3D Animation Pipeline

**Last Updated**: 2025-11-28
**Total LoRAs**: 97 safetensors files
**Total Storage**: ~8.6 GB

---

## Quick Reference

### SDXL LoRAs (Ready for Composition Testing)

| Character | Movie | Best Checkpoint | Path | Size |
|-----------|-------|----------------|------|------|
| Miguel | Coco | `miguel_identity_lora_sdxl-000004.safetensors` | `/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/` | 436MB |
| Luca | Luca | `luca_sdxl_RECOMMENDED.safetensors` | `/mnt/c/ai_models/lora/luca/sdxl_trial1/` | 871MB |
| Alberto | Luca | `alberto_identity_lora_sdxl-000002.safetensors` | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/` | 218MB |
| Elio | Elio | `elio_identity_lora_sdxl-000004.safetensors` | `/mnt/data/training/lora/elio/elio_identity/` | 436MB |

### SD 1.5 LoRAs (Full Collection)

| Character | Movie | Best Checkpoint | Path | Epochs |
|-----------|-------|----------------|------|--------|
| Miguel | Coco | `miguel_rivera_identity_lora.safetensors` | `/mnt/c/ai_models/lora/coco/miguel_identity/` | 12 |
| Alberto | Luca | `alberto_scorfano_identity_lora_BEST.safetensors` | `/mnt/c/ai_models/lora/luca/alberto_identity/` | 10 |
| Giulia | Luca | `giulia_identity_lora.safetensors` | `/mnt/c/ai_models/lora/luca/giulia_identity/` | 12 |
| Luca (Trial35) | Luca | `luca_trial35.safetensors` | `/mnt/c/ai_models/lora/luca/trial35/` | 16 |
| Bryce | Elio | `bryce_markwell_identity_lora.safetensors` | `/mnt/c/ai_models/lora/elio/bryce_identity/` | 12 |
| Caleb | Elio | `caleb_identity_lora.safetensors` | `/mnt/c/ai_models/lora/elio/caleb_identity/` | 15 |
| Ian | Onward | `ian_lightfoot_identity_lora.safetensors` | `/mnt/c/ai_models/lora/onward/ian_lightfoot_identity/` | 12 |
| Barley | Onward | `barley_lightfoot_identity_lora.safetensors` | `/mnt/c/ai_models/lora/onward/barley_lightfoot_identity/` | 12 |
| Tyler | Turning Red | `tyler_identity_lora.safetensors` | `/mnt/c/ai_models/lora/turning-red/tyler_identity/` | 12 |
| Russell | Up | `russell_identity_lora.safetensors` | `/mnt/c/ai_models/lora/up/russell_identity/` | 14 |
| Orion | Orion | `orion_identity_lora.safetensors` | `/mnt/c/ai_models/lora/orion/orion_identity/` | 10 |

---

## Detailed SDXL LoRA Inventory

### 1. Miguel (Coco) - SDXL

**Location**: `/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/`
**Size**: 436MB per checkpoint
**Training Status**: ✅ Completed (Partial - 3 epochs only)
**Last Modified**: 2025-11-23

**Available Checkpoints**:
- `miguel_identity_lora_sdxl-000002.safetensors` (Epoch 2) - 436MB
- `miguel_identity_lora_sdxl-000004.safetensors` (Epoch 4) - 436MB ⭐ **Recommended**
- `miguel_identity_lora_sdxl-000006.safetensors` (Epoch 6) - 436MB

**Evaluation**:
- Directory: `simple_test_20251123_004808/`
- Contains test images for all 3 checkpoints

**Notes**:
- Training stopped early (only 6 epochs vs typical 10-12)
- Epoch 4 shows good balance between fidelity and flexibility
- Missing: Epoch 8, 10, 12, final checkpoint

---

### 2. Luca - SDXL (Multiple Trials)

#### Trial 1 (Recommended) ⭐

**Location**: `/mnt/c/ai_models/lora/luca/sdxl_trial1/`
**Size**: 871MB per checkpoint (High rank - 2x standard)
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-16

**Available Checkpoints**:
- `luca_sdxl-000006.safetensors` (Epoch 6) - 871MB
- `luca_sdxl-000008.safetensors` (Epoch 8) - 871MB
- `luca_sdxl-000010.safetensors` (Epoch 10) - 871MB
- `luca_sdxl.safetensors` (Final) - 871MB
- `luca_sdxl_RECOMMENDED.safetensors` (Marked best) - 871MB ⭐ **Use This**

**Evaluation**:
- Directory: `sample/` with generated test images

**Notes**:
- Higher capacity network (871MB vs typical 436MB)
- Likely uses rank 128 instead of 64
- RECOMMENDED version is manually curated as best

#### Trial 1 Old (10x20 - Archived)

**Location**: `/mnt/c/ai_models/lora/luca/sdxl_trial1_old_10x20/`
**Size**: 871MB per checkpoint
**Training Status**: ⚠️ Superseded by Trial 1

**Available Checkpoints**:
- `luca_sdxl-000002.safetensors` (Epoch 2) - 871MB
- `luca_sdxl-000004.safetensors` (Epoch 4) - 871MB

**Notes**:
- Earlier training run with different hyperparameters
- Kept for comparison purposes
- Use Trial 1 RECOMMENDED instead

---

### 3. Alberto (Luca) - SDXL

**Location**: `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/`
**Size**: 218MB per checkpoint (Lower rank)
**Training Status**: 🔄 Early Training (Just started Nov 24)
**Last Modified**: 2025-11-24

**Available Checkpoints**:
- `alberto_identity_lora_sdxl-000001.safetensors` (Epoch 1) - 218MB
- `alberto_identity_lora_sdxl-000002.safetensors` (Epoch 2) - 218MB ⭐ **Latest**

**Backup**:
- `backup_20251124_201728/alberto_identity_lora_sdxl-000002.safetensors`

**Evaluation**:
- Directory: `sample/` with early test images

**Notes**:
- Recently started training (Nov 24, 2025)
- Lower file size (218MB) suggests rank 32 or lower
- Only 2 epochs completed so far
- May be actively training or paused

---

### 4. Elio - SDXL

**Location**: `/mnt/data/training/lora/elio/elio_identity/`
**Size**: 436MB per checkpoint
**Training Status**: 🔄 **Active Training**
**Last Modified**: 2025-11-28 08:56 (Most Recent!)

**Available Checkpoints**:
- `elio_identity_lora_sdxl-000002.safetensors` (Epoch 2) - 436MB
- `elio_identity_lora_sdxl-000004.safetensors` (Epoch 4) - 436MB ⭐ **Latest**

**Logs Location**: `/mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/logs/`

**Notes**:
- **Currently in active training** (last update Nov 28, 09:00)
- Standard SDXL rank (436MB)
- Split location: checkpoints in `/mnt/data/training/`, logs in `/mnt/c/ai_models/`
- Expected to continue to epoch 10-12

**Trash**:
- `/mnt/c/.Trash-1000/files/elio_identity_lora_sdxl-000002.safetensors` (436MB, Nov 28)
  - Old version, superseded by version in training directory
  - Can be safely deleted

---

### 5. Orion - SDXL ⚠️ CRITICAL ISSUE

**Expected Location**: `/mnt/c/ai_models/diffusion/lora/sdxl/orion/orion_identity/`
**Training Status**: ❌ **CHECKPOINTS MISSING**
**Last Evaluation**: 2025-11-23 12:25

**Problem**:
- **NO safetensors files found** in the training directory
- **BUT**: Extensive evaluation results exist with 48 test images

**Evidence of Completed Training**:
- `eval_orion_identity_lora_sdxl/` - 24 test images
- `eval_orion_identity_lora_sdxl-000002/` - 24 test images
- Evaluation metadata shows successful generation
- Total evaluation data: 1GB+

**Possible Locations** (TO BE SEARCHED):
- External backup drives (if mounted)
- `/mnt/*/backups/orion/`
- `/mnt/*/archive/lora/orion/`
- Cloud backup (Google Drive, Dropbox, etc.)

**Action Required**:
1. Search all mounted drives for `orion*sdxl*.safetensors`
2. Check training logs for original output path
3. If not found, mark for retraining

---

## Detailed SD 1.5 LoRA Inventory

### Coco

#### Miguel Rivera

**Location**: `/mnt/c/ai_models/lora/coco/miguel_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-21 03:59

**Checkpoints** (7 total):
- `miguel_rivera_identity_lora-000002.safetensors` (Epoch 2)
- `miguel_rivera_identity_lora-000004.safetensors` (Epoch 4)
- `miguel_rivera_identity_lora-000006.safetensors` (Epoch 6)
- `miguel_rivera_identity_lora-000008.safetensors` (Epoch 8)
- `miguel_rivera_identity_lora-000010.safetensors` (Epoch 10)
- `miguel_rivera_identity_lora-000012.safetensors` (Epoch 12)
- `miguel_rivera_identity_lora.safetensors` (Final) ⭐

**Evaluations**: 4 quality test directories

---

### Luca

#### Alberto Scorfano

**Location**: `/mnt/c/ai_models/lora/luca/alberto_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed + Marked BEST
**Last Modified**: 2025-11-21 06:54

**Checkpoints** (7 total):
- `alberto_scorfano_identity_lora-000002.safetensors` (Epoch 2)
- `alberto_scorfano_identity_lora-000004.safetensors` (Epoch 4)
- `alberto_scorfano_identity_lora-000006.safetensors` (Epoch 6)
- `alberto_scorfano_identity_lora-000008.safetensors` (Epoch 8)
- `alberto_scorfano_identity_lora-000010.safetensors` (Epoch 10)
- `alberto_scorfano_identity_lora.safetensors` (Final)
- `alberto_scorfano_identity_lora_BEST.safetensors` ⭐ **Use This**

**Evaluations**: Full evaluation directory with detailed results

---

#### Giulia

**Location**: `/mnt/c/ai_models/lora/luca/giulia_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-22 03:54

**Checkpoints** (7 total):
- Epochs 2, 4, 6, 8, 10, 12 + Final

**Evaluations**: `evaluations_20251121_195640/`

---

#### Luca Trial 35 (High Capacity)

**Location**: `/mnt/c/ai_models/lora/luca/trial35/`
**Size**: 145MB per checkpoint (2x standard size - Higher rank)
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-15 00:29

**Checkpoints** (9 total):
- Epochs 2, 4, 6, 8, 10, 12, 14, 16 + Final

**Special Features**:
- Higher capacity network (145MB vs 73MB)
- Likely rank 128 vs standard rank 64
- More detailed character representation
- Includes evaluation/ and sample/ directories

---

### Elio

#### Bryce Markwell

**Location**: `/mnt/c/ai_models/lora/elio/bryce_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-22 07:19

**Checkpoints** (5 total):
- Epochs 3, 6, 9, 12 + Final

**Evaluations**: `evaluations_20251122/`

**Notes**: Non-standard epoch intervals (every 3 epochs)

---

#### Caleb

**Location**: `/mnt/c/ai_models/lora/elio/caleb_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-22 07:34

**Checkpoints** (6 total):
- Epochs 3, 6, 9, 12, 15 + Final

**Evaluations**: `evaluations_20251122/`

**Notes**: Extended to epoch 15 (unusual)

---

### Onward

#### Ian Lightfoot

**Location**: `/mnt/c/ai_models/lora/onward/ian_lightfoot_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-21 16:43

**Checkpoints** (7 total):
- Epochs 2, 4, 6, 8, 10, 12 + Final

**Evaluations**: `evaluations_20251121_152947/`

---

#### Barley Lightfoot

**Location**: `/mnt/c/ai_models/lora/onward/barley_lightfoot_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-22 00:42

**Checkpoints** (7 total):
- Epochs 2, 4, 6, 8, 10, 12 + Final

**Evaluations**: `evaluations_20251121_195640/`

---

### Turning Red

#### Tyler

**Location**: `/mnt/c/ai_models/lora/turning-red/tyler_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-21 23:21

**Checkpoints** (7 total):
- Epochs 2, 4, 6, 8, 10, 12 + Final

**Evaluations**: `evaluations_20251121_195640/`

---

### Up

#### Russell

**Location**: `/mnt/c/ai_models/lora/up/russell_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed
**Last Modified**: 2025-11-21 21:28

**Checkpoints** (8 total):
- Epochs 2, 4, 6, 8, 10, 12, 14 + Final

**Evaluations**: `evaluations_20251121_195640/`

**Notes**: Extended training to epoch 14

---

### Orion

#### Orion

**Location**: `/mnt/c/ai_models/lora/orion/orion_identity/`
**Size**: 73MB per checkpoint
**Training Status**: ✅ Completed (SD 1.5)
**Last Modified**: 2025-11-21 10:59

**Checkpoints** (6 total):
- Epochs 2, 4, 6, 8, 10 + Final

**Evaluations**: `evaluations_20251121_093117/`

**Notes**: Missing some intermediate epochs

---

## SDXL Training Status Summary

### ✅ Completed & Ready
- Miguel (Coco) - 3 checkpoints
- Luca (Luca) - 5 checkpoints + RECOMMENDED
- Alberto (Luca) - 2 checkpoints (early stage)
- Elio (Elio) - 2 checkpoints (active training)

### ❌ Missing Checkpoints
- **Orion** - Evaluation exists, checkpoints missing (CRITICAL)

### 📁 Empty Directories (Not Started)
- Giulia (Luca SDXL) - `/mnt/c/ai_models/diffusion/lora/sdxl/luca/giulia_identity/`
- Bryce (Elio SDXL) - `/mnt/c/ai_models/diffusion/lora/sdxl/elio/bryce_identity/`
- Caleb (Elio SDXL) - `/mnt/c/ai_models/diffusion/lora/sdxl/elio/caleb_identity/`
- Glordon (Elio SDXL) - `/mnt/c/ai_models/diffusion/lora/sdxl/elio/glordon_identity/`

### ⏹️ Training Completed But Checkpoints Removed
- Barley (Onward SDXL) - `.training_complete` marker exists
- Ian (Onward SDXL) - `.training_complete` marker exists
- **Note**: User confirmed these characters don't need SDXL versions

### 🔄 Logs Only (Training Started But Not Completed)
- Tyler (Turning Red SDXL) - Logs from Nov 23, no checkpoints yet

### ❓ Not Started
- Russell (Up SDXL) - No SDXL directory exists

---

## Recommended Usage for Composition Testing

### SDXL Multi-Character Combinations

**Best for Generation Quality (1024x1024)**:
1. **Miguel + Luca** - Two protagonists from different films
2. **Miguel + Alberto** - Different checkpoint sizes (436MB + 218MB)
3. **Luca + Alberto** - Same film characters, different training methods
4. **All four**: Miguel + Luca + Alberto + Elio - Grand ensemble!

**Weight Recommendations**:
- Balanced: `(1.0, 1.0)` for equal presence
- Primary/Secondary: `(1.0, 0.7)` for main character focus
- Subtle: `(0.8, 0.8)` to reduce LoRA intensity
- Gradient: `(1.0, 0.8, 0.6)` for three characters

### SD 1.5 Combinations

**Cross-Film Ensembles**:
- Kids group: Miguel + Russell + Tyler
- Adventure team: Ian + Barley + Luca + Alberto
- Full cast: All 11 characters!

**Same-Film Groups**:
- Luca family: Luca + Alberto + Giulia (3 characters)
- Onward brothers: Ian + Barley
- Elio friends: Bryce + Caleb

---

## Storage Breakdown

### By Format
- **SDXL**: ~3.4 GB (10 files + 2 in-progress)
- **SD 1.5**: ~5.1 GB (70 files)
- **High-rank variants**: ~1.5 GB (Trial35 + SDXL Trial1)
- **Total**: ~8.6 GB

### By Movie
- **Luca**: ~2.8 GB (most variants)
- **Elio**: ~1.5 GB
- **Coco**: ~0.9 GB
- **Onward**: ~1.0 GB
- **Others**: ~2.4 GB

---

## Action Items

### Immediate
- [ ] Search for missing Orion SDXL checkpoints
- [ ] Delete old Elio checkpoint from trash
- [ ] Document Alberto SDXL training completion

### Future
- [ ] Consider training Giulia SDXL (to complete Luca trio)
- [ ] Monitor Elio SDXL training progress
- [ ] Evaluate if Russell/Tyler/others need SDXL versions

---

## Notes

- All SD 1.5 LoRAs have comprehensive evaluation directories
- SDXL training appears to use different epoch intervals than SD 1.5
- File sizes indicate different network ranks:
  - 73MB = Standard SD 1.5 (rank 64)
  - 145MB = High-rank SD 1.5 (rank 128)
  - 218MB = Low-rank SDXL (rank 32)
  - 436MB = Standard SDXL (rank 64)
  - 871MB = High-rank SDXL (rank 128)

**Last Updated**: 2025-11-28 09:00 UTC+0
**Maintained By**: 3D Animation LoRA Pipeline Project
