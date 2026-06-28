# Pose/Action/Expression LoRA Training Pipeline Plan

**Status:** Planning Phase
**Date:** 2025-12-04
**Current Progress:** 23,821 synthetic images generated (14 characters × 3 LoRA types)

---

## Overview

Transform synthetic training data into production-ready pose/action/expression LoRAs for multi-LoRA composition workflows.

**Goal:** Enable fine-grained control over character rendering through specialized LoRAs:
- **Pose LoRA**: Body position, stance, limb placement
- **Action LoRA**: Dynamic movements, gestures, activities
- **Expression LoRA**: Facial expressions, emotions

**Current Assets:**
- ✅ 14 BEST identity LoRAs (SDXL)
- ✅ ~23,821 synthetic images generated
- ✅ Template-based prompts with metadata
- ✅ Existing tools: dataset_organizer, training_config_generator, training_launcher

---

## Pipeline Architecture

```
Generated Images (23,821)
  ↓
Quality Filtering & Validation
  ↓
Caption Expansion & Refinement
  ↓
Dataset Organization (Kohya_ss format)
  ↓
Training Config Generation
  ↓
LoRA Training (Automated)
  ↓
Checkpoint Evaluation & Selection
  ↓
Multi-LoRA Composition Testing
```

---

## Phase 1: Quality Filtering & Validation

### Objectives
- Filter out low-quality/failed generations
- Validate image dimensions and format
- Remove near-duplicates
- Organize images by quality tiers

### Implementation Plan

#### 1.1 Quality Filter Module
**File:** `scripts/generic/training/quality_filters/synthetic_image_filter.py`

**Features:**
- **Technical validation:**
  - Resolution check (must be 1024×1024 for SDXL)
  - Format validation (PNG, no corruption)
  - File size sanity check (reject too small = failed generation)

- **Visual quality metrics:**
  - Blur detection (Laplacian variance)
  - Contrast/exposure check (histogram analysis)
  - Artifact detection (JPEG artifacts, noise)

- **Character presence validation:**
  - YOLO-based person detection
  - Face detection (ensure character is present)
  - Bounding box size (reject if character too small)

- **Deduplication:**
  - Perceptual hashing (pHash)
  - SSIM similarity threshold
  - Keep only best quality among duplicates

**Quality Tiers:**
- **Tier A** (Premium): Perfect quality, use for training
- **Tier B** (Good): Minor issues, use if needed
- **Tier C** (Reject): Failed generations, artifacts, blur

**CLI Interface:**
```bash
python scripts/generic/training/quality_filters/synthetic_image_filter.py \
  --input-dir /mnt/data/ai_data/synthetic_lora_data/generated_data \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data \
  --character luca \
  --lora-type pose \
  --tier-threshold 0.7 \
  --enable-dedup \
  --dedup-threshold 0.92 \
  --report filtered_report.json
```

**Output Structure:**
```
filtered_data/
  {character}/
    {lora_type}/
      tier_a/  # Premium quality
        pose_round001_img001.png
        ...
      tier_b/  # Good quality
        pose_round050_img005.png
        ...
      tier_c/  # Rejected
        pose_round100_img010.png  # (for debugging)
        ...
      filtering_report.json
```

**Filtering Report Schema:**
```json
{
  "character": "luca",
  "lora_type": "pose",
  "total_input": 610,
  "tier_a_count": 480,
  "tier_b_count": 95,
  "tier_c_rejected": 35,
  "rejection_reasons": {
    "blur": 12,
    "no_person_detected": 8,
    "artifacts": 7,
    "duplicate": 5,
    "low_contrast": 3
  },
  "deduplication": {
    "duplicates_found": 5,
    "duplicates_removed": 5
  },
  "recommended_for_training": 480
}
```

---

## Phase 2: Caption Expansion & Refinement

### Objectives
- Expand template-based captions with detailed descriptions
- Emphasize LoRA-specific attributes (pose/action/expression)
- Ensure caption consistency within each LoRA type
- Maintain character identity triggers

### Implementation Plan

#### 2.1 Caption Expansion Module
**File:** `scripts/generic/training/caption_engines/synthetic_caption_expander.py`

**Strategy:**
Current captions are template-based and generic. We need to expand them to be:
1. **Descriptive** - Detail the specific pose/action/expression
2. **Consistent** - Same terminology across similar images
3. **Trigger-preserving** - Keep character name + identity markers
4. **LoRA-focused** - Emphasize the attribute being trained

**Expansion Methods:**

##### Method 1: Template Enhancement (Fast, Deterministic)
Extract pose/action/expression from original prompt metadata and expand:

**Original Caption:**
```
luca, 3d animation, pixar style, full body view standing straight with neutral expression
```

**Expanded Caption (Pose LoRA):**
```
luca, full body shot, standing upright pose with arms at sides, neutral stance, feet shoulder-width apart, front-facing view, 3d animated character, pixar style, studio lighting
```

**Expanded Caption (Expression LoRA):**
```
luca, close-up portrait, neutral facial expression, relaxed eyebrows, slight smile, calm eyes, natural mouth position, 3d animated character face, pixar style, soft lighting
```

**Expanded Caption (Action LoRA):**
```
luca, dynamic action pose, running forward motion, arms swinging naturally, forward lean, active movement, energetic stance, 3d animated character, pixar style, motion blur effect
```

##### Method 2: VLM-Assisted Enhancement (Slower, More Accurate)
Use Qwen2-VL to analyze images and generate detailed captions:

**Process:**
1. Sample 20% of images per character/type
2. Send to Qwen2-VL with schema-guided prompt
3. Extract structured description
4. Use as templates for similar images
5. Manual review of VLM outputs

**VLM Prompt Template:**
```
Analyze this 3D animated character image and describe the {lora_type}:

Character: {character_name}
LoRA Type: {lora_type}

Provide a detailed description focusing on:
- [If pose] Body position, limb placement, stance, viewing angle
- [If action] Movement, activity, dynamic elements, motion
- [If expression] Facial features, emotion, eye/mouth/eyebrow position

Output Format (JSON):
{
  "character_trigger": "character name",
  "primary_attribute": "main pose/action/expression",
  "detailed_description": "comprehensive description",
  "technical_terms": ["term1", "term2"],
  "final_caption": "optimized training caption"
}
```

##### Method 3: Hybrid Approach (Recommended)
1. Use template enhancement for bulk processing (fast)
2. VLM-sample 10% for quality validation
3. Build vocabulary from VLM outputs
4. Apply vocabulary consistently to template-enhanced captions

**Caption Vocabulary Database:**
```json
{
  "pose_terms": {
    "standing": ["upright stance", "standing pose", "vertical position"],
    "sitting": ["seated position", "sitting pose", "chair sitting"],
    "walking": ["walking stance", "mid-stride", "walking motion"],
    "running": ["running pose", "forward lean", "dynamic running"]
  },
  "action_terms": {
    "jumping": ["mid-air jump", "jumping motion", "leap"],
    "waving": ["hand wave gesture", "greeting wave", "arm raised"],
    "pointing": ["pointing gesture", "extended arm", "finger pointing"]
  },
  "expression_terms": {
    "happy": ["smiling", "joyful expression", "bright eyes", "wide smile"],
    "sad": ["downcast eyes", "frowning", "sad expression", "tears"],
    "surprised": ["wide eyes", "open mouth", "raised eyebrows", "shocked"]
  }
}
```

**CLI Interface:**
```bash
# Template-based expansion (fast)
python scripts/generic/training/caption_engines/synthetic_caption_expander.py \
  --input-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data/luca/pose/tier_a \
  --prompts-json /mnt/data/ai_data/synthetic_lora_data/generated_data/luca/pose/prompts_converted.json \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data/luca/pose/captioned \
  --method template \
  --vocabulary configs/training/caption_vocabulary.json

# VLM-assisted expansion (slower, more accurate)
python scripts/generic/training/caption_engines/synthetic_caption_expander.py \
  --input-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data/luca/pose/tier_a \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data/luca/pose/captioned \
  --method vlm \
  --vlm-model qwen2_vl \
  --sample-rate 0.2 \
  --device cuda
```

**Output:**
- Each image gets corresponding `.txt` caption file
- `caption_expansion_report.json` with statistics
- `vocabulary_extracted.json` for future use

---

## Phase 3: Dataset Organization

### Objectives
- Convert filtered+captioned images into Kohya_ss format
- Organize by character and LoRA type
- Apply correct repeat counts
- Generate dataset metadata

### Implementation Plan

#### 3.1 Use Existing `dataset_organizer.py`
We already have `scripts/generic/training/dataset_organizer.py` - extend it for synthetic data:

**Dataset Structure (Kohya_ss format):**
```
datasets/
  {character}_{lora_type}/
    12_concept/              # 12 = repeat count
      luca_pose_001.png
      luca_pose_001.txt
      luca_pose_002.png
      luca_pose_002.txt
      ...
    dataset_metadata.json
```

**Repeat Count Strategy:**
- **Pose LoRA:** 8-12 repeats (stable, structural)
- **Action LoRA:** 10-15 repeats (more variation needed)
- **Expression LoRA:** 8-10 repeats (focused on face)

**Dataset Size Targets:**
- **Minimum:** 200 images per LoRA type (for stable training)
- **Recommended:** 400-500 images (better generalization)
- **Maximum:** 800 images (diminishing returns, risk overfitting)

Given ~610 images per character/type after generation, and filtering to ~480 tier A:
- ✅ Sufficient for quality training
- Can afford aggressive quality filtering

**CLI Interface:**
```bash
python scripts/generic/training/dataset_organizer.py \
  --input-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data/luca/pose/tier_a \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/datasets/luca_pose \
  --repeat-count 10 \
  --concept-name luca_pose \
  --target-resolution 1024 \
  --copy-images
```

#### 3.2 Batch Organization Script
**File:** `scripts/batch/organize_all_synthetic_datasets.sh`

Automate organization for all characters × LoRA types:

```bash
#!/bin/bash
# Organize all synthetic data into training datasets

FILTERED_ROOT="/mnt/data/ai_data/synthetic_lora_data/filtered_data"
DATASET_ROOT="/mnt/data/ai_data/synthetic_lora_data/datasets"

for char in alberto bryce caleb elio giulia ian_lightfoot luca miguel orion russell tyler alberto_seamonster luca_seamonster barley_lightfoot; do
  for lora_type in pose action expression; do
    echo "Organizing ${char} ${lora_type}..."

    python scripts/generic/training/dataset_organizer.py \
      --input-dir "${FILTERED_ROOT}/${char}/${lora_type}/tier_a" \
      --output-dir "${DATASET_ROOT}/${char}_${lora_type}" \
      --repeat-count 10 \
      --concept-name "${char}_${lora_type}" \
      --target-resolution 1024 \
      --copy-images
  done
done
```

---

## Phase 4: Training Configuration Generation

### Objectives
- Generate optimized TOML configs for each LoRA type
- Tune hyperparameters based on LoRA type characteristics
- Enable automated batch training

### Implementation Plan

#### 4.1 Training Config Templates

**Base SDXL LoRA Config** (`configs/training/sdxl_lora_base.toml`):
```toml
# Base configuration for SDXL LoRA training
[model]
pretrained_model_name_or_path = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
vae = "/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors"

[dataset]
resolution = "1024,1024"
batch_size = 1
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 2048
bucket_reso_steps = 64

[training]
max_train_epochs = 12
save_every_n_epochs = 2
mixed_precision = "bf16"
gradient_checkpointing = true
gradient_accumulation_steps = 1

[optimizer]
optimizer_type = "AdamW8bit"
learning_rate = 1e-4
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

[network]
network_module = "networks.lora"
network_dim = 32
network_alpha = 16

[output]
output_name = "lora"
save_model_as = "safetensors"
save_precision = "bf16"

[logging]
logging_dir = "./logs"
log_with = "tensorboard"
```

**Pose LoRA Config** (`configs/training/pose_lora_template.toml`):
```toml
# Optimized for pose/stance learning

[model]
pretrained_model_name_or_path = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"

[training]
max_train_epochs = 10  # Pose is structural, converges faster
learning_rate = 8e-5   # Slightly lower LR for stability
network_dim = 32       # Medium rank for body structure
network_alpha = 16

# Pose focuses on body position - stable training
# Lower LR prevents overfitting to specific angles
```

**Action LoRA Config** (`configs/training/action_lora_template.toml`):
```toml
# Optimized for dynamic movement/action

[training]
max_train_epochs = 12  # Actions need more epochs for variation
learning_rate = 1e-4   # Standard LR for dynamic content
network_dim = 48       # Higher rank for motion complexity
network_alpha = 24

# Actions are more complex - need higher capacity
# More epochs to learn diverse movements
```

**Expression LoRA Config** (`configs/training/expression_lora_template.toml`):
```toml
# Optimized for facial expressions

[training]
max_train_epochs = 8   # Facial features converge quickly
learning_rate = 6e-5   # Lower LR for fine facial details
network_dim = 24       # Lower rank, focused on face region
network_alpha = 12

# Expressions are localized to face - lower rank sufficient
# Faster convergence, risk of overfitting if too many epochs
```

#### 4.2 Config Generator Enhancement

Extend `scripts/generic/training/training_config_generator.py` to support LoRA-type-specific tuning:

```bash
python scripts/generic/training/training_config_generator.py \
  --dataset-dir /mnt/data/ai_data/synthetic_lora_data/datasets/luca_pose \
  --output-dir /mnt/c/ai_models/diffusion/lora/sdxl/luca/pose \
  --config-template configs/training/pose_lora_template.toml \
  --character-name luca \
  --lora-type pose \
  --output-name luca_pose_lora
```

#### 4.3 Batch Config Generation Script
**File:** `scripts/batch/generate_all_lora_configs.sh`

```bash
#!/bin/bash
# Generate training configs for all synthetic LoRAs

DATASET_ROOT="/mnt/data/ai_data/synthetic_lora_data/datasets"
CONFIG_ROOT="/mnt/c/ai_models/diffusion/lora/sdxl"

for char in alberto bryce caleb elio giulia ian_lightfoot luca miguel orion russell tyler alberto_seamonster luca_seamonster barley_lightfoot; do
  for lora_type in pose action expression; do
    template="configs/training/${lora_type}_lora_template.toml"

    python scripts/generic/training/training_config_generator.py \
      --dataset-dir "${DATASET_ROOT}/${char}_${lora_type}" \
      --output-dir "${CONFIG_ROOT}/${char}/${lora_type}" \
      --config-template "${template}" \
      --character-name "${char}" \
      --lora-type "${lora_type}" \
      --output-name "${char}_${lora_type}_lora"
  done
done
```

---

## Phase 5: Automated Training

### Objectives
- Train all pose/action/expression LoRAs systematically
- Monitor training progress
- Auto-save best checkpoints based on loss curves
- Handle GPU memory efficiently

### Implementation Plan

#### 5.1 Training Orchestrator
**File:** `scripts/batch/train_all_synthetic_loras.sh`

**Strategy:**
- Sequential training (one LoRA at a time to avoid GPU contention)
- Group by type (all pose → all action → all expression)
- Checkpoint-based resumption
- Automatic error recovery

```bash
#!/bin/bash
# Automated training for all synthetic LoRAs
# Trains in batches: pose → action → expression

set -euo pipefail

CONDA_ENV="kohya_ss"
SCRIPT="/mnt/c/sd-scripts/sdxl_train_network.py"
CONFIG_ROOT="/mnt/c/ai_models/diffusion/lora/sdxl"
LOG_DIR="/mnt/data/ai_data/synthetic_lora_data/logs/training"

mkdir -p "$LOG_DIR"

# Characters to train
CHARACTERS=(
  alberto bryce caleb elio giulia ian_lightfoot
  luca miguel orion russell tyler
  alberto_seamonster luca_seamonster barley_lightfoot
)

# Train by type (pose first, then action, then expression)
for lora_type in pose action expression; do
  echo "========================================"
  echo "Training ${lora_type} LoRAs for all characters"
  echo "========================================"

  for char in "${CHARACTERS[@]}"; do
    config_file="${CONFIG_ROOT}/${char}/${lora_type}/config.toml"

    if [ ! -f "$config_file" ]; then
      echo "Config not found: $config_file, skipping..."
      continue
    fi

    echo "[$(date)] Training ${char} ${lora_type}..."

    conda run -n "$CONDA_ENV" accelerate launch \
      --num_cpu_threads_per_process=8 \
      "$SCRIPT" \
      --config_file="$config_file" \
      2>&1 | tee "${LOG_DIR}/${char}_${lora_type}_$(date +%Y%m%d_%H%M%S).log"

    echo "[$(date)] Completed ${char} ${lora_type}"

    # Brief cooldown between training runs
    sleep 10
  done
done

echo "========================================"
echo "All training completed!"
echo "========================================"
```

#### 5.2 Training Monitor Dashboard
**File:** `scripts/batch/monitor_synthetic_training.sh`

Real-time monitoring of training progress:

```bash
#!/bin/bash
# Monitor training progress for synthetic LoRAs

watch -n 5 '
echo "=== Synthetic LoRA Training Progress ==="
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""
echo "Active Training Processes:"
ps aux | grep "sdxl_train_network.py" | grep -v grep
echo ""
echo "Recent Checkpoints (last 10):"
find /mnt/c/ai_models/diffusion/lora/sdxl -name "*.safetensors" -type f -mmin -120 | tail -10
echo ""
echo "Latest Training Logs:"
tail -5 /mnt/data/ai_data/synthetic_lora_data/logs/training/*.log 2>/dev/null | tail -20
'
```

---

## Phase 6: Checkpoint Evaluation

### Objectives
- Evaluate trained checkpoints systematically
- Select best checkpoint per character/type
- Test multi-LoRA composition
- Generate comparison reports

### Implementation Plan

#### 6.1 Checkpoint Evaluation Module
**File:** `scripts/evaluation/evaluate_synthetic_lora_checkpoints.py`

**Evaluation Strategy:**

**Single LoRA Testing:**
- Generate test images with fixed prompts
- Measure attribute fidelity (pose/action/expression adherence)
- Check identity preservation
- Assess visual quality

**Test Prompts (Pose LoRA):**
```json
{
  "test_prompts": [
    "luca, standing pose, arms at sides, front view, 3d animated character",
    "luca, sitting pose, crossed legs, side view, 3d animated character",
    "luca, walking pose, mid-stride, three-quarter view, 3d animated character",
    "luca, running pose, dynamic stance, front view, 3d animated character"
  ]
}
```

**Test Prompts (Expression LoRA):**
```json
{
  "test_prompts": [
    "luca, happy expression, wide smile, bright eyes, close-up face",
    "luca, sad expression, downcast eyes, frown, close-up face",
    "luca, surprised expression, wide eyes, open mouth, close-up face",
    "luca, neutral expression, calm face, relaxed, close-up face"
  ]
}
```

**Metrics:**
- **Visual Quality**: CLIP aesthetic score, sharpness
- **Attribute Fidelity**: Does it follow the pose/action/expression?
- **Identity Consistency**: Face similarity to reference
- **Composition Stability**: No artifacts, proper anatomy

**CLI:**
```bash
python scripts/evaluation/evaluate_synthetic_lora_checkpoints.py \
  --checkpoint-dir /mnt/c/ai_models/diffusion/lora/sdxl/luca/pose \
  --character luca \
  --lora-type pose \
  --test-prompts configs/evaluation/pose_test_prompts.json \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/evaluation/luca_pose \
  --device cuda
```

#### 6.2 Multi-LoRA Composition Testing
**File:** `scripts/evaluation/test_multi_lora_composition.py`

**Test combinations:**
- Identity + Pose
- Identity + Expression
- Identity + Action
- Identity + Pose + Expression (triple composition)

**Example:**
```bash
python scripts/evaluation/test_multi_lora_composition.py \
  --identity-lora /mnt/c/ai_models/diffusion/lora/sdxl/BEST_CHECKPOINTS_COLLECTION/BEST_luca_lora_sdxl.safetensors \
  --pose-lora /mnt/c/ai_models/diffusion/lora/sdxl/luca/pose/luca_pose_lora-000008.safetensors \
  --expression-lora /mnt/c/ai_models/diffusion/lora/sdxl/luca/expression/luca_expression_lora-000006.safetensors \
  --prompts "luca, running pose, happy expression, 3d animated character" \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/evaluation/multi_lora/luca \
  --lora-scales 0.8,0.6,0.5
```

**Output:**
- Grid comparisons (identity only vs identity+pose vs identity+expression, etc.)
- Optimal scale recommendations
- Composition stability report

---

## Phase 7: Production Deployment

### Objectives
- Select and archive best checkpoints
- Create LoRA usage documentation
- Build example workflows
- Integrate with existing animation pipeline

### Implementation Plan

#### 7.1 Best Checkpoint Selection & Archiving
**File:** `scripts/batch/archive_best_synthetic_loras.sh`

Based on evaluation results, copy best checkpoints to production location:

```bash
PROD_DIR="/mnt/c/ai_models/diffusion/lora/sdxl/PRODUCTION_SYNTHETIC_LORAS"

mkdir -p "$PROD_DIR"/{pose,action,expression}

# Copy best checkpoints (manually selected or auto-selected from evaluation)
cp /mnt/c/ai_models/diffusion/lora/sdxl/luca/pose/BEST_luca_pose_lora.safetensors \
   "$PROD_DIR/pose/"

# Repeat for all characters and types...
```

#### 7.2 Usage Documentation
**File:** `docs/guides/MULTI_LORA_COMPOSITION_GUIDE.md`

- How to combine identity + pose/action/expression LoRAs
- Recommended LoRA scales
- Example prompts
- Troubleshooting common issues

#### 7.3 Integration Scripts

**Quick Generation Script:**
```bash
#!/bin/bash
# Quick multi-LoRA generation

python scripts/evaluation/sdxl_multi_lora_compositor.py \
  --identity-lora /path/to/identity.safetensors \
  --pose-lora /path/to/pose.safetensors \
  --prompt "luca, running pose, 3d animated character" \
  --output output.png
```

---

## Implementation Phases Summary

### Phase 1: Quality Filtering (Week 1)
- [ ] Implement `synthetic_image_filter.py`
- [ ] Test on sample character (luca)
- [ ] Run batch filtering for all characters
- [ ] Review filtering reports, adjust thresholds

### Phase 2: Caption Expansion (Week 1-2)
- [ ] Build caption vocabulary database
- [ ] Implement template-based expansion
- [ ] (Optional) VLM sampling for validation
- [ ] Apply to all filtered images

### Phase 3: Dataset Organization (Week 2)
- [ ] Extend `dataset_organizer.py` if needed
- [ ] Create batch organization script
- [ ] Organize all datasets
- [ ] Validate Kohya_ss format

### Phase 4: Training Config Generation (Week 2)
- [ ] Create LoRA-type-specific config templates
- [ ] Generate configs for all characters/types
- [ ] Validate config correctness

### Phase 5: Training Execution (Week 3-4)
- [ ] Start automated training (pose → action → expression)
- [ ] Monitor training progress
- [ ] Handle any errors/failures
- [ ] Collect all checkpoints

### Phase 6: Evaluation & Selection (Week 4-5)
- [ ] Implement evaluation scripts
- [ ] Test all checkpoints
- [ ] Run multi-LoRA composition tests
- [ ] Select best checkpoints

### Phase 7: Production Deployment (Week 5)
- [ ] Archive best checkpoints
- [ ] Write documentation
- [ ] Create usage examples
- [ ] Integration testing

---

## Expected Outputs

### Training Outputs
- **42 LoRA types total:** 14 characters × 3 types (pose/action/expression)
- **~6-12 checkpoints per LoRA** (saved every 2 epochs)
- **Total checkpoints:** ~250-500 .safetensors files

### Dataset Statistics (Estimated)
| LoRA Type | Images/Char | Total Images (14 chars) | Training Epochs | Checkpoints/Char |
|-----------|-------------|-------------------------|-----------------|------------------|
| Pose      | ~480        | ~6,720                  | 10              | 5                |
| Action    | ~480        | ~6,720                  | 12              | 6                |
| Expression| ~480        | ~6,720                  | 8               | 4                |

### Storage Requirements
- **Datasets:** ~20 GB (organized images + captions)
- **Training checkpoints:** ~40-80 GB (before selection)
- **Final production LoRAs:** ~4-6 GB (best checkpoints only)
- **Evaluation outputs:** ~10-15 GB (test images, reports)

---

## Risk Mitigation

### Potential Issues

1. **Quality Filtering Too Aggressive**
   - **Risk:** Filter removes too many images, insufficient training data
   - **Mitigation:** Set conservative thresholds initially, review tier B images

2. **Caption Quality Inconsistent**
   - **Risk:** Poor captions lead to weak LoRA performance
   - **Mitigation:** Manual review of sample captions, use VLM validation

3. **Overfitting**
   - **Risk:** LoRAs memorize training images, poor generalization
   - **Mitigation:** Monitor validation loss, early stopping, diverse test prompts

4. **Multi-LoRA Conflicts**
   - **Risk:** LoRAs interfere when combined (identity + pose + expression)
   - **Mitigation:** Systematic composition testing, adjust LoRA scales

5. **Training Instability**
   - **Risk:** Loss divergence, mode collapse
   - **Mitigation:** Conservative learning rates, gradient clipping, monitoring

---

## Next Steps

**Immediate Actions:**
1. ✅ Stop synthetic data generation (DONE)
2. Create quality filtering module
3. Test filtering on one character (luca pose)
4. Review filtered results and adjust

**User Decision Points:**
- Confirm filtering thresholds and quality tiers
- Choose caption expansion method (template vs VLM vs hybrid)
- Approve training config templates
- Review Phase 1 outputs before proceeding

---

## Notes

- This plan builds on existing tools and infrastructure
- Modular design allows independent testing of each phase
- Can start with single character (luca) as pilot before scaling
- Training can be parallelized if multiple GPUs available
- Plan assumes SDXL base model + existing identity LoRAs

**End of Plan Document**
