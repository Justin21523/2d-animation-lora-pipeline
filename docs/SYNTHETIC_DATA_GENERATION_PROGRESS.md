# Synthetic Data Generation System - Implementation Progress

## Project Overview

**Goal**: Use existing character identity LoRAs to generate diverse synthetic training data for Expression, Pose, and Action LoRAs.

**Timeline**: 5 weeks
**Total Characters**: 15 available LoRAs
**Target Output**: ~50,000 raw images → ~25,000-30,000 filtered high-quality images
**Data Mixing Ratio**: 70% synthetic / 30% original

## Available Character LoRAs (15 total)

### By Movie
- **Elio** (4): Glordon, Bryce, Elio, Caleb
- **Luca** (5): Luca (human), Luca (seamonster), Alberto (human), Alberto (seamonster), Giulia
- **Coco** (1): Miguel
- **Up** (1): Russell
- **Turning Red** (1): Tyler
- **Onward** (2): Ian Lightfoot, Barley Lightfoot
- **Orion and the Dark** (1): Orion

### LoRA Model Locations
```
/mnt/c/ai_models/diffusion/lora/sdxl/
├── elio/
│   ├── glordon_identity/
│   ├── bryce_identity/
│   ├── elio_identity/
│   └── caleb_identity/
├── luca/
│   ├── luca_identity/
│   ├── luca_seamonster_identity/
│   ├── alberto_identity/
│   ├── alberto_seamonster_identity/
│   └── giulia_identity/
├── coco/
│   └── miguel_identity/
├── up/
│   └── russell_identity/
├── turning-red/
│   └── tyler_identity/
├── onward/
│   ├── ian_lightfoot_identity/
│   └── barley_lightfoot_identity/
└── orion/
    └── orion_identity/
```

## Implementation Plan (5 Weeks)

### Week 1: Core Vocabulary & Prompt Generation ✅ IN PROGRESS

#### Completed Tasks ✅

1. **Vocabulary YAMLs** - DONE
   - ✅ `prompts/generation/vocabulary/expressions.yaml`
     - 6 basic emotions (happy, sad, angry, fearful, surprised, disgusted)
     - 9 complex emotions (confused, excited, nervous, determined, embarrassed, proud, mischievous, thoughtful, curious)
     - 3 neutral states
     - Detailed attributes (intensity, mouth_states, eye_states, brow_positions)
     - Prompt templates and incompatible pairs

   - ✅ `prompts/generation/vocabulary/poses.yaml`
     - Body poses: standing (4 variants), sitting (3), action (8), dynamic (4)
     - Camera angles: horizontal (5), vertical (6)
     - Framing: full_body, 3/4, medium, close-up, extreme_close_up
     - Distribution targets for balanced dataset
     - 20+ total pose combinations

   - ✅ `prompts/generation/vocabulary/actions.yaml`
     - Movement: locomotion (6), rotation (2)
     - Hand actions: gestures (5), reaching (3), holding (2)
     - Interaction: communication (4), physical (4)
     - Activities: sports (3), expressive (3)
     - Resting/static (4), contemplative (3), reactions (4)
     - Expression-action compatibility matrix

2. **Directory Structure** - DONE
   - ✅ Created `/mnt/c/ai_projects/3d-animation-lora-pipeline/prompts/generation/vocabulary/`
   - ✅ Created `/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/synthesis/`

#### Pending Tasks ⏳

3. **Prompt Generator** - TODO
   - ⏳ `scripts/generic/synthesis/prompt_generator.py`
   - Core engine for generating diverse prompts
   - Reads vocabulary YAMLs
   - Enforces category balance (<20% per category)
   - Implements weighted sampling based on distribution targets

4. **Semantic Variator** - TODO
   - ⏳ `scripts/generic/synthesis/semantic_variator.py`
   - Synonym replacement
   - Attribute permutation
   - Template variation

### Week 2: Image Generation Engine (Pending)

Planned components:
- `scripts/generic/synthesis/generator_engine.py` - SDXL wrapper with LoRA loading
- `scripts/generic/synthesis/checkpoint_manager.py` - Fault-tolerant checkpointing (every 100 images)
- `scripts/generic/synthesis/quality_filter.py` - Multi-tier quality filtering
  - Tier 1: InsightFace (face quality)
  - Tier 2: CLIP consistency
  - Tier 3: MUSIQ aesthetic score

### Week 3: Batch Processing (Pending)

- Job planner and batch orchestrator
- Progress monitoring
- GPU memory management

### Week 4: Dataset Integration (Pending)

- Caption converter (synthetic → Kohya format)
- Dataset mixer (70/30 synthetic/original)
- Integration with existing `prepare_training_data.py`

### Week 5: CLI & Testing (Pending)

- User-facing CLI tools
- Configuration YAMLs
- End-to-end testing

## Target Dataset Distribution

### Expression LoRA Dataset
- 15 expression types × 100 images/type = 1,500 images per character
- 15 characters × 1,500 = 22,500 total images (before filtering)
- After 90% quality filter: ~20,000 images
- Mixed with original: 20,000 × 0.7 + original × 0.3 = final dataset

### Pose LoRA Dataset
- 64 pose combinations (8 body × 8 camera) × 30 images = 1,920 images per character
- 15 characters × 1,920 = 28,800 total images (before filtering)
- After filtering and mixing: similar ratio

### Action LoRA Dataset
- 20 action types × 100 images/type = 2,000 images per character
- 15 characters × 2,000 = 30,000 total images (before filtering)

### Total Estimated Output
- Raw generation: ~82,000 images
- After quality filter (90%): ~74,000 images
- Final datasets (after mixing): ~50,000-60,000 images across 3 LoRA types

## Quality Filtering Strategy

### Multi-Tier Filtering (90% retention target)

1. **Tier 1: Face Quality (InsightFace)**
   - Reject: No face detected, severely distorted faces
   - Threshold: Face confidence > 0.7
   - Expected rejection: ~5%

2. **Tier 2: CLIP Consistency**
   - Measure: CLIP similarity between prompt and generated image
   - Threshold: Similarity > 0.25
   - Expected rejection: ~3%

3. **Tier 3: Aesthetic Quality (MUSIQ)**
   - Measure: Overall image aesthetic score
   - Threshold: Score > 50 (out of 100)
   - Expected rejection: ~2%

**Total expected retention**: ~90% (22,500 rejected from 82,000)

## Checkpointing Strategy

- **Checkpoint every 100 images** to prevent data loss
- Checkpoint format: JSON with metadata
  ```json
  {
    "character": "bryce",
    "lora_type": "expression",
    "progress": {
      "total_planned": 1500,
      "generated": 847,
      "passed_filter": 763,
      "failed_filter": 84
    },
    "last_prompt": "...",
    "timestamp": "2025-11-30T10:00:00Z"
  }
  ```

## Vocabulary Statistics

### Expressions (expressions.yaml)
- **Basic emotions**: 6 types
- **Complex emotions**: 9 types
- **Neutral states**: 3 types
- **Total unique expressions**: 18
- **Intensity modifiers**: 5 levels
- **Mouth states**: 9 variants
- **Eye states**: 8 variants
- **Brow positions**: 5 variants
- **Prompt templates**: 4 patterns

### Poses (poses.yaml)
- **Body poses**: 20+ variants
- **Camera horizontal angles**: 5 (front, 3/4, side, back, OTS)
- **Camera vertical angles**: 6 (eye level, low, high, Dutch, extreme variants)
- **Framing options**: 5 (full body to extreme close-up)
- **Total pose combinations**: 64 (8 body × 8 camera)

### Actions (actions.yaml)
- **Movement actions**: 8 types
- **Hand gestures**: 10 types
- **Interaction actions**: 8 types
- **Physical activities**: 6 types
- **Contemplative actions**: 3 types
- **Reaction actions**: 4 types
- **Total unique actions**: 39

## Next Steps (After Training Completes)

1. ✅ Wait for Batch 2 retraining to complete (Alberto, Giulia, Russell)
2. ⏳ Implement `prompt_generator.py`
3. ⏳ Implement `semantic_variator.py`
4. ⏳ Test prompt generation with sample outputs
5. ⏳ Proceed to Week 2 implementation

## Dependencies Required

```python
# Already available in environment
- torch
- diffusers
- transformers
- safetensors
- pyyaml
- omegaconf

# May need to add
- insightface  # For face quality filtering
- open_clip    # For CLIP consistency check
- pyiqa        # For MUSIQ aesthetic scoring (or implement alternative)
```

## File Structure (Current)

```
3d-animation-lora-pipeline/
├── prompts/
│   └── generation/
│       └── vocabulary/
│           ├── expressions.yaml ✅
│           ├── poses.yaml ✅
│           └── actions.yaml ✅
├── scripts/
│   └── generic/
│       └── synthesis/
│           ├── prompt_generator.py (TODO)
│           ├── semantic_variator.py (TODO)
│           ├── generator_engine.py (Week 2)
│           ├── checkpoint_manager.py (Week 2)
│           ├── quality_filter.py (Week 2)
│           ├── batch_orchestrator.py (Week 3)
│           └── ... (more to come)
└── docs/
    ├── TRAINING_LOG.md ✅
    └── SYNTHETIC_DATA_GENERATION_PROGRESS.md ✅ (this file)
```

---

*Status: Week 1 - 40% Complete (Vocabularies done, prompt generator pending)*
*Last Updated: 2025-11-30 09:15*
