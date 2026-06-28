# CPU-Based LoRA Data Preparation Guide

Complete guide for preparing multi-type LoRA training data using CPU-intensive tasks.

---

## Overview

This pipeline prepares training data for **5 types of LoRA models**:

| LoRA Type | Purpose | Data Source | CPU Tasks |
|-----------|---------|-------------|-----------|
| **Character** | Character identity | SAM2 instances | Face clustering (already done) |
| **Background** | Scene/environment | Inpainted backgrounds | CLIP clustering |
| **Pose/Action** | Body poses/actions | SAM2 instances | Visual clustering |
| **Expression** | Facial expressions | SAM2 instances | Face detection + clustering |
| **Style** | Visual style/rendering | Original frames | Quality filtering + sampling |

**Key Advantage**: All tasks are **CPU-based** and can run in parallel with GPU SAM2 processing.

---

## Prerequisites

### 1. Required Data

Each film needs:
- ✅ `{film}_instances_sam2_v2/instances/` - Character instances (from SAM2)
- ✅ `backgrounds_lama_v2/` - Inpainted backgrounds (from LaMa)
- ✅ `frames_final/` - Original frames (for style LoRA)
- ✅ `identity_clusters/` - Face-clustered identities (for character LoRA)

### 2. Python Dependencies

```bash
conda activate ai_env
pip install open_clip_torch pillow opencv-python scikit-learn umap-learn imagehash
```

---

## Quick Start

### Process All Films, All LoRA Types

```bash
bash scripts/batch/batch_lora_data_preparation.sh all
```

This will process **coco, elio, luca, onward, turning-red** for:
- Background clustering
- Action clustering
- Expression clustering
- Style frame selection

### Process Specific Films

```bash
bash scripts/batch/batch_lora_data_preparation.sh luca onward
```

### Process Specific LoRA Types

```bash
bash scripts/batch/batch_lora_data_preparation.sh all --lora-types background,style
```

---

## Individual Tools

Each tool can be run independently:

### 1. Background Clustering (CLIP-based)

Groups background images by visual similarity (scenes, locations).

```bash
conda run -n ai_env python scripts/generic/clustering/clip_character_clustering.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2 \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/background_clusters \
  --min-cluster-size 20 \
  --min-samples 3 \
  --device cpu
```

**Output**:
```
lora_data/background_clusters/
├── character_0/   # Scene type 1 (e.g., seaside town)
├── character_1/   # Scene type 2 (e.g., underwater)
├── character_2/   # Scene type 3 (e.g., indoor house)
└── noise/         # Unclassified
```

### 2. Action Clustering (Visual-based)

Groups character instances by pose/action.

```bash
conda run -n ai_env python scripts/generic/clustering/action_clustering.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/action_clusters \
  --method visual \
  --device cpu \
  --min-cluster-size 15
```

**Output**:
```
lora_data/action_clusters/
├── action_000/   # Pose group 1 (e.g., standing)
├── action_001/   # Pose group 2 (e.g., running)
├── action_002/   # Pose group 3 (e.g., sitting)
└── noise/        # Unclassified
```

### 3. Expression Classification

Groups faces by expression.

```bash
conda run -n ai_env python scripts/generic/face/expression_classification.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2/instances \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/expression_clusters \
  --method clip \
  --device cpu
```

**Output**:
```
lora_data/expression_clusters/
├── expression_000/   # Expression type 1 (e.g., happy)
├── expression_001/   # Expression type 2 (e.g., surprised)
├── expression_002/   # Expression type 3 (e.g., neutral)
└── unclassified/     # Unclassified
```

### 4. Style Frame Selection

Selects representative frames for style LoRA.

```bash
conda run -n ai_env python scripts/generic/quality/style_frame_selector.py \
  /mnt/data/ai_data/datasets/3d-anime/luca/frames_final \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/style_frames \
  --target-count 400 \
  --device cpu
```

**Output**:
```
lora_data/style_frames/
├── frame_00001.jpg
├── frame_00045.jpg
├── ...
└── style_selection.json
```

---

## Monitoring Progress

### Real-time Monitor

```bash
# Continuous monitoring (refreshes every 10 seconds)
watch -n 10 'bash scripts/batch/monitor_lora_preparation.sh'
```

### Manual Checks

```bash
# Check active processes
ps aux | grep "clustering\|expression\|style" | grep python

# Check CPU usage
top -bn1 | grep python | head -10

# Check output directories
ls -lh /mnt/data/ai_data/datasets/3d-anime/*/lora_data/

# Check logs
tail -f logs/lora_preparation/*.log
```

---

## Output Structure

After processing, each film will have:

```
/mnt/data/ai_data/datasets/3d-anime/luca/lora_data/
├── background_clusters/      # For Background LoRA
│   ├── character_0/  (~500 images)
│   ├── character_1/  (~400 images)
│   └── ...
├── action_clusters/           # For Pose LoRA
│   ├── action_000/   (~200 images)
│   ├── action_001/   (~150 images)
│   └── ...
├── expression_clusters/       # For Expression LoRA
│   ├── expression_000/  (~100 images)
│   ├── expression_001/  (~80 images)
│   └── ...
└── style_frames/              # For Style LoRA
    ├── frame_*.jpg   (400 images)
    └── style_selection.json
```

---

## Next Steps: Caption Generation

After clustering, manually review and rename clusters, then generate captions:

### 1. Manual Review

```bash
# Browse clusters visually
nautilus /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/background_clusters/

# Rename clusters with descriptive names
mv character_0 portorosso_town
mv character_1 underwater_scenes
mv character_2 luca_house_interior
```

### 2. Generate Captions (LLMProvider API)

For **Background LoRA**:

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/background_clusters/portorosso_town \
  --output-dir /mnt/data/ai_data/training_data/luca/background_portorosso \
  --lora-type background \
  --scene-info "Italian coastal town with colorful buildings, narrow streets, Mediterranean architecture" \
  --film-info "Luca (2021), Pixar Animation, warm sunny colors, hand-painted feel" \
  --api-provider llm_vendor \
  --model llm_provider-3-haiku-20240307
```

For **Pose LoRA**:

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/action_clusters/action_000 \
  --output-dir /mnt/data/ai_data/training_data/luca/pose_running \
  --lora-type pose \
  --pose-info "running pose, forward lean, arms swinging, dynamic motion" \
  --api-provider llm_vendor
```

For **Expression LoRA**:

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/expression_clusters/expression_000 \
  --output-dir /mnt/data/ai_data/training_data/luca/expression_happy \
  --lora-type expression \
  --expression-info "happy expression, wide smile, bright eyes, joyful" \
  --api-provider llm_vendor
```

For **Style LoRA**:

```bash
conda run -n ai_env python scripts/generic/training/api_caption_generator.py \
  --input-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/style_frames \
  --output-dir /mnt/data/ai_data/training_data/luca/style_pixar \
  --lora-type style \
  --style-info "Pixar 3D animation style, smooth PBR shading, warm Mediterranean lighting, vibrant colors" \
  --api-provider llm_vendor
```

---

## Estimated Processing Times

| Task | Per Film | All 5 Films (Parallel) |
|------|----------|------------------------|
| Background Clustering | 30-45 min | ~45 min |
| Action Clustering | 2-3 hours | ~3 hours |
| Expression Clustering | 1-2 hours | ~2 hours |
| Style Frame Selection | 20-30 min | ~30 min |
| **Total** | **4-6 hours** | **~4 hours** |

**Note**: Times assume CPU-only processing with 3-4 parallel jobs.

---

## Troubleshooting

### Issue: CLIP model download fails

```bash
# Pre-download CLIP model
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"
```

### Issue: Out of memory

Reduce parallel jobs in `batch_lora_data_preparation.sh`:

```bash
PARALLEL_JOBS=2  # Change from 3 to 2
```

### Issue: No clusters found

Check `min-cluster-size`:
- For backgrounds: Try 15-25
- For actions: Try 10-20
- For expressions: Try 8-15

```bash
# Lower thresholds
--min-cluster-size 10
```

---

## Cost Estimates (LLMProvider API Captions)

Assuming LLMProvider 3 Haiku ($0.25/1M input, $1.25/1M output):

| LoRA Type | Images/Film | Cost/Film | 5 Films Total |
|-----------|-------------|-----------|---------------|
| Background | 500 | $0.60 | $3.00 |
| Pose | 200 | $0.24 | $1.20 |
| Expression | 100 | $0.12 | $0.60 |
| Style | 400 | $0.48 | $2.40 |
| **Total** | **1,200** | **$1.44** | **$7.20** |

**Very affordable** compared to training time saved!

---

## FAQ

**Q: Can I run this while SAM2 is processing?**
A: Yes! All tasks are CPU-only and won't affect GPU workload.

**Q: How long does the full pipeline take?**
A: ~4-6 hours for all 5 films, all LoRA types (CPU parallel).

**Q: What if I only want background and style LoRAs?**
A: Use `--lora-types background,style` flag.

**Q: Do I need to process all films?**
A: No, you can specify individual films or process incrementally.

**Q: Can I use GPU for faster processing?**
A: Yes, but not recommended while SAM2 is running. After SAM2 completes, use `--device cuda`.

---

## Related Documentation

- `docs/training/multi-type-lora-system.md` - LoRA system overview
- `docs/training/lora_types/BACKGROUND_LORA_DEEP_DIVE.md` - Background LoRA details
- `docs/training/lora_types/POSE_LORA_DEEP_DIVE.md` - Pose LoRA details
- `docs/training/lora_types/EXPRESSION_LORA_DEEP_DIVE.md` - Expression LoRA details
- `docs/films/FILM_CHARACTER_CONTEXT.yaml` - Character context for captions
