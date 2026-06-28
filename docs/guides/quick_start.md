# Quick Start Guide

> **Get started with the 2D Animation LoRA Pipeline in 10 minutes**

This guide walks you through training your first character LoRA from a 2D animated video.

---

## 📋 Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+) or WSL2
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space for datasets

### Software Requirements

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or 12.x
- **Conda**: Miniconda or Anaconda

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/2d-animation-lora-pipeline.git
cd 2d-animation-lora-pipeline
```

### Step 2: Create Environment

```bash
# Create conda environment
conda create -n ai_env python=3.10
conda activate ai_env

# Install PyTorch (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install pipeline dependencies
pip install -r requirements/all.txt
```

### Step 3: Install Optional Dependencies

```bash
# For face-based identity clustering (recommended)
pip install insightface

# For advanced segmentation (optional)
pip install onnxruntime-gpu

# For pose extraction (optional)
pip install mediapipe
```

### Step 4: Verify Installation

```bash
# Quick verification
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Pipeline smoke test
python scripts/run_pipeline.py --help
```

---

## 🎬 Your First Pipeline Run

### Example: Training Homer Simpson LoRA

Let's train a LoRA for Homer Simpson from The Simpsons.

#### Step 1: Prepare Video

```bash
# Create project directory
mkdir -p /path/to/data/simpsons

# Place your video file
cp /path/to/simpsons_episode.mp4 /path/to/data/simpsons/input.mp4
```

#### Step 2: Create Project Config

Create `configs/projects/simpsons.yaml`:

```yaml
project: simpsons
animation_mode: 2d

paths:
  base_dir: /path/to/data/simpsons
  input_video: ${paths.base_dir}/input.mp4
  frames: ${paths.base_dir}/frames
  segmented: ${paths.base_dir}/segmented
  clustered: ${paths.base_dir}/clustered
  training_data: ${paths.base_dir}/training_data

# Frame extraction settings
frame_extraction:
  mode: scene  # Extract frames at scene changes
  scene_threshold: 0.3
  quality: high

# Multi-character extraction
multi_character_extraction:
  # YOLO tracking
  tracking:
    use_stub: true  # Use stub mode for testing
    min_track_length: 10
    tracker_type: bytetrack

  # ToonOut segmentation
  segmentation:
    use_stub: true  # Use stub mode for testing
    backend: stub

  # Identity clustering
  clustering:
    min_cluster_size: 20  # 2D default
    device: cuda

# Dataset building
dataset:
  target_size: 500
  generate_captions: true

# LoRA training
training:
  epochs: 10
  learning_rate: 1e-4
  batch_size: 2
```

#### Step 3: Run Pipeline (Stub Mode)

First, test in stub mode (no model weights needed):

```bash
python scripts/run_pipeline.py \
    --project simpsons \
    --mode 2d \
    --dry-run  # Preview configuration
```

If configuration looks good:

```bash
python scripts/run_pipeline.py \
    --project simpsons \
    --mode 2d
```

#### Step 4: Monitor Progress

The pipeline will show progress for each stage:

```
================================================================================
Starting full pipeline for project: simpsons
Animation mode: 2d
================================================================================

[1/7] Executing stage: frame_extraction - Extract frames from video
  ✓ Extracted 1,234 frames

[2/7] Executing stage: multi_character_extraction
  [2.1] YOLO Detection + Multi-Object Tracking...
    ✓ Found 3 valid tracks
  [2.2] Per-Track ToonOut Segmentation...
    ✓ Segmented 1,234 foreground instances
  [2.3] Face-Based Identity Clustering...
    ✓ Characters found: 2 (character_000: Homer, character_001: Marge)

[3/7] Executing stage: dataset_building
  ✓ Built dataset: 523 images

[4/7] Executing stage: lora_training
  ✓ Training complete: 10 epochs

================================================================================
✓ Pipeline completed successfully (1234.5s)
================================================================================
```

#### Step 5: Check Outputs

```bash
# View directory structure
tree /path/to/data/simpsons -L 2

simpsons/
├── frames/              # Extracted frames
├── segmented/           # Segmented characters by track
│   ├── track_001/
│   ├── track_002/
│   └── track_003/
├── clustered/           # Characters grouped by identity
│   ├── character_000/   # Homer
│   └── character_001/   # Marge
├── training_data/       # Organized training datasets
│   └── character_000/
│       ├── images/
│       └── captions/
└── checkpoints/         # Pipeline checkpoints
```

---

## 🎨 Running with Real Models

Once you're comfortable with stub mode, use real models:

### Step 1: Download Model Weights

```bash
# YOLO detection model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt \
    -O /path/to/models/yolov11n.pt

# (Optional) ToonOut segmentation model
# Download from ToonOut repository
```

### Step 2: Update Config

```yaml
multi_character_extraction:
  tracking:
    use_stub: false
    backend: pytorch
    model_path: /path/to/models/yolov11n.pt

  segmentation:
    use_stub: false  # or true if you don't have ToonOut weights
    backend: onnx
    model_path: /path/to/models/toonout.onnx

  clustering:
    device: cuda  # Uses InsightFace for face recognition
```

### Step 3: Run Pipeline

```bash
python scripts/run_pipeline.py \
    --project simpsons \
    --mode 2d \
    --device cuda
```

---

## 🔧 Common Workflows

### Workflow 1: Single Character Only

```bash
# Skip multi-character extraction, use simple segmentation
python scripts/run_pipeline.py \
    --project single_char \
    --stages frame_extraction,toonout_segmentation,dataset_building,lora_training
```

### Workflow 2: Resume from Checkpoint

```bash
# Pipeline saves checkpoints automatically
python scripts/run_pipeline.py \
    --project simpsons \
    --start-from dataset_building  # Resume from this stage
```

### Workflow 3: Run Specific Stages

```bash
# Extract frames only
python scripts/run_pipeline.py \
    --project simpsons \
    --stages frame_extraction

# Multi-character extraction only
python scripts/run_pipeline.py \
    --project simpsons \
    --stages multi_character_extraction
```

### Workflow 4: Test Configuration

```bash
# Preview without execution
python scripts/run_pipeline.py \
    --project simpsons \
    --dry-run

# Outputs: pipeline_config.json
```

---

## 📊 Understanding the Output

### Frame Extraction

```
frames/
└── video_001/
    ├── frame_000001.png
    ├── frame_000002.png
    └── ...
```

### Multi-Character Extraction

**Track-level segmentation:**
```
segmented/
├── video_001_t0/     # Track 0 (Homer)
│   ├── character/
│   ├── background/
│   └── masks/
└── video_001_t1/     # Track 1 (Marge)
    └── ...
```

**Identity-level clustering:**
```
clustered/
├── character_000/    # Homer (merged from tracks 0, 3, 5)
│   ├── instance_001.png
│   ├── instance_002.png
│   └── ...
└── character_001/    # Marge (merged from tracks 1, 4)
    └── ...
```

### Training Dataset

```
training_data/
└── homer_simpson/
    ├── images/
    │   ├── 001.png
    │   ├── 002.png
    │   └── ...
    └── captions/
        ├── 001.txt
        ├── 002.txt
        └── ...
```

---

## 🛠️ Troubleshooting

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in config
2. Use smaller YOLO model (`yolov11n.pt` instead of `yolov11x.pt`)
3. Enable stub mode for testing
4. Process fewer frames at once

```yaml
multi_character_extraction:
  tracking:
    model_path: /path/to/yolov11n.pt  # Smaller model

training:
  batch_size: 1  # Reduce from 2
```

### Issue: No Characters Detected

```
WARNING: Found 0 valid tracks
```

**Solutions:**
1. Check if video contains visible characters
2. Lower confidence threshold
3. Increase max_dets_per_frame

```yaml
multi_character_extraction:
  tracking:
    conf_threshold: 0.20  # Lower from 0.25
    max_dets_per_frame: 5  # Increase from 1
```

### Issue: Wrong Character Clustering

```
INFO: Characters found: 5 (expected 2)
```

**Solutions:**
1. Adjust clustering parameters
2. Use face-based clustering (requires InsightFace)
3. Manual review and merge clusters

```yaml
multi_character_extraction:
  clustering:
    min_cluster_size: 30  # Increase to merge similar clusters
    similarity_threshold: 0.70  # Lower for more aggressive merging
```

### Issue: Poor Caption Quality

```
Caption: "a person standing"  # Too generic
```

**Solutions:**
1. Use better VLM (Qwen2-VL, InternVL2)
2. Provide character-specific prompts
3. Manual caption editing

```yaml
dataset:
  caption_model: qwen2_vl  # Instead of default
  caption_prefix: "homer simpson character, yellow skin"
```

---

## 🎓 Next Steps

### Learn More

1. **[Multi-Character Workflow](multi_character_workflow.md)**: Advanced multi-character handling
2. **[Configuration Guide](configuration_system.md)**: Deep dive into OmegaConf
3. **[2D vs 3D Parameters](2d_vs_3d_parameters.md)**: Parameter optimization guide
4. **[Migration Guide](../migration/v1_to_v2_migration.md)**: Upgrading from old pipeline

### Advanced Topics

- Custom stage development
- Hyperparameter tuning
- Batch processing multiple projects
- Integration with training frameworks

### Community

- Join discussions on GitHub
- Share your trained LoRAs
- Report issues and feature requests

---

## ✅ Quick Start Checklist

Before running your first pipeline:

- [ ] Python 3.10+ and CUDA installed
- [ ] Repository cloned and dependencies installed
- [ ] Project config created in `configs/projects/`
- [ ] Video file prepared
- [ ] Enough disk space (50GB+ recommended)
- [ ] Test with `--dry-run` flag
- [ ] Start with stub mode for testing

---

**Ready to train your first 2D character LoRA! 🚀**

Need help? Check the [full documentation](../../README.md) or [open an issue](https://github.com/yourusername/2d-animation-lora-pipeline/issues).
