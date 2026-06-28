# AI_WAREHOUSE 3.0 Migration Guide

> **Centralized storage configuration for the 2D Animation LoRA Pipeline**
>
> This guide explains the AI_WAREHOUSE 3.0 storage architecture and how to migrate from deprecated paths.

---

## Overview

AI_WAREHOUSE 3.0 organizes storage across two physical drives for optimal performance:

| Drive | Size | Purpose | Speed |
|-------|------|---------|-------|
| `/mnt/c/` | 2TB SSD | Models, Code, Cache | Fast |
| `/mnt/data/` | 4TB HDD | Datasets, Training outputs | Large |

---

## Storage Layout

### `/mnt/c/` (2TB SSD - Fast Storage)

```
/mnt/c/
├── ai_models/              # All model weights
│   ├── segmentation/       # SAM2, ToonOut, BiRefNet
│   ├── detection/          # YOLO models
│   ├── pose/               # DWPose, RTMPose
│   ├── clip/               # CLIP models
│   ├── face/               # ArcFace, RetinaFace
│   ├── vlm/                # Qwen2-VL, InternVL2
│   ├── inpainting/         # LaMa, PowerPaint
│   ├── upscaling/          # RealESRGAN, CodeFormer
│   ├── lora/               # SD1.5 LoRA outputs
│   ├── lora_sdxl/          # SDXL LoRA outputs
│   └── stable-diffusion/   # Base SD models
│       ├── checkpoints/
│       └── vae/
│
├── ai_projects/            # Git repositories
│   ├── 2d-animation-lora-pipeline/  # This project
│   └── 3d-animation-lora-pipeline/
│
├── ai_tools/               # Standalone tools
│   ├── kohya_ss/           # LoRA training
│   └── comfyui/            # Image generation
│
└── ai_cache/               # Cache directories
    ├── huggingface/        # HF_HOME
    ├── torch/              # TORCH_HOME
    └── pip/                # PIP_CACHE_DIR
```

### `/mnt/data/` (4TB HDD - Large Storage)

```
/mnt/data/
├── datasets/
│   ├── general/            # Animation datasets by project
│   │   ├── simpsons/
│   │   ├── luca/
│   │   └── ...
│   └── pixar/              # Alternative organization
│
├── training/
│   ├── lora/
│   │   ├── 2d_characters/  # 2D character LoRA training data
│   │   ├── 3d_characters/  # 3D character LoRA training data
│   │   └── evaluation/     # Evaluation outputs
│   ├── runs/               # Training run artifacts
│   └── logs/               # Training logs
│
├── extracted/
│   ├── frames/             # Extracted video frames
│   └── sam_masks/          # Segmentation masks
│
└── videos/                 # Source video files
```

---

## Environment Variables

Set these environment variables to configure cache locations:

```bash
# Add to ~/.bashrc or ~/.zshrc
export HF_HOME=/mnt/c/ai_cache/huggingface
export TRANSFORMERS_CACHE=/mnt/c/ai_cache/huggingface
export TORCH_HOME=/mnt/c/ai_cache/torch
export XDG_CACHE_HOME=/mnt/c/ai_cache
export PIP_CACHE_DIR=/mnt/c/ai_cache/pip
```

Or use the pipeline's built-in setup:

```python
from anime_pipeline.config import setup_warehouse_environment
setup_warehouse_environment()
```

---

## Configuration Files

### `configs/global/warehouse.yaml`

Central configuration for all paths:

```yaml
version: "3.0"

roots:
  models: "/mnt/c/ai_models"
  projects: "/mnt/c/ai_projects"
  cache: "/mnt/c/ai_cache"
  datasets: "/mnt/data/datasets"
  training: "/mnt/data/training"

animation_2d:
  datasets: "${roots.datasets}/general"
  training_data: "${roots.training}/lora/2d_characters"
  lora_outputs: "${roots.models}/lora"
  lora_sdxl_outputs: "${roots.models}/lora_sdxl"
```

### `configs/global/models.yaml`

Centralized model registry:

```yaml
version: "1.0"
model_root: "/mnt/c/ai_models"

segmentation:
  sam2_hiera_large:
    path: "${model_root}/segmentation/sam2_hiera_large.pt"
  toonout:
    path: "${model_root}/segmentation/toonout.onnx"

detection:
  yolov11n:
    path: "${model_root}/detection/yolov11n.pt"
```

---

## Deprecated Paths (DO NOT USE)

These paths are from older versions and should NOT be used:

| Deprecated Path | New Path |
|-----------------|----------|
| `/mnt/data/ai_data/` | `/mnt/data/datasets/` or `/mnt/data/training/` |
| `/mnt/data/ai_data/datasets/` | `/mnt/data/datasets/general/` |
| `/mnt/data/ai_data/training_data/` | `/mnt/data/training/lora/` |
| `/mnt/data/ai_data/models/` | `/mnt/c/ai_models/` |
| `~/ai_data/` | Use absolute paths |
| `$HOME/datasets/` | `/mnt/data/datasets/` |

---

## Migration Steps

### Step 1: Validate Current Configuration

```bash
# Check for deprecated paths in configs
python scripts/utils/validate_warehouse_config.py --verbose

# Scan codebase for deprecated paths
python scripts/utils/scan_deprecated_paths.py
```

### Step 2: Update Configuration Files

Replace deprecated paths in your configs:

```yaml
# OLD (deprecated)
paths:
  base_dir: /mnt/data/ai_data/datasets/3d-anime/luca

# NEW (AI_WAREHOUSE 3.0)
paths:
  base_dir: /mnt/data/datasets/general/luca
```

### Step 3: Migrate Existing Data (if needed)

```bash
# Move datasets (example)
mv /mnt/data/ai_data/datasets/3d-anime/* /mnt/data/datasets/general/

# Move models to SSD
mv /mnt/data/ai_data/models/* /mnt/c/ai_models/

# Update symlinks if any
ln -sf /mnt/data/datasets/general/luca /path/to/old/location
```

### Step 4: Validate Dataset Structure

```bash
# Check all project directories
python scripts/utils/validate_dataset_structure.py

# Check specific project
python scripts/utils/validate_dataset_structure.py --project luca
```

### Step 5: Generate Project Configs

```bash
# Auto-generate configs for all detected projects
python scripts/utils/generate_project_configs.py --verbose

# Generate for specific project
python scripts/utils/generate_project_configs.py --project simpsons --force
```

### Step 6: Test Configuration Loading

```bash
# Parallel dry-run test
python scripts/utils/parallel_dry_run.py --verbose
```

---

## Using the Config API

### Loading Warehouse Configuration

```python
from anime_pipeline.config import (
    load_warehouse_config,
    get_warehouse_path,
    get_model_path,
    setup_warehouse_environment
)

# Setup environment variables
setup_warehouse_environment()

# Load full warehouse config
warehouse = load_warehouse_config()
print(warehouse.roots.models)  # /mnt/c/ai_models

# Get specific paths
datasets_path = get_warehouse_path("datasets", "2d")
print(datasets_path)  # /mnt/data/datasets/general

# Get model path
sam2_path = get_model_path("segmentation", "sam2_hiera_large")
print(sam2_path)  # /mnt/c/ai_models/segmentation/sam2_hiera_large.pt
```

### Using in Project Configs

```yaml
# Use variable interpolation
project:
  name: my_project

paths:
  # These will resolve correctly
  base_dir: "${animation_2d.datasets}/${project.name}"
  training_data: "${animation_2d.training_data}/${project.name}"
  lora_output: "${animation_2d.lora_sdxl_outputs}/${project.name}"
```

---

## Validation Tools

### `validate_warehouse_config.py`

Validates all YAML configs against AI_WAREHOUSE 3.0:

```bash
python scripts/utils/validate_warehouse_config.py
python scripts/utils/validate_warehouse_config.py --verbose
python scripts/utils/validate_warehouse_config.py --output report.json
```

### `scan_deprecated_paths.py`

Scans entire codebase for deprecated paths:

```bash
python scripts/utils/scan_deprecated_paths.py
python scripts/utils/scan_deprecated_paths.py --include-docs
python scripts/utils/scan_deprecated_paths.py --output findings.json
```

### `validate_dataset_structure.py`

Validates dataset directory structures:

```bash
python scripts/utils/validate_dataset_structure.py
python scripts/utils/validate_dataset_structure.py --project luca
python scripts/utils/validate_dataset_structure.py --output structure.json
```

### `parallel_dry_run.py`

Tests configuration loading in parallel:

```bash
python scripts/utils/parallel_dry_run.py
python scripts/utils/parallel_dry_run.py --filter simpsons
python scripts/utils/parallel_dry_run.py --output test_results.json
```

### `generate_project_configs.py`

Auto-generates project configs from datasets:

```bash
python scripts/utils/generate_project_configs.py
python scripts/utils/generate_project_configs.py --project simpsons
python scripts/utils/generate_project_configs.py --force
```

---

## Common Issues

### Issue: "Deprecated path detected"

```
WARNING: Deprecated path in 'paths.base_dir': /mnt/data/ai_data/datasets/...
```

**Fix:** Update the path to use AI_WAREHOUSE 3.0 structure:

```yaml
# Change from
paths:
  base_dir: /mnt/data/ai_data/datasets/3d-anime/project

# To
paths:
  base_dir: /mnt/data/datasets/general/project
```

### Issue: "Model path not found"

```
ERROR: Model not found: /old/path/to/model.pt
```

**Fix:** Use centralized model paths from `models.yaml`:

```python
from anime_pipeline.config import get_model_path
model_path = get_model_path("segmentation", "sam2_hiera_large")
```

### Issue: "Environment variable not set"

```
WARNING: HF_HOME not set, using default cache
```

**Fix:** Call `setup_warehouse_environment()` at startup:

```python
from anime_pipeline.config import setup_warehouse_environment
setup_warehouse_environment()
```

---

## Best Practices

1. **Always use config variables** instead of hardcoded paths
2. **Run validation tools** before major pipeline runs
3. **Store models on SSD** (`/mnt/c/ai_models`) for faster loading
4. **Store datasets on HDD** (`/mnt/data/datasets`) for capacity
5. **Use `setup_warehouse_environment()`** at script startup
6. **Keep configs in version control** but not model weights

---

## Quick Reference

| What | Where |
|------|-------|
| Model weights | `/mnt/c/ai_models/` |
| Source datasets | `/mnt/data/datasets/general/` |
| Training data | `/mnt/data/training/lora/` |
| LoRA outputs | `/mnt/c/ai_models/diffusion/lora/sdxl/` |
| Cache | `/mnt/c/ai_cache/` |
| Logs | `/mnt/data/training/logs/` |

---

**AI_WAREHOUSE 3.0 - Organized storage for efficient ML workflows**
