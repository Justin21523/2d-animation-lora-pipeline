# V1 to V2 Migration Guide

> **Upgrading from the old 2D pipeline to the new unified architecture**

This guide helps you migrate from the original 2D animation pipeline to the new architecture that integrates proven patterns from the 3D pipeline.

---

## 🎯 What's New in V2?

### Major Improvements

1. **Unified Configuration System**
   - ❌ Old: Dataclass-based configs scattered across modules
   - ✅ New: OmegaConf hierarchical configuration with automatic merging

2. **Stage-Based Architecture**
   - ❌ Old: Manual script execution, no dependency management
   - ✅ New: Automatic dependency resolution, checkpoint/resume support

3. **Multi-Character Support**
   - ❌ Old: Single character per frame assumption
   - ✅ New: Proper multi-character handling with identity tracking

4. **Code Quality**
   - ❌ Old: Duplicated stub patterns (~800 lines)
   - ✅ New: Unified `StubMode` and `MetadataIO` utilities

5. **Python CLI**
   - ❌ Old: Bash scripts for orchestration
   - ✅ New: Comprehensive Python CLI with argparse

---

## 📋 Migration Checklist

### Phase 1: Backup Your Data

```bash
# Backup existing project data
cp -r /path/to/old_pipeline_data /path/to/backup

# Backup existing configs
cp -r /path/to/old_configs /path/to/backup/configs
```

### Phase 2: Install V2

```bash
# Pull latest changes
cd /mnt/c/ai_projects/2d-animation-lora-pipeline
git pull origin main

# Update dependencies
conda activate ai_env
pip install -r requirements/all.txt
```

### Phase 3: Migrate Configurations

#### Old Format (Dataclass)

```python
# Old: anime_pipeline/config/loader.py
from dataclasses import dataclass

@dataclass
class FrameExtractionConfig:
    input_path: str
    output_dir: str
    mode: str = "scene"
    fps: int = None
```

#### New Format (YAML)

```yaml
# New: configs/projects/my_project.yaml
project: my_project
animation_mode: 2d

paths:
  input_video: /path/to/video.mp4
  frames: /path/to/output/frames

frame_extraction:
  mode: scene
  scene_threshold: 0.3
  fps: null
  quality: high
```

**Migration Script:**

```python
# convert_config.py
from pathlib import Path
import yaml

# Read old config
old_config = {
    "input_path": "/path/to/video.mp4",
    "output_dir": "/path/to/frames",
    "mode": "scene",
    "fps": None
}

# Convert to new format
new_config = {
    "project": "my_project",
    "animation_mode": "2d",
    "paths": {
        "input_video": old_config["input_path"],
        "frames": old_config["output_dir"]
    },
    "frame_extraction": {
        "mode": old_config["mode"],
        "fps": old_config["fps"]
    }
}

# Save as YAML
with open("configs/projects/my_project.yaml", "w") as f:
    yaml.dump(new_config, f, default_flow_style=False)
```

### Phase 4: Update Script Calls

#### Old Way (Bash Scripts)

```bash
# Old: Multiple manual script calls
bash bash/run_extract_frames.sh
bash bash/run_yolo_detection.sh
bash bash/run_toonout_segmentation.sh
bash bash/run_clustering.sh
```

#### New Way (Unified CLI)

```bash
# New: Single command with automatic dependency management
python scripts/run_pipeline.py \
    --project my_project \
    --mode 2d
```

**Or run specific stages:**

```bash
python scripts/run_pipeline.py \
    --project my_project \
    --stages frame_extraction,multi_character_extraction,dataset_building
```

### Phase 5: Update Custom Code

If you've built custom modules on top of the old pipeline:

#### Update Imports

```python
# Old imports
from anime_pipeline.config.loader import load_config
from anime_pipeline.detection.yolo_detector import run_yolo_tracking

# New imports
from anime_pipeline.config import load_config  # Now uses OmegaConf
from anime_pipeline.detection.yolo_detector import run_yolo_tracking_with_grouping
from anime_pipeline.core import StubMode, MetadataIO
```

#### Use New Utilities

```python
# Old: Manual stub handling
def my_function(config, logger):
    if config.use_stub:
        return generate_stub()
    try:
        model = load_model(config.model_path)
        return run_inference(model)
    except Exception as e:
        logger.warning(f"Failed: {e}, using stub")
        return generate_stub()

# New: Use StubMode
from anime_pipeline.core import StubMode, StubConfig

def my_function(config, logger):
    return StubMode.run_with_fallback(
        model_loader=load_model,
        real_inference=run_inference,
        stub_inference=generate_stub,
        config=StubConfig(**config),
        logger=logger
    )
```

#### Use MetadataIO

```python
# Old: Manual parquet/CSV handling
def load_detections(path, logger):
    if not path.exists():
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
    if not path.exists():
        return []
    import pandas as pd
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df.to_dict(orient="records")

# New: Use MetadataIO
from anime_pipeline.core import MetadataIO

def load_detections(path, logger):
    return MetadataIO.load_records(path, logger)
```

---

## 🔄 Feature Mapping

### Configuration System

| V1 Feature | V2 Equivalent | Notes |
|------------|---------------|-------|
| Dataclass configs | OmegaConf YAML | Hierarchical, supports variable interpolation |
| Manual path management | Automatic path resolution | Use `${paths.base_dir}/subdir` |
| Hard-coded defaults | Global + stage + project configs | Multi-level override system |

### Pipeline Execution

| V1 Feature | V2 Equivalent | Notes |
|------------|---------------|-------|
| Bash orchestration | `run_pipeline.py` | Dependency management, checkpoint/resume |
| Manual stage order | Automatic topological sort | Stages run in correct order |
| No progress tracking | Built-in progress reporting | JSON checkpoints, resource monitoring |

### Multi-Character Handling

| V1 Feature | V2 Equivalent | Notes |
|------------|---------------|-------|
| Single character assumption | `multi_character_extraction` stage | Proper track-level processing |
| CLIP clustering | Face-based identity clustering | Correctly separates different characters |
| No track merging | Automatic track merging | Same character across cuts |

---

## 🚨 Breaking Changes

### 1. Configuration Format

**Impact:** All configs must be converted to YAML

**Migration:**
- Create YAML configs in `configs/projects/`
- Use provided conversion script
- Test with `--dry-run` flag

### 2. CLI Interface

**Impact:** Bash scripts are deprecated

**Migration:**
- Replace all bash calls with `run_pipeline.py`
- See command mapping table below

### 3. Module Imports

**Impact:** Some import paths have changed

**Migration:**
- Update imports as shown in examples above
- Use unified utilities from `anime_pipeline.core`

---

## 📊 Command Mapping

| Old Bash Script | New Python CLI |
|----------------|----------------|
| `bash/run_extract_frames.sh` | `python scripts/run_pipeline.py --stages frame_extraction` |
| `bash/run_yolo_detection.sh` | `python scripts/run_pipeline.py --stages multi_character_extraction` |
| `bash/run_toonout_segmentation.sh` | (Integrated into `multi_character_extraction`) |
| `bash/run_clustering.sh` | (Integrated into `multi_character_extraction`) |
| `bash/run_full_pipeline.sh` | `python scripts/run_pipeline.py` |

---

## 🧪 Testing Your Migration

### Step 1: Dry Run

```bash
# Test configuration loading without execution
python scripts/run_pipeline.py \
    --project my_project \
    --dry-run
```

### Step 2: Single Stage Test

```bash
# Test frame extraction only
python scripts/run_pipeline.py \
    --project my_project \
    --stages frame_extraction
```

### Step 3: Full Pipeline Test

```bash
# Run full pipeline on small sample
python scripts/run_pipeline.py \
    --project my_project_sample \
    --mode 2d
```

### Step 4: Compare Outputs

```python
# Compare V1 vs V2 outputs
import pandas as pd

v1_detections = pd.read_parquet("v1_output/detections.parquet")
v2_detections = pd.read_parquet("v2_output/metadata/detections.parquet")

print(f"V1 detections: {len(v1_detections)}")
print(f"V2 detections: {len(v2_detections)}")
print(f"Difference: {abs(len(v1_detections) - len(v2_detections))}")
```

---

## 💡 Best Practices

### 1. Gradual Migration

Don't migrate everything at once. Start with:
1. Configuration files
2. Frame extraction
3. Single character project
4. Multi-character project
5. Custom modules

### 2. Use Stub Mode

Test the pipeline without model weights first:

```yaml
# Test config with stub mode
segmentation:
  use_stub: true
  backend: stub

clustering:
  use_stub: true
```

### 3. Leverage 2D/3D Conversion

If migrating from 3D parameters:

```bash
# Automatically convert to 2D parameters
python scripts/run_pipeline.py \
    --project my_3d_project \
    --mode 2d  # Converts 3D params to 2D
```

### 4. Monitor Resource Usage

```python
from anime_pipeline.core import ResourceMonitor

monitor = ResourceMonitor(device="cuda")
stats = monitor.get_current_stats()

print(f"GPU Memory: {stats.gpu_memory_used_mb} MB")
print(f"CPU Usage: {stats.cpu_percent}%")
```

---

## 🆘 Troubleshooting

### Issue: "Config not found"

```
ERROR: Config file not found: configs/projects/my_project.yaml
```

**Solution:**
1. Check config file exists
2. Use absolute path or ensure working directory is correct
3. Try `--dry-run` to see config search paths

### Issue: "Stage 'xyz' not found"

```
ERROR: Unknown stage: xyz
```

**Solution:**
1. Check available stages: `python scripts/run_pipeline.py --help`
2. Use correct stage names (see STAGE_REGISTRY in `stages.py`)
3. Old stage names may have changed (see command mapping table)

### Issue: "Circular dependency detected"

```
ERROR: Circular dependency detected in pipeline stages
```

**Solution:**
1. Check `dependencies` in stage configs
2. Ensure no circular references (A depends on B, B depends on A)
3. Review `configs/global/pipeline_stages.yaml`

### Issue: "Model weights not found"

```
WARNING: Model path does not exist, using stub mode
```

**Solution:**
1. This is expected if model weights aren't available
2. Set `use_stub: true` in config to acknowledge
3. Or provide correct `model_path` in config

---

## 📞 Getting Help

If you encounter issues during migration:

1. **Check Documentation**: See [full documentation](../README.md)
2. **GitHub Issues**: Report bugs or ask questions
3. **Dry-Run Mode**: Always test with `--dry-run` first
4. **Stub Mode**: Test logic without model weights

---

## ✅ Migration Verification

After migration, verify:

- [ ] All configs converted to YAML format
- [ ] Pipeline runs with `--dry-run`
- [ ] Single stage execution works
- [ ] Full pipeline completes successfully
- [ ] Output formats match expectations
- [ ] Resource usage is acceptable
- [ ] Custom code updated to use new APIs

---

**Migration complete! Welcome to V2! 🎉**
