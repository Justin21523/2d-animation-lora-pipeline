# Configuration Usage Guide

Guide for using the unified configuration system in scripts.

## Overview

The pipeline uses a hierarchical configuration system based on OmegaConf, with configurations organized in `configs/` directory:

```
configs/
├── global/          # Pipeline-wide settings
│   ├── pipeline.yaml
│   └── models.yaml
├── stages/          # Stage-specific configs
│   ├── segmentation/
│   ├── clustering/
│   ├── inpainting/
│   └── enhancement/
├── characters/      # Per-character definitions
│   └── luca.yaml
├── projects/        # Per-project configs
│   └── luca.yaml
└── training/        # Training configs
    └── *.toml
```

## Basic Usage

### 1. Import Configuration Loader

```python
from scripts.core.utils.config_loader import load_config, get_config
```

### 2. Load Single Configuration

```python
# Load global pipeline config
config = load_config("pipeline", config_type="global")

# Load stage config
seg_config = load_config("segmentation", config_type="stage")

# Load character config
char_config = load_config("luca", config_type="character")

# Load project config
project_config = load_config("luca", config_type="project")
```

### 3. Load Merged Configuration

For scripts that need combined settings from multiple sources:

```python
# Merge global + character + project configs
config = get_config(project="luca", character="luca")

# Merge global + stage + project configs
config = get_config(project="luca", stage="segmentation")

# Full merge: global + stage + character + project
config = get_config(project="luca", character="luca", stage="segmentation")
```

## Configuration Access

### Accessing Values

```python
# OmegaConf uses dot notation
device = config.hardware.gpu.device  # "cuda:0"
batch_size = config.processing.batch_size  # 8

# Access with get() for optional values
threshold = config.get('segmentation', {}).get('alpha_threshold', 0.15)

# Convert to dict if needed
config_dict = OmegaConf.to_container(config, resolve=True)
```

### Variable Resolution

Configs support variable interpolation:

```yaml
# In config file
paths:
  base_dir: "/mnt/data/ai_data/datasets/3d-anime/luca"
  frames: "${paths.base_dir}/frames"  # Resolves to full path
  instances: "${paths.base_dir}/instances"
```

## Integration Patterns

### Pattern 1: Simple Config Loading

For scripts that use a single configuration source:

```python
import argparse
from scripts.core.utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='pipeline', help='Config name')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, config_type="global")

    # Use config
    device = config.hardware.gpu.device
    batch_size = config.processing.batch_size

    # Run processing...
```

### Pattern 2: Merged Config with Override

For scripts that need merged configs but allow CLI overrides:

```python
import argparse
from scripts.core.utils.config_loader import get_config
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--character', default=None)
    parser.add_argument('--device', default=None, help='Override device')
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    # Load merged config
    config = get_config(project=args.project, character=args.character)

    # Apply CLI overrides
    if args.device:
        config.hardware.gpu.device = args.device
    if args.batch_size:
        config.processing.batch_size = args.batch_size

    # Use config...
```

### Pattern 3: Config + Explicit Paths

For scripts that still accept explicit paths but use config as fallback:

```python
import argparse
from pathlib import Path
from scripts.core.utils.config_loader import get_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--project', default='luca')
    parser.add_argument('--use-config', action='store_true',
                       help='Use paths from config instead of args')
    args = parser.parse_args()

    # Load config
    config = get_config(project=args.project)

    # Determine paths (CLI args take priority unless --use-config)
    if args.use_config or not args.input_dir:
        input_dir = Path(config.paths.frames)
    else:
        input_dir = Path(args.input_dir)

    if args.use_config or not args.output_dir:
        output_dir = Path(config.paths.instances)
    else:
        output_dir = Path(args.output_dir)

    # Processing...
```

### Pattern 4: Class-Based Configuration

For scripts with class-based architecture:

```python
from pathlib import Path
from scripts.core.utils.config_loader import get_config
from omegaconf import DictConfig

class DataPreparer:
    def __init__(self, project: str, character: str = None, config_overrides: dict = None):
        """
        Initialize with project configuration.

        Args:
            project: Project name (e.g., 'luca')
            character: Character name for character-specific config
            config_overrides: Dictionary of config overrides
        """
        # Load merged config
        self.config = get_config(project=project, character=character)

        # Apply overrides
        if config_overrides:
            self.config = OmegaConf.merge(self.config, config_overrides)

        # Extract commonly used values
        self.device = self.config.hardware.gpu.device
        self.batch_size = self.config.processing.batch_size
        self.output_dir = Path(self.config.paths.training_data)

    def prepare_dataset(self):
        # Use self.config throughout...
        alpha_threshold = self.config.quality.alpha_threshold
        # Processing...
```

## Adding Config Arguments to Existing Scripts

### Step 1: Add Config Import

```python
# Add to imports
from scripts.core.utils.config_loader import get_config
```

### Step 2: Add Config Arguments

```python
parser.add_argument('--project', type=str, default=None,
                   help='Project name (e.g., luca) - loads project config')
parser.add_argument('--character', type=str, default=None,
                   help='Character name - loads character config')
parser.add_argument('--use-config-paths', action='store_true',
                   help='Use paths from config instead of CLI args')
```

### Step 3: Load and Merge Config

```python
# After parsing args
config = None
if args.project:
    config = get_config(project=args.project, character=args.character)
```

### Step 4: Use Config with Fallbacks

```python
# Use config value if available, otherwise use CLI arg or default
if config and args.use_config_paths:
    input_dir = Path(config.paths.frames)
else:
    input_dir = Path(args.input_dir) if args.input_dir else Path("./data/frames")

# For non-path values, use config as default
device = config.hardware.gpu.device if config else args.device
batch_size = config.processing.batch_size if config else args.batch_size
```

## Configuration Best Practices

### 1. Keep CLI Compatibility

Don't break existing CLI interfaces. Add config support while maintaining backwards compatibility:

```python
# Good: Config is optional, CLI args still work
parser.add_argument('--device', default='cuda', help='Device (overrides config)')
parser.add_argument('--project', default=None, help='Project config (optional)')

# Bad: Force users to use config
parser.add_argument('--project', required=True)  # Breaks existing scripts
```

### 2. Clear Priority Order

Document and implement clear precedence:
1. Explicit CLI arguments (highest priority)
2. Project/character config
3. Global config
4. Script defaults (lowest priority)

### 3. Validate Critical Values

```python
# Validate required config sections exist
if config:
    try:
        _ = config.paths.base_dir
    except (AttributeError, KeyError):
        logger.warning("Config missing paths.base_dir, using defaults")
```

### 4. Log Config Source

```python
if config:
    logger.info(f"Loaded config from project: {args.project}")
    if args.device and args.device != config.hardware.gpu.device:
        logger.info(f"Overriding device: {config.hardware.gpu.device} -> {args.device}")
```

## Testing Configuration

### Test Config Loading

```bash
# Test single config
python scripts/core/utils/config_loader.py --name pipeline --type global

# Test merged config
python scripts/core/utils/config_loader.py --project luca --character luca

# Save merged config for inspection
python scripts/core/utils/config_loader.py --project luca --output test_config.yaml
```

### Test Script with Config

```bash
# Run with project config
python scripts/your_script.py --project luca --use-config-paths

# Run with config but override specific values
python scripts/your_script.py --project luca --device cuda:1 --batch-size 16

# Run without config (old CLI style)
python scripts/your_script.py --input-dir ./data --output-dir ./output
```

## Migration Checklist

When migrating a script to use unified config:

- [ ] Import config_loader
- [ ] Add --project and optional --character arguments
- [ ] Load config conditionally (if --project provided)
- [ ] Apply config values with CLI override logic
- [ ] Test both config mode and legacy CLI mode
- [ ] Update script documentation
- [ ] Add config usage example to --help text

## Common Patterns

### Access Nested Config

```python
# Safe access with fallback
alpha = config.get('quality', {}).get('alpha_threshold', 0.15)

# Direct access (may raise AttributeError)
alpha = config.quality.alpha_threshold
```

### Merge CLI Overrides

```python
from omegaconf import OmegaConf

# Create override dict from CLI args
overrides = {}
if args.device:
    overrides['hardware'] = {'gpu': {'device': args.device}}
if args.batch_size:
    overrides['processing'] = {'batch_size': args.batch_size}

# Merge overrides into config
if overrides:
    config = OmegaConf.merge(config, overrides)
```

### Convert Config Sections

```python
# Extract specific section as dict
paths_dict = OmegaConf.to_container(config.paths, resolve=True)

# Use with existing code that expects dict
process_data(**paths_dict)
```

## Troubleshooting

### Variable Not Resolved

**Problem:** `InterpolationKeyError: Interpolation key 'base_dir' not found`

**Solution:** Use full path in variable references:
```yaml
# Wrong
paths:
  base_dir: "/path/to/data"
  frames: "${base_dir}/frames"  # base_dir not found

# Correct
paths:
  base_dir: "/path/to/data"
  frames: "${paths.base_dir}/frames"  # Full path reference
```

### HuggingFace Model IDs Converted to Paths

**Problem:** `ViT-L/14` becomes `/path/to/repo/ViT-L/14`

**Solution:** This is fixed in config_loader.py. Model IDs without file extensions are preserved.

### Config Not Found

**Problem:** `FileNotFoundError: Configuration file not found`

**Solution:** Check config exists in correct directory:
```bash
ls configs/projects/luca.yaml
ls configs/characters/luca.yaml
```

## Examples

See migrated scripts for full examples:
- `scripts/generic/segmentation/instance_segmentation.py` - Merged config with CLI overrides
- `scripts/generic/training/prepare_pose_lora_data.py` - Class-based config usage
- `scripts/generic/clustering/character_clustering.py` - Config + explicit paths pattern

## Summary

The unified config system provides:
- ✅ Centralized configuration management
- ✅ Hierarchical config merging
- ✅ Variable interpolation for DRY configs
- ✅ Backwards compatibility with CLI args
- ✅ Easy project/character switching
- ✅ Reduced duplication across scripts

Use it to simplify script maintenance and ensure consistency across the pipeline.
