# 2D vs 3D Animation Parameters Guide

> **Understanding parameter differences and automatic conversion**

This guide explains why 2D and 3D animation pipelines need different parameters and how the automatic conversion system works.

---

## 🎨 Fundamental Differences

### 3D Animation Characteristics

**Examples**: Pixar (Toy Story, Luca), DreamWorks (Shrek), Disney 3D (Frozen)

- ✅ **Consistent 3D models**: Same character model rendered in all scenes
- ✅ **Smooth anti-aliased edges**: Soft transitions at boundaries
- ✅ **PBR materials**: Physically-based rendering with consistent colors
- ✅ **Cinematic DoF**: Intentional blur effects
- ✅ **Global illumination**: Realistic lighting across frames

### 2D Animation Characteristics

**Examples**: The Simpsons, Family Guy, Rick and Morty, Adventure Time

- ⚠️ **Hand-drawn variation**: Different animators, episodes have style drift
- ⚠️ **Hard line-art edges**: Sharp boundaries, no anti-aliasing
- ⚠️ **Cel shading**: Flat colors, hard shadows
- ⚠️ **Sharp focus**: Usually maintains clarity across frame
- ⚠️ **Stylized faces**: May fail realistic face detectors

---

## 📊 Complete Parameter Comparison

### Segmentation Parameters

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `alpha_threshold` | **0.15** | **0.25** | 2D has hard edges, need higher threshold to capture line-art |
| `blur_threshold` | **80** | **100** | 2D maintains sharper focus, stricter blur rejection |
| `edge_detection_mode` | `soft_gradient` | `hard_edge` | 2D uses distinct black outlines |
| `min_instance_size` | **128px** | **128px** | Same (both reject tiny instances) |

**Example Config:**

```yaml
# 3D config
segmentation:
  alpha_threshold: 0.15  # Soft anti-aliased edges
  blur_threshold: 80     # Tolerate cinematic DoF

# 2D config
segmentation:
  alpha_threshold: 0.25  # Sharp line-art boundaries
  blur_threshold: 100    # Stricter focus requirement
```

### Clustering Parameters

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `min_cluster_size` | **10-15** | **20-25** | 2D varies more across episodes/animators |
| `min_samples` | **2** | **3-5** | More samples needed for style variation |
| `similarity_threshold` | **0.70** | **0.75** | Stricter threshold to separate similar characters |
| `use_face_detection` | **true** | **false** ⚠️ | Stylized 2D faces fail realistic detectors |
| `distance_metric` | `euclidean` | `euclidean` | Same |

**Example Config:**

```yaml
# 3D config
clustering:
  min_cluster_size: 10   # 3D models are consistent
  min_samples: 2
  similarity_threshold: 0.70

# 2D config
clustering:
  min_cluster_size: 20   # Account for style drift
  min_samples: 3
  similarity_threshold: 0.75  # Stricter separation
```

### Dataset Parameters

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `target_size` | **200-500** | **500-1000** | 2D needs more examples for style coverage |
| `dedup_threshold` | **0.95** | **0.92** | More aggressive dedup for TV shows |
| `quality_min_score` | **0.7** | **0.6** | 2D has more variation, lower threshold |

**Example Config:**

```yaml
# 3D config
dataset:
  target_size: 300       # Fewer examples needed
  dedup_threshold: 0.95  # Keep similar frames

# 2D config
dataset:
  target_size: 800       # More examples for variation
  dedup_threshold: 0.92  # Remove duplicates aggressively
```

### Augmentation Parameters

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `color_jitter` | **false** ❌ | **true** ✅ | 2D has color variation across episodes |
| `horizontal_flip` | **false** ❌ | **false** ❌ | Both avoid asymmetric features |
| `rotation_range` | **0°** | **5°** | Slight rotation for 2D variance |
| `brightness_range` | **0** | **0.1** | 2D lighting varies across scenes |

**Example Config:**

```yaml
# 3D config
augmentation:
  color_jitter: false  # Breaks PBR materials
  horizontal_flip: false
  rotation_range: 0

# 2D config
augmentation:
  color_jitter: true   # Helps generalization across episodes
  horizontal_flip: false
  rotation_range: 5
  brightness_range: 0.1
```

### Training Parameters

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `learning_rate` | **1e-4** | **1e-4** | Same |
| `epochs` | **8-12** | **10-15** | 2D may need more epochs |
| `batch_size` | **4** | **2** | 2D images are typically higher res |
| `network_dim` | **32** | **32** | Same |
| `network_alpha` | **16** | **16** | Same |

**Example Config:**

```yaml
# Both similar, adjust based on results
training:
  learning_rate: 1e-4
  epochs: 10
  batch_size: 2
```

### Captioning Parameters

| Parameter | 3D Default | 2D Default | Reason |
|-----------|------------|------------|--------|
| `style_prefix` | `"3d animated, pixar style, smooth shading"` | `"2d animated, western animation, cel shading"` | Different art styles |
| `include_materials` | **true** | **false** | 3D has PBR materials, 2D has flat colors |
| `include_lighting` | **true** | **false** | 3D has complex lighting, 2D has simple shading |
| `max_length` | **77 tokens** | **77 tokens** | Same (CLIP limit) |

**Example Config:**

```yaml
# 3D config
captioning:
  prefix: "a 3d animated character, pixar style, smooth shading, studio lighting"
  include_materials: true
  include_lighting: true

# 2D config
captioning:
  prefix: "a 2d animated character, western animation, cel shading"
  include_materials: false
  include_lighting: false
```

---

## 🔄 Automatic Parameter Conversion

The pipeline can automatically convert parameters between 2D and 3D modes.

### How It Works

When you specify `animation_mode` in your config:

```yaml
# Project config
project: my_project
animation_mode: 2d  # or 3d

# Stage configs will be automatically adjusted
```

The `ParameterConverter` applies the appropriate defaults from `configs/global/param_mapping.yaml`.

### Conversion Examples

#### Example 1: 3D → 2D Conversion

```python
from anime_pipeline.config import ParameterConverter

converter = ParameterConverter()

# Start with 3D parameters
config_3d = {
    "segmentation": {
        "alpha_threshold": 0.15,
        "blur_threshold": 80
    },
    "clustering": {
        "min_cluster_size": 10
    }
}

# Convert to 2D
config_2d = converter.convert_config(
    config=config_3d,
    source_style="3d",
    target_style="2d"
)

print(config_2d)
# {
#     "segmentation": {
#         "alpha_threshold": 0.25,  # Scaled
#         "blur_threshold": 100
#     },
#     "clustering": {
#         "min_cluster_size": 20  # Scaled
#     }
# }
```

#### Example 2: Using CLI Flag

```bash
# Automatically use 2D parameters
python scripts/run_pipeline.py \
    --project my_project \
    --mode 2d  # Applies 2D defaults

# Or 3D parameters
python scripts/run_pipeline.py \
    --project my_project \
    --mode 3d  # Applies 3D defaults
```

### Parameter Scaling Rules

The converter uses two strategies:

1. **Numeric Scaling**: Proportional adjustment
   ```python
   # Example: min_cluster_size
   # 3D default: 10 → 2D default: 20 (2x multiplier)
   scale = 20 / 10  # 2.0
   converted_value = original_value * scale
   ```

2. **Direct Replacement**: For non-numeric values
   ```python
   # Example: edge_detection_mode
   # 3D: "soft_gradient" → 2D: "hard_edge" (direct swap)
   converted_value = target_default
   ```

---

## 🎯 Parameter Selection Guide

### When to Use 3D Parameters

Use `animation_mode: 3d` when processing:

- **Pixar films**: Toy Story, Finding Nemo, Luca, Turning Red
- **DreamWorks films**: Shrek, How to Train Your Dragon
- **Disney 3D**: Frozen, Moana, Encanto
- **Other 3D**: Illumination (Despicable Me), Sony (Spiderverse - though stylized)

**Characteristics:**
- Consistent character models
- Smooth surfaces
- Realistic lighting
- Cinematic effects

### When to Use 2D Parameters

Use `animation_mode: 2d` when processing:

- **TV animation**: The Simpsons, Family Guy, Rick and Morty
- **Adult animation**: South Park, Archer, BoJack Horseman
- **Cartoon Network**: Adventure Time, Regular Show
- **Anime**: (though may need anime-specific tuning)

**Characteristics:**
- Hand-drawn or vector art
- Hard line-art edges
- Flat cel shading
- Style variation across episodes

### Hybrid Cases

Some content is **stylized 3D** that looks like 2D:

- **Spider-Verse**: 3D with 2D-style rendering
- **Arcane**: 3D with painterly effects
- **Klaus**: 2D with 3D-like lighting

**Recommendation**: Start with `3d` mode, then adjust if needed.

---

## 🔬 Parameter Tuning Guide

### Step 1: Start with Defaults

```yaml
animation_mode: 2d  # or 3d
# No explicit parameters - use defaults
```

### Step 2: Run and Evaluate

```bash
python scripts/run_pipeline.py \
    --project test \
    --mode 2d \
    --stop-at clustering
```

### Step 3: Adjust Based on Results

**Problem**: Too many small clusters

```yaml
clustering:
  min_cluster_size: 30  # Increase from 20
  similarity_threshold: 0.70  # Lower from 0.75
```

**Problem**: Characters merged together

```yaml
clustering:
  min_cluster_size: 15  # Decrease from 20
  similarity_threshold: 0.80  # Increase from 0.75
```

**Problem**: Poor segmentation quality

```yaml
segmentation:
  alpha_threshold: 0.30  # Increase from 0.25
  blur_threshold: 90     # Decrease from 100
```

### Step 4: Document Your Overrides

```yaml
# configs/projects/my_project.yaml
project: my_project
animation_mode: 2d

# Custom overrides (documented)
segmentation:
  alpha_threshold: 0.28  # Adjusted for thick line-art in this show

clustering:
  min_cluster_size: 25   # Increased due to many similar episodes
```

---

## 📈 Performance Impact

### Parameter Tradeoffs

| Parameter Change | Speed Impact | Quality Impact | Memory Impact |
|------------------|--------------|----------------|---------------|
| Increase `min_cluster_size` | ✅ Faster | ⚠️ May miss characters | ➡️ Same |
| Decrease `alpha_threshold` | ➡️ Same | ⬇️ More noise | ➡️ Same |
| Increase `target_size` | ⬇️ Slower | ✅ Better LoRA | ⬆️ More disk |
| Enable `color_jitter` | ⬇️ Slower | ✅ Better generalization | ➡️ Same |

---

## 🧪 A/B Testing Parameters

### Testing Framework

```python
from anime_pipeline.core import PipelineOrchestrator

# Test A: 2D defaults
orchestrator_a = PipelineOrchestrator(
    project="test",
    animation_mode="2d"
)
result_a = orchestrator_a.run_full_pipeline()

# Test B: Custom parameters
config_b = load_config("test", overrides={
    "clustering": {"min_cluster_size": 30}
})
orchestrator_b = PipelineOrchestrator(
    project="test",
    config=config_b
)
result_b = orchestrator_b.run_full_pipeline()

# Compare results
print(f"A: {result_a['num_characters']} characters")
print(f"B: {result_b['num_characters']} characters")
```

---

## 📚 Reference

### Full Parameter Mapping

See `configs/global/param_mapping.yaml` for the complete parameter mapping table.

### Parameter Validation

The `ParameterConverter` includes validation rules:

```python
# Example validation rules
if mode == "2d":
    if alpha_threshold < 0.20:
        raise ValueError("alpha_threshold too low for 2D (use >= 0.25)")

    if min_cluster_size < 15:
        raise ValueError("min_cluster_size too small for 2D (use >= 20)")
```

---

## ✅ Best Practices

1. **Start with auto-detection**: Use `--mode 2d` or `--mode 3d`
2. **Test before full run**: Use `--dry-run` to check parameters
3. **Document overrides**: Add comments explaining custom values
4. **Iterate**: Run → Evaluate → Adjust → Repeat
5. **Share findings**: Contribute successful parameters back to the project

---

## 🆘 Common Issues

### Issue: "Invalid parameter for 2d mode"

```
ValueError: Invalid parameter for 2d mode: alpha_threshold too low for 2D
```

**Solution**: Increase the parameter to meet 2D requirements:

```yaml
segmentation:
  alpha_threshold: 0.25  # Minimum for 2D
```

### Issue: Parameters not applied

```
INFO: Using 3D defaults (parameter conversion not applied)
```

**Solution**: Explicitly set `animation_mode` in config:

```yaml
animation_mode: 2d  # At top level of project config
```

---

**Master the parameters for optimal 2D LoRA training! 🎨**
