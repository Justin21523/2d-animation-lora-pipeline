# Style LoRA Deep Dive: Technical Implementation Guide

**Target Audience:** ML engineers implementing style/aesthetic LoRA for 3D animation visual style
**Difficulty Level:** 🟡 **MEDIUM-HIGH** (Abstract style features + multi-domain application)
**Prerequisites:** Understanding of LoRA, color theory, computer graphics, rendering concepts

---

## Table of Contents

1. [Why Style LoRA is Unique](#1-why-style-lora-is-unique)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Style Feature Taxonomy](#3-style-feature-taxonomy)
4. [Data Preparation Pipeline](#4-data-preparation-pipeline)
5. [Implementation Details](#5-implementation-details)
6. [Training Strategy](#6-training-strategy)
7. [Testing and Evaluation](#7-testing-and-evaluation)
8. [Common Problems](#8-common-problems)

---

## 1. Why Style LoRA is Unique

### 1.1 The Core Problem

Style LoRA aims to capture the **overall visual aesthetic** of a film or scene collection, including:

- **Lighting style** (soft vs hard, naturalistic vs dramatic)
- **Color grading** (warm/cool palette, saturation, contrast)
- **Rendering style** (smooth PBR vs painterly, level of detail)
- **Material appearance** (how skin, cloth, metal look)
- **Atmospheric effects** (bloom, god rays, fog, depth of field)

**Key challenge:** Style is **abstract and pervasive** - it affects everything but is hard to isolate.

### 1.2 Style vs Other LoRA Types

| LoRA Type | Target | Scope | Application |
|-----------|--------|-------|-------------|
| **Character** | Identity (face, body) | Single entity | Characters only |
| **Expression** | Facial emotions | Face region | Faces only |
| **Pose** | Body configuration | Full body | Characters only |
| **Background** | Scene environment | Scene elements | Backgrounds only |
| **Style** | **Visual aesthetic** | **Everything** | **All domains** |

**Critical difference:** Style LoRA is **cross-domain** - applies to characters, backgrounds, props, everything.

### 1.3 3D Animation Style Characteristics

**Pixar-style 3D animation has distinctive style features:**

#### Lighting Style:
- **Cinematic three-point lighting** (key, fill, rim)
- **Soft shadows** (raytraced area lights)
- **Volumetric effects** (god rays, atmospheric scattering)
- **HDR lighting** (high dynamic range)
- **Color temperature** (warm key + cool rim common)

#### Material/Rendering Style:
- **Smooth PBR shading** (physically-based rendering)
- **Subsurface scattering** (skin translucency)
- **Specular highlights** (wet-look eyes, shiny surfaces)
- **Texture detail level** (painterly, not photorealistic)
- **Edge softness** (anti-aliasing, motion blur)

#### Color Grading:
- **Saturated colors** (more vibrant than real life)
- **Warm color palette** (often golden/orange tones)
- **High contrast** (deep shadows, bright highlights)
- **Color harmony** (carefully chosen palettes)
- **Film-like look** (subtle color shifts)

#### Atmospheric Style:
- **Depth of field** (camera-like bokeh)
- **Lens effects** (bloom, flares, chromatic aberration)
- **Atmospheric perspective** (haze on distant objects)
- **Water/underwater** (caustics, light rays)

**Example:** "Luca" has distinct style:
- Warm Mediterranean color palette (ochre, azure, coral)
- Soft, naturalistic lighting with occasional dramatic accents
- Smooth PBR materials with painterly texture detail
- Underwater scenes: strong volumetric light rays, cyan-green tint
- Overall: sunny, warm, inviting aesthetic

### 1.4 Why Style LoRA is Valuable

1. **Consistent aesthetics** across all generated content
2. **Film identity preservation**: Keep the "look" of the source material
3. **Cross-domain application**: One Style LoRA affects characters + backgrounds
4. **Art direction control**: Separate content (what) from style (how)
5. **Efficient workflow**: Apply film style to any base model output

---

## 2. Theoretical Foundation

### 2.1 What is Style LoRA?

Style LoRA learns **perceptual and rendering characteristics** in the latent space:

```
Base Model Latent → Style LoRA → Stylized Latent
```

**Key idea:** Style features should be **content-agnostic** - work on any subject (character, background, object).

### 2.2 Style Transfer vs Style LoRA

| Approach | Traditional Style Transfer | Style LoRA |
|----------|---------------------------|------------|
| **Method** | Per-image optimization | Learned adaptation |
| **Speed** | Slow (~30s per image) | Fast (inference time) |
| **Flexibility** | Arbitrary styles | Trained style only |
| **Quality** | Can be inconsistent | Stable, predictable |
| **Use case** | Artistic experiments | Production pipeline |

Style LoRA is like "baking" style transfer into the model.

### 2.3 Style LoRA Rank & Layers

**Rank recommendation:** 64-96 (higher than other LoRA types)

**Why higher rank?**
- Style affects **global** image properties (color, lighting)
- Style affects **local** details (textures, edges)
- Need capacity to learn both global and local features

**Layer targeting:**
- **UNet blocks:** All (style is pervasive)
- **Text encoder:** Yes (style keywords)
- **Cross-attention:** Critical (style-content disentanglement)

---

## 3. Style Feature Taxonomy

### 3.1 Lighting Style Features

#### Global Illumination:
- **Lighting direction:** Top-down (overhead), side-lit, backlit
- **Light hardness:** Soft (diffuse) vs hard (sharp shadows)
- **Light color:** Warm (golden) vs cool (blue), neutral
- **Intensity:** Bright (high-key) vs dark (low-key)
- **Contrast:** High (dramatic) vs low (flat)

#### Specific Lighting Setups:
- **Three-point** (key + fill + rim)
- **Rembrandt** (45° key with triangle of light on cheek)
- **Butterfly** (overhead key, glamour lighting)
- **Rim/Edge** (backlight separating subject from background)
- **Volumetric** (god rays, light shafts through fog/water)

### 3.2 Color Grading Features

#### Color Palette:
- **Dominant hues:** Warm (red/orange/yellow) vs cool (blue/cyan/green)
- **Saturation:** Vibrant vs muted vs monochrome
- **Brightness:** Light (high-key) vs dark (low-key)
- **Contrast:** High (punchy) vs low (soft)

#### Color Relationships:
- **Complementary** (opposite on color wheel, e.g., blue-orange)
- **Analogous** (adjacent, e.g., blue-cyan-green)
- **Triadic** (evenly spaced, e.g., red-yellow-blue)
- **Monochromatic** (single hue with variations)

#### Technical Color Grading:
- **Lift/Gamma/Gain** (shadows/midtones/highlights adjustment)
- **Color temperature shift** (warmer/cooler)
- **Tonal curves** (S-curve for contrast, custom curves)
- **Split toning** (different colors in shadows vs highlights)

### 3.3 Rendering Style Features

#### Shading Model:
- **PBR (Physically-Based):** Realistic material response to light
- **NPR (Non-Photorealistic):** Toon shading, cel shading, painterly
- **Hybrid:** Realistic base with stylized accents (common in 3D animation)

#### Material Properties:
- **Diffuse reflection:** Surface color, brightness
- **Specular reflection:** Shininess, highlight sharpness
- **Roughness/Glossiness:** Matte vs glossy
- **Subsurface scattering:** Skin translucency, wax, jade
- **Metalness:** Metallic vs dielectric materials

#### Texture Detail:
- **Level of detail:** Photorealistic vs simplified
- **Texture style:** Painterly, hand-painted, procedural
- **Normal map intensity:** Subtle vs pronounced surface detail
- **Displacement:** Geometric detail vs texture detail

### 3.4 Post-Processing & Camera Effects

#### Depth of Field:
- **Bokeh shape:** Circular, hexagonal, custom
- **Blur intensity:** Subtle vs strong
- **Focus distance:** Foreground, midground, background

#### Lens Effects:
- **Bloom:** Glow around bright areas
- **Lens flares:** Sun/light source flares
- **Chromatic aberration:** Color fringing at edges
- **Vignetting:** Darkening at image corners

#### Atmospheric Effects:
- **Fog/Haze:** Distance attenuation
- **Volumetric lighting:** Light scattering in atmosphere
- **Underwater effects:** Caustics, light rays, color shift

### 3.5 Edge & Detail Rendering

#### Edge Treatment:
- **Anti-aliasing:** Smooth vs pixelated edges
- **Edge softness:** Sharp vs soft
- **Rim lighting:** Edge highlights
- **Outlines:** Stylized black outlines (rare in Pixar, common in 2D)

#### Motion & Temporal:
- **Motion blur:** Camera motion, object motion
- **Frame rate simulation:** 24fps film look vs 60fps smooth
- **Temporal effects:** Film grain, noise patterns

---

## 4. Data Preparation Pipeline

### 4.1 Pipeline Overview

```
Source Frames (all scenes, characters, backgrounds)
  ↓
Style Feature Extraction
  ├─ Color Histogram & Statistics
  ├─ Lighting Analysis (brightness, contrast, color temp)
  ├─ Texture & Detail Analysis (edge density, frequency)
  └─ Global Style Embedding (CLIP/DINOv2)
  ↓
Style Consistency Filtering (reject outliers)
  ↓
Representative Sample Selection (diverse but consistent)
  ↓
Caption Generation (style-focused)
  ↓
Dataset Assembly (Kohya_ss format)
```

### 4.2 Data Selection Strategy

**Key principle:** Style LoRA needs **diverse content** with **consistent style**.

#### Diversity Requirements:
- **Multiple characters** (not just one)
- **Multiple scenes** (indoor, outdoor, underwater, etc.)
- **Multiple poses/actions** (standing, running, close-up, wide shot)
- **Multiple expressions** (happy, sad, neutral, etc.)

#### Consistency Requirements:
- **Same film** (don't mix multiple films)
- **Same lighting conditions** (or separate by lighting if needed)
- **Same rendering pipeline** (consistent material/shader settings)
- **Same color grading** (consistent LUT/color correction)

**Example Dataset for "Luca" Style LoRA:**
```
- 100 frames: Luca character in various poses/expressions
- 100 frames: Alberto character in various poses/expressions
- 50 frames: Giulia character
- 100 frames: Background scenes (plaza, streets, underwater)
- 50 frames: Close-ups (faces, hands, objects)
- 100 frames: Wide shots (establishing shots, landscapes)
```
**Total:** ~500 frames with high style consistency, high content diversity.

### 4.3 Style Feature Extraction

#### Method 1: Automated Style Analysis

```python
import cv2
import numpy as np
from pathlib import Path

def extract_style_features(image_path):
    """
    Extract quantitative style features from an image

    Returns:
        Dict with color, lighting, texture, and edge features
    """
    img = cv2.imread(str(image_path))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    features = {}

    # === Color Features ===
    # Dominant hue
    hue = img_hsv[:, :, 0]
    features['dominant_hue'] = np.median(hue)  # 0-180

    # Saturation statistics
    sat = img_hsv[:, :, 1]
    features['saturation_mean'] = np.mean(sat)
    features['saturation_std'] = np.std(sat)

    # Value (brightness) statistics
    val = img_hsv[:, :, 2]
    features['brightness_mean'] = np.mean(val)
    features['brightness_std'] = np.std(val)

    # Color temperature (approximation)
    # Warm = more red/yellow, Cool = more blue/cyan
    b, g, r = cv2.split(img)
    features['color_temperature'] = (np.mean(r) - np.mean(b)) / 255.0  # -1 to 1

    # === Lighting Features ===
    # Contrast (std of luminance)
    luminance = img_lab[:, :, 0]
    features['contrast'] = np.std(luminance)

    # Dynamic range
    features['dynamic_range'] = luminance.max() - luminance.min()

    # Histogram spread (how evenly distributed are brightness values)
    hist = cv2.calcHist([luminance], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    features['histogram_entropy'] = -np.sum(hist * np.log(hist + 1e-7))

    # === Texture & Detail Features ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    features['edge_density'] = edges.sum() / edges.size

    # Texture detail (Laplacian variance)
    features['texture_detail'] = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Frequency content (FFT analysis)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    # High-frequency content (sharp details)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 4

    mask_high_freq = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_high_freq, (center_x, center_y), radius, 0, -1)  # Mask low freq
    mask_high_freq = 255 - mask_high_freq

    high_freq_energy = np.sum(magnitude_spectrum * (mask_high_freq / 255.0))
    total_energy = np.sum(magnitude_spectrum)
    features['high_freq_ratio'] = high_freq_energy / (total_energy + 1e-7)

    # === Perceptual Sharpness ===
    # Blur score (inverse of Laplacian variance)
    features['perceived_sharpness'] = features['texture_detail']

    return features

# Usage
features = extract_style_features('frame_001.jpg')
print(f"Dominant hue: {features['dominant_hue']:.1f}°")
print(f"Color temperature: {features['color_temperature']:.2f}")
print(f"Contrast: {features['contrast']:.2f}")
print(f"Edge density: {features['edge_density']:.4f}")
```

---

#### Method 2: Deep Learning Style Embeddings

```python
import torch
from transformers import AutoModel, AutoProcessor

def extract_style_embedding_clip(image_path, model_name="openai/clip-vit-large-patch14"):
    """
    Extract CLIP visual embedding (captures high-level style)
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cuda')

    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to('cuda')

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()


def extract_style_embedding_dinov2(image_path):
    """
    Extract DINOv2 embedding (pure visual, no text bias)
    Better for style clustering
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model = AutoModel.from_pretrained("facebook/dinov2-large").to('cuda')

    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token
        features = outputs.last_hidden_state[:, 0, :]
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()
```

---

### 4.4 Style Consistency Filtering

**Goal:** Remove frames with inconsistent style (wrong lighting, color grading errors, etc.)

```python
def filter_style_consistency(frame_features, threshold=2.0):
    """
    Filter frames by style consistency using statistical outlier detection

    Args:
        frame_features: List of feature dicts from extract_style_features()
        threshold: Z-score threshold for outliers (2.0 = 95% confidence)

    Returns:
        filtered_frame_indices: Indices of consistent frames
    """
    import numpy as np
    from scipy import stats

    # Convert to numpy array
    feature_names = ['dominant_hue', 'saturation_mean', 'brightness_mean',
                     'color_temperature', 'contrast', 'edge_density']

    feature_matrix = np.array([
        [frame[feat] for feat in feature_names]
        for frame in frame_features
    ])

    # Compute Z-scores for each feature
    z_scores = np.abs(stats.zscore(feature_matrix, axis=0))

    # Mark as outlier if ANY feature is beyond threshold
    is_outlier = (z_scores > threshold).any(axis=1)

    # Keep non-outliers
    filtered_indices = np.where(~is_outlier)[0]

    print(f"✅ Style consistency filtering:")
    print(f"   Kept: {len(filtered_indices)} / {len(frame_features)}")
    print(f"   Removed: {is_outlier.sum()} outliers")

    return filtered_indices
```

---

### 4.5 Representative Sample Selection

**Strategy:** Maximize content diversity while maintaining style consistency.

```python
def select_representative_samples(
    frame_paths,
    frame_features,
    content_embeddings,  # DINOv2 or CLIP embeddings
    target_count=500,
    diversity_weight=0.7
):
    """
    Select representative samples that maximize content diversity

    Args:
        frame_paths: List of frame file paths
        frame_features: List of style feature dicts
        content_embeddings: (N, D) content embeddings (for diversity)
        target_count: Target number of samples
        diversity_weight: Weight for diversity vs random (0-1)

    Returns:
        selected_indices: Indices of selected frames
    """
    import numpy as np
    from sklearn.metrics import pairwise_distances

    n_frames = len(frame_paths)

    if n_frames <= target_count:
        return list(range(n_frames))

    # Compute pairwise distances (content diversity)
    distances = pairwise_distances(content_embeddings, metric='cosine')

    # Greedy selection: maximize minimum distance to selected set
    selected = []
    remaining = set(range(n_frames))

    # Start with a random frame
    first_idx = np.random.choice(list(remaining))
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Greedily select frames that are most different from selected set
    for _ in range(target_count - 1):
        if len(remaining) == 0:
            break

        # For each remaining frame, compute min distance to selected set
        min_distances = []
        for idx in remaining:
            dists_to_selected = [distances[idx, s] for s in selected]
            min_dist = min(dists_to_selected)
            min_distances.append((idx, min_dist))

        # Select frame with maximum min_distance (farthest from selected)
        if np.random.rand() < diversity_weight:
            # Diversity-based selection
            best_idx, _ = max(min_distances, key=lambda x: x[1])
        else:
            # Random selection
            best_idx, _ = min_distances[np.random.randint(len(min_distances))]

        selected.append(best_idx)
        remaining.remove(best_idx)

    print(f"✅ Selected {len(selected)} representative samples")

    return selected
```

---

### 4.6 Caption Generation

**Caption strategy for Style LoRA:**

Captions should **emphasize style keywords** while keeping content generic.

```python
def generate_style_caption(
    content_type='scene',  # 'character', 'scene', 'object', 'wide_shot'
    lighting_style='natural soft lighting',
    color_palette='warm mediterranean colors',
    rendering_style='smooth PBR shading',
    atmosphere='bright sunny atmosphere',
    film_name='pixar style'
):
    """
    Generate style-focused caption

    Args:
        content_type: Generic content description
        lighting_style: Lighting keywords
        color_palette: Color/grading keywords
        rendering_style: Material/shader keywords
        atmosphere: Mood/ambiance keywords
        film_name: Film/studio style reference

    Returns:
        Caption string
    """
    parts = []

    # Generic content (de-emphasized)
    content_templates = {
        'character': 'a 3d animated character',
        'scene': 'a 3d animated scene',
        'object': 'a 3d animated object',
        'wide_shot': 'a 3d animated wide shot scene',
        'close_up': 'a 3d animated close-up',
    }
    parts.append(content_templates[content_type])

    # Style keywords (emphasized)
    parts.append(lighting_style)
    parts.append(color_palette)
    parts.append(rendering_style)
    parts.append(atmosphere)

    # Film style reference
    parts.append(film_name)
    parts.extend([
        'detailed textures',
        'cinematic composition',
    ])

    caption = ", ".join(parts)
    return caption

# Examples for Luca Style LoRA
caption1 = generate_style_caption(
    content_type='scene',
    lighting_style='soft natural sunlight, warm key light',
    color_palette='warm mediterranean colors, ochre and azure tones',
    rendering_style='smooth PBR shading, painterly textures',
    atmosphere='bright sunny italian coastal atmosphere',
    film_name='pixar luca style'
)
# "a 3d animated scene, soft natural sunlight, warm key light, warm mediterranean colors, ochre and azure tones, smooth PBR shading, painterly textures, bright sunny italian coastal atmosphere, pixar luca style, detailed textures, cinematic composition"

caption2 = generate_style_caption(
    content_type='character',
    lighting_style='three-point lighting, rim light separation',
    color_palette='saturated colors, high contrast',
    rendering_style='smooth skin shader, subsurface scattering',
    atmosphere='dramatic lighting mood',
    film_name='pixar style'
)
```

**VLM-based style caption generation:**

```python
def generate_vlm_style_caption(image_path):
    """
    Use VLM to analyze and describe style

    Prompt engineering is critical for style focus
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = """Analyze the visual STYLE of this 3D animated image (NOT the content/story).

Describe in detail:
1. **Lighting style**: Direction, softness, color temperature, contrast
2. **Color palette**: Dominant colors, saturation, warmth/coolness
3. **Rendering style**: Materials (skin, cloth, surfaces), shading model, level of detail
4. **Atmospheric effects**: Depth of field, bloom, fog, volumetric lighting
5. **Overall aesthetic**: Mood, film reference (if recognizable)

Focus on HOW it looks, not WHAT it shows.

Format as comma-separated tags: "lighting description, color description, rendering description, atmosphere description, style reference"

Keep to 60-80 tokens total.
"""

    from PIL import Image
    image = Image.open(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=120)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return caption
```

---

### 4.7 Dataset Assembly

```python
def assemble_style_dataset(
    selected_frame_paths,
    output_dir,
    caption_generator,
    style_config
):
    """
    Assemble final style LoRA training dataset

    Args:
        selected_frame_paths: List of selected frame paths
        output_dir: Output directory
        caption_generator: Function to generate style captions
        style_config: Dict with lighting_style, color_palette, etc.
    """
    import shutil

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Categorize frames by content type (for caption variation)
    # Simple heuristic: check if frame has large centered object (character)
    # vs wide scene (background)

    for i, frame_path in enumerate(tqdm(selected_frame_paths, desc="Assembling dataset")):
        frame_path = Path(frame_path)

        # Copy image
        dst_img_path = images_dir / f"style_{i:04d}.jpg"
        shutil.copy(frame_path, dst_img_path)

        # Determine content type (simple heuristic or manual labels)
        # For now, alternate between types for diversity
        content_types = ['scene', 'character', 'wide_shot', 'close_up', 'object']
        content_type = content_types[i % len(content_types)]

        # Generate caption
        caption = caption_generator(
            content_type=content_type,
            **style_config
        )

        # Save caption
        caption_path = captions_dir / f"style_{i:04d}.txt"
        caption_path.write_text(caption, encoding='utf-8')

    # Save metadata
    metadata = {
        'total_images': len(selected_frame_paths),
        'style_config': style_config,
        'caption_method': 'template',
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Style LoRA dataset assembled: {len(selected_frame_paths)} images")
    print(f"   Style: {style_config}")

# Usage
style_config = {
    'lighting_style': 'soft natural sunlight, warm key light, subtle rim light',
    'color_palette': 'warm mediterranean colors, ochre and azure tones, high saturation',
    'rendering_style': 'smooth PBR shading, painterly textures, detailed materials',
    'atmosphere': 'bright sunny italian coastal atmosphere, inviting mood',
    'film_name': 'pixar luca style',
}

assemble_style_dataset(
    selected_frame_paths=representative_frames,
    output_dir="/path/to/luca_style_lora",
    caption_generator=generate_style_caption,
    style_config=style_config
)
```

---

## 5. Implementation Details

See: `scripts/generic/training/prepare_style_lora_data.py` (to be implemented)

---

## 6. Training Strategy

### 6.1 Style LoRA Training Config

```toml
[model]
pretrained_model_name_or_path = "/path/to/sd_xl_base_1.0.safetensors"
vae = "/path/to/sdxl_vae.safetensors"

[dataset]
train_data_dir = "/path/to/luca_style_lora/images"
caption_extension = ".txt"
resolution = "1024,1024"
batch_size = 4
num_workers = 4

# Augmentation strategy for style LoRA
color_aug = false  # CRITICAL: Preserve color style!
flip_aug = true  # OK for style (content-agnostic)
random_crop = true  # OK, different compositions

[network]
network_module = "networks.lora"
network_dim = 80            # HIGHER rank for style (64-96)
network_alpha = 40

# Train both UNet and text encoder
network_train_unet_only = false

[training]
output_dir = "/path/to/lora_outputs/luca_style"
output_name = "luca_style"

learning_rate = 0.0002      # Slightly lower for style
unet_lr = 0.0002
text_encoder_lr = 0.00005   # Much lower for text encoder

lr_scheduler = "constant_with_warmup"  # Or cosine
lr_warmup_steps = 200

max_train_epochs = 15       # Longer training for style
save_every_n_epochs = 3

mixed_precision = "fp16"

[optimizer]
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.05"]  # Lower weight decay

[noise]
noise_offset = 0.1  # Higher noise offset for style variation

[saving]
save_model_as = "safetensors"
save_precision = "fp16"

[sample]
sample_every_n_epochs = 3
sample_prompts = "/path/to/prompts/style_test_prompts.txt"
```

### 6.2 Style-Specific Training Considerations

**Higher Rank (64-96):**
- Style affects global and local features
- Need capacity for color, lighting, materials, edges

**Lower Learning Rate:**
- Style is subtle, avoid overfitting
- Gentle adaptation of base model's style tendencies

**Longer Training (15-20 epochs):**
- Style features are abstract, need more iterations
- Monitor for overfitting (style becoming too specific)

**NO Color Augmentation:**
- **Critical:** Color augmentation destroys color palette learning
- Would randomize the very style you're trying to capture

**Higher Noise Offset:**
- Helps learn style across different noise levels
- Improves robustness

---

## 7. Testing and Evaluation

### 7.1 Test Prompts

**Test with varied content, same style keywords:**

```text
# Characters (different identities)
a 3d animated young boy character, luca style, warm mediterranean lighting, smooth shading
a 3d animated female character, luca style, soft sunlight, vibrant colors

# Scenes (different environments)
a 3d animated italian coastal town, luca style, golden hour lighting, ochre and azure palette
a 3d animated underwater ocean scene, luca style, volumetric light rays, cyan tones

# Objects
a 3d animated vintage vespa scooter, luca style, bright sunny day, detailed textures

# Abstract prompts
a beautiful landscape, luca style, cinematic lighting
a portrait, luca style, natural lighting
```

### 7.2 Evaluation Metrics

1. **Color Histogram Similarity**: Compare generated vs source frame color distributions
2. **Lighting Consistency**: Brightness/contrast statistics
3. **Perceptual Style Similarity**: LPIPS distance
4. **User Study**: Side-by-side comparison with source material

### 7.3 Style Transfer Strength Control

**At inference time, adjust LoRA weight:**

```python
# Subtle style influence
pipe.set_adapters(["luca_style"], adapter_weights=[0.5])

# Strong style influence
pipe.set_adapters(["luca_style"], adapter_weights=[1.2])

# Combine with other LoRAs
pipe.load_lora_weights(character_lora, adapter_name="character")
pipe.load_lora_weights(style_lora, adapter_name="style")
pipe.set_adapters(["character", "style"], adapter_weights=[1.0, 0.8])
```

---

## 8. Common Problems

### Problem 1: Style LoRA Too Weak

**Symptoms:** Generated images don't show style influence

**Solutions:**
- Increase LoRA weight (1.0 → 1.5)
- Include style keywords in prompt
- Retrain with higher rank (96)
- Longer training epochs

### Problem 2: Style LoRA Too Strong / Overfitting

**Symptoms:** All generated images look nearly identical, loss of variety

**Solutions:**
- Decrease LoRA weight (1.0 → 0.6)
- Reduce training epochs
- Lower learning rate
- More diverse training data

### Problem 3: Style Conflicts with Character LoRA

**Symptoms:** Character LoRA + Style LoRA produce weird results

**Solutions:**
- Adjust relative weights (character 1.0, style 0.7)
- Ensure character LoRA was trained with neutral/varied lighting
- Train character LoRA and style LoRA with compatible captions

### Problem 4: Color Palette Not Transferring

**Cause:** Color augmentation during training, or insufficient color keywords in captions

**Solutions:**
- **Disable color_aug** in training config
- Add explicit color keywords to all captions
- Retrain with color-focused captions

### Problem 5: Only Works on Specific Content

**Cause:** Training data too homogeneous (e.g., only characters, no backgrounds)

**Solutions:**
- Add more content diversity to training data
- Include characters, backgrounds, objects, close-ups, wide shots
- Use representative sampling strategy (Section 4.5)

---

## Summary: Style LoRA Implementation Checklist

- [ ] Collect diverse frames from same film (characters, scenes, objects, various angles)
- [ ] Extract style features (color, lighting, texture, edges)
- [ ] Filter for style consistency (remove outliers)
- [ ] Select representative samples (maximize content diversity)
- [ ] Generate style-focused captions (emphasize lighting, color, materials)
- [ ] Assemble Kohya_ss dataset
- [ ] Configure training (high rank 64-96, NO color aug, longer epochs)
- [ ] Train style LoRA
- [ ] Test on varied content types
- [ ] Evaluate color/lighting consistency
- [ ] Adjust LoRA weight for desired strength
- [ ] Combine with other LoRAs (character, background)
- [ ] Debug common issues

---

**This completes the Style LoRA deep-dive guide.**

**All LoRA type guides:**
1. `01_CHARACTER_LORA_DEEP_DIVE.md` - Identity preservation
2. `02_EXPRESSION_LORA_DEEP_DIVE.md` - Facial emotions
3. `03_POSE_LORA_DEEP_DIVE.md` - Body poses and actions
4. `04_BACKGROUND_LORA_DEEP_DIVE.md` - Scene environments
5. `05_STYLE_LORA_DEEP_DIVE.md` - Visual aesthetics ✅

**This guide is part of the 3D Animation LoRA Pipeline project.**
