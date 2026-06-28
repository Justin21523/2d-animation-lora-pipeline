# Background/Scene LoRA Deep Dive: Technical Implementation Guide

**Target Audience:** ML engineers implementing background/scene LoRA for 3D animated environments
**Difficulty Level:** 🟡 **MEDIUM** (Scene understanding + inpainting)
**Prerequisites:** Basic understanding of LoRA, scene segmentation, visual embeddings

---

## Table of Contents

1. [Why Background LoRA is Important](#1-why-background-lora-is-important)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [Model Selection Framework](#3-model-selection-framework)
4. [Data Preparation Pipeline](#4-data-preparation-pipeline)
5. [Implementation Details](#5-implementation-details)
6. [Training Strategy](#6-training-strategy)
7. [Testing and Evaluation](#7-testing-and-evaluation)
8. [Common Problems](#8-common-problems)

---

## 1. Why Background LoRA is Important

### 1.1 The Core Problem

Background/Scene LoRA aims to teach the model specific environmental settings and scene styles **independently of characters**. This enables:

- Consistent scene generation across different character compositions
- Reusable environment assets ("Italian coastal town", "underwater cave", "sunny plaza")
- Compositional generation: Character LoRA + Background LoRA for complete scenes
- Art direction control (lighting, atmosphere, architectural style)

**Key challenge:** Extract clean backgrounds from frames with characters, then cluster similar scenes.

### 1.2 3D Animation Background Characteristics

| Aspect | Real Photographs | 3D Animated Backgrounds | Impact |
|--------|------------------|-------------------------|--------|
| **Lighting** | Natural, variable | Stylized, cinematic, artist-controlled | Distinct lighting signatures |
| **Textures** | Photorealistic | Painterly, simplified, PBR materials | Unique material appearance |
| **Depth of field** | Optical physics | Artistic blur for focus | Intentional bokeh effects |
| **Color grading** | Natural tones | Saturated, mood-driven palettes | Strong color identity |
| **Architecture** | Real-world | Stylized proportions, fantasy elements | Unique spatial geometry |
| **Atmospheric effects** | Realistic | Exaggerated (god rays, mist, glow) | Dramatic visual style |

**Example:** Pixar's "Luca" has distinct Italian coastal town aesthetics with:
- Warm Mediterranean color palette (ochre, terracotta, azure)
- Textured stone buildings with stylized imperfections
- Sunlit plazas with dramatic shadows
- Underwater scenes with volumetric light rays

### 1.3 Why Background LoRA is Valuable

1. **Scene consistency** across generated images
2. **Compositional control**: Mix character + background LoRAs
3. **Art style preservation**: Maintain film's visual identity
4. **Efficient asset reuse**: One background LoRA works for any character
5. **Lighting transfer**: Apply film's lighting to new scenes

---

## 2. Theoretical Foundation

### 2.1 What is Background LoRA?

Background LoRA learns **environmental and scene-specific features** in the latent space:

```
Base Model Latent → Background LoRA → Scene-Conditioned Latent
```

**Key idea:** Background features should be **character-free** and capture scene essence (architecture, lighting, atmosphere, color palette).

### 2.2 LoRA Mathematics (Quick Recap)

Same as other LoRA types:

```
W' = W + ΔW
ΔW = A × B

where:
- rank typically 48-64 for backgrounds (moderate complexity)
```

### 2.3 Background LoRA vs Other LoRA Types

| Aspect | Character LoRA | Expression LoRA | Pose LoRA | Background LoRA |
|--------|---------------|-----------------|-----------|-----------------|
| **Target** | Identity | Facial emotion | Body action | Scene/environment |
| **Dataset size** | 200-500 | 100-300/class | 150-400/action | 200-400 scenes |
| **Diversity requirement** | Varied poses | Varied identities | Varied angles | Varied compositions |
| **Caption focus** | Character name | Expression | Action verbs | Scene descriptors |
| **Training epochs** | 8-12 | 6-10 | 8-12 | 10-15 |
| **Rank (dim)** | 32-64 | 32-48 | 48-64 | 48-64 |
| **Key preprocessing** | Face clustering | Face detection | Pose estimation | **Inpainting** |

**Critical difference:** Background LoRA requires **character removal** via inpainting before training.

---

## 3. Model Selection Framework

### 3.1 Task Breakdown

Background LoRA data preparation requires **4 sub-tasks**:

1. **Background Extraction**: Remove characters from frames (SAM2 + inpainting)
2. **Scene Understanding**: Extract visual features for clustering
3. **Scene Clustering**: Group similar scenes (architecture, lighting, location)
4. **Scene Classification**: Categorize scene types (outdoor/indoor, time of day, location)

---

### 3.2 Background Extraction (Inpainting)

**Goal:** Remove characters from frames to obtain clean backgrounds

#### Option 1: LaMa (Large Mask Inpainting) ⭐ **Recommended**

**Pros:**
- **State-of-the-art quality** for large mask regions
- Fast inference (~0.5s per image on GPU)
- Handles complex textures and patterns well
- Open-source, easy to deploy

**Cons:**
- May struggle with very large masks (>50% of image)
- Requires mask dilation for clean edges

**When to use:** Default choice for character removal

**Implementation:**
```python
from simple_lama_inpainting import SimpleLama
import cv2
import numpy as np

# Initialize LaMa
lama = SimpleLama(device='cuda')

# Load background layer with character mask
background = cv2.imread('background.png')  # SAM2 output with masked region
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Character mask

# Optional: Dilate mask for cleaner removal
kernel = np.ones((20, 20), np.uint8)  # 20px dilation
mask_dilated = cv2.dilate(mask, kernel, iterations=1)

# Inpaint
inpainted = lama(background, mask_dilated)

# Save clean background
cv2.imwrite('background_clean.jpg', inpainted)
```

**3D-specific tuning:**
```python
# Adjust mask dilation based on character-background boundary
# 3D animation has soft anti-aliased edges
dilation_px = 20  # Larger for softer edges
```

---

#### Option 2: PowerPaint (ECCV 2024) 🔥 **Best for Context-Aware**

**Pros:**
- **Text-guided inpainting** (can describe what should fill the region)
- Better context understanding
- Handles large masks better than LaMa

**Cons:**
- Slower inference (~2-3s per image)
- Requires text prompts (more complex)
- Larger model size

**When to use:** Complex scenes, need semantic control

**Implementation:**
```python
from powerpaint import PowerPaintInpainter

inpainter = PowerPaintInpainter(device='cuda')

# Text-guided inpainting
prompt = "italian coastal town building, stone wall, sunny day, pixar style"
inpainted = inpainter.inpaint(
    image=background,
    mask=mask_dilated,
    prompt=prompt,
    negative_prompt="character, person, human",
    num_inference_steps=50
)
```

---

#### Option 3: OpenCV Inpainting (Fast Fallback)

**Pros:**
- **Extremely fast** (<0.1s per image)
- No GPU required
- Good for small masks

**Cons:**
- Lower quality than LaMa/PowerPaint
- Poor on large regions

**When to use:** Quick preview, CPU-only

**Implementation:**
```python
# Telea algorithm
inpainted = cv2.inpaint(background, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Or NS (Navier-Stokes)
inpainted = cv2.inpaint(background, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
```

---

### 3.3 Scene Understanding Models

**Goal:** Extract visual features for scene clustering and classification

#### Option 1: EVA-CLIP (5B parameters) ⭐ **Recommended SOTA**

**Pros:**
- **State-of-the-art** vision-language model (5 billion parameters!)
- Excellent scene understanding
- Strong on artistic/stylized content
- Good for 3D animation aesthetics

**Cons:**
- Large model size (~20GB)
- Slower inference (~1s per image)
- High VRAM requirements (16GB+)

**When to use:** Best quality scene embeddings, GPU available

**Implementation:**
```python
import torch
from transformers import AutoModel, AutoProcessor

# Load EVA-CLIP
model_name = "BAAI/EVA-CLIP-8B"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

def extract_scene_embedding(image_path):
    """Extract scene embedding with EVA-CLIP"""
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to('cuda')

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()

# Usage
embedding = extract_scene_embedding('background_clean.jpg')  # Shape: (768,) or (1024,)
```

---

#### Option 2: SigLIP (Google, Newer than CLIP) 🚀 **Fast & Strong**

**Pros:**
- **Better than CLIP** on many benchmarks
- Faster inference than EVA-CLIP
- Good scene understanding
- Multiple sizes available

**Cons:**
- Not as strong as EVA-CLIP on artistic content
- Fewer parameters than EVA-CLIP

**When to use:** Balance of speed and quality

**Implementation:**
```python
from transformers import AutoModel, AutoProcessor

model_name = "google/siglip-so400m-patch14-384"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to('cuda')

def extract_scene_embedding_siglip(image_path):
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)

    return outputs.cpu().numpy().flatten()
```

---

#### Option 3: DINOv2 (Self-Supervised) 🎯 **No Text Bias**

**Pros:**
- **Pure visual features** (no text bias from CLIP-style training)
- Excellent for clustering visually similar scenes
- Multiple sizes (small, base, large, giant)
- Good on out-of-distribution data

**Cons:**
- No text alignment (can't query with scene descriptions)
- Requires more scenes for good clustering

**When to use:** Unsupervised clustering, no scene labels

**Implementation:**
```python
import torch
from transformers import AutoImageProcessor, AutoModel

model_name = "facebook/dinov2-large"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to('cuda')

def extract_scene_embedding_dino(image_path):
    from PIL import Image

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1024)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten()
```

---

#### Option 4: BLIP2 (Vision-Language, Good for Captioning)

**Pros:**
- Can generate scene descriptions automatically
- Good for creating captions
- Understands scene context

**Cons:**
- Slower
- Not optimized for pure embeddings

**When to use:** Need automatic scene captioning

---

#### Option 5: Hand-Crafted Features (Baseline)

**Pros:**
- No ML model needed
- Fast
- Interpretable

**Cons:**
- Lower quality clustering
- Misses semantic information

**Implementation:**
```python
def extract_scene_features_handcrafted(image_path):
    """
    Extract hand-crafted scene features

    Features:
    1. Color histogram (dominant colors)
    2. Spatial color distribution (4x4 grid)
    3. Edge density (scene complexity)
    4. Brightness statistics
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Color histogram (HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256])

    # Normalize
    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-6)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-6)
    hist_v = hist_v.flatten() / (hist_v.sum() + 1e-6)

    # 2. Spatial color distribution (4x4 grid mean colors)
    h, w = img.shape[:2]
    grid_features = []
    for i in range(4):
        for j in range(4):
            y1, y2 = i * h // 4, (i + 1) * h // 4
            x1, x2 = j * w // 4, (j + 1) * w // 4
            patch = img[y1:y2, x1:x2]
            mean_color = patch.mean(axis=(0, 1))
            grid_features.extend(mean_color)

    grid_features = np.array(grid_features) / 255.0

    # 3. Edge density
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / edges.size

    # 4. Brightness statistics
    brightness_mean = img_hsv[:, :, 2].mean() / 255.0
    brightness_std = img_hsv[:, :, 2].std() / 255.0

    # Concatenate all features
    features = np.concatenate([
        hist_h,  # 32
        hist_s,  # 32
        hist_v,  # 32
        grid_features,  # 48 (4x4x3)
        [edge_density, brightness_mean, brightness_std]  # 3
    ])  # Total: 147 dimensions

    return features
```

---

### 3.4 Scene Clustering Methods

**Goal:** Group similar scenes together

#### Method 1: HDBSCAN (Density-Based) ⭐ **Recommended**

**Pros:**
- Automatic number of clusters
- Handles noise well
- No need to specify k

**Cons:**
- Sensitive to hyperparameters
- May create too many small clusters

**Implementation:**
```python
import hdbscan
import umap
from sklearn.preprocessing import normalize

def cluster_scenes_hdbscan(embeddings, min_cluster_size=15, min_samples=3):
    """
    Cluster scene embeddings using UMAP + HDBSCAN

    Args:
        embeddings: (N, D) scene embeddings
        min_cluster_size: Minimum cluster size (3D: 10-20)
        min_samples: Minimum samples for core points (3D: 2-3)

    Returns:
        labels: (N,) cluster assignments
    """
    # Normalize embeddings
    embeddings_norm = normalize(embeddings)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    embeddings_reduced = reducer.fit_transform(embeddings_norm)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_epsilon=0.0
    )
    labels = clusterer.fit_predict(embeddings_reduced)

    # Print cluster info
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    print(f"✅ Clustering complete:")
    print(f"   Clusters: {n_clusters}")
    print(f"   Noise: {n_noise}")

    return labels
```

---

#### Method 2: KMeans (Fixed k)

**Pros:**
- Simple, fast
- Predictable number of clusters

**Cons:**
- Must specify k (number of scenes)
- Assumes spherical clusters

**When to use:** Know exact number of scene types

**Implementation:**
```python
from sklearn.cluster import KMeans

def cluster_scenes_kmeans(embeddings, n_clusters=8):
    """Simple KMeans clustering"""
    embeddings_norm = normalize(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_norm)

    return labels
```

---

#### Method 3: Agglomerative (Hierarchical)

**Pros:**
- Can explore hierarchy
- Flexible distance metrics

**Cons:**
- Slower on large datasets

---

### 3.5 Scene Classification (Optional)

**Goal:** Assign semantic labels to scenes

#### Approach 1: Zero-shot with CLIP/EVA-CLIP

```python
def classify_scene_zero_shot(image_path, scene_categories):
    """
    Zero-shot scene classification using CLIP

    Args:
        image_path: Path to background image
        scene_categories: List of scene descriptions
            e.g., ["outdoor sunny plaza", "indoor cozy room", "underwater scene"]

    Returns:
        category (str), confidence (float)
    """
    import clip
    from PIL import Image

    model, preprocess = clip.load("ViT-L/14", device='cuda')

    # Prepare prompts
    prompts = [f"a 3d animated scene of {cat}, pixar style" for cat in scene_categories]
    text_tokens = clip.tokenize(prompts).to('cuda')

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to('cuda')

    # Encode
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        confidence, idx = similarity.max(dim=-1)

    category = scene_categories[idx.item()]
    confidence = confidence.item()

    return category, confidence
```

---

### 3.6 Model Selection Decision Tree

```
Background Extraction (Inpainting)
├─ Default → LaMa (fast, high quality)
├─ Complex scenes → PowerPaint (text-guided)
└─ Quick preview → OpenCV (fast, CPU)

Scene Understanding
├─ Best quality → EVA-CLIP 5B (SOTA)
├─ Balanced → SigLIP (fast, strong)
├─ Unsupervised → DINOv2 (no text bias)
└─ Lightweight → Hand-crafted features

Scene Clustering
├─ Unknown k → HDBSCAN (auto clusters)
├─ Known k → KMeans (fixed clusters)
└─ Hierarchical → Agglomerative

Scene Classification
├─ With labels → Zero-shot CLIP
└─ No labels → Skip, use clustering only
```

---

### 3.7 Recommended Pipeline (3D Animated Backgrounds)

**For Pixar-style 3D backgrounds (like Luca):**

```yaml
background_extraction:
  method: lama
  config:
    mask_dilation: 20  # px, for soft edges
    device: cuda

scene_understanding:
  model: eva_clip  # SOTA for 3D scenes
  config:
    model_name: BAAI/EVA-CLIP-8B
    device: cuda

scene_clustering:
  method: hdbscan
  config:
    min_cluster_size: 15  # 3D: 10-20
    min_samples: 3
    metric: cosine

scene_classification:
  method: zero_shot_clip  # Optional
  categories:
    - "sunny outdoor italian coastal town plaza"
    - "indoor cozy home interior"
    - "underwater ocean scene with light rays"
    - "narrow cobblestone street"
    - "seaside dock and boats"
```

---

## 4. Data Preparation Pipeline

### 4.1 Pipeline Overview

```
SAM2 Backgrounds (with character masks)
  ↓
LaMa Inpainting (remove characters)
  ↓
Quality Filtering (blur, artifacts, completeness)
  ↓
Scene Feature Extraction (EVA-CLIP or DINOv2)
  ↓
Scene Clustering (HDBSCAN)
  ↓
Manual Review & Labeling
  ↓
Caption Generation (scene-focused)
  ↓
Dataset Assembly (Kohya_ss format)
```

### 4.2 Step-by-Step Implementation

#### Step 1: LaMa Inpainting

**Goal:** Remove characters from SAM2 background layers

```python
from simple_lama_inpainting import SimpleLama
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json

class BackgroundInpainter:
    """Inpaint SAM2 backgrounds to remove characters"""

    def __init__(self, mask_dilation=20, device='cuda'):
        self.lama = SimpleLama(device=device)
        self.mask_dilation = mask_dilation

    def inpaint_background(self, background_path, mask_path):
        """
        Inpaint a single background

        Args:
            background_path: SAM2 background layer (with masked regions)
            mask_path: Character mask

        Returns:
            inpainted image (np.array)
        """
        # Load background and mask
        background = cv2.imread(str(background_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if background is None or mask is None:
            return None

        # Dilate mask for cleaner removal
        if self.mask_dilation > 0:
            kernel = np.ones((self.mask_dilation, self.mask_dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Inpaint
        inpainted = self.lama(background, mask)

        return inpainted

# Usage
inpainter = BackgroundInpainter(mask_dilation=20, device='cuda')

sam2_dir = Path("/path/to/sam2_output")
output_dir = Path("/path/to/backgrounds_clean")
output_dir.mkdir(exist_ok=True)

# Process all SAM2 backgrounds
background_metadata = []

for frame_dir in tqdm(sorted((sam2_dir / "frames").iterdir())):
    if not frame_dir.is_dir():
        continue

    # Find background layer
    background_path = frame_dir / "background.png"

    # Aggregate all character masks for this frame
    masks = list(frame_dir.glob("instance_*_mask.png"))

    if not background_path.exists() or len(masks) == 0:
        continue

    # Combine all character masks
    combined_mask = np.zeros_like(cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE))
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        combined_mask = np.maximum(combined_mask, mask)

    # Save combined mask temporarily
    temp_mask_path = frame_dir / "combined_mask.png"
    cv2.imwrite(str(temp_mask_path), combined_mask)

    # Inpaint
    inpainted = inpainter.inpaint_background(background_path, temp_mask_path)

    if inpainted is None:
        continue

    # Save clean background
    output_filename = f"{frame_dir.name}_background_clean.jpg"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), inpainted)

    # Metadata
    background_metadata.append({
        'frame_name': frame_dir.name,
        'background_clean_path': str(output_path),
        'original_background_path': str(background_path),
        'num_characters_removed': len(masks),
    })

# Save metadata
with open(output_dir / "background_metadata.json", 'w') as f:
    json.dump(background_metadata, f, indent=2)

print(f"✅ Inpainted {len(background_metadata)} backgrounds")
```

---

#### Step 2: Quality Filtering

```python
def filter_background_quality(background_metadata, config):
    """
    Filter backgrounds by quality

    Quality criteria:
    1. Blur detection (reject blurry backgrounds)
    2. Inpainting artifacts (detect obvious seams/distortions)
    3. Completeness (enough non-masked area)
    """
    from collections import defaultdict

    filtered = []
    rejection_stats = defaultdict(int)

    for bg in tqdm(background_metadata, desc="Quality filtering"):
        bg_path = Path(bg['background_clean_path'])
        img = cv2.imread(str(bg_path))

        if img is None:
            rejection_stats['read_failed'] += 1
            continue

        # 1. Blur detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < config['min_blur_score']:
            rejection_stats['too_blurry'] += 1
            continue

        # 2. Color variance (detect if mostly solid color = bad inpainting)
        color_variance = img.std()
        if color_variance < config['min_color_variance']:
            rejection_stats['low_color_variance'] += 1
            continue

        # 3. Edge density (too few edges = potential artifact)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.sum() / edges.size

        if edge_density < config['min_edge_density']:
            rejection_stats['low_edge_density'] += 1
            continue

        bg['blur_score'] = blur_score
        bg['color_variance'] = float(color_variance)
        bg['edge_density'] = edge_density

        filtered.append(bg)

    print(f"✅ Quality filtering: {len(filtered)} / {len(background_metadata)} kept")
    print("❌ Rejections:")
    for reason, count in sorted(rejection_stats.items()):
        print(f"  - {reason}: {count}")

    return filtered

# Config
quality_config = {
    'min_blur_score': 100.0,  # Laplacian variance
    'min_color_variance': 10.0,  # Std of pixel values
    'min_edge_density': 0.01,  # Canny edge percentage
}

filtered_backgrounds = filter_background_quality(background_metadata, quality_config)
```

---

#### Step 3: Scene Feature Extraction

```python
def extract_all_scene_features(filtered_backgrounds, model_type='eva_clip'):
    """
    Extract scene features for all backgrounds

    Args:
        filtered_backgrounds: List of background dicts
        model_type: 'eva_clip', 'siglip', 'dinov2', or 'handcrafted'

    Returns:
        embeddings (np.array): (N, D) embeddings
    """
    if model_type == 'eva_clip':
        from transformers import AutoModel, AutoProcessor

        model_name = "BAAI/EVA-CLIP-8B"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')

        def extract_fn(img_path):
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to('cuda')

            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()

    elif model_type == 'dinov2':
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-large"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to('cuda')

        def extract_fn(img_path):
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt").to('cuda')

            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()

    elif model_type == 'handcrafted':
        extract_fn = extract_scene_features_handcrafted  # From Section 3.3

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Extract features
    embeddings = []
    for bg in tqdm(filtered_backgrounds, desc="Extracting features"):
        embedding = extract_fn(bg['background_clean_path'])
        embeddings.append(embedding)
        bg['embedding'] = embedding.tolist()  # Save for later

    embeddings = np.array(embeddings)
    print(f"✅ Extracted features: {embeddings.shape}")

    return embeddings

# Usage
embeddings = extract_all_scene_features(filtered_backgrounds, model_type='eva_clip')
```

---

#### Step 4: Scene Clustering

```python
def cluster_scenes(embeddings, filtered_backgrounds, method='hdbscan', **kwargs):
    """
    Cluster scenes by similarity

    Args:
        embeddings: (N, D) scene embeddings
        filtered_backgrounds: List of background dicts
        method: 'hdbscan' or 'kmeans'
        **kwargs: Method-specific parameters

    Returns:
        labels: (N,) cluster assignments
    """
    if method == 'hdbscan':
        labels = cluster_scenes_hdbscan(
            embeddings,
            min_cluster_size=kwargs.get('min_cluster_size', 15),
            min_samples=kwargs.get('min_samples', 3)
        )

    elif method == 'kmeans':
        labels = cluster_scenes_kmeans(
            embeddings,
            n_clusters=kwargs.get('n_clusters', 8)
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    # Assign labels to backgrounds
    for i, bg in enumerate(filtered_backgrounds):
        bg['scene_cluster'] = int(labels[i])

    # Cluster statistics
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    print("Cluster distribution:")
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        if cluster_id == -1:
            print(f"  Noise: {count}")
        else:
            print(f"  Cluster {cluster_id}: {count}")

    return labels

# Usage
labels = cluster_scenes(
    embeddings,
    filtered_backgrounds,
    method='hdbscan',
    min_cluster_size=15,
    min_samples=3
)
```

---

#### Step 5: Organize by Scene Cluster

```python
def organize_by_scene_cluster(filtered_backgrounds, output_dir):
    """
    Organize backgrounds into folders by scene cluster

    Output structure:
        output_dir/
        ├── scene_0/
        ├── scene_1/
        ├── scene_2/
        └── noise/
    """
    import shutil

    output_dir = Path(output_dir)

    # Group by cluster
    cluster_groups = defaultdict(list)
    for bg in filtered_backgrounds:
        cluster_id = bg['scene_cluster']
        cluster_groups[cluster_id].append(bg)

    # Create folders and copy files
    for cluster_id, backgrounds in cluster_groups.items():
        if cluster_id == -1:
            cluster_dir = output_dir / "noise"
        else:
            cluster_dir = output_dir / f"scene_{cluster_id}"

        cluster_dir.mkdir(parents=True, exist_ok=True)

        for bg in backgrounds:
            src_path = Path(bg['background_clean_path'])
            dst_path = cluster_dir / src_path.name
            shutil.copy(src_path, dst_path)
            bg['organized_path'] = str(dst_path)

        print(f"📁 {'Noise' if cluster_id == -1 else f'Scene {cluster_id}'}: {len(backgrounds)} backgrounds")

    return cluster_groups

cluster_groups = organize_by_scene_cluster(
    filtered_backgrounds,
    output_dir="/path/to/scenes_organized"
)
```

---

#### Step 6: Manual Review & Labeling

**Interactive tool to:**
- View all clusters
- Merge similar clusters
- Split mixed clusters
- Assign semantic labels ("sunny plaza", "underwater", "indoor home")
- Remove outliers

```bash
# Use existing interactive selector (adapt from character clustering)
python scripts/generic/clustering/interactive_scene_selector.py \
  --scene-dir /path/to/scenes_organized \
  --output-dir /path/to/scenes_reviewed
```

---

#### Step 7: Caption Generation

**Caption strategy for Background LoRA:**

Captions should:
1. **Describe scene setting** (primary focus)
2. Include architectural/environmental details
3. Specify lighting and atmosphere
4. Mention color palette
5. Add 3D style descriptors

```python
def generate_background_caption(
    scene_type: str,
    location: str = "",
    lighting: str = "natural lighting",
    atmosphere: str = "",
    include_3d_style: bool = True
):
    """
    Generate caption for background LoRA training

    Args:
        scene_type: Scene category (outdoor, indoor, underwater, etc.)
        location: Specific location (italian coastal town, home interior, etc.)
        lighting: Lighting description (sunny, golden hour, soft indoor, etc.)
        atmosphere: Mood/atmosphere (peaceful, dramatic, mysterious, etc.)
        include_3d_style: Add 3D style tags

    Returns:
        Caption string
    """
    parts = []

    # Scene setting (primary focus)
    if location:
        parts.append(f"a 3d animated scene of {location}")
    else:
        parts.append(f"a 3d animated {scene_type} scene")

    # Lighting
    parts.append(lighting)

    # Atmosphere
    if atmosphere:
        parts.append(atmosphere)

    # 3D style tags
    if include_3d_style:
        parts.extend([
            "pixar style",
            "detailed environment",
            "cinematic lighting",
            "vibrant colors",
        ])

    caption = ", ".join(parts)
    return caption

# Examples
caption1 = generate_background_caption(
    scene_type="outdoor",
    location="sunny italian coastal town plaza with stone buildings",
    lighting="warm sunlight, golden hour",
    atmosphere="peaceful mediterranean atmosphere"
)
# "a 3d animated scene of sunny italian coastal town plaza with stone buildings, warm sunlight, golden hour, peaceful mediterranean atmosphere, pixar style, detailed environment, cinematic lighting, vibrant colors"

caption2 = generate_background_caption(
    scene_type="underwater",
    location="underwater ocean scene with coral and rocks",
    lighting="volumetric light rays from surface",
    atmosphere="serene underwater atmosphere"
)
```

**VLM-based captions (optional):**

```python
def generate_vlm_background_caption(background_path):
    """Use VLM to auto-generate scene description"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = """Describe this 3D animated background scene in detail.
Focus on:
1. Scene setting (outdoor/indoor, location type)
2. Architectural/environmental elements
3. Lighting (time of day, light quality, shadows)
4. Color palette (dominant colors, mood)
5. Atmosphere and style

Keep description concise (50-70 tokens). Use tags separated by commas.
Do NOT mention characters or people.

Example format: "a 3d animated scene of [location], [architectural details], [lighting], [atmosphere], pixar style, detailed environment"
"""

    from PIL import Image
    image = Image.open(background_path)

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

    output_ids = model.generate(**inputs, max_new_tokens=100)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return caption
```

---

#### Step 8: Dataset Assembly

```python
def assemble_background_dataset(
    cluster_groups,
    output_dir,
    caption_generator,
    scene_labels=None  # Optional: {cluster_id: scene_description}
):
    """
    Assemble final background LoRA training dataset

    Output structure:
        output_dir/
        ├── images/
        │   ├── scene0_001.jpg
        │   ├── scene0_002.jpg
        │   ├── scene1_001.jpg
        │   └── ...
        ├── captions/
        │   ├── scene0_001.txt
        │   └── ...
        └── metadata.json
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    captions_dir = output_dir / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = {
        'total_images': 0,
        'scene_counts': {},
    }

    global_idx = 0

    for cluster_id, backgrounds in cluster_groups.items():
        if cluster_id == -1:  # Skip noise
            continue

        scene_count = 0

        # Get scene label if provided
        if scene_labels and cluster_id in scene_labels:
            scene_desc = scene_labels[cluster_id]
        else:
            scene_desc = None

        for bg in backgrounds:
            # Copy image
            src_path = Path(bg['organized_path'])
            img_filename = f"scene{cluster_id}_{global_idx:04d}.jpg"
            dst_img_path = images_dir / img_filename
            shutil.copy(src_path, dst_img_path)

            # Generate caption
            if scene_desc:
                # Use provided scene description
                caption = caption_generator(**scene_desc)
            else:
                # Generic caption
                caption = f"a 3d animated scene, pixar style, detailed environment, cinematic lighting"

            # Save caption
            caption_path = captions_dir / f"scene{cluster_id}_{global_idx:04d}.txt"
            caption_path.write_text(caption, encoding='utf-8')

            global_idx += 1
            scene_count += 1

        dataset_stats['scene_counts'][f'scene_{cluster_id}'] = scene_count
        dataset_stats['total_images'] += scene_count

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    print(f"✅ Background LoRA dataset assembled: {dataset_stats['total_images']} images")
    print("Scene distribution:")
    for scene, count in sorted(dataset_stats['scene_counts'].items()):
        print(f"  {scene}: {count}")

    return dataset_stats

# Usage with manual scene labels
scene_labels = {
    0: {
        'scene_type': 'outdoor',
        'location': 'sunny italian coastal town plaza',
        'lighting': 'warm sunlight',
        'atmosphere': 'peaceful mediterranean'
    },
    1: {
        'scene_type': 'underwater',
        'location': 'underwater ocean scene',
        'lighting': 'volumetric light rays',
        'atmosphere': 'serene'
    },
    # ... more scenes
}

dataset_stats = assemble_background_dataset(
    cluster_groups=cluster_groups,
    output_dir="/path/to/final_dataset",
    caption_generator=generate_background_caption,
    scene_labels=scene_labels
)
```

---

## 5. Implementation Details

### 5.1 Complete Pipeline Script Reference

See existing: `scripts/generic/training/prepare_background_lora_data.py` (already implemented)

---

## 6. Training Strategy

### 6.1 Background LoRA Training Config

```toml
[model]
pretrained_model_name_or_path = "/path/to/sd_xl_base_1.0.safetensors"
vae = "/path/to/sdxl_vae.safetensors"

[dataset]
train_data_dir = "/path/to/final_dataset/images"
caption_extension = ".txt"
resolution = "1024,1024"
batch_size = 4
num_workers = 4

# Augmentation acceptable for backgrounds (unlike character/pose)
color_aug = false  # Preserve color palette
flip_aug = true  # OK for symmetric scenes
random_crop = true  # OK, gets different compositions

[network]
network_module = "networks.lora"
network_dim = 56            # Moderate rank for scenes
network_alpha = 28

# Train both UNet and text encoder
network_train_unet_only = false

[training]
output_dir = "/path/to/lora_outputs/background_lora"
output_name = "luca_backgrounds"

learning_rate = 0.0003
unet_lr = 0.0003
text_encoder_lr = 0.0001

lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

max_train_epochs = 12  # Longer for backgrounds
save_every_n_epochs = 2

mixed_precision = "fp16"

[optimizer]
optimizer_type = "AdamW8bit"
optimizer_args = ["weight_decay=0.1"]

[noise]
noise_offset = 0.05

[saving]
save_model_as = "safetensors"
save_precision = "fp16"

[sample]
sample_every_n_epochs = 2
sample_prompts = "/path/to/prompts/background_test_prompts.txt"
```

### 6.2 Background-Specific Considerations

**Longer Training (12-15 epochs):**
- Backgrounds have more visual complexity
- Need to learn architectural details, lighting patterns

**Flip Augmentation OK:**
- Unlike pose LoRA, flipping backgrounds doesn't break semantics
- Helps with data augmentation

**Random Crop Acceptable:**
- Different crops = different compositions
- Helps learn scene elements at multiple scales

---

## 7. Testing and Evaluation

### 7.1 Test Prompts

```text
# Scene-only prompts
a 3d animated scene of sunny italian coastal town plaza, warm lighting, pixar style, detailed environment
a 3d animated underwater ocean scene, volumetric light rays, serene atmosphere, pixar style
a 3d animated cozy home interior, soft indoor lighting, warm colors, pixar style

# Combined with character LoRA
[luca] standing in sunny italian coastal town plaza, three-quarter view, pixar style
a 3d animated character swimming in underwater scene, volumetric light, pixar style
```

---

## 8. Common Problems

### Problem 1: Inpainting Artifacts

**Symptoms:** Visible seams, blurry regions where characters were removed

**Solutions:**
- Increase mask dilation (20-30px)
- Use PowerPaint instead of LaMa for complex cases
- Manual cleanup in Photoshop for training set

### Problem 2: Background LoRA Affects Characters

**Symptoms:** Characters look wrong when background LoRA is active

**Cause:** Background LoRA learned character remnants from poor inpainting

**Solutions:**
- Stricter quality filtering
- Manual review of training images
- Lower LoRA weight during inference

### Problem 3: Too Many Scene Clusters

**Cause:** HDBSCAN creates too many small clusters

**Solutions:**
- Increase `min_cluster_size` (20-30)
- Manually merge similar clusters
- Use KMeans with fixed k

### Problem 4: Background LoRA No Effect

**Solutions:**
- Increase LoRA weight
- Check caption keywords match training
- Longer training epochs

---

## Summary: Background LoRA Checklist

- [ ] SAM2 segmentation complete (background layers)
- [ ] LaMa inpainting (remove characters)
- [ ] Quality filtering (blur, artifacts)
- [ ] Scene feature extraction (EVA-CLIP or DINOv2)
- [ ] Scene clustering (HDBSCAN)
- [ ] Manual review and scene labeling
- [ ] Generate scene-focused captions
- [ ] Assemble Kohya_ss dataset
- [ ] Configure training (rank 48-64, longer epochs, flip OK)
- [ ] Train background LoRA
- [ ] Test scene-only and with character LoRA
- [ ] Evaluate scene consistency
- [ ] Debug common issues

---

**Next guide:**
- `05_STYLE_LORA_DEEP_DIVE.md` - Artistic style transfer LoRA (lighting, color grading, rendering style)

**This guide is part of the 3D Animation LoRA Pipeline project.**
