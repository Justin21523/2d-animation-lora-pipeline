# Post-Segmentation Pipeline Implementation Plan

## Overview

This document outlines the complete implementation plan for pipeline stages **after** SAM2 instance segmentation and inpainting are complete. The goal is to transform segmented character instances into high-quality, well-organized training datasets ready for LoRA fine-tuning.

**Pipeline Flow:**
```
Segmented Instances (SAM2 output)
  ↓
Instance Pre-filtering (Remove backgrounds/noise)
  ↓
Face Detection & Identity Clustering (Group by character)
  ↓
Interactive Cluster Review (Human-in-the-loop refinement)
  ↓
Optional: Pose/View Subclustering (Balance training angles)
  ↓
VLM Captioning (Schema-guided descriptions)
  ↓
Dataset Assembly & Augmentation
  ↓
Quality Validation & Export
  ↓
Ready for LoRA Training
```

---

## Stage 1: Instance Pre-filtering

**Purpose:** Remove 80-90% of background instances before expensive clustering operations.

**Location:** `scripts/generic/clustering/instance_prefilter.py`

**Status:** ✅ Already implemented

**Key Features:**
- Semantic filtering (character vs. background classification)
- Heuristic filtering (size, aspect ratio, edge complexity)
- Face detection pre-check
- Batch reporting with statistics

**Input:** `segmented/characters/` (SAM2 output)
**Output:** `filtered_instances/` (character-only instances)

**Configuration:**
```yaml
# configs/stages/clustering/instance_prefilter.yaml
mode: balanced  # conservative | balanced | aggressive
enable_semantic: true
min_face_confidence: 0.3
min_size: 128
max_aspect_ratio: 3.0
edge_complexity_threshold: 0.15
```

**Usage:**
```bash
python scripts/generic/clustering/instance_prefilter.py \
  --input-dir .../segmented/characters \
  --output-dir .../filtered_instances \
  --mode balanced \
  --enable-semantic \
  --batch-report .../prefilter_report.json
```

---

## Stage 2: Face-Centric Identity Clustering

**Purpose:** Group character instances by WHO they are (identity), not visual similarity.

**Location:** `scripts/generic/clustering/face_identity_clustering.py`

**Status:** ✅ Already implemented

**Key Components:**

### A) Face Detection
- **Model:** RetinaFace or MTCNN
- **Purpose:** Locate face regions in each instance
- **Output:** Bounding boxes + confidence scores

### B) Face Recognition Embeddings
- **Model:** ArcFace (ResNet-100)
- **Purpose:** Extract 512-D identity-invariant embeddings
- **Key Property:** Same person → similar embedding, regardless of expression/lighting/angle

### C) Identity Clustering
- **Algorithm:** HDBSCAN
- **Parameters:**
  ```yaml
  min_cluster_size: 12  # 3D characters need fewer samples
  min_samples: 2
  metric: cosine
  cluster_selection_method: eom
  ```
- **Output:** Character-specific folders (character_0/, character_1/, ..., noise/)

### D) Cluster Naming
- **Interactive prompt:** "Please name this character based on sample images"
- **Auto-naming (optional):** Use CLIP text similarity with character name list

**Configuration:**
```yaml
# configs/stages/clustering/face_identity_clustering.yaml
face_detection:
  model: retinaface
  min_confidence: 0.7

face_recognition:
  model: arcface_resnet100
  normalize_embeddings: true

clustering:
  algorithm: hdbscan
  min_cluster_size: 12
  min_samples: 2
  metric: cosine

quality_filter:
  min_face_size: 64
  blur_threshold: 80
  brightness_range: [20, 235]
```

**Implementation Plan:**

**File:** `scripts/generic/clustering/face_identity_clustering.py`

```python
#!/usr/bin/env python3
"""
Face-centric identity clustering for multi-character 3D animation scenes.

This script groups character instances by WHO they are (identity), not visual
similarity, using face detection + recognition + HDBSCAN clustering.
"""

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import cv2

# Face detection
from retinaface import RetinaFace

# Face recognition
from facenet_pytorch import InceptionResnetV1

# Clustering
import hdbscan
from sklearn.preprocessing import normalize

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class FaceIdentityClusterer:
    """
    Multi-character identity clustering using face recognition.
    """

    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.face_detector = self._init_face_detector()
        self.face_recognizer = self._init_face_recognizer()

    def _init_face_detector(self):
        """Initialize RetinaFace detector."""
        # RetinaFace is a function-based API, no initialization needed
        return None

    def _init_face_recognizer(self):
        """Initialize ArcFace (InceptionResnetV1 trained on VGGFace2)."""
        model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        return model

    def detect_faces(self, image_path: Path):
        """
        Detect faces in an image.

        Returns:
            List of face bounding boxes and confidence scores.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        # RetinaFace detection
        faces = RetinaFace.detect_faces(img)

        if not isinstance(faces, dict):
            return []

        results = []
        for key, face_info in faces.items():
            bbox = face_info['facial_area']  # [x1, y1, x2, y2]
            confidence = face_info['score']

            if confidence >= self.config['face_detection']['min_confidence']:
                results.append({
                    'bbox': bbox,
                    'confidence': confidence
                })

        return results

    def extract_face_embedding(self, image_path: Path, bbox: list):
        """
        Extract face recognition embedding from detected face region.

        Args:
            image_path: Path to image
            bbox: [x1, y1, x2, y2]

        Returns:
            512-D face embedding (numpy array)
        """
        img = Image.open(image_path).convert('RGB')

        # Crop face region
        x1, y1, x2, y2 = bbox
        face_crop = img.crop((x1, y1, x2, y2))

        # Resize to 160x160 (InceptionResnetV1 input size)
        face_crop = face_crop.resize((160, 160), Image.BILINEAR)

        # Convert to tensor
        face_tensor = torch.from_numpy(np.array(face_crop)).permute(2, 0, 1).float()
        face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        # Extract embedding
        with torch.no_grad():
            embedding = self.face_recognizer(face_tensor).cpu().numpy()[0]

        # Normalize
        if self.config['face_recognition']['normalize_embeddings']:
            embedding = normalize(embedding.reshape(1, -1))[0]

        return embedding

    def process_instances(self, input_dir: Path):
        """
        Process all instances: detect faces and extract embeddings.

        Returns:
            dict: {
                'embeddings': np.ndarray (N, 512),
                'image_paths': list of Path,
                'face_boxes': list of bbox,
                'confidences': list of float
            }
        """
        image_paths = []
        embeddings = []
        face_boxes = []
        confidences = []

        # Collect all images
        all_images = sorted(input_dir.glob('**/*.png')) + sorted(input_dir.glob('**/*.jpg'))

        print(f"Processing {len(all_images)} instances...")

        for img_path in tqdm(all_images, desc="Extracting face embeddings"):
            # Detect faces
            faces = self.detect_faces(img_path)

            if not faces:
                continue  # Skip instances without detected faces

            # Use the largest/most confident face
            best_face = max(faces, key=lambda x: x['confidence'])

            # Extract embedding
            try:
                embedding = self.extract_face_embedding(img_path, best_face['bbox'])

                image_paths.append(img_path)
                embeddings.append(embedding)
                face_boxes.append(best_face['bbox'])
                confidences.append(best_face['confidence'])

            except Exception as e:
                print(f"Warning: Failed to extract embedding from {img_path}: {e}")
                continue

        return {
            'embeddings': np.array(embeddings),
            'image_paths': image_paths,
            'face_boxes': face_boxes,
            'confidences': confidences
        }

    def cluster_identities(self, embeddings: np.ndarray):
        """
        Cluster face embeddings by identity using HDBSCAN.

        Returns:
            np.ndarray: Cluster labels (-1 for noise)
        """
        cfg = self.config['clustering']

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg['min_cluster_size'],
            min_samples=cfg['min_samples'],
            metric=cfg['metric'],
            cluster_selection_method=cfg['cluster_selection_method']
        )

        labels = clusterer.fit_predict(embeddings)

        return labels

    def save_clusters(self, data: dict, labels: np.ndarray, output_dir: Path):
        """
        Save clustered instances to character-specific folders.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                cluster_dir = output_dir / 'noise'
            else:
                cluster_dir = output_dir / f'character_{label}'

            cluster_dir.mkdir(exist_ok=True)

            # Copy images to cluster folder
            indices = np.where(labels == label)[0]

            for idx in indices:
                src_path = data['image_paths'][idx]
                dst_path = cluster_dir / src_path.name

                # Copy file
                import shutil
                shutil.copy2(src_path, dst_path)

        # Save metadata
        metadata = {
            'total_instances': len(labels),
            'num_characters': len([l for l in unique_labels if l != -1]),
            'noise_count': np.sum(labels == -1),
            'cluster_sizes': {
                int(label): int(np.sum(labels == label))
                for label in unique_labels
            }
        }

        with open(output_dir / 'cluster_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Clustering complete:")
        print(f"   Total instances: {metadata['total_instances']}")
        print(f"   Characters found: {metadata['num_characters']}")
        print(f"   Noise instances: {metadata['noise_count']}")

        return metadata

    def run(self, input_dir: Path, output_dir: Path):
        """
        Complete identity clustering pipeline.
        """
        # Step 1: Process instances (detect faces, extract embeddings)
        print("Step 1: Detecting faces and extracting embeddings...")
        data = self.process_instances(input_dir)

        if len(data['embeddings']) == 0:
            print("❌ No faces detected in any instance. Aborting.")
            return

        # Step 2: Cluster by identity
        print(f"\nStep 2: Clustering {len(data['embeddings'])} faces by identity...")
        labels = self.cluster_identities(data['embeddings'])

        # Step 3: Save clusters
        print("\nStep 3: Saving clusters...")
        metadata = self.save_clusters(data, labels, output_dir)

        # Step 4: Visualize (optional)
        self.visualize_clusters(data['embeddings'], labels, output_dir)

    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, output_dir: Path):
        """
        Create UMAP visualization of identity clusters.
        """
        try:
            from umap import UMAP

            # Reduce to 2D
            reducer = UMAP(n_components=2, metric='cosine', random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)

            # Plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab20',
                alpha=0.6,
                s=50
            )
            plt.colorbar(scatter, label='Character ID')
            plt.title('Face Identity Clustering (UMAP Projection)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.tight_layout()
            plt.savefig(output_dir / 'identity_clustering_visualization.png', dpi=150)
            plt.close()

            print(f"✅ Visualization saved to {output_dir / 'identity_clustering_visualization.png'}")

        except ImportError:
            print("⚠️ UMAP not installed, skipping visualization")


def main():
    parser = argparse.ArgumentParser(description='Face-centric identity clustering')
    parser.add_argument('--input-dir', type=Path, required=True,
                      help='Directory with filtered instances')
    parser.add_argument('--output-dir', type=Path, required=True,
                      help='Output directory for clustered characters')
    parser.add_argument('--config', type=Path,
                      default=Path('configs/stages/clustering/face_identity_clustering.yaml'),
                      help='Configuration file')

    args = parser.parse_args()

    # Run clustering
    clusterer = FaceIdentityClusterer(args.config)
    clusterer.run(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
```

---

## Stage 3: Interactive Cluster Review

**Purpose:** Human-in-the-loop refinement of automated clustering results.

**Location:** `scripts/generic/clustering/interactive_character_selector.py`

**Status:** ✅ Already implemented (basic version)

**Enhancement Plan:** Add web-based UI for better UX

**New Implementation:** `scripts/generic/clustering/web_cluster_review.py`

**Features:**
- Visual grid view of all clusters
- Drag-and-drop to move instances between clusters
- Merge/split cluster operations
- Rename clusters (assign character names)
- Mark instances for deletion
- Export corrected results

**UI Design:**

```html
<!-- cluster_review.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Character Cluster Review</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }

        .cluster-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }

        .cluster-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            width: 300px;
        }

        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .cluster-name {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }

        .instance-count {
            background: #007bff;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
        }

        .image-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .instance-image {
            width: 100%;
            height: 90px;
            object-fit: cover;
            border-radius: 4px;
            cursor: move;
            border: 2px solid transparent;
        }

        .instance-image.selected {
            border-color: #007bff;
        }

        .cluster-actions {
            margin-top: 10px;
            display: flex;
            gap: 8px;
        }

        button {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }

        .btn-rename {
            background: #28a745;
            color: white;
        }

        .btn-merge {
            background: #ffc107;
            color: black;
        }

        .btn-delete {
            background: #dc3545;
            color: white;
        }

        #toolbar {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
        }

        #save-button {
            background: #007bff;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div id="toolbar">
        <h2 style="margin: 0;">Character Cluster Review</h2>
        <button id="save-button">Save Changes</button>
        <span id="status-text"></span>
    </div>

    <div id="cluster-container" class="cluster-container">
        <!-- Clusters will be dynamically loaded here -->
    </div>

    <script src="cluster_review.js"></script>
</body>
</html>
```

**Backend:** Simple Flask server to serve images and save corrections

```python
# scripts/generic/clustering/web_cluster_review.py

from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import json
import shutil

app = Flask(__name__)

CLUSTER_DIR = None
CURRENT_DATA = {}

@app.route('/')
def index():
    return send_from_directory('scripts/generic/clustering/interactive_ui', 'cluster_review.html')

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Load and return all cluster data."""
    global CLUSTER_DIR, CURRENT_DATA

    clusters = {}
    for cluster_path in CLUSTER_DIR.iterdir():
        if cluster_path.is_dir():
            cluster_name = cluster_path.name
            images = [str(img.relative_to(CLUSTER_DIR)) for img in cluster_path.glob('*.png')]
            clusters[cluster_name] = {
                'name': cluster_name,
                'count': len(images),
                'images': images
            }

    CURRENT_DATA = clusters
    return jsonify(clusters)

@app.route('/api/rename', methods=['POST'])
def rename_cluster():
    """Rename a cluster."""
    data = request.json
    old_name = data['old_name']
    new_name = data['new_name']

    old_path = CLUSTER_DIR / old_name
    new_path = CLUSTER_DIR / new_name

    if old_path.exists():
        shutil.move(str(old_path), str(new_path))
        return jsonify({'success': True})

    return jsonify({'success': False, 'error': 'Cluster not found'})

@app.route('/api/move', methods=['POST'])
def move_instance():
    """Move an instance from one cluster to another."""
    data = request.json
    image_path = data['image_path']
    target_cluster = data['target_cluster']

    src = CLUSTER_DIR / image_path
    dst = CLUSTER_DIR / target_cluster / Path(image_path).name

    if src.exists():
        shutil.move(str(src), str(dst))
        return jsonify({'success': True})

    return jsonify({'success': False})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve cluster images."""
    return send_from_directory(CLUSTER_DIR, filename)

def launch_review_ui(cluster_dir: Path, port: int = 5000):
    """Launch the web-based cluster review UI."""
    global CLUSTER_DIR
    CLUSTER_DIR = cluster_dir

    print(f"\n🌐 Launching cluster review UI at http://localhost:{port}")
    print(f"   Cluster directory: {cluster_dir}")
    print(f"\n   Press Ctrl+C to save and exit\n")

    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    launch_review_ui(args.cluster_dir, args.port)
```

---

## Stage 4: Pose/View Subclustering (Optional)

**Purpose:** Further subdivide each character cluster by pose and viewing angle for balanced training.

**Location:** `scripts/generic/clustering/pose_subclustering.py`

**Status:** 🚧 To be implemented

**Algorithm:**

1. **Pose Estimation:** RTM-Pose to extract 17-point skeleton keypoints
2. **View Classification:** Automatic front/3-quarter/profile/back based on keypoint geometry
3. **Pose Normalization:** Remove scale/translation effects
4. **Subcluster:** UMAP + HDBSCAN within each identity cluster

**Benefits:**
- Balanced angle distribution in training data
- Better LoRA generalization across viewpoints
- Consistent captions within pose buckets

**Implementation Outline:**

```python
# scripts/generic/clustering/pose_subclustering.py

def estimate_poses(image_dir: Path):
    """
    Estimate pose keypoints for all images using RTM-Pose.

    Returns:
        dict: {image_path: keypoints (17, 3)}
    """
    pass

def classify_view(keypoints: np.ndarray) -> str:
    """
    Classify viewing angle based on keypoint geometry.

    Returns:
        'front' | 'three_quarter' | 'profile' | 'back'
    """
    # Heuristic based on shoulder/hip width ratio and nose position
    pass

def extract_pose_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract normalized pose features (scale/translation invariant).
    """
    # Normalize to root joint (pelvis), scale by torso height
    pass

def subcluster_by_pose(identity_cluster_dir: Path, output_dir: Path):
    """
    Subdivide an identity cluster into pose/view buckets.
    """
    # 1. Estimate poses
    # 2. Classify views
    # 3. Extract features
    # 4. UMAP + HDBSCAN or KMeans
    # 5. Save to pose_* subdirectories
    pass
```

---

## Stage 5: VLM Captioning

**Purpose:** Generate high-quality, schema-guided captions emphasizing 3D rendering characteristics.

**Location:** `scripts/generic/training/vlm_caption_generator.py`

**Status:** 🚧 To be implemented (Qwen2-VL integration)

**Caption Schema:**

```json
{
  "character": "character_name",
  "identity": "distinctive features (hair, clothing, accessories)",
  "pose": "body pose description (standing, running, sitting)",
  "view": "camera view (front view, three-quarter, profile)",
  "expression": "facial expression (smiling, neutral, surprised)",
  "materials": {
    "skin": "smooth 3d skin shader, subsurface scattering",
    "clothing": "fabric material type (cotton, denim, leather)",
    "hair": "hair rendering style (strand-based, smooth)"
  },
  "lighting": {
    "key": "primary light direction and color",
    "fill": "fill light softness",
    "rim": "rim/back light presence"
  },
  "camera": {
    "shot_type": "close-up | medium shot | full body",
    "angle": "eye-level | low-angle | high-angle",
    "depth_of_field": "sharp | shallow DoF with bokeh"
  },
  "background": "background description or 'transparent'",
  "final_caption": "complete natural language caption for training"
}
```

**Implementation:**

```python
# scripts/generic/training/vlm_caption_generator.py

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch
from PIL import Image
import json

class VLMCaptioner:
    """
    Schema-guided caption generation using Qwen2-VL.
    """

    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_caption(self, image_path: Path, character_name: str = None) -> dict:
        """
        Generate structured caption for a 3D character image.

        Args:
            image_path: Path to image
            character_name: Optional character name hint

        Returns:
            dict with schema fields
        """
        image = Image.open(image_path).convert('RGB')

        # Construct prompt with JSON schema guidance
        prompt = f"""
Analyze this 3D animated character image and provide a structured description.

Character name: {character_name if character_name else "unknown"}

Please output a JSON object with these fields:
- character: character name or description
- identity: distinctive visual features (hair style/color, clothing, accessories)
- pose: body pose (standing, walking, running, sitting, etc.)
- view: camera view (front view, three-quarter view, profile view, back view)
- expression: facial expression (smiling, neutral, surprised, angry, etc.)
- materials: object with skin, clothing, hair material descriptions (3D rendering terms)
- lighting: object with key, fill, rim light descriptions
- camera: object with shot_type (close-up/medium/full body), angle, depth_of_field
- background: background description or "transparent" if isolated
- final_caption: complete natural language caption suitable for image generation training

Focus on 3D rendering characteristics: materials, lighting, camera properties.
Use precise 3D terminology (e.g., "subsurface scattering", "PBR materials", "rim lighting").

JSON:
"""

        # Generate
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=False
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse JSON from response
        try:
            # Extract JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            caption_data = json.loads(json_str)
        except:
            # Fallback to simple caption
            caption_data = {
                'character': character_name or 'unknown',
                'final_caption': f"a 3d animated character, {character_name or 'character'}, 3d cgi render"
            }

        return caption_data

    def batch_caption_cluster(self, cluster_dir: Path, character_name: str = None):
        """
        Generate captions for all images in a character cluster.

        Saves captions as .txt files alongside images.
        """
        images = sorted(cluster_dir.glob('*.png'))

        for img_path in tqdm(images, desc=f"Captioning {character_name or cluster_dir.name}"):
            caption_data = self.generate_caption(img_path, character_name)

            # Save caption as .txt file
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(caption_data['final_caption'])

            # Also save full structured data as .json
            json_path = img_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, indent=2, ensure_ascii=False)
```

**Usage:**

```bash
python scripts/generic/training/vlm_caption_generator.py \
  --cluster-dir .../clustered/character_0 \
  --character-name "luca" \
  --model qwen2_vl \
  --output-format both  # txt + json
```

---

## Stage 6: Dataset Assembly & Augmentation

**Purpose:** Organize captioned images into final training dataset with metadata and optional augmentation.

**Location:** `scripts/generic/training/dataset_assembler.py`

**Status:** 🚧 To be implemented

**Tasks:**

1. **Collect all images + captions** from character cluster
2. **Quality filtering:**
   - Remove blurry images (Laplacian variance)
   - Remove duplicates (pHash)
   - Remove low-resolution images
3. **Augmentation** (3D-appropriate only):
   - Brightness/contrast adjustment (±10%)
   - Slight color temperature shift (±5%)
   - **NO horizontal flip** (breaks asymmetric features)
   - **NO color jitter** (breaks PBR materials)
4. **Caption refinement:**
   - Add character-specific prefix
   - Ensure token length 40-77
   - Add style tags (pixar style, 3d cgi, smooth shading)
5. **Train/val split by scene** (not random)
6. **Export metadata:**
   - `metadata.json` with stats
   - `file_list.txt` for training script

**Implementation:**

```python
# scripts/generic/training/dataset_assembler.py

class DatasetAssembler:
    """
    Assemble final training dataset from captioned clusters.
    """

    def __init__(self, config):
        self.config = config

    def quality_filter(self, image_paths: List[Path]) -> List[Path]:
        """
        Filter out low-quality images.
        """
        filtered = []

        for img_path in tqdm(image_paths, desc="Quality filtering"):
            img = cv2.imread(str(img_path))

            # Blur detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < self.config['blur_threshold']:
                continue  # Too blurry

            # Resolution check
            h, w = img.shape[:2]
            if h < self.config['min_resolution'] or w < self.config['min_resolution']:
                continue

            filtered.append(img_path)

        return filtered

    def deduplicate(self, image_paths: List[Path]) -> List[Path]:
        """
        Remove near-duplicate images using perceptual hashing.
        """
        from imagehash import phash

        hashes = {}
        unique = []

        for img_path in tqdm(image_paths, desc="Deduplicating"):
            img = Image.open(img_path)
            hash_val = phash(img)

            # Check if similar hash exists
            is_duplicate = False
            for existing_hash in hashes.keys():
                if hash_val - existing_hash < self.config['phash_threshold']:
                    is_duplicate = True
                    break

            if not is_duplicate:
                hashes[hash_val] = img_path
                unique.append(img_path)

        print(f"Removed {len(image_paths) - len(unique)} duplicates")
        return unique

    def augment_dataset(self, image_paths: List[Path], output_dir: Path):
        """
        Apply 3D-appropriate augmentation.
        """
        from PIL import ImageEnhance

        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(image_paths, desc="Augmenting"):
            img = Image.open(img_path)

            # Original
            shutil.copy2(img_path, output_dir / img_path.name)

            # Brightness variants (±10%)
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(1.1)
            bright_img.save(output_dir / f"{img_path.stem}_bright{img_path.suffix}")

            dark_img = enhancer.enhance(0.9)
            dark_img.save(output_dir / f"{img_path.stem}_dark{img_path.suffix}")

            # Contrast variant (+5%)
            enhancer = ImageEnhance.Contrast(img)
            contrast_img = enhancer.enhance(1.05)
            contrast_img.save(output_dir / f"{img_path.stem}_contrast{img_path.suffix}")

    def refine_captions(self, caption_files: List[Path], character_name: str, prefix: str):
        """
        Add character-specific prefix and ensure token length constraints.
        """
        for caption_path in caption_files:
            with open(caption_path, 'r', encoding='utf-8') as f:
                original_caption = f.read().strip()

            # Add prefix
            refined_caption = f"{prefix}, {character_name}, {original_caption}"

            # Ensure token length (approximate: 1 token ≈ 4 chars)
            if len(refined_caption) > 300:  # ~75 tokens
                refined_caption = refined_caption[:300].rsplit(' ', 1)[0]  # Cut at word boundary

            # Save
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(refined_caption)

    def split_train_val(self, image_paths: List[Path], val_ratio: float = 0.1):
        """
        Split dataset by scene (not random) to avoid leakage.

        Assumes images are named with frame numbers: frame_0001.png, frame_0002.png, etc.
        """
        # Group by scene (consecutive frame ranges)
        scenes = []
        current_scene = []
        prev_frame_num = -1

        for img_path in sorted(image_paths):
            # Extract frame number
            frame_num = int(img_path.stem.split('_')[-1])

            if prev_frame_num == -1 or frame_num == prev_frame_num + 1:
                current_scene.append(img_path)
            else:
                # New scene
                scenes.append(current_scene)
                current_scene = [img_path]

            prev_frame_num = frame_num

        if current_scene:
            scenes.append(current_scene)

        # Split scenes into train/val
        num_val_scenes = max(1, int(len(scenes) * val_ratio))
        val_scenes = scenes[:num_val_scenes]
        train_scenes = scenes[num_val_scenes:]

        train_paths = [img for scene in train_scenes for img in scene]
        val_paths = [img for scene in val_scenes for img in scene]

        return train_paths, val_paths

    def export_metadata(self, output_dir: Path, stats: dict):
        """
        Export dataset metadata.
        """
        metadata = {
            'character': stats['character'],
            'total_images': stats['total_images'],
            'train_images': stats['train_images'],
            'val_images': stats['val_images'],
            'augmentation': stats['augmentation'],
            'caption_prefix': stats['caption_prefix'],
            'created_at': datetime.now().isoformat()
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
```

---

## Stage 7: Quality Validation & Export

**Purpose:** Final QA before training to ensure dataset quality.

**Checks:**

1. **All images have captions** (.png → .txt pairs)
2. **No corrupted images** (can open with PIL)
3. **Caption length distribution** (warn if >10% too long/short)
4. **Resolution distribution** (warn if many low-res)
5. **Perceptual diversity** (ensure not all similar poses/views)

**Export formats:**

- **Kohya_ss format:**
  ```
  dataset/
  ├── images/
  │   ├── 001_character_name.png
  │   ├── 002_character_name.png
  │   └── ...
  ├── captions/
  │   ├── 001_character_name.txt
  │   ├── 002_character_name.txt
  │   └── ...
  └── metadata.json
  ```

- **HuggingFace Dataset format** (optional):
  ```python
  from datasets import Dataset, Features, Image, Value

  def export_hf_dataset(image_paths, captions, output_dir):
      dataset = Dataset.from_dict({
          'image': [str(p) for p in image_paths],
          'caption': captions
      })
      dataset.save_to_disk(output_dir)
  ```

---

## Integrated Pipeline Script

**Location:** `scripts/pipelines/post_segmentation_pipeline.py`

**Purpose:** Run all stages sequentially with progress tracking.

```python
#!/usr/bin/env python3
"""
Complete post-segmentation pipeline:
  Instance prefilter → Identity clustering → Interactive review
  → (Optional pose subclustering) → VLM captioning → Dataset assembly
"""

import argparse
from pathlib import Path
import subprocess
import json

def run_stage(script_path: str, args: list):
    """Run a pipeline stage script with arguments."""
    cmd = ['python', script_path] + args
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description='Post-segmentation pipeline orchestrator')
    parser.add_argument('--input-dir', type=Path, required=True,
                       help='Directory with SAM2 segmented instances')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Final dataset output directory')
    parser.add_argument('--project-name', type=str, required=True,
                       help='Project name (e.g., luca, turning_red)')
    parser.add_argument('--skip-review', action='store_true',
                       help='Skip interactive cluster review')
    parser.add_argument('--enable-pose-subclustering', action='store_true',
                       help='Enable pose/view subclustering')

    args = parser.parse_args()

    # Create intermediate directories
    work_dir = args.output_dir / args.project_name / 'pipeline_temp'
    work_dir.mkdir(parents=True, exist_ok=True)

    filtered_dir = work_dir / 'filtered_instances'
    clustered_dir = work_dir / 'identity_clusters'
    reviewed_dir = work_dir / 'reviewed_clusters'
    captioned_dir = work_dir / 'captioned'
    final_dir = args.output_dir / args.project_name / 'training_data'

    # Stage 1: Instance prefiltering
    run_stage('scripts/generic/clustering/instance_prefilter.py', [
        '--input-dir', str(args.input_dir),
        '--output-dir', str(filtered_dir),
        '--mode', 'balanced',
        '--enable-semantic'
    ])

    # Stage 2: Face identity clustering
    run_stage('scripts/generic/clustering/face_identity_clustering.py', [
        '--input-dir', str(filtered_dir),
        '--output-dir', str(clustered_dir),
        '--config', 'configs/stages/clustering/face_identity_clustering.yaml'
    ])

    # Stage 3: Interactive review (optional)
    if not args.skip_review:
        print("\n🌐 Launching interactive cluster review UI...")
        print("   Open http://localhost:5000 in your browser")
        print("   Review clusters, rename, merge, move instances")
        print("   Press Ctrl+C when done to continue pipeline\n")

        run_stage('scripts/generic/clustering/web_cluster_review.py', [
            '--cluster-dir', str(clustered_dir)
        ])

        # After review, results are saved back to clustered_dir
        reviewed_dir = clustered_dir
    else:
        reviewed_dir = clustered_dir

    # Stage 4: Pose subclustering (optional)
    if args.enable_pose_subclustering:
        for character_dir in reviewed_dir.iterdir():
            if character_dir.is_dir() and character_dir.name != 'noise':
                run_stage('scripts/generic/clustering/pose_subclustering.py', [
                    '--input-dir', str(character_dir),
                    '--output-dir', str(character_dir / 'pose_subclusters')
                ])

    # Stage 5: VLM captioning
    for character_dir in reviewed_dir.iterdir():
        if character_dir.is_dir() and character_dir.name != 'noise':
            character_name = character_dir.name.replace('character_', '')

            run_stage('scripts/generic/training/vlm_caption_generator.py', [
                '--cluster-dir', str(character_dir),
                '--character-name', character_name,
                '--model', 'qwen2_vl',
                '--output-format', 'both'
            ])

    # Stage 6: Dataset assembly
    for character_dir in reviewed_dir.iterdir():
        if character_dir.is_dir() and character_dir.name != 'noise':
            character_name = character_dir.name.replace('character_', '')
            character_output = final_dir / character_name

            run_stage('scripts/generic/training/dataset_assembler.py', [
                '--input-dir', str(character_dir),
                '--output-dir', str(character_output),
                '--character-name', character_name,
                '--augment',
                '--target-size', '400'
            ])

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!")
    print(f"   Final datasets saved to: {final_dir}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
```

**Usage:**

```bash
python scripts/pipelines/post_segmentation_pipeline.py \
  --input-dir /mnt/data/datasets/3d-anime/luca/segmented/characters \
  --output-dir /mnt/data/datasets/3d-anime \
  --project-name luca \
  --enable-pose-subclustering
```

---

## Dependencies

**New requirements to add to `requirements/clustering.txt`:**

```txt
# Face detection & recognition
retinaface-pytorch
facenet-pytorch
mtcnn

# Clustering
hdbscan
umap-learn

# VLM captioning
transformers>=4.35.0
accelerate
bitsandbytes  # For quantization

# Image processing
imagehash
opencv-python
pillow

# Web UI
flask
flask-cors

# Utilities
tqdm
pyyaml
```

---

## Testing Plan

**Unit tests:**

```python
# tests/test_face_clustering.py

def test_face_detection():
    """Test face detection on sample image."""
    pass

def test_embedding_extraction():
    """Test face embedding extraction."""
    pass

def test_identity_clustering():
    """Test HDBSCAN clustering on synthetic embeddings."""
    pass
```

**Integration test:**

```bash
# Process a small test dataset end-to-end
bash tests/integration/test_post_segmentation_pipeline.sh
```

---

## Rollout Plan

**Week 1: Core Implementation**
- ✅ Day 1-2: Implement `face_identity_clustering.py`
- ✅ Day 3-4: Implement web-based cluster review UI
- ✅ Day 5: Test on small dataset (50 instances)

**Week 2: VLM Integration**
- ⏳ Day 1-2: Implement `vlm_caption_generator.py` with Qwen2-VL
- ⏳ Day 3: Create caption templates and schema
- ⏳ Day 4-5: Test captioning quality, tune prompts

**Week 3: Dataset Assembly & Pipeline Integration**
- ⏳ Day 1-2: Implement `dataset_assembler.py`
- ⏳ Day 3: Implement `post_segmentation_pipeline.py` orchestrator
- ⏳ Day 4-5: End-to-end test on full project (Luca/Alberto)

**Week 4: Optimization & Documentation**
- ⏳ Day 1-2: Performance optimization (batch processing, GPU utilization)
- ⏳ Day 3-4: Write comprehensive usage guides
- ⏳ Day 5: Final QA and deployment

---

## Success Metrics

**Quality Metrics:**
- ✅ Face detection recall > 95% on character instances
- ✅ Identity clustering purity > 90%
- ✅ Caption quality score (CLIP similarity) > 0.75
- ✅ Training dataset diversity (perceptual hash spread)

**Efficiency Metrics:**
- ✅ Full pipeline runtime < 2 hours for 1000 instances
- ✅ Interactive review session < 30 minutes per project
- ✅ GPU utilization > 80% during VLM captioning

**Usability Metrics:**
- ✅ Zero-config pipeline for new projects
- ✅ Clear progress indicators and ETAs
- ✅ Intuitive web UI requiring minimal instructions

---

## Documentation Deliverables

1. **`docs/guides/POST_SEGMENTATION_PIPELINE.md`** - User guide
2. **`docs/api/FACE_CLUSTERING_API.md`** - API reference
3. **`docs/tutorials/MULTI_CHARACTER_CLUSTERING.md`** - Step-by-step tutorial
4. **`configs/stages/clustering/` and `configs/stages/captioning/`** - Config templates

---

## Next Steps

After implementing this plan, the pipeline will be:
1. **Fully automated** with optional human review
2. **Production-ready** for any 3D animation project
3. **Extensible** (easy to swap VLM models, add new quality filters)
4. **Well-documented** for future users

**Ready to begin implementation!** 🚀
