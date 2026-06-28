# Expression LoRA Script V2 - Improvement Plan

**Author:** LLMProvider Tooling
**Date:** 2025-11-20
**Purpose:** Plan and implement next-generation expression processing with state-of-the-art AI models

---

## Executive Summary

The current expression processing script (`prepare_expression_lora_data.py`) uses:
- **InsightFace** for face detection (87% success rate on Pixar characters)
- **CLIP ViT-L/14** for zero-shot expression classification
- **Basic intensity estimation** via simple heuristics
- **Perceptual hashing** for deduplication

**Key Problems Identified:**
1. InsightFace is CPU-only (ONNX Runtime issue) → slow processing
2. Single-model approach → limited accuracy on 3D animated faces
3. No specialized emotion recognition → relies on generic CLIP
4. Missing interactive review tools → manual QA is difficult
5. Limited face quality assessment → some low-quality faces pass through

**Improvement Goal:** Increase processing speed 3-5x and accuracy 15-20% through multi-model ensemble and GPU optimization.

---

## Improvement Strategy

### Phase 1: Multi-Face Detector Support ✅ **Priority: HIGH**

**Rationale:** InsightFace works well (87%) but MediaPipe has better generalization to animated faces.

**Implementation:**
```python
class MultiFaceDetector:
    """Ensemble face detector supporting multiple backends"""

    def __init__(self, methods=['insightface', 'mediapipe'], device='cuda'):
        self.methods = methods
        self.detectors = {}

    def detect_faces(self, image, mode='best'):
        """
        Modes:
        - 'best': Use primary detector, fallback to others if fails
        - 'ensemble': Combine results from all detectors via NMS
        - 'voting': Keep faces detected by majority
        """
```

**Detectors to Support:**
1. **InsightFace** (current default) - Best for realistic faces
   - Pros: High precision, robust landmarks
   - Cons: Domain gap to 3D animation, CPU-only issue

2. **MediaPipe Face Detection** (NEW) - Better generalization
   - Pros: Works well on animated faces, fast, GPU-accelerated
   - Cons: Fewer landmarks than InsightFace

3. **RetinaFace** (optional) - Alternative deep learning detector
   - Pros: State-of-the-art accuracy
   - Cons: Slower than MediaPipe

**Expected Improvement:** +5-10% face detection rate, 2x speed with MediaPipe

---

### Phase 2: Multi-Model Expression Classifier ✅ **Priority: CRITICAL**

**Rationale:** CLIP zero-shot is good but not specialized for emotions. Ensemble approach combines strengths.

**Architecture:**
```python
class EnsembleExpressionClassifier:
    """Multi-model expression classifier with weighted voting"""

    def __init__(self, models=['clip', 'emoca', 'hsemotion'], weights=[0.4, 0.4, 0.2]):
        self.models = models
        self.weights = weights

    def classify(self, face_crop):
        """
        Returns:
        - Primary emotion (weighted vote)
        - Confidence score (weighted average)
        - Per-model predictions (for debugging)
        - Intensity level (from EMOCA valence/arousal)
        """
```

**Models to Integrate:**

1. **CLIP ViT-L/14** (current) - Zero-shot baseline
   - Weight: 0.4
   - Strength: Generalization to 3D style
   - Weakness: Not specialized for emotions

2. **EMOCA** (NEW) **🔥 Recommended**
   - Paper: ECCV 2024, specializes in 3D face emotions
   - Features: 53 expression parameters, valence/arousal estimation
   - Weight: 0.4
   - Strength: Designed for 3D faces, intensity estimation
   - Integration: `pip install emoca`

3. **HSEmotion** (NEW) - Lightweight emotion CNN
   - Pre-trained on FER datasets
   - Weight: 0.2
   - Strength: Fast, good for basic 6 emotions
   - Integration: `pip install hsemotion`

4. **EmoNet** (optional) - Alternative to EMOCA
   - Similar capabilities, lighter weight
   - Good for resource-constrained scenarios

**Voting Strategy:**
```python
def weighted_vote(predictions, weights):
    """
    Weighted soft voting:
    1. Normalize each model's confidences to [0,1]
    2. Multiply by model weight
    3. Sum across models
    4. Argmax for final emotion
    """
    scores = defaultdict(float)
    for (emotion, conf), weight in zip(predictions, weights):
        scores[emotion] += conf * weight
    return max(scores.items(), key=lambda x: x[1])
```

**Expected Improvement:** +15-20% classification accuracy on 3D animated faces

---

### Phase 3: Advanced Intensity Estimation ✅ **Priority: HIGH**

**Current Approach:** Simple heuristic based on CLIP confidence
**Problem:** Doesn't capture true emotion intensity

**Improved Approach:**

1. **EMOCA-based Intensity** (valence + arousal dimensions)
   ```python
   valence = emoca_output['valence']  # -1 (negative) to +1 (positive)
   arousal = emoca_output['arousal']  # 0 (calm) to +1 (excited)

   intensity_score = np.sqrt(valence**2 + arousal**2)  # Euclidean distance from neutral

   # Map to levels
   if intensity_score < 0.3: level = 'subtle'
   elif intensity_score < 0.7: level = 'moderate'
   else: level = 'strong'
   ```

2. **Facial Action Unit (AU) Analysis**
   - Use OpenFace or PyFeat to extract AU intensities
   - AU12 (lip corner puller) → happiness intensity
   - AU04 (brow lowerer) → anger/sadness intensity

3. **Optical Flow Motion** (optional)
   - Analyze temporal dynamics for dynamic expressions
   - Higher motion → stronger expression

**Expected Improvement:** Better caption granularity, more balanced training data

---

### Phase 4: GPU Optimization & Batch Processing ✅ **Priority: CRITICAL**

**Current Issue:** InsightFace on CPU (ONNX Runtime), CLIP processes images one-by-one

**Optimizations:**

1. **Fix ONNX Runtime GPU** (already in progress)
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu==1.20.0
   ```

2. **Batch Processing for CLIP**
   ```python
   # Current: Loop through faces one by one
   for face in faces:
       result = classifier.classify(face)  # Individual forward pass

   # Improved: Batch processing
   batch_size = 32
   for i in range(0, len(faces), batch_size):
       batch = faces[i:i+batch_size]
       results = classifier.classify_batch(batch)  # Single batched forward pass
   ```
   Expected speedup: 3-5x

3. **Multi-GPU Support** (if available)
   ```python
   if torch.cuda.device_count() > 1:
       model = nn.DataParallel(model)
   ```

4. **Mixed Precision (FP16)**
   ```python
   from torch.cuda.amp import autocast

   with autocast():
       image_features = model.encode_image(images)
   ```
   Expected speedup: 1.5-2x, lower VRAM usage

**Expected Improvement:** 3-5x total processing speed

---

### Phase 5: Enhanced Quality Filtering ✅ **Priority: MEDIUM**

**Current Filters:**
- Detection confidence
- Expression confidence
- Blur score (Laplacian)

**Additional Filters:**

1. **Face Occlusion Detection**
   ```python
   def estimate_occlusion(face_landmarks):
       """
       Check if critical facial features are visible:
       - Eyes (landmarks 36-47)
       - Nose (landmarks 27-35)
       - Mouth (landmarks 48-67)

       Returns occlusion_ratio (0-1)
       """
   ```
   Threshold: occlusion_ratio < 0.3

2. **Lighting Quality Assessment**
   ```python
   def assess_lighting(face_crop):
       """
       Check for:
       - Over-exposure (histogram clipping)
       - Under-exposure (mean intensity < 50)
       - Even lighting (stddev within reasonable range)
       """
   ```

3. **Face Pose Estimation**
   ```python
   def estimate_pose(landmarks):
       """
       Compute pitch, yaw, roll
       Filter out extreme angles (e.g., |yaw| > 45°)
       """
   ```
   Keep: frontal to three-quarter views (|yaw| < 45°)

4. **Expression Ambiguity Detection**
   ```python
   def is_ambiguous(expression_scores):
       """
       If top-2 emotions have similar scores, mark as ambiguous
       """
       sorted_scores = sorted(expression_scores.values(), reverse=True)
       margin = sorted_scores[0] - sorted_scores[1]
       return margin < 0.15  # Ambiguous if top 2 within 15%
   ```

**Expected Improvement:** Higher dataset quality, better LoRA training results

---

### Phase 6: Interactive Review UI ✅ **Priority: HIGH**

**Problem:** Manual review of 1000+ expression images is tedious

**Solution:** Web-based interactive UI (following project guidelines)

**Features:**
1. **Grid View** - Thumbnail display of all faces per expression
2. **Quick Actions:**
   - Move to different expression folder
   - Mark as low-quality (delete)
   - Adjust intensity level
   - Add custom tags
3. **Filtering:**
   - By confidence score
   - By intensity level
   - By model agreement (ensemble consensus)
4. **Batch Operations:**
   - Select multiple images
   - Apply bulk actions
5. **Export:**
   - Save corrected classifications
   - Re-run final dataset assembly with corrections

**Technology Stack:**
```
Backend: Flask (lightweight Python web framework)
Frontend: HTML + Vanilla JS + CSS Grid
No dependencies: Runs in any browser without installation
```

**Implementation:**
```python
# scripts/generic/training/interactive_expression_review.py
from flask import Flask, render_template, request, jsonify
import json
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def index():
    """Render main review interface"""

@app.route('/api/expressions')
def get_expressions():
    """Return all expressions with metadata"""

@app.route('/api/move', methods=['POST'])
def move_image():
    """Move image to different expression folder"""

@app.route('/api/delete', methods=['POST'])
def delete_image():
    """Mark image for deletion"""

@app.route('/api/export')
def export_corrections():
    """Export all corrections as JSON"""
```

**Expected Improvement:** 50-70% faster manual review, better data quality

---

### Phase 7: Advanced Features (Future) ✅ **Priority: LOW**

1. **Temporal Expression Analysis**
   - Process video sequences to capture dynamic expressions
   - Use 3D-CNNs or transformer-based models (TimeSformer)

2. **Multi-task Learning**
   - Joint training for expression + intensity + AU
   - Single model with multiple heads

3. **Active Learning**
   - Identify low-confidence samples for human labeling
   - Iteratively improve classifier

4. **Zero-shot to Few-shot Transfer**
   - Fine-tune CLIP on small labeled dataset of 3D faces
   - Improve domain adaptation

---

## Implementation Roadmap

### Week 1: Core Improvements
- [x] Install GPU ONNX Runtime
- [ ] Implement `MultiFaceDetector` with MediaPipe support
- [ ] Implement `EnsembleExpressionClassifier` with EMOCA + HSEmotion
- [ ] Add batch processing for CLIP
- [ ] Test on Miguel (449 images) to verify improvements

### Week 2: Quality & UI
- [ ] Implement advanced quality filters (occlusion, lighting, pose)
- [ ] Create interactive review UI (Flask-based)
- [ ] Process all 6 characters with new pipeline
- [ ] Manual QA via interactive UI

### Week 3: Evaluation & Refinement
- [ ] Compare V1 vs V2 results (accuracy, speed, quality)
- [ ] Fine-tune ensemble weights based on validation data
- [ ] Document best practices in `docs/training/lora_types/02_EXPRESSION_LORA_DEEP_DIVE.md`

---

## Success Metrics

| Metric | Current (V1) | Target (V2) | Status |
|--------|-------------|-------------|--------|
| Face Detection Rate | 87% | 92-95% | Pending |
| Expression Accuracy | ~60% (estimated) | 75-80% | Pending |
| Processing Speed | 3-5 img/s (CPU) | 15-25 img/s (GPU) | In Progress |
| Manual Review Time | 2-3 hours | 30-45 min | Pending |
| Dataset Quality (subjective) | Good | Excellent | Pending |

---

## Technical Dependencies

### New Python Packages
```bash
# Face detection
pip install mediapipe  # GPU-accelerated face detection

# Expression recognition
pip install emoca  # 3D face emotion analysis (ECCV 2024)
pip install hsemotion  # Lightweight emotion CNN

# Alternative options
pip install emonet  # Alternative to EMOCA
pip install py-feat  # Facial Action Unit extraction

# Web UI
pip install flask  # Lightweight web framework
```

### Model Downloads
- EMOCA checkpoint: Auto-downloaded on first use
- HSEmotion weights: Auto-downloaded via pip
- MediaPipe models: Auto-downloaded on first use

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| EMOCA domain gap to 3D | Medium | High | Use ensemble with CLIP, adjust weights |
| GPU memory overflow | Low | Medium | Reduce batch size, use FP16 |
| Installation issues | Medium | Low | Provide detailed setup guide |
| Ensemble slower than single model | High | Medium | Optimize with batch processing, caching |

---

## Conclusion

The improved Expression LoRA script (V2) will provide:
- **Higher accuracy** through multi-model ensemble
- **Faster processing** via GPU optimization and batch processing
- **Better data quality** with advanced filtering
- **Easier QA workflow** with interactive review UI

This positions the pipeline as **state-of-the-art for 3D animated character expression recognition**, surpassing existing open-source solutions.

**Next Steps:**
1. Implement `MultiFaceDetector` class
2. Integrate EMOCA + HSEmotion
3. Create batch processing wrapper for CLIP
4. Build interactive review UI
5. Test and iterate
