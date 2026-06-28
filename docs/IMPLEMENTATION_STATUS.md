# Post-Segmentation Pipeline - Implementation Status

**Last Updated:** 2025-12-16

This document tracks implementation status for the post-SAM2 pipeline stages.

---

## Overview

**Pipeline Stages:**
```
SAM2 Segmentation ✅ (Complete)
  ↓
Instance Pre-filtering ✅ (Complete)
  ↓
Face Identity Clustering 🚧 (In Progress)
  ↓
Interactive Cluster Review 🚧 (In Progress)
  ↓
Pose/View Subclustering ⏳ (Planned)
  ↓
VLM Captioning ⏳ (Planned)
  ↓
Dataset Assembly ⏳ (Planned)
  ↓
Quality Validation ⏳ (Planned)
```

---

## Stage Status

### ✅ Stage 0: SAM2 Instance Segmentation

**Status:** Complete and production-ready

**Files:**
- ✅ `scripts/generic/segmentation/instance_segmentation.py`
- ✅ `configs/stages/segmentation/sam2.yaml`

**Recent Updates:**
- Added async I/O support (75%+ GPU utilization)
- Optimized prefetch buffer for faster processing
- Tested on multiple projects (Super Wings, Luca, etc.)

**Next Improvements:**
- Consider adding SAM2.1 support when released

---

### ✅ Stage 1: Instance Pre-filtering

**Status:** Complete and production-ready

**Files:**
- ✅ `scripts/generic/clustering/instance_prefilter.py`
- ✅ `configs/stages/clustering/instance_prefilter.yaml`

**Capabilities:**
- Semantic filtering (character vs. background)
- Heuristic filtering (size, aspect ratio, edge complexity)
- Face detection pre-check
- Removes 80-90% of background instances

**Performance:**
- ~500-1000 instances/second
- Reduces clustering time by 5-10x

**Next Improvements:**
- Add CLIP-based semantic filtering option
- Improve edge complexity metrics for cartoon characters

---

### 🚧 Stage 2: Face Identity Clustering

**Status:** In Progress (70% complete)

**Files:**
- 🚧 `scripts/generic/clustering/face_identity_clustering.py` (NEW - needs implementation)
- ✅ `configs/stages/clustering/face_identity_clustering.yaml` (Created)
- ⚠️ `scripts/generic/clustering/clip_character_clustering.py` (OLD - visual similarity only)

**What's Done:**
- ✅ Config file created with HDBSCAN parameters
- ✅ Architecture design complete
- ✅ Dependencies identified (RetinaFace, ArcFace)

**What's Needed:**
- ⏳ Implement `FaceIdentityClusterer` class
- ⏳ Integrate RetinaFace detection
- ⏳ Integrate ArcFace embeddings
- ⏳ Test on multi-character scenes
- ⏳ Add cluster naming prompt

**Blockers:**
- None

**Target Completion:** End of Week 1

**Testing Plan:**
```bash
# Test on Super Wings dataset (8 characters)
python scripts/generic/clustering/face_identity_clustering.py \
  --input-dir /mnt/data/datasets/general/super-wings/filtered_instances \
  --output-dir /mnt/data/datasets/general/super-wings/identity_clusters \
  --config configs/stages/clustering/face_identity_clustering.yaml

# Verify cluster purity (should be >90%)
python scripts/evaluation/cluster_purity_metrics.py \
  --cluster-dir /mnt/data/datasets/general/super-wings/identity_clusters
```

---

### 🚧 Stage 3: Interactive Cluster Review

**Status:** In Progress (40% complete)

**Files:**
- ✅ `scripts/generic/clustering/interactive_character_selector.py` (Basic CLI version exists)
- ⏳ `scripts/generic/clustering/web_cluster_review.py` (NEW - needs implementation)
- ⏳ `scripts/generic/clustering/interactive_ui/cluster_review.html` (NEW - needs creation)
- ⏳ `scripts/generic/clustering/interactive_ui/cluster_review.js` (NEW - needs creation)

**What's Done:**
- ✅ Basic CLI review tool works
- ✅ Architecture design for web UI complete

**What's Needed:**
- ⏳ Implement Flask backend (`web_cluster_review.py`)
- ⏳ Create HTML/CSS/JS frontend
- ⏳ Add drag-and-drop functionality
- ⏳ Add merge/split cluster operations
- ⏳ Add keyboard shortcuts for efficiency
- ⏳ Test on real clustering output

**UI Features Required:**
- [ ] Grid view of all clusters
- [ ] Thumbnail preview (3x3 or 4x4 grid per cluster)
- [ ] Drag-and-drop instances between clusters
- [ ] Rename cluster (assign character name)
- [ ] Merge clusters (combine two identities)
- [ ] Split cluster (separate by pose/view)
- [ ] Delete/mark noise instances
- [ ] Save/export corrections

**Target Completion:** End of Week 1

**Demo Data:**
```bash
# Generate test clustering for UI development
python scripts/generic/clustering/face_identity_clustering.py \
  --input-dir tests/fixtures/multi_character_instances \
  --output-dir tests/fixtures/test_clusters

# Launch web UI for testing
python scripts/generic/clustering/web_cluster_review.py \
  --cluster-dir tests/fixtures/test_clusters \
  --port 5000
```

---

### ⏳ Stage 4: Pose/View Subclustering

**Status:** Planned (0% complete)

**Files:**
- ⏳ `scripts/generic/clustering/pose_subclustering.py` (NEW - needs implementation)
- ⏳ `configs/stages/clustering/pose_subclustering.yaml` (NEW - needs creation)

**What's Needed:**
- ⏳ Integrate RTM-Pose for keypoint detection
- ⏳ Implement view classification (front/3-quarter/profile/back)
- ⏳ Implement pose normalization
- ⏳ Implement UMAP + HDBSCAN subclustering
- ⏳ Test on single-character clusters
- ⏳ Validate pose bucket balance

**Dependencies:**
- `mmpose` library for RTM-Pose
- `mmcv` for model loading

**Target Completion:** End of Week 3

**Expected Benefits:**
- Balanced angle distribution (no over-representation of front views)
- Better caption consistency within pose buckets
- Improved LoRA generalization

**Testing Criteria:**
- Each pose bucket should have similar size (±20%)
- Intra-bucket visual similarity > 0.85 (CLIP)
- Inter-bucket visual similarity < 0.70

---

### ⏳ Stage 5: VLM Captioning

**Status:** Planned (10% complete)

**Files:**
- ⏳ `scripts/generic/training/vlm_caption_generator.py` (NEW - needs implementation)
- ✅ `configs/stages/captioning/vlm_3d_character_captioning.yaml` (Created)
- ⚠️ `scripts/generic/training/qwen_caption_generator.py` (OLD - basic version exists)

**What's Done:**
- ✅ Config file created with schema definition
- ✅ Prompt template designed
- ✅ Caption refinement rules defined

**What's Needed:**
- ⏳ Implement `VLMCaptioner` class
- ⏳ Integrate Qwen2-VL model
- ⏳ Implement JSON schema enforcement
- ⏳ Implement caption refinement (prefix, token length)
- ⏳ Add quality control checks
- ⏳ Batch processing optimization
- ⏳ Test caption quality (CLIP score, consistency)

**Dependencies:**
- `transformers>=4.35.0` (Qwen2-VL support)
- `bitsandbytes` (4-bit quantization)
- `accelerate` (model loading)

**Target Completion:** End of Week 2

**Caption Quality Metrics:**
```python
# Evaluate caption quality
python scripts/evaluation/caption_quality_metrics.py \
  --image-dir /path/to/captioned_images \
  --metrics clip_score,bleu,diversity

# Expected results:
# - CLIP score > 0.75
# - Caption diversity > 0.8 (unique trigrams)
# - Token length: 40-77 (99% of captions)
```

---

### ⏳ Stage 6: Dataset Assembly

**Status:** Planned (0% complete)

**Files:**
- ⏳ `scripts/generic/training/dataset_assembler.py` (NEW - needs implementation)
- ⏳ `configs/stages/dataset/assembly.yaml` (NEW - needs creation)

**What's Needed:**
- ⏳ Implement quality filtering (blur, duplicates, resolution)
- ⏳ Implement 3D-appropriate augmentation
- ⏳ Implement caption refinement
- ⏳ Implement scene-based train/val split
- ⏳ Export to Kohya_ss format
- ⏳ Generate metadata

**Target Completion:** End of Week 3

**Expected Output Structure:**
```
training_data/character_name/
├── images/
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── captions/
│   ├── 001.txt
│   ├── 002.txt
│   └── ...
├── metadata.json
└── validation/
    ├── images/
    └── captions/
```

---

### ⏳ Stage 7: Pipeline Orchestrator

**Status:** Planned (0% complete)

**Files:**
- ⏳ `scripts/pipelines/post_segmentation_pipeline.py` (NEW - needs implementation)

**What's Needed:**
- ⏳ Implement stage orchestration
- ⏳ Add progress tracking (% complete, ETA)
- ⏳ Add resume capability (checkpoint intermediate results)
- ⏳ Add error handling and recovery
- ⏳ Add logging and reporting
- ⏳ Create CLI with rich output

**Target Completion:** End of Week 3

**Example Usage:**
```bash
python scripts/pipelines/post_segmentation_pipeline.py \
  --input-dir /mnt/data/datasets/3d-anime/luca/segmented/characters \
  --output-dir /mnt/data/datasets/3d-anime/luca/training_data \
  --project-name luca \
  --enable-pose-subclustering \
  --auto-caption \
  --review-clusters  # Launch web UI for review
```

---

## Priority Implementation Order

**Week 1:**
1. ✅ Create config files (DONE)
2. 🚧 Implement `face_identity_clustering.py` (IN PROGRESS)
3. 🚧 Implement web cluster review UI (IN PROGRESS)
4. ⏳ Test on Super Wings dataset

**Week 2:**
5. ⏳ Implement `vlm_caption_generator.py`
6. ⏳ Test caption quality on sample clusters
7. ⏳ Tune prompts and schemas

**Week 3:**
8. ⏳ Implement `pose_subclustering.py` (optional but recommended)
9. ⏳ Implement `dataset_assembler.py`
10. ⏳ Implement `post_segmentation_pipeline.py` orchestrator

**Week 4:**
11. ⏳ End-to-end testing on full project
12. ⏳ Performance optimization
13. ⏳ Documentation and guides

---

## Dependencies Installation

**New requirements needed:**

```bash
# Face detection & recognition
pip install retinaface-pytorch facenet-pytorch mtcnn

# Clustering
pip install hdbscan umap-learn

# VLM captioning
pip install transformers>=4.35.0 accelerate bitsandbytes

# Web UI
pip install flask flask-cors

# Image processing
pip install imagehash opencv-python pillow

# Pose estimation (for Stage 4)
pip install mmpose mmcv
```

**Update `requirements/clustering.txt`:**
```bash
# Add to requirements/clustering.txt
retinaface-pytorch
facenet-pytorch
mtcnn
hdbscan
umap-learn
imagehash
flask
flask-cors
```

**Update `requirements/training.txt`:**
```bash
# Add to requirements/training.txt
transformers>=4.35.0
accelerate
bitsandbytes
```

---

## Testing Strategy

**Unit Tests:**
```bash
# Test face detection
pytest tests/unit/test_face_detection.py

# Test identity clustering
pytest tests/unit/test_identity_clustering.py

# Test VLM captioning
pytest tests/unit/test_vlm_captioner.py
```

**Integration Tests:**
```bash
# Test complete pipeline on small dataset
bash tests/integration/test_post_segmentation_pipeline.sh
```

**End-to-End Test:**
```bash
# Process Super Wings dataset (known ground truth)
python scripts/pipelines/post_segmentation_pipeline.py \
  --input-dir /mnt/data/datasets/general/super-wings/segmented/characters \
  --output-dir /mnt/data/datasets/general/super-wings/training_data \
  --project-name super_wings

# Validate results
python scripts/evaluation/validate_training_dataset.py \
  --dataset-dir /mnt/data/datasets/general/super-wings/training_data
```

---

## Known Issues

1. **Face detection failures on extreme angles**
   - RetinaFace struggles with >75° yaw
   - **Solution:** Add back-view detection fallback or use pose-based clustering

2. **VLM hallucination on accessories**
   - Qwen2-VL sometimes invents non-existent details
   - **Solution:** Add consistency checks, reject hallucinated captions

3. **Memory usage during VLM batch processing**
   - 7B VLM model uses ~14GB VRAM even with 4-bit quantization
   - **Solution:** Reduce batch size to 2-4, enable gradient checkpointing

---

## Success Metrics

**Stage 2: Face Clustering**
- ✅ Face detection recall > 95%
- ✅ Identity clustering purity > 90%
- ✅ Processing speed > 100 instances/minute

**Stage 5: VLM Captioning**
- ✅ CLIP score > 0.75
- ✅ Caption diversity > 0.8
- ✅ Token length compliance > 99%

**Stage 6: Dataset Assembly**
- ✅ Duplicate removal > 95%
- ✅ Quality filtering recall > 90%
- ✅ Train/val split leak rate < 1%

**Overall Pipeline**
- ✅ End-to-end runtime < 2 hours for 1000 instances
- ✅ Zero-config for new projects
- ✅ Interactive review session < 30 minutes

---

## Next Actions

**Immediate (This Week):**
1. Start implementing `face_identity_clustering.py`
2. Create basic web UI for cluster review
3. Test face detection on Super Wings dataset

**Short-term (Next 2 Weeks):**
4. Implement VLM captioning pipeline
5. Test caption quality on sample clusters
6. Begin dataset assembly implementation

**Long-term (Next 4 Weeks):**
7. Complete all stages
8. End-to-end testing on multiple projects
9. Performance optimization
10. Documentation and user guides

---

## Notes

- 所有實作都應該遵循 3D 動畫專用的參數設定
- 確保每個階段都有清晰的輸入輸出格式契約
- 優先考慮批次處理效能和 GPU 利用率
- 保持程式碼模組化，方便後續替換模型

**Ready to start implementation!** 🚀
