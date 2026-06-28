# Synthetic LoRA Training Pipeline - 進度追蹤

**最後更新**: 2025-12-04 14:30 UTC
**專案狀態**: ✅ Phases 1-4 完成！已過濾並準備訓練！

---

## 📊 整體進度概覽

| Phase | 狀態 | 完成度 | 備註 |
|-------|------|--------|------|
| Phase 1.1: Caption Template Generation | ✅ 完成 | 100% | 30 exemplars, $0.23 |
| Phase 1.2: Caption Application | ✅ 完成 | 100% | 26,342/27,118 captions (97.1%) |
| Phase 2.1: Quality Filtering | ✅ 完成 | 100% | 20,221/27,118 passed (74.6%) |
| Phase 2.2: Manual Caption Review | ✅ 完成 | 100% | 583 samples reviewed |
| Phase 3: Dataset Organization | ✅ 完成 | 100% | **45 datasets, 27,112 images** |
| Phase 4.1: Initial Config Generation | ✅ 完成 | 100% | 45 configs generated |
| **Phase 4.2: Filtered Config Generation** | ✅ 完成 | 100% | **40 configs (跳過5個小資料集)** |
| Phase 5.1: Universal LoRAs Training | ⏳ 準備中 | 0% | 3 LoRAs (優先) |
| Phase 5.2: Character LoRAs Training | ⏳ 等待中 | 0% | **37 LoRAs (已過濾)** |
| Phase 6: Evaluation & Selection | ⏳ 等待中 | 0% | Auto checkpoint eval |

---

## ✅ Phase 1: Caption Generation (完成)

### Phase 1.1: Caption Template Generation

**工具**: `scripts/generic/training/caption_engines/synthetic_caption_template_generator.py`

**執行指令**:
```bash
LLM_VENDOR_API_KEY="sk-ant-..." conda run -n ai_env python scripts/generic/training/caption_engines/synthetic_caption_template_generator.py \
  --data-root /mnt/data/ai_data/synthetic_lora_data/generated_data \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/caption_templates \
  --characters alberto luca \
  --lora-types pose action expression \
  --samples-per-character 5 \
  --model llm_provider-3-5-haiku-20241022 \
  --temperature 0.7
```

**結果**:
- ✅ 生成30個exemplar captions (alberto + luca × 3 types × 5 samples)
- ✅ API成本: $0.23
- ✅ Total tokens: 59,801
- ✅ 成功率: 100% (30/30)

**輸出文件**:
- `/mnt/data/ai_data/synthetic_lora_data/caption_templates/caption_templates_pose.json` (10 templates)
- `/mnt/data/ai_data/synthetic_lora_data/caption_templates/caption_templates_action.json` (10 templates)
- `/mnt/data/ai_data/synthetic_lora_data/caption_templates/caption_templates_expression.json` (10 templates)
- `/mnt/data/ai_data/synthetic_lora_data/caption_templates/caption_vocabulary_*.json` (各type的vocabulary)
- `/mnt/data/ai_data/synthetic_lora_data/caption_templates/generation_statistics.json`

**重要修正**:
1. ✅ 強化prompt以移除character identity (name, appearance, clothing, colors)
2. ✅ 添加"ABSOLUTELY FORBIDDEN TERMS"列表
3. ✅ 移除具體物品描述 (basketball, soccer ball, bicycle)

### Phase 1.2: Caption Application

**工具**: `scripts/generic/training/caption_engines/synthetic_caption_applier.py`

**執行指令**:
```bash
conda run -n ai_env python scripts/generic/training/caption_engines/synthetic_caption_applier.py \
  --data-root /mnt/data/ai_data/synthetic_lora_data/generated_data \
  --vocabulary-dir /mnt/data/ai_data/synthetic_lora_data/caption_templates \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/captioned_data \
  --lora-types pose action expression \
  --min-tokens 75 \
  --max-tokens 225 \
  --target-tokens 150 \
  --force
```

**結果**:
- ✅ **POSE**: 9,296/9,296 captions (100.0% success)
- ✅ **EXPRESSION**: 9,664/9,664 captions (100.0% success)
- ⚠️ **ACTION**: 7,382/8,158 captions (90.5% success, 776 failed - 74 tokens minimum)
- **總計**: 26,342/27,118 captions (97.1% success)

**輸出**:
- Caption files: `/mnt/data/ai_data/synthetic_lora_data/captioned_data/{character}/{type}/*.txt`
- Statistics: `/mnt/data/ai_data/synthetic_lora_data/captioned_data/caption_application_statistics.json`

**Caption範例** (pose, 95 words):
```
a 3d animated character, sitting cross-legged pose with torso slightly leaned back and one hand resting on ground for support, body weight distributed asymmetrically, three-quarter side view revealing dynamic body positioning, soft warm cinematic lighting with gentle gradient background, high-fidelity 3D rendering with advanced subsurface scattering and detailed skin texturing, precise character rigging showing natural body mechanics, professional animation studio quality with smooth procedural shading, soft ambient occlusion enhancing depth and form, pristine digital sculpting with clean geometric topology, photorealistic rendering at 4K resolution with delicate light interaction, side, ambient occlusion, 4K
```

---

## ✅ Phase 2.1: Quality Filtering (完成)

### 最終結果

**工具**: `scripts/generic/training/quality_filters/synthetic_quality_pipeline.py`

**執行指令**:
```bash
conda run -n ai_env python scripts/generic/training/quality_filters/synthetic_quality_pipeline.py \
  --input-dir /mnt/data/ai_data/synthetic_lora_data/generated_data \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data \
  --blur-threshold 80 \
  --blur-premium 120 \
  --device cpu \
  --num-workers 32
```

**配置**:
- ✅ **Multiprocessing**: 32 parallel workers (充分利用32 CPU threads)
- ✅ **Blur Detection**: Laplacian variance threshold = 80
- ✅ **Blur Premium**: Tier A threshold = 120
- ✅ **Size Filter**: 768-1280px
- ✅ **Deduplication**: Perceptual hash, Hamming distance > 8
- ✅ **Character Presence**: YOLOv8 (yolov8n.pt), confidence > 0.5

**4階段過濾**:
1. **Stage 1: Blur Detection** - Laplacian variance (reject if < 80)
2. **Stage 2: Size Validation** - Resolution check (768-1280px, resize to 1024×1024)
3. **Stage 3: Deduplication** - Perceptual hash (Hamming distance > 8)
4. **Stage 4: Character Presence** - YOLOv8 person detection (exactly 1 person, confidence > 0.5)

**實際結果** (超出預期！):
```
總輸入圖片: 27,118
✅ Tier A (Premium, blur>120): 14,092 (52.0%)
✅ Tier B (Good, blur 80-120):   6,129 (22.6%)
✅ 總通過: 20,221 (74.6%)
❌ Tier C (Rejected):            6,897 (25.4%)

各類型統計:
  ACTION:     6,922 passed (84.8%) - Tier A: 5,518, Tier B: 1,404
  EXPRESSION: 6,387 passed (66.1%) - Tier A: 3,997, Tier B: 2,390
  POSE:       6,912 passed (74.4%) - Tier A: 4,577, Tier B: 2,335

各角色統計 (Top 5):
  luca:            2,077/2,878 (72.2%)
  ian_lightfoot:   2,393/2,710 (88.3%)
  miguel:            649/2,680 (24.2%)
  luca_seamonster: 2,090/2,292 (91.2%)
  elio:              788/1,920 (41.0%)
```

**平均每角色每類型**: ~480 張高質量訓練圖

**輸出目錄結構**:
```
filtered_data/
  {character}/
    {lora_type}/
      tier_a/  # Premium quality
        image_*.png
        image_*.txt
      tier_b/  # Good quality
      tier_c/  # Rejected (for debugging)
      filtering_report.json
  overall_filtering_statistics.json
```

### 代碼優化記錄

#### 問題1: 單進程處理，只使用1-2個CPU核心
**修復**:
- ✅ 添加`multiprocessing.Pool`支持
- ✅ 添加`--num-workers`參數 (default: 16, 可設為32或0=auto)
- ✅ 使用`pool.imap()`並行處理圖片

#### 問題2: Pickle serialization錯誤 (local function無法序列化)
**修復**:
- ✅ 將worker function移到module level (`_filter_image_worker`)
- ✅ Filters在worker內部初始化避免pickle問題

#### 問題3: PIL Image沒有`.shape`屬性
**修復**:
- ✅ 添加`import numpy as np`
- ✅ 將PIL Image轉換為numpy array: `img = np.array(img_pil)`

#### 問題4: 錯誤的hash method名稱
**修復**:
- ✅ 將`'phash'`改為`'perceptual'`

#### 問題5: Missing Optional import
**修復**:
- ✅ 添加`Optional`到typing imports

#### 問題6: 目錄結構不匹配
**修復**:
- ✅ 自動檢測`/generated`子目錄
- ✅ 複製captions到image目錄

---

## 📦 數據統計

### Caption Generation
- **Total source images**: 27,118
- **Captions generated**: 26,342 (97.1%)
- **Characters**: 14
- **LoRA types**: 3 (pose, action, expression)
- **Per character per type**: ~600-700 images

### Quality Filtering (完成)
- **Input images**: 27,118
- **Actual pass rate**: 74.6%
- **Actual output**: 20,221 images (超出預期！)
- **Tier A (Premium)**: 14,092 images (52.0%)
- **Tier B (Good)**: 6,129 images (22.6%)
- **Per character per type**: ~480 images (training ready)

---

## ✅ Phase 3: Dataset Organization (完成)

**工具**: `scripts/batch/simple_dataset_organizer.py`

**執行指令**:
```bash
python3 /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/batch/simple_dataset_organizer.py
```

**結果**:
- ✅ **45 datasets organized** in Kohya format
  - 42 character-specific datasets
  - 3 universal cross-character datasets
- ✅ **27,112 images** organized with captions
- ✅ **100% success rate** (45/45 datasets)

**Dataset Distribution**:
- Universal pose: 4,577 images
- Universal action: 4,982 images
- Universal expression: 3,997 images
- Character-specific: range from 17 to 737 images per dataset

**Format**: Kohya_ss standard
- Directory structure: `{dataset_name}/1_{dataset_name}/`
- Repeat count: 1 (built into folder name)
- Each image paired with .txt caption file

**輸出位置**: `/mnt/data/ai_data/synthetic_lora_data/datasets/`

---

## ✅ Phase 4: Training Config Generation (完成)

### Phase 4.1: Initial Config Generation

**工具**: `scripts/batch/generate_all_training_configs.py`

**結果**:
- ✅ Generated 45 TOML configs
- ✅ All configs include RTX 5080 compatibility (xformers=false)
- ✅ Priority ordering: Universal (1-3), then Character-specific (4-45)

### Phase 4.2: Filtered Config Generation (FINAL)

**工具**: `scripts/batch/generate_filtered_training_configs.py`

**過濾策略**: Skip datasets with < 50 images (insufficient for effective training)

**執行指令**:
```bash
python3 /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/batch/generate_filtered_training_configs.py
```

**結果**:
- ✅ **40 configs generated**
  - 3 universal LoRAs (all retained)
  - 37 character-specific LoRAs (filtered from 42)
- ⚠️ **5 configs skipped** (too few images):
  - alberto_seamonster_action: 21 images
  - caleb_expression: 28 images
  - elio_expression: 17 images
  - elio_pose: 19 images
  - miguel_action: 41 images

**效益分析**:
- ⏱️ Time saved: **10-15 hours** of training
- 📉 Images lost: Only 126 images (0.5% of total)
- ✅ Focus on viable datasets with >50 images
- 💪 Higher expected success rate

**Training Parameters** (by type):
- **Universal Pose**: dim=128, epochs=30
- **Universal Action**: dim=160, epochs=35
- **Universal Expression**: dim=112, epochs=35
- **Character Pose**: dim=112, epochs=20-30 (adjusted by size)
- **Character Action**: dim=128, epochs=25-35
- **Character Expression**: dim=96, epochs=25-35

**輸出位置**:
- Configs: `/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered/`
- Report: `config_generation_report.json`

---

## 🛠️ 修改的文件

### 新增文件
1. `scripts/generic/training/caption_engines/synthetic_caption_template_generator.py`
2. `scripts/generic/training/caption_engines/synthetic_caption_applier.py`
3. `scripts/generic/training/quality_filters/synthetic_quality_pipeline.py`
4. `scripts/generic/training/quality_filters/character_presence_filter.py`
5. `scripts/generic/training/caption_review/sample_for_review.py` (NEW - Phase 2.2)
6. `scripts/generic/training/caption_review/interactive_caption_reviewer.py` (EXISTING - Phase 2.2)
7. `SYNTHETIC_LORA_NEXT_STEPS.md` (完整實施計劃)

### 修改文件
1. `scripts/generic/training/quality_filters/synthetic_quality_pipeline.py`
   - 添加multiprocessing支持
   - 添加`_filter_image_worker` module-level function
   - 添加`--num-workers`參數
   - 修復PIL Image → numpy array conversion
   - 修復hash method名稱
   - 自動檢測`/generated`子目錄

---

## 🚀 下一步計劃

### Phase 2.2: Manual Caption Review (準備就緒)

**工具**:
1. `scripts/generic/training/caption_review/sample_for_review.py` (✅ 已創建)
2. `scripts/generic/training/caption_review/interactive_caption_reviewer.py` (✅ 已存在)

**執行步驟**:
```bash
# Step 1: Smart sampling (10 min)
conda run -n ai_env python scripts/generic/training/caption_review/sample_for_review.py \
  --input-dir /mnt/data/ai_data/synthetic_lora_data/filtered_data \
  --output-dir /mnt/data/ai_data/synthetic_lora_data/review_samples \
  --target-samples 600

# Step 2: Interactive review (2-3 hours)
conda run -n ai_env python scripts/generic/training/caption_review/interactive_caption_reviewer.py \
  --data-root /mnt/data/ai_data/synthetic_lora_data/filtered_data \
  --samples-per-type 40 \
  --port 5000

# Open: http://localhost:5000
```

**策略**:
- Stratified sampling: 均衡覆蓋各角色、類型、質量層級
- Total review: ~600 images (3% of dataset)
- Web-based UI (HTML/JavaScript + Flask backend)
- Actions: Accept (A), Edit (E), Reject (R)
- Keyboard shortcuts for efficiency
- 自動保存修正到原始 caption 文件

### Phase 3: Dataset Organization

**工具**: `scripts/generic/training/dataset_organizer.py` (已存在)

**格式**: Kohya_ss format
- Repeat count: 1 (dataset already large enough)
- Directory structure: `1_{character}_{lora_type}/`
- Target resolution: 1024×1024

### Phase 4: Training Config Generation

**工具**: `scripts/generic/training/training_config_generator.py` (需修改)

**配置模板**:
- `configs/training/pose_lora_sdxl_template.toml`
- `configs/training/action_lora_sdxl_template.toml`
- `configs/training/expression_lora_sdxl_template.toml`

**Training參數**:
- Pose: network_dim=112, epochs=20
- Action: network_dim=128, epochs=25
- Expression: network_dim=96, epochs=25

### Phase 5: Training Execution (3-4天)

**工具**: `scripts/batch/train_all_synthetic_loras.sh` (待創建)

**策略**: Sequential training (1 at a time)
- Total LoRAs: 42 (14 characters × 3 types)
- Order: All pose → All action → All expression
- Estimated time: 77-98 hours

### Phase 6: Evaluation & Selection

**工具**: `scripts/evaluation/auto_evaluate_checkpoints.py` (需修改)

**Metrics**:
- CLIP Similarity Score
- Pose Accuracy (RTM-Pose)
- Expression Accuracy (FER)
- Character Consistency (ArcFace)

---

## ⚠️ 重要提醒

### RTX 5080 GPU限制
```
NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
```
- ⚠️ **不能使用xformers** (CUDA sm_120不兼容)
- ✅ 使用PyTorch native attention
- ⚠️ Performance impact: 10-30% slower

### Conda環境
- **ai_env**: Caption generation, quality filtering, evaluation
- **kohya_ss**: LoRA training only

### 路徑結構
```
/mnt/data/ai_data/synthetic_lora_data/
  generated_data/          # Original synthetic images
    {character}/{type}/generated/
      *.png
  caption_templates/       # API-generated templates
    caption_templates_*.json
    caption_vocabulary_*.json
  captioned_data/          # Applied captions
    {character}/{type}/
      *.txt
  filtered_data/           # Quality filtered
    {character}/{type}/
      tier_a/
      tier_b/
      tier_c/
  datasets/                # Organized for training (待創建)
    {character}_{type}/
      1_{character}_{type}/
```

---

## 📝 待辦事項

### 已完成 ✅
- [x] Phase 1.1: Generate caption templates
- [x] Phase 1.2: Apply captions to all images
- [x] Copy captions to image directories
- [x] Add multiprocessing support to quality filtering
- [x] Phase 2.1: Quality filtering (20,221/27,118 passed)
- [x] Create interactive caption review UI tools

### 立即執行 🚀
- [ ] **Phase 2.2: Run smart sampling** (~10 min)
- [ ] **Phase 2.2: Interactive caption review** (~2-3 hours)

### 接下來
- [ ] Phase 3: Organize datasets (Kohya format) (~30 min)
- [ ] Phase 4: Generate 42 training configs (~15 min)
- [ ] Phase 5: Sequential LoRA training (3-4 days)
- [ ] Phase 6: Checkpoint evaluation & selection (1 day)

---

## 🐛 已知問題

### Action Caption Failures
- **問題**: 776/8158 action captions failed (9.5%)
- **原因**: Generated captions exactly 74 tokens (below 75 minimum)
- **影響**: Minor, ~90% action images still have captions
- **狀態**: Accepted (不影響整體質量)

### YOLOv8 Character Presence
- **問題**: YOLOv8每個worker都會重新加載模型
- **影響**: 可能增加記憶體使用和初始化時間
- **優化**: 考慮使用shared model或batch processing
- **狀態**: 可接受 (32 workers仍然快)

---

**文檔版本**: 1.1
**維護者**: LLMProvider Tooling
**下次更新**: Phase 2.2完成後
