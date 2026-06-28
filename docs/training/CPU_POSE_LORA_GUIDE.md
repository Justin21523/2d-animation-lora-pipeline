這是一份將說明文字翻譯為全中文（保留關鍵技術名詞中英對照）的文檔版本，內容結構與數據部分完全保持不變。

-----

# 僅使用 CPU 的 Pose LoRA 數據準備指南 (CPU-Only Pose LoRA Data Preparation Guide)

## 概覽 (Overview)

本指南說明如何使用 MediaPipe 姿勢偵測 (pose detection) 透過 **僅使用 CPU 處理 (CPU-only processing)** 來準備 **Pose LoRA 訓練數據集**。這允許您在 GPU 被角色身分 (character identity) LoRA 訓練佔用時，並行準備姿勢數據。

## 目前流程狀態 (Current Pipeline Status)

**處理範圍** (截至 2025-11-21):

### 電影 (共 7 部)

  - Coco (2017) - Pixar
  - Elio (2025) - Pixar
  - Luca (2021) - Pixar
  - Onward (2020) - Pixar
  - Turning Red (2022) - Pixar
  - Up (2009) - Pixar
  - Orion and the Dark (2024) - DreamWorks

### 角色 (共 14 位)

| 電影 (Movie) | 角色 (Character) | 圖像數 (Images) | 來源目錄 (Source Directory) |
|-------|-----------|--------|------------------|
| **Coco** | miguel | 449 | characters\_inpainted/miguel |
| **Elio** | elio | 680 | characters\_inpainted/elio |
| **Elio** | bryce | 48→201 | characters\_inpainted/bryce → characters\_augmented/bryce |
| **Elio** | caleb | 30→195 | characters\_inpainted/caleb → characters\_augmented/caleb |
| **Elio** | glordon | 112→201 | characters\_inpainted/glordon → characters\_augmented/glordon |
| **Elio** | olga | 93→204 | characters\_inpainted/olga → characters\_augmented/olga |
| **Luca** | luca\_human | 1231 | characters\_inpainted/luca\_human |
| **Luca** | alberto\_human | 509 | characters\_inpainted/alberto\_human |
| **Luca** | giulia | 273 | characters\_inpainted/giulia |
| **Onward** | ian | 359 | characters\_inpainted/ian |
| **Onward** | barley | 258 | characters\_inpainted/barley |
| **Turning Red** | tyler | 46→276 | characters\_inpainted/tyler → characters\_augmented/tyler |
| **Up** | russell | 162→243 | characters\_inpainted/russell → characters\_augmented/russell |
| **Orion** | orion | 261 | characters\_inpainted/orion |

**需準備的總數據集**: 70 (14 位角色 × 5 種動作)

**動作**: standing (站立), walking (行走), running (奔跑), jumping (跳躍), sitting (坐著)

## 為什麼選擇僅使用 CPU 處理？ (Why CPU-Only Processing?)

**問題**: 在角色身分 (character identity) LoRA 訓練期間 GPU 被完全佔用 (\~14-15GB VRAM)，沒有資源進行姿勢偵測。

**解決方案**: 使用 MediaPipe (CPU 優化) 代替 RTM-Pose (需要 GPU) 來進行姿勢關鍵點提取 (pose keypoint extraction)。

**優點**:

  - ✅ **真正的並行處理**: 在訓練運行時準備姿勢數據
  - ✅ **高吞吐量**: 在現代 CPU 上達到 60+ FPS (MediaPipe 模型複雜度 2)
  - ✅ **無 GPU 依賴**: 適用於任何具有足夠 CPU 核心的系統
  - ✅ **相同的輸出格式**: 與整個流程相容的 COCO 17 關鍵點格式

## 架構 (Architecture)

### 流程階段 (Pipeline Stages)

```
Character Instances (Inpainted/Augmented) [角色實例 (已修補/已增強)]
  ↓
MediaPipe Pose Detection (CPU, 60+ FPS) [MediaPipe 姿勢偵測]
  → 33 keypoints → automatic conversion → COCO 17 format [33 關鍵點 → 自動轉換 → COCO 17 格式]
  ↓
Rule-Based Pose Classification (Geometric Analysis) [基於規則的姿勢分類 (幾何分析)]
  → standing, walking, running, jumping, sitting
  ↓
View Classification (Automatic) [視角分類 (自動)]
  → front, back, side, three-quarter [正面, 背面, 側面, 四分之三側]
  ↓
Quality Filtering & Diversity Sampling [品質過濾與多樣性採樣]
  → Perceptual deduplication (pHash) [感知去重]
  → Farthest-point sampling in pose space [姿勢空間中的最遠點採樣]
  ↓
Caption Generation (VLM-assisted) [標註生成 (VLM 輔助)]
  → "character_name, pixar style, standing pose, front view, studio lighting"
  ↓
Dataset Assembly [數據集組裝]
  → images/ + captions/ + pose_metadata.json
```

### MediaPipe 與 RTM-Pose 比較

| 特徵 | MediaPipe (CPU) | RTM-Pose (GPU) |
|---------|----------------|----------------|
| **設備** | 僅限 CPU | 需要 CUDA GPU |
| **速度** | 60+ FPS (CPU) | 100+ FPS (GPU) |
| **準確度** | 良好 | 優異 |
| **關鍵點** | 33 → 17 (COCO) | 17 (COCO 原生) |
| **記憶體** | \~200MB RAM | \~2-3GB VRAM |
| **並行訓練** | ✅ 是 | ❌ 否 (GPU 衝突) |
| **安裝** | `pip install mediapipe` | 需要 MMPose + CUDA |

## 安裝 (Installation)

### 先決條件 (Prerequisites)

```bash
# 啟用環境
conda activate ai_env

# 安裝 MediaPipe
pip install mediapipe opencv-python

# 驗證安裝
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)"
```

預期輸出:

```
MediaPipe version: 0.10.9
```

## 用法 (Usage)

### 單一角色/動作對 (Single Character/Action Pair)

使用 MediaPipe (僅限 CPU) 處理特定動作的單個角色：

```bash
conda run -n ai_env python scripts/generic/training/prepare_pose_lora_data.py \
  --character-instances /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/characters_inpainted/luca_human \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/poses/luca_human_standing \
  --action-name "standing" \
  --character-name "luca_human" \
  --style-description "pixar style, 3d animation, smooth shading" \
  --device cpu \
  --pose-detector mediapipe
```

**關鍵參數**:

  - `--pose-detector mediapipe`: 使用 CPU 優化的 MediaPipe (預設: rtmpose)
  - `--device cpu`: 在 CPU 上處理
  - `--character-instances`: 包含角色圖像的目錄 (已修補或已增強)
  - `--action-name`: 目標動作 (standing, walking, running, jumping, sitting)
  - `--output-dir`: 儲存姿勢數據集的位置

### 批量處理 (所有角色，所有動作)

處理所有電影中的所有角色，每位角色 5 個動作：

```bash
bash scripts/batch/prepare_all_pose_lora_cpu.sh
```

**它的作用**:

1.  自動從 `characters_inpainted/` 或 `characters_augmented/` 中發現角色
2.  處理每位角色的 5 個動作: standing, walking, running, jumping, sitting
3.  並行運行 4 個並發作業 (視 CPU 核心數而定)
4.  生成綜合日誌和統計數據

**預期運行時間**:

  - 每位角色約 5-10 分鐘 (300-500 張圖像)
  - 總計: 6 部電影 × 每部 2-3 位角色 × 5 個動作，約 2-3 小時

**預期輸出**:

```
============================================
Batch Pose LoRA Data Preparation (CPU-Only)
============================================
Started: 2025-01-21 20:00:00

Films: coco elio luca onward turning-red up
Actions: standing walking running jumping sitting

Discovering characters...
Film: luca - Found 3 character(s): luca_human alberto_human giulia
Film: coco - Found 1 character(s): miguel
...

[INFO] Starting task: luca_luca_human_standing
[INFO] Processing 387 images for luca_luca_human_standing...
[SUCCESS] Completed: luca_luca_human_standing

...

============================================
Processing Complete
============================================
Total tasks: 45
Completed: 42
Failed: 1
Skipped: 2
Elapsed time: 2h 15m 30s
```

### 自定義並行作業數量 (Custom Parallel Job Count)

覆蓋預設的 4 個並發作業：

```bash
bash scripts/batch/prepare_all_pose_lora_cpu.sh --max-jobs 8
```

**建議**: 每 2 個 CPU 核心對應 1 個作業 (例如: 8 核 CPU → 4 個作業, 16 核 CPU → 8 個作業)

## 輸出結構 (Output Structure)

每個 角色 × 動作 的組合會產生：

```
/mnt/data/ai_data/datasets/3d-anime/FILM/lora_data/poses/CHARACTER_ACTION/
├── images/                          # 精選姿勢圖像
│   ├── standing_0001.png
│   ├── standing_0002.png
│   └── ...
├── captions/                        # 對應的標註
│   ├── standing_0001.txt
│   ├── standing_0002.txt
│   └── ...
├── pose_data/                       # 關鍵點可視化
│   ├── standing_0001_pose.png
│   └── ...
├── pose_metadata.json               # 統計數據和元數據
└── rejected/                        # 低品質的剔除數據
    └── ...
```

### 元數據文件範例 (Metadata File Example)

```json
{
  "character_name": "luca_human",
  "action_name": "standing",
  "style_description": "pixar style, 3d animation, smooth shading",
  "total_processed": 387,
  "total_selected": 156,
  "rejected_count": 231,
  "rejection_reasons": {
    "low_confidence": 98,
    "blur_detected": 67,
    "duplicate": 44,
    "incomplete_pose": 22
  },
  "view_distribution": {
    "front": 68,
    "three_quarter": 54,
    "side": 22,
    "back": 12
  },
  "avg_keypoint_confidence": 0.87,
  "timestamp": "2025-01-21T20:15:30"
}
```

## 監控進度 (Monitoring Progress)

### 檢查正在運行的作業

```bash
# 查看活動的 Python 進程
ps aux | grep prepare_pose_lora_data.py

# 計算正在運行的作業
jobs -r | wc -l
```

### 查看日誌

```bash
# 主要批量日誌
tail -f logs/pose_lora_prep/batch_pose_prep_TIMESTAMP.log

# 個別任務日誌
tail -f logs/pose_lora_prep/pose_prep_luca_luca_human_standing_TIMESTAMP.log
```

### 檢查完成情況

```bash
# 計算已完成的任務 (檢查是否存在 pose_metadata.json)
find /mnt/data/ai_data/datasets/3d-anime/*/lora_data/poses -name "pose_metadata.json" | wc -l

# 預期: ~45 個文件 (6 部電影 × 2-3 位角色 × 5 個動作)
```

## 與 LoRA 訓練整合 (Integration with LoRA Training)

在姿勢數據準備完成後，訓練 Pose LoRAs：

### 步驟 1: 驗證數據集

```bash
# 檢查一個角色的姿勢數據
ls -lh /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/poses/luca_human_standing/

# 應包含: images/, captions/, pose_data/, pose_metadata.json
```

### 步驟 2: 建立訓練配置

```toml
# configs/training/pose_loras/luca_human_standing_pose.toml

[model]
base_model = "/path/to/sd-v1-5.safetensors"
output_name = "luca_human_standing_pose"

[dataset]
train_data_dir = "/mnt/data/ai_data/datasets/3d-anime/luca/lora_data/poses/luca_human_standing"
resolution = 512

[training]
max_train_epochs = 15
learning_rate = 1e-4
network_dim = 32
network_alpha = 16

[optimizer]
optimizer_type = "AdamW8bit"
lr_scheduler = "cosine_with_restarts"
```

### 步驟 3: 訓練 Pose LoRA

```bash
cd /path/to/kohya_ss/sd-scripts

conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=4 train_network.py \
  --config_file=/path/to/configs/training/pose_loras/luca_human_standing_pose.toml
```

## 疑難排解 (Troubleshooting)

### 問題: "MediaPipe not installed"

**錯誤**:

```
ImportError: MediaPipe not installed. Install with: pip install mediapipe
```

**修復**:

```bash
conda activate ai_env
pip install mediapipe opencv-python
```

### 問題: 腳本卡住或運行緩慢

**症狀**: CPU 使用率低，任務沒有進展

**可能原因**:

1.  可用的 CPU 核心無法負擔太多的並行作業
2.  I/O 瓶頸 (磁碟讀取慢)

**修復**:

```bash
# 減少並行作業
bash scripts/batch/prepare_all_pose_lora_cpu.sh --max-jobs 2

# 檢查 CPU 使用率
htop  # 或: top
```

### 問題: 未發現角色 (No characters discovered)

**錯誤**:

```
[WARNING] No characters found for film: luca
```

**修復**: 確保角色目錄存在:

```bash
# 檢查預期位置
ls /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/characters_inpainted/
ls /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/characters_augmented/
```

### 問題: 高剔除率 (High rejection rate)

**症狀**: `pose_metadata.json` 顯示 80%+ 的剔除率

**可能原因**:

1.  MediaPipe 置信度閾值太嚴格
2.  圖像品質本身較低 (模糊、遮擋)

**修復**: 在腳本中降低 MediaPipe 置信度:

```python
# 在 scripts/core/pose_estimation/mediapipe_detector.py 中
detector = MediaPipePoseDetector(
    model_complexity=2,
    confidence=0.3,  # 從 0.5 降低
    logger=logger
)
```

## 效能優化 (Performance Optimization)

### CPU 核心利用率 (CPU Core Utilization)

**公式**: `max_parallel_jobs = (CPU_cores // 2)`

**範例**:

  - 8 核 CPU → 4 個並行作業
  - 16 核 CPU → 8 個並行作業
  - 32 核 CPU → 16 個並行作業

### 模型複雜度調整 (Model Complexity Tuning)

MediaPipe 提供 3 個複雜度級別：

```python
# 在 prepare_pose_lora_data.py 中修改:
detector = MediaPipePoseDetector(
    model_complexity=0,  # 0=Lite (最快), 1=Full (完整), 2=Heavy (最準確)
    confidence=0.5,
    logger=logger
)
```

**基準測試** (Intel i7-10700K, 8 cores):

  - `complexity=0`: \~80 FPS, 中等準確度
  - `complexity=1`: \~55 FPS, 良好準確度
  - `complexity=2`: \~35 FPS, 最佳準確度

### 磁碟 I/O 優化 (Disk I/O Optimization)

如果使用 HDD (非 SSD)，減少並行作業以避免磁碟震盪 (thrashing)：

```bash
bash scripts/batch/prepare_all_pose_lora_cpu.sh --max-jobs 2
```

## 成本分析 (Cost Analysis)

### 時間成本 (Time Cost)

| 階段 (Stage) | 每位角色所需時間 (Time per Character) |
|-------|-------------------|
| 姿勢偵測 (300 張圖) | 5-8 分鐘 |
| 分類 | 30 秒 |
| 品質過濾 | 1 分鐘 |
| 標註生成 | 2-3 分鐘 |
| **總計** | **\~10-15 分鐘** |

**完整批量** (6 部電影 × 平均 2.5 位角色 × 5 個動作): \~3 小時

### 資源成本 (Resource Cost)

  - **CPU 使用率**: 分配的核心上 100% (4 核心 = 400%)
  - **RAM 使用率**: 每個並發作業約 2GB (4 個作業 = 總計 8GB)
  - **磁碟 I/O**: 中等 (讀取圖像，寫入可視化)
  - **GPU 使用率**: 0% (與訓練完全並行)

## 進階選項 (Advanced Options)

### 自定義動作類型 (Custom Action Types)

擴展預設的 5 個動作以外：

```bash
# 編輯 scripts/batch/prepare_all_pose_lora_cpu.sh
ACTIONS=("standing" "walking" "running" "jumping" "sitting" "crouching" "reaching")
```

### 特定電影處理 (Film-Specific Processing)

僅處理特定電影：

```bash
# 編輯 scripts/batch/prepare_all_pose_lora_cpu.sh
FILMS=("luca" "coco")  # 僅處理 Luca 和 Coco
```

### 覆蓋風格描述 (Override Style Description)

```bash
python scripts/generic/training/prepare_pose_lora_data.py \
  --character-instances /path/to/character \
  --output-dir /path/to/output \
  --action-name "standing" \
  --character-name "character_name" \
  --style-description "dreamworks style, 3d render, cinematic lighting" \
  --device cpu \
  --pose-detector mediapipe
```

## 相關文檔 (Related Documentation)

  - **技術深度剖析**: `docs/training/lora_types/POSE_LORA_DEEP_DIVE.md`
  - **多角色聚類**: `docs/guides/MULTI_CHARACTER_CLUSTERING.md`
  - **訓練流程修復**: `docs/training/TRAINING_PIPELINE_FIXES.md`
  - **批量處理指南**: `docs/training/BATCH_TRAINING_GUIDE.md`

## 參見 (See Also)

  - **MediaPipe Pose**: [https://google.github.io/mediapipe/solutions/pose.html](https://google.github.io/mediapipe/solutions/pose.html)
  - **COCO Keypoints Format**: [https://cocodataset.org/\#keypoints-2020](https://cocodataset.org/#keypoints-2020)
  - **ControlNet Pose Conditioning**: [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)

## 更新日誌 (Changelog)

  - **2025-01-21**: 首次發布，支援 MediaPipe 僅限 CPU 處理
  - **2025-01-21**: 新增針對所有角色/動作的批量處理腳本