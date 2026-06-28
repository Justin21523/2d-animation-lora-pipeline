# 批次處理工具使用指南

## 🎯 概述

通用批次處理工具能夠自動發現多部電影並依序執行處理任務（SAM2 分割、LaMa inpainting 等），支援中斷恢復、錯誤重試和詳細日誌記錄。

## 📋 功能特點

- ✅ **自動發現電影**：掃描 `/mnt/data/ai_data/datasets/3d-anime/` 目錄
- ✅ **依賴管理**：自動處理任務依賴（SAM2 → LaMa → Clustering）
- ✅ **完成檢測**：雙重驗證（exit code + 實際輸出）
- ✅ **中斷恢復**：可隨時恢復中斷的處理
- ✅ **錯誤重試**：自動重試失敗的任務（可配置次數）
- ✅ **詳細日誌**：每個任務獨立日誌文件
- ✅ **可擴充**：YAML 配置即可新增操作
- ✅ **順序執行**：一次處理一部電影（避免 GPU 衝突）
- ✅ **Watchdog 監控**：自動重啟崩潰的進程

---

## ✅ 當前系統狀態 (2025-11-17 23:32)

**系統已啟動並運行中**:
- 🐕 Watchdog PID: 1718408 ✅
- 🤖 Batch Processor PID: 1718457 ✅
- 🔄 當前任務: Coco SAM2 分割 (PID: 1718561)
- 📊 處理佇列: 6 部電影（Coco → Elio → Onward → Orion → Turning-red → Up）
- ⏱️ 預計完成: 2025-11-20 上午（約 56 小時）

**即時監控**:
```bash
watch -n 10 'bash scripts/batch/monitor.sh'
```

---

## 🚀 快速開始

### 1. 智能啟動（推薦，防止 GPU 競爭）

```bash
# 自動等待 GPU 空閒後才啟動批次處理
# 適用於有其他 GPU 任務正在運行的情況
nohup bash scripts/batch/smart_batch_launcher.sh configs/batch/sam2_lama.yaml > logs/smart_batch_launcher.log 2>&1 &

# 查看等待狀態
tail -f logs/smart_batch_launcher.log

# 確認智能啟動器正在運行
ps aux | grep smart_batch_launcher | grep -v grep
```

**智能啟動器會：**
- ✅ 每 5 分鐘檢查 GPU 記憶體使用量（< 5GB）
- ✅ 偵測是否有競爭的 SAM2/LaMa 進程
- ✅ 條件滿足時自動啟動批次處理
- ✅ 可安全斷線（nohup 保護）

### 2. 直接啟動（確保 GPU 空閒時使用）

```bash
# 立即執行批次處理（需確保無其他 GPU 任務）
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml
```

### 3. 測試配置（Dry Run）

```bash
# 測試配置文件，不實際執行
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml --dry-run
```

### 4. 重新開始（忽略進度）

```bash
# 從頭開始處理，忽略之前的進度
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml --no-resume
```

---

## 📁 配置文件

### `configs/batch/sam2_lama.yaml` - SAM2 + LaMa 完整流程

處理所有電影的 SAM2 分割和 LaMa inpainting：

```yaml
discovery:
  base_dir: "/mnt/data/ai_data/datasets/3d-anime"
  exclude:
    - "luca"  # 排除已處理的電影

jobs:
  - name: "sam2_segmentation"  # 第一階段：SAM2 分割
    ...

  - name: "lama_inpainting"    # 第二階段：LaMa inpainting
    depends_on: "sam2_segmentation"
    ...
```

### `configs/batch/sam2_only.yaml` - 僅 SAM2 分割

只執行 SAM2 分割，不進行 inpainting：

```bash
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_only.yaml
```

---

## 🔧 自訂配置

### 新增要處理的電影

編輯配置文件中的 `exclude` 列表：

```yaml
discovery:
  exclude:
    - "luca"      # 已處理
    - "coco"      # 暫時跳過
```

### 新增處理階段

在 `jobs` 下新增任務定義：

```yaml
jobs:
  - name: "face_clustering"
    script: "scripts/generic/clustering/face_identity_clustering.py"
    conda_env: "ai_env"
    depends_on: "lama_inpainting"  # 等待 LaMa 完成

    args:
      template: "{base_dir}/{film}_instances_sam2_v2/instances"
      named:
        --output-dir: "{base_dir}/{film}/identity_clusters"
        --min-cluster-size: "10"
        --device: "cuda"

    completion_check:
      type: "file_exists"
      path: "{base_dir}/{film}/identity_clusters/identity_clustering.json"

    retry:
      max_attempts: 2
      backoff_seconds: 60
```

---

## 📊 監控進度

### 查看整體進度

```bash
# 實時查看進度文件
watch -n 5 'cat logs/batch_processing/progress.json | jq ".jobs[] | {name, film, status}"'
```

### 查看特定任務日誌

```bash
# 列出所有日誌
ls -lht logs/batch_processing/

# 查看最新的 SAM2 日誌
tail -f logs/batch_processing/sam2_segmentation_coco_*.log
```

### 統計處理狀態

```bash
# 統計各狀態的任務數量
cat logs/batch_processing/progress.json | jq '.jobs | group_by(.status) | map({status: .[0].status, count: length})'
```

---

## 🎯 處理流程

### 典型批次流程（6 部電影）

```
發現電影: coco, elio, onward, orion, turning-red, up
  ↓
對每部電影：
  ├─ SAM2 分割
  │   ├─ 檢查是否已完成
  │   ├─ 執行 instance_segmentation.py
  │   ├─ 驗證輸出（instances/目錄 ≥ 10個文件）
  │   └─ 保存進度
  │
  └─ LaMa Inpainting（等待 SAM2 完成）
      ├─ 檢查是否已完成
      ├─ 執行 sam2_background_inpainting.py
      ├─ 驗證輸出（inpainting_metadata.json 存在）
      └─ 保存進度
```

### 預估時間

| 電影 | Frames | SAM2 (13.4s/frame) | LaMa (2.5s/frame) | 總計 |
|------|--------|-------------------|------------------|------|
| coco | 2,058 | ~7.7 小時 | ~1.4 小時 | ~9.1 小時 |
| elio | 1,910 | ~7.1 小時 | ~1.3 小時 | ~8.4 小時 |
| onward | 2,318 | ~8.6 小時 | ~1.6 小時 | ~10.2 小時 |
| orion | 2,805 | ~10.4 小時 | ~1.9 小時 | ~12.3 小時 |
| turning-red | 1,894 | ~7.0 小時 | ~1.3 小時 | ~8.3 小時 |
| up | 1,642 | ~6.1 小時 | ~1.1 小時 | ~7.2 小時 |
| **總計** | **12,627** | **~47 小時** | **~8.7 小時** | **~56 小時** |

---

## 🔍 完成檢測機制（已增強）

### 雙重驗證系統

系統會同時檢查：
1. ✅ 進程退出碼（exit code == 0）
2. ✅ **實際輸出驗證**（根據 completion_check 配置）

只有兩者都通過才標記為完成。這避免了進程正常退出但輸出不完整的情況。

### 支援的檢測類型

#### 1. `backgrounds_complete` - SAM2 背景完成度檢測（推薦）

```yaml
completion_check:
  type: "backgrounds_complete"
  path: "{base_dir}/{film}/{film}_instances_sam2_v2/backgrounds"
  frames_dir: "{frames_dir}"
  tolerance: 0.95  # 至少 95% 的幀必須處理完成
```

**檢測邏輯**:
- 統計原始 frames 目錄的總幀數
- 統計 backgrounds 目錄的輸出數量
- 計算完成度 = 輸出數量 / 總幀數
- 只有 >= tolerance 才算完成

**優點**: 能準確檢測是否完整處理，避免只處理部分幀就停止的問題。

#### 2. `directory_exists` - 目錄存在 + 最小文件數

```yaml
completion_check:
  type: "directory_exists"
  path: "{base_dir}/{film}_instances_sam2_v2/instances"
  min_files: 10  # 至少 10 個文件
```

#### 3. `file_exists` - 特定文件存在

```yaml
completion_check:
  type: "file_exists"
  path: "{base_dir}/{film}/backgrounds_lama_v2/inpainting_metadata.json"
```

#### 4. `metadata_key` - JSON 文件中的特定鍵值

```yaml
completion_check:
  type: "metadata_key"
  path: "{base_dir}/{film}/cluster_report.json"
  key: "total_clusters"  # 檢查 total_clusters > 0
```

---

## ⚠️ 常見問題

### Q1: 如何跳過已處理的電影？

在配置文件的 `exclude` 列表中添加電影名稱：

```yaml
discovery:
  exclude:
    - "luca"
    - "coco"  # 新增要跳過的電影
```

### Q2: 如何恢復中斷的處理？

直接重新執行相同的命令，系統會自動從上次中斷處繼續：

```bash
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml
```

### Q3: 如何調整重試次數？

修改配置文件中的 `retry` 設定：

```yaml
retry:
  max_attempts: 5      # 最多重試 5 次
  backoff_seconds: 300  # 每次重試間隔 5 分鐘
```

### Q4: 如何只處理特定幾部電影？

使用 `film_pattern` 和 `exclude` 組合：

```yaml
discovery:
  base_dir: "/mnt/data/ai_data/datasets/3d-anime"
  film_pattern: "*"
  exclude:
    - "luca"
    - "elio"
    - "onward"
    # 只處理 coco, orion, turning-red, up
```

### Q5: 處理失敗了怎麼辦？

1. 查看日誌找出錯誤原因：
   ```bash
   tail -100 logs/batch_processing/sam2_segmentation_coco_*.log
   ```

2. 修正問題後重新執行（自動跳過已完成的任務）

3. 如果需要重新處理特定電影，刪除其輸出目錄：
   ```bash
   rm -rf /mnt/data/ai_data/datasets/3d-anime/coco/coco_instances_sam2_v2
   ```

---

## 📈 輸出結構

處理完成後，每部電影會有以下結構：

```
/mnt/data/ai_data/datasets/3d-anime/{film}/
├── frames/  或  frames_final/    # 原始 frames
├── {film}_instances_sam2_v2/     # SAM2 輸出
│   ├── instances/                # 角色實例（PNG）
│   ├── masks/                    # Binary masks（PNG）
│   └── backgrounds/              # 初步背景（OpenCV TELEA）
└── backgrounds_lama_v2/          # 最終高品質背景（LaMa）
    ├── *.jpg                     # 清理後的背景圖片
    └── inpainting_metadata.json  # 處理統計
```

---

## 🎓 進階用法

### 僅測試特定電影

創建自訂配置文件：

```bash
# 複製模板
cp configs/batch/sam2_lama.yaml configs/batch/test_coco.yaml

# 編輯配置，只保留 coco
vim configs/batch/test_coco.yaml

# 執行
bash scripts/batch/run_batch_processing.sh configs/batch/test_coco.yaml
```

### 並行處理多部電影（CPU-only 任務）

修改配置中的 `execution.mode`：

```yaml
execution:
  mode: "parallel"  # 並行處理（僅適用於 CPU-only 任務）
  max_parallel: 2   # 最多同時 2 個任務
```

**注意**：GPU 任務（SAM2、LaMa）建議使用 `sequential` 模式避免衝突。

---

## 🔬 技術機制詳解

### GPU 競爭防護（Smart Launcher）

**問題**：單 GPU 系統無法同時運行多個 SAM2/LaMa 任務，會導致：
- ❌ 記憶體溢出（OOM）
- ❌ 兩個任務都變慢（GPU 時間切片）
- ❌ 不可預測的行為

**解決方案**：智能啟動器（`smart_batch_launcher.sh`）

```bash
# 腳本位置
scripts/batch/smart_batch_launcher.sh

# 核心檢查邏輯
check_gpu_memory() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
}

check_competing_processes() {
    ps aux | grep -E "instance_segmentation.py|sam2_background_inpainting.py"
}

# 啟動條件（兩者都必須滿足）
if [ "$gpu_mem" -lt 5000 ] && [ "$competing_procs" -eq 0 ]; then
    # 啟動批次處理
fi
```

**使用場景**：

| 情境 | 建議啟動方式 | 原因 |
|------|------------|------|
| GPU 完全空閒 | 直接啟動 | 無競爭風險 |
| Luca SAM2 正在運行 | **智能啟動** | 自動等待 Luca 完成 |
| 其他電影正在處理 | **智能啟動** | 避免 GPU 衝突 |
| 多日無人值守處理 | **智能啟動** | 自動接續，無需人工介入 |

**監控智能啟動器**：

```bash
# 查看等待日誌
tail -f logs/smart_batch_launcher.log

# 確認智能啟動器運行中
ps aux | grep smart_batch_launcher | grep -v grep

# 強制停止等待（不影響已啟動的批次處理）
pkill -f smart_batch_launcher
```

---

### Tmux Session 管理

**為什麼需要 Tmux？**

批次處理任務（SAM2、LaMa）通常需要 8-12 小時，使用 tmux session 可以：
- ✅ **SSH 斷線時持續執行**：即使終端關閉，任務仍在背景運行
- ✅ **隨時附加查看**：可以 `tmux attach` 查看實時輸出
- ✅ **多會話管理**：每個任務獨立 session，互不干擾
- ✅ **斷點續傳**：重啟後自動檢測並同步現有 session

**Tmux Session 生命週期**

```python
# 1. 啟動新任務時創建 tmux session
session_name = f"batch_{job.name}_{job.film}_{timestamp}"
tmux.create_session(session_name, command, log_file)

# 2. 任務運行中：定期檢查進程狀態
if tmux.is_process_running(job.pid):
    job.status = "running"
else:
    # 檢查是否完成或失敗
    if check_completion(job):
        job.status = "completed"
        tmux.kill_session(session_name)
    else:
        job.status = "failed"

# 3. 恢復時：自動同步現有 sessions
existing_sessions = tmux.list_sessions()
for job in jobs:
    if job.tmux_session in existing_sessions:
        # 檢測 PID 是否仍在運行
        if tmux.is_process_running(job.pid):
            print(f"📡 檢測到運行中的任務: {job.name}/{job.film}")
```

**常用 Tmux 命令**

```bash
# 列出所有批次處理 sessions
tmux ls | grep batch_

# 附加到特定任務查看實時輸出
tmux attach -t batch_sam2_segmentation_coco_20251116_143052

# 分離 session（任務繼續運行）
Ctrl+B, D

# 強制終止某個 session
tmux kill-session -t batch_sam2_segmentation_coco_20251116_143052
```

---

### 自動完成檢測機制

**核心理念：檢查實際輸出，而非進程狀態**

批次處理器**不依賴**進程退出碼（exit code），而是驗證**實際輸出文件**是否符合預期。這樣即使進程崩潰，也能正確恢復。

**檢測流程圖**

```
任務執行完成（或進程退出）
  ↓
調用 _check_completion(job)
  ↓
根據 completion_check.type 選擇驗證方式：
  ├─ directory_exists → 檢查目錄是否存在 + 文件數量 ≥ min_files
  ├─ file_exists → 檢查特定文件是否存在
  └─ metadata_key → 讀取 JSON 文件，驗證特定鍵值 > 0
  ↓
驗證通過 → status = "completed"
驗證失敗 → status = "failed" → 進入重試流程（如果 retry 未達上限）
```

**三種檢測類型的內部實現**

```python
def _check_completion(self, job: Job) -> bool:
    """Check if job has completed successfully by validating actual outputs"""
    check = job.completion_check
    check_type = check['type']

    if check_type == "directory_exists":
        # 例如：檢查 SAM2 instances/ 目錄是否至少有 10 個文件
        path = Path(self._resolve_template(check['path'], job.film))
        if not path.exists():
            return False
        min_files = check.get('min_files', 0)
        if min_files > 0:
            file_count = len(list(path.iterdir()))
            self.logger.info(f"Found {file_count} files in {path} (required: {min_files})")
            return file_count >= min_files
        return True

    elif check_type == "file_exists":
        # 例如：檢查 inpainting_metadata.json 是否存在
        path = Path(self._resolve_template(check['path'], job.film))
        exists = path.exists()
        self.logger.info(f"Checking {path}: {'✅ exists' if exists else '❌ not found'}")
        return exists

    elif check_type == "metadata_key":
        # 例如：檢查 cluster_report.json 中 total_clusters > 0
        path = Path(self._resolve_template(check['path'], job.film))
        if not path.exists():
            return False
        with open(path) as f:
            data = json.load(f)
        key = check['key']
        has_key = key in data and data[key] > 0
        self.logger.info(f"Checking {path}['{key}']: {data.get(key, 'N/A')}")
        return has_key
```

**為什麼不用 exit code？**

- ❌ **不可靠**：進程可能正常退出但輸出不完整（磁碟滿、OOM、手動中止）
- ❌ **難恢復**：無法區分「已完成」與「失敗後手動清理」
- ✅ **輸出驗證更準確**：直接檢查 instances/ 目錄有足夠的文件
- ✅ **支援斷點續傳**：刪除輸出目錄 = 自動重新處理

---

### 任務依賴與自動接續

**依賴聲明（YAML 配置）**

```yaml
jobs:
  - name: "sam2_segmentation"
    # ... (無 depends_on，可立即執行)

  - name: "lama_inpainting"
    depends_on: "sam2_segmentation"  # ← 聲明依賴
    # ... (必須等待 SAM2 完成)

  - name: "face_clustering"
    depends_on: "lama_inpainting"    # ← 鏈式依賴
```

**依賴解析邏輯**

```python
def _dependency_met(self, job: Job) -> bool:
    """Check if job's dependency has been satisfied"""
    if not job.depends_on:
        return True  # No dependency, always runnable

    # Find the dependency job for the same film
    dep_jobs = [j for j in self.jobs
                if j.name == job.depends_on and j.film == job.film]

    if not dep_jobs:
        self.logger.warning(f"Dependency '{job.depends_on}' not found for {job}")
        return False

    dep_job = dep_jobs[0]
    is_met = dep_job.status == "completed"

    if not is_met:
        self.logger.debug(f"{job.name}/{job.film} waiting for {dep_job.name} (status: {dep_job.status})")

    return is_met
```

**自動接續流程**

```
主循環（每 30 秒輪詢）
  ↓
1. 更新運行中任務的狀態（檢查 PID + 完成驗證）
  ↓
2. 查找可運行的新任務：
   for job in jobs:
       if job.status == "pending" and dependency_met(job):
           runnable_jobs.append(job)
  ↓
3. 啟動可運行任務（在 tmux session 中）
  ↓
4. 保存進度到 progress.json
  ↓
5. 檢查是否所有任務完成
   - 是 → 退出主循環
   - 否 → sleep(30) 後重複
```

**示例：Luca 完成後自動觸發批次處理**

假設配置如下：

```yaml
discovery:
  exclude:
    - "luca"  # Luca 手動處理，批次處理跳過

jobs:
  - name: "sam2_segmentation"
  - name: "lama_inpainting"
    depends_on: "sam2_segmentation"
```

**執行順序**：

```
T0: 啟動批次處理
  ├─ 發現電影: coco, elio, onward, orion, turning-red, up (排除 luca)
  └─ 創建 12 個任務（6 部電影 × 2 任務）

T1: 啟動第一部電影（coco）的 SAM2
  ├─ 創建 tmux session: batch_sam2_segmentation_coco_20251116_150000
  ├─ status: "running"
  └─ lama_inpainting 狀態: "pending" (等待 SAM2 完成)

T2 (7 小時後): coco SAM2 完成
  ├─ 檢測到進程退出
  ├─ 驗證 instances/ 目錄: 2,058 個文件 ≥ 10 ✅
  ├─ SAM2 status → "completed"
  ├─ 檢查 lama_inpainting 依賴: dependency_met() = True
  └─ 自動啟動 coco LaMa inpainting

T3 (1.5 小時後): coco LaMa 完成
  ├─ 驗證 inpainting_metadata.json 存在 ✅
  ├─ LaMa status → "completed"
  └─ coco 全部任務完成，開始處理下一部電影（elio）

...（依此類推，處理所有 6 部電影）

T_final (~56 小時後): 所有任務完成
  └─ 批次處理器退出，報告統計結果
```

**關鍵特性**：

- ✅ **無需人工介入**：SAM2 完成後自動觸發 LaMa
- ✅ **電影間串行**：一次處理一部電影，避免 GPU 衝突
- ✅ **任務間依賴**：LaMa 必定在 SAM2 後執行
- ✅ **可中斷恢復**：重啟後自動跳過已完成的任務

---

### 進度持久化與恢復

**progress.json 結構**

```json
{
  "config_file": "configs/batch/sam2_lama.yaml",
  "started_at": "2025-11-16T15:00:00",
  "last_updated": "2025-11-16T22:30:45",
  "jobs": [
    {
      "name": "sam2_segmentation",
      "film": "coco",
      "status": "completed",
      "started_at": "2025-11-16T15:00:05",
      "completed_at": "2025-11-16T22:30:12",
      "tmux_session": "batch_sam2_segmentation_coco_20251116_150000",
      "pid": 1234567,
      "attempts": 1
    },
    {
      "name": "lama_inpainting",
      "film": "coco",
      "status": "running",
      "started_at": "2025-11-16T22:30:45",
      "tmux_session": "batch_lama_inpainting_coco_20251116_223045",
      "pid": 1234890,
      "attempts": 1
    },
    ...
  ]
}
```

**恢復邏輯（`--resume` 默認啟用）**

```python
def _load_progress(self) -> None:
    """Load progress from previous run and sync with tmux sessions"""
    if not self.progress_file.exists():
        return

    with open(self.progress_file) as f:
        data = json.load(f)

    # Restore job status
    for job_data in data['jobs']:
        job = self._find_job(job_data['name'], job_data['film'])
        job.status = job_data['status']
        job.tmux_session = job_data.get('tmux_session')
        job.pid = job_data.get('pid')
        job.attempts = job_data.get('attempts', 0)

    # Sync with existing tmux sessions (may have been running in background)
    self._sync_with_tmux()
```

**Tmux 同步邏輯**

```python
def _sync_with_tmux(self) -> None:
    """Sync job status with existing tmux sessions"""
    existing_sessions = self.tmux.list_sessions()

    for job in self.jobs:
        if job.tmux_session and job.tmux_session in existing_sessions:
            # Session still exists, check if process is running
            if job.pid and self.tmux.is_process_running(job.pid):
                job.status = "running"
                print(f"📡 檢測到運行中的任務: {job.name}/{job.film} (PID {job.pid})")
            else:
                # Process finished, validate completion
                if self._check_completion(job):
                    job.status = "completed"
                    self.tmux.kill_session(job.tmux_session)
                    print(f"✅ 任務已完成: {job.name}/{job.film}")
                else:
                    job.status = "failed"
                    print(f"❌ 任務失敗: {job.name}/{job.film}")
```

**恢復情境示例**

| 情境 | 行為 |
|------|------|
| 批次處理器崩潰（tmux sessions 仍運行） | 重啟後自動檢測運行中的 sessions，同步狀態 |
| SSH 斷線後重新連接 | tmux sessions 持續運行，重新附加即可查看 |
| 手動中止批次處理器（Ctrl+C） | tmux sessions 繼續運行，重啟後自動同步 |
| 系統重啟（tmux sessions 丟失） | 根據輸出驗證結果更新狀態，跳過已完成任務 |
| 刪除某電影輸出目錄 | 完成檢測失敗 → 自動重新處理 |

---

### 關於當前 Luca 處理的說明

**目前狀態**：

- Luca SAM2 處理使用 `nohup`（非 tmux）
- 進程 PID: 1141506（可能已變化）
- 進度: ~1,181/14,410 幀（8.2%）
- 預計剩餘時間: ~49 小時

**是否需要遷移到 tmux？**

**建議：讓當前進程繼續，未來批次使用 tmux**

| 選項 | 優點 | 缺點 |
|------|------|------|
| **保持 nohup 完成** | ✅ 不損失進度<br>✅ 無風險 | ❌ 無法實時附加查看<br>❌ 需透過 log 文件監控 |
| **遷移到 tmux** | ✅ 可隨時附加查看<br>✅ 與批次處理統一管理 | ❌ 需中止進程（損失 ~6 小時進度）<br>❌ 重啟風險 |

**推薦做法**：

```bash
# 1. 讓 Luca 繼續用 nohup 完成
tail -f logs/sam2_segmentation_v2.log  # 監控進度

# 2. 待 Luca SAM2 完成後，手動執行 LaMa（使用 tmux）
tmux new -s luca_lama
conda run -n ai_env python scripts/generic/inpainting/sam2_background_inpainting.py \
  --sam2-dir /mnt/data/ai_data/datasets/3d-anime/luca/luca_instances_sam2_v2 \
  --output-dir /mnt/data/ai_data/datasets/3d-anime/luca/backgrounds_lama_v2 \
  --method lama \
  --mask-dilate 20 \
  --batch-size 4 \
  --device cuda

# 3. 之後的批次處理（其他 6 部電影）將自動使用 tmux
bash scripts/batch/run_batch_processing.sh configs/batch/sam2_lama.yaml
```

---

## 🔄 順序執行模式（2025-11-17 更新）

**重要變更**: 系統已修改為**一次只處理一個任務**（順序執行），避免 GPU 記憶體衝突。

**執行順序**:
```
Coco (SAM2 → LaMa) → Elio (SAM2 → LaMa) → Onward (SAM2 → LaMa) → ...
```

**好處**:
- ✅ 避免 GPU OOM（記憶體溢出）
- ✅ 每個任務獲得完整的 GPU 資源
- ✅ 更穩定的處理速度
- ✅ 更容易監控進度

---

## 📊 輸出數據位置

**重要**: 所有處理結果存儲在數據集目錄，**不使用** `outputs/` 目錄：

```
/mnt/data/ai_data/datasets/3d-anime/{film}/
├── frames_final/                    # 原始 frames
├── {film}_instances_sam2_v2/        # SAM2 輸出
│   ├── instances/                   # 角色實例（PNG，透明背景）
│   ├── masks/                       # Binary masks
│   └── backgrounds/                 # 初步背景（OpenCV TELEA）
└── backgrounds_lama_v2/             # LaMa 最終背景
    ├── *.jpg                        # 清理後的背景圖片
    └── inpainting_metadata.json    # 處理統計
```

**注意**:
- ❌ 舊的 `outputs/luca_backgrounds_lama_cleaned/` **不再使用**
- ✅ Luca 和其他電影的 inpainting 結果統一在 `backgrounds_lama_v2/`

---

## ✅ 總結

批次處理工具提供了：
- 🚀 **自動化**：一次設定，處理所有電影
- 🔄 **可靠**：中斷恢復，不重複處理
- 📊 **可追蹤**：詳細日誌和進度文件
- 🔧 **可擴充**：YAML 配置即可新增操作
- ⚡ **高效**：自動跳過已完成的任務
- 🛡️ **健壯**：Watchdog 自動重啟，SSH 斷線不中斷
- 🔍 **智能**：雙重驗證（exit code + 實際輸出）
- 🔗 **依賴管理**：自動解析任務依賴，確保執行順序
- 🎯 **順序執行**：一次一部電影，避免 GPU 衝突

**當前狀態**: 🟢 系統運行中，預計 2025-11-20 完成所有 6 部電影的處理。

建議使用 `watch -n 10 'bash scripts/batch/monitor.sh'` 監控進度！
