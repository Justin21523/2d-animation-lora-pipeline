# Phase 5: 安全訓練指南

**最後更新**: 2025-12-04
**狀態**: 準備就緒，包含完整記憶體保護機制

---

## 📋 總覽

本指南說明如何安全地執行 40 個合成 LoRA 的訓練：
- **3 個通用 LoRAs**（優先訓練）
- **37 個角色專屬 LoRAs**

**估計總時間**: ~320 小時（13.3 天）

**關鍵安全特性**:
- ✅ CUDA OOM 錯誤保護
- ✅ 自動記憶體清理
- ✅ 系統資源監控
- ✅ Checkpoint 自動評估
- ✅ 訓練進度追蹤與恢復

---

## 🛡️ 記憶體保護機制

### 1. 自動記憶體清理

所有腳本包含積極的記憶體管理：

```python
# Before loading models
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# After inference
del pipe
gc.collect()
torch.cuda.empty_cache()
```

### 2. CUDA OOM 錯誤處理

Checkpoint 評估腳本專門捕獲 OOM 錯誤：

```python
try:
    # Load and run inference
    pipe = StableDiffusionXLPipeline.from_pretrained(...)
    image = pipe(prompt)
except torch.cuda.OutOfMemoryError as e:
    print(f"❌ CUDA OOM: {e}")
    # Cleanup and continue
finally:
    cleanup_memory()
```

### 3. 系統資源監控

`training_safety_monitor.sh` 持續監控：
- **RAM 使用率** - 警告閾值: 90%
- **GPU 記憶體** - 警告閾值: 14 GB / 16 GB
- **GPU 溫度** - 警告閾值: 85°C
- **Swap 使用** - 任何 swap 使用都會警告

### 4. RTX 5080 特殊注意事項

**CUDA Compute Capability sm_120 不支援 xformers！**

所有訓練配置已自動設置：
```toml
xformers = false
mem_eff_attn = false
```

改用 PyTorch 原生 attention（約慢 10-30%，但穩定）。

---

## 🚀 訓練執行步驟

### Step 1: 環境準備

確認環境：
```bash
# 檢查 GPU
nvidia-smi

# 檢查 Kohya_ss 環境
conda activate kohya_ss
which accelerate

# 檢查 sd-scripts 位置
ls ~/sd-scripts/sdxl_train_network.py
```

### Step 2: 啟動安全監控（推薦）

在**另一個終端**運行監控腳本：

```bash
bash scripts/batch/training_safety_monitor.sh
```

這會每 60 秒檢查一次系統資源並記錄到 `/tmp/training_safety_monitor.log`。

**監控內容**:
- RAM / GPU 記憶體使用率
- GPU 溫度
- Swap 使用（性能殺手）
- 訓練進程狀態

### Step 3: 啟動 TensorBoard（可選）

在**另一個終端**啟動 TensorBoard 監控損失曲線：

```bash
# 安裝 TensorBoard (如果尚未安裝)
pip install tensorboard

# 啟動 TensorBoard
tensorboard --logdir /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/synthetic_training --port 6006

# 打開瀏覽器: http://localhost:6006
```

### Step 4: 啟動訓練編排器

**完整訓練（從頭開始）:**

```bash
python3 scripts/batch/train_all_synthetic_loras_sequential.py \
  --config-dir /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered \
  --sd-scripts /home/justin/sd-scripts
```

**從特定 LoRA 恢復訓練:**

```bash
python3 scripts/batch/train_all_synthetic_loras_sequential.py \
  --config-dir /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered \
  --sd-scripts /home/justin/sd-scripts \
  --start-from universal_action  # 從 universal_action 開始
```

**測試運行（不實際訓練）:**

```bash
python3 scripts/batch/train_all_synthetic_loras_sequential.py \
  --config-dir /mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/synthetic_loras_filtered \
  --sd-scripts /home/justin/sd-scripts \
  --dry-run
```

### Step 5: (可選) Checkpoint 自動評估

如果想在訓練期間自動評估 checkpoint：

**監控模式（持續檢查新 checkpoint）:**

```bash
python3 scripts/batch/auto_evaluate_checkpoints.py \
  --lora-dir /mnt/c/ai_models/diffusion/lora/sdxl/synthetic/universal_pose \
  --output-dir /mnt/c/ai_projects/3d-animation-lora-pipeline/evaluation/universal_pose \
  --monitor \
  --interval 300  # 每 5 分鐘檢查一次
```

**一次性評估所有現有 checkpoint:**

```bash
python3 scripts/batch/auto_evaluate_checkpoints.py \
  --lora-dir /mnt/c/ai_models/diffusion/lora/sdxl/synthetic/universal_pose \
  --output-dir /mnt/c/ai_projects/3d-animation-lora-pipeline/evaluation/universal_pose
```

---

## 📊 訓練進度追蹤

### 進度文件位置

訓練編排器會自動維護進度文件：
```bash
cat /tmp/synthetic_lora_training_progress.json
```

**內容**:
```json
{
  "started_at": "2025-12-04T15:30:00",
  "completed": ["universal_pose", "universal_action"],
  "failed": [],
  "current": "universal_expression",
  "last_updated": "2025-12-04T18:45:00"
}
```

### 實時監控訓練

```bash
# 查看訓練編排器輸出
tail -f /tmp/synthetic_lora_training_progress.log

# 查看安全監控日誌
tail -f /tmp/training_safety_monitor.log

# 查看 GPU 使用率
watch -n 5 nvidia-smi
```

---

## ⚠️ 常見問題與解決方案

### 問題 1: CUDA Out of Memory (OOM)

**症狀**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**解決方案**:

1. **降低 batch size**（編輯訓練 TOML 配置）:
   ```toml
   train_batch_size = 1  # 從 2 降到 1
   ```

2. **啟用 gradient checkpointing**:
   ```toml
   gradient_checkpointing = true
   ```

3. **降低 resolution**（不推薦，會影響品質）:
   ```toml
   resolution = "896,896"  # 從 1024 降到 896
   ```

4. **清理 GPU 記憶體並重啟**:
   ```bash
   # 殺掉所有 Python 進程
   pkill -9 python

   # 清理 GPU
   nvidia-smi --gpu-reset

   # 重新啟動訓練
   python3 scripts/batch/train_all_synthetic_loras_sequential.py --start-from <failed_lora>
   ```

### 問題 2: 系統使用 Swap（性能降級）

**症狀**:
```
[WARNING] System is using swap: 8/16 GB - Performance degraded!
```

**解決方案**:

1. **停止其他應用程式**（瀏覽器、IDE 等）

2. **清理系統記憶體**:
   ```bash
   # Drop caches (需要 sudo)
   sudo sync
   sudo echo 3 > /proc/sys/vm/drop_caches
   ```

3. **重啟系統**（如果持續使用 swap）

### 問題 3: GPU 過熱

**症狀**:
```
[WARNING] GPU temperature critical: 87°C
```

**解決方案**:

1. **檢查風扇運作**

2. **改善機箱通風**

3. **降低功率限制**:
   ```bash
   # 設定 GPU 功率限制為 250W（RTX 5080 TDP 320W）
   sudo nvidia-smi -pl 250
   ```

4. **暫停訓練並冷卻**

### 問題 4: 訓練中斷需要恢復

**解決方案**:

訓練編排器自動記錄進度，使用 `--start-from` 恢復：

```bash
# 查看已完成的 LoRAs
cat /tmp/synthetic_lora_training_progress.json | grep completed

# 從下一個未完成的開始
python3 scripts/batch/train_all_synthetic_loras_sequential.py \
  --start-from <next_lora_name>
```

---

## 📈 預期時間表

### 優先訓練（通用 LoRAs）

| LoRA | 圖片數 | Epochs | 預估時間 |
|------|--------|--------|----------|
| universal_pose | 4,577 | 30 | ~57 小時 |
| universal_action | 4,982 | 35 | ~73 小時 |
| universal_expression | 3,997 | 35 | ~58 小時 |
| **小計** | **13,556** | - | **~188 小時 (7.8 天)** |

### 角色專屬 LoRAs

- **數量**: 37 個
- **平均圖片數**: ~360 張
- **平均 Epochs**: 20-25
- **預估時間**: ~132 小時（5.5 天）

### 總計

- **總 LoRAs**: 40 個
- **總圖片數**: ~27,000 張
- **總訓練時間**: **~320 小時（13.3 天）**

---

## ✅ 訓練完成後

### 1. Checkpoint 選擇

每個 LoRA 會產生多個 checkpoints（每 2 epochs 一個）：

```bash
# 列出所有 checkpoints
ls /mnt/c/ai_models/diffusion/lora/sdxl/synthetic/universal_pose/

# 評估所有 checkpoints
python3 scripts/batch/auto_evaluate_checkpoints.py \
  --lora-dir /mnt/c/ai_models/diffusion/lora/sdxl/synthetic/universal_pose \
  --output-dir evaluation/universal_pose
```

### 2. 最佳 Checkpoint 選擇策略

通常選擇：
- **中後期 checkpoint**（例如 epoch 20-25 out of 30）
- **避免最後一個**（可能過擬合）
- **參考 TensorBoard 損失曲線**

### 3. 測試組合效果

測試通用 + 角色專屬 LoRA 組合：

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(...)

# 載入多個 LoRAs
pipe.load_lora_weights("universal_pose.safetensors", adapter_name="pose")
pipe.load_lora_weights("luca_identity.safetensors", adapter_name="character")

# 設定權重
pipe.set_adapters(["pose", "character"], adapter_weights=[0.8, 1.0])

image = pipe("luca standing with arms crossed, pixar style")
```

---

## 🔧 疑難排解

### 檢查依賴

```bash
# Kohya_ss 環境
conda activate kohya_ss
python -c "import torch; print(torch.cuda.is_available())"
python -c "import accelerate; print(accelerate.__version__)"

# AI環境 (evaluation)
conda activate ai_env
python -c "import diffusers; print(diffusers.__version__)"
```

### 清理殘留進程

```bash
# 查看所有訓練相關進程
ps aux | grep train_network

# 殺掉卡住的訓練進程
pkill -9 -f train_network

# 清理 GPU
nvidia-smi --gpu-reset  # 需要 sudo
```

### 磁碟空間檢查

訓練會產生大量 checkpoints：

```bash
# 檢查磁碟空間
df -h /mnt/c/ai_models

# 檢查 LoRA 輸出大小
du -sh /mnt/c/ai_models/diffusion/lora/sdxl/synthetic/*

# 清理舊 checkpoints（僅保留每個 LoRA 的最佳幾個）
# 手動操作，小心刪除！
```

---

## 📚 相關文件

- `SYNTHETIC_LORA_TRAINING_PROGRESS.md` - Phase 1-4 完整記錄
- `SDXL_LORA_TRAINING_RESULTS.md` - 之前的 SDXL 訓練結果參考
- `configs/training/synthetic_loras_filtered/` - 所有訓練配置
- `/tmp/synthetic_lora_training_progress.json` - 實時訓練進度

---

## 🎯 開始訓練！

確認準備就緒後，執行：

```bash
# Terminal 1: 安全監控
bash scripts/batch/training_safety_monitor.sh

# Terminal 2: TensorBoard（可選）
tensorboard --logdir logs/synthetic_training --port 6006

# Terminal 3: 訓練編排器
python3 scripts/batch/train_all_synthetic_loras_sequential.py \
  --config-dir configs/training/synthetic_loras_filtered \
  --sd-scripts ~/sd-scripts
```

**祝訓練順利！🚀**
