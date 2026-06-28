# SDXL LoRA Training Monitoring Guide

本指南說明如何即時監控 SDXL LoRA 訓練進度，包括時間速率、預估時間和進度條。

## 問題說明

由於 `conda run` 和 `accelerate launch` 的輸出緩衝機制，訓練日誌無法即時輸出到文本文件。kohya_ss 的訓練進度主要記錄在 TensorBoard 日誌中。

## 方法 1：使用即時監控腳本（推薦）

### 啟動監控

```bash
# 監控當前訓練（自動檢測）
bash scripts/batch/monitor_training_progress.sh

# 或指定輸出目錄
bash scripts/batch/monitor_training_progress.sh /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity
```

### 監控內容

監控腳本每 5 秒更新一次，顯示：

1. **GPU 狀態**
   - GPU 使用率（目標：80-99%）
   - VRAM 使用量
   - 溫度
   - 功耗

2. **訓練進程信息**
   - 進程 PID
   - 運行時間
   - CPU/內存使用

3. **Checkpoints**
   - 已保存的 checkpoint 數量
   - 最新的 checkpoint 文件
   - 新 checkpoint 通知

4. **TensorBoard 日誌**
   - 最新日誌文件大小
   - 最後更新時間
   - TensorBoard 訪問命令

5. **訓練配置**
   - 總 epochs
   - 保存間隔
   - Batch size

### 示例輸出

```
================================================================================
🔍 SDXL LoRA Training Progress Monitor
================================================================================
Output directory: /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity
Refresh interval: 5s
Press Ctrl+C to exit
================================================================================

=== 2025-11-22 13:37:27 ===

GPU Status:
  GPU 0:  99% | VRAM:  15722/ 16303 MB | Temp:  46°C | Power:  96.35W

Training Process:
  PID: 2682121 | Runtime: 16:14 | CPU: 94.8% | MEM: 53.3%

Checkpoints:
  Total checkpoints: 2
  Latest checkpoints:
    2025-11-22 13:35  miguel_identity-000002.safetensors
    2025-11-22 13:30  miguel_identity-000001.safetensors
  🔔 New checkpoint detected!

TensorBoard Metrics:
  Latest log: events.out.tfevents.1763789068.MyComputer.2682121.0
  Size: 20K | Last updated: 2025-11-22 13:37:25
  View in TensorBoard: tensorboard --logdir='/mnt/data/.../logs' --port=6006

Training Configuration:
  Config: coco_miguel_identity_sdxl.toml
  Epochs: 10 | Save interval: every 2 epochs | Batch size: 2

================================================================================
Next update in 5s... (Ctrl+C to exit)
```

## 方法 2：使用 TensorBoard 查看詳細指標

### 啟動 TensorBoard

```bash
# 快速啟動（單個角色）
tensorboard --logdir='/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/logs' --port=6006

# 或使用提供的腳本
bash /tmp/start_tensorboard.sh
```

### 訪問 TensorBoard

在瀏覽器中打開：`http://localhost:6006`

### TensorBoard 顯示內容

1. **SCALARS 標籤**：
   - `loss/total` - 總損失（應該逐步下降）
   - `loss/text_encoder` - 文本編碼器損失
   - `lr` - 學習率變化
   - `max_steps` - 總步數

2. **實時更新**：
   - 每 30 秒自動刷新
   - 可查看訓練曲線和趨勢
   - 支持縮放和下載圖表

### 優點

- ✅ 完整的訓練指標曲線
- ✅ 多個實驗對比
- ✅ 視覺化損失下降趨勢
- ✅ 導出數據和圖表

## 方法 3：直接查看 tmux session

如果訓練在 tmux 中運行，可以直接連接查看：

```bash
# 列出所有 tmux sessions
tmux list-sessions

# 連接到訓練 session（只讀模式）
tmux attach-session -t miguel_sdxl_train -r

# 按 Ctrl+B 然後 D 離開 tmux
```

## 方法 4：計算預估時間

### 基於已保存的 checkpoints

```bash
# 查看所有 checkpoints
ls -lhtr /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors

# 計算時間間隔
# 如果 save_every_n_epochs = 2，且已有 2 個 checkpoints
# 時間間隔 = checkpoint2_time - checkpoint1_time
# 預估總時間 = (時間間隔 / 2) * 10 epochs
```

### 基於訓練配置

對於 Miguel (449 images, 3 repeats, 10 epochs, batch_size=2):
- 每 epoch 步數 = (449 × 3) / 2 = 674 steps
- 總步數 = 674 × 10 = 6740 steps
- 每步時間 ≈ 2-2.5 秒（SDXL typical）
- **預估總時間** = 6740 × 2.25 ≈ **4.2 小時**

## 方法 5：使用 Python 腳本解析 TensorBoard 日誌

```bash
# 查看最新的訓練步數和時間
python << 'PYEOF'
import os
from tensorboard.backend.event_processing import event_accumulator

log_dir = "/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/logs"
latest_run = sorted([d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))])[-1]
tfevents_dir = os.path.join(log_dir, latest_run, "network_train")

ea = event_accumulator.EventAccumulator(tfevents_dir)
ea.Reload()

if 'loss/total' in ea.Tags()['scalars']:
    loss_events = ea.Scalars('loss/total')
    print(f"Current step: {loss_events[-1].step}")
    print(f"Current loss: {loss_events[-1].value:.4f}")
    print(f"Elapsed time: {loss_events[-1].wall_time - loss_events[0].wall_time:.0f}s")
PYEOF
```

## 故障排除

### Q: 監控腳本顯示 "No training process found"

**A:** 檢查訓練進程是否真的在運行：
```bash
ps aux | grep sdxl_train_network
nvidia-smi  # 檢查 GPU 使用率
```

### Q: TensorBoard 無法連接

**A:** 確認端口沒有被佔用：
```bash
lsof -i:6006
# 如果被佔用，使用其他端口：
tensorboard --logdir='...' --port=6007
```

### Q: 訓練運行了很久但沒有 checkpoints

**A:** 檢查配置文件中的保存設置：
```bash
grep -E "(save_every_n_epochs|save_model_as)" config.toml
```

確保：
- `save_every_n_epochs` 設置合理（如 2 或 3）
- 輸出目錄有寫入權限

### Q: 如何知道訓練是否正常進行？

**A:** 健康訓練的特徵：
- ✅ GPU 使用率 80-99%
- ✅ VRAM 使用穩定（不應該持續增長）
- ✅ TensorBoard 日誌文件大小持續增長
- ✅ Loss 值逐步下降（查看 TensorBoard）
- ✅ 定期保存 checkpoints

## 總結

| 方法 | 優點 | 缺點 | 推薦度 |
|------|------|------|--------|
| **監控腳本** | 即時、全面、易用 | 需要單獨終端 | ⭐⭐⭐⭐⭐ |
| **TensorBoard** | 詳細指標、視覺化 | 需要瀏覽器 | ⭐⭐⭐⭐ |
| **tmux** | 原始輸出 | 可能沒有進度條 | ⭐⭐⭐ |
| **計算預估** | 快速估算 | 不夠準確 | ⭐⭐ |
| **Python解析** | 精確數據 | 需要編程 | ⭐⭐⭐ |

**最佳實踐**：
1. 使用監控腳本 實時跟蹤進度
2. 同時開啟 TensorBoard 查看訓練曲線
3. 在 tmux 中運行訓練以防斷開
