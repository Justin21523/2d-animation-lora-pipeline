# 📊 SDXL LoRA 訓練監控指南

## 🚀 快速開始

### 最常用的 3 個指令

```bash
# 1. 查看即時進度條
tmux attach -t batch_training

# 2. 啟動 TensorBoard
bash scripts/batch/start_tensorboard.sh orion

# 3. 監控 GPU
watch -n 2 nvidia-smi
```

### 一鍵監控腳本
```bash
bash scripts/batch/monitor_training_live.sh
```

---

## 📺 查看即時進度條

### 方法 1: 附加到 tmux（推薦）

```bash
tmux attach -t batch_training
# 或簡寫
tmux a -t batch_training
```

**離開 tmux（不中斷訓練）:**
1. 按 `Ctrl+B`
2. 再按 `D`

### 方法 2: 查看輸出但不進入

```bash
# 最後 50 行
tmux capture-pane -t batch_training -p | tail -50

# 只看關鍵字
tmux capture-pane -t batch_training -p | grep -E "epoch|loss|step"
```

---

## 📊 TensorBoard 視覺化

### 啟動 TensorBoard

```bash
# 各角色
bash scripts/batch/start_tensorboard.sh orion   # Port 6006
bash scripts/batch/start_tensorboard.sh elio    # Port 6006
bash scripts/batch/start_tensorboard.sh bryce
bash scripts/batch/start_tensorboard.sh caleb
bash scripts/batch/start_tensorboard.sh alberto
bash scripts/batch/start_tensorboard.sh tyler

# 自訂 port
bash scripts/batch/start_tensorboard.sh orion 6007
```

### 瀏覽器訪問

```
http://localhost:6006
```

### 手動啟動（Orion 範例）

```bash
conda run -n kohya_ss tensorboard \
  --logdir /mnt/c/ai_models/diffusion/lora/sdxl/orion/orion_identity/logs \
  --port 6006 \
  --bind_all
```

---

## 🎮 GPU 監控

### 即時監控

```bash
watch -n 2 nvidia-smi
```

### 單次查詢

```bash
nvidia-smi

# 簡化輸出
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
```

### 正常狀態

- **GPU 使用率:** 70-95%
- **VRAM:** 14-16 GB / 16 GB
- **溫度:** 50-80°C
- **功耗:** 150-250W

---

## 📦 檢查 Checkpoints

```bash
# Orion
ls -lth /mnt/c/ai_models/diffusion/lora/sdxl/orion/orion_identity/*.safetensors

# 所有角色
find /mnt/c/ai_models/diffusion/lora/sdxl -name "*.safetensors"
```

### Checkpoint 保存時間表

| Epoch | 預估時間 | 檔名 |
|-------|---------|------|
| 2     | ~1.5 小時 | `*-000002.safetensors` |
| 4     | ~3 小時   | `*-000004.safetensors` |

---

## 🔍 常用檢查指令

```bash
# 訓練總覽
bash scripts/batch/monitor_training_live.sh

# 訓練進程
ps aux | grep sdxl_train | grep -v grep

# tmux sessions
tmux ls

# 記憶體使用
free -h

# 日誌檔案
tail -50 logs/*.log
```

---

## ⚠️ 疑難排解

### tmux session 不存在

```bash
# 查看所有 sessions
tmux ls
```

### TensorBoard port 被佔用

```bash
# 終止佔用 port 的進程
lsof -ti:6006 | xargs kill -9
```

### 記憶體不足

```bash
# 檢查重複進程
ps aux | grep sdxl_train | wc -l

# 終止所有訓練進程
pkill -9 -f sdxl_train_network.py
```

### GPU 使用率為 0%

```bash
# 檢查進程
ps aux | grep sdxl_train

# 查看 tmux 輸出
tmux capture-pane -t batch_training -p | tail -20
```

---

## 進階監控

### 持續監控（每 30 秒）

```bash
watch -n 30 bash scripts/batch/monitor_training_live.sh
```

### 記錄監控日誌

```bash
while true; do
  echo "=== $(date) ===" >> logs/training_monitor.log
  bash scripts/batch/monitor_training_live.sh >> logs/training_monitor.log
  sleep 300
done
```

---

預祝訓練順利！🚀
