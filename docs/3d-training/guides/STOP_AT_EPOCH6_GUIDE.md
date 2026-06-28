# ⏸️ Epoch 6 停止操作指南

## 📊 當前狀態

- ✅ **已完成**: Epoch 2, 4
- ⏳ **進行中**: Epoch 5-6
- 📅 **Epoch 6 預計完成**: 凌晨 **00:23**（約 42 分鐘後）

---

## 🎯 操作方案

### 方案 A：自動監控並停止（推薦 ⭐）

**在另一個終端運行監控腳本：**

```bash
# 新開一個終端或 tmux pane
bash scripts/batch/watch_for_epoch6_and_stop.sh
```

**腳本會：**
1. ✅ 每 60 秒檢查 Epoch 6 checkpoint 是否生成
2. ✅ 檢測到後等待 30 秒確保文件完整
3. ✅ 自動停止所有訓練進程
4. ✅ 顯示可用的 checkpoints
5. ✅ 提示下一步操作

---

### 方案 B：手動監控並停止

#### 1. 監控 Epoch 6 完成

```bash
# 方式 1: 查看 tmux 訓練進度
tmux attach -t sdxl_training
# 看到 "Epoch 6" 完成時按 Ctrl+B 然後 D 離開

# 方式 2: 監控 checkpoint 文件
watch -n 30 'ls -lht /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors'
# 看到 miguel_identity_lora_sdxl-000006.safetensors 出現即可

# 方式 3: 監控日誌
tail -f logs/coco_miguel_identity_sdxl_training_*.log | grep -E "epoch.*6|Saving"
```

#### 2. Epoch 6 完成後手動停止

```bash
# 方式 1: 找出並停止訓練進程
ps aux | grep "sdxl_train_network.py" | grep -v grep
kill <PID>

# 方式 2: 使用 pkill
pkill -f "sdxl_train_network.py"

# 方式 3: 停止 tmux session（會完全終止）
tmux kill-session -t sdxl_training
```

---

## 📋 停止後立即執行

### 1️⃣ **確認 3 個 Checkpoints 已保存**

```bash
ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors
```

**應該看到：**
```
miguel_identity_lora_sdxl-000002.safetensors  (Epoch 2)
miguel_identity_lora_sdxl-000004.safetensors  (Epoch 4)
miguel_identity_lora_sdxl-000006.safetensors  (Epoch 6)
```

---

### 2️⃣ **手動測試 3 個 Checkpoints**

```bash
# 執行完整評估（包含所有 SOTA 指標）
bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel
```

**這會：**
1. 評估所有 3 個 checkpoints
2. 每個生成 24 張測試圖（6 prompts × 4 variations）
3. 計算完整評估指標（CLIP, Face Consistency, LPIPS, MUSIQ）
4. 生成排名報告
5. **總時間: 約 15-30 分鐘**

**評估完成後查看結果：**

```bash
# 查看排名報告
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.json | python -m json.tool

# 查看比較圖
eog /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.png

# 查看測試圖（查看 Epoch 6 的生成圖）
eog /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/miguel_identity_lora_sdxl-000006/image_*.png
```

---

### 3️⃣ **根據測試結果決定**

#### 選項 A：滿意 Epoch 6，直接使用

```bash
# 複製最佳 checkpoint 到工作目錄
cp /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/miguel_identity_lora_sdxl-000006.safetensors \
   /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/BEST_miguel.safetensors

echo "✅ Miguel 訓練完成！"
```

#### 選項 B：想繼續訓練到 Epoch 8 或 10

```bash
# 重新啟動訓練（會從 Epoch 6 繼續）
tmux new-session -d -s sdxl_training \
  "bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/coco_miguel_identity_sdxl.toml"

# 然後再次監控 Epoch 8 或 10
```

---

### 4️⃣ **啟動批量訓練其他角色**

一旦決定 Miguel 完成，立即啟動批量訓練：

#### 方案 1：訓練剩餘所有 11 個角色

```bash
tmux new-session -d -s batch_training \
  "bash scripts/batch/train_all_sdxl_sequential.sh bryce caleb elio glordon alberto giulia barley_lightfoot ian_lightfoot orion tyler russell"
```

#### 方案 2：分批訓練（推薦）

**第一批：Elio 電影角色（4個）**
```bash
tmux new-session -d -s batch_training \
  "bash scripts/batch/train_all_sdxl_sequential.sh bryce caleb elio glordon"
```

**第二批：Luca 和 Onward**
```bash
bash scripts/batch/train_all_sdxl_sequential.sh alberto giulia barley_lightfoot ian_lightfoot
```

**第三批：其他**
```bash
bash scripts/batch/train_all_sdxl_sequential.sh orion tyler russell
```

#### 方案 3：使用配置文件

```bash
# 創建訓練列表
cat > next_batch.txt << EOF
# Elio characters
bryce
caleb
elio
glordon

# Luca characters
alberto
giulia
EOF

# 啟動批量訓練
tmux new-session -d -s batch_training \
  "bash scripts/batch/train_all_sdxl_sequential.sh --config next_batch.txt"
```

---

## 📊 批量訓練監控

```bash
# 查看批量訓練 tmux
tmux attach -t batch_training

# 查看批量訓練日誌
tail -f logs/batch_training_*.log

# 查看 GPU 狀態
nvidia-smi

# 查看當前訓練的角色
ps aux | grep "sdxl_train_network.py" | grep -v grep
```

---

## ⏱️ 時間估算

### Miguel 測試階段
- 評估 3 個 checkpoints: **15-30 分鐘**
- 查看結果並決定: **5-10 分鐘**

### 批量訓練時間（每角色 ~7 小時）
- 4 個角色: ~28 小時（1.2 天）
- 8 個角色: ~56 小時（2.3 天）
- 11 個角色: ~77 小時（3.2 天）

---

## 🔧 快速命令參考

```bash
# === Epoch 6 監控 ===
# 自動監控並停止
bash scripts/batch/watch_for_epoch6_and_stop.sh

# 手動檢查 checkpoint
ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors

# 手動停止訓練
pkill -f "sdxl_train_network.py"

# === 測試 Checkpoints ===
# 評估所有 checkpoints
bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel

# 查看結果
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.json | python -m json.tool

# === 啟動批量訓練 ===
# 訓練特定角色
bash scripts/batch/train_all_sdxl_sequential.sh bryce caleb elio

# 訓練所有剩餘角色
bash scripts/batch/train_all_sdxl_sequential.sh bryce caleb elio glordon alberto giulia barley_lightfoot ian_lightfoot orion tyler russell

# 使用配置文件
bash scripts/batch/train_all_sdxl_sequential.sh --config next_batch.txt

# === 監控 ===
# 查看訓練
tmux attach -t batch_training

# 查看日誌
tail -f logs/batch_training_*.log
```

---

## 💡 最佳實踐建議

### 1. **現在立即運行監控腳本**
```bash
bash scripts/batch/watch_for_epoch6_and_stop.sh
```
這樣你就不需要一直盯著，腳本會自動停止。

### 2. **Epoch 6 完成後立即評估**
不要等待，立即運行評估以便快速決策。

### 3. **分批訓練**
建議分 2-3 批訓練，每批 3-5 個角色，這樣：
- ✅ 可以更靈活地調整
- ✅ 出問題時影響較小
- ✅ 可以根據前幾個角色的結果調整策略

### 4. **使用 Tmux**
所有長時間任務都在 tmux 中運行，即使斷線也不影響。

---

## 🎯 **推薦工作流程**

```bash
# 步驟 1: 現在運行（在新終端或tmux pane）
bash scripts/batch/watch_for_epoch6_and_stop.sh

# 步驟 2: 等待自動停止（約 42 分鐘）

# 步驟 3: 停止後立即評估
bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel

# 步驟 4: 查看結果（15-30分鐘後）
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.json | python -m json.tool

# 步驟 5: 決定並啟動批量訓練
tmux new-session -d -s batch_training \
  "bash scripts/batch/train_all_sdxl_sequential.sh bryce caleb elio glordon"
```

**全自動化，無需手動干預！** 🚀
