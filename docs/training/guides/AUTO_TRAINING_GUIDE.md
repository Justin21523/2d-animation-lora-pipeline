# 🚀 自動訓練評估系統使用指南

## 📋 功能概述

完全自動化的 SDXL LoRA 訓練和評估系統：

✅ **每 2 個 Epochs 自動保存 Checkpoint**（Epoch 2, 4, 6, 8, 10）
✅ **每個 Checkpoint 保存後立即自動評估**
✅ **完整的 SOTA 評估指標**（CLIP, InsightFace, LPIPS, MUSIQ）
✅ **訓練結束後自動生成比較報告**
✅ **自動選出最佳 Checkpoint 並複製**
✅ **支持多角色連續訓練**
✅ **靈活的角色配置**（命令行參數、配置文件、電影分組）

---

## 🎯 快速開始

### 單一角色訓練（帶自動評估）

```bash
# 訓練單一角色（會自動評估每個checkpoint）
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/coco_miguel_identity_sdxl.toml
```

**流程：**
1. 開始訓練 → Epoch 2 完成 → 自動評估 Epoch 2 checkpoint
2. 繼續訓練 → Epoch 4 完成 → 自動評估 Epoch 4 checkpoint
3. ...依此類推直到 Epoch 10
4. 訓練結束 → 生成完整比較報告 → 複製最佳 checkpoint

---

### 批量訓練多個角色

#### 方法 1：訓練所有 12 個角色

```bash
bash scripts/batch/train_all_sdxl_sequential.sh
```

#### 方法 2：指定特定角色

```bash
# 按角色名稱
bash scripts/batch/train_all_sdxl_sequential.sh miguel bryce alberto

# 按電影（會訓練該電影的所有角色）
bash scripts/batch/train_all_sdxl_sequential.sh coco elio luca
```

#### 方法 3：使用配置文件

```bash
# 1. 創建配置文件
cat > my_batch.txt << EOF
miguel
bryce
alberto
giulia
EOF

# 2. 運行批量訓練
bash scripts/batch/train_all_sdxl_sequential.sh --config my_batch.txt
```

#### 方法 4：查看可用角色列表

```bash
bash scripts/batch/train_all_sdxl_sequential.sh --list
```

---

## 📁 輸出結構

每個角色訓練完成後會產生：

```
/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/
├── miguel_identity_lora_sdxl-000002.safetensors  (Epoch 2)
├── miguel_identity_lora_sdxl-000004.safetensors  (Epoch 4)
├── miguel_identity_lora_sdxl-000006.safetensors  (Epoch 6)
├── miguel_identity_lora_sdxl-000008.safetensors  (Epoch 8)
├── miguel_identity_lora_sdxl-000010.safetensors  (Epoch 10)
├── BEST_miguel_identity_lora_sdxl-000006.safetensors  ⭐ 最佳checkpoint
│
├── eval_miguel_identity_lora_sdxl-000002/        評估結果
│   └── miguel_identity_lora_sdxl-000002/
│       ├── image_0000.png ... image_0023.png    (24張測試圖)
│       └── evaluation.json                       (完整評估指標)
├── eval_miguel_identity_lora_sdxl-000004/
│   └── ...
├── eval_miguel_identity_lora_sdxl-000006/
├── eval_miguel_identity_lora_sdxl-000008/
├── eval_miguel_identity_lora_sdxl-000010/
│
└── final_comparison_summary.txt                  ⭐ 完整比較報告
```

---

## 📊 評估報告解讀

### 查看最終比較報告

```bash
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/final_comparison_summary.txt
```

**報告內容：**
```
=== SDXL LoRA CHECKPOINT COMPARISON ===
Character: coco/miguel

Checkpoint Rankings (by Aggregate Score):

🏆 RANK 1: miguel_identity_lora_sdxl-000006 (BEST)
   Aggregate: 0.6234 | CLIP: 0.3456 | Face: 0.8123 | Diversity: 0.4521 | Quality: 72.3

   RANK 2: miguel_identity_lora_sdxl-000008
   Aggregate: 0.6012 | CLIP: 0.3398 | Face: 0.7956 | Diversity: 0.4123 | Quality: 71.8

   RANK 3: miguel_identity_lora_sdxl-000004
   ...

================================================================================
🏆 RECOMMENDED CHECKPOINT: miguel_identity_lora_sdxl-000006
================================================================================
```

### 評估指標說明

| 指標 | 目標值 | 說明 |
|------|--------|------|
| **Aggregate Score** | > 0.60 | 綜合評分（越高越好） |
| **CLIP Score** | > 0.30 | Prompt 對齊度 |
| **Face Consistency** | > 0.75 | 角色臉部一致性⭐最重要 |
| **Diversity** | 0.3-0.6 | 圖像多樣性（太低=mode collapse） |
| **Quality** | > 70 | 技術圖像質量 |

---

## 🔧 高級用法

### 自定義訓練批次

創建自己的訓練列表：

```bash
# 1. 創建配置文件
cat > coco_characters.txt << EOF
# Coco 電影所有角色
coco_miguel_identity_sdxl
EOF

cat > elio_characters.txt << EOF
# Elio 電影所有角色
bryce
caleb
elio
glordon
EOF

# 2. 按配置訓練
bash scripts/batch/train_all_sdxl_sequential.sh --config coco_characters.txt
```

### 監控訓練進度

```bash
# 查看即時訓練
tmux attach -t sdxl_training
# 離開: Ctrl+B 然後 D

# 查看批量訓練日誌
tail -f logs/batch_training_*.log

# 查看 GPU 狀態
nvidia-smi
```

### 訓練時間估算

| 角色數 | 估計時間（假設每角色 7-8 小時） |
|--------|----------------------------------|
| 1 個角色 | ~7-8 小時 |
| 3 個角色 | ~21-24 小時（1 天） |
| 6 個角色 | ~42-48 小時（2 天） |
| 12 個角色 | ~84-96 小時（3.5-4 天） |

**建議：**
- 1-3 個角色：直接運行
- 4-6 個角色：建議在 tmux 中運行（overnight）
- 7+ 個角色：建議分批訓練，每批 3-4 個

---

## 🎯 最佳實踐

### 1. 使用 Tmux 運行（長時間訓練）

```bash
# 創建 tmux session
tmux new-session -s training

# 在 tmux 中運行訓練
bash scripts/batch/train_all_sdxl_sequential.sh miguel bryce alberto

# 離開 tmux（訓練繼續運行）
Ctrl+B, 然後 D

# 稍後重新連接
tmux attach -t training
```

### 2. 定期檢查進度

```bash
# 查看最新訓練日誌
tail -100 logs/batch_training_*.log | grep -E "RANK|Training:|completed"

# 查看已保存的 checkpoints
ls -lht /mnt/c/ai_models/diffusion/lora/sdxl/*/*/miguel_identity_lora_sdxl-*.safetensors
```

### 3. 失敗處理

如果某個角色訓練失敗：

```bash
# 1. 查看失敗原因
cat logs/batch_training_*.log | grep -A 10 "FAILED"

# 2. 單獨重新訓練該角色
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/failed_character.toml

# 3. 或創建剩餘角色列表繼續
cat > remaining.txt << EOF
character_that_failed
character_not_trained_yet
EOF

bash scripts/batch/train_all_sdxl_sequential.sh --config remaining.txt
```

---

## 📋 完整工作流範例

### 場景：訓練 Coco 電影的所有角色

```bash
# Step 1: 列出可用角色
bash scripts/batch/train_all_sdxl_sequential.sh --list | grep coco

# Step 2: 創建訓練列表
cat > coco_batch.txt << EOF
# Coco characters
miguel
EOF

# Step 3: 在 tmux 中啟動訓練
tmux new-session -d -s coco_training \
  "bash scripts/batch/train_all_sdxl_sequential.sh --config coco_batch.txt"

# Step 4: 監控進度
tmux attach -t coco_training

# Step 5: 訓練完成後查看結果
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/final_comparison_summary.txt

# Step 6: 使用最佳 checkpoint
ls /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/BEST_*.safetensors
```

---

## 🆘 常見問題

### Q: 評估失敗了怎麼辦？

**A:** 評估失敗不會影響訓練。訓練會繼續進行，只是該 checkpoint 沒有評估結果。可以訓練結束後手動評估：

```bash
bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel
```

### Q: 可以暫停後繼續嗎？

**A:** 目前腳本不支持斷點續傳。如果訓練中斷：
1. 檢查哪些角色已完成
2. 創建剩餘角色列表
3. 使用 `--config` 參數繼續訓練

### Q: 如何選擇最佳 Epoch？

**A:** 系統會自動選出最佳 checkpoint（通常是 Epoch 4-6）。查看 `final_comparison_summary.txt` 中的排名。

### Q: 訓練速度慢怎麼辦？

**A:**
- 確認 GPU 利用率 > 90%（`nvidia-smi`）
- 檢查是否有其他進程佔用 GPU
- 評估會暫時佔用 GPU，可以選擇訓練完成後統一評估

---

## 🎉 總結

**單一角色：**
```bash
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/coco_miguel_identity_sdxl.toml
```

**多個角色：**
```bash
# 所有角色
bash scripts/batch/train_all_sdxl_sequential.sh

# 指定角色
bash scripts/batch/train_all_sdxl_sequential.sh miguel bryce alberto

# 配置文件
bash scripts/batch/train_all_sdxl_sequential.sh --config my_list.txt
```

**查看結果：**
```bash
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/final_comparison_summary.txt
```

**系統會自動處理一切！** 🚀
