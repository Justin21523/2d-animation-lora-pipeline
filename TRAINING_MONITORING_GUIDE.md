# Inazuma Eleven SDXL LoRA 訓練監控指南

**訓練狀態**: 🟢 進行中
**開始時間**: 2025-12-19 18:42
**預計完成**: 2025-12-22 03:00 (~63小時)

---

## 快速狀態檢查

### 1. 查看訓練進程
```bash
ps aux | grep "[s]dxl_train_network"
```
**正常輸出**: 應該看到Python進程，CPU使用率 > 100%

### 2. 檢查GPU狀態
```bash
watch -n 5 nvidia-smi
```
**預期狀態**:
- VRAM使用: 12-16GB
- GPU利用率: 80-100%
- 溫度: <80°C

### 3. 連接訓練Session
```bash
tmux attach -t inazuma_training
```
**離開tmux**: 按 `Ctrl+B` 然後按 `D`

---

## 詳細監控命令

### 批次訓練進度日誌
```bash
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/inazuma_batch_training_*.log
```
**顯示內容**:
- 當前訓練角色 [X/7]
- 每個角色完成時間
- 預估剩餘時間

### 當前角色訓練日誌
```bash
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/inazuma_endou_mamoru_sdxl_training_*.log
```
**顯示內容**:
- Epoch進度
- Loss值
- Steps/秒

### 檢查已保存的Checkpoint
```bash
# 查看所有角色的checkpoints
for char in endou_mamoru gouenji_shuuya fudou_akio matsukaze_tenma inamori_asuto nosaka_yuuma utsunomiya_toramaru; do
  echo "=== $char ==="
  ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/${char}_identity/*.safetensors 2>/dev/null | tail -3 || echo "尚無checkpoint"
  echo ""
done
```

---

## 訓練時間表 (預估)

| 角色 | 順序 | 預估時間 | 累計時間 |
|------|------|----------|----------|
| Endou Mamoru | 1/7 | 9h | 9h |
| Gouenji Shuuya | 2/7 | 9h | 18h |
| Fudou Akio | 3/7 | 9h | 27h (Day 2) |
| Matsukaze Tenma | 4/7 | 9h | 36h |
| Inamori Asuto | 5/7 | 9h | 45h (Day 3) |
| Nosaka Yuuma | 6/7 | 9h | 54h |
| Utsunomiya Toramaru | 7/7 | 9h | **63h** |

**總時長**: ~63小時 (2.6天)

---

## Checkpoint時間點 (每個角色)

- **Epoch 2**: 訓練開始後 ~1.5小時
- **Epoch 4**: 訓練開始後 ~3小時
- **Epoch 6**: 訓練開始後 ~4.5小時
- **Epoch 8**: 訓練開始後 ~6小時
- **Epoch 10**: 訓練開始後 ~7.5小時
- **Epoch 12** (最終): 訓練開始後 ~9小時

---

## 輸出位置

### LoRA模型
```
/mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/
├── endou_mamoru_identity/
│   ├── inazuma_endou_mamoru_lora_sdxl-000002.safetensors
│   ├── inazuma_endou_mamoru_lora_sdxl-000004.safetensors
│   ├── ...
│   └── inazuma_endou_mamoru_lora_sdxl-000012.safetensors
├── gouenji_shuuya_identity/
└── ...
```

### 訓練日誌
```
/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/
├── inazuma_batch_training_20251219_184220.log
├── inazuma_endou_mamoru_sdxl_training_20251219_184220.log
└── ...
```

### TensorBoard日誌
```
/mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/{character}_identity/logs/
```

---

## TensorBoard監控 (可選)

```bash
# 啟動TensorBoard
conda run -n kohya_ss tensorboard \
  --logdir /mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/endou_mamoru_identity/logs \
  --port 6006

# 在瀏覽器打開: http://localhost:6006
```

**可查看**:
- Loss曲線
- Learning rate變化
- 訓練指標

---

## 常見問題排查

### 1. 訓練意外停止
```bash
# 檢查tmux session是否還在
tmux ls

# 重新連接
tmux attach -t inazuma_training

# 如果session不存在，重新啟動
tmux new-session -s inazuma_training
bash scripts/batch/train_inazuma_sdxl_loras.sh
```

### 2. VRAM不足 (OOM)
```bash
# 檢查GPU進程
nvidia-smi

# 查找占用GPU的進程
ps aux | grep python | grep -E "(train|sdxl)"

# 如需終止
kill -9 <PID>
```

### 3. 查看詳細錯誤
```bash
# 搜尋錯誤訊息
grep -i error /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/*.log

# 查看tmux完整輸出
tmux capture-pane -t inazuma_training -p -S -1000 > /tmp/training_debug.log
less /tmp/training_debug.log
```

---

## 訓練完成後

### 1. 檢查所有Checkpoint
```bash
bash scripts/batch/check_training_progress.sh
```

### 2. 評估最終Checkpoint
```bash
# 評估所有角色的epoch 12 checkpoint
python scripts/batch/evaluate_inazuma_loras.py --checkpoint-epoch 12
```

### 3. 查看評估結果
```
outputs/lora_evaluation/inazuma_sdxl/
├── endou_mamoru/
│   ├── inazuma_endou_mamoru_lora_sdxl-000012/
│   │   ├── prompt_01.png (timeline_original)
│   │   ├── prompt_02.png (timeline_go)
│   │   └── ...
└── ...
```

---

## 快速監控腳本

創建一個快速狀態腳本：

```bash
cat > /tmp/quick_status.sh << 'EOF'
#!/bin/bash
echo "=== Inazuma SDXL Training Status ==="
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo ""
echo "Training Process:"
ps aux | grep "[s]dxl_train_network" | awk '{print "PID: "$2" CPU: "$3"% MEM: "$4"%"}'
echo ""
echo "Latest Batch Log:"
tail -5 /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/inazuma_batch_training_*.log 2>/dev/null || echo "No batch log yet"
echo ""
echo "Checkpoints Count:"
for char in endou_mamoru gouenji_shuuya fudou_akio matsukaze_tenma inamori_asuto nosaka_yuuma utsunomiya_toramaru; do
  count=$(ls /mnt/c/ai_models/diffusion/lora/sdxl/inazuma-eleven/${char}_identity/*.safetensors 2>/dev/null | wc -l)
  echo "  $char: $count checkpoints"
done
EOF

chmod +x /tmp/quick_status.sh

# 使用方式
/tmp/quick_status.sh
```

---

## 預期結果

### 成功標準
- ✅ 每個角色產生6個checkpoints (epoch 2,4,6,8,10,12)
- ✅ 每個checkpoint約90-100MB
- ✅ 訓練日誌無Error訊息
- ✅ Loss曲線平穩下降

### 品質標準
- ✅ 角色辨識度高（用canonical caption測試）
- ✅ Timeline conditioning有效（不同prompt產生對應樣式）
- ✅ 無prompt bleeding（通用prompt不觸發角色）
- ✅ 清晰的2D anime風格線條

---

**訓練進行中，請耐心等待！預計2025-12-22完成。** 🚀
