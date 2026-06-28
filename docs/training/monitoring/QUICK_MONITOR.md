# 🎯 快速查看訓練進度

## 方法 1: 進入 Tmux（推薦⭐）

```bash
tmux attach -t sdxl_training
```

**你會看到類似這樣的即時進度條**：

```
epoch 1/10:  85%|████████████████▌   | 2288/2694 [30:12<05:21,  1.26it/s, loss=0.0421]
              ↑                         ↑          ↑           ↑        ↑
           進度%                    當前/總steps   已用時間  剩餘時間  速度  當前loss
```

**離開 tmux（不停止訓練）**：
```
按 Ctrl+B，然後按 D
```

---

## 方法 2: 查看捕獲的輸出

```bash
# 即時追蹤
tail -f /tmp/training_output.log

# 或使用腳本
bash scripts/batch/watch_training_live.sh
```

---

## 方法 3: 一鍵監控腳本

```bash
bash scripts/batch/monitor_sdxl_training.sh
```

自動顯示：
- GPU 狀態
- 訓練進程
- Checkpoints
- 每 5 秒刷新

---

## 關鍵指標解讀

### 速度指標

| 顯示 | 含義 | 好壞 |
|------|------|------|
| `1.26 it/s` | 每秒 1.26 個 step | ✅ 很好 |
| `0.95 it/s` | 每秒 0.95 個 step | ✅ 正常 |
| `2.35 s/it` | 每個 step 2.35 秒 | ⚠️ 稍慢 |

**目標速度**（優化後）：
- ✅ **0.8-1.3 it/s** 或 **0.8-1.2 秒/step**
- ⭐ **每 epoch 35-45 分鐘**

### Loss 值

| Epoch | 正常範圍 |
|-------|----------|
| 1-2 | 0.05-0.15 |
| 3-5 | 0.03-0.08 |
| 6+ | 0.02-0.05 |

---

## 故障排除

### 看不到進度條

**問題**：Tmux 裡只看到靜態文字，沒有進度條

**解決**：
1. 確認訓練在運行：`ps aux | grep sdxl_train`
2. 可能還在初始化（前 5 分鐘）
3. 等待 epoch 完成會顯示輸出

### 進度條卡住

**問題**：進度條不更新

**解決**：
1. 檢查 GPU：`nvidia-smi`（應該 > 90% 利用率）
2. 檢查進程：`ps aux | grep sdxl_train`
3. 如果 GPU 閒置，訓練可能崩潰了

---

## 預期時間線（Miguel）

```
19:37-19:40  Epoch 1 完成（你會看到第一個進度輸出）
20:12-20:18  Epoch 2 完成（第一個 checkpoint ⭐）
21:22-21:35  Epoch 4
22:32-22:50  Epoch 6
23:42-00:05  Epoch 8
00:52-01:20  Epoch 10 完成 ✅
```

**總時間**: ~6 小時

---

## 快速命令參考

```bash
# 查看訓練（最直觀）
tmux attach -t sdxl_training

# 查看 GPU
nvidia-smi

# 查看進程
ps aux | grep sdxl_train

# 查看 checkpoint
ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors

# 監控腳本
bash scripts/batch/monitor_sdxl_training.sh
```

---

**💡 現在就運行 `tmux attach -t sdxl_training` 查看即時進度吧！**
