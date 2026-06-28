# SDXL Character LoRA 訓練完整指南

**最後更新:** 2025-11-28

本指南整合了 SDXL 角色 LoRA 訓練的完整流程，包括配置、啟動、監控和故障排除。

---

## 目錄

- [快速開始](#快速開始)
- [訓練配置](#訓練配置)
- [啟動訓練](#啟動訓練)
- [監控訓練](#監控訓練)
- [TensorBoard 使用](#tensorboard-使用)
- [訓練完成後](#訓練完成後)
- [故障排除](#故障排除)
- [下一步計畫](#下一步計畫)

---

## 快速開始

### 前置條件

確保以下路徑和依賴正確：

```bash
# 驗證 Kohya_ss 安裝
ls /mnt/c/ai_projects/kohya_ss/sd-scripts/

# 驗證基礎模型
ls /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors

# 檢查 conda 環境
conda activate kohya_ss
python -c "import bitsandbytes; print('✅ bitsandbytes installed')"
python -c "import tensorboard; print('✅ tensorboard installed')"
```

### 啟動訓練（以 Elio 為例）

```bash
# 創建 tmux session
tmux new-session -d -s elio_sdxl_training

# 進入工作目錄
tmux send-keys -t elio_sdxl_training "cd /mnt/c/ai_projects/kohya_ss/sd-scripts" C-m

# 激活 conda 環境
tmux send-keys -t elio_sdxl_training "conda activate kohya_ss" C-m

# 啟動訓練
tmux send-keys -t elio_sdxl_training "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml" C-m

# 連接到 session 查看進度
tmux attach -t elio_sdxl_training
```

**離開 tmux:** 按 `Ctrl+B` 然後按 `D` (detach，訓練繼續)

---

## 訓練配置

### 關鍵參數說明

所有角色配置位於：`configs/training/character_loras_sdxl/`

#### 基礎配置

```toml
[model]
pretrained_model_name_or_path = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
sdxl = true
network_dim = 64        # LoRA rank
network_alpha = 32      # 通常設為 rank 的一半

[paths]
train_data_dir = "/mnt/data/datasets/general/{movie}/lora_data/training_data_sdxl/{character}_identity"
output_dir = "/mnt/data/training/lora/{movie}/{character}_identity"
logging_dir = "/mnt/data/training/lora/{movie}/{character}_identity/logs"

[training]
max_train_epochs = 10           # 訓練 epochs（從 4 增加到 10）
save_every_n_epochs = 2         # 每 2 epochs 保存
save_last_n_epochs = 3          # 保留最後 3 個 checkpoints
learning_rate = 0.0001          # UNet 學習率
text_encoder_lr = 0.00006       # Text encoder 學習率
```

#### 數據集結構

```
train_data_dir/
└── {repeats}_{character}/
    ├── image_001.png
    ├── image_001.txt      # caption
    ├── image_002.png
    ├── image_002.txt
    └── ...
```

**Repeats 配置建議:**
- 200-300 張圖片: 5x repeats
- 300-400 張圖片: 4-5x repeats
- 400+ 張圖片: 3-4x repeats

**目前配置:**
- Elio: 301 images × 5 repeats = 1,505 images/epoch → 753 steps/epoch
- Tyler: 276 images × 9 repeats = 2,484 images/epoch → 1,242 steps/epoch
- Caleb: 195 images × 13 repeats = 2,535 images/epoch → 1,268 steps/epoch ⚠️ 可能需要降低

---

## 啟動訓練

### 使用通用訓練腳本（推薦）

```bash
# 查看用法
bash scripts/batch/train_sdxl_with_auto_eval.sh --help

# 啟動訓練（會自動創建 tmux session）
bash scripts/batch/train_sdxl_with_auto_eval.sh \
  configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml
```

### 手動啟動（更靈活）

參考上方「快速開始」中的步驟。

### 啟動 TensorBoard

```bash
# 創建 TensorBoard session
tmux new-session -d -s elio_tensorboard

# 進入日誌目錄
tmux send-keys -t elio_tensorboard "cd /mnt/data/training/lora/elio/elio_identity/logs" C-m

# 激活環境
tmux send-keys -t elio_tensorboard "conda activate kohya_ss" C-m

# 啟動 TensorBoard
tmux send-keys -t elio_tensorboard "tensorboard --logdir . --port 6006 --bind_all" C-m
```

---

## 監控訓練

### 方法 1: 通用監控腳本（推薦）

```bash
bash scripts/batch/monitor_sdxl_training.sh elio
```

**顯示內容:**
- GPU 狀態（溫度、使用率、VRAM、功耗）
- 訓練進程資訊
- 最新訓練輸出
- Checkpoints 列表
- 訓練日誌摘要

### 方法 2: 直接連接 tmux session

```bash
# 連接到訓練 session
tmux attach -t elio_sdxl_training

# 列出所有 sessions
tmux list-sessions

# 離開 tmux（不停止訓練）
# 按 Ctrl+B 然後 D
```

### 方法 3: 持續監控 GPU

```bash
watch -n 2 nvidia-smi
```

### 方法 4: 查看訓練日誌

```bash
# 實時查看
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log

# 查看最後 100 行
tail -100 /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log

# 搜尋關鍵指標
grep -E "(epoch|loss)" /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log | tail -20
```

### 關鍵監控指標

#### ✅ 健康指標

| 指標 | 正常範圍 | 說明 |
|------|---------|------|
| **GPU 使用率** | 85-100% | 訓練充分利用 GPU |
| **VRAM 使用** | 12-15 GB | SDXL 需要較多 VRAM |
| **訓練速度** | 1.2-2.5 sec/step | 依硬體而異 |
| **GPU 溫度** | 50-75°C | 散熱正常 |
| **功耗** | 200-350W | RTX 5080/4090 範圍 |
| **Loss (Epoch 1)** | 0.05-0.15 | 起始 loss |

#### ⚠️ 警告信號

- GPU 使用率 < 80% → 數據載入瓶頸
- VRAM > 15.5 GB → 接近 OOM
- 訓練速度 > 3 sec/step → 異常慢
- Loss 出現 NaN 或 Inf → 訓練爆炸
- Loss 在 Epoch 3 後不下降 → 學習率或數據問題

### 預期 Loss 曲線

| Epoch Range | 預期 Loss | 狀態 |
|-------------|----------|------|
| **Epoch 1-2** | 0.05 - 0.15 | 起始階段 |
| **Epoch 3-5** | 0.03 - 0.08 | 快速下降 |
| **Epoch 6-10** | 0.02 - 0.05 | 收斂穩定 |

如果 Epoch 5 後 Loss > 0.1，可能需要檢查配置。

---

## TensorBoard 使用

### 訪問 TensorBoard

**本地訪問:**
```
http://localhost:6006
```

**遠程訪問:**
```
http://YOUR_SERVER_IP:6006
```

### TensorBoard 顯示內容

- **SCALARS:** Loss 曲線、Learning rate 變化
- **IMAGES:** 樣本生成圖片（每 2 epochs）
- **GRAPHS:** 模型架構圖
- **DISTRIBUTIONS:** 權重分佈

### 重啟 TensorBoard

```bash
# 檢查是否運行
tmux has-session -t elio_tensorboard

# 停止
tmux kill-session -t elio_tensorboard

# 重新啟動
tmux new-session -d -s elio_tensorboard \
  "cd /mnt/data/training/lora/elio/elio_identity/logs && \
   conda activate kohya_ss && \
   tensorboard --logdir . --port 6006 --bind_all"
```

---

## 訓練完成後

### 1. 檢查 Checkpoints

```bash
# 列出所有 checkpoints（按時間排序）
ls -lht /mnt/data/training/lora/elio/elio_identity/*.safetensors

# 檢查 checkpoint 大小（應該約 1.5-2 GB）
du -h /mnt/data/training/lora/elio/elio_identity/*.safetensors
```

**預期 Checkpoints:**
- `{character}_identity_lora_sdxl-000002.safetensors` (Epoch 2)
- `{character}_identity_lora_sdxl-000004.safetensors` (Epoch 4)
- `{character}_identity_lora_sdxl-000006.safetensors` (Epoch 6) ⭐ 重點測試
- `{character}_identity_lora_sdxl-000008.safetensors` (Epoch 8) ⭐ 重點測試
- `{character}_identity_lora_sdxl-000010.safetensors` (Epoch 10) ⭐ 重點測試

### 2. 評估 Checkpoints

```bash
python scripts/evaluation/sdxl_lora_evaluator.py \
  --checkpoint /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000006.safetensors \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir outputs/evaluation/elio_epoch6
```

### 3. 比較不同 Epochs

重點測試 Epoch 6, 8, 10，比較：
- 角色相似度
- 細節保留
- 泛化能力（不同 prompts）
- 過擬合跡象

### 4. 查看 TensorBoard 曲線

訪問 TensorBoard 查看完整訓練過程，確認：
- Loss 是否正常下降
- 沒有異常波動
- 樣本圖片質量提升

---

## 故障排除

### 訓練突然停止

```bash
# 查看最後的錯誤訊息
tail -50 /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log

# 檢查 GPU 狀態
nvidia-smi

# 檢查磁碟空間
df -h /mnt/data/training/

# 檢查 tmux session 狀態
tmux list-sessions
tmux capture-pane -t elio_sdxl_training -p -S -200
```

**常見原因:**
- OOM（記憶體不足）→ 減少 batch_size 或啟用 gradient_checkpointing
- 磁碟空間不足 → 清理舊 checkpoints
- 數據集路徑錯誤 → 檢查 train_data_dir

### Loss 出現 NaN

```bash
# 停止訓練
tmux send-keys -t elio_sdxl_training C-c

# 檢查配置
cat configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml | grep -E "(learning_rate|min_snr_gamma)"
```

**解決方案:**
- 降低學習率（例如從 0.0001 → 0.00005）
- 檢查 min_snr_gamma 設置
- 檢查數據集是否有損壞的圖片

### TensorBoard 無法訪問

```bash
# 檢查是否運行
ps aux | grep tensorboard

# 檢查端口佔用
lsof -i :6006

# 重啟 TensorBoard
tmux kill-session -t elio_tensorboard
tmux new-session -d -s elio_tensorboard \
  "cd /mnt/data/training/lora/elio/elio_identity/logs && \
   conda activate kohya_ss && \
   tensorboard --logdir . --port 6006 --bind_all"
```

### 訓練速度過慢

**檢查瓶頸:**

```bash
# GPU 使用率
nvidia-smi dmon -s u

# 數據載入器設置
grep "max_data_loader_n_workers" configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml
```

**優化建議:**
- 增加 `max_data_loader_n_workers` (當前 6)
- 啟用 `cache_latents = true`
- 啟用 `cache_latents_to_disk` (如果 RAM 不足)
- 減少 `vae_batch_size` (如果 VRAM 緊張)

### 緊急停止訓練

```bash
# 優雅停止（完成當前 step）
tmux send-keys -t elio_sdxl_training C-c

# 強制停止 tmux session
tmux kill-session -t elio_sdxl_training

# 查找並終止所有訓練進程
pkill -f sdxl_train_network
```

---

## 下一步計畫

### 訓練其他角色

#### Tyler (Turning Red)

```bash
tmux new-session -d -s tyler_sdxl_training
tmux send-keys -t tyler_sdxl_training "cd /mnt/c/ai_projects/kohya_ss/sd-scripts" C-m
tmux send-keys -t tyler_sdxl_training "conda activate kohya_ss" C-m
tmux send-keys -t tyler_sdxl_training "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/turning-red_tyler_identity_sdxl.toml" C-m

# 監控
bash scripts/batch/monitor_sdxl_training.sh tyler
```

**預估時間:** 3-4 小時 (276 images × 9 repeats × 10 epochs)

#### Caleb (Elio)

⚠️ **建議先調整 repeats:**

```bash
# 當前: 195 images × 13 repeats = 2,535 steps/epoch (可能過多)
# 建議: 195 images × 6-7 repeats = 1,200-1,400 steps/epoch

# 修改配置文件中的資料夾名稱
# 從 13_caleb 改為 6_caleb 或 7_caleb
```

### 時間估算表

| 角色 | 圖片數 | Repeats | Steps/Epoch | 總 Steps | 預估時間 |
|------|--------|---------|-------------|---------|---------|
| Elio | 301 | 5 | 753 | 7,530 | ~3.8 小時 |
| Tyler | 276 | 9 | 1,242 | 12,420 | ~6.2 小時 |
| Caleb | 195 | 13 | 1,268 | 12,680 | ~6.3 小時 |
| **總計** | - | - | - | 32,630 | ~16.3 小時 |

**建議策略:**
1. 先完成並評估 Elio (~4 小時)
2. 根據 Elio 結果調整 Tyler/Caleb 參數
3. 依序訓練 Tyler 和 Caleb，或使用多 GPU 並行

### 批次訓練（順序）

如果單 GPU，可以創建順序訓練腳本：

```bash
#!/bin/bash
# scripts/batch/train_all_sdxl_sequential.sh

# Elio
bash scripts/batch/train_sdxl_with_auto_eval.sh \
  configs/training/character_loras_sdxl/elio_elio_identity_sdxl.toml

# Tyler
bash scripts/batch/train_sdxl_with_auto_eval.sh \
  configs/training/character_loras_sdxl/turning-red_tyler_identity_sdxl.toml

# Caleb
bash scripts/batch/train_sdxl_with_auto_eval.sh \
  configs/training/character_loras_sdxl/elio_caleb_identity_sdxl.toml
```

---

## 附錄

### 重要檔案位置

**配置文件:**
- 訓練配置: `configs/training/character_loras_sdxl/*.toml`
- 訓練腳本: `scripts/batch/train_sdxl_with_auto_eval.sh`
- 監控腳本: `scripts/batch/monitor_sdxl_training.sh`

**輸出位置:**
- Checkpoints: `/mnt/data/training/lora/{movie}/{character}_identity/*.safetensors`
- TensorBoard 日誌: `/mnt/data/training/lora/{movie}/{character}_identity/logs/`
- 訓練日誌: `/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/*_training_*.log`

**數據集:**
- SDXL 訓練數據: `/mnt/data/datasets/general/{movie}/lora_data/training_data_sdxl/{character}_identity/`

### 快速命令速查

```bash
# === 訓練管理 ===
# 啟動訓練
tmux new-session -d -s {char}_sdxl_training
tmux send-keys -t {char}_sdxl_training "cd /mnt/c/ai_projects/kohya_ss/sd-scripts && conda activate kohya_ss" C-m
tmux send-keys -t {char}_sdxl_training "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/{movie}_{char}_identity_sdxl.toml" C-m

# 連接 session
tmux attach -t {char}_sdxl_training

# 停止訓練
tmux send-keys -t {char}_sdxl_training C-c

# === 監控 ===
# 通用監控
bash scripts/batch/monitor_sdxl_training.sh {char}

# GPU 監控
watch -n 2 nvidia-smi

# 日誌監控
tail -f logs/{movie}_{char}_identity_sdxl_training_*.log

# === TensorBoard ===
# 啟動
tmux new-session -d -s {char}_tensorboard "cd /mnt/data/training/lora/{movie}/{char}_identity/logs && conda activate kohya_ss && tensorboard --logdir . --port 6006 --bind_all"

# 訪問
http://localhost:6006

# === Checkpoints ===
# 列出
ls -lht /mnt/data/training/lora/{movie}/{char}_identity/*.safetensors

# 檢查大小
du -h /mnt/data/training/lora/{movie}/{char}_identity/*.safetensors
```

---

## 版本歷史

- **v1.0** (2025-11-28): 初始版本，整合監控、啟動、故障排除指南
- 修正所有路徑問題 (PROJECT_ROOT, KOHYA_SS_DIR, BASE_MODEL)
- 增加 epochs 從 4 到 10
- 添加 TensorBoard 支援

---

**訓練愉快！🚀**
