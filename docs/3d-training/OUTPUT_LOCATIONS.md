# SDXL LoRA Training - Checkpoints 和 Evaluation 結果位置

**最後更新:** 2025-11-28

本文檔詳細說明所有訓練輸出、checkpoints 和 evaluation 結果的存放位置。

---

## 📁 Checkpoints 位置

### 主要輸出目錄

**正確的基礎路徑:**
```
/mnt/c/ai_models/diffusion/lora/sdxl/{movie}/{character}_identity/
```

**範例 (Elio):**
```
/mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/
```

**⚠️ 注意:** 當前 Elio 訓練使用臨時路徑 `/mnt/data/training/lora/elio/elio_identity/`，訓練完成後需要移動到正確位置。Tyler 和 Caleb 已配置為正確路徑。

### Checkpoint 檔案結構

訓練完成後，目錄結構如下：

```
/mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/
├── elio_identity_lora_sdxl-000002.safetensors    # Epoch 2 checkpoint (~1.8 GB)
├── elio_identity_lora_sdxl-000004.safetensors    # Epoch 4 checkpoint
├── elio_identity_lora_sdxl-000006.safetensors    # Epoch 6 checkpoint ⭐ 重點測試
├── elio_identity_lora_sdxl-000008.safetensors    # Epoch 8 checkpoint ⭐ 重點測試
├── elio_identity_lora_sdxl-000010.safetensors    # Epoch 10 checkpoint (final) ⭐ 重點測試
├── logs/
│   └── elio_sdxl{timestamp}/
│       ├── network_train/
│       │   └── events.out.tfevents.*              # TensorBoard 日誌
│       └── sample/                                 # 訓練期間生成的樣本圖片
│           ├── sample-000002.png                   # Epoch 2 樣本
│           ├── sample-000004.png                   # Epoch 4 樣本
│           ├── sample-000006.png                   # Epoch 6 樣本
│           ├── sample-000008.png                   # Epoch 8 樣本
│           └── sample-000010.png                   # Epoch 10 樣本
└── eval_{timestamp}/                               # Evaluation 結果
    ├── elio_identity_lora_sdxl-000006/
    │   ├── prompt_001.png
    │   └── ...
    └── comparison_report.json
```

### 所有角色的 Checkpoint 位置

| 角色 | 電影 | Checkpoint 位置 |
|------|------|----------------|
| **Elio** | Elio (2025) | `/mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/*.safetensors` ⚠️ 需移動 |
| **Tyler** | Turning Red | `/mnt/c/ai_models/diffusion/lora/sdxl/turning-red/tyler_identity/*.safetensors` |
| **Caleb** | Elio (2025) | `/mnt/c/ai_models/diffusion/lora/sdxl/elio/caleb_identity/*.safetensors` |
| **Orion** | Elio (2025) | `/mnt/c/ai_models/diffusion/lora/sdxl/orion/orion_identity/*.safetensors` ✅ |
| **Miguel** | Up (2009) | `/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors` ✅ |
| **Alberto** | Luca (2021) | `/mnt/c/ai_models/diffusion/lora/sdxl/luca/alberto_identity/*.safetensors` ✅ |

### 列出 Checkpoints 的命令

```bash
# 列出所有 checkpoints（按時間排序，最新在前）
ls -lht /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/*.safetensors

# 列出所有 checkpoints（按名稱排序）
ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/*.safetensors | sort

# 檢查 checkpoint 大小
du -h /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/*.safetensors

# 統計 checkpoint 數量
ls /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/*.safetensors | wc -l
```

### ⚠️ Elio 訓練完成後：移動 Checkpoints

當前 Elio 訓練輸出到臨時位置，訓練完成後需要移動到正確位置：

```bash
# 使用提供的腳本自動移動（推薦）
bash scripts/batch/move_elio_checkpoints.sh

# 或手動移動
mkdir -p /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity
mv /mnt/data/training/lora/elio/elio_identity/*.safetensors /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/
mv /mnt/data/training/lora/elio/elio_identity/logs /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity/

# 創建 symlink（可選，保持向後兼容）
ln -s /mnt/c/ai_models/diffusion/lora/sdxl/elio/elio_identity /mnt/data/training/lora/elio/elio_identity
```

---

## 📊 訓練期間自動生成的樣本圖片

### 樣本圖片位置

訓練過程中，Kohya_ss 會根據 `sample_prompts` 文件每 2 個 epochs 生成樣本圖片：

```
/mnt/data/training/lora/elio/elio_identity/logs/elio_sdxl{timestamp}/sample/
├── sample-000002.png         # Epoch 2 生成的樣本
├── sample-000004.png         # Epoch 4 生成的樣本
├── sample-000006.png         # Epoch 6 生成的樣本
├── sample-000008.png         # Epoch 8 生成的樣本
└── sample-000010.png         # Epoch 10 生成的樣本
```

### Sample Prompts 配置位置

```
/mnt/c/ai_projects/3d-animation-lora-pipeline/prompts/lora_testing/
├── elio_identity_test.txt
├── tyler_identity_test.txt
└── caleb_identity_test.txt
```

### 查看樣本圖片

```bash
# 列出所有樣本圖片
ls -lht /mnt/data/training/lora/elio/elio_identity/logs/*/sample/*.png

# 使用圖片查看器打開
# Linux
eog /mnt/data/training/lora/elio/elio_identity/logs/*/sample/*.png

# 或複製到容易訪問的位置
cp /mnt/data/training/lora/elio/elio_identity/logs/*/sample/*.png ~/Desktop/elio_samples/
```

---

## 🧪 Evaluation 結果位置

### 評估腳本輸出位置

使用 `sdxl_lora_evaluator.py` 評估 checkpoints 時，結果會存放在：

**默認位置:**
```
/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/evaluation/{character}_epoch{N}/
```

**範例 (評估 Elio Epoch 6):**
```
/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/evaluation/elio_epoch6/
├── evaluation_report.json              # 評估指標和統計數據
├── prompt_001_seed_42.png             # 測試圖片 1
├── prompt_002_seed_42.png             # 測試圖片 2
├── prompt_003_seed_42.png             # 測試圖片 3
├── ...
├── comparison_grid.png                 # 所有測試圖片的網格對比
└── metadata.json                       # 評估配置和參數
```

### Evaluation 命令範例

```bash
# 評估單個 checkpoint (Epoch 6)
python scripts/evaluation/sdxl_lora_evaluator.py \
  --checkpoint /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000006.safetensors \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir outputs/evaluation/elio_epoch6

# 評估多個 checkpoints（批次比較）
python scripts/evaluation/sdxl_lora_evaluator.py \
  --checkpoint /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000006.safetensors \
  --checkpoint /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000008.safetensors \
  --checkpoint /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000010.safetensors \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir outputs/evaluation/elio_comparison
```

### 完整的 Evaluation 目錄結構

```
/mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/
└── evaluation/
    ├── elio_epoch6/                    # Epoch 6 評估結果
    │   ├── evaluation_report.json
    │   ├── *.png
    │   └── comparison_grid.png
    ├── elio_epoch8/                    # Epoch 8 評估結果
    ├── elio_epoch10/                   # Epoch 10 評估結果
    ├── elio_comparison/                # 多 checkpoint 對比
    │   ├── epoch6_vs_epoch8_vs_epoch10.png
    │   ├── detailed_comparison.json
    │   └── ...
    ├── tyler_epoch6/                   # Tyler 評估結果
    └── caleb_epoch6/                   # Caleb 評估結果
```

---

## 📝 訓練日誌位置

### 主要日誌文件

**訓練執行日誌:**
```
/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/
├── elio_elio_identity_sdxl_training_20251128_070639.log
├── turning-red_tyler_identity_sdxl_training_*.log
└── elio_caleb_identity_sdxl_training_*.log
```

**TensorBoard 日誌:**
```
/mnt/data/training/lora/{movie}/{character}_identity/logs/{character}_sdxl{timestamp}/
└── network_train/
    └── events.out.tfevents.*
```

### 查看日誌

```bash
# 即時查看訓練日誌
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log

# 查看最後 100 行
tail -100 /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log

# 搜尋特定內容（例如 loss）
grep -i "loss\|error" /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/elio_elio_identity_sdxl_training_*.log

# 查看所有訓練日誌
ls -lht /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/*_training_*.log
```

---

## 🗂️ 完整的資料夾結構總覽

```
/mnt/data/training/lora/
├── elio/
│   ├── elio_identity/
│   │   ├── *.safetensors                       # ✅ Checkpoints
│   │   └── logs/
│   │       └── elio_sdxl{timestamp}/
│   │           ├── network_train/              # ✅ TensorBoard 日誌
│   │           └── sample/                     # ✅ 訓練樣本圖片
│   ├── caleb_identity/
│   │   └── ...
│   └── orion_identity/
│       └── ...
├── turning-red/
│   └── tyler_identity/
│       └── ...
└── up/
    └── miguel_identity/
        └── ...

/mnt/c/ai_projects/3d-animation-lora-pipeline/
├── outputs/
│   └── evaluation/                             # ✅ Evaluation 結果
│       ├── elio_epoch6/
│       ├── elio_epoch8/
│       └── ...
└── logs/                                       # ✅ 訓練執行日誌
    ├── elio_elio_identity_sdxl_training_*.log
    └── ...
```

---

## 🔍 快速查找命令

### Checkpoints

```bash
# 查找所有 SDXL checkpoints
find /mnt/data/training/lora -name "*_lora_sdxl-*.safetensors" -type f

# 查找特定角色的 checkpoints
find /mnt/data/training/lora -path "*/elio_identity/*.safetensors" -type f

# 查找最新的 checkpoint
find /mnt/data/training/lora -name "*.safetensors" -type f -printf '%T+ %p\n' | sort -r | head -5
```

### 樣本圖片

```bash
# 查找所有訓練樣本圖片
find /mnt/data/training/lora -path "*/sample/*.png" -type f

# 查找特定 epoch 的樣本
find /mnt/data/training/lora -name "sample-000006.png" -type f
```

### Evaluation 結果

```bash
# 列出所有 evaluation 結果
ls -lh /mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/evaluation/

# 查找所有 evaluation reports
find /mnt/c/ai_projects/3d-animation-lora-pipeline/outputs/evaluation -name "evaluation_report.json"
```

---

## 💾 磁碟空間估算

### 預期檔案大小

| 項目 | 單個大小 | Elio (10 epochs) | Tyler (10 epochs) | Caleb (10 epochs) |
|------|---------|------------------|-------------------|-------------------|
| **Checkpoint** | ~1.8 GB | 5 × 1.8 GB = 9 GB | 5 × 1.8 GB = 9 GB | 5 × 1.8 GB = 9 GB |
| **TensorBoard 日誌** | ~50-200 MB | ~200 MB | ~200 MB | ~200 MB |
| **樣本圖片** | ~2-5 MB/張 | 5 × 3 MB = 15 MB | 5 × 3 MB = 15 MB | 5 × 3 MB = 15 MB |
| **訓練日誌** | ~10-50 MB | ~50 MB | ~50 MB | ~50 MB |
| **總計** | - | **~9.3 GB** | **~9.3 GB** | **~9.3 GB** |

**3 個角色總計:** ~28 GB

### 清理舊 Checkpoints

```bash
# 只保留最後 3 個 checkpoints（配置中已設定）
# save_last_n_epochs = 3

# 手動刪除特定 checkpoint
rm /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000002.safetensors

# 清理所有非最終 checkpoints（小心使用！）
# 先列出要刪除的文件
find /mnt/data/training/lora -name "*-00000[24].safetensors" -type f

# 確認後再刪除
# find /mnt/data/training/lora -name "*-00000[24].safetensors" -type f -delete
```

---

## 📋 Checkpoints 管理最佳實踐

### 1. 定期備份重要 Checkpoints

```bash
# 備份最佳 checkpoint
BACKUP_DIR="/mnt/data/backups/lora_checkpoints/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
cp /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000006.safetensors "$BACKUP_DIR/"
```

### 2. 重命名最佳 Checkpoint

```bash
# 重命名為更有意義的名稱
cp /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl-000006.safetensors \
   /mnt/data/training/lora/elio/elio_identity/elio_identity_lora_sdxl_FINAL_BEST.safetensors
```

### 3. 評估後整理

評估完成後，建議：
- 保留 **Epoch 6, 8, 10** 的 checkpoints
- 刪除 Epoch 2, 4 (如果磁碟空間不足)
- 備份最佳 checkpoint 到安全位置
- 保留 evaluation 報告和對比圖片

---

## 🔗 相關文件

- [SDXL Training Complete Guide](./SDXL_TRAINING_COMPLETE_GUIDE.md) - 完整訓練流程
- [Evaluation Guide](./guides/EVALUATE_AFTER_TRAINING.md) - 評估方法
- [Data Model Structure](../configuration/data_model_structure.md) - 數據結構說明

---

**最後更新:** 2025-11-28
