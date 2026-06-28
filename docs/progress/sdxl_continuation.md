# SDXL LoRA Training Plan - Batch 2 (Optimized)
## 基於 Batch 1 經驗的優化訓練

---

## 📋 待訓練角色

| 角色 | 電影 | 圖片數 | Repeats | Steps/Epoch | Learning Rate | Dropout |
|------|------|--------|---------|-------------|---------------|---------|
| Barley Lightfoot | Onward | 254 | **4x** | **1016** | 0.00005 (低) | 0.05 |
| Alberto (人類) | Luca | 509 | **2x** | **1018** | 0.00007 (中) | 0.03 |
| Giulia | Luca | 273* | **2x** | **546** | 0.00007 (中) | 0.03 |
| Russell | Up | 243 | **4x** | **972** | 0.00005 (低) | 0.05 |

**Total:** 4 角色, 8 checkpoints (2 epochs × 4 characters)

***註:** Giulia 實際使用 273 張圖片 (從 10_giulia 子目錄複製),原預期 546 張來自兩個不同子目錄。273 × 2 repeats = 546 steps/epoch,仍然足夠訓練。*

**Repeats 策略:**
- **小數據集 (< 300 images):** 4 repeats → 確保約 1000 steps/epoch
- **大數據集 (> 500 images):** 2 repeats → 避免過度重複,保持約 1000 steps/epoch

---

## 🎯 優化策略

### 基於 Batch 1 的關鍵發現:

**問題:**
- 所有角色在 Epoch 1 達到最佳效果
- Epoch 2-5 出現品質下降和模糊化
- Learning Rate 0.0001 對小數據集過於激進

**解決方案:**

1. **降低 Learning Rate**
   - 小數據集 (< 300 images): `0.00005` (原先的 50%)
   - 中數據集 (> 500 images): `0.00007` (原先的 70%)

2. **早停 (Early Stopping)**
   - 只訓練 **2 epochs**
   - 每個 epoch 都保存 checkpoint

3. **輕量正則化**
   - 小數據集: dropout 0.05
   - 大數據集: dropout 0.03

4. **保持成功配置**
   - Min SNR Gamma: 5.0
   - Noise Offset: 0.05
   - Cosine LR scheduler (single cycle)
   - AdamW8bit optimizer

---

## 📊 配置對比

### Batch 1 (有 Overfitting 問題)
```toml
learning_rate = 0.0001
max_train_epochs = 5
network_dropout = 0.0
結果: Epoch 1 最佳,後續品質下降
```

### Batch 2 (優化版本)
```toml
# 小數據集
learning_rate = 0.00005  # ↓ 50%
max_train_epochs = 2      # ↓ 60%
network_dropout = 0.05    # ↑ 加入輕量正則化

# 中數據集
learning_rate = 0.00007  # ↓ 30%
max_train_epochs = 2      # ↓ 60%
network_dropout = 0.03    # ↑ 加入最小正則化
```

---

## 🔧 配置文件

已創建以下配置:

1. `/configs/training/character_loras_sdxl/onward_barley_lightfoot_sdxl.toml`
2. `/configs/training/character_loras_sdxl/luca_alberto_human_sdxl.toml`
3. `/configs/training/character_loras_sdxl/luca_giulia_sdxl.toml`
4. `/configs/training/character_loras_sdxl/up_russell_sdxl.toml`

---

## 📁 數據準備狀態

### 需要從 training_data 複製到 training_data_sdxl:

```bash
# Barley
Source: /mnt/data/datasets/general/onward/lora_data/training_data/barley_lightfoot_identity
Target: /mnt/data/datasets/general/onward/lora_data/training_data_sdxl/barley_lightfoot_identity
Status: ⏳ 待複製

# Alberto (human)
Source: /mnt/data/datasets/general/luca/lora_data/training_data/alberto_identity
Target: /mnt/data/datasets/general/luca/lora_data/training_data_sdxl/alberto_identity
Status: ⏳ 待複製

# Giulia
Source: /mnt/data/datasets/general/luca/lora_data/training_data/giulia_identity
Target: /mnt/data/datasets/general/luca/lora_data/training_data_sdxl/giulia_identity
Status: ⏳ 待複製

# Russell
Source: /mnt/data/datasets/general/up/lora_data/training_data/russell_identity
Target: /mnt/data/datasets/general/up/lora_data/training_data_sdxl/russell_identity
Status: ⏳ 待複製
```

---

## ⏱️ 預估訓練時間

基於 Batch 1 的經驗 (~49 min/epoch):

| 角色 | Epochs | 預估時間/角色 |
|------|--------|--------------|
| Barley | 2 | ~1.6 hours |
| Alberto | 2 | ~1.6 hours |
| Giulia | 2 | ~1.6 hours |
| Russell | 2 | ~1.6 hours |

**總預估時間: 約 6.5-7 小時** (Sequential training)

如果使用 parallel training (GPU 允許):
**總預估時間: 約 1.6-2 hours**

---

## 🚀 執行步驟

### 1. 準備訓練數據
```bash
# 執行數據準備腳本
bash scripts/batch/prepare_batch2_data.sh
```

### 2. 開始訓練
```bash
# Sequential training (推薦 - 避免 GPU OOM)
bash scripts/batch/train_batch2_sequential.sh

# 或手動逐一訓練
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/onward_barley_lightfoot_sdxl.toml
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/luca_alberto_human_sdxl.toml
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/luca_giulia_sdxl.toml
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/up_russell_sdxl.toml
```

### 3. 評估結果
```bash
# 評估所有 Batch 2 checkpoints
bash scripts/batch/evaluate_batch2_checkpoints.sh
```

---

## 📈 預期結果

基於優化的配置,預期:

1. **Epoch 1 vs Epoch 2 品質相近**
   - 不再出現顯著的品質下降
   - 兩個 epochs 都應該保持高品質

2. **更好的泛化能力**
   - 降低的 LR 應該防止 overfitting
   - Dropout 提供額外的正則化

3. **最佳 Checkpoint 選擇**
   - Epoch 2 可能與 Epoch 1 相當甚至更好
   - 需要視覺評估來確認

---

## 🔍 評估指標

評估時關注:

1. **圖像清晰度** - 不應該模糊
2. **特徵準確性** - 角色辨識度高
3. **Prompt 響應** - 能生成多樣化姿勢/角度
4. **一致性** - 同一 seed 生成穩定
5. **泛化能力** - 未見過的 prompts 表現良好

---

## 📝 訓練紀錄模板

訓練完成後更新:

```markdown
## Batch 2 Training Results

**Training Date:** [日期]
**Training Duration:** [時間]
**Status:** ✅ COMPLETE

### Best Checkpoints:

| Character | Best Epoch | Quality Score | Notes |
|-----------|------------|---------------|-------|
| Barley | Epoch X | X/10 | [觀察] |
| Alberto | Epoch X | X/10 | [觀察] |
| Giulia | Epoch X | X/10 | [觀察] |
| Russell | Epoch X | X/10 | [觀察] |

### Validation:

- ✅/❌ Epoch 2 quality maintained
- ✅/❌ No blurring observed
- ✅/❌ Good prompt response
- ✅/❌ Consistent generation

### Comparison with Batch 1:

[觀察到的改進或差異]
```

---

## 🎓 預期學習成果

如果此批次成功:

1. **驗證優化策略有效**
   - 降低 LR 確實防止 overfitting
   - 2 epochs 足夠學習角色特徵

2. **建立最佳實踐**
   - 小數據集 (< 300): LR 0.00005, 2 epochs, dropout 0.05
   - 中數據集 (300-600): LR 0.00007, 2 epochs, dropout 0.03
   - 大數據集 (> 600): LR 0.0001, 3-4 epochs, dropout 0.0

3. **優化訓練流程**
   - 減少浪費的計算 (不需要訓練 5 epochs)
   - 更快的迭代週期
   - 更可預測的結果

---

## ⚠️ 風險與應對

### 潛在問題:

1. **LR 太低導致欠擬合**
   - **症狀:** Epoch 2 仍不夠清晰,特徵學習不足
   - **應對:** 下次微調提高 LR 到 0.00006

2. **Dropout 影響學習速度**
   - **症狀:** 2 epochs 不足以學習特徵
   - **應對:** 降低 dropout 或增加到 3 epochs

3. **數據品質問題**
   - **症狀:** 即使優化配置,結果仍差
   - **應對:** 檢查數據集品質,可能需要重新篩選

---

## 📚 參考資料

- Batch 1 訓練紀錄: `TRAINING_RESULTS_BATCH1.md`
- 原始配置 (Batch 1): `configs/training/character_loras_sdxl/*_sdxl.toml` (舊版)
- 優化配置 (Batch 2): 本批次配置文件

---

*Created: 2025-11-29*
*Batch: 2 (Optimized)*
*Status: ⏳ READY TO START*
