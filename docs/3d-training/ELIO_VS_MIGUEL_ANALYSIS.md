# Elio vs Miguel SDXL Training 對比分析

**分析日期:** 2025-11-28

## ✅ 確認：當前訓練的是 SDXL 模型

是的，當前 Elio 訓練使用的是 **SDXL (Stable Diffusion XL)** 模型：

```toml
[model]
pretrained_model_name_or_path = "/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors"
sdxl = true
```

---

## 📊 訓練配置對比

### 基本參數 (完全相同)

| 參數 | Elio | Miguel | 狀態 |
|------|------|--------|------|
| **模型** | SDXL (sd_xl_base_1.0) | SDXL (sd_xl_base_1.0) | ✅ 相同 |
| **Network Dim** | 64 | 64 | ✅ 相同 |
| **Network Alpha** | 32 | 32 | ✅ 相同 |
| **Learning Rate** | 0.0001 | 0.0001 | ✅ 相同 |
| **Text Encoder LR** | 0.00006 | 0.00006 | ✅ 相同 |
| **Optimizer** | AdamW8bit | AdamW8bit | ✅ 相同 |
| **Mixed Precision** | bf16 | bf16 | ✅ 相同 |
| **Batch Size** | 1 | 1 | ✅ 相同 |
| **Gradient Accum** | 2 | 2 | ✅ 相同 |
| **Max Epochs** | 10 | 10 | ✅ 相同 |
| **Save Every** | 2 epochs | 2 epochs | ✅ 相同 |
| **LR Scheduler** | cosine_with_restarts | cosine_with_restarts | ✅ 相同 |

**結論：訓練超參數配置完全一致。**

---

## 🔍 關鍵差異：數據集

### 數據集規模對比

| 項目 | Elio | Miguel | 差異 |
|------|------|--------|------|
| **圖片數量** | 301 | 449 | ⚠️ Elio 少 33% |
| **Repeats** | 5 | 6 | ⚠️ Elio 少 1x |
| **每 Epoch Steps** | ~753 | ~1,347 | ⚠️ Elio 少 44% |
| **總訓練 Steps** | ~7,530 | ~13,470 | ⚠️ Elio 少 44% |

**🔑 關鍵發現：Elio 的訓練數據量明顯少於 Miguel**

---

## 📝 Caption 質量對比

### Caption 長度

| 角色 | 平均字數 | 範圍 |
|------|---------|------|
| **Elio** | ~145 words | 129-175 words |
| **Miguel** | ~138 words | 118-152 words |

**狀態：** ✅ Caption 長度相近

### Caption 質量評估

**Miguel Captions (範例):**
```
miguel_rivera, a young boy with short dark hair and large, expressive brown eyes,
rendered in Pixar's signature photorealistic animation style. He wears a crisp
white collared shirt, vibrant red vest, and tailored brown pants, with a distinctive
guitar-shaped pendant hanging around his neck. The 3D character is meticulously
crafted using advanced physically-based rendering (PBR) techniques...
```

**Elio Captions (範例):**
```
A photorealistic 3D portrait of Elio Solis, a young boy with tousled dark hair
and piercing bright blue eyes, wearing a vibrant emerald green jacket adorned
with intricate golden star patterns and a shimmering iridescent collar. Rendered
in Pixar-inspired animation style with advanced studio lighting...
```

**評估：**
- ✅ 兩者 caption 質量都很高
- ✅ 都包含詳細的技術描述（PBR, subsurface scattering, lighting）
- ✅ 都強調角色特徵和服裝細節
- ✅ 語言風格和結構相似

**狀態：** ✅ Caption 質量相當，不是問題所在

---

## ⚠️ 可能導致質量差異的原因

### 1. **訓練數據量不足** (最可能)

**Miguel:**
- 449 images × 6 repeats × 10 epochs = **26,940 optimization steps**
- 每個圖片被訓練 60 次

**Elio:**
- 301 images × 5 repeats × 10 epochs = **15,050 optimization steps**
- 每個圖片被訓練 50 次

**差異：Elio 的訓練步數少了 44%，每張圖片的訓練次數也少了 17%**

### 2. **數據多樣性可能不足**

**Miguel: 449 張圖片**
- 更多場景變化
- 更多角度和表情
- 更多服裝和光照條件

**Elio: 301 張圖片**
- 相對較少的多樣性
- 可能某些角度/表情/場景覆蓋不足

### 3. **角色本身的複雜度差異**

**Elio 的特點：**
- 有眼罩（獨特識別特徵）
- 綠色外套with星星圖案（複雜紋理）
- 藍色眼睛（顏色特徵）
- 可能服裝變化較多

**Miguel 的特點：**
- 簡單服裝（白襯衫/紅背心）
- 吉他項鍊（標誌性配件）
- 深色頭髮和棕色眼睛
- 相對簡單的視覺特徵

**可能性：Elio 的視覺複雜度更高，需要更多訓練數據來充分學習**

---

## 💡 建議改進方案

### 方案 1：增加訓練數據量（最推薦）

**目標：將 Elio 數據量提升到與 Miguel 相當的水平**

```bash
# 當前
301 images × 5 repeats = 1,505 images/epoch

# 建議選項 A：增加 repeats
301 images × 7 repeats = 2,107 images/epoch  (接近 Miguel)

# 建議選項 B：增加圖片數量
增加到 ~400-450 images × 5-6 repeats = 2,000-2,700 images/epoch
```

**修改配置：**
```toml
# 將資料夾從 5_elio 重命名為 7_elio
# 或者增加更多篩選後的圖片
```

### 方案 2：延長訓練時間

**當前計畫：10 epochs**

**建議：增加到 12-15 epochs**

```toml
max_train_epochs = 15  # 從 10 增加
save_every_n_epochs = 2
save_last_n_epochs = 5  # 保留更多 checkpoints
```

**優點：**
- 不需要修改數據集
- 簡單直接
- 可以通過 loss 曲線判斷是否過擬合

**缺點：**
- 訓練時間增加 50%
- 可能出現過擬合（需密切監控）

### 方案 3：調整學習率策略

**當前使用：cosine_with_restarts (3 cycles)**

**建議：對於較小數據集，可以嘗試：**

```toml
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 5  # 從 3 增加到 5
lr_warmup_steps = 200        # 從 100 增加

# 或者使用更激進的學習率
learning_rate = 0.00012      # 從 0.0001 微調
```

### 方案 4：混合方案（推薦）

**結合多個策略：**

1. **增加 repeats 到 7** (不需要重新準備數據)
2. **訓練 12 epochs** (適度延長)
3. **密切監控 TensorBoard** loss 曲線

**預期效果：**
```
301 images × 7 repeats × 12 epochs = 25,284 steps
接近 Miguel 的 26,940 steps
```

---

## 📈 當前訓練狀態評估

### 已經比之前好的原因

1. ✅ **Epochs 從 4 增加到 10** - 訓練時間增加 150%
2. ✅ **TensorBoard logging 啟用** - 可以準確監控進度
3. ✅ **配置參數正確** - 與 Miguel 完全一致

### 為何可能還未達到 Miguel 水準

1. ⚠️ **訓練數據量不足 44%**
2. ⚠️ **當前只完成了很小一部分** (Epoch 1/10)
3. ⚠️ **Elio 角色可能更複雜** (眼罩、複雜服裝)

---

## 🎯 下一步行動建議

### 短期（當前訓練）

1. **繼續觀察當前訓練**
   - 等待訓練完成
   - 查看 Epoch 6, 8, 10 的效果
   - 通過 TensorBoard 監控 loss 曲線

2. **評估結果**
   - 測試最終 checkpoints
   - 與 Miguel 做 A/B 對比
   - 記錄具體差異

### 中期（如果當前結果不理想）

**選項 A：增加 Repeats（最快）**
```bash
# 1. 重命名資料夾
mv /mnt/data/datasets/general/elio/lora_data/training_data_sdxl/elio_identity/5_elio \
   /mnt/data/datasets/general/elio/lora_data/training_data_sdxl/elio_identity/7_elio

# 2. 更新配置註釋
# Dataset: 301 images × 7 repeats = 2107 steps/epoch
# Target: 12 epochs (25,284 total steps)

# 3. 修改 max_train_epochs
max_train_epochs = 12

# 4. 重新訓練
```

**選項 B：增加圖片數量**
```bash
# 從原始 clustered data 篩選更多高質量圖片
# 目標：400-450 images
```

### 長期

1. **建立數據質量標準**
   - 記錄 Miguel 成功的數據特徵
   - 制定數據準備 checklist
   - 確保未來角色達到相同標準

2. **自動化評估**
   - 開發自動化測試腳本
   - 建立質量基準線（benchmark）
   - 早期發現問題

---

## 📋 Quick Reference

### Miguel 成功配置

```
數據：449 images × 6 repeats × 10 epochs = 26,940 steps
結果：✅ 優秀
Checkpoints：Epoch 6 最佳
```

### Elio 當前配置

```
數據：301 images × 5 repeats × 10 epochs = 15,050 steps
狀態：🔄 訓練中
預期：可能不如 Miguel（數據量少 44%）
```

### Elio 建議配置

```
選項 1：301 images × 7 repeats × 12 epochs = 25,284 steps
選項 2：450 images × 6 repeats × 10 epochs = 27,000 steps
預期：✅ 應該達到 Miguel 水準
```

---

## 結論

**Q: 目前訓練的是 SDXL 嗎？**
**A:** ✅ 是的，100% 確認是 SDXL。

**Q: 為什麼還沒達到 Miguel 水準？**
**A:** 最可能的原因是 **訓練數據量不足**：
- Elio: 15,050 steps (301 images × 5 repeats × 10 epochs)
- Miguel: 26,940 steps (449 images × 6 repeats × 10 epochs)
- 差異: 44% 更少的訓練步數

**Q: 這樣的差異是正常的嗎？**
**A:** ✅ 是的，非常正常。LoRA 訓練對數據量很敏感：
- 數據量減少 44% 通常會導致明顯的質量下降
- 不同角色的複雜度也會影響所需數據量
- Elio 的眼罩和複雜服裝可能需要更多數據

**Q: 如何改善？**
**A:** 最簡單有效的方法：
1. **增加 repeats 到 7** (重命名資料夾即可)
2. **延長到 12 epochs**
3. 總 steps 會接近 Miguel 的水準 (25,284 vs 26,940)

---

**最後更新：** 2025-11-28
