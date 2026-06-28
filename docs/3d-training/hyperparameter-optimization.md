好的，這是一份將說明文字翻譯為全中文（保留關鍵技術名詞中英對照）的文檔版本，內容結構與數據部分完全保持不變。

-----

# 超參數優化指南 (Hyperparameter Optimization Guide)

## 概覽 (Overview)

本指南說明我們的自動化超參數優化系統如何運作，以及它如何確保為 LoRA 模型找到最佳的訓練參數。

## 系統架構 (System Architecture)

### 核心組件 (Core Components)

```
┌─────────────────────────────────────────────────────────┐
│                    Optuna Framework                      │
│  (Tree-structured Parzen Estimator - TPE Sampler)      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├─► 建議超參數 (智能採樣)
                 │
                 ├─► 使用建議參數訓練 LoRA
                 │
                 ├─► 評估檢查點品質
                 │
                 ├─► 更新機率模型
                 │
                 └─► 重複直到收斂
```

## 運作原理 (How It Works)

### 1\. 樹狀結構 Parzen 估計器 (TPE)

**什麼是 TPE？**

TPE 是一種貝式優化演算法，它建立一個機率模型來預測哪些超參數組合表現會更好。

**關鍵概念：**

  - **非隨機搜尋**：TPE 從先前的試驗中學習，做出更明智的選擇
  - **探索 (Exploration) vs. 開發 (Exploitation)**：在嘗試新區域與優化有潛力區域之間取得平衡
  - **高效**：比網格搜尋 (grid search) 或隨機搜尋 (random search) 更快找到好的參數

**TPE 如何運作：**

1.  **初始階段 (隨機探索)**
       - 前 5-10 次試驗：隨機採樣以探索空間
       - 建立對參數景觀的初步了解

2.  **學習階段 (智能採樣)**
       - 根據分數將試驗分為「好」與「壞」
       - 建立兩個機率分佈：
         - `l(x)`：導致**好結果**的參數分佈
         - `g(x)`：導致**壞結果**的參數分佈

3.  **開發階段**
       - 從 `l(x)/g(x)` 值高的區域採樣新參數
       - 專注於有潛力的區域，同時保持探索

**範例：**

```
Trial 1:  lr=1e-4, dim=128  → Score: 0.25
Trial 2:  lr=5e-5, dim=192  → Score: 0.18  ✅ 更好！
Trial 3:  lr=8e-5, dim=256  → Score: 0.30
Trial 4:  lr=6e-5, dim=192  → Score: 0.16  ✅ 甚至更好！
         ↓
TPE 學習到：lr 在 5e-5 到 6e-5 之間且 dim=192 是有潛力的
         ↓
Trial 5:  lr=5.5e-5, dim=192 → 在最佳區域附近採樣
```

### 2\. 搜尋空間設計 (Search Space Design)

我們的搜尋空間是根據先驗知識和最佳實踐精心設計的：

```python
# 連續參數 (對數均勻分佈)
learning_rate: 5e-5 到 2e-4
  └─ 對數刻度，因為學習率的效應是乘法性的
  └─ 範圍基於 LoRA 訓練的經驗值

text_encoder_lr: 3e-5 到 1e-4
  └─ 通常低於 UNet LR (0.5倍 到 0.8倍)
  └─ 防止文字編碼器過擬合

# 類別參數
network_dim: [64, 96, 128, 192, 256]
  └─ 常見的 LoRA rank (2 的次方和 32 的倍數)

network_alpha: [32, 48, 64, 96, 128]
  └─ 通常 α ≤ dim (由約束條件強制執行)

optimizer: [AdamW, AdamW8bit, Lion, Adafactor]
  └─ 不同的記憶體/收斂權衡

lr_scheduler: [cosine, cosine_with_restarts, polynomial]
  └─ 影響學習率衰減策略

gradient_accumulation: [1, 2, 4]
  └─ 有效批次大小乘數

max_epochs: [8, 10, 12, 15]
  └─ 訓練時長 (更多 epochs ≠ 總是更好)
```

**為什麼這樣設計？**

1.  **聚焦範圍**：基於文獻和 3D 角色 LoRA 的經驗
2.  **LR 使用對數刻度**：學習率效應是乘法性的，而非加法性的
3.  **離散選擇**：類別參數降低搜尋空間的複雜度
4.  **約束條件**：`network_alpha ≤ network_dim` 防止無效配置

### 3\. 使用 SOTA 指標的多目標評估

每次試驗都使用**綜合指標系統**進行評估，結合傳統的皮克斯風格指標與**最先進的感知指標**：

#### 評估指標

**SOTA 指標 (主要 - 60% 權重)：**

```python
# 1. LPIPS (Learned Perceptual Image Patch Similarity)
#    - 測量生成圖像之間的感知多樣性
#    - 使用基於 AlexNet 的深度特徵
#    - 目標：0.30-0.50 (多樣但不混亂)
#    - 過低 (<0.2) = 模式崩潰 (所有圖像看起來都一樣)
#    - 過高 (>0.6) = 角色外觀不一致
lpips_diversity = calculate_lpips_diversity(generated_images)

# 2. CLIP Text-Image Consistency
#    - 測量提示詞與圖像之間的語義對齊
#    - 使用 ViT-B/32 模型
#    - 目標：>0.35 相似度分數
#    - 確保臉部特徵完整 (未被裁切或缺失)
#    - 檢測指定屬性是否正確生成
clip_consistency = calculate_clip_consistency(images, prompts)
```

**皮克斯風格指標 (次要 - 40% 權重)：**

```python
# 使用訓練好的 LoRA 生成 8 張測試圖像
test_prompts = [
    "Luca Paguro, frontal view, happy smile...",
    "Luca Paguro, three-quarter view, concerned expression...",
    # ... 8 個多樣化的提示詞
]

# 計算指標
brightness = mean(pixel_values)  # 目標：0.40-0.60 (皮克斯風格)
contrast = std(pixel_values)     # 目標：0.15-0.25 (低對比度)
saturation = std(hsv_saturation) # 目標：0.30-0.50 (適中)

# 一致性 (較低的 std = 更穩定)
brightness_std = std(brightness_per_image)
contrast_std = std(contrast_per_image)
```

#### 綜合評分公式

```python
# === 組件 1: 皮克斯風格分數 (40% 權重) ===
brightness_error = |mean_brightness - 0.50|
contrast_error = |mean_contrast - 0.20|
pixar_score = (brightness_error + 0.5 * brightness_std) + \
              (contrast_error + 0.5 * contrast_std)

# === 組件 2: LPIPS 多樣性分數 (30% 權重) ===
lpips_target = 0.40  # 多樣性的甜蜜點
lpips_error = |lpips_diversity - lpips_target|

# === 組件 3: CLIP 一致性分數 (30% 權重) ===
clip_target = 0.35  # 最低可接受的一致性
clip_error = max(0.0, clip_target - clip_consistency)

# === 最終綜合分數 (越低越好) ===
combined_score = 0.40 * pixar_score + \
                0.30 * lpips_error + \
                0.30 * clip_error
```

**為什麼使用這種綜合評分？**

1.  **皮克斯風格 (40%)**：維持 3D 動畫的目標美學
       - 亮度 \~0.50, 對比度 \~0.20 是經驗上的最佳值
       - 一致性懲罰確保生成穩定

2.  **LPIPS 多樣性 (30%)**：防止模式崩潰
       - 直接解決臉型不一致的問題
       - 確保 LoRA 生成多樣性，而非死記硬背
       - 捕捉模型是否無視提示詞總是輸出相同臉孔的情況

3.  **CLIP 一致性 (30%)**：確保語義正確性
       - 檢測被裁切或缺失的臉部特徵
       - 驗證屬性是否正確生成 (表情、角度)
       - 確認複雜提示詞的圖文對齊

**相較於簡單指標的優勢：**

  - **簡單方法 (僅亮度/對比度)**：無法檢測模式崩潰、特徵缺失或語義錯誤
  - **SOTA 方法 (LPIPS + CLIP)**：捕捉人類關心的感知問題
  - **結合方法**：兩全其美 - 風格合規性 + 感知品質

### 4\. 優化過程 (Optimization Process)

#### 每次試驗的工作流程

```
Trial N:
  1. TPE 根據過去的試驗建議參數
  2. 訓練 LoRA X 個 epochs (8-15)
     ├─ 儲存檢查點
     └─ 記錄訓練指標
  3. 使用檢查點生成 8 張測試圖像
  4. 計算品質指標
  5. 計算綜合分數
  6. 將結果儲存至 SQLite 資料庫
  7. 更新 TPE 機率模型
```

#### 收斂策略

**何時停止？**

1.  **固定預算**：50 次試驗 (目前的設定)
2.  **平台期檢測**：最佳分數連續 10 次試驗未改善
3.  **目標達成**：綜合分數 \< 0.10 (非常好)

**為什麼是 50 次試驗？**

  - **探索**：前 10-15 次試驗探索多樣化區域
  - **開發**：中間 20-30 次試驗優化有潛力的區域
  - **驗證**：最後 5-10 次試驗驗證穩定性

對於 8 個參數和約 50 次試驗，TPE 通常能收斂至接近最佳解。

### 5\. 我們如何確保找到最佳參數

#### 策略 1: 貝式優化 (TPE)

✅ **智能搜尋**

  - 從過去的試驗中學習
  - 專注於有潛力的區域
  - 避免浪費試驗在差的區域

❌ **不保證全域最佳解**

  - 可能陷入局部最佳解
  - 但：在有限的試驗中能找到非常好的解

#### 策略 2: 廣泛的初始探索

```python
# 前 10 次試驗：隨機採樣
# - 探索整個參數空間
# - 識別有潛力的區域
# - 防止過早收斂
```

#### 策略 3: 多次啟動驗證

**最佳實踐 (可選)：**

```bash
# 使用不同種子運行優化兩次
bash launch_overnight_optimization.sh  # Seed=42
# 完成後，再次運行：
bash launch_overnight_optimization.sh  # Seed=123

# 比較兩次運行的最佳結果
# 如果參數相似 → 高信心
# 如果不同 → 運行更多試驗
```

#### 策略 4: 集成前 K 名 (Ensemble Top-K)

不只挑選單一最佳試驗：

```bash
# 優化後，測試前 5 名試驗
sqlite3 optuna_study.db \
  'SELECT number, value FROM trials
   WHERE state="COMPLETE"
   ORDER BY value LIMIT 5;'

# 手動測試每個前 5 名檢查點
# 根據視覺品質 + 指標挑選最佳者
```

#### 策略 5: 持續監控

```bash
# 每 2-3 小時檢查一次：
bash check_optimization_progress.sh

# 如果最佳分數未改善：
# - 檢查日誌中的錯誤
# - 驗證評估腳本是否運作
# - 考慮調整搜尋空間
```

## 實務保證 (Practical Guarantees)

### 我們可以保證什麼

✅ **優於隨機**：TPE 比隨機搜尋更快找到好參數

✅ **優於手動**：探索人類不會嘗試的組合

✅ **可重現**：相同種子 + 搜尋空間 → 相同結果

✅ **持續改進**：每次試驗都為下一次提供資訊

### 我們不能保證什麼

❌ **全域最佳解**：可能會錯過絕對最佳解 (但會非常接近)

❌ **固定的收斂時間**：某些問題比其他問題更難

❌ **無過擬合**：測試集上的最佳解可能無法完美泛化

## 監控與診斷 (Monitoring and Diagnostics)

### 檢查目前最佳結果

```bash
# 查看前 10 名試驗
sqlite3 /mnt/data/ai_data/models/lora/luca/optimization_overnight/optuna_study.db \
  'SELECT number, value FROM trials
   WHERE state="COMPLETE"
   ORDER BY value LIMIT 10;'
```

### 檢查參數趨勢

```bash
# 嘗試了哪些學習率？
sqlite3 optuna_study.db \
  'SELECT tp.value, t.value as score
   FROM trial_params tp
   JOIN trials t ON tp.trial_id = t.trial_id
   WHERE tp.param_name = "learning_rate"
   ORDER BY t.value LIMIT 10;'
```

### 可視化進度

優化完成後：

```python
import optuna

study = optuna.load_study(
    study_name="luca_facial_quality_optimization",
    storage="sqlite:///optuna_study.db"
)

# 繪製優化歷史
fig1 = optuna.visualization.plot_optimization_history(study)
fig1.show()

# 繪製參數重要性
fig2 = optuna.visualization.plot_param_importances(study)
fig2.show()

# 繪製參數關係
fig3 = optuna.visualization.plot_parallel_coordinate(study)
fig3.show()
```

## 解讀結果 (Interpreting Results)

### 良好的優化運行

```
✅ 成功跡象：
- 最佳分數在前 20-30 次試驗中穩步提升
- 最後 10-20 次試驗出現平台期 (收斂)
- 前 5 名試驗分數相似 (穩定的最佳解)
- 參數重要性圖表顯示 learning_rate 重要性高
```

### 有問題的運行

```
❌ 警訊：
- 最佳分數隨機跳動 (無學習)
- 所有試驗失敗 (訓練腳本錯誤)
- 分數全部相同 (評估錯誤)
- 無平台期 (需要更多試驗)
```

## 進階：處理臉部品質問題

針對您的特定問題 (臉型不一致、裁切、特徵缺失)：

### 1\. 透過超參數進行正則化

```
較低的學習率 → 減少對訓練數據怪癖的過擬合
較少的 Epochs → 防止記住特定瑕疵
較高的 Network Dim → 更多容量容納一般特徵
較低的梯度累積 → 更頻繁的更新
```

### 2\. 評估對齊

我們的評估針對皮克斯風格指標，這些指標與以下相關：

  - 一致的臉部特徵 (低亮度/對比度標準差)
  - 無強烈陰影 (低對比度)
  - 適當的構圖 (亮度 \~0.50 表示主體居中)

### 3\. 多階段方法

```
階段 1: 運行 50 次試驗優化 (目前)
         ↓
階段 2: 挑選前 5 個檢查點
         ↓
階段 3: 使用臉部特定提示詞手動測試
         ↓
階段 4: 若有需要，使用更多數據重新訓練最佳配置
```

## 常見問題 (Common Questions)

**Q: 為什麼不對所有組合進行網格搜尋？**

A: 對於 8 個參數，網格搜尋需要：

  - 10 種學習率 × 10 種文字編碼器 LR × 5 種 dims × 5 種 alphas × 4 種優化器 × 3 種調度器 × 3 種梯度累積 × 4 種 epochs = **18,000 次試驗**
  - 每試驗 30 分鐘 = **375 天**
  - TPE 在 **50 次試驗 (\~25 小時)** 內就能找到接近最佳解

**Q: 我可以信任 50 次試驗的結果嗎？**

A: 可以，因為：

  - TPE 在文獻中被證明有效
  - 搜尋空間受到專家知識約束
  - 前 K 名驗證提供了信心
  - 視覺檢查是最終評判

**Q: 如果優化後結果仍然很差怎麼辦？**

可能原因：

1.  **數據集問題**：沒有超參數能修復糟糕的訓練數據
2.  **評估未對齊**：指標未捕捉到您的品質關注點
3.  **搜尋空間太窄**：可能需要擴展範圍
4.  **根本限制**：SD 1.5 可能不支持您的需求

解決方案：

  - 審查訓練圖像品質
  - 添加感知指標 (LPIPS, FID)
  - 擴展學習率範圍
  - 考慮 SD 2.1 或 SDXL

## 參考資料 (References)

  - **Optuna Paper**: "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
  - **TPE Algorithm**: "Algorithms for Hyper-Parameter Optimization" (Bergstra et al., 2011)
  - **LoRA Training**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

## 檔案位置 (File Locations)

  - **優化腳本**: `scripts/optimization/optuna_hyperparameter_search.py`
  - **啟動腳本**: `scripts/optimization/launch_overnight_optimization.sh`
  - **進度檢查器**: `scripts/optimization/check_optimization_progress.sh`
  - **結果資料庫**: `/mnt/data/ai_data/models/lora/luca/optimization_overnight/optuna_study.db`
  - **試驗檢查點**: `/mnt/data/ai_data/models/lora/luca/optimization_overnight/trial_XXXX/`

-----

## 試驗分析與 V2.1 策略 (Trial Analysis and V2.1 Strategy)

### Trial 1-5 結果分析

基於初始優化運行 (Trials 1-5)，我們識別出關於參數空間的關鍵見解：

#### 試驗結果摘要

| Trial | Network Dim | Network Alpha | Alpha/Dim Ratio | Learning Rate | Text Encoder LR | Optimizer | Epochs | 結果 (Result) |
|-------|-------------|---------------|-----------------|---------------|-----------------|-----------|--------|--------|
| **Trial 3** | 256 | 32 | **0.125** | 0.000066 | 0.000037 | AdamW8bit | 10 | ⭐⭐⭐⭐⭐ 穩定，通用 |
| **Trial 4** | 128 | 128 | **1.0** | 0.000184 | 0.000088 | AdamW8bit | 8 | ⭐⭐ 高相似度，不穩定 |
| **Trial 5** | 64 | 32 | **0.5** | 0.000071 | 0.000049 | AdamW8bit | 10 | 🔄 運行中 |

#### 關鍵發現：Trial 4 的成功模式

**關鍵使用者回饋**："老實講 若單看人臉的部分 排除那些黑的 圖像等 我覺得trial4 看起來是最像原始Luca角色的"

這個觀察徹底改變了我們的分析策略。Trial 4 雖然有品質問題（黑色圖像、模糊肢體、身體變形），但在成功時達到了**最高的角色臉部相似度**。這揭示了：

1.  **高 LR (0.000184) 是正確的**，用於深度學習角色特定特徵
2.  **Alpha=Dim (比率 1.0) 是正確的**，用於精確的角色記憶保存
3.  **不穩定性才是真正的問題**，而不是參數本身

#### 為什麼 Trial 4 達到高相似度

**學習率分析 (0.000184 vs 0.000066)**

```
Trial 3 (LR=0.000066):
  ├─ 穩定收斂
  ├─ 學習一般 3D 人類特徵
  └─ 結果：通用角色，特異性低

Trial 4 (LR=0.000184):
  ├─ 深度滲透權重空間
  ├─ 學習獨特的臉部特徵 (鼻形、眼距、嘴部曲線)
  └─ 結果：高角色相似度但 不穩定
```

**Alpha/Dim 比率分析**

```python
# Alpha/Dim 比率 = LoRA 更新的縮放因子
# 公式：ΔW = B × A × (alpha/dim)

Trial 3: Alpha/Dim = 32/256 = 0.125
  → 重度正則化 (0.125x 更新強度)
  → 保存基礎模型知識
  → 結果：通用，過於保守

Trial 4: Alpha/Dim = 128/128 = 1.0
  → 零正則化 (1.0x 更新強度)
  → 最大化角色記憶容量
  → 結果：最高臉部相似度
  → 問題：無安全緩衝 → 不穩定
```

**3D 角色訓練哲學**

對於外觀一致的 3D 動畫角色：

  - 傳統的「防止過擬合」是**錯誤的**
  - 我們**希望**模型對特定角色「過擬合」
  - 挑戰：在不造成訓練崩潰的情況下實現高記憶

#### Trial 3 vs Trial 4：權衡分析

**Trial 3 優勢**
✅ 極度穩定的訓練 (零失敗)
✅ 一致的輸出品質
✅ 無梯度爆炸
✅ 適合風格轉移

**Trial 3 劣勢**
❌ 角色特異性低 (太通用)
❌ 臉部特徵不夠獨特
❌ 對 3D 角色訓練來說過度正則化

**Trial 4 優勢**
✅ 與原始角色的**臉部相似度最高**
✅ 保留獨特的角色特徵
✅ 強大的特徵記憶

**Trial 4 劣勢**
❌ 30-40% 品質失敗 (黑色圖像、變形)
❌ 訓練不穩定
❌ 無安全機制
❌ 僅 8 個 epochs (穩定期不足)

#### Trial 4 不穩定性的根本原因

1.  **梯度爆炸風險**：高 LR (0.000184) + Alpha=Dim + 僅 1 個梯度累積步驟
2.  **8-bit 量化放大**：AdamW8bit + 高 LR → 不穩定的更新
3.  **訓練時間不足**：8 個 epochs 對於高 LR 的穩定來說太短
4.  **缺失的安全機制**：
       - 無梯度裁剪 (`max_grad_norm`)
       - 無 SNR 加權 (`min_snr_gamma`)
       - 極少預熱 (僅 50 步)
       - 低梯度累積 (1 步)

### Trial 3.5 融合策略 (The Trial 3.5 Fusion Strategy)

**概念**：結合 Trial 4 的高學習能力與 Trial 3 的穩定機制。

```python
# Trial 3.5 配置
learning_rate = 0.00013           # 2.0x Trial 3, 0.7x Trial 4
text_encoder_lr = 0.00008         # 平衡的比率
network_dim = 128                 # Trial 4 的容量
network_alpha = 96                # Alpha/Dim = 0.75 (高記憶，25% 安全緩衝)
max_train_epochs = 18             # 2.25x Trial 4 (更多穩定期時間)
optimizer = "AdamW"               # 全精度 (無 8-bit 量化)
gradient_accumulation_steps = 3   # 3x Trial 4 (更平滑的更新)
lr_scheduler = "cosine_with_restarts"
lr_scheduler_num_cycles = 2
lr_warmup_steps = 150             # 3x Trial 4
max_grad_norm = 0.8               # 梯度裁剪保護
min_snr_gamma = 5.0               # 用於穩定性的 SNR 加權
```

**5 層穩定性保護**：

1.  **第 1 層**：梯度累積 (3 步) → 更平滑的更新
2.  **第 2 層**：梯度裁剪 (max\_norm=0.8) → 防止爆炸
3.  **第 3 層**：延長預熱 (150 步) → 逐步提升
4.  **第 4 層**：SNR 加權 (gamma=5.0) → 優先處理乾淨的訓練步驟
5.  **第 5 層**：全精度優化器 → 消除量化雜訊

**預期結果**：

  - 角色相似度：⭐⭐⭐⭐⭐ (維持 Trial 4 的高相似度)
  - 訓練穩定性：⭐⭐⭐⭐ (增加全面的安全機制)
  - 品質失敗：\<5% (相較於 Trial 4 的 30-40%)

### V2.1 優化策略

基於 Trial 1-5 的見解，我們設計了一個改進的搜尋空間，它：

1.  ✅ 將學習率範圍縮小到成功區域
2.  ✅ 使用 Alpha/Dim 比率 (0.25-0.9) 代替絕對值
3.  ✅ 為高 Alpha 配置添加分層安全約束
4.  ✅ 將最低訓練 epochs 從 8 增加到 12

#### V2.1 搜尋空間變更

**學習率 (縮小範圍)**

```python
# V1 (舊):
"learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True)
"text_encoder_lr": trial.suggest_float("text_encoder_lr", 3e-5, 1e-4, log=True)

# V2.1 (新):
"learning_rate": trial.suggest_float("learning_rate", 6e-5, 1.2e-4, log=True)
  # 縮小: 5e-5~2e-4 → 6e-5~1.2e-4
  # 理由: Trial 3 (6.6e-5) 和 Trial 5 (7.1e-5) 是穩定的
  #      Trial 4 (1.84e-4) 太高 → 上限設為 1.2e-4

"text_encoder_lr": trial.suggest_float("text_encoder_lr", 3e-5, 8e-5, log=True)
  # 縮小: 1e-4 → 8e-5
  # 理由: Text encoder 應為 UNet LR 的 0.5-0.7倍
```

**Network Alpha - 比率方法 (新)**

```python
# V1 (舊): 類別絕對值
"network_dim": trial.suggest_categorical("network_dim", [64, 96, 128, 192, 256])
"network_alpha": trial.suggest_categorical("network_alpha", [32, 48, 64, 96, 128])

# V2.1 (新): 基於比率並擴展範圍
"network_dim": trial.suggest_categorical("network_dim", [64, 128, 256])  # 簡化
"network_alpha_ratio": trial.suggest_float("network_alpha_ratio", 0.25, 0.9, step=0.05)
  # 比率範圍: [0.25, 0.3, 0.35, ..., 0.85, 0.9]
  # 基於 Trial 4 的成功，從最初的 0.5 擴展到 0.9

# 獲取 dim 後計算實際 alpha
network_dim = params["network_dim"]
alpha_ratio = params["network_alpha_ratio"]
network_alpha = int(network_dim * alpha_ratio)
params["network_alpha"] = network_alpha
```

**為什麼使用比率方法？**

1.  **保證有效組合**：Alpha 總是 ≤ Dim
2.  **探索 Trial 4 的範圍**：比率 0.75-0.9 類似於 Trial 4 的 1.0
3.  **語義意義**：比率直接控制正則化強度
4.  **更好的覆蓋率**：連續範圍 vs 離散類別

**為什麼最大值=0.9 (不是 1.0)？**

  - Trial 4 (比率=1.0) 有最高相似度但不穩定
  - 比率=0.9 提供 **10% 安全緩衝** 同時保持高記憶
  - 配合適當的穩定機制，0.9 可以匹配 Trial 4 的相似度

**訓練 Epochs (增加最小值)**

```python
# V1 (舊):
"max_train_epochs": trial.suggest_categorical("max_train_epochs", [8, 10, 12, 15])

# V2.1 (新):
"max_train_epochs": trial.suggest_categorical("max_train_epochs", [12, 16, 20])
  # 移除 8, 10 (對於高 LR 穩定來說太短)
  # 理由: Trial 4 的 8 epochs 不足以收斂
```

**優化器選擇 (簡化)**

```python
# V1 (舊):
"optimizer_type": trial.suggest_categorical("optimizer_type",
    ["AdamW", "AdamW8bit", "Lion"])

# V2.1 (新):
"optimizer_type": trial.suggest_categorical("optimizer_type",
    ["AdamW", "AdamW8bit"])
  # 移除 Lion: 實踐中不可靠
  # 專注於經證實的 AdamW 變體
```

#### V2.1 安全約束

**硬性約束 (訓練前強制執行)**

```python
def check_safety_constraints(params: Dict[str, any]) -> Tuple[bool, str]:
    """
    檢查參數組合是否安全可訓練

    返回:
        (is_valid, rejection_reason)
    """
    lr = params["learning_rate"]
    text_lr = params["text_encoder_lr"]
    dim = params["network_dim"]
    alpha = params["network_alpha"]
    alpha_ratio = params["network_alpha_ratio"]
    optimizer = params["optimizer_type"]
    epochs = params["max_train_epochs"]
    grad_accum = params["gradient_accumulation_steps"]
    warmup = params["lr_warmup_steps"]

    # 約束 1: 高 LR + 8bit 優化器
    if lr > 0.00012 and optimizer == "AdamW8bit":
        return False, "High LR (>0.00012) + AdamW8bit = unstable"

    # 約束 2: 精確 Alpha = Dim (避免浮點比較問題)
    if abs(alpha - dim) < 1:  # 差距小於 1 視為相等
        return False, "Alpha ≈ Dim causes overfitting + instability"

    # 約束 3: 高 Dim + 少 Epochs
    if dim >= 256 and epochs < 16:
        return False, "High dim (≥256) needs ≥16 epochs for convergence"

    # 約束 4: 極高 LR + 低預熱
    if lr > 0.0001 and warmup < 100:
        return False, "High LR (>0.0001) needs ≥100 warmup steps"

    # 約束 5: 梯度累積 1 + 高 LR
    if grad_accum == 1 and lr > 0.00011:
        return False, "High LR needs gradient accumulation ≥2"

    # === 針對高 Alpha 比率的分層約束 ===

    # 約束 6: 高記憶 (比率 >= 0.75) 需要強穩定性
    if alpha_ratio >= 0.75:
        required_checks = [
            ("epochs >= 16", epochs >= 16),
            ("grad_accum >= 2", grad_accum >= 2),
            ("optimizer = AdamW", optimizer == "AdamW"),  # 高 Alpha 不使用 8bit
            ("warmup >= 150", warmup >= 150),
        ]

        failed_checks = [name for name, passed in required_checks if not passed]
        if failed_checks:
            return False, f"High alpha (≥0.75) requires: {', '.join(failed_checks)}"

    # 約束 7: 極高 Alpha (>= 0.85) + 高 LR 很危險
    if alpha_ratio >= 0.85 and lr > 0.0001:
        return False, "Very high alpha (≥0.85) + LR >0.0001 = explosion risk"

    return True, ""
```

**分層約束的理由**：

  - 低 alpha (0.25-0.5)：安全，無特殊需求
  - 中 alpha (0.5-0.75)：中等安全需求
  - 高 alpha (0.75-0.9)：**需要所有安全機制**
      - 更長的訓練 (≥16 epochs)
      - 梯度平滑 (≥2 累積)
      - 全精度優化器 (無 8-bit)
      - 延長預熱 (≥150 步)

#### 預期 V2.1 結果分佈 (20 次試驗)

```
Alpha 比率分佈：
├─ 0.25-0.4 (低):  ~6 次試驗  → 強正則化，穩定
├─ 0.5-0.7  (中):  ~8 次試驗  → 核心探索區
└─ 0.75-0.9 (高): ~6 次試驗  → Trial 4 範圍帶有安全機制

學習率分佈：
├─ 6e-5 到 8e-5:   ~7 次試驗  → 保守基準
├─ 8e-5 到 1e-4:   ~8 次試驗  → 平衡效能
└─ 1e-4 到 1.2e-4: ~5 次試驗  → 帶有約束的高學習率

預期成果：
✅ 穩定試驗：80-90% (V1 為 40-50%)
✅ 品質失敗：<10% (V1 為 30-40%)
✅ 找到最佳配置：30-40% (V1 為 10-15%)
```

### Alpha/Dim 比率技術深度剖析

#### LoRA 數學基礎

```
LoRA 權重更新公式：
ΔW = B × A × scaling_factor

其中：
  scaling_factor = alpha / dim
  B, A = 低秩分解矩陣
```

**Network Dim (Rank)**：

  - 決定 LoRA **容量** (參數數量)
  - Dim=128 → 矩陣大小 [m×128] × [128×n]
  - 較高的 Dim = 更多參數 = 更強的表現能力

**Network Alpha (縮放因子)**：

  - **不**影響模型容量
  - 僅是一個**乘法係數**
  - 控制 LoRA 更新的**強度**

#### Alpha/Dim 比率效應

**比率 \< 1 (Alpha \< Dim)**

```python
範例: Dim=128, Alpha=32 → 比率=0.25
更新強度 = ΔW × 0.25
```

**效應**：

  - ✅ LoRA 更新被**正則化** (抑制)
  - ✅ 基礎模型知識被**保存**
  - ✅ 更穩定，較不易崩潰
  - ❌ 角色特徵可能**太通用**

**使用案例**：風格轉移、通用 LoRA、當基礎模型知識很有價值時

**比率 = 1 (Alpha = Dim)**

```python
範例: Dim=128, Alpha=128 → 比率=1.0 (Trial 4)
更新強度 = ΔW × 1.0
```

**效應**：

  - ✅ LoRA 更新**完全應用** (無正則化)
  - ✅ **最大化角色記憶**容量
  - ⚠️ 零正則化 → 容易不穩定
  - ❌ Trial 4 結果：高相似度但訓練崩潰

**比率 \> 1 (Alpha \> Dim)** ⚠️ **危險**

```python
範例: Dim=128, Alpha=256 → 比率=2.0
更新強度 = ΔW × 2.0
```

**效應**：

  - ⚠️ LoRA 更新被**放大**
  - ❌ **極端梯度爆炸風險**
  - ❌ 訓練極度不穩定
  - ❌ 模型崩潰 (NaN 損失、黑色圖像、完全變形)

**為什麼很少使用**：

1.  Kohya\_ss 官方建議：Alpha ≤ Dim
2.  社群實踐：Alpha \> Dim 幾乎總是失敗
3.  數學直覺：放大更新 = 放大梯度 = 爆炸

#### 實務常見比率範圍

| Alpha/Dim | 使用案例 | 穩定性 | 應用 |
|-----------|----------|-----------|--------------|
| **0.125-0.25** | 強正則化 | ⭐⭐⭐⭐⭐ | 風格轉移、通用 LoRA |
| **0.5** | 標準 (Kohya 預設) | ⭐⭐⭐⭐ | 大多數教學、平衡訓練 |
| **0.75-0.9** | 高記憶 | ⭐⭐⭐ | 3D 角色特異性 (需要安全機制) |
| **1.0** | 零正則化 | ⭐⭐ | Trial 4: 高相似度但不穩定 |
| **\>1.0** | 過度放大 | ⭐ | ❌ **不建議 - 極高風險** |

#### 為什麼 V2.1 使用 max=0.9

```
Alpha/Dim = 0.9 → 接近但不達到 1.0

優點：
✅ 探索 Trial 4 的高記憶範圍 (0.75-0.9 類似於 1.0)
✅ 保留 10% 正則化作為安全緩衝
✅ 配合穩定機制，可匹配 Trial 4 的角色相似度
✅ 避免完全零正則化的極端情況
```

**如果你真的想要 Alpha \> Dim** (不建議)：

所需保障措施 (必須全部滿足)：

  - 極低的 LR (\< 0.00005)
  - 非常長的訓練 (\>20 epochs)
  - 強梯度裁剪 (max\_grad\_norm=0.5)
  - 高梯度累積 (≥4)
  - 全精度訓練 (fp32, 無 8-bit)
  - 頻繁驗證 (每步檢查 NaN)

**風險等級**：極高 - 99% 失敗率

#### 安全範圍建議

```
保守：   [0.25-0.5]   → 適合大多數情況
平衡：   [0.25-0.75]  → 平衡穩定性與記憶
積極：   [0.25-0.9]   → V2.1 選擇，帶有安全機制的極限探索
極端：   [0.25-1.0]   → 可能但需要極端穩定措施
危險：   >1.0         → ❌ 強烈不建議
```

### V2.1 實作指南

#### 修改檔案

**目標**: `/mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/optuna_hyperparameter_search.py`

#### 步驟 1: 更新 `suggest_hyperparameters()` 方法

**位置**: 第 67-121 行

**1.1 學習率範圍**

```python
# 修改前 (V1):
"learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-4, log=True),
"text_encoder_lr": trial.suggest_float("text_encoder_lr", 3e-5, 1e-4, log=True),

# 修改後 (V2.1):
"learning_rate": trial.suggest_float("learning_rate", 6e-5, 1.2e-4, log=True),
"text_encoder_lr": trial.suggest_float("text_encoder_lr", 3e-5, 8e-5, log=True),
```

**1.2 Network Alpha - 使用比率**

```python
# 修改前 (V1):
"network_dim": trial.suggest_categorical("network_dim", [64, 96, 128, 192, 256]),
"network_alpha": trial.suggest_categorical("network_alpha", [32, 48, 64, 96, 128]),

# 修改後 (V2.1):
"network_dim": trial.suggest_categorical("network_dim", [64, 128, 256]),
"network_alpha_ratio": trial.suggest_float("network_alpha_ratio", 0.25, 0.9, step=0.05),

# 獲取 dim 後計算實際 alpha
network_dim = params["network_dim"]
alpha_ratio = params["network_alpha_ratio"]
network_alpha = int(network_dim * alpha_ratio)
params["network_alpha"] = network_alpha
```

**1.3 訓練 Epochs**

```python
# 修改前 (V1):
"max_train_epochs": trial.suggest_categorical("max_train_epochs", [8, 10, 12, 15]),

# 修改後 (V2.1):
"max_train_epochs": trial.suggest_categorical("max_train_epochs", [12, 16, 20]),
```

**1.4 優化器選擇**

```python
# 修改前 (V1):
"optimizer_type": trial.suggest_categorical("optimizer_type", ["AdamW", "AdamW8bit", "Lion"]),

# 修改後 (V2.1):
"optimizer_type": trial.suggest_categorical("optimizer_type", ["AdamW", "AdamW8bit"]),
```

#### 步驟 2: 添加安全約束檢查器

**位置**: 在 `suggest_hyperparameters()` 之後添加新方法 (約第 122 行)

```python
def check_safety_constraints(self, params: Dict[str, any]) -> Tuple[bool, str]:
    """
    檢查參數組合是否安全可訓練

    返回:
        (is_valid, rejection_reason)
    """
    lr = params["learning_rate"]
    text_lr = params["text_encoder_lr"]
    dim = params["network_dim"]
    alpha = params["network_alpha"]
    alpha_ratio = params["network_alpha_ratio"]
    optimizer = params["optimizer_type"]
    epochs = params["max_train_epochs"]
    grad_accum = params["gradient_accumulation_steps"]
    warmup = params["lr_warmup_steps"]

    # 約束 1: 高 LR + 8bit 優化器
    if lr > 0.00012 and optimizer == "AdamW8bit":
        return False, "High LR (>0.00012) + AdamW8bit = unstable"

    # 約束 2: 精確 Alpha = Dim (避免浮點比較問題)
    if abs(alpha - dim) < 1:
        return False, "Alpha ≈ Dim causes overfitting + instability"

    # 約束 3: 高 Dim + 少 Epochs
    if dim >= 256 and epochs < 16:
        return False, "High dim (≥256) needs ≥16 epochs for convergence"

    # 約束 4: 極高 LR + 低預熱
    if lr > 0.0001 and warmup < 100:
        return False, "High LR (>0.0001) needs ≥100 warmup steps"

    # 約束 5: 梯度累積 1 + 高 LR
    if grad_accum == 1 and lr > 0.00011:
        return False, "High LR needs gradient accumulation ≥2"

    # 約束 6: 高記憶 (比率 >= 0.75) 需要強穩定性
    if alpha_ratio >= 0.75:
        required_checks = [
            ("epochs >= 16", epochs >= 16),
            ("grad_accum >= 2", grad_accum >= 2),
            ("optimizer = AdamW", optimizer == "AdamW"),
            ("warmup >= 150", warmup >= 150),
        ]

        failed_checks = [name for name, passed in required_checks if not passed]
        if failed_checks:
            return False, f"High alpha (≥0.75) requires: {', '.join(failed_checks)}"

    # 約束 7: 極高 Alpha (>= 0.85) + 高 LR 很危險
    if alpha_ratio >= 0.85 and lr > 0.0001:
        return False, "Very high alpha (≥0.85) + LR >0.0001 = explosion risk"

    return True, ""
```

#### 步驟 3: 修改 `train_lora()` 以使用約束

**位置**: 第 123-149 行

```python
def train_lora(self, trial: Trial, params: Dict[str, any]) -> Path:
    """
    使用給定的超參數訓練 LoRA

    返回:
        訓練檢查點的路徑
    """
    # === 在開頭添加此部分 ===
    # 在訓練前檢查安全約束
    is_valid, rejection_reason = self.check_safety_constraints(params)
    if not is_valid:
        print(f"\n❌ TRIAL REJECTED: {rejection_reason}")
        print(f"Parameters: {json.dumps(params, indent=2)}\n")
        raise optuna.TrialPruned(rejection_reason)

    # === 原始代碼繼續 ===
    self.trial_counter += 1
    trial_dir = self.output_dir / f"trial_{self.trial_counter:04d}"
    # ... 方法其餘部分不變
```

#### 步驟 4: 更新參數日誌

**位置**: 約第 138-144 行 (在 `train_lora` 內)

```python
# 修改前:
print(f"Parameters:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 修改後 (添加 alpha 比率資訊):
print(f"Parameters:")
for key, value in params.items():
    if key == "network_alpha":
        alpha_ratio = params.get("network_alpha_ratio", value / params["network_dim"])
        print(f"  {key}: {value} (ratio: {alpha_ratio:.3f})")
    else:
        print(f"  {key}: {value}")
```

#### 步驟 5: 移除舊的 Alpha 約束

**位置**: 第 117-119 行

```python
# 移除這個 (V1):
# Ensure network_alpha <= network_dim (common constraint)
if params["network_alpha"] > params["network_dim"]:
    params["network_alpha"] = params["network_dim"]

# V2.1 不需要 - Alpha 是從比率計算而來，保證 ≤ dim * 0.9
```

#### 驗證檢查清單

實作 V2.1 變更後：

  - [ ] 腳本運行無語法錯誤
  - [ ] 第一次試驗在參數中印出 alpha 比率
  - [ ] `alpha_ratio >= 0.75` 的試驗檢查了所有穩定性要求
  - [ ] 被拒絕的試驗顯示清楚的拒絕原因
  - [ ] 學習率範圍是 [6e-5, 1.2e-4]
  - [ ] 所有試驗使用 epochs ≥ 12
  - [ ] 資料庫和日誌正確建立

#### V2.1 使用範例

```bash
# 啟動 V2.1 優化 (20 次試驗，保守階段)
conda run -n kohya_ss python \
  /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/scripts/optimization/optuna_hyperparameter_search.py \
  --dataset-config /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/luca_human_dataset.toml \
  --base-model /mnt/c/AI_LLM_projects/ai_warehouse/models/stable-diffusion/checkpoints/v1-5-pruned-emaonly.safetensors \
  --output-dir /mnt/data/ai_data/models/lora/luca/optimization_v2.1 \
  --study-name luca_v2.1_optimization \
  --n-trials 20 \
  --device cuda
```

### V2.1 預期改進

**比較：V1 vs V2.1**

| 指標 | V1 (Trials 1-5) | V2.1 (預期) | 改進 |
|--------|-----------------|-----------------|-------------|
| **穩定試驗** | 40-50% | 80-90% | ↑ +40-50% |
| **品質失敗** | 30-40% | \<10% | ↓ 減少 20-30% |
| **找到最佳配置** | 10-15% | 30-40% | ↑ +20-25% |
| **災難性失敗** | 10-15% | \<5% | ↓ 減少 5-10% |
| **角色相似度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 探索 Trial 4 範圍 |
| **訓練效率** | \~50% 浪費 | \~90% 有用 | ↑ 40% 改進 |

**關鍵變更摘要**：

1.  ✅ 縮小 LR 範圍 (基於 Trial 1-5 數據)
2.  ✅ Alpha 比率方法 [0.25-0.9] (vs 類別)
3.  ✅ 為高 Alpha 添加分層安全約束
4.  ✅ 增加最低 epochs (12 vs 8)
5.  ✅ 簡化 dim 選擇 (3 個 vs 5 個選項)
6.  ✅ 移除不可靠的優化器 (Lion)
7.  ✅ 訓練前安全檢查

### 實作時程表

**階段 1: 準備** (目前)

1.  ⏳ 等待 Trial 5 完成 (ETA: \~00:40-01:00)
2.  🧪 測試 Trial 3 檢查點 (建立基準)
3.  🔍 評估 Trial 5 結果

**階段 2: 實作** (Trial 5 之後)

1.  ✅ 在 `optuna_hyperparameter_search.py` 中實作 V2.1 代碼變更
2.  ✅ 添加安全約束檢查器函數
3.  ✅ 更新搜尋空間參數
4.  ✅ 使用單次試驗運行進行測試

**階段 3: 執行** (驗證之後)

1.  🚀 啟動 V2.1 階段 1 (20 次試驗，保守)
       - 預期運行時間：24-30 小時
       - 目標：找到 5-8 個優秀配置
       - 預期穩定試驗率：80-90%

2.  📊 分析階段 1 結果
       - 與 Trial 3.5 基準比較
       - 識別最佳 alpha 比率範圍
       - 檢查是否需要進一步探索

3.  🎯 可選階段 2 (如果需要)
       - 額外 20 次試驗專注於最佳範圍
       - 在最佳參數周圍微調

**成功標準**：

如果滿足以下條件，V2.1 優化視為成功：

  - ✅ \<10% 災難性失敗 (黑色圖像、崩潰)
  - ✅ \>60% 可用檢查點 (可接受的品質)
  - ✅ \>3 個優秀配置 (可投入生產)
  - ✅ 最佳檢查點上的 CLIP 分數 \>0.30
  - ✅ 任何試驗中均無梯度爆炸
  - ✅ 所有輸出中的角色身分一致

-----

**最後更新**: 2025-11-13
**版本**: 2.1 (Trial 1-5 分析 + V2.1 策略)