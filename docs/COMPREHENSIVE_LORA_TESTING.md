# Super Wings 綜合 LoRA 測試系統

## 概述

這是一個全面的 LoRA 評估系統，為每個 Super Wings 角色生成 **180 張多樣化測試圖片**，涵蓋：
- **10 種動作** (Actions)
- **10 種表情** (Expressions)
- **10 種姿勢** (Poses)

每個場景使用：
- **2 種 LoRA 強度**: 0.8, 1.0
- **3 個不同 seeds**: 42, 123, 456

**總計**: 30 prompts × 2 strengths × 3 seeds = **180 張圖片/角色**

---

## 文件結構

```
configs/evaluation/super_wings_comprehensive_test.yaml  # 測試配置
scripts/evaluation/comprehensive_lora_test.py           # 主測試腳本
/tmp/run_comprehensive_test_3chars.sh                   # 批次執行腳本
```

---

## 10 種動作 (Actions)

每個動作都經過精心設計，展現角色的動態能力：

1. **taking_off** - 起飛姿勢，輪子離地，引擎發光
2. **mid_flight** - 飛行中，穿越雲層，平穩飛行
3. **landing_preparation** - 降落準備，輪子放下，專注表情
4. **turning_spinning** - 轉身/旋轉，桶滾動作，動態模糊
5. **diving** - 俯衝，高速下降，風阻效果
6. **climbing** - 爬升，引擎全功率，尾跡雲
7. **victory_spread_pose** ⭐ **必須場景** - 張開雙臂雙腿，嘴巴張大自信微笑
8. **waving_hello** - 揮手打招呼，友善手勢
9. **speed_dash** - 快速衝刺，音爆效果
10. **victory_pose_standing** - 勝利姿勢，單臂高舉

---

## 10 種表情 (Expressions)

展現角色豐富的情感表達：

1. **happy_smile** - 開心微笑，溫暖真誠
2. **excited** - 興奮，大笑，充滿活力
3. **surprised** - 驚訝，眼睛睜大，嘴巴O型
4. **confident** - 自信，輕微微笑，堅定眼神
5. **focused_determined** - 專注，嚴肅，任務準備
6. **friendly** - 友善，溫柔微笑，親切
7. **curious** - 好奇，歪頭，探索眼神
8. **proud** - 驕傲，挺胸，滿足微笑
9. **encouraging** - 鼓勵，豎起大拇指，正面能量
10. **joyful_laughing** - 歡樂大笑，眼睛瞇起

---

## 10 種姿勢 (Poses)

全方位角度展示：

1. **front_standing** - 正面站立，對稱構圖
2. **side_profile** - 側面輪廓，90度角
3. **three_quarter_view** - 三分之三視角，45度
4. **back_view** - 背面視角，回頭看
5. **top_down_angle** - 俯視角度，鳥瞰
6. **low_angle_heroic** - 仰視角度，英雄視角
7. **dynamic_tilted** - 動態傾斜，荷蘭角
8. **hovering_stationary** - 靜止懸停，平靜
9. **compact_tucked** - 緊湊姿勢，流線型
10. **extended_stretched** - 伸展姿勢，翅膀展開

---

## 角色特色

### Jett (紅白色)
- **性格**: 樂觀、熱心助人、充滿活力
- **特徵**: 紅色機身配白色裝飾，小巧噴射機，兩隻藍眼睛
- **顏色**: red and white

### Jerome (藍色)
- **性格**: 經驗豐富、智慧、冷靜、導師風格
- **特徵**: 藍色機身，大型運輸機，兩隻藍眼睛
- **顏色**: blue

### Donnie (黃藍色)
- **性格**: 建築專家、機械能手、實用、技術精湛
- **特徵**: 黃色機身配藍色裝飾，建築設備，兩隻藍眼睛
- **顏色**: yellow and blue

---

## Negative Prompts (共用)

為了確保高品質輸出，所有測試使用統一的 negative prompts：

```
人類相關: human, person, people, humanoid, human face/body
多角色: multiple characters, group, crowd, duo
多眼睛: extra eyes, three eyes, four eyes, multiple eyes
顏色錯誤: wrong colors, incorrect colors, color swap
品質問題: blurry, low quality, distorted, deformed
2D風格: anime style, cartoon, illustration, painting
照片風格: photographic, photo, photograph
裁切問題: cropped, cut off, frame, border
其他: watermark, text, signature, noise
```

---

## 使用方法

### 方法 1: 批次執行所有 3 個角色

```bash
chmod +x /tmp/run_comprehensive_test_3chars.sh
bash /tmp/run_comprehensive_test_3chars.sh
```

這會自動依序測試 Jett → Jerome → Donnie，每個角色 180 張圖片。

**預計時間**: ~8-10 小時 (3 角色 × ~3 小時)

---

### 方法 2: 單獨測試一個角色

**測試 Jett:**
```bash
conda run -n kohya_ss python \
  /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/evaluation/comprehensive_lora_test.py \
  --character jett \
  --lora-path /mnt/data/training/lora/super-wings/jett_identity/super-wings-jett-identity-sdxl.safetensors \
  --output-dir /mnt/data/training/lora/super-wings/comprehensive_evaluation
```

**測試 Jerome:**
```bash
conda run -n kohya_ss python \
  /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/evaluation/comprehensive_lora_test.py \
  --character jerome \
  --lora-path /mnt/data/training/lora/super-wings/jerome_identity/super-wings-jerome-identity-sdxl.safetensors \
  --output-dir /mnt/data/training/lora/super-wings/comprehensive_evaluation
```

**測試 Donnie:**
```bash
conda run -n kohya_ss python \
  /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/evaluation/comprehensive_lora_test.py \
  --character donnie \
  --lora-path /mnt/data/training/lora/super-wings/donnie_identity/super-wings-donnie-identity-sdxl.safetensors \
  --output-dir /mnt/data/training/lora/super-wings/comprehensive_evaluation
```

---

## 輸出結構

```
/mnt/data/training/lora/super-wings/comprehensive_evaluation/
├── jett/
│   ├── action_taking_off_lora0.8_seed42.png
│   ├── action_taking_off_lora0.8_seed123.png
│   ├── action_taking_off_lora0.8_seed456.png
│   ├── action_taking_off_lora1.0_seed42.png
│   ├── action_taking_off_lora1.0_seed123.png
│   ├── action_taking_off_lora1.0_seed456.png
│   ├── action_mid_flight_lora0.8_seed42.png
│   ├── ... (180 張圖片)
│
├── jerome/
│   ├── action_taking_off_lora0.8_seed42.png
│   ├── ... (180 張圖片)
│
└── donnie/
    ├── action_taking_off_lora0.8_seed42.png
    ├── ... (180 張圖片)
```

**檔名格式**: `{category}_{name}_lora{strength}_seed{seed}.png`

---

## 測試參數

```yaml
LoRA strengths: [0.8, 1.0]
Seeds: [42, 123, 456]
Steps: 40
CFG scale: 8.0
Resolution: 1024 × 1024
```

---

## 關鍵特色場景

### ⭐ 必須場景: Victory Spread Pose

這是用戶特別要求的關鍵場景：

**Prompt 重點**:
- arms spread wide open (雙臂大幅張開)
- legs extended apart (雙腿伸展分開)
- mouth open with big confident smile (嘴巴張大自信微笑)
- extremely happy triumphant expression (極度開心勝利表情)
- eyes shining with joy (眼睛閃耀喜悅)
- starry celebration background (星空慶祝背景)
- spotlights, confetti (聚光燈、彩帶)

這個場景完美展現角色的自信、活力和勝利姿態！

---

## 品質保證

每個 prompt 都包含：

✅ **角色特定資訊**: 顏色、性格、特徵
✅ **嚴格品質控制**: 避免多角色、多眼睛、人類化
✅ **3D 動畫風格**: Pixar、DreamWorks 品質
✅ **詳細場景描述**: 背景、光線、構圖
✅ **表情與動作**: 明確的情感和動作指示

---

## 評估建議

完成測試後，建議從以下角度評估：

1. **角色一致性**: 顏色、造型是否正確
2. **表情豐富度**: 10 種表情是否有明顯區別
3. **動作準確性**: 動作是否符合描述
4. **LoRA 強度比較**: 0.8 vs 1.0 的差異
5. **Seed 穩定性**: 同一 prompt 不同 seeds 的一致性
6. **最佳場景**: 哪些場景最成功

---

## 故障排除

**GPU 記憶體不足:**
- 關閉其他訓練程序
- 降低 batch size (腳本已設為 1)
- 使用較小的 resolution (修改 YAML config)

**生成速度慢:**
- 每張圖約 8-10 秒
- 180 張 × 10 秒 ≈ 30 分鐘/角色
- 可調整 steps 降低至 30 加快速度

**顏色不正確:**
- 檢查 YAML 中角色 colors 欄位
- 確認 negative prompts 包含 "wrong colors"

---

## 後續擴展

系統設計支援輕鬆擴展：

1. **新增角色**: 在 YAML 的 `characters` 區塊新增
2. **新增場景**: 在 `actions`/`expressions`/`poses` 新增
3. **調整參數**: 修改 `test_config` 區塊
4. **自訂 LoRA 強度**: 調整 `lora_strengths` 列表

---

## 總結

這個綜合測試系統提供：

- ✅ **全面評估**: 30 種不同場景
- ✅ **參數測試**: 2 種 LoRA 強度 + 3 個 seeds
- ✅ **角色特化**: 每個角色的獨特 prompts
- ✅ **品質保證**: 嚴格的 negative prompts
- ✅ **自動化**: 一鍵批次執行
- ✅ **可擴展**: 易於新增角色和場景

開始測試，享受看到角色在各種場景下的精彩表現！🎬✨
