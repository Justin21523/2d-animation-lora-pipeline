# Kohya_ss GUI SDXL LoRA 訓練指南

## 快速開始

### 1. 啟動 Kohya_ss GUI

```bash
cd /mnt/c/ai_projects/kohya_ss
conda activate kohya_ss
python kohya_gui.py
```

GUI 會在瀏覽器中啟動，通常是 `http://localhost:7860`

---

## 稲妻11 角色 SDXL LoRA 訓練配置步驟

### 第一步：配置基本設置 (Settings)

1. **進入 Settings 標籤**
   - 點擊上方的 "Settings"

2. **設置基本路徑**
   - Models directory: `/mnt/c/ai_models/stable-diffusion/checkpoints`
   - Pretrained model name or path: `sd_xl_base_1.0.safetensors`
   - Output directory: `/mnt/data/training/lora/inazuma_eleven`
   - Logging directory: `/mnt/data/training/lora/inazuma_eleven/logs`

3. **保存設置**
   - 點擊 "Save Config" 保存

---

### 第二步：配置訓練基本參數 (Train LoRA)

1. **進入 Train LoRA 標籤**
   - 點擊上方的 "Train LoRA"

2. **Model 部分**
   ```
   Pretrained model name or path: /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors
   ```

3. **Dataset 部分 - 重要！**

   **第一個角色示例: Endou Mamoru**

   - Train data dir: `/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/Endou Mamoru`
   - Image extension: `.png` 或 `.jpg`
   - Caption extension: `.txt`
   - Enable Bucketing: ✅ (勾選)
   - Batch size: `2` (根據 GPU 記憶體調整，4GB 用 1，8GB+ 用 2-4)

4. **Output 部分**
   ```
   Output directory: /mnt/data/training/lora/inazuma_eleven/endou_mamoru
   Output name: endou_mamoru_sdxl_lora
   Save model as: safetensors
   Save precision: bf16
   ```

5. **Training 部分**
   ```
   Max train epochs: 10
   Max train steps: 1000  (或根據需要調整)
   Save every n epochs: 1
   Save every n steps: 200
   Learning rate: 0.0001 (1e-4)
   LR Scheduler: cosine
   Optimizer: AdamW8bit
   ```

6. **LoRA Network 部分**
   ```
   Network module: lycoris.kohya
   Network dim: 64
   Network alpha: 32
   Network dropout: 0.1
   ```

7. **Advanced 部分**
   ```
   Cache latents: ✅
   Full BF16: ✅
   Clip skip: 2
   ```

8. **SDXL 專用部分**
   ```
   SDXL Cache text encoder outputs: ✅
   No half VAE: ✅
   ```

9. **解析度設置**
   ```
   Resolution: 1024,1024 (或 768,1024)
   Min bucket reso: 256
   Max bucket reso: 2048
   ```

---

### 第三步：配置採樣和驗證 (Sampling)

1. **Sample prompts 部分**
   ```
   Sample every n epochs: 1
   Sample sampler: euler_a
   Sample prompts:
   - "a 3d anime character, inazuma_endou_mamoru, anime_style, looking_at_viewer"
   - "1boy, inazuma_endou_mamoru, soccer uniform, determined expression"

   Negative prompts:
   - "bad quality, low quality, blurry"
   ```

2. **Validation 部分** (可選)
   ```
   Validation every n epochs: 1
   ```

---

### 第四步：保存配置並開始訓練

1. **保存配置**
   - 在 Train LoRA 標籤最下方，點擊 "Save Config"
   - 選擇保存位置：
     ```
     /mnt/data/training/lora/inazuma_eleven/configs/endou_mamoru_sdxl_gui.json
     ```

2. **開始訓練**
   - 點擊 "Train" 按鈕
   - GUI 會顯示訓練進度
   - 訓練過程中可以在下方看到日誌輸出

3. **監控訓練**
   - 實時進度條顯示
   - 損失值變化圖表
   - Checkpoint 自動生成在輸出目錄

---

## 為所有 7 個角色重複配置

### 角色列表和配置概覽

| 角色 | 訓練資料夾 | 輸出資料夾 | 特殊設置 |
|-----|----------|---------|--------|
| Endou Mamoru | `Endou Mamoru` | `endou_mamoru` | 標準設置 |
| Fudou Akio | `Fudou Akio` | `fudou_akio` | 標準設置 |
| Gouenji Shuuya | `Gouenji Shuuya` | `gouenji_shuuya` | 標準設置 |
| Inamori Asuto | `Inamori Asuto` | `inamori_asuto` | 標準設置 |
| Matsukaze Tenma | `Matsukaze Tenma` | `matsukaze_tenma` | 標準設置 |
| Nosaka Yuuma | `Nosaka Yuuma` | `nosaka_yuuma` | 標準設置 |
| Utsunomiya Toramaru | `Utsunomiya Toramaru` | `utsunomiya_toramaru` | 標準設置 |

### 快速複製配置方法

1. 配置完第一個角色 (Endou Mamoru) 後
2. 點擊 "Load Config" 重新加載該配置
3. **修改以下部分**：
   - Train data dir: 改為下一個角色
   - Output directory: 改為下一個角色
   - Output name: 改為下一個角色 ID
4. 保存為新配置文件
5. 點擊 "Train" 開始訓練

---

## 訓練期間會發生的事

### 1. 初始化階段 (2-5 分鐘)
- 加載基模型
- 初始化 LoRA 網絡
- 準備數據集

### 2. 訓練階段 (30分鐘 - 2小時，取決於設置)
- 進度條顯示當前 epoch 和步數
- 損失值逐漸下降（正常現象）
- 每 200 步自動保存 checkpoint

### 3. Checkpoint 生成
- 每個 epoch 結束時生成完整 checkpoint
- 保存位置：`/mnt/data/training/lora/inazuma_eleven/<character>/`
- 檔案格式：`.safetensors`

### 4. 採樣圖片生成
- 每個 epoch 使用示例提示詞生成預覽圖片
- 保存在 output 目錄的 `samples/` 文件夾

---

## Checkpoint 管理和驗證

### 訓練完成後

1. **檢查 Checkpoint**
   ```bash
   ls -lh /mnt/data/training/lora/inazuma_eleven/endou_mamoru/
   ```
   應該看到多個 `.safetensors` 檔案

2. **選擇最佳 Checkpoint**
   - 通常最後一個 epoch 的 checkpoint 最好
   - 或者查看 `samples/` 中的生成圖片，選擇視覺效果最好的

3. **重命名最佳 Checkpoint**（可選）
   ```bash
   cp /path/to/checkpoint_best.safetensors \
      /mnt/data/training/lora/inazuma_eleven/endou_mamoru/endou_mamoru_final.safetensors
   ```

---

## 故障排除

### 問題 1: 訓練開始時 GPU 記憶體不足

**解決方案**：
- 減小 `batch_size` (改為 1)
- 啟用 `Enable gradient checkpointing`
- 減小 `network_dim` (改為 32)

### 問題 2: 訓練很慢

**解決方案**：
- 啟用 `Full BF16` (已啟用)
- 檢查 GPU 使用率（應該 > 80%）
- 增加 `batch_size`（如果有記憶體）

### 問題 3: 生成的圖片品質差

**解決方案**：
- 增加訓練 epoch (改為 20-30)
- 調整 learning rate (試試 1e-5 或 5e-5)
- 檢查訓練資料質量

### 問題 4: GUI 連不上

**解決方案**：
```bash
# 確認 kohya_ss 進程在運行
ps aux | grep kohya_gui

# 檢查端口
netstat -tuln | grep 7860

# 如果卡住，重新啟動
pkill -f kohya_gui
# 等待 2 秒後重新啟動
cd /mnt/c/ai_projects/kohya_ss && conda run -n kohya_ss python kohya_gui.py
```

---

## 訓練完成後的下一步

### 1. 測試 LoRA

```bash
python scripts/evaluation/test_lora_checkpoints.py \
  /mnt/data/training/lora/inazuma_eleven/endou_mamoru/ \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir /mnt/data/training/lora/inazuma_eleven/evaluation/endou_mamoru/
```

### 2. 生成測試圖片

使用 ComfyUI 或其他 Stable Diffusion UI 加載 LoRA 並生成圖片

### 3. 批量訓練所有角色

訓練完一個角色後，重複步驟 2-4 訓練其他角色

---

## 訓練時間預估

基於 GPU (推測 NVIDIA A100 或 RTX 4090):

| 設置 | 時間 |
|------|------|
| 10 epochs, 1000 steps | 30-60 分鐘 |
| 20 epochs, 2000 steps | 60-120 分鐘 |
| 完整 7 個角色 | 7-14 小時 (順序執行) |

---

## 重要提示

### 📌 Captions 檔案必須正確放置

```
/mnt/data/datasets/general/inazuma-eleven/lora_data/characters_augmented/
├── Endou Mamoru/
│   ├── image1.png
│   ├── image1.txt ✅ 必須有
│   ├── image2.png
│   ├── image2.txt ✅ 必須有
│   └── ...
```

### 📌 每個 Caption 應包含唯一的角色 Token

```
inazuma_eleven, inazuma_endou_mamoru, anime_style, teenage_boy, ...
                            ↑ 唯一識別符，必須包含
```

### 📌 訓練中不要關閉 GUI

如果需要停止訓練：
1. 點擊 GUI 中的 "Stop training" 按鈕（如果有）
2. 或使用 Ctrl+C 優雅地停止
3. 已保存的 checkpoint 不會丟失

---

## 完整操作流程總結

```
1. 啟動 GUI → 2. 配置第一個角色 → 3. 開始訓練
   ↓
4. 訓練進行中 → 5. 監控進度 → 6. 訓練完成
   ↓
7. 驗證 Checkpoint → 8. 選擇最佳版本 → 9. 複製配置
   ↓
10. 配置下一個角色 → 11. 重複步驟 2-8
   ↓
12. 所有角色訓練完成 → 13. 測試所有 LoRA → 14. 完成！
```

---

## 聯繫和問題

如有問題或 GUI 遇到困難，請檢查：
- 數據夾路徑是否正確
- Caption 檔案是否存在
- GPU 驅動是否最新
- Conda 環境是否正確激活

祝你訓練順利！🚀
