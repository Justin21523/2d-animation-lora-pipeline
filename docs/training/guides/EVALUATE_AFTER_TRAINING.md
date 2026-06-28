# 📊 訓練完成後評估指南

## 🎯 訓練完成後執行

當 Miguel 的 SDXL LoRA 訓練完成（預計凌晨 01:50）後，執行以下命令評估所有 checkpoints：

```bash
bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel
```

---

## 📦 預期輸出

訓練完成後會有 **5 個 checkpoints**：
- `miguel_identity_lora_sdxl-000002.safetensors` (Epoch 2)
- `miguel_identity_lora_sdxl-000004.safetensors` (Epoch 4)
- `miguel_identity_lora_sdxl-000006.safetensors` (Epoch 6)
- `miguel_identity_lora_sdxl-000008.safetensors` (Epoch 8)
- `miguel_identity_lora_sdxl-000010.safetensors` (Epoch 10)

---

## ⏱️ 評估時間

每個 checkpoint 評估時間：**約 5-10 分鐘**

總評估時間：**25-50 分鐘**（5個checkpoints）

評估過程：
1. 載入 SDXL 基礎模型
2. 載入 LoRA checkpoint
3. 生成 24 張測試圖（6 prompts × 4 variations）
4. 計算評估指標：
   - CLIP Score (提示對齊)
   - InsightFace Consistency (臉部一致性)
   - LPIPS Diversity (圖像多樣性)
   - MUSIQ Quality (技術質量)
5. 生成綜合評分

---

## 📊 評估指標說明

### 1. CLIP Score (0-1，越高越好)
- 衡量生成圖像與提示詞的對齊程度
- **目標：> 0.30** 表示良好對齊
- **> 0.35** 表示優秀

### 2. Face Consistency (0-1，越高越好)
- 使用 InsightFace ArcFace 衡量角色臉部一致性
- **目標：> 0.70** 表示同一角色
- **> 0.80** 表示高度一致

### 3. LPIPS Diversity (0-1，越高越好)
- 衡量生成圖像的感知多樣性
- **目標：0.3-0.6** 表示適當多樣性
- 太低 (<0.2) = mode collapse
- 太高 (>0.7) = 不一致

### 4. MUSIQ Quality (0-100，越高越好)
- 技術圖像質量評分
- **目標：> 60** 表示可接受質量
- **> 70** 表示良好質量

### 5. Aggregate Score (0-1，越高越好)
- 加權綜合評分
- 權重：CLIP 35% + Face 35% + Diversity 15% + Quality 15%
- **目標：> 0.50** 表示良好 LoRA
- **> 0.60** 表示優秀 LoRA

---

## 📁 輸出結構

```
/mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/
├── miguel_identity_lora_sdxl-000002.safetensors
├── miguel_identity_lora_sdxl-000004.safetensors
├── ...
└── evaluation_results_YYYYMMDD_HHMMSS/
    ├── miguel_identity_lora_sdxl-000002/
    │   ├── image_0000.png ... image_0023.png  (24張測試圖)
    │   └── evaluation.json                     (詳細指標)
    ├── miguel_identity_lora_sdxl-000004/
    │   └── ...
    ├── ...
    ├── checkpoint_comparison.json              (排名和最佳checkpoint)
    └── checkpoint_comparison.png               (視覺化比較圖)
```

---

## 🏆 查看結果

### 查看最佳 checkpoint
```bash
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.json | python -m json.tool | grep -A 2 "best_checkpoint"
```

### 查看排名
```bash
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.json | python -m json.tool | grep -A 10 "ranking"
```

### 查看單一 checkpoint 詳細結果
```bash
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/miguel_identity_lora_sdxl-000010/evaluation.json | python -m json.tool
```

### 查看測試圖
```bash
eog /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/miguel_identity_lora_sdxl-000010/image_*.png
```

### 查看比較圖
```bash
eog /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.png
```

---

## 🎯 選擇最佳 Checkpoint 的建議

通常最佳 checkpoint **不是** Epoch 10！根據經驗：

- **Epoch 4-6** 通常是最佳平衡點
- Epoch 2 可能欠擬合（CLIP score 低）
- Epoch 8-10 可能過擬合（diversity 低，mode collapse）

**選擇標準（優先級）：**
1. ✅ Face Consistency > 0.75（最重要！）
2. ✅ CLIP Score > 0.30
3. ✅ Diversity 0.3-0.6（避免 mode collapse）
4. ✅ Aggregate Score 最高

---

## ⚡ 快速開始（訓練完成後）

```bash
# 1. 確認訓練完成
ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors

# 2. 運行批量評估
bash scripts/batch/evaluate_all_sdxl_checkpoints.sh miguel

# 3. 查看最佳結果
cat /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.json | python -m json.tool

# 4. 查看比較圖
eog /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/evaluation_results_*/checkpoint_comparison.png
```

---

## 🔍 監控訓練進度（現在）

當前訓練狀態：
```bash
# 查看即時訓練進度
tmux attach -t sdxl_training
# 離開: Ctrl+B 然後 D

# 查看 GPU 狀態
nvidia-smi

# 查看已保存的 checkpoints
ls -lht /mnt/c/ai_models/diffusion/lora/sdxl/coco/miguel_identity/*.safetensors
```

---

**💡 提示：** 評估腳本會自動使用所有可用的評估模型（CLIP, InsightFace, LPIPS, MUSIQ），並生成完整的比較報告！
