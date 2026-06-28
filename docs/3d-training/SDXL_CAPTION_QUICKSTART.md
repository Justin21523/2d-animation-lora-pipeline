# SDXL Caption Expansion - 5 分鐘快速開始

## 快速開始 3 步驟

### 1. 設定 API Key

```bash
export LLM_VENDOR_API_KEY="your-api-key-here"
```

### 2. 測試單一角色 (5 個 captions, ~$0.02)

```bash
# 測試 alberto
bash scripts/batch/quick_test_sdxl_expansion.sh alberto

# 或測試 elio
bash scripts/batch/quick_test_sdxl_expansion.sh elio
```

### 3. 批次處理所有角色

```bash
# 乾跑 - 查看將處理哪些角色
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py --dry-run

# 執行 - 擴展所有角色 (~$10-15, 30-60分鐘)
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py --execute
```

---

## 支援的角色

| Film | Characters |
|------|------------|
| **Luca** | alberto, giulia |
| **Onward** | ian, barley |
| **Turning Red** | tyler |
| **Up** | russell |
| **Orion** | orion |
| **Elio** | elio, bryce, caleb, glordon |
| **Coco** | miguel |

**總計**: 12 個角色

---

## 輸出範例

**原始 SD1.5 caption (45 tokens)**:
```
alberto, a 3d animated human character, pixar style, teenage boy with curly hair,
shirtless, standing on beach, natural lighting, medium shot
```

**擴展後 SDXL caption (142 tokens)**:
```
alberto scorfano, a 3d animated human character rendered in pixar animation style,
teenage boy with wild curly dark brown hair, warm tan italian complexion,
shirtless showing detailed skin shader with subsurface scattering, standing
confidently on sunny italian beach with natural three-point lighting setup including
soft key light from upper left creating gentle shadows on torso, subtle rim lighting
separating character from background, global illumination providing ambient fill
with coastal atmosphere, medium full shot composition with professional camera
framing at eye level, shallow depth of field with bokeh ocean background, 1024px
high resolution render with award-winning pixar animation quality, detailed 3d
character model with realistic skin materials, sharp focus on subject with cinematic
color grading, smooth shading with painterly italian coastal aesthetic
```

---

## 檢查品質

```bash
# 查看擴展後的 caption
cd /mnt/data/ai_data/datasets/3d-anime/luca/lora_data/training_data_sdxl/alberto_identity

# 隨機抽樣 3 個
for f in $(ls *.txt | shuf -n 3); do
    echo "=== $f ==="
    cat "$f"
    echo ""
done
```

---

## 常見問題

**Q: 成本多少？**
- 單一角色: ~$0.80-1.20 (300-500 captions)
- 全部 12 角色: ~$10-15 USD

**Q: 需要多久？**
- 單一角色: 2-5 分鐘
- 批次處理: 30-60 分鐘

**Q: 如何只處理部分角色？**
```bash
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py \
  --execute \
  --characters alberto giulia elio bryce caleb
```

**Q: 如何重新擴展已存在的角色？**
```bash
conda run -n ai_env python scripts/batch/expand_all_sdxl_captions.py \
  --execute \
  --no-skip-existing \
  --characters alberto
```

---

## 下一步

擴展完成後：

1. **檢查品質**: 隨機抽樣檢查擴展後的 captions
2. **生成訓練配置**: 為 SDXL LoRA 訓練創建 TOML 配置
3. **訓練 SDXL LoRAs**: 使用擴展後的 captions 訓練
4. **評估比較**: 與 SD1.5 版本比較品質

---

## 完整文件

詳細工作流程請參閱:
- [SDXL Caption Expansion Workflow](./SDXL_CAPTION_EXPANSION_WORKFLOW.md)
- [SDXL Caption Expansion Guide](./SDXL_CAPTION_EXPANSION.md)
