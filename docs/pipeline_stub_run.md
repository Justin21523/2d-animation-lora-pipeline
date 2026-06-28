# Stub 全流程執行指南

在尚未放入實際權重前，可用 stub/CPU 流程確認資料流與輸出格式。

## 前置
- 確保 `ai_env` 已安裝必需套件（pyyaml, pandas, pillow, pytest 等）。
- 目前所有 config 預設 `use_stub: true`，不需模型權重。

## 一鍵執行（推薦）
```bash
bash/run_full_pipeline_stub.sh
```
步驟：抽幀 → 去重 → 偵測/追蹤 → 分割 → 姿勢 → embeddings → datasets → 推理 → 動畫 → 超分 → 插幀。
輸出位置：
- `data_*` / `metadata/`：中繼資料與階段產物
- `outputs/`：推理、動畫、超分、插幀結果
- `logs/`：執行 log

## 單步執行
- 抽幀：`bash/run_extract_frames.sh --use-stub`
- 去重：`bash/run_dedupe_frames.sh`
- 偵測/追蹤：`bash/run_yolo_tracking.sh --use-stub`
- 分割：`bash/run_segment_fg_bg.sh --use-stub`
- 姿勢：`bash/run_extract_pose.sh --use-stub`
- Embeddings：`bash/build_embeddings.sh --use-stub`
- LoRA dataset：`bash/build_lora_dataset_characters.sh --use-stub`
- ControlNet pose dataset：`bash/build_controlnet_pose_dataset.sh --use-stub`
- 推理：`bash/infer_lora_controlnet_pose.sh --use-stub`
- 動畫：`bash/generate_animation_lora_controlnet_pose.sh --use-stub`
- 超分：`bash/run_upscale_realesrgan.sh --use-stub`
- 插幀：`bash/run_interpolate_rife.sh --use-stub`

## 切換到真實模型
- 將權重放入 `/mnt/c/ai_models/` 對應子資料夾，更新 config 中的 `model_path`、`backend`（pytorch/onnx/tensorrt）、`device`。
- 若模型缺失或載入失敗會自動回退 stub，log 會提示。
