# 專案進度摘要

## 已完成（stub/骨架）
- M0-M3：抽幀、去重、YOLO 偵測/追蹤（含 ONNX/TensorRT 骨架）、分割（ToonOut/BiRefNet ONNX/TRT 骨架）、姿勢（DWpose ONNX/TRT 骨架）。
- M4：Embeddings / Dataset builders（角色 LoRA、ControlNet pose）。
- M5/M6：LoRA/ControlNet 訓練：stub Tiny 模型 + diffusers LoRA/ControlNet 簡易骨架（無權重時回退 stub）。
- M7：推理/動畫：diffusers 推理、AnimateDiff 全序列生成（含 motion bucket/fps/插幀參數），ffmpeg 影片合成；超分/插幀 (Real-ESRGAN/RIFE) ONNX/TRT 前後處理骨架。
- 工具與 QC：一鍵 stub 流程、log/metadata 檢查、loss/影片/可視化 QC。

## 待接入（需權重/實際資料）
- 實際權重：YOLO、ToonOut/BiRefNet、DWpose、Real-ESRGAN、RIFE、SD/ControlNet/LoRA、AnimateDiff/motion 模組。
- 實戰前處理/後處理確認：TensorRT/ONNX engine 的 I/O 格式校準。
- 真實訓練資料：接入 train_data_dir（影像 + prompt/condition），完善 diffusers 訓練流程（資料管線、保存 safetensors）。
- AnimateDiff 進一步參數校準（若有特定 motion bucket/fps/插幀設定）。
- QC/監控：可再加入遮罩/pose/偵測抽樣自動生成報表、訓練指標彙整。

## 下一步建議
1. 放置權重至 `/mnt/c/ai_models/` 對應子資料夾並更新 config 的 `model_path/backend/device`。
2. 用小樣本驗證 ONNX/TensorRT 路徑（YOLO、DWpose、Segmentation、Real-ESRGAN、RIFE），必要時微調前處理/後處理。
3. 將 diffusers 訓練切換為真實資料（關閉 use_stub、開啟 enable_diffusers），並檢查 loss 監控。
4. AnimateDiff：填入 `animatediff_path`（motion 模組可選），跑一次全序列 + 影片 mux，使用 `qc_video.sh` 驗證影片資訊。
