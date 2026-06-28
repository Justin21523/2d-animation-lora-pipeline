# 模型與資源存放規範 (/mnt/c/ai_models)

本文檔說明各類模型/資源建議放置的子資料夾與命名，程式碼預設會從這些位置尋找；若不存在則回退 stub/mock。

## 目錄對應
- detection/：YOLO 權重（pt/onnx/engine），例如 `detection/yolo11n.pt`, `detection/yolo11n.onnx`, `detection/yolo11n.engine`。
- segmentation/：ToonOut、BiRefNet 去背/分割權重（onnx/engine），例如 `segmentation/toonout.onnx`, `segmentation/birefnet.engine`。
- pose/（可新增）：DWpose/OpenPose 權重，例 `pose/dwpose.onnx`, `pose/dwpose.engine`。
- controlnet/：ControlNet 基底（pose/canny/lineart…）與自訓權重；可依任務再分子資料夾。
- stable-diffusion/：SD/SDXL 基底模型 safetensors/ckpt，例如 `stable-diffusion/sd15.safetensors`, `stable-diffusion/sdxl_base.safetensors`。
- lora/：SD1.5 LoRA 權重；lora_sdxl/：SDXL LoRA。
- embeddings/：CLIP/ArcFace/其他特徵模型，例 `embeddings/clip-ViT-B-16`, `embeddings/arcface.onnx`。
- interpolation/：RIFE/Practical-RIFE 等插幀模型（onnx/engine/pt）。
- inpainting/：LaMa/背景補洞/影像修復模型。
- flow/：光流模型（RAFT/GMFlow 等）。
- video/：影片復原/特化模型；upscale/（可新增）放 Real-ESRGAN 系列更清晰。
- 其他現有資料夾（audio、reranker、safety、vlm、llm、clip）保留原用途，非本管線主線。

## 程式碼預期與覆寫方式
- YOLO 偵測：`configs/run_yolo_tracking.yaml` 中可指定 `model_path`, `backend`（stub/pytorch/onnx/tensorrt），`device`。預設路徑可指向 `/mnt/c/ai_models/detection/<model>`.
- 分割：`configs/segment_fg_bg.yaml` 可指定 `model_path` + `backend`；建議放 `/mnt/c/ai_models/segmentation/<model>`.
- Pose：`configs/extract_pose.yaml` 可指定 DWpose/OpenPose `model_path` + `backend`；建議放 `/mnt/c/ai_models/pose/<model>`.
- ControlNet/LoRA/SD：訓練/推理腳本可指向 `stable-diffusion/`, `controlnet/`, `lora/`, `lora_sdxl/` 下的權重；如需自定可透過 config 覆寫。
- 插幀/超分：RIFE、Real-ESRGAN 等可放 `interpolation/`、`upscale/`（或 `video/`），供後續腳本引用。

## 命名建議
- 盡量包含模型名+版本：`yolo11n.onnx`, `dwpose-v1.1.onnx`, `realesrgan-anime.engine`, `rife-v4.6.onnx`.
- TensorRT engine 可用 `.engine` 後綴；PyTorch 用 `.pt`/`.pth`/`.safetensors`；ONNX 用 `.onnx`。

## 其他注意
- 未放入的權重會觸發 stub/mock 流程，請於放置後在對應 config 內填入 `model_path` 並將 `backend` 從 stub 換為 pytorch/onnx/tensorrt。
- 若新增子資料夾（例如 pose/、upscale/），保持小寫、無空白，方便程式與 bash 腳本引用。 
