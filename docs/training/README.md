# Training Documentation Index

This directory contains comprehensive guides for training LoRA models in the 3D Animation LoRA Pipeline.

## 📚 Available Guides

### SDXL Training

- **[SDXL Training Complete Guide](./SDXL_TRAINING_COMPLETE_GUIDE.md)** - 完整的 SDXL 角色 LoRA 訓練指南
  - 快速開始和配置
  - 啟動和監控訓練
  - TensorBoard 使用
  - 故障排除和優化
  - 批次訓練策略

### LoRA Types

- **[Character Identity LoRA](./lora_types/)** - 角色身份 LoRA 訓練
- **[Style LoRA](./lora_types/)** - 風格 LoRA 訓練
- **[Pose/Action LoRA](./lora_types/)** - 姿勢/動作 LoRA 訓練

## 🚀 Quick Start

### 訓練角色 Identity LoRA (SDXL)

```bash
# 1. 準備數據集
# 確保數據集位於: /mnt/data/datasets/general/{movie}/lora_data/training_data_sdxl/{character}_identity/

# 2. 啟動訓練
tmux new-session -d -s {char}_sdxl_training
tmux send-keys -t {char}_sdxl_training "cd /mnt/c/ai_projects/kohya_ss/sd-scripts" C-m
tmux send-keys -t {char}_sdxl_training "conda activate kohya_ss" C-m
tmux send-keys -t {char}_sdxl_training "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 4 sdxl_train_network.py --config_file=/mnt/c/ai_projects/3d-animation-lora-pipeline/configs/training/character_loras_sdxl/{movie}_{char}_identity_sdxl.toml" C-m

# 3. 監控訓練
bash scripts/batch/monitor_sdxl_training.sh {char}

# 4. 啟動 TensorBoard
tmux new-session -d -s {char}_tensorboard "cd /mnt/data/training/lora/{movie}/{char}_identity/logs && conda activate kohya_ss && tensorboard --logdir . --port 6006 --bind_all"
```

## 🎯 Training Workflow

```
數據準備 → 配置設置 → 啟動訓練 → 監控進度 → 評估 Checkpoints → 選擇最佳模型
   ↓            ↓            ↓            ↓              ↓                ↓
Dataset    TOML Config   Tmux +     Monitor +    Test Multiple      Deploy
Curation   Generation    Accelerate  TensorBoard  Epochs (6,8,10)   Best LoRA
```

## 📊 Training Status

### Current Training Sessions

查看當前訓練狀態：

```bash
# 列出所有 tmux sessions
tmux list-sessions

# 監控特定角色
bash scripts/batch/monitor_sdxl_training.sh {character}
```

### Recently Completed

- ✅ Miguel (Up) - SDXL Identity LoRA - 10 epochs - **成功**
- ✅ Orion (Elio) - SDXL Identity LoRA - 4 epochs - **成功**
- 🔄 Elio (Elio) - SDXL Identity LoRA - 10 epochs - **訓練中**
- ⏳ Tyler (Turning Red) - SDXL Identity LoRA - 10 epochs - **待啟動**
- ⏳ Caleb (Elio) - SDXL Identity LoRA - 10 epochs - **待啟動**

## 🔧 Common Commands

### Training Management

```bash
# Start training
bash scripts/batch/train_sdxl_with_auto_eval.sh configs/training/character_loras_sdxl/{config}.toml

# Monitor training
bash scripts/batch/monitor_sdxl_training.sh {character}

# Stop training gracefully
tmux send-keys -t {char}_sdxl_training C-c

# Force stop
tmux kill-session -t {char}_sdxl_training
```

### Monitoring

```bash
# GPU monitoring
watch -n 2 nvidia-smi

# Training log
tail -f /mnt/c/ai_projects/3d-animation-lora-pipeline/logs/{movie}_{char}_identity_sdxl_training_*.log

# TensorBoard
# Access at http://localhost:6006
```

### Checkpoints

```bash
# List checkpoints
ls -lht /mnt/data/training/lora/{movie}/{char}_identity/*.safetensors

# Evaluate checkpoint
python scripts/evaluation/sdxl_lora_evaluator.py \
  --checkpoint /mnt/data/training/lora/{movie}/{char}_identity/{char}_identity_lora_sdxl-000006.safetensors \
  --base-model /mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors \
  --output-dir outputs/evaluation/{char}_epoch6
```

## 📁 Important Locations

### Configuration Files

- Training configs: `configs/training/character_loras_sdxl/*.toml`
- Training scripts: `scripts/batch/train_sdxl_with_auto_eval.sh`
- Monitoring scripts: `scripts/batch/monitor_sdxl_training.sh`

### Output Locations

- Checkpoints: `/mnt/data/training/lora/{movie}/{character}_identity/*.safetensors`
- TensorBoard logs: `/mnt/data/training/lora/{movie}/{character}_identity/logs/`
- Training logs: `/mnt/c/ai_projects/3d-animation-lora-pipeline/logs/*_training_*.log`

### Datasets

- SDXL training data: `/mnt/data/datasets/general/{movie}/lora_data/training_data_sdxl/{character}_identity/`

## 🐛 Troubleshooting

See the [SDXL Training Complete Guide](./SDXL_TRAINING_COMPLETE_GUIDE.md#故障排除) for comprehensive troubleshooting.

**Common Issues:**

- Training stops unexpectedly → Check logs and GPU memory
- Loss is NaN → Reduce learning rate
- TensorBoard not accessible → Restart TensorBoard session
- Slow training speed → Check data loader workers and GPU utilization

## 📚 Additional Resources

- [Kohya_ss Documentation](https://github.com/kohya-ss/sd-scripts)
- [SDXL Training Guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md)
- [LoRA Training Best Practices](../../docs/guides/)

---

**Last Updated:** 2025-11-28
