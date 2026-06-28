# ComfyUI Integration Guide

ComfyUI is installed at `/mnt/c/ai_tools/comfyui/` to provide visual workflow design and LoRA testing capabilities for the 3D animation pipeline.

## Quick Start

### Launch ComfyUI

```bash
/mnt/c/ai_tools/comfyui/start_comfyui.sh
```

Access the web interface at: **http://localhost:8188**

### First Time Setup

1. **Verify Installation**
   - Launch ComfyUI using the command above
   - Check that ComfyUI Manager appears in the UI (button on the right side)
   - Verify no red error messages in the console

2. **Check Custom Nodes**
   - Click "Manager" button
   - Go to "Installed" tab
   - Verify these nodes are installed:
     - ComfyUI-Manager
     - ComfyUI_IPAdapter_plus
     - ComfyUI-InstantID
     - ComfyUI-Frame-Interpolation
     - ComfyUI-Impact-Pack

3. **Download Required Models**
   - See `/mnt/c/ai_tools/comfyui/MODELS_DOWNLOAD_GUIDE.md` for detailed instructions
   - Essential models:
     - SDXL VAE fp16-fix
     - IPAdapter SDXL models
     - ControlNet OpenPose SDXL

## Directory Structure

```
/mnt/c/ai_tools/comfyui/
├── start_comfyui.sh          # Main launcher with RTX 5080 optimizations
├── setup_models.sh            # Model symlink setup script
├── .env                       # Environment variables
├── MODELS_DOWNLOAD_GUIDE.md   # Model download instructions
├── models/                    # Symlinks to /mnt/c/ai_models/
│   ├── checkpoints -> /mnt/c/ai_models/stable-diffusion/checkpoints
│   ├── loras_sdxl -> /mnt/c/ai_models/diffusion/lora/sdxl
│   ├── vae -> /mnt/c/ai_models/stable-diffusion/vae
│   └── ... (other model symlinks)
├── workflows/                 # Your custom workflows
│   ├── lora_testing/
│   ├── multi_character/
│   ├── video_generation/
│   └── controlnet_pose/
├── custom_nodes/              # Installed custom nodes
└── output -> /mnt/data/outputs/comfyui/
```

## Pre-configured Workflows

### 1. LoRA Checkpoint Comparison Testing

**Purpose**: Test multiple LoRA checkpoints side-by-side with identical parameters.

**Location**: `workflows/lora_testing/checkpoint_comparison.json` (to be created in UI)

**How to create**:
1. Load SDXL base checkpoint
2. Add multiple LoRA loaders (one per checkpoint to test)
3. Use same prompt and seed for all
4. Arrange outputs in a grid using `Image Batch` node
5. Save workflow

**Use case**: Select best checkpoint from Kohya training (epoch 3, 6, 9, 12)

### 2. Multi-Character Scene Composition

**Purpose**: Combine multiple character LoRAs in one scene.

**Location**: `workflows/multi_character/scene_composition.json`

**Key nodes**:
- InstantID (for each character)
- Regional Conditioning
- Multiple LoRA loaders
- Latent Composite

**Use case**: Test if LoRAs interfere with each other (e.g., Luca + Alberto)

### 3. Video Generation Pipeline

**Purpose**: Generate video from LoRA-based frames.

**Location**: `workflows/video_generation/batch_frame_to_video.json`

**Pipeline**:
1. Batch generate frames with LoRA
2. RIFE frame interpolation (2x or 4x)
3. FFmpeg video assembly

**Use case**: Create character animation showcases

### 4. ControlNet Pose Control

**Purpose**: Generate training data with specific poses.

**Location**: `workflows/controlnet_pose/pose_controlled_generation.json`

**Pipeline**:
1. Load reference image
2. Extract pose with OpenPose
3. Apply ControlNet conditioning
4. Generate with character LoRA

**Use case**: Augment training dataset with underrepresented poses

## Python API Integration

The `comfyui_lora_tester.py` script provides programmatic access to ComfyUI.

### Basic Usage

```python
from pathlib import Path
from scripts.evaluation.comfyui_lora_tester import ComfyUIClient, LoRACheckpointTester

# Initialize client
client = ComfyUIClient(host="localhost", port=8188)

# Load workflow template
workflow_path = Path("/mnt/c/ai_tools/comfyui/workflows/lora_testing/checkpoint_comparison.json")
tester = LoRACheckpointTester(workflow_path, client)

# Test checkpoints
checkpoint_dir = Path("/mnt/c/ai_models/diffusion/lora/sdxl/luca/")
output_dir = Path("/mnt/data/outputs/comfyui/lora_tests/luca/")

results = tester.test_checkpoints(
    checkpoint_paths=list(checkpoint_dir.glob("*.safetensors")),
    output_dir=output_dir,
    prompts=["a 3d animated boy character, pixar style, neutral pose"],
    seeds=[42, 123, 456]
)

print(f"Generated {sum(len(imgs) for imgs in results.values())} images")
```

### CLI Usage

```bash
conda activate ai_env

python /mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/evaluation/comfyui_lora_tester.py \
    /mnt/c/ai_models/diffusion/lora/sdxl/luca/ \
    --workflow /mnt/c/ai_tools/comfyui/workflows/lora_testing/checkpoint_comparison.json \
    --output-dir /mnt/data/outputs/comfyui/lora_tests/luca/ \
    --prompts "a 3d animated boy character, pixar style, standing pose" \
              "a 3d animated boy character, pixar style, smiling expression"
```

## RTX 5080 Optimizations

The launch script includes several GPU-specific optimizations:

```bash
# TF32 acceleration
export NVIDIA_TF32_OVERRIDE=1
export TORCH_BACKENDS_CUDA_MATMUL_ALLOW_TF32=1

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fast module loading
export CUDA_MODULE_LOADING=LAZY
```

**Expected Performance**:
- SDXL generation: ~25-35 seconds (30 steps, DPM++ 2M Karras)
- GPU utilization: >80% during generation
- VRAM usage: ~10-14 GB (SDXL + LoRA)

## Troubleshooting

### Custom Nodes Not Loading

**Symptom**: Red error messages about missing nodes

**Solution**:
```bash
cd /mnt/c/ai_tools/comfyui/custom_nodes/

# Check each node
for dir in ComfyUI-*; do
    echo "=== $dir ==="
    ls -la "$dir/requirements.txt" 2>/dev/null || echo "No requirements"
done

# Reinstall dependencies if needed
conda activate ai_env
pip install -r ComfyUI-Manager/requirements.txt
pip install -r ComfyUI-InstantID/requirements.txt
pip install -r ComfyUI-Impact-Pack/requirements.txt
```

### Models Not Appearing

**Symptom**: Dropdown menus are empty

**Solution**:
```bash
# Verify symlinks
ls -la /mnt/c/ai_tools/comfyui/models/

# Recreate if needed
/mnt/c/ai_tools/comfyui/setup_models.sh

# Restart ComfyUI
```

### CUDA Out of Memory

**Symptom**: Generation fails with OOM error

**Solution**:
1. Enable `--lowvram` flag in `start_comfyui.sh`:
   ```bash
   python main.py --listen --lowvram ...
   ```

2. Reduce batch size in workflows

3. Use `--cpu-vae` if VAE decoding causes issues

### API Connection Failed

**Symptom**: Python script cannot connect

**Solution**:
```bash
# Check ComfyUI is running
curl http://localhost:8188/system_stats

# Check firewall
sudo ufw allow 8188/tcp

# Verify port not in use
netstat -tuln | grep 8188
```

## Advantages Over Existing test_lora_checkpoints.py

| Feature | test_lora_checkpoints.py | ComfyUI |
|---------|--------------------------|---------|
| **Visual workflow** | ❌ Code-only | ✅ Drag-and-drop nodes |
| **Multi-LoRA testing** | ❌ One at a time | ✅ Side-by-side comparison |
| **InstantID/IPAdapter** | ❌ Not available | ✅ Character consistency |
| **ControlNet** | ❌ Limited support | ✅ Full integration |
| **Frame interpolation** | ❌ External tool | ✅ Built-in RIFE |
| **Workflow sharing** | ❌ Hard to reproduce | ✅ Export JSON |
| **Real-time preview** | ❌ Wait until done | ✅ See progress |

**Recommendation**: Use ComfyUI for:
- Checkpoint comparison and selection
- Multi-character scene testing
- Video generation
- Workflow prototyping

Use `test_lora_checkpoints.py` for:
- Automated batch testing
- CI/CD integration
- Scripted evaluation

## Next Steps

1. **Download essential models** (see MODELS_DOWNLOAD_GUIDE.md)
2. **Launch ComfyUI** and verify installation
3. **Create your first workflow** (basic SDXL + LoRA test)
4. **Save workflow template** for reuse
5. **Test Python API integration**
6. **Create custom workflows** for your specific needs

## Additional Resources

- [ComfyUI Official Docs](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager Guide](https://github.com/ltdrdata/ComfyUI-Manager)
- [InstantID Guide](https://github.com/cubiq/ComfyUI_InstantID)
- [ComfyUI Workflows Community](https://comfyworkflows.com/)
