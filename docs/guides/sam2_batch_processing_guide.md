# SAM2 Instance Segmentation Batch Processing Guide

## Overview

This guide documents the optimized SAM2 instance segmentation batch processing pipeline with async I/O optimization for maximum GPU utilization.

## Current Optimized Configuration (MUST USE)

**As of 2025-12-11**, all SAM2 batch processing MUST use these parameters:

```python
# Optimized async I/O parameters for >90% GPU utilization
PREFETCH_SIZE = 32      # Image prefetch buffer size (increased from 16)
SAVE_WORKERS = 8        # Async save threads (increased from 4)
MAX_CONCURRENT = 3      # Max concurrent SAM2 processes (for 16GB GPU)
```

### Performance Results

| Metric | Old Configuration | New Configuration | Improvement |
|--------|------------------|-------------------|-------------|
| **GPU Utilization** | 4% (I/O bottleneck) | **75%** | **18.75x** |
| Prefetch Size | 16 | **32** | 2x |
| Save Workers | 4 | **8** | 2x |
| Max Concurrent | 2-3 | **3** | - |
| Estimated Time (45 episodes) | 20-24 hours | **4-6 hours** | **4-5x faster** |

## Scripts

### Primary Script: `process_all_super_wings.py`

**Location**: `/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/batch/process_all_super_wings.py`

**Features**:
- Processes all episodes in a batch
- Deletes first half of frames before processing (reduces dataset size)
- Manages concurrent SAM2 processes with OOM detection and automatic retry
- GPU memory management with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Progress tracking with real-time status updates
- **Automatic OOM Recovery**: Detects CUDA out of memory errors and retries up to 3 times with GPU memory clearing

**Usage**:
```bash
cd /mnt/c/ai_projects/3d-animation-lora-pipeline
nohup python scripts/batch/process_all_super_wings.py > /tmp/super_wings_optimized.log 2>&1 &
```

**Configuration**:
```python
# Configuration (UPDATED FOR OOM PREVENTION)
FRAMES_BASE = Path("/mnt/data/datasets/general/super-wings/frames")
INSTANCES_BASE = Path("/mnt/data/datasets/general/super-wings/instances")
MAX_CONCURRENT = 2  # Max concurrent SAM2 processes (REDUCED from 3 to avoid OOM)
CHECK_INTERVAL = 30  # Check progress every 30 seconds
MIN_INSTANCES_THRESHOLD = 500  # Consider complete if > 500 instances

# Optimized async I/O parameters for better GPU utilization
PREFETCH_SIZE = 32      # Increased from 16 - larger prefetch buffer
SAVE_WORKERS = 8        # Increased from 4 - more async save threads

# OOM Detection and Recovery (NEW)
OOM_KEYWORDS = ["CUDA out of memory", "OutOfMemoryError", "out of memory"]
MAX_OOM_RETRIES = 3     # Max retries for OOM errors

SCRIPT_PATH = "/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/segmentation/instance_segmentation.py"
LOG_DIR = Path("/tmp/super_wings_sam2_logs")
```

**SAM2 Command Line**:
```bash
conda run -n ai_env python instance_segmentation.py \
  <frames_dir> \
  --output-dir <output_dir> \
  --model sam2_hiera_large \
  --min-size 4096 \
  --device cuda \
  --context-mode transparent \
  --use-async-io \
  --prefetch-size 32 \
  --save-workers 8
```

### Deprecated Scripts (DO NOT USE)

❌ **`process_super_wings_half.py`** - Old version with inefficient I/O parameters
  - Used prefetch-size=16, save-workers=4
  - Resulted in only 4% GPU utilization
  - **DELETED** as of 2025-12-11

## Instance Segmentation Core: `instance_segmentation.py`

**Location**: `/mnt/c/ai_projects/3d-animation-lora-pipeline/scripts/generic/segmentation/instance_segmentation.py`

**Key Features**:
- **AsyncImageLoader**: Prefetches images in background threads
- **AsyncImageSaver**: Saves images asynchronously to disk
- **Producer-Consumer Pipeline**: GPU never waits for I/O
- **Memory Management**: PyTorch expandable segments

**CLI Arguments** (REQUIRED):
```bash
--use-async-io           # Enable async I/O optimization
--prefetch-size 32       # Prefetch buffer size (MUST be 32)
--save-workers 8         # Async save threads (MUST be 8)
```

**Other Important Arguments**:
```bash
--model sam2_hiera_large    # SAM2 model variant
--min-size 4096             # Minimum mask size (pixels)
--device cuda               # Use GPU
--context-mode transparent  # Output transparent PNGs
```

## Monitoring Processing Progress

### Check Running Processes
```bash
ps aux | grep process_all_super_wings
ps aux | grep instance_segmentation
```

### Check GPU Utilization
```bash
nvidia-smi dmon -s u
```

Expected GPU utilization: **>70%** (should be sustained)

### Check Log Files
```bash
# Main batch processing log
tail -f /tmp/super_wings_optimized.log

# Individual episode logs
ls /tmp/super_wings_sam2_logs/
tail -f /tmp/super_wings_sam2_logs/<episode_name>_sam2.log
```

### Check Output Progress
```bash
# Count completed instances
for dir in /mnt/data/datasets/general/super-wings/instances/*/; do
  count=$(find "$dir" -name "*.png" | wc -l)
  echo "$(basename "$dir"): $count instances"
done | sort
```

## Troubleshooting

### Low GPU Utilization (<50%)

**Symptom**: GPU utilization drops below 50%, processing is slow

**Solution**: Restart with optimized parameters
```bash
# Kill old processes
pkill -f "process_all_super_wings.py"
pkill -f "instance_segmentation.py.*super-wings"

# Restart with optimized parameters
cd /mnt/c/ai_projects/3d-animation-lora-pipeline
nohup python scripts/batch/process_all_super_wings.py > /tmp/super_wings_optimized.log 2>&1 &
```

### Out of Memory (OOM)

**Symptom**: CUDA out of memory errors

**✅ AUTOMATIC RECOVERY (NEW)**:
The batch processor now automatically detects and recovers from OOM errors:
1. Detects OOM keywords in log files ("CUDA out of memory", "OutOfMemoryError")
2. Waits 10 seconds for GPU memory to clear
3. Automatically retries failed episode up to 3 times
4. If still failing after 3 retries, marks as failed and continues to next episode

**Manual Solution** (if needed): Reduce MAX_CONCURRENT in script
```python
MAX_CONCURRENT = 1      # Most conservative - never OOM
```

**Best Practice**:
- For 16GB GPU: MAX_CONCURRENT = 2 (safe, good utilization)
- For 24GB GPU: MAX_CONCURRENT = 3 (optimal)
- For 12GB GPU: MAX_CONCURRENT = 1 (required)

### Process Stuck / No Progress

**Symptom**: No new files created for >5 minutes

**Check**:
```bash
# Check if processes are alive
ps aux | grep instance_segmentation

# Check latest file creation time
find /mnt/data/datasets/general/super-wings/instances -name "*.png" -mmin -5 | wc -l
```

**Solution**: Restart stuck episode
```bash
# Identify stuck PID
ps aux | grep instance_segmentation

# Kill stuck process
kill -9 <PID>

# Batch processor will auto-restart
```

## Best Practices

1. **Always use optimized parameters** (PREFETCH_SIZE=32, SAVE_WORKERS=8)
2. **Monitor GPU utilization** - should be >70%
3. **Run in background** with nohup to prevent SSH disconnection
4. **Check logs regularly** for errors
5. **Don't run multiple batch processors** - they will conflict
6. **Keep MAX_CONCURRENT=3** for 16GB GPU

## Architecture: Async I/O Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   ASYNC I/O PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Image Files on Disk]                                      │
│       │                                                     │
│       ↓                                                     │
│  ┌────────────────────────────┐                            │
│  │ AsyncImageLoader           │ (8 worker threads)         │
│  │ - ThreadPoolExecutor       │                            │
│  │ - Prefetch next 32 frames  │                            │
│  └────────────────────────────┘                            │
│       │                                                     │
│       ↓                                                     │
│  [LOAD_QUEUE] (size=32)                                    │
│       │                                                     │
│       ↓                                                     │
│  ┌────────────────────────────┐                            │
│  │ MAIN GPU THREAD            │ (single thread)            │
│  │ - Pop from LOAD_QUEUE      │                            │
│  │ - SAM2 inference           │ ← TARGET: 75%+ GPU util    │
│  │ - Push to SAVE_QUEUE       │                            │
│  └────────────────────────────┘                            │
│       │                                                     │
│       ↓                                                     │
│  [SAVE_QUEUE] (size=64)                                    │
│       │                                                     │
│       ↓                                                     │
│  ┌────────────────────────────┐                            │
│  │ AsyncImageSaver            │ (8 worker threads)         │
│  │ - ThreadPoolExecutor       │                            │
│  │ - Async disk writes        │                            │
│  └────────────────────────────┘                            │
│       │                                                     │
│       ↓                                                     │
│  [Saved Instance Masks]                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Historical Notes

### 2025-12-11: I/O Bottleneck Resolution

**Problem**: GPU utilization was only 4%, indicating severe I/O bottleneck

**Root Cause**:
- Insufficient prefetch buffer (16 → GPU starved)
- Insufficient save workers (4 → GPU blocked on saves)
- Sequential I/O pattern → GPU idle >90% of time

**Solution**:
- Increased PREFETCH_SIZE from 16 to 32
- Increased SAVE_WORKERS from 4 to 8
- Result: GPU utilization jumped from 4% to 75%

**Performance Impact**:
- Processing time reduced from 20-24 hours to 4-6 hours
- 18.75x improvement in GPU utilization
- 4-5x faster overall processing

## Related Documentation

- [SAM2 Model Documentation](../models/sam2.md)
- [Instance Segmentation Guide](./instance_segmentation_guide.md)
- [Batch Processing Best Practices](./batch_processing_guide.md)
- [GPU Memory Management](./gpu_memory_guide.md)

## References

- SAM2 Paper: "Segment Anything Model 2" (Meta AI, 2024)
- AsyncIO Best Practices: Python Threading and Multiprocessing
- GPU Utilization Optimization: NVIDIA CUDA Best Practices Guide
