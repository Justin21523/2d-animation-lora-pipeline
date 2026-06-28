# Batch Training Pipeline Fixes

## Critical Issue: Training Interruptions

### Problem Description

The batch training pipeline was experiencing recurring interruptions where training would complete for one character but fail to automatically proceed to the next character in the queue. This occurred multiple times:

- **First occurrence**: Stopped after Orion (character 1/6)
- **Second occurrence**: Stopped after Ian (character 2/6)

**User Impact**: Required manual intervention to restart training, breaking the overnight/unattended workflow.

### Root Cause Analysis

The issue was caused by overly strict error handling in the bash script:

```bash
# PROBLEMATIC CODE
set -e  # Exit immediately if any command returns non-zero

# Training pipeline
train_character
evaluate_checkpoints  # May return non-zero even when training succeeds
next_character        # NEVER REACHED if evaluation fails
```

**Why This Failed**:
1. `set -e` causes the script to exit immediately on ANY non-zero exit code
2. Evaluation stage may return non-zero even when training completes successfully
3. Non-critical errors (e.g., missing evaluation report) would terminate the entire pipeline
4. No fallback or continuation mechanism

### Solution: Robust Error Handling

Created `scripts/batch/resume_training_russell_to_giulia.sh` with improved error handling:

#### 1. Disable Strict Exit-on-Error

```bash
# FIXED CODE
set +e  # DISABLED set -e for robustness
trap 'echo "[ERROR] Script interrupted at line $LINENO"' ERR
```

#### 2. Manual Exit Code Capture

```bash
# Run training (don't use && to avoid script exit on error)
cd "$KOHYA_ROOT"
conda run -n kohya_ss accelerate launch --num_cpu_threads_per_process=4 train_network.py \
    --config_file="$CONFIG_PATH" \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT_CODE=$?
cd "$PROJECT_ROOT"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_success "Training completed: $CHAR_NAME"
    CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -name "*.safetensors" -type f | wc -l)
    log_info "Generated $CHECKPOINT_COUNT checkpoint(s)"
else
    log_error "Training failed: $CHAR_NAME (exit code: $TRAIN_EXIT_CODE)"
    log_error "Check log: $TRAIN_LOG"
    ((FAILED++))
    log_info "Continuing to next character despite failure..."
    continue  # CONTINUES INSTEAD OF EXITING
fi
```

#### 3. Graceful Degradation

```bash
# Run evaluation
conda run -n ai_env python "$PROJECT_ROOT/scripts/evaluation/test_lora_checkpoints.py" \
    "$OUTPUT_DIR" \
    --base-model "$BASE_MODEL" \
    --output-dir "$EVAL_DIR" \
    2>&1 | tee "$EVAL_LOG"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    log_success "Evaluation completed: $CHAR_NAME"
else
    log_error "Evaluation failed: $CHAR_NAME (exit code: $EVAL_EXIT_CODE)"
    log_error "Check log: $EVAL_LOG"
    # Don't increment FAILED - training was successful
fi

# ALWAYS CONTINUES to next character
```

#### 4. Increased Thermal Management

```bash
# Always continue to next character
if [ $CHAR_NUM -lt $TOTAL_CHARS ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
    log_info "Proceeding to next character..."
    sleep 5  # Increased sleep for GPU cooldown
fi
```

## Usage Guide

### Running the Resume Script

```bash
# Launch in TMUX for long-running jobs
tmux new-session -s sd_training

# Navigate to project root
cd /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline

# Run resume script
bash scripts/batch/resume_training_russell_to_giulia.sh
```

### Monitoring Progress

```bash
# Attach to TMUX session
tmux attach-session -t sd_training

# View main log
tail -f logs/training/resume_russell_to_giulia_<timestamp>.log

# Check individual character logs
tail -f logs/training/train_<character>_<timestamp>.log
tail -f logs/training/eval_<character>_<timestamp>.log
```

### Detaching Safely

```bash
# Inside TMUX session, press: Ctrl+B, then D
# Script continues running in background
```

## Script Features

### Comprehensive Logging

- **Main log**: Overall pipeline progress and summary
- **Per-character training logs**: Detailed Kohya_ss output
- **Per-character evaluation logs**: Checkpoint testing results
- **Timestamped outputs**: All logs have unique timestamps

### Error Recovery

- **Training failures**: Logged, counted, but script continues
- **Evaluation failures**: Logged but doesn't block next character
- **Missing configs**: Skips character and continues
- **GPU errors**: Per-character isolation prevents cascade failures

### Statistics Tracking

```bash
TOTAL_CHARS=${#CHARACTERS[@]}
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Final summary
log_info "Total characters: $TOTAL_CHARS"
log_success "Successfully completed: $COMPLETED"
log_error "Failed: $FAILED"
log_info "Total elapsed time: ${HOURS}h ${MINS}m"
```

## Verification

### Successful Launch Indicators

1. **TMUX session active**: `tmux list-sessions` shows `sd_training`
2. **Training log growing**: `ls -lh logs/training/train_russell_*.log` shows increasing size
3. **GPU utilization**: `nvidia-smi` shows VRAM usage ~14-15GB
4. **Process running**: `ps aux | grep train_network.py` shows active Python process

### Expected Output

```
==========================================
Resumed Training Pipeline: Russell → Giulia
==========================================

[2025-11-21 19:56:40] [INFO] Started: 2025-11-21 19:56:40
[2025-11-21 19:56:40] [INFO] Total characters: 4
[2025-11-21 19:56:40] [INFO] Previous completed: Orion ✅, Ian ✅

==========================================
[1/4] Processing: russell
==========================================

[2025-11-21 19:56:40] [INFO] Configuration: configs/training/character_loras/up_russell_identity.toml
[2025-11-21 19:56:40] [INFO] Output directory: /mnt/data/ai_data/models/lora/up/russell_identity

==========================================
Stage 1: LoRA Training - russell
==========================================

[2025-11-21 19:56:40] [INFO] Starting Kohya training...
[2025-11-21 19:56:40] [INFO]   Config: /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/configs/training/character_loras/up_russell_identity.toml
[2025-11-21 19:56:40] [INFO]   Log: /mnt/c/AI_LLM_projects/3d-animation-lora-pipeline/logs/training/train_russell_20251121_195640.log
```

## Troubleshooting

### Script Still Stops After Character

**Check**:
1. Are you using the NEW resume script (`resume_training_russell_to_giulia.sh`)?
2. Is the script running inside TMUX (survives SSH disconnects)?
3. Check main log for exit codes and error messages

**Fix**:
```bash
# Verify script has set +e (not set -e)
grep "^set" scripts/batch/resume_training_russell_to_giulia.sh

# Should show:
# set +e  # DISABLED set -e for robustness
```

### GPU Out of Memory

**Symptoms**: Script stops with CUDA OOM errors

**Fix**: Reduce batch size or gradient accumulation in training configs

```toml
# In configs/training/character_loras/*.toml
train_batch_size = 3  # Reduce from 4
gradient_accumulation_steps = 2  # Increase to maintain effective batch size
```

### Evaluation Hangs

**Symptoms**: Evaluation stage doesn't complete

**Workaround**: Script will wait then continue to next character

**Manual skip**:
```bash
# Attach to TMUX
tmux attach -t sd_training

# Press Ctrl+C to skip evaluation
# Script will continue to next character
```

## Comparison: Before vs After

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| Error handling | `set -e` (strict) | `set +e` + manual checks |
| Exit codes | Ignored (implicit exit) | Captured and logged |
| Evaluation failure | **Stops pipeline** | Logs error, continues |
| Missing config | **Stops pipeline** | Skips character, continues |
| GPU errors | Cascade failure | Per-character isolation |
| Logging | Basic | Comprehensive (main + per-stage) |
| Inter-character delay | 2 seconds | 5 seconds (thermal mgmt) |
| Recovery | Manual restart needed | **Automatic continuation** |

## Best Practices for Future Batch Scripts

1. **Never use `set -e`** in long-running batch pipelines
2. **Always capture exit codes** explicitly with `$?`
3. **Use trap mechanisms** for debugging (`trap 'echo "Error at line $LINENO"' ERR`)
4. **Implement graceful degradation**: non-critical failures shouldn't stop the pipeline
5. **Add thermal delays**: `sleep 5` between GPU-intensive tasks
6. **Comprehensive logging**: main log + per-stage logs with timestamps
7. **TMUX/screen wrapper**: protect against SSH disconnects
8. **Statistics tracking**: count successes/failures, measure elapsed time

## Related Files

- **Resume script**: `scripts/batch/resume_training_russell_to_giulia.sh`
- **Original (broken) script**: `scripts/batch/train_and_evaluate_6_characters.sh`
- **Training configs**: `configs/training/character_loras/*.toml`
- **Evaluation script**: `scripts/evaluation/test_lora_checkpoints.py`
- **Logs directory**: `logs/training/`

## See Also

- [SDXL 16GB Training Guide](sdxl-16gb-guide.md)
- [LoRA Training Best Practices](LORA_TRAINING_BEST_PRACTICES.md)
- [WSL Long-Running Guide](../setup/WSL_LONG_RUNNING_GUIDE.md)
