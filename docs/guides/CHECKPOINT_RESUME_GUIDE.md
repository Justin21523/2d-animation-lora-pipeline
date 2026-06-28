# Checkpoint & Resume Guide

Guide for adding checkpoint/resume capability to long-running scripts using the `CheckpointManager` class.

## Overview

The `CheckpointManager` provides automatic progress saving and resume functionality for scripts that process large numbers of files. This is essential for:

- **Long-running operations** (hours to days)
- **Unreliable environments** (network interruptions, power outages)
- **Resource limitations** (GPU memory, system crashes)
- **Iterative development** (testing and debugging)

## Quick Start

### 1. Basic Usage Pattern

```python
from scripts.core.utils.checkpoint_manager import CheckpointManager
from pathlib import Path

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager(
    checkpoint_path=output_dir / ".checkpoint.json",
    save_interval=100,  # Save every 100 items
    logger=logger
)

# Load existing checkpoint if resuming
if checkpoint_mgr.exists():
    checkpoint_mgr.load()
    logger.info(f"Resuming: {len(checkpoint_mgr)} items already processed")

# Get unprocessed items
image_files = list(input_dir.glob("*.jpg"))
unprocessed_files = checkpoint_mgr.get_unprocessed_items(image_files)

logger.info(f"Processing {len(unprocessed_files)} remaining files...")

# Process each item
for img_path in tqdm(unprocessed_files):
    # Your processing logic here
    result = process_image(img_path)

    # Mark as processed (auto-saves at intervals)
    checkpoint_mgr.mark_processed(img_path)

# Clean up checkpoint on successful completion
checkpoint_mgr.cleanup()
logger.info("Processing complete! Checkpoint removed.")
```

### 2. Manual Save Control

```python
# Disable auto-save for manual control
checkpoint_mgr = CheckpointManager(
    checkpoint_path=output_dir / ".checkpoint.json",
    save_interval=0,  # Disable auto-save
    logger=logger
)

# Process items
for i, item in enumerate(items):
    process_item(item)
    checkpoint_mgr.mark_processed(item)

    # Manual save at custom intervals
    if (i + 1) % 50 == 0:
        checkpoint_mgr.save(force=True)
        logger.info(f"Checkpoint saved at {i + 1} items")
```

### 3. Batch Processing

```python
# Process in batches
batch_size = 32

for i in range(0, len(unprocessed_files), batch_size):
    batch = unprocessed_files[i:i+batch_size]

    # Process batch
    results = process_batch(batch)

    # Mark entire batch as processed
    checkpoint_mgr.mark_batch_processed(batch)

    logger.info(f"Batch {i//batch_size + 1} complete")
```

### 4. Metadata Storage

```python
# Store additional metadata
checkpoint_mgr.update_metadata('total_images', len(all_images))
checkpoint_mgr.update_metadata('start_time', datetime.now().isoformat())

# Retrieve metadata
total = checkpoint_mgr.get_metadata('total_images', default=0)
```

## Integration Examples

### Example 1: Pose Detection Script

```python
class PoseLoRADataPreparer:
    def __init__(self, ...):
        # ... existing init code ...

        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_path=self.output_dir / ".pose_detection_checkpoint.json",
            save_interval=50,
            logger=self.logger
        )

    def _detect_poses(self, instance_files: List[Path]) -> List[Dict]:
        """Detect poses with checkpoint/resume support."""
        # Load checkpoint if exists
        if self.checkpoint_mgr.exists():
            self.checkpoint_mgr.load()
            self.logger.info(f"Resuming pose detection...")

        # Get unprocessed files
        unprocessed = self.checkpoint_mgr.get_unprocessed_items(instance_files)
        self.logger.info(f"Processing {len(unprocessed)} remaining images...")

        results = []

        for img_path in tqdm(unprocessed, desc="Detecting poses"):
            # Detect pose
            result = self.pose_detector.detect_with_person_bbox(str(img_path))

            results.append({
                'image_path': img_path,
                'keypoints': result['keypoints'],
                'scores': result['scores'],
                'success': result['success'],
                'num_valid': result['num_valid']
            })

            # Mark as processed (auto-saves every 50 items)
            self.checkpoint_mgr.mark_processed(img_path)

        # Save final checkpoint
        self.checkpoint_mgr.save(force=True)

        return results
```

### Example 2: SAM2 Instance Segmentation

```python
def segment_frames_with_checkpoint(
    input_dir: Path,
    output_dir: Path,
    model,
    logger
):
    """Segment frames with checkpoint/resume."""

    # Initialize checkpoint
    checkpoint_mgr = CheckpointManager(
        checkpoint_path=output_dir / ".sam2_checkpoint.json",
        save_interval=10,  # SAM2 is slow, save more frequently
        logger=logger
    )

    # Load checkpoint
    if checkpoint_mgr.exists():
        checkpoint_mgr.load()

    # Get all frames
    frame_files = sorted(input_dir.glob("*.jpg"))
    unprocessed = checkpoint_mgr.get_unprocessed_items(frame_files)

    logger.info(f"Segmenting {len(unprocessed)} frames...")

    for frame_path in tqdm(unprocessed, desc="Segmenting"):
        # Run SAM2 segmentation
        instances = segment_frame(frame_path, model)

        # Save instances
        save_instances(instances, output_dir)

        # Mark as processed
        checkpoint_mgr.mark_processed(frame_path)

    # Clean up
    checkpoint_mgr.cleanup()
```

### Example 3: Feature Extraction with Batching

```python
def extract_features_with_checkpoint(
    image_paths: List[Path],
    output_dir: Path,
    model,
    batch_size: int = 32
):
    """Extract features in batches with checkpointing."""

    checkpoint_mgr = CheckpointManager(
        checkpoint_path=output_dir / ".feature_checkpoint.json",
        save_interval=0,  # Manual save after each batch
        logger=logger
    )

    if checkpoint_mgr.exists():
        checkpoint_mgr.load()

    unprocessed = checkpoint_mgr.get_unprocessed_items(image_paths)

    for i in range(0, len(unprocessed), batch_size):
        batch = unprocessed[i:i+batch_size]

        # Extract features for batch
        features = model.extract_batch(batch)

        # Save features
        save_features(features, output_dir)

        # Mark batch as processed
        checkpoint_mgr.mark_batch_processed(batch)

        # Force save after each batch
        checkpoint_mgr.save(force=True)

    checkpoint_mgr.cleanup()
```

## Best Practices

### 1. Checkpoint File Location

Place checkpoint files in the output directory:
```python
# Good: Hidden file in output directory
checkpoint_path = output_dir / ".checkpoint.json"

# Bad: Shared location (conflicts with parallel runs)
checkpoint_path = Path("/tmp/checkpoint.json")
```

### 2. Save Frequency

Choose `save_interval` based on processing speed:

- **Fast operations** (< 1s/item): `save_interval=100-500`
- **Medium operations** (1-10s/item): `save_interval=50-100`
- **Slow operations** (> 10s/item): `save_interval=10-50`
- **Very slow** (> 60s/item): `save_interval=1-10`

### 3. Error Handling

Wrap processing in try-except and save checkpoint on errors:

```python
try:
    for item in unprocessed_items:
        result = process_item(item)
        checkpoint_mgr.mark_processed(item)
except Exception as e:
    logger.error(f"Error processing {item}: {e}")
    # Force save checkpoint before exiting
    checkpoint_mgr.save(force=True)
    raise
```

### 4. Parallel Processing

For multi-GPU or multi-process setups, use separate checkpoints:

```python
# GPU 0
checkpoint_mgr_0 = CheckpointManager(
    checkpoint_path=output_dir / ".checkpoint_gpu0.json",
    ...
)

# GPU 1
checkpoint_mgr_1 = CheckpointManager(
    checkpoint_path=output_dir / ".checkpoint_gpu1.json",
    ...
)
```

### 5. Memory Management

For large datasets, don't load all processed items into memory:

```python
# Instead of loading all results, only track which files processed
if not checkpoint_mgr.is_processed(item):
    result = process_item(item)
    # Save result to disk immediately
    save_result(result, output_dir)
    checkpoint_mgr.mark_processed(item)
```

## Advanced Features

### Thread-Safe Operations

CheckpointManager is thread-safe for multi-threaded processing:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for item in unprocessed_items:
        future = executor.submit(process_and_checkpoint, item, checkpoint_mgr)
        futures.append(future)

    # Wait for all to complete
    for future in futures:
        future.result()
```

### Custom Metadata

Store processing statistics in checkpoint:

```python
checkpoint_mgr.update_metadata('errors_count', error_count)
checkpoint_mgr.update_metadata('avg_processing_time', avg_time)
checkpoint_mgr.update_metadata('model_version', '1.0.0')

# On resume, check metadata
if checkpoint_mgr.exists():
    checkpoint_mgr.load()
    model_version = checkpoint_mgr.get_metadata('model_version')

    if model_version != '1.0.0':
        logger.warning("Model version mismatch! Starting fresh.")
        checkpoint_mgr.cleanup()
```

### Atomic Saves

CheckpointManager uses atomic file operations:
1. Writes to `.checkpoint.tmp`
2. Renames to `.checkpoint.json` (atomic operation)
3. Prevents corruption from interrupted saves

## Troubleshooting

### Checkpoint Not Loading

```python
if checkpoint_mgr.exists():
    success = checkpoint_mgr.load()
    if not success:
        logger.warning("Failed to load checkpoint, starting fresh")
        checkpoint_mgr.cleanup()
```

### Stale Checkpoints

If checkpoint is from old run with different parameters:

```python
# Store parameters in metadata
checkpoint_mgr.update_metadata('params', {
    'model': 'rtmpose-m',
    'threshold': 0.3,
    'batch_size': 32
})

# On resume, validate parameters
if checkpoint_mgr.exists():
    checkpoint_mgr.load()
    stored_params = checkpoint_mgr.get_metadata('params')

    if stored_params != current_params:
        logger.warning("Parameters changed, starting fresh")
        checkpoint_mgr.cleanup()
```

### Manual Cleanup

If script crashes before cleanup:

```bash
# Remove all checkpoint files
find /path/to/output -name ".checkpoint*.json" -delete
```

## Performance Impact

Checkpoint overhead is minimal:
- **Save operation**: ~1-10ms for 1000-10000 items
- **Load operation**: ~10-50ms for 10000 items
- **Memory overhead**: ~100 bytes per item

For 10,000 processed items with `save_interval=100`:
- Total checkpoint saves: ~100
- Total overhead: ~1 second
- Negligible compared to actual processing time

## Summary

CheckpointManager provides:
✅ Automatic progress tracking
✅ Resume from interruption
✅ Thread-safe operations
✅ Minimal performance overhead
✅ Easy integration (3-5 lines of code)
✅ Metadata storage
✅ Atomic file operations

**Always use checkpointing for operations taking > 10 minutes!**
