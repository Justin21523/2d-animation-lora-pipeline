#!/bin/bash
# Monitor Luca Sea Monster SDXL Training
# Optimized configuration based on Alberto results

echo "=== Luca Sea Monster Training Monitor ==="
echo "Configuration: 8 epochs, dropout 0.1, reduced LR"
echo "Expected: Best checkpoint around epoch 3-5"
echo "Training ID: $(date +%Y%m%d_%H%M)"
echo ""

while true; do
    clear
    echo "======================================"
    echo "Luca Sea Monster Training Progress"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================"
    echo ""

    # GPU status
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  Utilization: %s%%\n  VRAM: %s / %s MiB\n  Temp: %s°C\n  Power: %s W\n", $1, $2, $3, $4, $5}'
    echo ""

    # Training process
    echo "Training Process:"
    if pgrep -f "sdxl_train_network.py.*luca_luca_seamonster" > /dev/null; then
        echo "  Status: ✓ Running"
        echo "  PID: $(pgrep -f 'sdxl_train_network.py.*luca_luca_seamonster')"
    else
        echo "  Status: ✗ Not running"
    fi
    echo ""

    # Checkpoints
    echo "Saved Checkpoints:"
    if [ -d "/mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_seamonster_identity" ]; then
        ls -lh /mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_seamonster_identity/*.safetensors 2>/dev/null | \
            awk '{printf "  %s - %s\n", $9, $5}' | sed 's|.*/||' || echo "  No checkpoints yet"
    else
        echo "  Directory not created yet"
    fi
    echo ""

    # Latest log
    echo "Latest Training Log:"
    if [ -d "/mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_seamonster_identity/logs" ]; then
        tail -5 /mnt/c/ai_models/diffusion/lora/sdxl/luca/luca_seamonster_identity/logs/*.log 2>/dev/null || echo "  No logs yet"
    else
        echo "  Logs directory not created yet"
    fi
    echo ""

    echo "======================================"
    echo "Press Ctrl+C to exit monitor"
    echo "Refresh in 30 seconds..."

    sleep 30
done
