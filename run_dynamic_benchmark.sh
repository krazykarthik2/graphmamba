#!/usr/bin/env bash
# ==============================================================================
# Dynamic Graph-Mamba (Win-At-All-Costs) vs Pure Mamba Baseline
# ==============================================================================
set -e

echo "=== System Environment & GPU Verification ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== Running Dynamic Graph-Mamba Benchmark (L=2048) ==="
if command -v python3 &> /dev/null; then
    python3 -u train_dynamic_graph_mamba.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
else
    python -u train_dynamic_graph_mamba.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
fi
