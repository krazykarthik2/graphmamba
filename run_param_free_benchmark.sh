#!/usr/bin/env bash
# ==============================================================================
# Parameter-Free Graph Diffusion Mamba vs Previous Pure Mamba Baseline
# Runs only the Parameter-Free model and compares directly with previous results.
# ==============================================================================
set -e

echo "=== System Environment & GPU Verification ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== Running Parameter-Free Graph Diffusion Mamba (L=2048) ==="
if command -v python3 &> /dev/null; then
    python3 -u train_param_free_graph_mamba.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
else
    python -u train_param_free_graph_mamba.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
fi
