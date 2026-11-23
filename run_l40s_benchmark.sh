#!/usr/bin/env bash
# ==============================================================================
# Graph-Mamba vs Pure Mamba L40S L2-Cache Resident Benchmark
# Single-command runner for Cloud VM with NVIDIA L40S GPU
# ==============================================================================
set -e

echo "=== System Environment & GPU Verification ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== Installing dependencies if needed ==="
pip install --upgrade pip torch torchvision torchaudio numpy --extra-index-url https://download.pytorch.org/whl/cu121 || pip install torch numpy

echo "=== Running Cache-Resident Long-Sequence Benchmark (L=2048) ==="
# Sized specifically to fit entirely within the 96 MB L2 cache of NVIDIA L40S
python -u train_l40s_cache_benchmark.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
