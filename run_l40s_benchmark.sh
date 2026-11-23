#!/usr/bin/env bash
# ==============================================================================
# Graph-Mamba vs Pure Mamba L40S L2-Cache Resident Benchmark
# Single-command runner for Cloud VM with NVIDIA L40S GPU (No virtualenv)
# ==============================================================================
set -e

echo "=== System Environment & GPU Verification ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== Checking PyTorch & GPU availability ==="
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')" || python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

echo "=== Running Cache-Resident Long-Sequence Benchmark (L=2048) ==="
# Sized specifically to fit entirely within the 96 MB L2 cache of NVIDIA L40S
# Runs directly in the host environment (no venv)
if command -v python3 &> /dev/null; then
    python3 -u train_l40s_cache_benchmark.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
else
    python -u train_l40s_cache_benchmark.py --seq_len 2048 --steps 300 --batch_size 4 --lr 0.002
fi

