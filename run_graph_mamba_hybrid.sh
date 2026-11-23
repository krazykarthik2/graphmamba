#!/usr/bin/env bash
# Head-to-head Mamba vs Graph-Mamba Hybrid benchmark wrapper
set -e

python -u train_graph_mamba_hybrid_benchmark.py --epochs 3 --n_train 1000 --n_test 200 --pool_factor 4
