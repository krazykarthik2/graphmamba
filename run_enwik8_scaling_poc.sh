#!/usr/bin/env bash
# Enwik8 Long-Sequence LM Scaling PoC Runner
set -e

python -u train_enwik8_scaling_poc.py --seq_len 1024 --steps 60 --batch_size 2
