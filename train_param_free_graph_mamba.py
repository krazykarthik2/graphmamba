"""
Train Parameter-Free Graph-Mamba vs Pure Mamba Baseline on Enwik8 (L=2048).
Memory is handled by 100% parameter-free multi-scale Causal Laplacian diffusion.
Decisions are handled by compact Mamba (d_state=4).
Directly compares with Pure Mamba Baseline (2,603,520 params, 1063.96s, Val BPC=3.409, Val PPL=10.62).
"""

import os
import sys
import time
import math
import argparse
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from model_param_free_graph_mamba import ParameterFreeGraphMambaLM

# Historical Pure Mamba Baseline values (from previous full run)
BASELINE_PURE_MAMBA = {
    "name": "Pure Mamba (d_state=16)",
    "params": 2603520,
    "time_s": 1063.96,
    "tok_s": 2310,
    "val_bpc": 3.409,
    "val_ppl": 10.62,
    "step1_loss": 244.5598,
    "step10_loss": 28.7065,
    "step50_loss": 5.1641,
    "step100_loss": 4.4118,
    "step200_loss": 2.5036,
    "step300_loss": 2.1952
}

def prepare_data(data_path="enwik8", n_train_bytes=20_000_000, n_val_bytes=2_000_000):
    if not os.path.exists(data_path):
        print(f"Dataset {data_path} not found. Downloading enwik8.zip (100 MB)...")
        sys.stdout.flush()
        urllib.request.urlretrieve("http://mattmahoney.net/dc/enwik8.zip", "enwik8.zip")
        with zipfile.ZipFile("enwik8.zip", "r") as z:
            z.extractall(".")
        print("Download and extraction complete.")
        sys.stdout.flush()

    with open(data_path, "rb") as f:
        raw_bytes = f.read(n_train_bytes + n_val_bytes)
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    train_data = data[:n_train_bytes]
    val_data = data[n_train_bytes:n_train_bytes + n_val_bytes]
    print(f"Loaded: {len(train_data):,} training bytes, {len(val_data):,} validation bytes.")
    sys.stdout.flush()
    return train_data, val_data

def get_batches(data, seq_len=2048, batch_size=4, device='cuda'):
    num_seqs = len(data) // (seq_len + 1)
    data = data[: num_seqs * (seq_len + 1)]
    tensor = torch.from_numpy(data.astype(np.int64))
    seqs = tensor.view(num_seqs, seq_len + 1)
    
    batches = []
    num_batches = num_seqs // batch_size
    for b in range(num_batches):
        batch = seqs[b * batch_size : (b + 1) * batch_size]
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        batches.append((x, y))
    return batches

def evaluate(model, val_batches, criterion, use_amp=True):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_batches:
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.reshape(-1, 256), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2.0)
    try:
        ppl = math.exp(min(avg_loss, 50.0))
    except OverflowError:
        ppl = float('inf')
    return bpc, ppl

def main():
    parser = argparse.ArgumentParser(description="Param-Free Graph-Mamba Benchmark")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_name} ({gpu_props.total_memory / (1024**3):.1f} GB VRAM)")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sys.stdout.flush()

    train_raw, val_raw = prepare_data()
    train_batches = get_batches(train_raw, seq_len=args.seq_len, batch_size=args.batch_size, device=device)
    val_batches = get_batches(val_raw, seq_len=args.seq_len, batch_size=args.batch_size, device=device)[:20]

    print("\n" + "=" * 80)
    print(" PARAMETER-FREE GRAPH DIFFUSION MAMBA (L=2048) VS PREVIOUS BASELINE")
    print("=" * 80)

    model = ParameterFreeGraphMambaLM(vocab_size=256, d_model=256, n_layers=6, d_state=4, num_scales=4).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    footprint_mb = (num_params * 2 + num_params * 4 * 2) / (1024 * 1024)
    print(f"Model: Parameter-Free Graph-Mamba (d_model=256, d_state=4)")
    print(f"Parameters: {num_params:,} (vs Pure Mamba Baseline: {BASELINE_PURE_MAMBA['params']:,})")
    print(f"L2 Cache Footprint: {footprint_mb:.1f} MB (Fits entirely in L40S 96 MB L2 Cache)")
    sys.stdout.flush()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    total_tokens_processed = 0

    print(f"\nStarting training loop ({args.steps} steps)...")
    sys.stdout.flush()

    for step in range(1, args.steps + 1):
        x, y = train_batches[step % len(train_batches)]
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.reshape(-1, 256), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens_processed += x.numel()

        if step % 10 == 0 or step <= 5 or step == args.steps:
            bpc_step = loss.item() / math.log(2.0)
            elapsed = time.time() - t0
            tok_s = total_tokens_processed / (elapsed + 1e-6)
            print(f"  Step {step:3d}/{args.steps} | Train Loss: {loss.item():.4f} | BPC: {bpc_step:.3f} | Speed: {tok_s:7.0f} tok/s | Elapsed: {elapsed:5.1f}s")
            sys.stdout.flush()

    if device == 'cuda':
        torch.cuda.synchronize()
    wall_time = time.time() - t0
    tok_per_sec = total_tokens_processed / wall_time

    print("\nRunning Validation Evaluation on 20 batches of L=2048...")
    sys.stdout.flush()
    val_bpc, val_ppl = evaluate(model, val_batches, criterion, use_amp=(device == 'cuda'))

    print("\n" + "=" * 90)
    print("                    HEAD-TO-HEAD COMPARISON WITH PREVIOUS BASELINE")
    print("=" * 90)
    print(f"  Metric                     | Pure Mamba Baseline | Param-Free Graph-Mamba | Advantage / Gain")
    print(f"  ---------------------------+---------------------+------------------------+------------------")
    print(f"  State Dimension (d_state)  | N = 16              | N = 4                  | 4x smaller state")
    print(f"  Parameters                 | {BASELINE_PURE_MAMBA['params']:19,d} | {num_params:22,d} | {BASELINE_PURE_MAMBA['params'] - num_params:,} fewer params ({100.0 * (BASELINE_PURE_MAMBA['params'] - num_params) / BASELINE_PURE_MAMBA['params']:.1f}%)")
    print(f"  Wall-Clock Training Time   | {BASELINE_PURE_MAMBA['time_s']:18.2f}s | {wall_time:21.2f}s | {BASELINE_PURE_MAMBA['time_s'] / wall_time:.2f}x speedup")
    print(f"  Token Throughput (tok/s)   | {BASELINE_PURE_MAMBA['tok_s']:19,d} | {tok_per_sec:22,.0f} | {tok_per_sec / BASELINE_PURE_MAMBA['tok_s']:.2f}x throughput")
    print(f"  Validation BPC             | {BASELINE_PURE_MAMBA['val_bpc']:19.3f} | {val_bpc:22.3f} | {BASELINE_PURE_MAMBA['val_bpc'] - val_bpc:+.3f} BPC")
    print(f"  Validation Perplexity      | {BASELINE_PURE_MAMBA['val_ppl']:19.2f} | {val_ppl:22.2f} | {BASELINE_PURE_MAMBA['val_ppl'] - val_ppl:+.2f} PPL margin")
    print("=" * 90)

if __name__ == '__main__':
    main()
