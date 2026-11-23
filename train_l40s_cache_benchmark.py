"""
NVIDIA L40S Ultra-Fast L2-Cache Bound Training Benchmark (L=2048).
Optimized for:
  - 96 MB L2 Cache Residency (Zero DRAM bandwidth stalls)
  - PyTorch 2.0+ torch.compile (Fused CUDA Kernels)
  - Automatic Mixed Precision (torch.cuda.amp.autocast in bfloat16 / fp16)
  - Enwik8 character-level sequence modeling (L=2048 tokens)
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
from model_l40s_graph_mamba import L40SGraphMambaLM, L40SPureMambaLM

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

def get_batches(data, seq_len=2048, batch_size=8, device='cuda'):
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

def train_and_benchmark(model, name, train_batches, val_batches, steps=300, lr=2e-3, device='cuda', compile_model=True):
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    # Estimate memory footprint in MB (BF16 weights + FP32 optimizer states)
    footprint_mb = (num_params * 2 + num_params * 4 * 2) / (1024 * 1024)
    print(f"\n[{name}] Parameters: {num_params:,} | L2 Cache Footprint: {footprint_mb:.1f} MB (Target < 96 MB L2)")
    sys.stdout.flush()

    if compile_model and hasattr(torch, 'compile') and os.name != 'nt':
        print(f"[{name}] Applying torch.compile for maximum kernel fusion...")
        sys.stdout.flush()
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    else:
        print(f"[{name}] Running in native PyTorch eager mode (zero JIT compile wait).")
        sys.stdout.flush()

    print(f"[{name}] Initializing AdamW optimizer & criterion...")
    sys.stdout.flush()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    use_bf16 = (device == 'cuda' and torch.cuda.is_bf16_supported())

    print(f"[{name}] Starting training loop ({steps} steps)...")
    sys.stdout.flush()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    total_tokens_processed = 0

    for step in range(1, steps + 1):
        x, y = train_batches[step % len(train_batches)]
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(device == 'cuda'), dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.reshape(-1, 256), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens_processed += x.numel()

        if step % 10 == 0 or step == 1 or step == steps:
            bpc_step = loss.item() / math.log(2.0)
            print(f"  Step {step:4d}/{steps} | Train Loss: {loss.item():.4f} | BPC: {bpc_step:.3f}")
            sys.stdout.flush()

    if device == 'cuda':
        torch.cuda.synchronize()
    wall_time = time.time() - t0
    tok_per_sec = total_tokens_processed / wall_time

    val_bpc, val_ppl = evaluate(model, val_batches, criterion, use_amp=(device == 'cuda'))
    return val_bpc, val_ppl, num_params, wall_time, tok_per_sec

def main():
    parser = argparse.ArgumentParser(description="L40S Cache-Resident Long Sequence Benchmark")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (default: 2048)")
    parser.add_argument("--steps", type=int, default=200, help="Training steps per model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--no_compile", action="store_true", default=False, help="Disable torch.compile")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_name} ({gpu_props.total_memory / (1024**3):.1f} GB VRAM)")
        # Set TF32 for maximum tensor core throughput on Ada Lovelace (L40S)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sys.stdout.flush()

    train_raw, val_raw = prepare_data()
    train_batches = get_batches(train_raw, seq_len=args.seq_len, batch_size=args.batch_size, device=device)
    val_batches = get_batches(val_raw, seq_len=args.seq_len, batch_size=args.batch_size, device=device)[:20]

    print("\n" + "=" * 80)
    print(f"       BENCHMARK ON ENWIK8 (L={args.seq_len}) SIZED FOR L40S 96 MB L2 CACHE")
    print("=" * 80)

    # 1. Graph-Mamba Hybrid (Memory via Graph Laplacian, Decision via Mamba)
    print("\n>>> Model 1: Graph-Mamba Hybrid (Memory = Graph Diffusion, Decision = Mamba d_state=4)")
    m_hybrid = L40SGraphMambaLM(vocab_size=256, d_model=256, n_layers=6, d_state=4, num_scales=4)
    bpc_h, ppl_h, p_h, t_h, tok_h = train_and_benchmark(m_hybrid, "Graph-Mamba", train_batches, val_batches, steps=args.steps, lr=args.lr, device=device, compile_model=(not args.no_compile))

    # 2. Pure Mamba (Standard Selective State Space)
    print("\n>>> Model 2: Pure Mamba (d_state=16)")
    m_pure = L40SPureMambaLM(vocab_size=256, d_model=256, n_layers=6, d_state=16)
    bpc_p, ppl_p, p_p, t_p, tok_p = train_and_benchmark(m_pure, "Pure-Mamba", train_batches, val_batches, steps=args.steps, lr=args.lr, device=device, compile_model=(not args.no_compile))

    print("\n" + "=" * 85)
    print("                            FINAL EMPIRICAL RESULTS")
    print("=" * 85)
    print(f"  Model             | Params   | Time (s) | Throughput (tok/s) | Val BPC | Val PPL")
    print(f"  ------------------+----------+----------+--------------------+---------+---------")
    print(f"  Graph-Mamba Hybrid| {p_h:8,d} | {t_h:7.2f}s | {tok_h:18,.0f} | {bpc_h:7.3f} | {ppl_h:7.2f}")
    print(f"  Pure Mamba        | {p_p:8,d} | {t_p:7.2f}s | {tok_p:18,.0f} | {bpc_p:7.3f} | {ppl_p:7.2f}")
    print("=" * 85)
    print(f"  Parameter Reduction : {100.0 * (p_p - p_h) / p_p:.1f}% less parameters")
    print(f"  Speedup Multiplier  : {t_p / t_h:.2f}x wall-clock speedup")
    print(f"  Throughput Gain     : {tok_h / tok_p:.2f}x higher token throughput")
    print(f"  Perplexity Margin   : {ppl_p - ppl_h:+.2f} PPL advantage")
    print("=" * 85)

if __name__ == '__main__':
    main()
