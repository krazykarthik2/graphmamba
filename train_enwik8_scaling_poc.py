"""
Enwik8 Benchmark: Pure Mamba vs Graph-Mamba Hybrid (L=2048)
Evaluates Bits-Per-Character (BPC), Perplexity, and Wall-Clock Speed.
Formula:
  BPC = CrossEntropyLoss / ln(2)
  Perplexity = exp(CrossEntropyLoss)
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from model_causal_graph_mamba import CausalGraphMambaLM, PureMambaLM

def load_enwik8(path='enwik8', n_train_bytes=2000000, n_val_bytes=200000):
    print(f"Reading enwik8 from {path}...")
    sys.stdout.flush()
    with open(path, 'rb') as f:
        data = f.read(n_train_bytes + n_val_bytes)
    raw = np.frombuffer(data, dtype=np.uint8)
    train_data = raw[:n_train_bytes]
    val_data = raw[n_train_bytes : n_train_bytes + n_val_bytes]
    print(f"Loaded {len(train_data):,} train bytes, {len(val_data):,} validation bytes.")
    sys.stdout.flush()
    return train_data, val_data

def get_batches(data, seq_len=1024, batch_size=4):
    num_seqs = len(data) // (seq_len + 1)
    data = data[: num_seqs * (seq_len + 1)]
    data_tensor = torch.from_numpy(data.astype(np.int64))
    
    # Reshape to (num_seqs, seq_len + 1)
    seqs = data_tensor.view(num_seqs, seq_len + 1)
    
    # Pack into batches
    batches = []
    num_batches = num_seqs // batch_size
    for b in range(num_batches):
        batch = seqs[b * batch_size : (b + 1) * batch_size]
        x = batch[:, :-1]
        y = batch[:, 1:]
        batches.append((x, y))
    return batches

def evaluate(model, val_batches, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, 256), y.reshape(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2.0)
    ppl = math.exp(avg_loss)
    return bpc, ppl

def train_lm(model, train_batches, val_batches, max_steps=200, lr=1e-3, device='cpu'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time()
    step = 0
    b_idx = 0

    while step < max_steps:
        x, y = train_batches[b_idx % len(train_batches)]
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        b_idx += 1
        if step % 25 == 0 or step == max_steps:
            bpc_step = loss.item() / math.log(2.0)
            print(f"  Step {step:3d}/{max_steps} | Train Loss: {loss.item():.4f} | Train BPC: {bpc_step:.3f}")
            sys.stdout.flush()

    train_time = time.time() - t0
    val_bpc, val_ppl = evaluate(model, val_batches, criterion, device=device)
    params = sum(p.numel() for p in model.parameters())

    return val_bpc, val_ppl, params, train_time

def main():
    parser = argparse.ArgumentParser(description="Enwik8 Long-Sequence LM Scaling PoC")
    parser.add_argument("--seq_len", type=int, default=1024, help="Context length L (e.g. 1024 or 2048)")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()

    train_raw, val_raw = load_enwik8()
    train_batches = get_batches(train_raw, seq_len=args.seq_len, batch_size=args.batch_size)
    val_batches = get_batches(val_raw, seq_len=args.seq_len, batch_size=args.batch_size)[:10]

    print(f"\nConfiguration: Sequence Length L={args.seq_len}, Steps={args.steps}, Batch Size={args.batch_size}")
    sys.stdout.flush()

    print("\n" + "=" * 70)
    print(" [1/2] Benchmarking Pure Mamba LM (d_model=128, d_state=16)")
    print("=" * 70)
    m_pure = PureMambaLM(vocab_size=256, d_model=128, n_layers=2, d_state=16)
    bpc_pure, ppl_pure, p_pure, t_pure = train_lm(m_pure, train_batches, val_batches, max_steps=args.steps)
    print(f" -> Pure Mamba: Val BPC={bpc_pure:.3f} | Val PPL={ppl_pure:.2f} | Time={t_pure:.2f}s | Params={p_pure:,}")
    sys.stdout.flush()

    print("\n" + "=" * 70)
    print(" [2/2] Benchmarking Graph-Mamba Hybrid (d_model=96, d_state=4, 4-scale Laplacian)")
    print("=" * 70)
    m_hybrid = CausalGraphMambaLM(vocab_size=256, d_model=96, n_layers=2, d_state=4, num_scales=4)
    bpc_hybrid, ppl_hybrid, p_hybrid, t_hybrid = train_lm(m_hybrid, train_batches, val_batches, max_steps=args.steps)
    print(f" -> Graph-Mamba: Val BPC={bpc_hybrid:.3f} | Val PPL={ppl_hybrid:.2f} | Time={t_hybrid:.2f}s | Params={p_hybrid:,}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("                 ENWIK8 LONG-SEQUENCE SCALING PROOF OF CONCEPT")
    print("=" * 80)
    print(f"  Model                   | Params  | Time (s) | Val BPC | Val PPL")
    print(f"  ------------------------+---------+----------+---------+--------")
    print(f"  Pure Mamba (d=128, N=16)| {p_pure:7,d} | {t_pure:7.2f}s | {bpc_pure:7.3f} | {ppl_pure:7.2f}")
    print(f"  Graph-Mamba (d=96, N=4) | {p_hybrid:7,d} | {t_hybrid:7.2f}s | {bpc_hybrid:7.3f} | {ppl_hybrid:7.2f}")
    print("=" * 80)
    print(f"  Parameter Reduction    : {100.0 * (p_pure - p_hybrid) / p_pure:.1f}% fewer parameters")
    print(f"  Wall-Clock Speedup     : {t_pure / t_hybrid:.2f}x faster training")
    print(f"  Perplexity Improvement : {ppl_pure - ppl_hybrid:+.2f} PPL margin")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
