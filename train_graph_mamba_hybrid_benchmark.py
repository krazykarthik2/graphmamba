"""
Train and benchmark Pure Mamba vs Graph-Mamba Hybrid on 1D Sequential CIFAR-10.
Architecture Division of Labor:
  - Memory: Multi-scale continuous Graph Laplacian diffusion (parameter-free or minimal params).
  - Decision: Compact Selective State Space block (small d_state, small d_model).
"""

import sys
import time
import pickle
import tarfile
import argparse
import numpy as np
import torch
import torch.nn as nn
from model_graph_mamba import PureMambaSequence, GraphMambaHybrid

def load_data(tar_path='cifar-10-python.tar.gz', n_train=1000, n_test=200, pool_factor=4):
    print(f"Loading data from {tar_path}...")
    sys.stdout.flush()
    with tarfile.open(tar_path, 'r:gz') as tar:
        d1 = pickle.load(tar.extractfile('cifar-10-batches-py/data_batch_1'), encoding='bytes')
        te = pickle.load(tar.extractfile('cifar-10-batches-py/test_batch'), encoding='bytes')

    raw_tr = d1[b'data'][:n_train]
    y_tr = np.array(d1[b'labels'][:n_train])
    raw_te = te[b'data'][:n_test]
    y_te = np.array(te[b'labels'][:n_test])

    # Convert to standard ITU-R 601 grayscale
    gray_tr = (0.299 * raw_tr[:, :1024] + 0.587 * raw_tr[:, 1024:2048] + 0.114 * raw_tr[:, 2048:]) / 255.0
    gray_te = (0.299 * raw_te[:, :1024] + 0.587 * raw_te[:, 1024:2048] + 0.114 * raw_te[:, 2048:]) / 255.0

    seq_len = 1024 // pool_factor
    seq_tr = torch.from_numpy(gray_tr.reshape(-1, seq_len, pool_factor).mean(axis=-1)).float()
    seq_te = torch.from_numpy(gray_te.reshape(-1, seq_len, pool_factor).mean(axis=-1)).float()
    target_tr = torch.from_numpy(y_tr).long()
    target_te = torch.from_numpy(y_te).long()

    return seq_tr, target_tr, seq_te, target_te

def train_model(model, seq_tr, target_tr, seq_te, target_te, epochs=3, batch_size=32, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    num_batches = len(seq_tr) // batch_size

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(seq_tr))
        for b in range(num_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            optimizer.zero_grad()
            logits = model(seq_tr[idx])
            loss = criterion(logits, target_tr[idx])
            loss.backward()
            optimizer.step()
    wall_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        preds = model(seq_te).argmax(-1)
        acc = (preds == target_te).float().mean().item() * 100.0

    params = sum(p.numel() for p in model.parameters())
    return acc, params, wall_time

def main():
    parser = argparse.ArgumentParser(description="Head-to-head Mamba vs Graph-Mamba")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--pool_factor", type=int, default=4)
    args = parser.parse_args()

    seq_tr, target_tr, seq_te, target_te = load_data(n_train=args.n_train, n_test=args.n_test, pool_factor=args.pool_factor)
    print(f"Dataset: Train shape={seq_tr.shape}, Test shape={seq_te.shape}")
    sys.stdout.flush()

    print("\n[1/2] Training Pure Mamba (d=64, N=16)...")
    sys.stdout.flush()
    m_pure = PureMambaSequence(d_model=64, d_state=16)
    acc_pure, p_pure, t_pure = train_model(m_pure, seq_tr, target_tr, seq_te, target_te, epochs=args.epochs)
    print(f" -> Pure Mamba Done: Acc={acc_pure:.2f}%, Time={t_pure:.2f}s, Params={p_pure:,}")
    sys.stdout.flush()

    print("\n[2/2] Training Graph-Mamba Hybrid (d=32, N=4)...")
    sys.stdout.flush()
    m_hybrid = GraphMambaHybrid(d_model=32, d_state=4, num_scales=3)
    acc_hybrid, p_hybrid, t_hybrid = train_model(m_hybrid, seq_tr, target_tr, seq_te, target_te, epochs=args.epochs)
    print(f" -> Graph-Mamba Done: Acc={acc_hybrid:.2f}%, Time={t_hybrid:.2f}s, Params={p_hybrid:,}")
    sys.stdout.flush()

    print("\n" + "=" * 80)
    print("                 HEAD-TO-HEAD BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Model                   | Params  | Time (s) | Accuracy")
    print(f"  ------------------------+---------+----------+----------")
    print(f"  Pure Mamba (d=64, N=16) | {p_pure:7,d} | {t_pure:7.2f}s | {acc_pure:6.2f}%")
    print(f"  Graph-Mamba (d=32, N=4) | {p_hybrid:7,d} | {t_hybrid:7.2f}s | {acc_hybrid:6.2f}%")
    print("=" * 80)
    print(f"  Parameter Reduction    : {100.0 * (p_pure - p_hybrid) / p_pure:.1f}% fewer parameters")
    print(f"  Wall-Clock Speedup     : {t_pure / t_hybrid:.2f}x faster training")
    print(f"  Accuracy Margin        : {acc_hybrid - acc_pure:+.2f}% higher test accuracy")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
