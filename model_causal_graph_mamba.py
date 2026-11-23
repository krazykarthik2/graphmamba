"""
Causal Graph-Mamba Language Model for Enwik8 (L=2048)
Architectural Division of Labor:
  - Memory: Multi-scale Continuous Causal Graph Laplacian temporal diffusion:
      h_t^(k) = alpha_k * h_{t-1}^(k) + (1 - alpha_k) * x_t
      where alpha_k = exp(-Delta tau_k) covers multi-scale time horizons.
      Fast O(L) scan / 1D causal depthwise recurrence with zero quadratic attention cost.
  - Decision: Selective State Space Model (Mamba) with compact state dimension (N=4)
      and reduced model width (d_model), eliminating parameter waste.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCausalLaplacianDiffusion(nn.Module):
    """
    Continuous Multi-Scale Causal Graph Laplacian Diffusion Operator.
    Propagates memory across L steps using log-spaced continuous diffusion rates.
    """
    def __init__(self, d_model, num_scales=4):
        super().__init__()
        self.d_model = d_model
        self.num_scales = num_scales
        # Log-spaced continuous decay rates: timescales ~ 2, 8, 32, 128 tokens
        init_timescales = torch.logspace(math.log10(2.0), math.log10(128.0), num_scales)
        # alpha = exp(-1.0 / timescale)
        init_alphas = torch.exp(-1.0 / init_timescales)
        # Store as unconstrained logit parameter
        self.alpha_logits = nn.Parameter(torch.logit(init_alphas))
        self.proj = nn.Linear(d_model * (num_scales + 1), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        alphas = torch.sigmoid(self.alpha_logits) # (num_scales,)
        
        diffused_scales = []
        for s in range(self.num_scales):
            a = alphas[s]
            # Fast causal exponential recurrence: h_t = a * h_{t-1} + (1 - a) * x_t
            # Computed via associative scan or sequential step in PyTorch
            h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            h_seq = []
            for t in range(L):
                h = a * h + (1.0 - a) * x[:, t, :]
                h_seq.append(h)
            h_stack = torch.stack(h_seq, dim=1) # (B, L, D)
            diffused_scales.append(h_stack)

        combined = torch.cat([x] + diffused_scales, dim=-1) # (B, L, D * (num_scales + 1))
        out = self.proj(combined)
        return self.norm(out)

class CompactMambaSelectiveBlock(nn.Module):
    """
    Mamba Selective State Space Block with compressed state dimension N.
    """
    def __init__(self, d_model, d_state=4, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # Selective projection: x_proj maps to (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, 1 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Continuous S4 / HiPPO diagonal init
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        # u: (B, L, D)
        B, L, _ = u.shape
        xz = self.in_proj(u) # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Causal Conv1d
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)

        # Selective gating
        ssm_params = self.x_proj(x_act)
        dt = F.softplus(self.dt_proj(ssm_params[:, :, :1])) # (B, L, d_inner)
        B_mat = ssm_params[:, :, 1 : 1 + self.d_state] # (B, L, d_state)
        C_mat = ssm_params[:, :, 1 + self.d_state :]   # (B, L, d_state)

        A = -torch.exp(self.A_log) # (d_inner, d_state)

        # Selective recurrence
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        y_list = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            dA = torch.exp(dt_t * A.unsqueeze(0)) # (B, d_inner, d_state)
            B_t = B_mat[:, t, :].unsqueeze(1) # (B, 1, d_state)
            dB = dt_t * B_t # (B, d_inner, d_state)

            x_t = x_act[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            h = h * dA + dB * x_t

            C_t = C_mat[:, t, :].unsqueeze(-1) # (B, d_state, 1)
            y_t = torch.matmul(h, C_t).squeeze(-1) # (B, d_inner)
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1) # (B, L, d_inner)
        y = y + x_act * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)

class CausalGraphMambaLM(nn.Module):
    """
    Causal Graph-Mamba Language Model.
    Layer stack combining Causal Graph Laplacian diffusion memory and Mamba selective decisions.
    """
    def __init__(self, vocab_size=256, d_model=128, n_layers=4, d_state=4, num_scales=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'graph_memory': MultiScaleCausalLaplacianDiffusion(d_model, num_scales=num_scales),
                'norm2': nn.LayerNorm(d_model),
                'mamba_decision': CompactMambaSelectiveBlock(d_model, d_state=d_state, d_conv=4, expand=2)
            }))

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight # Weight tying

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embedding(input_ids)
        for layer in self.layers:
            # Memory step
            x = x + layer['graph_memory'](layer['norm1'](x))
            # Decision step
            x = x + layer['mamba_decision'](layer['norm2'](x))
        x = self.final_norm(x)
        logits = self.head(x) # (B, L, vocab_size)
        return logits

class PureMambaLM(nn.Module):
    """
    Standard Pure Mamba Language Model (requires higher d_state=16 and d_model to match capacity).
    """
    def __init__(self, vocab_size=256, d_model=192, n_layers=4, d_state=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'mamba': CompactMambaSelectiveBlock(d_model, d_state=d_state, d_conv=4, expand=2)
            }))

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer['mamba'](layer['norm'](x))
        x = self.final_norm(x)
        logits = self.head(x)
        return logits
