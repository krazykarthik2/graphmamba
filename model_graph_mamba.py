import torch
import torch.nn as nn
import torch.nn.functional as F

class PureMambaSequence(nn.Module):
    def __init__(self, d_model=64, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.embed = nn.Linear(1, d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.x_proj = nn.Linear(d_model, 1 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        u = self.embed(x.unsqueeze(-1))
        B, L, D = u.shape
        xz = self.in_proj(u)
        x_m, z = xz.chunk(2, dim=-1)
        x_m = F.silu(self.conv1d(x_m.transpose(1, 2)).transpose(1, 2))
        
        ssm_p = self.x_proj(x_m)
        dt = F.softplus(self.dt_proj(ssm_p[..., :1]))
        B_ssm = ssm_p[..., 1:1+self.d_state]
        C_ssm = ssm_p[..., 1+self.d_state:]
        A = -torch.exp(self.A_log)
        
        h = torch.zeros(B, D, self.d_state, device=u.device)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A.unsqueeze(0))
            dB = dt_t * B_ssm[:, t, :].unsqueeze(1)
            h = dA * h + dB * x_m[:, t, :].unsqueeze(-1)
            y_t = (h * C_ssm[:, t, :].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) + x_m * self.D.unsqueeze(0).unsqueeze(0)
        out = self.out_proj(y * F.silu(z))
        return self.head(out.mean(dim=1))

class GraphMambaHybrid(nn.Module):
    def __init__(self, d_model=48, d_state=4, num_scales=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.embed = nn.Linear(1, d_model)
        
        # 1. Graph Laplacian Memory Bank
        alphas = torch.sigmoid(torch.linspace(-2.0, 3.0, num_scales))
        self.register_buffer('alphas', alphas)
        self.graph_mix = nn.Linear(d_model * num_scales, d_model, bias=False)
        
        # 2. Compact Mamba Selective Decision Block
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.x_proj = nn.Linear(d_model, 1 + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        u = self.embed(x.unsqueeze(-1))
        B, L, D = u.shape
        
        # Phase 1: Multi-Scale Graph Laplacian Diffusion
        diffs = []
        for a in self.alphas:
            y_diff = torch.zeros_like(u)
            cur = torch.zeros(B, D, device=u.device)
            for t in range(L):
                cur = a * cur + (1.0 - a) * u[:, t, :]
                y_diff[:, t, :] = cur
            diffs.append(y_diff)
        h_graph = self.graph_mix(torch.cat(diffs, dim=-1))
        
        u_fused = u + h_graph
        
        # Phase 2: Mamba Selective Decision
        xz = self.in_proj(u_fused)
        x_m, z = xz.chunk(2, dim=-1)
        x_m = F.silu(self.conv1d(x_m.transpose(1, 2)).transpose(1, 2))
        
        ssm_p = self.x_proj(x_m)
        dt = F.softplus(self.dt_proj(ssm_p[..., :1]))
        B_ssm = ssm_p[..., 1:1+self.d_state]
        C_ssm = ssm_p[..., 1+self.d_state:]
        A = -torch.exp(self.A_log)
        
        h = torch.zeros(B, D, self.d_state, device=u.device)
        ys = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A.unsqueeze(0))
            dB = dt_t * B_ssm[:, t, :].unsqueeze(1)
            h = dA * h + dB * x_m[:, t, :].unsqueeze(-1)
            y_t = (h * C_ssm[:, t, :].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1) + x_m * self.D.unsqueeze(0).unsqueeze(0)
        out = self.out_proj(y * F.silu(z))
        return self.head(out.mean(dim=1))

if __name__ == "__main__":
    m_pure = PureMambaSequence(d_model=64, d_state=16)
    m_hybrid = GraphMambaHybrid(d_model=48, d_state=4)
    p_pure = sum(p.numel() for p in m_pure.parameters())
    p_hybrid = sum(p.numel() for p in m_hybrid.parameters())
    print(f"Pure Mamba Parameters       : {p_pure:,}")
    print(f"Graph-Mamba Parameters      : {p_hybrid:,} ({100*(p_hybrid-p_pure)/p_pure:.1f}% reduction!)")
