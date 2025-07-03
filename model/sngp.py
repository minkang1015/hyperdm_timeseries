# models/sngp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralDense(nn.Linear):
    def forward(self, x):
        # 1-step power iteration
        v = F.normalize(self.weight.data.t() @ self.u, dim=0)
        self.u = F.normalize(self.weight.data @ v, dim=0)
        w_sn = self.weight / (self.u @ (self.weight @ v))
        return F.linear(x, w_sn, self.bias)

class SNGPHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dense = SpectralDense(in_dim, 1, bias=False)
        self.u     = nn.Parameter(torch.randn(in_dim), requires_grad=False)
        self.rho   = nn.Parameter(torch.tensor(1.0))

    def forward(self, h):
        m = self.dense(h).squeeze(-1)
        v = self.rho.expand_as(m)
        return m, v
