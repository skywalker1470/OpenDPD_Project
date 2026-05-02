"""
my_dpd_backbone.py  —  Drop this into backbones/

Two custom models that plug directly into the dpdOpen framework:
  - MyMLPDPD   : feedforward MLP with residual connection
  - MyLSTMDPD  : bidirectional LSTM with attention

Usage in dpdOpen:
    opendpd.train_dpd(
        dataset_name='DPA_200MHz',
        DPD_backbone='my_mlp_dpd',     # or 'my_lstm_dpd'
        DPD_hidden_size=64,
        PA_backbone='gru',
        PA_hidden_size=23,
        n_epochs=100,
        ...
    )

Registration:
    Add to backbones/__init__.py:
        from .my_dpd_backbone import MyMLPDPD, MyLSTMDPD

    Add to models.py backbone_map dict:
        'my_mlp_dpd':  MyMLPDPD,
        'my_lstm_dpd': MyLSTMDPD,
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared convention (mirrors every other backbone in this repo):
#   forward(x)  where x : (batch, seq_len, 2)   — real [I, Q] pairs
#   returns     y : (batch, 2)                  — pre-distorted [I, Q]
# ─────────────────────────────────────────────────────────────────────────────


class _ResBlock(nn.Module):
    """Two-layer residual block used inside MyMLPDPD."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class MyMLPDPD(nn.Module):
    """
    Feedforward DPD with residual blocks.

    Architecture:
        flatten(seq_len * 2)  →  project to hidden_size
        → N x ResBlock(hidden_size)
        → Linear(hidden_size, 2)

    Args:
        input_size  : always 2  (I and Q)
        hidden_size : width of each hidden layer  (DPD_hidden_size)
        seq_len     : memory depth — how many past samples are visible
        n_blocks    : number of residual blocks (default 3)
    """
    def __init__(self, input_size=2, hidden_size=64, seq_len=10, n_blocks=3):
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Sequential(
            nn.Linear(input_size * seq_len, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(*[_ResBlock(hidden_size) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x : (B, seq_len, 2)
        B = x.size(0)
        out = self.input_proj(x.reshape(B, -1))   # (B, hidden_size)
        out = self.res_blocks(out)                 # (B, hidden_size)
        return self.output_proj(out)               # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────

class _ScaledDotAttention(nn.Module):
    """Single-head scaled dot-product attention over time steps."""
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, h):
        # h : (B, T, dim)
        Q = self.q(h)
        K = self.k(h)
        V = self.v(h)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # (B, T, T)
        attn   = torch.softmax(scores, dim=-1)
        out    = torch.bmm(attn, V)                              # (B, T, dim)
        return out[:, -1, :]                                     # last timestep


class MyLSTMDPD(nn.Module):
    """
    Bidirectional LSTM DPD with single-head attention.

    Architecture:
        BiLSTM(input_size=2, hidden_size, num_layers=2)
        → attention over time steps
        → LayerNorm → Linear(hidden_size*2, 2)

    Args:
        input_size  : always 2  (I and Q)
        hidden_size : LSTM hidden units per direction  (DPD_hidden_size)
        num_layers  : stacked LSTM layers (default 2)
        dropout     : dropout between LSTM layers (default 0.1)
    """
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )
        attn_dim = hidden_size * 2          # bidirectional doubles the dim
        self.attn = _ScaledDotAttention(attn_dim)
        self.norm = nn.LayerNorm(attn_dim)
        self.head = nn.Linear(attn_dim, input_size)

    def forward(self, x):
        # x : (B, seq_len, 2)
        h, _ = self.lstm(x)                 # (B, seq_len, hidden*2)
        ctx  = self.attn(h)                 # (B, hidden*2)
        ctx  = self.norm(ctx)
        return self.head(ctx)               # (B, 2)
