"""
Layers and cells used in DEARI.
"""

# Created by <you>
# License: BSD-3-Clause

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class FeatureRegression(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W * Variable(self.m), self.b)


class Decay(nn.Module):
    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))
        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        return torch.exp(-gamma)


# ---- Optional components: kept for parity with CSAI and potential extensions.
class TorchTransformerEncoder(nn.Module):
    def __init__(self, heads: int = 8, layers: int = 1, channels: int = 64):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x)


class Conv1dWithInit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ---- Bayesian GRU cell (for bayesian=True). If blitz is absent, raise a helpful error.
class BayesianGRUCell(nn.Module):
    """
    A minimal Bayesian GRUCell using blitz BayesianLinear layers.

    If `blitz` (blitz-bayesian-pytorch) is not installed, importing this cell will fail with
    ImportError. In that case, set bayesian=False or install `blitz-bayesian-pytorch`.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        try:
            from blitz.modules import BayesianLinear
        except Exception as e:
            raise ImportError(
                "Bayesian GRU requires `blitz-bayesian-pytorch`. "
                "Install with `pip install blitz-bayesian-pytorch`."
            ) from e

        self.hidden_size = hidden_size
        # x->h
        self.W_zx = BayesianLinear(input_size, hidden_size)
        self.W_rx = BayesianLinear(input_size, hidden_size)
        self.W_nx = BayesianLinear(input_size, hidden_size)
        # h->h
        self.W_zh = BayesianLinear(hidden_size, hidden_size)
        self.W_rh = BayesianLinear(hidden_size, hidden_size)
        self.W_nh = BayesianLinear(hidden_size, hidden_size)

        self._bayesian_layers = [
            self.W_zx, self.W_rx, self.W_nx, self.W_zh, self.W_rh, self.W_nh
        ]

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.W_zx(x) + self.W_zh(h_prev))
        r = torch.sigmoid(self.W_rx(x) + self.W_rh(h_prev))
        n = torch.tanh(self.W_nx(x) + self.W_nh(r * h_prev))
        h = (1 - z) * n + z * h_prev
        return h

    def kl_loss(self) -> torch.Tensor:
        # Aggregate KL from all BayesianLinear layers
        kl = 0.0
        for lyr in self._bayesian_layers:
            # blitz BayesianLinear exposes kl_loss() method
            if hasattr(lyr, "kl_loss"):
                kl = kl + lyr.kl_loss()
        return kl if isinstance(kl, torch.Tensor) else torch.tensor(kl)


class BayesianLSTMCell(nn.Module):
    """
    A minimal Bayesian LSTMCell using blitz BayesianLinear layers.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        try:
            from blitz.modules import BayesianLinear
        except Exception as e:
            raise ImportError(
                "Bayesian LSTM requires `blitz-bayesian-pytorch`. "
                "Install with `pip install blitz-bayesian-pytorch`."
            ) from e

        self.hidden_size = hidden_size
        self.W_ih = BayesianLinear(input_size, hidden_size * 4)
        self.W_hh = BayesianLinear(hidden_size, hidden_size * 4)
        self._bayesian_layers = [self.W_ih, self.W_hh]

    def forward(self, x: torch.Tensor, state):
        h_prev, c_prev = state
        gates = self.W_ih(x) + self.W_hh(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def kl_loss(self) -> torch.Tensor:
        kl = 0.0
        for lyr in self._bayesian_layers:
            if hasattr(lyr, "kl_loss"):
                kl = kl + lyr.kl_loss()
        return kl if isinstance(kl, torch.Tensor) else torch.tensor(kl)
