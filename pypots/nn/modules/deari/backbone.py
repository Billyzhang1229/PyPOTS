"""
Backbone for DEARI (and its bidirectional variant) in PyPOTS style.
"""

# Created by Ao Zhang <ao.zhang@kcl.ac.uk>
# License: BSD-3-Clause

import math
import torch
import torch.nn as nn

from .layers import FeatureRegression, Decay, BayesianGRUCell, BayesianLSTMCell
from ..loss import Criterion, MAE


class BackboneDEARI(nn.Module):
    """
    A DEARI-style imputation backbone (single direction, one layer).

    Parameters
    ----------
    n_steps : int
        Sequence length (time steps)
    n_features : int
        Number of features
    rnn_hidden_size : int
        Hidden size of the recurrent cell
    rnn_type : str
        "gru" or "lstm"
    attention : bool
        Whether to use attention-based hidden aggregation when stacking layers.
    hidden_agg : str
        One of {"cls", "last", "mean"}.
    n_attn_heads : int
        Number of attention heads for the transformer encoder.
    n_attn_layers : int
        Number of transformer encoder layers.
    bayesian : bool
        If True, use Bayesian recurrent cells (requires blitz).
    training_loss : Criterion
        Training loss defined on reconstructed values under observation mask.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        rnn_type: str = "lstm",
        attention: bool = True,
        hidden_agg: str = "cls",
        n_attn_heads: int = 4,
        n_attn_layers: int = 2,
        bayesian: bool = False,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()

        self.n_steps = n_steps
        self.input_size = n_features
        self.hidden_size = rnn_hidden_size
        self.rnn_type = rnn_type.lower()
        self.attention = attention
        self.hidden_agg = hidden_agg
        self.training_loss = training_loss
        self.bayesian = bayesian

        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag=False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag=True)

        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        if self.attention:
            if self.hidden_size % n_attn_heads != 0:
                raise ValueError("rnn_hidden_size must be divisible by n_attn_heads for attention.")
            self.class_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size, nhead=n_attn_heads, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)
        elif self.hidden_agg == "cls":
            raise ValueError("hidden_agg='cls' requires attention=True.")

        if self.rnn_type == "gru":
            if bayesian:
                self.rnn = BayesianGRUCell(self.input_size * 2, self.hidden_size)
            else:
                self.rnn = nn.GRUCell(self.input_size * 2, self.hidden_size)
        elif self.rnn_type == "lstm":
            if bayesian:
                self.rnn = BayesianLSTMCell(self.input_size * 2, self.hidden_size)
            else:
                self.rnn = nn.LSTMCell(self.input_size * 2, self.hidden_size)
        else:
            raise ValueError("rnn_type must be one of {'gru', 'lstm'}.")

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def _aggregate_hidden(self, prev_hiddens: torch.Tensor) -> torch.Tensor:
        if prev_hiddens.dim() != 3:
            return prev_hiddens

        if self.hidden_agg == "cls":
            if not self.attention:
                raise ValueError("hidden_agg='cls' requires attention=True.")
            cls_token = self.class_token.expand(prev_hiddens.size(0), -1, -1)
            h_in = torch.cat([cls_token, prev_hiddens], dim=1)
            h_out = self.transformer_encoder(h_in)
            return h_out[:, 0, :]
        attended = self.transformer_encoder(prev_hiddens) if self.attention else prev_hiddens
        if self.hidden_agg == "last":
            return attended[:, -1, :]
        if self.hidden_agg == "mean":
            return torch.mean(attended, dim=1)
        raise ValueError("hidden_agg must be one of {'cls', 'last', 'mean'}.")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        deltas: torch.Tensor,
        h: torch.Tensor = None,
        c: torch.Tensor = None,
        return_hidden_sequence: bool = False,
    ):
        """
        Parameters
        ----------
        x : [B, T, F]
        mask : [B, T, F]  (1=observed, 0=missing)
        deltas : [B, T, F]
        h : [B, H] or [B, T, H] (optional)
        c : [B, H] (optional, for LSTM)
        return_hidden_sequence : bool
            If True, also return hidden states across time steps.

        Returns
        -------
        x_imp : [B, T, F]
        reconstruction : [B, T, F]  (the x_comb estimates before mixing with observed x)
        h : [B, H]  final hidden
        x_loss : scalar  accumulated reconstruction loss
        kl_loss : scalar  Bayesian KL term (0 if non-bayesian)
        hidden_seq : [B, T, H]  (optional, only if return_hidden_sequence=True)
        """
        B, T, _ = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)
        else:
            h = self._aggregate_hidden(h)

        if self.rnn_type == "lstm":
            if c is None:
                c = torch.zeros(B, self.hidden_size, device=x.device)

        x_loss = torch.zeros(1, device=x.device)
        kl_accum = torch.zeros(1, device=x.device)
        x_imp_list = []
        reconstruction = []
        hidden_seq = []

        for t in range(x.size(1)):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decay hidden
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h

            # History estimation
            x_h = self.hist(h)

            # Replace missing with history to stabilize feature regression
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # Feature regression with temporal decay
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)

            # Combine feature regression and history via learned weights
            beta = torch.sigmoid(self.weight_combine(torch.cat([gamma_x, m_t], dim=1)))
            x_comb_t = beta * xu + (1 - beta) * x_h

            # Loss on observed positions
            x_loss = x_loss + self.training_loss(x_comb_t, x_t, m_t)

            # Final imputation
            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_comb_t)
            # RNN input: imputed values + mask
            rnn_in = torch.cat([x_imp_t, m_t], dim=1)

            # Step recurrent cell
            if self.rnn_type == "gru":
                h = self.rnn(rnn_in, h)
            else:
                h, c = self.rnn(rnn_in, (h, c))

            reconstruction.append(x_comb_t.unsqueeze(1))
            hidden_seq.append(h.unsqueeze(1))
            x_imp_list.append(x_imp_t.unsqueeze(1))

        x_imp = torch.cat(x_imp_list, dim=1)
        reconstruction = torch.cat(reconstruction, dim=1)
        hidden_seq = torch.cat(hidden_seq, dim=1)
        if self.bayesian and hasattr(self.rnn, "kl_loss"):
            kl_accum = self.rnn.kl_loss()
        else:
            kl_accum = torch.tensor(0.0, device=x.device)

        if return_hidden_sequence:
            return x_imp, reconstruction, h, x_loss.squeeze(), kl_accum.squeeze(), hidden_seq
        return x_imp, reconstruction, h, x_loss.squeeze(), kl_accum.squeeze()


class BackboneBDEARI(nn.Module):
    """
    Bidirectional DEARI assembling forward & backward backbones.
    """
    # Weight for consistency loss computation between forward and backward directions
    CONSISTENCY_LOSS_WEIGHT = 0.1
    
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_layers: int = 8,
        rnn_type: str = "lstm",
        attention: bool = True,
        hidden_agg: str = "cls",
        n_attn_heads: int = 4,
        n_attn_layers: int = 2,
        bayesian: bool = False,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.model_f = nn.ModuleList(
            [
                BackboneDEARI(
                    n_steps,
                    n_features,
                    rnn_hidden_size,
                    rnn_type=rnn_type,
                    attention=attention,
                    hidden_agg=hidden_agg,
                    n_attn_heads=n_attn_heads,
                    n_attn_layers=n_attn_layers,
                    bayesian=bayesian,
                    training_loss=training_loss,
                )
                for _ in range(n_layers)
            ]
        )
        self.model_b = nn.ModuleList(
            [
                BackboneDEARI(
                    n_steps,
                    n_features,
                    rnn_hidden_size,
                    rnn_type=rnn_type,
                    attention=attention,
                    hidden_agg=hidden_agg,
                    n_attn_heads=n_attn_heads,
                    n_attn_layers=n_attn_layers,
                    bayesian=bayesian,
                    training_loss=training_loss,
                )
                for _ in range(n_layers)
            ]
        )
        self.bayesian = bayesian

    def forward(self, xdata: dict):
        # forward streams
        x = xdata["forward"]["X"]
        m = xdata["forward"]["missing_mask"]
        d_f = xdata["forward"]["deltas"]

        # backward streams
        m_b = xdata["backward"]["missing_mask"]
        d_b = xdata["backward"]["deltas"]

        x_imp_layers = []
        xreg_losses = []
        consistency_losses = []
        kl_losses = []
        f_hidden_layers = []
        b_hidden_layers = []
        f_recon = None
        b_recon = None
        f_hidden_last = None
        b_hidden_last = None

        for i in range(self.n_layers):
            prev_f_hiddens = None if i == 0 else f_hidden_layers[-1]
            prev_b_hiddens = None if i == 0 else b_hidden_layers[-1]
            x_f = x if i == 0 else x_imp_layers[-1]

            ret_f = self.model_f[i](
                x=x_f,
                mask=m,
                deltas=d_f,
                h=prev_f_hiddens,
                return_hidden_sequence=True,
            )
            f_imputed, f_recon, f_hidden_last, f_loss, f_kl, f_hidden_seq = ret_f

            x_b = x_f.flip(dims=[1])
            ret_b = self.model_b[i](
                x=x_b,
                mask=m_b,
                deltas=d_b,
                h=prev_b_hiddens,
                return_hidden_sequence=True,
            )
            b_imputed, b_recon, b_hidden_last, b_loss, b_kl, b_hidden_seq = ret_b

            imp = (f_imputed + b_imputed.flip(dims=[1])) / 2
            xreg_losses.append(f_loss + b_loss)
            consistency_losses.append(torch.abs(f_imputed - b_imputed.flip(dims=[1])).mean() * self.CONSISTENCY_LOSS_WEIGHT)

            x_imp_layers.append((x * m) + ((1 - m) * imp))
            f_hidden_layers.append(f_hidden_seq)
            b_hidden_layers.append(b_hidden_seq)
            kl_losses.append(f_kl + b_kl)

        x_imp = torch.mean(torch.stack(x_imp_layers, dim=1), dim=1)
        reconstruction_loss = torch.mean(torch.stack(xreg_losses, dim=0), dim=0)
        consistency_loss = torch.mean(torch.stack(consistency_losses, dim=0), dim=0)
        if self.bayesian:
            kl_loss = torch.sum(torch.stack(kl_losses, dim=0), dim=0)
        else:
            kl_loss = torch.tensor(0.0, device=x.device)

        return (
            x_imp,
            f_recon,
            b_recon,
            f_hidden_last,
            b_hidden_last,
            consistency_loss,
            reconstruction_loss,
            kl_loss,
        )
