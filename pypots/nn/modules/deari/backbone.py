"""
Backbone for DEARI (and its bidirectional variant) in PyPOTS style.
"""

# Created by <you>
# License: BSD-3-Clause

import math
import torch
import torch.nn as nn

from .layers import FeatureRegression, Decay, BayesianGRUCell
from ..loss import Criterion, MAE


class BackboneDEARI(nn.Module):
    """
    A DEARI-style imputation backbone (single direction).

    Parameters
    ----------
    n_steps : int
        Sequence length (time steps)
    n_features : int
        Number of features
    rnn_hidden_size : int
        Hidden size of the GRU cell
    bayesian : bool
        If True, use BayesianGRUCell (requires blitz). Otherwise nn.GRUCell.
    training_loss : Criterion
        Training loss defined on reconstructed values under observation mask.
    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        bayesian: bool = False,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()

        self.n_steps = n_steps
        self.input_size = n_features
        self.hidden_size = rnn_hidden_size
        self.training_loss = training_loss
        self.bayesian = bayesian

        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag=False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag=True)

        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        if bayesian:
            self.gru = BayesianGRUCell(self.input_size * 2, self.hidden_size)
        else:
            self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, deltas: torch.Tensor, h: torch.Tensor = None):
        """
        Parameters
        ----------
        x : [B, T, F]
        mask : [B, T, F]  (1=observed, 0=missing)
        deltas : [B, T, F]
        h : [B, H] (optional)

        Returns
        -------
        x_imp : [B, T, F]
        reconstruction : [B, T, F]  (the x_comb estimates before mixing with observed x)
        h : [B, H]  final hidden
        x_loss : scalar  accumulated reconstruction loss
        kl_loss : scalar  Bayesian KL term (0 if non-bayesian)
        """
        B, T, F = x.shape
        if h is None:
            h = torch.zeros(B, self.hidden_size, device=x.device)

        x_loss = torch.zeros(1, device=x.device)
        kl_accum = torch.zeros(1, device=x.device)
        x_imp = x.clone()
        reconstruction = []

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
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            # Loss on observed positions
            x_loss = x_loss + self.training_loss(x_comb_t, x_t, m_t)

            # Final imputation
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)
            # RNN input: imputed values + mask
            rnn_in = torch.cat([x_imp[:, t, :], m_t], dim=1)

            # Step GRU
            h = self.gru(rnn_in, h)

            # Bayesian KL accumulation if any
            if self.bayesian and hasattr(self.gru, "kl_loss"):
                kl_accum = kl_accum + self.gru.kl_loss()

            reconstruction.append(x_comb_t.unsqueeze(1))

        reconstruction = torch.cat(reconstruction, dim=1)
        # normalize KL by time steps for stable scaling across sequence lengths
        kl_accum = kl_accum / x.size(1)
        return x_imp, reconstruction, h, x_loss.squeeze(), kl_accum.squeeze()


class BackboneBDEARI(nn.Module):
    """
    Bidirectional DEARI assembling forward & backward backbones.
    """
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        bayesian: bool = False,
        training_loss: Criterion = MAE(),
    ):
        super().__init__()
        self.model_f = BackboneDEARI(n_steps, n_features, rnn_hidden_size, bayesian, training_loss)
        self.model_b = BackboneDEARI(n_steps, n_features, rnn_hidden_size, bayesian, training_loss)

    def forward(self, xdata: dict):
        # forward streams
        x = xdata["forward"]["X"]
        m = xdata["forward"]["missing_mask"]
        d_f = xdata["forward"]["deltas"]

        # backward streams
        x_b = xdata["backward"]["X"]
        m_b = xdata["backward"]["missing_mask"]
        d_b = xdata["backward"]["deltas"]

        # Forward direction
        f_imputed, f_recon, f_hidden, f_loss, f_kl = self.model_f(x, m, d_f)

        # Backward direction
        b_imputed, b_recon, b_hidden, b_loss, b_kl = self.model_b(x_b, m_b, d_b)

        # Average imputation
        x_imp = (f_imputed + b_imputed.flip(dims=[1])) / 2
        imputed_data = (x * m) + ((1 - m) * x_imp)

        # Consistency loss
        consistency_loss = torch.abs(f_imputed - b_imputed.flip(dims=[1])).mean() * 1e-1

        # Reconstruction loss (sum of both directions)
        reconstruction_loss = f_loss + b_loss
        kl_loss = f_kl + b_kl

        return (
            imputed_data,
            f_recon,
            b_recon,
            f_hidden,
            b_hidden,
            consistency_loss,
            reconstruction_loss,
            kl_loss,
        )
