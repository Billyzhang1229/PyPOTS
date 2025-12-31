"""
Core module assembling bidirectional DEARI.
"""

# Created by Ao Zhang <ao.zhang@kcl.ac.uk>
# License: BSD-3-Clause

from typing import Dict, Any

import torch
import torch.nn as nn

from ...nn.modules.loss import Criterion, MAE, MSE
from ...nn.modules.deari import BackboneBDEARI


class _BDEARI(nn.Module):
    """Core wrapper for bidirectional DEARI.

    Notes
    -----
    - When ``calc_criterion=True``, forward returns only lightweight scalars suitable for
      logging in the training loop: keys include ``loss`` (when computed),
      ``reconstruction_loss``, ``consistency_loss``, ``kl_loss``, and optionally ``metric``
      if ground-truth masks are provided. ``imputation`` is returned only when
      ``calc_criterion=False``.
    - When ``return_details=True`` and ``calc_criterion=False``, forward additionally
      returns heavy tensors such as per-direction reconstructions/hidden state sequences for analysis.
    """
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
        imputation_weight: float = 0.3,
        consistency_weight: float = 0.1,
        kl_weight: float = 1e-4,
        training_loss: Criterion = MAE(),
        validation_metric: Criterion = MSE(),
    ):
        super().__init__()
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.kl_weight = kl_weight
        self.validation_metric = validation_metric

        self.backbone = BackboneBDEARI(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            n_layers=n_layers,
            rnn_type=rnn_type,
            attention=attention,
            hidden_agg=hidden_agg,
            n_attn_heads=n_attn_heads,
            n_attn_layers=n_attn_layers,
            bayesian=bayesian,
            training_loss=training_loss,
        )

    def forward(
        self,
        inputs: Dict[str, Any],
        calc_criterion: bool = False,
        return_details: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        (
            imputed_data,
            f_recon,
            b_recon,
            f_hidden,
            b_hidden,
            consistency_loss,
            reconstruction_loss,
            kl_loss,
        ) = self.backbone(inputs)

        results: Dict[str, Any] = {}
        if calc_criterion:
            results.update(
                {
                    "consistency_loss": consistency_loss.detach(),
                    "reconstruction_loss": reconstruction_loss.detach(),
                    "kl_loss": (
                        kl_loss.detach()
                        if torch.is_tensor(kl_loss)
                        else torch.tensor(0.0, device=imputed_data.device)
                    ),
                }
            )
        else:
            results["imputation"] = imputed_data
            if return_details:
                results.update(
                    {
                        "f_reconstruction": f_recon,
                        "b_reconstruction": b_recon.flip(dims=[1]),
                        "f_hidden_states": f_hidden,
                        "b_hidden_states": b_hidden.flip(dims=[1]),
                    }
                )

        # compute and return training criterion if requested by the trainer
        if calc_criterion:
            total_loss = (
                self.imputation_weight * reconstruction_loss
                + self.consistency_weight * consistency_loss
            )
            if torch.is_tensor(kl_loss) and kl_loss.numel() > 0:
                total_loss = total_loss + self.kl_weight * kl_loss
            results["loss"] = total_loss

        # always compute validation metric if ground truth is provided
        if "X_ori" in inputs and "indicating_mask" in inputs:
            with torch.no_grad():
                if type(self.validation_metric) is Criterion:
                    metric_source = results.get("loss", reconstruction_loss)
                    results["metric"] = metric_source.detach()
                else:
                    X_ori = inputs["X_ori"]
                    indicating_mask = inputs["indicating_mask"]
                    metric = self.validation_metric(imputed_data, X_ori, indicating_mask)
                    results["metric"] = metric.detach() if torch.is_tensor(metric) else metric

        return results
