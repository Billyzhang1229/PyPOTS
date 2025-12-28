"""
The PyTorch implementation of DEARI for imputation in PyPOTS.
"""

# Created by <you>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .core import _BDEARI
from .data import DatasetForDEARI
from ..base import BaseNNImputer
from ...data.checking import key_in_data_set
from ...nn.functional import gather_listed_dicts
from ...nn.modules.loss import Criterion, MAE, MSE
from ...optim.adam import Adam
from ...optim.base import Optimizer
from ...utils.logging import logger


class DEARI(BaseNNImputer):
    """
    PyPOTS integration of DEARI (bidirectional, optional Bayesian GRU).

    Parameters
    ----------
    n_steps : int
    n_features : int
    rnn_hidden_size : int
    n_layers : int
        Number of stacked DEARI layers.
    rnn_type : str
        "gru" or "lstm".
    attention : bool
        Whether to use attention-based hidden aggregation between layers.
    hidden_agg : str
        One of {"cls", "last", "mean"} for attention aggregation.
    n_attn_heads : int
        Number of attention heads in the transformer encoder.
    n_attn_layers : int
        Number of transformer encoder layers.
    imputation_weight : float
    consistency_weight : float
    bayesian : bool
        If True, use Bayesian recurrent cells (requires package ``blitz-bayesian-pytorch``).
        Otherwise uses standard GRU/LSTM cells based on rnn_type.
    kl_weight : float
        Weight for KL regularization when bayesian=True.
    batch_size, epochs, patience, training_loss, validation_metric, optimizer,
    num_workers, device, saving_path, model_saving_strategy, verbose
        Same semantics as other PyPOTS models (e.g., BRITS/CSAI).
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
        imputation_weight: float = 0.3,
        consistency_weight: float = 0.1,
        bayesian: bool = False,
        kl_weight: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 100,
        patience: Optional[int] = None,
        training_loss: Union[Criterion, type] = MAE,
        validation_metric: Union[Criterion, type] = MSE,
        optimizer: Union[Optimizer, type] = Adam,
        num_workers: int = 0,
        device: Optional[Union[str, torch.device, list]] = None,
        saving_path: str = None,
        model_saving_strategy: Optional[str] = "best",
        verbose: bool = True,
    ):
        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            training_loss=training_loss,
            validation_metric=validation_metric,
            num_workers=num_workers,
            device=device,
            saving_path=saving_path,
            model_saving_strategy=model_saving_strategy,
            verbose=verbose,
        )

        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.attention = attention
        self.hidden_agg = hidden_agg
        self.n_attn_heads = n_attn_heads
        self.n_attn_layers = n_attn_layers
        self.imputation_weight = imputation_weight
        self.consistency_weight = consistency_weight
        self.bayesian = bayesian
        self.kl_weight = kl_weight
        # Optional: set to an integer to enable linear KL annealing over first N epochs
        # (no-op by default; provided for future extension without altering the base training loop)
        self._kl_anneal_steps: Optional[int] = None

        # set up the model
        self.model = _BDEARI(
            n_steps=self.n_steps,
            n_features=self.n_features,
            rnn_hidden_size=self.rnn_hidden_size,
            n_layers=self.n_layers,
            rnn_type=self.rnn_type,
            attention=self.attention,
            hidden_agg=self.hidden_agg,
            n_attn_heads=self.n_attn_heads,
            n_attn_layers=self.n_attn_layers,
            bayesian=self.bayesian,
            imputation_weight=self.imputation_weight,
            consistency_weight=self.consistency_weight,
            kl_weight=self.kl_weight,
            training_loss=self.training_loss,
            validation_metric=self.validation_metric,
        )
        self._send_model_to_given_device()
        self._print_model_size()
        logger.info(f"DEARI config - bayesian={self.bayesian}, kl_weight={self.kl_weight}")

        # set up optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer()
            assert isinstance(self.optimizer, Optimizer)
        self.optimizer.init_optimizer(self.model.parameters())

    # -------- assemble inputs like BRITS/CSAI wrappers
    def _assemble_input_for_training(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
        }
        return inputs

    def _assemble_input_for_validating(self, data: list) -> dict:
        (
            indices,
            X,
            missing_mask,
            deltas,
            back_X,
            back_missing_mask,
            back_deltas,
            X_ori,
            indicating_mask,
        ) = self._send_data_to_given_device(data)

        inputs = {
            "indices": indices,
            "forward": {
                "X": X,
                "missing_mask": missing_mask,
                "deltas": deltas,
            },
            "backward": {
                "X": back_X,
                "missing_mask": back_missing_mask,
                "deltas": back_deltas,
            },
            "X_ori": X_ori,
            "indicating_mask": indicating_mask,
        }
        return inputs

    def _assemble_input_for_testing(self, data: list) -> dict:
        # identical to training assembly for testing (no X_ori)
        return self._assemble_input_for_training(data)

    # -------- train / predict
    def fit(self, train_set: Union[dict, str], val_set: Optional[Union[dict, str]] = None, file_type: str = "hdf5") -> None:
        # datasets
        train_dataset = DatasetForDEARI(train_set, return_X_ori=False, return_y=False, file_type=file_type)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = None
        if val_set is not None:
            if not key_in_data_set("X_ori", val_set):
                raise ValueError("val_set must contain 'X_ori' for model validation.")
            val_set = DatasetForDEARI(val_set, return_X_ori=True, return_y=False, file_type=file_type)
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # train
        self._train_model(train_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    @torch.no_grad()
    def predict(self, test_set: Union[dict, str], file_type: str = "hdf5") -> dict:
        self.model.eval()
        test_dataset = DatasetForDEARI(test_set, return_X_ori=False, return_y=False, file_type=file_type)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        dict_result_collector = []
        for _, data in enumerate(test_loader):
            inputs = self._assemble_input_for_testing(data)
            results = self.model(inputs, calc_criterion=False)
            dict_result_collector.append(results)
        result_dict = gather_listed_dicts(dict_result_collector)
        # expose the conventional key used by BaseImputer.impute() and tests
        # (keep both for backward-compatibility)
        if "imputed_data" in result_dict:
            result_dict["imputation"] = result_dict["imputed_data"]
        return result_dict

    def impute(self, test_set: Union[dict, str], file_type: str = "hdf5") -> np.ndarray:
        return self.predict(test_set, file_type=file_type)["imputation"]
