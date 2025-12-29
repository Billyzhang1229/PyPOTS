"""
The PyTorch implementation of DEARI for imputation in PyPOTS.
"""

# Created by Ao Zhang <ao.zhang@kcl.ac.uk>
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
    """The PyTorch implementation of the DEARI model for time-series imputation.

    DEARI (Deep Bidirectional RNN with Attention for Imputation) is a deep learning model that
    uses bidirectional recurrent neural networks with attention mechanisms to impute missing values
    in time-series data. It processes the data in both forward and backward directions and uses
    multi-layer stacking with attention-based hidden state aggregation.

    Parameters
    ----------
    n_steps : int
        The number of time steps in the time-series data sample.

    n_features : int
        The number of features in the time-series data sample.

    rnn_hidden_size : int
        The size of the RNN hidden state, also the number of hidden units in the RNN cell.

    n_layers : int, default=8
        Number of stacked DEARI layers. Each layer processes the output from the previous layer.

    rnn_type : str, default="lstm"
        Type of recurrent cell to use. Must be either "gru" or "lstm".

    attention : bool, default=True
        Whether to use attention-based hidden state aggregation between layers.
        When False, no attention mechanism is applied.

    hidden_agg : str, default="cls"
        Strategy for aggregating hidden states when attention is enabled.
        Must be one of {"cls", "last", "mean"}:
        - "cls": Use a class token with transformer encoder
        - "last": Use the last hidden state from the sequence
        - "mean": Use the mean of all hidden states

    n_attn_heads : int, default=4
        Number of attention heads in the transformer encoder.
        Only used when attention=True.

    n_attn_layers : int, default=2
        Number of transformer encoder layers for attention mechanism.
        Only used when attention=True.

    imputation_weight : float, default=0.3
        Weight for the reconstruction loss in the total loss computation.

    consistency_weight : float, default=0.1
        Weight for the consistency loss between forward and backward directions.

    bayesian : bool, default=False
        If True, use Bayesian recurrent cells which provide uncertainty estimates.
        Requires the ``blitz-bayesian-pytorch`` package to be installed.
        Otherwise uses standard GRU/LSTM cells based on rnn_type.

    kl_weight : float, default=1e-4
        Weight for KL divergence regularization when bayesian=True.
        Ignored when bayesian=False.

    batch_size : int, default=32
        The batch size for training and evaluating the model.

    epochs : int, default=100
        The number of epochs for training the model.

    patience : int, optional
        The patience for the early-stopping mechanism. Given a positive integer, the training process will be
        stopped when the model does not perform better after that number of epochs.
        Leaving it default as None will disable the early-stopping.

    training_loss : Criterion or type, default=MAE
        The loss function for training the model.
        If not given, will use Mean Absolute Error (MAE).

    validation_metric : Criterion or type, default=MSE
        The metric function for validating the model.
        If not given, will use Mean Squared Error (MSE).

    optimizer : Optimizer or type, default=Adam
        The optimizer for model training.
        If not given, will use the Adam optimizer.

    num_workers : int, default=0
        The number of subprocesses to use for data loading.
        `0` means data loading will be in the main process, i.e. there won't be subprocesses.

    device : str or torch.device or list, optional
        The device for the model to run on. It can be a string, a :class:`torch.device` object, or a list of them.
        If not given, will try to use CUDA devices first (will use the default CUDA device if there are multiple),
        then CPUs, considering CUDA and CPU are so far the main devices for people to train ML models.
        If given a list of devices, e.g. ['cuda:0', 'cuda:1'], or [torch.device('cuda:0'), torch.device('cuda:1')],
        the model will be parallely trained on the multiple devices (so far only support parallel training on CUDA devices).
        Other devices like Google TPU and Apple Silicon accelerator MPS may be added in the future.

    saving_path : str, optional
        The path for automatically saving model checkpoints and tensorboard files (i.e. loss values recorded during
        training into a tensorboard file). Will not save if not given.

    model_saving_strategy : str, default="best"
        The strategy to save model checkpoints. It has to be one of [None, "best", "better", "all"].
        No model will be saved when it is set as None.
        The "best" strategy will only automatically save the best model after the training finished.
        The "better" strategy will automatically save the model during training whenever the model performs
        better than in previous epochs.
        The "all" strategy will save every model after each epoch training.

    verbose : bool, default=True
        Whether to print out the training logs during the training process.

    Notes
    -----
    When using bayesian=True, ensure that the ``blitz-bayesian-pytorch`` package is installed:
    ``pip install blitz-bayesian-pytorch``
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
    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        return_details: bool = False,
    ) -> dict:
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
            results = self.model(inputs, calc_criterion=False, return_details=return_details)
            dict_result_collector.append(results)
        result_dict = gather_listed_dicts(dict_result_collector)
        return result_dict

    def impute(self, test_set: Union[dict, str], file_type: str = "hdf5") -> np.ndarray:
        return self.predict(test_set, file_type=file_type)["imputation"]
