"""
Test cases for DEARI imputation model.
"""

# Created by Ao Zhang <ao.zhang@kcl.ac.uk>
# License: BSD-3-Clause

import os.path
import unittest

import numpy as np
import pytest

from pypots.imputation.deari import DEARI
from pypots.optim import Adam
from pypots.utils.logging import logger
from pypots.nn.functional import calc_mse
from tests.global_test_config import (
    DATA,
    EPOCHS,
    DEVICE,
    TRAIN_SET,
    VAL_SET,
    TEST_SET,
    GENERAL_H5_TRAIN_SET_PATH,
    GENERAL_H5_VAL_SET_PATH,
    GENERAL_H5_TEST_SET_PATH,
    RESULT_SAVING_DIR_FOR_IMPUTATION,
    check_tb_and_model_checkpoints_existence,
)


class TestDEARI(unittest.TestCase):
    logger.info("Running tests for the DEARI imputation model...")

    saving_path = os.path.join(RESULT_SAVING_DIR_FOR_IMPUTATION, "DEARI")
    model_save_name = "saved_DEARI_model.pypots"

    optimizer = Adam(lr=0.001, weight_decay=1e-5)

    deari = DEARI(
        n_steps=DATA["n_steps"],
        n_features=DATA["n_features"],
        rnn_hidden_size=64,
        n_layers=2,
        imputation_weight=1.0,
        consistency_weight=0.1,
        bayesian=False,           # flip to True if blitz is available
        kl_weight=1e-4,
        epochs=EPOCHS,
        optimizer=optimizer,
        device=DEVICE,
        saving_path=saving_path,
        model_saving_strategy="best",
        verbose=True,
    )

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_0_fit(self):
        self.deari.fit(TRAIN_SET, VAL_SET)

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_1_predict_keys(self):
        results = self.deari.predict(TEST_SET)
        assert "imputation" in results
        assert "f_hidden_states" not in results
        assert "b_hidden_states" not in results
        assert "f_reconstruction" not in results
        assert "b_reconstruction" not in results

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_2_predict_details(self):
        results = self.deari.predict(TEST_SET, return_details=True)
        assert "imputation" in results
        assert "f_hidden_states" in results
        assert "b_hidden_states" in results
        assert "f_reconstruction" in results
        assert "b_reconstruction" in results
        assert results["imputation"].ndim == 3
        assert results["f_reconstruction"].shape == results["imputation"].shape
        assert results["b_reconstruction"].shape == results["imputation"].shape

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_3_impute(self):
        imputed_X = self.deari.impute(TEST_SET)
        assert not np.isnan(imputed_X).any(), "Output still has missing values after running impute()."
        test_MSE = calc_mse(imputed_X, DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        logger.info(f"DEARI test_MSE: {test_MSE}")

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_4_parameters(self):
        assert hasattr(self.deari, "model") and self.deari.model is not None
        assert hasattr(self.deari, "optimizer") and self.deari.optimizer is not None
        assert hasattr(self.deari, "best_loss")
        self.assertNotEqual(self.deari.best_loss, float("inf"))
        assert hasattr(self.deari, "best_model_dict") and self.deari.best_model_dict is not None

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_5_saving_path(self):
        assert os.path.exists(self.saving_path), f"file {self.saving_path} does not exist"
        check_tb_and_model_checkpoints_existence(self.deari)
        saved_model_path = os.path.join(self.saving_path, self.model_save_name)
        self.deari.save(saved_model_path)
        self.deari.load(saved_model_path)

    @pytest.mark.xdist_group(name="imputation-deari")
    def test_6_lazy_loading(self):
        # guard: skip if the general h5 dataset is not available in the environment
        if not (os.path.exists(GENERAL_H5_TRAIN_SET_PATH) and os.path.exists(GENERAL_H5_VAL_SET_PATH)):
            pytest.skip("General H5 dataset files are not available; skipping lazy-loading test.")

        # Fit on lazy-loaded data (when available)
        self.deari.fit(GENERAL_H5_TRAIN_SET_PATH, GENERAL_H5_VAL_SET_PATH)
        # Predict with lazy-loading
        results = self.deari.predict(GENERAL_H5_TEST_SET_PATH)
        assert not np.isnan(results["imputation"]).any(), "Output still has missing values after running impute()."

        test_MSE = calc_mse(results["imputation"], DATA["test_X_ori"], DATA["test_X_indicating_mask"])
        logger.info(f"Lazy-loading DEARI test_MSE: {test_MSE}")


if __name__ == "__main__":
    unittest.main()
