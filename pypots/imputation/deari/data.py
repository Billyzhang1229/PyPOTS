"""
Dataset for DEARI (bidirectional, forward/backward + deltas).
"""

# Created by <you>
# License: BSD-3-Clause

from typing import Union
import torch
from torch.utils.data import Dataset

from ...data.dataset.base import BaseDataset
from ...data.utils import parse_delta


class DatasetForDEARI(BaseDataset):
    """
    Wraps data for DEARI. Produces forward/backward streams and deltas.

    Parameters
    ----------
    data : dict or path
        Must contain 'X'. If return_X_ori=True, must also contain 'X_ori'.
    return_X_ori : bool
        Whether to return 'X_ori' and 'indicating_mask' (for validation).
    return_y : bool
        Should be False for pure imputation. Kept for API parity.
    file_type : str
        File type when `data` is a path (e.g., "hdf5").
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
    ):
        super().__init__(data, return_X_ori, return_X_pred=False, return_y=return_y, file_type=file_type)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # BaseDataset returns: [idx, X, missing_mask, (X_ori, indicating_mask if requested)]
        sample = super()._fetch_data_from_array(idx) if isinstance(self.data, dict) else super()._fetch_data_from_file(idx)
        # unpack
        ptr = 0
        indices = sample[ptr]; ptr += 1
        X = sample[ptr]; ptr += 1              # [T, F] or [F,T]? Base returns [T,F] per fill_and_get_mask_torch -> consistent
        missing_mask = sample[ptr]; ptr += 1
        X_ori, indicating_mask = None, None
        if self.return_X_ori:
            X_ori = sample[ptr]; ptr += 1
            indicating_mask = sample[ptr]; ptr += 1

        # Ensure shape [T, F]
        assert X.dim() == 2 and missing_mask.shape == X.shape, "X/mask must be [T, F] per BaseDataset."

        # Forward deltas
        deltas = parse_delta(missing_mask)  # [T, F]

        # Backward streams
        back_X = torch.flip(X, dims=[0])
        back_missing_mask = torch.flip(missing_mask, dims=[0])
        back_deltas = parse_delta(back_missing_mask)

        collated = [
            indices,
            X,                  # forward X [T,F]
            missing_mask,       # forward mask [T,F]
            deltas,             # forward deltas [T,F]
            back_X,             # backward X [T,F]
            back_missing_mask,  # backward mask [T,F]
            back_deltas,        # backward deltas [T,F]
        ]
        if self.return_X_ori:
            collated.extend([X_ori, indicating_mask])
        return collated
