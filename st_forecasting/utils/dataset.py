import torch
import numpy as np
from typing import Union, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from .functions import to_timeseries


class GeoFramesDataset(Dataset):
    def __init__(
            self,
            x: np.ndarray,
            indexes: Union[list, np.ndarray],
            memory: int,
            horizon: int,
            train: bool = True,
            scaler: StandardScaler = None,
            masked: bool = True
    ):
        """
        PyTorch Dataset with spatio-temporal snapshots.

        :param x: Input array with all spatio-temporal snapshots.
        :param indexes: Indexes from input array to use
        :param n_steps_past: Number of past steps to use for forecasting
        :param horizon: Forecasting horizon
        :param train: whether it is a training or testing dataset
        :param half: whether to use half precision (not supported on MPS)
        :param scaler: Object of fitted scaler class (applied to past steps sequence only)
        """
        self.data = x
        self.indexes = indexes
        self.memory = memory
        self.horizon = horizon
        self.train = train
        self.scaler = scaler
        self.masked = masked

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        x, y = to_timeseries(self.indexes[index], self.data, self.memory, self.horizon)

        assert x.sum() != 0.0, f'zero-sum-sequence appeared as x at index {index}'
        assert y.sum() != 0.0, f'zero-sum-sequence appeared as y at index {index}'

        size = x.shape

        if self.scaler is not None:
            x = self.scaler.transform(x.flatten().reshape(-1, 1)).reshape(size)

        x = torch.from_numpy(x).to(torch.float32)
        mask = torch.sum(x, dim=0)

        if self.masked:
            eta = 1e-8
            mask = torch.where(mask > 0.0, 1.0, eta).to(x.dtype)
        else:
            mask = torch.ones_like(mask)

        if self.train:
            y = torch.from_numpy(y)
            
            if len(y.shape) == 3:
                y = y.unsqueeze(dim=0)
            assert len(x.shape) == len(y.shape), 'Number of dimensions in x and y do not match!'

            y = y.to(torch.float32)
            
            return x, mask, y

        else:
            return x, mask
