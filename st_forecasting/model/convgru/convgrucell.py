import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from typing import Tuple


class Conv2dGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool, init: str = 'zeros'):
        """
        ConvGRU cell.

        Parameters
        ----------
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether to add the bias.
        :param init: str
            Initialization for LSTM.
            'zeros' for initialization with zeros or 'xavier_normal'|'glorot_normal' for xavier normal initialization.
            default = 'zeros'
        """

        super(Conv2dGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.bias = bias
        self.init = init

        self.x2h = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim * 3,
            kernel_size=kernel_size,
            padding=padding,
            bias=self.bias
        )

        self.h2h = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim * 3,
            kernel_size=kernel_size,
            padding=padding,
            bias=self.bias
        )

        self.reset_parameters()  # Not sure that this is necessary

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input_tensor: Tensor, cur_state: Tensor) -> Tensor:
        # Inputs:
        #       input: of shape (batch_size, input_size, height_size, width_size)
        #       hx: of shape (batch_size, hidden_size, height_size, width_size)
        # Outputs:
        #       hy: of shape (batch_size, hidden_size, height_size, width_size)

        x_t = self.x2h(input_tensor)
        h_t = self.h2h(cur_state)

        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * cur_state + (1 - update_gate) * new_gate

        return hy

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> Tensor:
        height, width = image_size

        if self.init == 'zeros':
            return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.x2h.weight.device)
        elif self.init == 'xavier_normal' or self.init == 'glorot_normal':
            size = torch.empty((batch_size, self.hidden_dim, height, width))
            return torch.nn.init.xavier_normal_(size, gain=1.0).to(self.x2h.weight.device)
        else:
            raise NotImplementedError(f'{self.init} is not implemented')
