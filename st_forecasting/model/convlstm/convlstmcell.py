import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool, init: str = 'zeros'):
        """
        ConvLSTM cell.

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

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.bias = bias
        self.init = init

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=self.bias
        )

    def forward(self, input_tensor: Tensor, cur_state: Tensor) -> (Tensor, Tensor):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> (Tensor, Tensor):
        height, width = image_size

        if self.init == 'zeros':
            return (
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            )
        elif self.init == 'xavier_normal' or self.init == 'glorot_normal':
            size = torch.empty((batch_size, self.hidden_dim, height, width))
            return (
                torch.nn.init.xavier_normal_(size, gain=1.0).to(self.conv.weight.device),
                torch.nn.init.xavier_normal_(size, gain=1.0).to(self.conv.weight.device)
            )
        else:
            raise NotImplementedError(f'{self.init} is not implemented')


class ResNetLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool, init: str = 'zeros'):
        """
        ResNetConvLSTM cell.

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

        super(ResNetLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.bias = bias
        self.init = init

        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=self.bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * self.hidden_dim)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=self.bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim),
            nn.Dropout2d(0.2)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=self.bias
            ),
            nn.ReLU(),
            nn.BatchNorm2d(4 * self.hidden_dim)
        )

    def resnet(self, x: Tensor) -> Tensor:
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(x2)

        return x3 + x1

    def forward(self, input_tensor: Tensor, cur_state: Tensor) -> (Tensor, Tensor):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.resnet(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> (Tensor, Tensor):
        height, width = image_size

        if self.init == 'zeros':
            return (
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.convblock1[0].weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.convblock1[0].weight.device)
            )
        elif self.init == 'xavier_normal' or self.init == 'glorot_normal':
            size = torch.empty((batch_size, self.hidden_dim, height, width))
            return (
                torch.nn.init.xavier_normal_(size, gain=1.0).to(self.convblock1[0].weight.device),
                torch.nn.init.xavier_normal_(size, gain=1.0).to(self.convblock1[0].weight.device)
            )
        else:
            raise NotImplementedError(f'{self.init} is not implemented')
