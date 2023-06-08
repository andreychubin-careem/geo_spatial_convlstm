import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .convlstmcell import ConvLSTMCell


class ConvLSTMNet(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            input_channels: int = 1,
            kernel_size: Tuple[int, int] = (3, 3),
            cell: nn.Module = ConvLSTMCell,
            bias: bool = True,
            init: str = 'zeros'
    ):
        """
        Encoder-Decoder ConvLSTM model.
        One LSTM_Block for Encoder and Decoder

        Parameters
        ----------
        :param hidden_channels: int
            Number of output channels.
        :param input_channels: int
            Number of input channels (features).
            default = 1
        :param kernel_size: Tuple[int, int]
            Size of the convolutional kernel.
            default = (3, 3)
        :param cell: nn.Module
            Class of ConvLSTMCell. Either ConvLSTMCell or ResNetLSTMCell.
            default = ConvLSTMCell
        :param bias: bool
            Whether to add bias (applies to all layers).
            default = True
        :param init: str
            Initialization for LSTM.
            'zeros' for initialization with zeros or 'xavier_normal'|'glorot_normal' for xavier normal initialization.
            default = 'zeros'
        """

        super(ConvLSTMNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.bias = bias

        padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.encoder = cell(
            input_dim=input_channels,
            hidden_dim=self.hidden_channels,
            kernel_size=kernel_size,
            bias=self.bias,
            init=init
        )

        self.decoder = cell(
            input_dim=self.hidden_channels,
            hidden_dim=self.hidden_channels,
            kernel_size=kernel_size,
            bias=self.bias,
            init=init
        )

        self.decoder_cnn = nn.Conv3d(
            in_channels=self.hidden_channels,
            out_channels=1,
            kernel_size=(1, kernel_size[0], kernel_size[1]),
            padding=(0, padding[0], padding[1])
        )

        self.relu = nn.ReLU()

    def autoencoder(
            self, x: Tensor, seq_len: int, horizon: int, h_t1: Tensor, c_t1: Tensor, h_t2: Tensor, c_t2: Tensor
    ) -> Tensor:
        outputs = []

        # encoder
        for t in range(seq_len):
            h_t1, c_t1 = self.encoder(
                input_tensor=x[:, t, :, :],
                cur_state=[h_t1, c_t1]
            )

        encoder_vector = h_t1

        # decoder
        for t in range(horizon):
            h_t2, c_t2 = self.decoder(
                input_tensor=encoder_vector,
                cur_state=[h_t2, c_t2]
            )

            encoder_vector = h_t2
            outputs += [h_t2]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4).cpu()

        decoder_cnn = self.decoder_cnn.cpu()
        outputs = decoder_cnn(outputs.cpu()).to('mps')  # no support for Conv3d on MPS
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

    def forward(self, x: Tensor, horizon: int) -> Tensor:
        b, seq_len, _, h, w = x.size()

        h_t1, c_t1 = self.encoder.init_hidden(batch_size=b, image_size=(h, w))
        h_t1, c_t1 = h_t1.to(x.dtype), c_t1.to(x.dtype)

        h_t2, c_t2 = self.decoder.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = h_t2.to(x.dtype), c_t2.to(x.dtype)

        outputs = self.autoencoder(x, seq_len, horizon, h_t1, c_t1, h_t2, c_t2)
        outputs = self.relu(outputs)

        return outputs


class ConvLSTMNetStacked(nn.Module):
    def __init__(
            self,
            hidden_channels: Tuple[int, int],
            input_channels: int = 1,
            cell: nn.Module = ConvLSTMCell,
            bias: bool = True,
            init: str = 'zeros'
    ):
        """
        Encoder-Decoder ConvLSTM model.
        Two stacked LSTM_Blocks for Encoder and Decoder.

        Parameters
        ----------
        :param hidden_channels: int
            Number of output channels.
        :param input_channels: int
            Number of input channels (features).
            default = 1
        :param kernel_size: Tuple[int, int]
            Size of the convolutional kernel.
            default = (3, 3)
        :param cell: nn.Module
            Class of ConvLSTMCell. Either ConvLSTMCell or ResNetLSTMCell.
            default = ConvLSTMCell
        :param bias: bool
            Whether to include bias (applies to all layers).
            default = True
        :param init: str
            Initialization for LSTM.
            'zeros' for initialization with zeros or 'xavier_normal'|'glorot_normal' for xavier normal initialization.
            default = 'zeros'
        """
        super(ConvLSTMNetStacked, self).__init__()

        self.hidden_channels = hidden_channels
        self.bias = bias

        self.encoder_1 = cell(
            input_dim=input_channels,
            hidden_dim=self.hidden_channels[0],
            kernel_size=(3, 3),
            bias=self.bias,
            init=init
        )

        self.encoder_2 = cell(
            input_dim=self.hidden_channels[0],
            hidden_dim=self.hidden_channels[1],
            kernel_size=(3, 3),
            bias=self.bias,
            init=init
        )

        self.decoder_1 = cell(
            input_dim=self.hidden_channels[1],
            hidden_dim=self.hidden_channels[0],
            kernel_size=(3, 3),
            bias=self.bias,
            init=init
        )

        self.decoder_2 = cell(
            input_dim=self.hidden_channels[0],
            hidden_dim=self.hidden_channels[0],
            kernel_size=(3, 3),
            bias=self.bias,
            init=init
        )

        self.decoder_cnn = nn.Conv3d(
            in_channels=self.hidden_channels[0],
            out_channels=1,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        self.relu = nn.ReLU()

    def autoencoder(
            self,
            x: Tensor,
            seq_len: int,
            horizon: int,
            h_t1: Tensor,
            c_t1: Tensor,
            h_t2: Tensor,
            c_t2: Tensor,
            h_t3: Tensor,
            c_t3: Tensor,
            h_t4: Tensor,
            c_t4: Tensor
    ) -> Tensor:
        outputs = []

        # encoder
        for t in range(seq_len):
            h_t1, c_t1 = self.encoder_1(
                input_tensor=x[:, t, :, :],
                cur_state=[h_t1, c_t1]
            )

            h_t2, c_t2 = self.encoder_2(
                input_tensor=h_t1,
                cur_state=[h_t2, c_t2]
            )

        encoder_vector = h_t2

        # decoder
        for t in range(horizon):
            h_t3, c_t3 = self.decoder_1(
                input_tensor=encoder_vector,
                cur_state=[h_t3, c_t3]
            )

            h_t4, c_t4 = self.decoder_2(
                input_tensor=h_t3,
                cur_state=[h_t4, c_t4]
            )

            encoder_vector = h_t4
            outputs += [h_t4]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4).cpu()

        decoder_cnn = self.decoder_cnn.cpu()
        outputs = decoder_cnn(outputs).to('mps')  # no support for Conv3d on MPS
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

    def forward(self, x: Tensor, horizon: int) -> Tensor:
        b, seq_len, _, h, w = x.size()

        h_t1, c_t1 = self.encoder_1.init_hidden(batch_size=b, image_size=(h, w))
        h_t1, c_t1 = h_t1.to(x.dtype), c_t1.to(x.dtype)
        
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = h_t2.to(x.dtype), c_t2.to(x.dtype)
        
        h_t3, c_t3 = self.decoder_1.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = h_t3.to(x.dtype), c_t3.to(x.dtype)
        
        h_t4, c_t4 = self.decoder_2.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = h_t4.to(x.dtype), c_t4.to(x.dtype)

        outputs = self.autoencoder(x, seq_len, horizon, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        outputs = self.relu(outputs)

        return outputs
