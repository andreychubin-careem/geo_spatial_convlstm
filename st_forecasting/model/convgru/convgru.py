import torch
import torch.nn as nn
from typing import Tuple

from .convgrucell import Conv2dGRUCell


class ConvGRUNet(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            input_channels: int = 1,
            kernel_size: Tuple[int, int] = (3, 3),
            cell: nn.Module = Conv2dGRUCell,
            bias: bool = True,
            init: str = 'zeros'
    ):
        """
        Encoder-Decoder ConvGRU model.
        One GRU_Block for Encoder and Decoder

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
            Class of Conv2dGRUCell. Only Conv2dGRUCell is available.
            default = Conv2dGRUCell
        :param bias: bool
            Whether to add bias (applies to all layers).
            default = True
        :param init: str
            Initialization for GRU.
            'zeros' for initialization with zeros or 'xavier_normal'|'glorot_normal' for xavier normal initialization.
            default = 'zeros'
        """
        super(ConvGRUNet, self).__init__()

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
            self, x: torch.Tensor, seq_len: int, horizon: int, h_x1: torch.Tensor, h_x2: torch.Tensor
    ) -> torch.Tensor:
        outputs = []

        # encoder
        for t in range(seq_len):
            h_x1 = self.encoder(
                input_tensor=x[:, t, :, :],
                cur_state=h_x1
            )

        encoder_vector = h_x1

        # decoder
        for t in range(horizon):
            h_x2 = self.decoder(
                input_tensor=encoder_vector,
                cur_state=h_x2
            )

            encoder_vector = h_x2
            outputs += [h_x2]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4).cpu()

        decoder_cnn = self.decoder_cnn.cpu()
        outputs = decoder_cnn(outputs.cpu()).to('mps')  # no support for Conv3d on MPS
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

    def forward(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        b, seq_len, _, h, w = x.size()

        h_x1 = self.encoder.init_hidden(batch_size=b, image_size=(h, w))
        h_x1 = h_x1.to(x.dtype)

        h_x2 = self.decoder.init_hidden(batch_size=b, image_size=(h, w))
        h_x2 = h_x2.to(x.dtype)

        outputs = self.autoencoder(x, seq_len, horizon, h_x1, h_x2)
        outputs = self.relu(outputs)

        return outputs
