import torch
import torch.nn as nn
from torch import Tensor
from typing import Iterable, Union, Optional

from .convlstmcell import ConvTTLSTMCell


class ConvTTLSTMNet(nn.Module):
    def __init__(
            self,
            input_channels: int,
            layers_per_block: Union[int, Iterable[int]],
            hidden_channels: Union[int, Iterable[int]],
            skip_stride: Optional[int] = None,
            cell_params: Optional[dict] = None,
            kernel_size: int = 3,
            bias: bool = True,
            teacher_forcing: bool = False,
            scheduled_sampling_ratio: int = 0
    ):
        """
        Initialization of a Conv-LSTM network.

        Arguments:
        ----------
        (Hyper-parameters of input interface)
        input_channels: int
            The number of channels for input video.
            Note: 3 for colored video, 1 for gray video.
        (Hyper-parameters of model architecture)
        layers_per_block: Union[int, Iterable[int]]
            Number of Conv-LSTM layers in each block.
        hidden_channels: Union[int, Iterable[int]]
            Number of output channels.
        Note: The length of hidden_channels (or layers_per_block) is equal to number of blocks.
        skip_stride: int
            The stride (in term of blocks) of the skip connections
            default: None, i.e. no skip connection

        cell_params: dictionary
            order: int
                The recurrent order of convolutional tensor-train cells.
                default: 3
            steps: int
                The number of previous steps used in the recurrent cells.
                default: 5
            rank: int
                The tensor-train rank of convolutional tensor-train cells.
                default: 16

        (Parameters of convolutional operations)
        kernel_size: int or (int, int)
            Size of the (squared) convolutional kernel.
            default: 3
        bias: bool
            Whether to add bias in the convolutional operation.
            default: True

        (Parameters for training)
        teacher_forcing: bool
            Whether the model is trained in teacher_forcing mode.
            Note 1: In test mode, teacher_forcing should be set as False.
            Note 2: If teacher_forcing mode is on,  # of frames in inputs = total_steps
                    If teacher_forcing mode is off, # of frames in inputs = input_frames
        scheduled_sampling_ratio: float between [0, 1]
            The ratio of ground-truth frames used in teacher_forcing mode.
            default: 0 (i.e. no teacher forcing effectively)
        """
        super(ConvTTLSTMNet, self).__init__()

        # Hyperparameters
        if cell_params is None:
            cell_params = {'order': 3, 'steps': 5, 'ranks': 8}
        self.layers_per_block = layers_per_block if isinstance(layers_per_block, Iterable) else [layers_per_block]
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, Iterable) else [hidden_channels]
        self.teacher_forcing = teacher_forcing
        self.scheduled_sampling_ratio = scheduled_sampling_ratio

        self.num_blocks = len(self.layers_per_block)
        assert self.num_blocks == len(self.hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        # Module type of convolutional LSTM layers
        cell = lambda in_channels, out_channels: ConvTTLSTMCell(
            input_channels=in_channels,
            hidden_channels=out_channels,
            order=cell_params["order"],
            steps=cell_params["steps"],
            ranks=cell_params["ranks"],
            kernel_size=(kernel_size, kernel_size),
            bias=bias
        )

        # Construction of convolutional tensor-train LSTM network

        # stack the convolutional-LSTM layers with skip connections
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for i in range(self.layers_per_block[b]):
                # number of input channels to the current layer
                if i > 0:
                    channels = self.hidden_channels[b]
                elif b == 0:  # if l == 0 and b == 0:
                    channels = input_channels
                else:  # if l == 0 and b > 0:
                    channels = self.hidden_channels[b - 1]
                    if b > self.skip_stride:
                        channels += self.hidden_channels[b - 1 - self.skip_stride]

                lid = "b{}l{}".format(b, i)  # layer ID
                self.layers[lid] = cell(channels, self.hidden_channels[b])

        # number of input channels to the last layer (output layer)
        channels = self.hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += self.hidden_channels[-1 - self.skip_stride]

        self.layers["output"] = nn.Conv2d(
            channels,
            input_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )

        self.relu = nn.ReLU()

    def autoencoder(self, inputs: Tensor, seq_len: int, horizon: int) -> Tensor:
        # TODO: include teacher_forcing support
        """
        Computation of Convolutional LSTM network.

        Arguments:
        ----------
        inputs: a 5-th order tensor of size [batch_size, input_frames, input_channels, height, width]
            Input tensor (video) to the deep Conv-LSTM network.
        seq_len: int
            The number of input frames to the model.
        horizon: int
            The number of future frames predicted by the model.

        Returns:
        --------
        outputs: a 5-th order tensor of size [batch_size, output_frames, hidden_channels, height, width]
            Output frames of the convolutional-LSTM module.
        """

        # compute the teacher forcing mask
        if self.teacher_forcing and self.scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(
                self.scheduled_sampling_ratio * torch.ones(inputs.size(0), horizon - 1, 1, 1, 1, device=inputs.device)
            )
        else:  # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            self.teacher_forcing = False
            # dummy
            teacher_forcing_mask = torch.bernoulli(
                torch.ones(inputs.size(0), horizon - 1, 1, 1, 1, device=inputs.device)
            )

        # the number of time steps in the computational graph
        total_steps = seq_len + horizon - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            if t < seq_len:
                input_ = inputs[:, t]
            elif not self.teacher_forcing:
                input_ = outputs[t - 1]
            else:  # if t >= seq_len and teacher_forcing:
                mask = teacher_forcing_mask[:, t - seq_len]
                input_ = inputs[:, t] * mask + outputs[t - 1] * (1 - mask)

            queue = []  # previous outputs for skip connection
            for b in range(self.num_blocks):
                for i in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, i)  # layer ID
                    input_ = self.layers[lid](input_, first_step=(t == 0))

                queue.append(input_)

                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim=1)  # concat over the channels

            # map the hidden states to predictive frames
            outputs[t] = self.layers["output"](input_)

        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        outputs = torch.stack([outputs[t] for t in range(horizon)], dim=1)

        return outputs

    def forward(self, inputs: Tensor, horizon: int) -> Tensor:
        b, seq_len, _, h, w = inputs.size()

        if self.teacher_forcing:
            seq_len = seq_len - horizon

        outputs = self.autoencoder(inputs, seq_len, horizon)
        outputs = self.relu(outputs)

        return outputs

    def set_teacher_forcing(self, value: bool) -> None:
        self.teacher_forcing = value
