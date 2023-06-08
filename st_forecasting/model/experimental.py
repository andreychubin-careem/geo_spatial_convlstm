import torch
import torch.nn as nn

from .layers import ConvLSTMCell


class SqueezeNet(nn.Module):
    """
    Not working :-) Unsqueezing does not learns properly
    """
    def __init__(self, hidden_channels: int, n_steps_past: int, in_channels: int = 1):
        super(SqueezeNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.batch_norm = nn.BatchNorm3d(n_steps_past)

        self.encoder = ConvLSTMCell(
            input_dim=in_channels,
            hidden_dim=self.hidden_channels,
            kernel_size=(3, 3),
            bias=True
        )

        self.decoder = ConvLSTMCell(
            input_dim=self.hidden_channels,
            hidden_dim=self.hidden_channels,
            kernel_size=(3, 3),
            bias=True
        )

        self.decoder_cnn = nn.Conv3d(
            in_channels=self.hidden_channels,
            out_channels=1,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)).cpu()

    def autoencoder(
            self,
            x: torch.Tensor,
            seq_len: int,
            horizon: int,
            h_t1: torch.Tensor,
            c_t1: torch.Tensor,
            h_t2: torch.Tensor,
            c_t2: torch.Tensor
    ) -> torch.Tensor:
        outputs = []

        h_t1, _ = self.pool(h_t1)
        c_t1, _ = self.pool(c_t1)
        h_t2, _ = self.pool(h_t2)
        c_t2, _ = self.pool(c_t2)
        indices = None

        # encoder
        for t in range(seq_len):
            x_p, indices = self.pool(x[:, t, :, :])
            h_t1, c_t1 = self.encoder(
                input_tensor=x_p,
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
        outputs = decoder_cnn(outputs)  # no support for Conv3d on MPS

        dummy = torch.ones(size=outputs.size(), dtype=indices.dtype) * indices.unsqueeze(dim=1).cpu()
        outputs = self.unpool(outputs, dummy).to('mps')
        outputs = outputs.permute(0, 2, 1, 3, 4)

        return outputs

    def forward(self, x: torch.Tensor, horizon: int = 0) -> torch.Tensor:
        mask = torch.where(torch.sum(x, dim=1) > 0, 1, 0).unsqueeze(dim=1).to(x.dtype)
        x = self.batch_norm(x)
        b, seq_len, _, h, w = x.size()

        h_t1, c_t1 = self.encoder.init_hidden(batch_size=b, image_size=(h, w))
        h_t1, c_t1 = h_t1.to(x.dtype), c_t1.to(x.dtype)

        h_t2, c_t2 = self.decoder.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = h_t2.to(x.dtype), c_t2.to(x.dtype)

        outputs = self.autoencoder(x, seq_len, horizon, h_t1, c_t1, h_t2, c_t2)
        outputs = self.relu(outputs)
        outputs = outputs * mask

        return outputs
