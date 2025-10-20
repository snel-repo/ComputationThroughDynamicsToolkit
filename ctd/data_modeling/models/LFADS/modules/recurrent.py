import torch
from torch import nn

from .initializers import init_gru_cell_


class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        scale_dim = input_size + hidden_size if is_encoder else None
        init_gru_cell_(self, scale_dim=scale_dim)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1 - z) * n
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden


class ClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.cell = ClippedGRUCell(
            input_size, hidden_size, clip_value=clip_value, is_encoder=True
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class BidirectionalClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.fwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h0_fwd, h0_bwd = h_0
        input_fwd = input
        input_bwd = torch.flip(input, [1])
        output_fwd, hn_fwd = self.fwd_gru(input_fwd, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(input_bwd, h0_bwd)
        output_bwd = torch.flip(output_bwd, [1])
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd])
        return output, h_n


class ClippedNODECell(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.clip_value = clip_value
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        net = nn.Sequential()
        net.add_module("linear_0", nn.Linear(input_size + latent_size, hidden_size))
        net.add_module("relu_0", nn.ReLU())
        for i in range(num_layers - 1):
            net.add_module(f"linear_{i+1}", nn.Linear(hidden_size, hidden_size))
            net.add_module(f"relu_{i+1}", nn.ReLU())
        net.add_module("linear_out", nn.Linear(hidden_size, latent_size))
        self.network = net

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        # concatenate input and hidden state
        x = torch.cat([input, hidden], dim=1)
        # pass through network
        output = hidden + 0.1 * self.network(x)
        # clamp output
        output = torch.clamp(output, -self.clip_value, self.clip_value)
        return output


class ClippedS2SNODECell(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_size: int,
        hidden_size: int,
        num_layers: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.clip_value = clip_value
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        net = nn.Sequential()
        net.add_module("linear_0", nn.Linear(input_size + latent_size, hidden_size))
        net.add_module("relu_0", nn.ReLU())
        for i in range(num_layers - 1):
            net.add_module(f"linear_{i+1}", nn.Linear(hidden_size, hidden_size))
            net.add_module(f"relu_{i+1}", nn.ReLU())
        net.add_module("linear_out", nn.Linear(hidden_size, latent_size))
        self.network = net

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        # concatenate input and hidden state
        x = torch.cat([input, hidden], dim=1)
        # pass through network
        output = self.network(x)
        # clamp output
        output = torch.clamp(output, -self.clip_value, self.clip_value)
        return output
