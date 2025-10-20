import math

import geotorch
import torch
from torch import nn

# import h5py


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers, out_features):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_features, hidden_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = self.output_layer(x)
        return x


class StiefelLinear(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        geotorch.orthogonal(
            self.linear, "weight"
        )  # Constrain weight to Stiefel manifold

    def forward(self, x):
        return self.linear(x)


# class PCRInitModuleList(nn.ModuleList):
#     def __init__(self, inits_path: str, modules: list[nn.Module]):
#         super().__init__(modules)
#         # Pull pre-computed initialization from the file, assuming correct order
#         with h5py.File(inits_path, "r") as h5file:
#             weights = [v["/" + k + "/matrix"][()] for k, v in h5file.items()]
#             biases = [v["/" + k + "/bias"][()] for k, v in h5file.items()]
#         # Load the state dict for each layer
#         for layer, weight, bias in zip(self, weights, biases):
#             state_dict = {"weight": torch.tensor(weight), "bias": torch.tensor(bias)}
#             layer.load_state_dict(state_dict)
