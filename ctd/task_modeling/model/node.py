import torch
from torch import nn

"""
All models must meet a few requirements
    1. They must have an init_model method that takes
    input_size and output_size as arguments
    2. They must have a forward method that takes inputs and hidden
    as arguments and returns output and hidden for one time step
    3. They must have a cell attribute that is the recurrent cell
    4. They must have a readout attribute that is the output layer
    (mapping from latent to output)

    Optionally,
    1. They can have an init_hidden method that takes
    batch_size as an argument and returns an initial hidden state
    2. They can have a model_loss method that takes a loss_dict
    as an argument and returns a loss (L2 regularization on latents, etc.)

"""


class NODE(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.generator = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )

    def init_hidden(self, batch_size):
        return self.latent_ics.unsqueeze(0).expand(batch_size, -1)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.generator = MLPCell(
            input_size, self.num_layers, self.layer_hidden_size, self.latent_size
        )
        self.readout = nn.Linear(self.latent_size, output_size)
        # Initialize weights and biases for the readout layer
        nn.init.normal_(
            self.readout.weight, mean=0.0, std=0.01
        )  # Small standard deviation
        nn.init.constant_(self.readout.bias, 0.0)  # Zero bias initialization

    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, self.latent_size), device=dev)
        hidden = self.generator(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class MLPCell(nn.Module):
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + 0.1 * self.vf_net(input_hidden)


class NODE_ICs(nn.Module):
    def __init__(
        self,
        num_layers,
        layer_hidden_size,
        latent_size,
        output_size=None,
        input_size=None,
        ic_size=3,
        gain_val=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.input_size = input_size
        self.generator = None
        self.readout = None
        self.encoder = None
        self.ic_size = ic_size
        self.gain_val = gain_val

    def init_hidden_from_ic(self, ics):
        return self.encoder(ics)

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.generator = MLPSlowCell(
            input_size,
            self.num_layers,
            self.layer_hidden_size,
            self.latent_size,
            gain_val=self.gain_val,
        )
        self.readout = nn.Identity()
        self.encoder = nn.Identity()

    def forward(self, inputs, hidden=None):
        n_samples, n_inputs = inputs.shape
        dev = inputs.device
        if hidden is None:
            hidden = torch.zeros((n_samples, self.latent_size), device=dev)
        hidden = self.generator(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class MLPSlowCell(nn.Module):
    def __init__(
        self, input_size, num_layers, layer_hidden_size, latent_size, gain_val=0.01
    ):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        self.gain_val = gain_val
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + self.gain_val * self.vf_net(input_hidden)
