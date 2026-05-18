import torch
from torch import nn
from torch.nn import GRUCell, RNNCell

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


class GRU_RNN(nn.Module):
    def __init__(
        self, latent_size, input_size=None, output_size=None, latent_ic_var=0.05
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class NoisyGRU(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class NoisyGRU_RNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden


class NoisyGRU_LatentL2(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        latent_ic_var=0.05,
        l2_wt=1e-2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        self.noise_level = noise_level
        self.l2_wt = l2_wt
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )
        self.latent_ic_var = latent_ic_var

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        hidden = self.cell(inputs, hidden)
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = hidden + noise
        return output, hidden

    def model_loss(self, loss_dict):
        latents = loss_dict["latents"]
        lats_flat = latents.view(-1, latents.shape[-1])
        latent_l2_loss = self.l2_wt * torch.norm(lats_flat, p=2, dim=1).mean()
        return latent_l2_loss


class DriscollRNN(nn.Module):
    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        noise_level=0.05,
        gamma=0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.readout = None
        self.noise_level = noise_level
        self.gamma = gamma
        self.act_func = nn.Tanh()

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

    def forward(self, inputs, hidden):
        noise = torch.randn_like(hidden) * self.noise_level
        output = self.readout(hidden)
        hidden = (1 - self.gamma) * self.recW(hidden) + self.gamma * self.act_func(
            self.recW(hidden) + self.inpW(inputs) + self.bias + noise
        )
        return output, hidden


class Vanilla_RNN(nn.Module):
    def __init__(self, latent_size, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = RNNCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)

    def forward(self, inputs, hidden=None):
        hidden = self.cell(inputs, hidden)
        output = self.readout(hidden)
        return output, hidden


class ChaoticVanillaRNN(nn.Module):
    """Tanh RNN with gain-controlled recurrent initialization."""

    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        recurrent_gain=1.5,
        noise_level=0.01,
        latent_ic_var=0.05,
        hidden_clip=5.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.recurrent_gain = recurrent_gain
        self.noise_level = noise_level
        self.latent_ic_var = latent_ic_var
        self.hidden_clip = hidden_clip

        self.inpW = None
        self.recW = None
        self.bias = None
        self.readout = None
        self.act = nn.Tanh()
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=True
        )

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.latent_size))
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

        rec_std = self.recurrent_gain / (self.latent_size**0.5)
        with torch.no_grad():
            self.recW.weight.normal_(mean=0.0, std=rec_std)
            self.inpW.weight.normal_(mean=0.0, std=1.0 / (self.input_size**0.5))

    def forward(self, inputs, hidden):
        pre = self.recW(hidden) + self.inpW(inputs) + self.bias
        if self.noise_level > 0:
            pre = pre + torch.randn_like(pre) * self.noise_level
        hidden = self.act(pre)
        if self.hidden_clip is not None:
            hidden = torch.clamp(hidden, -self.hidden_clip, self.hidden_clip)
        output = self.readout(hidden)
        return output, hidden


class ChaoticRateRNN(nn.Module):
    """Leaky rate tanh RNN with gain-controlled recurrent initialization."""

    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        recurrent_gain=1.5,
        noise_level=0.01,
        latent_ic_var=0.05,
        hidden_clip=5.0,
        alpha=0.2,
        use_bias=False,
        learnable_ics=True,
        init_hidden_dist="gaussian",
        init_hidden_uniform_bound=0.1,
        input_trainable=True,
        input_init_dist="normal",
        input_uniform_bound=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.recurrent_gain = recurrent_gain
        self.noise_level = noise_level
        self.latent_ic_var = latent_ic_var
        self.hidden_clip = hidden_clip
        self.alpha = alpha
        self.use_bias = use_bias
        self.learnable_ics = learnable_ics
        self.init_hidden_dist = init_hidden_dist
        self.init_hidden_uniform_bound = init_hidden_uniform_bound
        self.input_trainable = input_trainable
        self.input_init_dist = input_init_dist
        self.input_uniform_bound = input_uniform_bound

        self.inpW = None
        self.recW = None
        self.bias = None
        self.readout = None
        self.act = nn.Tanh()
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=learnable_ics
        )

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        if self.init_hidden_dist == "uniform":
            ic_noise = (
                2.0 * torch.rand_like(init_h) - 1.0
            ) * self.init_hidden_uniform_bound
        else:
            ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.recW = nn.Linear(self.latent_size, self.latent_size, bias=False)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.latent_size))
        else:
            self.bias = None
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

        rec_std = self.recurrent_gain / (self.latent_size**0.5)
        with torch.no_grad():
            self.recW.weight.normal_(mean=0.0, std=rec_std)
            if self.input_init_dist == "uniform":
                self.inpW.weight.uniform_(
                    -self.input_uniform_bound, self.input_uniform_bound
                )
            else:
                self.inpW.weight.normal_(mean=0.0, std=1.0 / (self.input_size**0.5))
        self.inpW.weight.requires_grad = self.input_trainable

    def forward(self, inputs, hidden):
        rates = self.act(hidden)
        drive = self.recW(rates) + self.inpW(inputs)
        if self.bias is not None:
            drive = drive + self.bias
        if self.noise_level > 0:
            drive = drive + torch.randn_like(drive) * self.noise_level
        hidden = (1.0 - self.alpha) * hidden + self.alpha * drive
        if self.hidden_clip is not None:
            hidden = torch.clamp(hidden, -self.hidden_clip, self.hidden_clip)
        output = self.readout(self.act(hidden))
        return output, hidden


class EIChaoticRNN(nn.Module):
    """Leaky-rate E/I RNN with Dale-constrained recurrent connectivity."""

    def __init__(
        self,
        latent_size,
        input_size=None,
        output_size=None,
        exc_fraction=0.8,
        inh_strength=1.2,
        recurrent_gain=1.5,
        noise_level=0.0,
        latent_ic_var=0.05,
        hidden_clip=None,
        alpha=0.1,
        activation="threshold_linear",
        activation_offset=0.0,
        activation_max=None,
        learnable_ics=True,
        input_trainable=True,
        readout_from_exc_only=False,
        sparsity=1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.exc_fraction = exc_fraction
        self.inh_strength = inh_strength
        self.recurrent_gain = recurrent_gain
        self.noise_level = noise_level
        self.latent_ic_var = latent_ic_var
        self.hidden_clip = hidden_clip
        self.alpha = alpha
        self.activation = activation
        self.activation_offset = activation_offset
        self.activation_max = activation_max
        self.learnable_ics = learnable_ics
        self.input_trainable = input_trainable
        self.readout_from_exc_only = readout_from_exc_only
        self.sparsity = sparsity

        self.n_exc = int(round(self.latent_size * self.exc_fraction))
        self.n_exc = min(max(self.n_exc, 1), self.latent_size - 1)
        self.n_inh = self.latent_size - self.n_exc

        self.inpW = None
        self.readout = None
        self.rec_weight = None
        self.register_buffer("rec_sign", None)
        self.register_buffer("rec_mask", None)
        self.register_buffer("readout_mask", None)
        self.latent_ics = torch.nn.Parameter(
            torch.zeros(latent_size), requires_grad=learnable_ics
        )

    def _rates(self, hidden):
        if self.activation == "relu":
            rates = torch.relu(hidden)
        else:
            rates = torch.relu(hidden + self.activation_offset)
        if self.activation_max is not None:
            rates = torch.clamp(rates, max=self.activation_max)
        return rates

    def init_hidden(self, batch_size):
        init_h = self.latent_ics.unsqueeze(0).expand(batch_size, -1)
        ic_noise = torch.randn_like(init_h) * self.latent_ic_var
        return init_h + ic_noise

    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.inpW = nn.Linear(self.input_size, self.latent_size, bias=False)
        self.readout = nn.Linear(self.latent_size, output_size, bias=True)

        self.rec_weight = nn.Parameter(torch.empty(self.latent_size, self.latent_size))

        sign = torch.ones(self.latent_size, dtype=torch.float32)
        sign[self.n_exc :] = -self.inh_strength
        self.rec_sign = sign.view(1, -1)

        if self.sparsity < 1.0:
            mask = (
                torch.rand(self.latent_size, self.latent_size) < self.sparsity
            ).float()
        else:
            mask = torch.ones(self.latent_size, self.latent_size)
        mask.fill_diagonal_(0.0)
        self.rec_mask = mask

        if self.readout_from_exc_only:
            readout_mask = torch.zeros(output_size, self.latent_size)
            readout_mask[:, : self.n_exc] = 1.0
            self.readout_mask = readout_mask
        else:
            self.readout_mask = None

        rec_std = self.recurrent_gain / (self.latent_size**0.5)
        with torch.no_grad():
            self.rec_weight.normal_(mean=0.0, std=rec_std)
            self.inpW.weight.normal_(mean=0.0, std=1.0 / (self.input_size**0.5))
            self.readout.weight.normal_(mean=0.0, std=1.0 / (self.latent_size**0.5))
            self.readout.bias.zero_()
        self.inpW.weight.requires_grad = self.input_trainable

    def forward(self, inputs, hidden):
        rates = self._rates(hidden)
        rec_eff = torch.abs(self.rec_weight) * self.rec_sign * self.rec_mask
        drive = torch.matmul(rates, rec_eff.t()) + self.inpW(inputs)
        if self.noise_level > 0:
            drive = drive + torch.randn_like(drive) * self.noise_level
        hidden = (1.0 - self.alpha) * hidden + self.alpha * drive
        if self.hidden_clip is not None:
            hidden = torch.clamp(hidden, -self.hidden_clip, self.hidden_clip)

        rates_out = self._rates(hidden)
        if hasattr(self.readout, "weight"):
            readout_weight = self.readout.weight
            if self.readout_mask is not None:
                readout_weight = readout_weight * self.readout_mask
            output = nn.functional.linear(rates_out, readout_weight, self.readout.bias)
        else:
            output = self.readout(rates_out)
        return output, hidden
