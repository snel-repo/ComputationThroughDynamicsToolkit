import torch
from torch import nn


class TT_Template(nn.Module):
    """
    Template for Task-Trained RNN models.

    All subclasses must implement:
      1. init_model(input_size, output_size)
      2. forward(inputs, hidden) -> (output, hidden)
      3. self.cell      # the recurrent cell module
      4. self.readout   # the output layer (latent → output)

    Optional hooks:
      - init_hidden(batch_size) -> Tensor
      - model_loss(loss_dict) -> Tensor
    """

    def __init__(
        self, latent_size: int, input_size: int = None, output_size: int = None
    ):
        super().__init__()
        # will be set in init_model():
        self.cell: nn.Module = None
        self.readout: nn.Module = None

        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size

    def init_model(self, input_size: int, output_size: int):
        """
        Instantiate:
          - self.cell    (e.g. GRUCell/LSTMCell)
          - self.readout (nn.Linear from latent_size→output_size)
        """
        raise NotImplementedError("Must implement init_model()")

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        (Optional) Return initial hidden state of shape
        (batch_size, latent_size).
        """
        raise NotImplementedError("Optional: implement init_hidden()")

    def forward(
        self, inputs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One timestep of RNN dynamics:
          inputs: Tensor [batch_size, input_size]
          hidden: Tensor [batch_size, latent_size]
        Returns:
          output: Tensor [batch_size, output_size]
          hidden: Tensor [batch_size, latent_size]
        """
        raise NotImplementedError("Must implement forward()")

    def model_loss(self, loss_dict: dict) -> torch.Tensor:
        """
        (Optional) Compute extra loss terms (e.g. latent regularization)
        from loss_dict and return a scalar Tensor.
        """
        raise NotImplementedError("Optional: implement model_loss()")
