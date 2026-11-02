import pytest
import torch

from ctd.task_modeling.model.rnn import DriscollRNN


@pytest.mark.parametrize("gamma", [0.0, 0.2, 1.0])
def test_driscoll_retains_hidden_state_outside_nonlinear_recurrence(gamma):
    model = DriscollRNN(latent_size=2, gamma=gamma, noise_level=0.0)
    model.init_model(input_size=1, output_size=1)
    with torch.no_grad():
        model.recW.weight.copy_(torch.tensor([[0.1, 0.4], [-0.2, 0.3]]))
        model.inpW.weight.copy_(torch.tensor([[0.5], [-0.4]]))
        model.bias.copy_(torch.tensor([0.1, -0.1]))
    inputs = torch.tensor([[0.2], [0.5]])
    hidden = torch.tensor([[1.0, -0.5], [-0.3, 0.8]])
    expected = (1 - gamma) * hidden + gamma * torch.tanh(
        model.recW(hidden) + model.inpW(inputs) + model.bias
    )

    torch.testing.assert_close(model.state_update(inputs, hidden), expected)
    torch.testing.assert_close(model.cell(inputs, hidden), expected)
    output, updated = model(inputs, hidden)
    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(output, model.readout(hidden))


def test_driscoll_zero_weights_preserve_leak_timescale_and_state_gradients():
    model = DriscollRNN(latent_size=2, gamma=0.2)
    model.init_model(input_size=1, output_size=1)
    with torch.no_grad():
        model.recW.weight.zero_()
        model.inpW.weight.zero_()
        model.bias.zero_()
    hidden = torch.tensor([[1.0, -2.0]], requires_grad=True)
    inputs = torch.zeros(1, 1)
    updated = model.cell(inputs, hidden)
    torch.testing.assert_close(updated, 0.8 * hidden)
    updated.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.full_like(hidden, 0.8))


def test_driscoll_noise_affects_only_nonlinear_branch(monkeypatch):
    model = DriscollRNN(latent_size=2, gamma=0.2, noise_level=0.3)
    model.init_model(input_size=1, output_size=1)
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    hidden = torch.ones(1, 2)
    inputs = torch.zeros(1, 1)
    pre = model.recW(hidden) + model.inpW(inputs) + model.bias
    expected = 0.8 * hidden + 0.2 * torch.tanh(pre + 0.3)
    _, updated = model(inputs, hidden)
    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(
        model.cell(inputs, hidden), 0.8 * hidden + 0.2 * torch.tanh(pre)
    )
