from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from ctd.comparison.analysis.tt.tt import Analysis_TT


@pytest.mark.parametrize("phase", ["train", "val", "all"])
@pytest.mark.parametrize("trim_inds", [None, (5, -1), (3, 12)])
def test_noisy_and_true_inputs_share_latent_time_window(phase, trim_inds):
    analysis = object.__new__(Analysis_TT)
    analysis.trim_inds = trim_inds
    analysis.train_inds = range(4)
    analysis.valid_inds = range(4, 5)
    clean = torch.arange(5 * 155 * 3, dtype=torch.float32).reshape(5, 155, 3)
    noisy = clean + 0.5
    ics = torch.zeros(5, 3)
    tensors = (ics, noisy, clean, ics, ics, ics, clean, clean)
    analysis.datamodule = SimpleNamespace(
        train_ds=TensorDataset(*(value[:4] for value in tensors)),
        valid_ds=TensorDataset(*(value[4:] for value in tensors)),
    )
    trials = {"train": slice(0, 4), "val": slice(4, 5), "all": slice(None)}[phase]
    time = slice(None) if trim_inds is None else slice(trim_inds[0] - 1, trim_inds[1])
    analysis.get_model_outputs = lambda phase: {"latents": clean[trials]}
    analysis.get_model_outputs_noiseless = analysis.get_model_outputs

    inputs = analysis.get_inputs(phase)
    true_inputs = analysis.get_true_inputs(phase)

    assert torch.equal(inputs, noisy[trials, time])
    assert torch.equal(true_inputs, clean[trials, time])
    assert torch.equal(inputs - true_inputs, torch.full_like(inputs, 0.5))
    assert inputs.shape == analysis.get_latents(phase).shape
    assert true_inputs.shape == analysis.get_latents_noiseless(phase).shape
    if trim_inds == (5, -1):
        assert true_inputs.shape[1] == 150

    # Model rollouts still need the full input sequence before analysis trimming.
    assert torch.equal(analysis.get_model_inputs(phase)[1], noisy[trials])
    assert torch.equal(analysis.get_model_inputs_noiseless(phase)[1], clean[trials])
