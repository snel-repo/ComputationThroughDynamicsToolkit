from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ctd.comparison.analysis.dd.dd import Analysis_DD_LFADS, Analysis_DD_SAE
from ctd.data_modeling.datamodules.LFADS.datamodule import SessionDataset
from ctd.data_modeling.datamodules.LFADS.tuples import SessionBatch


def make_analysis(kind, task, extras, keys=("inputs", "latents", "activity"), steps=10):
    cls = Analysis_DD_SAE if kind == "sae" else Analysis_DD_LFADS
    analysis = object.__new__(cls)
    datasets = []
    for extra in extras:
        values = torch.zeros(len(extra), steps, 3)
        if kind == "sae":
            dataset = TensorDataset(values, values, values, extra)
        else:
            batch = SessionBatch(values, values, values, values, values)
            metadata = tuple(extra if key == "extra" else values for key in keys)
            dataset = SessionDataset(batch, metadata)
        datasets.append(dataset)
    analysis.datamodule = SimpleNamespace(
        hparams={"prefix": f"tt_{task}", "batch_keys": list(keys)},
        # Historical SAE default must not override the actual dataset prefix.
        system="3BFF",
        train_ds=datasets[0],
        valid_ds=datasets[1],
        train_dataloader=lambda shuffle=False: DataLoader(
            datasets[0], batch_size=1, shuffle=shuffle
        ),
        val_dataloader=lambda: DataLoader(datasets[1], batch_size=1),
    )
    analysis.model = SimpleNamespace(
        hparams=SimpleNamespace(recon_seq_len=steps, ic_enc_seq_len=2)
    )
    return analysis


@pytest.mark.parametrize("kind", ["sae", "lfads"])
@pytest.mark.parametrize(
    "task",
    ["3bff", "NBFF", "RandomTarget", "PhaseCodedMemory", "ChaoticDelayedMatching"],
)
@pytest.mark.parametrize("phase, count", [("train", 2), ("val", 1), ("all", 3)])
def test_fixed_length_tasks_use_saved_window_not_extra(kind, task, phase, count):
    # Zeros in 3BFF and nonzero labels/timing in other tasks are not lengths.
    extras = [torch.tensor([[0.0, 1.0], [3.0, 0.0]]), torch.tensor([[2.0, 1.0]])]
    analysis = make_analysis(kind, task, extras, steps=146)
    lengths = analysis.get_trial_lens(phase)
    assert lengths.dtype == torch.long
    assert lengths.tolist() == [146] * count


@pytest.mark.parametrize("kind", ["sae", "lfads"])
@pytest.mark.parametrize("extra_index", [0, 1, 3])
@pytest.mark.parametrize(
    "phase, expected", [("train", [5, 10]), ("val", [7]), ("all", [5, 10, 7])]
)
def test_multitask_uses_response_ends_and_named_metadata(
    kind, extra_index, phase, expected
):
    keys = ["inputs", "latents", "activity"]
    keys.insert(extra_index, "extra")
    extras = [torch.tensor([[1.0, 5.0], [4.0, 10.0]]), torch.tensor([[3.0, 7.0]])]
    analysis = make_analysis(kind, "MultiTask", extras, keys)
    assert analysis.get_trial_lens(phase).tolist() == expected


@pytest.mark.parametrize("kind", ["sae", "lfads"])
def test_multitask_clips_trial_ends_to_reconstruction_window(kind):
    extras = [torch.tensor([[1.0, 8.0]]), torch.tensor([[2.0, 4.0]])]
    analysis = make_analysis(kind, "MultiTask", extras, keys=("extra",), steps=6)
    assert analysis.get_trial_lens().tolist() == [6, 4]


def test_lfads_uses_configured_decoder_window_not_encoder_length():
    extras = [torch.zeros(1, 1), torch.zeros(1, 1)]
    analysis = make_analysis("lfads", "3bff", extras, steps=12)
    analysis.model.hparams.recon_seq_len = 10
    assert analysis.get_trial_lens().tolist() == [10, 10]


def test_lfads_multitask_rejects_missing_named_metadata():
    analysis = make_analysis("lfads", "MultiTask", [torch.zeros(1, 2)] * 2)
    with pytest.raises(ValueError, match="'extra' in LFADS batch_keys"):
        analysis.get_trial_lens()


@pytest.mark.parametrize("kind", ["sae", "lfads"])
@pytest.mark.parametrize(
    "extra",
    [
        torch.zeros(1, 1),
        torch.zeros(1, 2),
        torch.tensor([[1.0, 2.5]]),
        torch.tensor([[1.0, float("nan")]]),
    ],
)
def test_multitask_rejects_invalid_trial_end_metadata(kind, extra):
    analysis = make_analysis(kind, "MultiTask", [extra] * 2, keys=("extra",))
    with pytest.raises(ValueError, match="MultiTask"):
        analysis.get_trial_lens()


@pytest.mark.parametrize("kind", ["sae", "lfads"])
def test_unknown_task_does_not_silently_assume_full_length(kind):
    analysis = make_analysis(kind, "CustomPaddedTask", [torch.ones(1, 2)] * 2)
    with pytest.raises(ValueError, match="Unknown trial-length schema"):
        analysis.get_trial_lens()


@pytest.mark.parametrize("kind", ["sae", "lfads"])
def test_invalid_phase_has_clear_error(kind):
    analysis = make_analysis(kind, "3bff", [torch.zeros(1, 1)] * 2)
    with pytest.raises(ValueError, match="Unknown phase"):
        analysis.get_trial_lens("test")
