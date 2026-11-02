from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ctd.comparison.analysis.dd.dd import Analysis_DD_LFADS, Analysis_DD_SAE
from ctd.comparison.analysis.tt.tasks.tt_MultiTask import Analysis_TT_MultiTask
from ctd.comparison.analysis.tt.tt import Analysis_TT


class ExportModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(3))
        self.dropout = nn.Dropout(0.8)
        self.frozen = nn.Linear(3, 3)
        self.frozen.eval()
        self.dynamic_noise = 0.2
        self.fail = False

    def forward(self, *inputs):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in self.modules())
        if self.fail:
            raise RuntimeError("export failed")
        values = self.dropout(torch.ones(2, 4, 3)) * self.weight
        return values, values + 1

    def predict_step(self, batch, batch_ix, sample_posteriors):
        assert sample_posteriors is False
        values, latents = self()
        return values, None, None, None, values + 2, None, latents


class ExportWrapper(ExportModel):
    def forward(self, *inputs):
        values, latents = super().forward(*inputs)
        return {"controlled": values, "latents": latents}


def make_analysis(kind):
    classes = {
        "sae": Analysis_DD_SAE,
        "lfads": Analysis_DD_LFADS,
        "tt": Analysis_TT,
        "multitask": Analysis_TT_MultiTask,
    }
    analysis = object.__new__(classes[kind])
    model = ExportWrapper() if kind in {"tt", "multitask"} else ExportModel()
    analysis.model = model
    analysis.wrapper = model
    analysis.trim_inds = None
    values = torch.ones(2, 4, 3)
    tensors = (values, values, values)
    analysis.get_model_inputs = lambda phase: tensors[:2] if kind == "sae" else tensors
    analysis.get_model_inputs_noiseless = lambda phase: tensors
    analysis.get_inputs_to_env = lambda phase: values
    analysis.datamodule = SimpleNamespace(
        train_dataloader=lambda shuffle=False: [[values]],
        val_dataloader=lambda: [[values]],
    )
    return analysis, model


EXPORTS = [
    ("sae", "get_model_outputs"),
    ("sae", "get_latents"),
    ("sae", "get_rates"),
    ("lfads", "get_model_outputs"),
    ("lfads", "get_latents"),
    ("lfads", "get_rates"),
    ("lfads", "get_inferred_inputs"),
    ("tt", "get_model_outputs"),
    ("tt", "get_model_outputs_noiseless"),
    ("tt", "get_latents"),
    ("tt", "get_latents_noiseless"),
    ("multitask", "get_model_outputs_noiseless"),
]


@pytest.mark.parametrize("phase", ["train", "val", "all"])
@pytest.mark.parametrize("kind, getter", EXPORTS)
def test_exports_disable_dropout_and_autograd_then_restore_modes(kind, getter, phase):
    analysis, model = make_analysis(kind)
    modes = [module.training for module in model.modules()]
    first = getattr(analysis, getter)(phase=phase)
    second = getattr(analysis, getter)(phase=phase)
    if isinstance(first, dict):
        first, second = tuple(first.values()), tuple(second.values())
    first = first if isinstance(first, tuple) else (first,)
    second = second if isinstance(second, tuple) else (second,)
    for actual, repeated in zip(first, second):
        assert torch.equal(actual, repeated)
        assert not actual.requires_grad
        assert actual.grad_fn is None
        # Unlike inference-mode tensors, exports remain valid autograd inputs.
        probe = nn.Linear(3, 1)
        probe(actual).sum().backward()
        assert probe.weight.grad is not None
    assert [module.training for module in model.modules()] == modes
    assert model.weight.requires_grad
    assert model.dynamic_noise == 0.2
    assert torch.is_grad_enabled()
    # Dynamics calculations outside the getter must retain state gradients.
    state = torch.ones(3, requires_grad=True)
    (state * model.weight).sum().backward()
    assert state.grad is not None


@pytest.mark.parametrize("kind, getter", EXPORTS)
def test_failed_exports_restore_modes_and_gradients(kind, getter):
    analysis, model = make_analysis(kind)
    modes = [module.training for module in model.modules()]
    model.fail = True
    with pytest.raises(RuntimeError, match="export failed"):
        getattr(analysis, getter)(phase="val")
    assert [module.training for module in model.modules()] == modes
    assert model.dynamic_noise == 0.2
    assert torch.is_grad_enabled()
