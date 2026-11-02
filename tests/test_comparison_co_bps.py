from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ctd.comparison.comparison import Comparison
from ctd.comparison.metrics import get_bps


@pytest.mark.parametrize("n_task_inputs", [1, 2, 3])
def test_co_bps_scores_only_heldout_neurons(n_task_inputs):
    n_heldin = 4
    spikes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 1.0, 3.0], [2.0, 1.0, 4.0, 3.0, 3.0, 1.0]]]
    )
    rates = spikes.clone()
    # Poor held-in predictions must never influence the held-out score.
    rates[..., :n_heldin] = 20.0
    latents = torch.zeros(1, 2, 2)
    task_inputs = torch.zeros(1, 2, n_task_inputs)

    reference = SimpleNamespace(
        run_name="teacher",
        env=SimpleNamespace(dataset_name="NBFF"),
        get_latents=lambda phase: latents,
        get_inputs=lambda phase: task_inputs,
    )

    def get_model_inputs(phase):
        assert phase == "val"
        return spikes[..., :n_heldin], task_inputs

    analysis = SimpleNamespace(
        run_name="student",
        get_inputs=lambda phase="val": task_inputs,
        get_model_inputs=get_model_inputs,
        get_model_outputs=lambda phase: (rates, latents),
        get_true_rates=lambda phase: spikes,
        get_spiking=lambda phase: spikes,
    )
    comparison = Comparison()
    comparison.load_analysis(reference, reference_analysis=True)
    comparison.load_analysis(analysis)

    result = comparison.compute_metrics(metric_dict_list={"co-bps": {}})

    expected = get_bps(rates[..., n_heldin:].numpy(), spikes[..., n_heldin:])
    contaminated = get_bps(
        rates[..., n_task_inputs:].numpy(), spikes[..., n_task_inputs:]
    )
    assert np.isfinite(expected)
    assert expected != pytest.approx(contaminated)
    assert result["co-bps"] == pytest.approx([expected])
