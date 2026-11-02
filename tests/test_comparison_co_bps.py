import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ctd.comparison.comparison import Comparison
from ctd.comparison.metrics import get_bps


@pytest.mark.parametrize("n_task_inputs", [1, 2, 3])
@pytest.mark.parametrize("has_heldout", [True, False])
def test_co_bps_scores_only_heldout_neurons(n_task_inputs, has_heldout):
    n_heldin = 4 if has_heldout else 6
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

    context = (
        nullcontext()
        if has_heldout
        else pytest.warns(RuntimeWarning, match="no held-out neurons")
    )
    with context:
        result = comparison.compute_metrics(metric_dict_list={"co-bps": {}, "bps": {}})
    assert result["bps"] == pytest.approx([get_bps(rates.numpy(), spikes)])
    if not has_heldout:
        assert np.isnan(result["co-bps"][0])
        assert result["run_name"] == ["student"]
        return

    expected = get_bps(rates[..., n_heldin:].numpy(), spikes[..., n_heldin:])
    contaminated = get_bps(
        rates[..., n_task_inputs:].numpy(), spikes[..., n_task_inputs:]
    )
    assert np.isfinite(expected)
    assert expected != pytest.approx(contaminated)
    assert result["co-bps"] == pytest.approx([expected])


@pytest.mark.parametrize(
    "co_bps, expected_key, expected_label",
    [
        ([0.1, 0.2], "co-bps", "co-BPS"),
        ([np.nan, np.nan], "bps", "Reconstruction BPS (all neurons)"),
        ([0.1, np.nan], "bps", "Reconstruction BPS (all neurons)"),
    ],
)
def test_figure6_labels_actual_reconstruction_metric(
    co_bps, expected_key, expected_label
):
    # Import only the helper: the figure script executes model loading at top level.
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/figures/Fig6InputInf/make_figure6_combined.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "reconstruction_bps_axis"
    )
    scope = {"np": np}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope
    )
    assert scope["reconstruction_bps_axis"]({"co-bps": co_bps, "bps": [0.3, 0.4]}) == (
        expected_key,
        expected_label,
    )
