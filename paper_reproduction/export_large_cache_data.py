#!/usr/bin/env python3
"""Thin the trusted multi-gigabyte S4/S5/S16 caches for public deposition.

Run this after export_release_data.py and point it at the same output directory.
It rewrites SHA256SUMS.txt after adding portable NPZ/CSV/JSON files. Loading
pickle can execute code, so only use caches from a trusted CtDToolkit checkout.
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_release_data as common  # noqa: E402

PCM_NEURONS = {
    "tt": (1, 2, 3, 10, 11, 14),
    "dd16": (1, 3, 4, 9, 10, 12),
}
PCM_TRIAL = 0
PCM_MAX_PCA_TRIALS = 240
PCM_SCATTER_STEP = 50
CDM_RASTER_TRIAL = 2
CDM_RASTER_NEURONS = 48


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=common.DEFAULT_CACHES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def selected_indices(total, maximum):
    if total <= maximum:
        return np.arange(total)
    return np.linspace(0, total - 1, maximum).astype(int)


def export_s4(cache_root, output):
    source = cache_root / "supplementary" / "make_phase_coded_memory_figure.cache.pkl"
    data = common.trusted_pickle(source)
    destination = output / "s4"
    destination.mkdir(parents=True, exist_ok=True)

    trials = selected_indices(data["lats_tt_pca"].shape[0], PCM_MAX_PCA_TRIALS)
    np.savez_compressed(
        destination / "pca_trajectories.npz",
        trial_indices=trials,
        extras=data["extras"][trials],
        is_stim_a=data["inds_a"][trials],
        is_stim_b=data["inds_b"][trials],
        tt_latents=data["lats_tt_pca"][trials],
        dd16_latents=data["lats_dd_pca"][trials],
        ground_truth_rates=data["rates_pca"][trials],
    )

    unit_rows = []
    for system, flat_key in (("tt", "flat_tt"), ("dd16", "flat_dd")):
        flat = data[flat_key]
        points = np.arange(
            0, min(flat["a"].shape[0], flat["b"].shape[0]), PCM_SCATTER_STEP
        )
        for stimulus in ("a", "b"):
            phase = flat[f"theta_{stimulus}"][points]
            values = flat[stimulus][points]
            for neuron in PCM_NEURONS[system]:
                actual_neuron = min(neuron, values.shape[1] - 1)
                unit_rows.extend(
                    {
                        "system": system,
                        "stimulus": stimulus.upper(),
                        "sample_index": int(point),
                        "phase": theta,
                        "neuron": int(actual_neuron),
                        "normalized_activity": value,
                    }
                    for point, theta, value in zip(
                        points, phase, values[:, actual_neuron]
                    )
                )
    common.rows_csv(
        destination / "single_unit_activity.csv",
        [
            "system",
            "stimulus",
            "sample_index",
            "phase",
            "neuron",
            "normalized_activity",
        ],
        unit_rows,
    )

    neuron_count = min(
        500,
        data["true_rates"].shape[-1],
        data["spikes"].shape[-1],
        data["inf_rates"].shape[-1],
    )
    np.savez_compressed(
        destination / "neural_data_trial.npz",
        trial_index=PCM_TRIAL,
        true_rates=data["true_rates"][PCM_TRIAL, :, :neuron_count],
        spikes=data["spikes"][PCM_TRIAL, :, :neuron_count],
        dd16_inferred_rates=data["inf_rates"][PCM_TRIAL, :, :neuron_count],
    )
    np.savez_compressed(
        destination / "stimulus_activity_differences.npz",
        tt=data["mean_delta_tt"],
        dd8=data["mean_delta_dd8"],
        dd16=data["mean_delta_dd"],
        dd128=data["mean_delta_dd128"],
    )
    common.json_file(
        destination / "metadata.json",
        {
            "source_cache": source.relative_to(cache_root).as_posix(),
            "trial_index": PCM_TRIAL,
            "max_pca_trials": PCM_MAX_PCA_TRIALS,
            "scatter_step": PCM_SCATTER_STEP,
            "selected_neurons": PCM_NEURONS,
            "dd16_metrics": data["bar_metrics"],
        },
    )
    del data
    gc.collect()


def export_s16(cache_root, output):
    source = cache_root / "supplementary" / "make_phase_coded_memory_simple.cache.pkl"
    data = common.trusted_pickle(source)
    destination = output / "s16"
    destination.mkdir(parents=True, exist_ok=True)

    trials = selected_indices(data["extras"].shape[0], PCM_MAX_PCA_TRIALS)
    arrays = {
        "trial_indices": trials,
        "extras": data["extras"][trials],
        "is_stim_a": data["inds_a"][trials],
        "is_stim_b": data["inds_b"][trials],
        "ground_truth_rate_pcs": data["gt"]["rates_pca"][trials],
        "ground_truth_latent_pcs": data["gt"]["lats_pca"][trials],
    }
    for label, key in (("dd8", "dd8"), ("dd16", "dd"), ("dd128", "dd128")):
        arrays[f"{label}_rate_pcs"] = data[key]["rates_pca"][trials]
        arrays[f"{label}_latent_pcs"] = data[key]["lats_pca"][trials]
    np.savez_compressed(destination / "pca_trajectories.npz", **arrays)
    np.savez_compressed(
        destination / "firing_rate_rasters.npz",
        trial_index=PCM_TRIAL,
        ground_truth=data["true_rates"][PCM_TRIAL],
        dd8=data["dd8"]["inf_rates"][PCM_TRIAL],
        dd16=data["dd"]["inf_rates"][PCM_TRIAL],
        dd128=data["dd128"]["inf_rates"][PCM_TRIAL],
    )
    common.json_file(
        destination / "metadata.json",
        {
            "source_cache": source.relative_to(cache_root).as_posix(),
            "trial_index": PCM_TRIAL,
            "max_pca_trials": PCM_MAX_PCA_TRIALS,
            "model_metrics": {
                label: {
                    "rate_r2": data[key]["rate_r2"],
                    "state_r2": data[key]["state_r2"],
                }
                for label, key in (
                    ("dd8", "dd8"),
                    ("dd16", "dd"),
                    ("dd128", "dd128"),
                )
            },
        },
    )
    del data
    gc.collect()


def first_cdm_trials_by_pair(extra):
    selected = {}
    for index, row in enumerate(extra):
        pair = (int(row[10]), int(row[11]))
        if pair not in selected:
            selected[pair] = index
        if len(selected) == 4:
            break
    ordered_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    return np.array([selected[pair] for pair in ordered_pairs if pair in selected])


def export_s5(cache_root, output):
    source = (
        cache_root / "supplementary" / "make_chaotic_delayed_memory_figure.cache.pkl"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = common.trusted_pickle(source)
    destination = output / "s5"
    destination.mkdir(parents=True, exist_ok=True)

    trial = int(data["trial_idx"])
    ic_trial = int(data["ic_trial_idx"])
    inputs = np.asarray(data["inputs_np"])
    np.savez_compressed(
        destination / "task_and_output_trial.npz",
        trial_index=trial,
        inputs=inputs[trial],
        targets=data["targets_np"][trial],
        model_output=data["controlled_np"][trial],
        phase_metadata=data["extra_np"][trial],
        perturbed_outputs=data["pert_out_np"][:, trial],
    )

    type_trials = first_cdm_trials_by_pair(data["extra_np"])
    np.savez_compressed(
        destination / "task_trained_dynamics.npz",
        cue_pair_trial_indices=type_trials,
        cue_pair_metadata=data["extra_np"][type_trials],
        cue_pair_latent_pcs=data["latents_pc"][type_trials],
        ic_trial_index=ic_trial,
        baseline_ic_latent_pcs=data["latents_pc"][ic_trial],
        perturbed_ic_latent_pcs=data["pert_pc"][:, ic_trial],
        baseline_initial_pc=data["base_h0_pc"][ic_trial],
        perturbed_initial_pcs=data["pert_h0_pc"][:, ic_trial],
        latent_perturbation_norm=data["latent_delta"][:, ic_trial],
        output_perturbation_norm=data["output_delta"][:, ic_trial],
    )

    dd = data["dd_data"]
    true_rates = dd["true_rates"]
    n_shared = min(true_rates.shape[-1], dd["rates"].shape[-1])
    variance = np.nanvar(true_rates[:, :, :n_shared], axis=(0, 1))
    neurons = np.argsort(variance)[::-1][:CDM_RASTER_NEURONS]
    raster_trial = min(CDM_RASTER_TRIAL, true_rates.shape[0] - 1)
    dd_trial = int(dd["trial_idx"])
    np.savez_compressed(
        destination / "data_driven_fit.npz",
        raster_trial_index=raster_trial,
        neuron_indices=neurons,
        spikes=dd["spikes"][raster_trial][:, neurons],
        true_rates=true_rates[raster_trial][:, neurons],
        predicted_rates=dd["rates"][raster_trial][:, neurons],
        phase_metadata=dd["extra"][raster_trial],
        example_trial_index=dd_trial,
        example_true_rates=true_rates[dd_trial][:, neurons[:4]],
        example_predicted_rates=dd["rates"][dd_trial][:, neurons[:4]],
        dd_latent_pcs=dd["latents_pc"][dd_trial],
        aligned_tt_latent_pcs=dd["tt_aligned_pc"][dd_trial],
    )

    perturbations = dd.get("perturbations", {}) or {}
    if perturbations:
        pert_trial = int(perturbations["trial_idx"])
        np.savez_compressed(
            destination / "data_driven_perturbations.npz",
            trial_index=pert_trial,
            baseline_latent_pcs=perturbations["baseline_pc"][pert_trial],
            perturbed_latent_pcs=perturbations["pert_pc"][:, pert_trial],
            perturbation_norm=perturbations["delta_norm"][:, pert_trial],
            tt_perturbation_norm=data["latent_delta"][:, ic_trial],
        )

    metrics = dict(dd.get("metrics", {}))
    metrics.pop("path", None)
    signature = dict(data.get("__signature__", {}))
    signature.pop("tt_run_path", None)
    signature.pop("dd_run_path", None)
    common.json_file(
        destination / "metadata.json",
        {
            "source_cache": source.relative_to(cache_root).as_posix(),
            "task_trained_lyapunov_max": data["TT_LYAP_MAX"],
            "task_trained_lyapunov_max_std": data["TT_LYAP_MAX_STD"],
            "data_driven_metrics": metrics,
            "cache_signature_without_local_paths": signature,
            "raster_neuron_count": CDM_RASTER_NEURONS,
        },
    )
    del data
    gc.collect()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    export_s4(args.cache_root, args.output)
    export_s16(args.cache_root, args.output)
    export_s5(args.cache_root, args.output)
    common.checksums(args.output)
    print(f"Added S4, S5, and S16 thin exports to {args.output}")


if __name__ == "__main__":
    main()
