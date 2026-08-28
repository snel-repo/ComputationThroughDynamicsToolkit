#!/usr/bin/env python3
"""Execute Figure 4 analyses and export only the displayed portable arrays.

This is a read-only extraction from an existing trusted model tree. It does not
train models. Run it with the repository's Python 3.10+ analysis environment,
after export_release_data.py, and point it at the same output directory.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import export_release_data as common  # noqa: E402

EPOCHS = (10, 50, 100, 250, 500)
MODEL_NAMES = ("TT", "LFADS", "GRU", "LDS")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Trusted CtDToolkit checkout containing content/trained_models.",
    )
    parser.add_argument(
        "--extra-site-packages",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional fallback package directory appended after the active "
            "environment."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def install_pickle_compatibility():
    if "DSA" not in sys.modules:
        dsa_stub = types.ModuleType("DSA")

        class UnavailableDSA:
            def __init__(self, *args, **kwargs):
                raise ImportError("DSA requires optional kooplearn dependencies")

        dsa_stub.DSA = UnavailableDSA
        sys.modules["DSA"] = dsa_stub
    try:
        import lightning_fabric.utilities.data as fabric_data

        if not hasattr(fabric_data, "AttributeDict"):

            class AttributeDict(dict):
                __getattr__ = dict.get
                __setattr__ = dict.__setitem__
                __delattr__ = dict.__delitem__

            fabric_data.AttributeDict = AttributeDict
    except ImportError:
        pass


def execute_figure4(data_root, extra_site_packages=()):
    source = ROOT / "examples" / "figures" / "Fig4Canonical"
    script = source / "make_figure4_canonical.py"
    old_cwd = Path.cwd()
    old_home_dir = os.environ.get("HOME_DIR")
    (source / "outputs").mkdir(parents=True, exist_ok=True)
    os.environ["HOME_DIR"] = str(data_root.resolve()) + os.sep
    import dotenv

    dotenv.find_dotenv = lambda *args, **kwargs: ""
    dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.path.insert(0, str(data_root / "libs" / "lfads-jslds"))
    sys.path.insert(0, str(data_root / "libs" / "DSA"))
    for package_dir in extra_site_packages:
        sys.path.append(str(package_dir))
    install_pickle_compatibility()
    try:
        os.chdir(source)
        namespace = runpy.run_path(str(script), run_name="__figure4_export__")
    finally:
        os.chdir(old_cwd)
        if old_home_dir is None:
            os.environ.pop("HOME_DIR", None)
        else:
            os.environ["HOME_DIR"] = old_home_dir
    return namespace


def export_learning_progression(namespace, destination):
    trial = int(namespace["trial_ind"])
    inputs = []
    outputs = []
    for epoch in EPOCHS:
        inputs.append(array(namespace[f"in_{epoch}"])[trial, :100])
        outputs.append(array(namespace[f"out_{epoch}"]["controlled"])[trial, :100])
    np.savez_compressed(
        destination / "panel_a_learning_progression.npz",
        epochs=np.asarray(EPOCHS),
        trial_index=trial,
        inputs=np.stack(inputs),
        outputs=np.stack(outputs),
    )


def export_3bff(namespace, destination):
    trial_indices = np.array([2, 1, 0])
    trajectories = np.stack(
        [
            np.asarray(latents)[trial_indices, :, :3]
            for latents in namespace["row1_latents"]
        ]
    )
    np.savez_compressed(
        destination / "panel_b_3bff_trajectories.npz",
        model_names=np.asarray(MODEL_NAMES),
        trial_indices=trial_indices,
        trajectories=trajectories,
        shared_axis_ranges=np.asarray(namespace["axis_ranges_r1"]),
    )


def export_multitask(namespace, destination):
    rows = []
    bins = np.asarray(namespace["memProBins"])
    for model, trajectories in zip(MODEL_NAMES, namespace["row2_data"]):
        for trial_index, trajectory in enumerate(trajectories):
            for time_index, point in enumerate(np.asarray(trajectory)[:, :3]):
                rows.append(
                    {
                        "model": model,
                        "trial_index": trial_index,
                        "time_index": time_index,
                        "response_angle_bin": int(bins[trial_index]),
                        "pc1": point[0],
                        "pc2": point[1],
                        "pc3": point[2],
                    }
                )
    common.rows_csv(
        destination / "panel_c_multitask_trajectories.csv",
        [
            "model",
            "trial_index",
            "time_index",
            "response_angle_bin",
            "pc1",
            "pc2",
            "pc3",
        ],
        rows,
    )


def export_random_target(namespace, destination):
    selected = []
    angle_bins = []
    for bin_index, trials in enumerate(namespace["trial_list"]):
        for trial in trials[:15]:
            selected.append(int(trial))
            angle_bins.append(bin_index)
    selected = np.asarray(selected, dtype=int)
    trajectories = np.stack(
        [np.asarray(latents)[selected, :, :3] for latents in namespace["row3_data"]]
    )
    np.savez_compressed(
        destination / "panel_d_random_target_trajectories.npz",
        model_names=np.asarray(MODEL_NAMES),
        trial_indices=selected,
        reach_angle_bins=np.asarray(angle_bins),
        trajectories=trajectories,
    )


def representative_runs(namespace):
    mappings = {
        "3bff_gru": "subfolders_GRU_3BFF",
        "3bff_lfads": "subfolders_LFADS_3BFF",
        "3bff_lds": "subfolders_LDS_3BFF",
        "multitask_gru": "subfolders_GRU_MT",
        "multitask_lfads": "subfolders_LFADS_MT",
        "multitask_lds": "subfolders_LDS_MT",
        "random_target_gru": "subfolders_GRU_RT",
        "random_target_lfads": "subfolders_LFADS_RT",
        "random_target_lds": "subfolders_LDS_RT",
    }
    return {
        label: [Path(path).name for path in namespace[key]]
        for label, key in mappings.items()
    }


def main():
    args = arguments()
    destination = args.output / "figure4"
    destination.mkdir(parents=True, exist_ok=True)
    namespace = execute_figure4(
        args.data_root, extra_site_packages=args.extra_site_packages
    )
    export_learning_progression(namespace, destination)
    export_3bff(namespace, destination)
    export_multitask(namespace, destination)
    export_random_target(namespace, destination)
    common.json_file(
        destination / "metadata.json",
        {
            "source_script": (
                "examples/figures/Fig4Canonical/make_figure4_canonical.py"
            ),
            "models": list(MODEL_NAMES),
            "representative_runs": representative_runs(namespace),
            "panel_b_trial_order": [2, 1, 0],
            "panel_c_trials": len(namespace["row2_data"][0]),
            "panel_d_max_trials_per_reach_bin": 15,
            "random_target_lds_selection": "seed=0 (explicit)",
        },
    )
    common.checksums(args.output)
    print(f"Added Figure 4 thin export to {args.output}")


if __name__ == "__main__":
    main()
