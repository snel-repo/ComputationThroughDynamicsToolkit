#!/usr/bin/env python3
"""Convert trusted figure caches into portable CSV/JSON/NPZ release data.

Loading pickle files can execute code. Only use caches from a trusted
CtDToolkit checkout. Exported files contain no pickle objects or Jacobians.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CACHES = ROOT / "examples" / "figures"
DEFAULT_TT = ROOT / "pretrained" / "20241017_NBFF_NoisyGRU_NewFinal"
FIG6_GOOD, FIG6_BAD, FIG6_TRIAL = 5, 0, 1
FIG6_Q, S8_Q, N_TRAJ = 1e-5, 1e-4, 10
S8_MODELS = ("NODE8", "NODE64", "GRU8", "GRU64")
S8_TRAJ_FRAC, S8_MERGE_FRAC = 0.80, 0.02


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHES)
    parser.add_argument("--tt-dir", type=Path, default=DEFAULT_TT)
    parser.add_argument("--si-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def trusted_pickle(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def rows_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: native(row.get(field, "")) for field in fields})


def columns_csv(path, columns):
    fields = list(columns)
    lengths = {len(columns[field]) for field in fields}
    if len(lengths) != 1:
        raise ValueError(f"Unequal column lengths for {path}: {lengths}")
    length = lengths.pop()
    rows_csv(
        path,
        fields,
        ({field: columns[field][index] for field in fields} for index in range(length)),
    )


def json_file(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def figure5(cache_root, output):
    source = (
        cache_root / "Fig5Metrics" / "make_figure5_reconstruction_simplicity.cache.pkl"
    )
    data = trusted_pickle(source)
    destination = output / "figure5"
    columns_csv(destination / "metrics.csv", data["metrics"])

    rec = data["reconstruction"]
    columns_csv(
        destination / "panel_b_reconstruction.csv",
        {
            "time_bin": rec["time"],
            "true_rate": rec["true"],
            "spike_count": rec["spikes"],
            "node_2_predicted_rate": rec[2]["pred"],
            "node_8_predicted_rate": rec[8]["pred"],
        },
    )
    simp = data["simplicity"]
    simp_columns = {"time_bin": simp["time"]}
    for size in (8, 32):
        for key in ("actual", "state_pred", "cycle_pred"):
            simp_columns[f"node_{size}_{key}"] = simp[size][key]
    columns_csv(destination / "panel_e_simplicity.csv", simp_columns)
    json_file(
        destination / "metadata.json",
        {
            "source_cache": source.relative_to(cache_root).as_posix(),
            "cache_config": data["config"],
            "reconstruction_annotations": {
                str(size): {
                    "co_bps": rec[size]["co_bps"],
                    "rate_r2": rec[size]["rate_r2"],
                }
                for size in (2, 8)
            },
            "simplicity_annotations": {
                str(size): {
                    "state_r2": simp[size]["state_r2"],
                    "cycle_r2": simp[size]["cycle_r2"],
                }
                for size in (8, 32)
            },
        },
    )


def rotate_inputs(inferred, truth):
    x = inferred.reshape(-1, inferred.shape[-1])
    y = truth.reshape(-1, truth.shape[-1])
    return LinearRegression().fit(x, y).predict(x).reshape(truth.shape)


def figure6_projection(entry):
    latents = np.asarray(entry["latents"])
    flat = latents.reshape(-1, latents.shape[-1])
    pca = PCA(n_components=3).fit(flat)
    fixed = entry["fps"]
    keep = np.asarray(fixed.qstar) < FIG6_Q
    return {
        "trajectories": pca.transform(flat).reshape(*latents.shape[:2], 3)[:N_TRAJ],
        "fixed_points": pca.transform(np.asarray(fixed.xstar)[keep]),
        "is_stable": np.asarray(fixed.is_stable)[keep],
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def figure6(cache_root, output):
    metrics_source = cache_root / "Fig6InputInf" / "make_figure6_combined.metrics.pkl"
    fps_source = cache_root / "Fig6InputInf" / "make_figure6_combined.fps.pkl"
    data = trusted_pickle(metrics_source)
    fps_data = trusted_pickle(fps_source)
    destination = output / "figure6"
    destination.mkdir(parents=True, exist_ok=True)
    columns_csv(destination / "metrics.csv", data["metrics"])

    info = data["full_analyses_info"]
    good_name = info[FIG6_GOOD]["run_name"]
    bad_name = info[FIG6_BAD]["run_name"]
    truth = np.asarray(data["true_inputs"])
    rotated_good = rotate_inputs(
        np.asarray(data["all_inferred_inputs"][good_name]), truth
    )
    rotated_bad = rotate_inputs(
        np.asarray(data["all_inferred_inputs"][bad_name]), truth
    )
    arrays = {
        "controlled": np.asarray(data["controlled"])[FIG6_TRIAL],
        "true_input": truth[FIG6_TRIAL],
        "effective_input": np.asarray(data["effective"])[FIG6_TRIAL],
        "ineffective_input": np.asarray(data["ineffective"])[FIG6_TRIAL],
        "good_inferred_rotated": rotated_good[FIG6_TRIAL],
        "bad_inferred_rotated": rotated_bad[FIG6_TRIAL],
    }
    trace_columns = {"time_bin": np.arange(next(iter(arrays.values())).shape[0])}
    for label, values in arrays.items():
        for channel in range(values.shape[1]):
            trace_columns[f"{label}_{channel + 1}"] = values[:, channel]
    columns_csv(destination / "panel_b_input_traces.csv", trace_columns)

    for label, run_name in (("good", good_name), ("bad", bad_name)):
        np.savez_compressed(
            destination / f"panel_{label}_fixed_points.npz",
            **figure6_projection(fps_data[run_name]),
        )
    json_file(
        destination / "metadata.json",
        {
            "metrics_source_cache": metrics_source.relative_to(cache_root).as_posix(),
            "fixed_points_source_cache": fps_source.relative_to(cache_root).as_posix(),
            "accepted_cache_metric_columns": list(data["metrics"]),
            "good_model_index": FIG6_GOOD,
            "good_model_run_name": good_name,
            "bad_model_index": FIG6_BAD,
            "bad_model_run_name": bad_name,
            "trial_index": FIG6_TRIAL,
            "fixed_point_q_threshold": FIG6_Q,
            "plotted_trajectories": N_TRAJ,
            "note": (
                "This accepted-figure cache predates the later "
                "input_r2_true_to_inferred comparison column."
            ),
        },
    )


def s8_target(tt_dir):
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

    import torch

    wrapper = trusted_pickle(tt_dir / "model.pkl")
    datamodule = trusted_pickle(tt_dir / "datamodule_sim.pkl")
    tensors = datamodule.valid_ds.tensors
    with torch.no_grad():
        controlled = wrapper(tensors[0], tensors[1], tensors[6])["controlled"]
    array = controlled.detach().cpu().numpy()
    return array.reshape(-1, array.shape[-1])


def s8_filter(xstar, stable, trajectory_points):
    center = trajectory_points.mean(axis=0)
    rms = np.sqrt(np.mean(np.sum((trajectory_points - center) ** 2, axis=1)))
    nearest = np.array(
        [np.min(np.linalg.norm(trajectory_points - point, axis=1)) for point in xstar]
    )
    keep = nearest <= S8_TRAJ_FRAC * rms
    xstar, stable = xstar[keep], stable[keep]

    clusters = []
    for index, point in enumerate(xstar):
        for cluster in clusters:
            if np.linalg.norm(point - xstar[cluster[0]]) <= S8_MERGE_FRAC * rms:
                cluster.append(index)
                break
        else:
            clusters.append([index])
    if not clusters:
        return np.empty((0, xstar.shape[1])), np.empty(0, dtype=bool)
    merged = np.array([xstar[cluster].mean(axis=0) for cluster in clusters])
    merged_stable = np.array(
        [stable[cluster].mean() >= 0.5 for cluster in clusters], dtype=bool
    )
    return merged, merged_stable


def s8(cache_root, tt_dir, output):
    source_dir = cache_root / "FigDSA"
    destination = output / "s8"
    destination.mkdir(parents=True, exist_ok=True)
    scatter = trusted_pickle(source_dir / "make_figure_dsa.scatter.pkl")
    rows_csv(
        destination / "state_rate_scatter.csv",
        ["family", "latent_size", "seed", "state_r2", "rate_r2"],
        scatter["points"],
    )

    dsa = trusted_pickle(source_dir / "make_figure_dsa.dsa.pkl")
    matrix_rows = []
    for family, entry in dsa.items():
        for row_index, row_label in enumerate(entry["labels"]):
            for column_index, column_label in enumerate(entry["labels"]):
                matrix_rows.append(
                    {
                        "family": family,
                        "row_model": row_label,
                        "column_model": column_label,
                        "dissimilarity": entry["matrix"][row_index, column_index],
                    }
                )
    rows_csv(
        destination / "dsa_matrices.csv",
        ["family", "row_model", "column_model", "dissimilarity"],
        matrix_rows,
    )

    target = s8_target(tt_dir)
    fps_data = trusted_pickle(source_dir / "make_figure_dsa.fps.pkl")
    counts = {}
    for model in S8_MODELS:
        entry = fps_data[model]
        latents = np.asarray(entry["latents"])
        flat = latents.reshape(-1, latents.shape[-1])
        fixed = entry["fps"]
        keep = np.asarray(fixed.qstar) < S8_Q
        xstar = np.asarray(fixed.xstar)[keep]
        stable = np.asarray(fixed.is_stable)[keep]
        indices = np.linspace(0, flat.shape[0] - 1, min(4000, len(flat))).astype(int)
        xstar, stable = s8_filter(xstar, stable, flat[indices])
        if target.shape[0] != flat.shape[0]:
            raise ValueError(
                f"S8 alignment target={target.shape[0]}, {model}={flat.shape[0]}"
            )
        regression = LinearRegression().fit(flat, target)
        projected_latents = regression.predict(flat).reshape(
            *latents.shape[:2], target.shape[1]
        )
        projected_fixed = (
            regression.predict(xstar) if len(xstar) else np.empty((0, target.shape[1]))
        )
        np.savez_compressed(
            destination / f"{model.lower()}_fixed_points.npz",
            trajectories=projected_latents[:N_TRAJ],
            fixed_points=projected_fixed,
            is_stable=stable,
            regression_coef=regression.coef_,
            regression_intercept=regression.intercept_,
        )
        counts[model] = len(projected_fixed)
    json_file(
        destination / "metadata.json",
        {
            "task_trained_alignment_source": tt_dir.name,
            "models": list(S8_MODELS),
            "fixed_point_q_threshold": S8_Q,
            "trajectory_distance_fraction": S8_TRAJ_FRAC,
            "merge_distance_fraction": S8_MERGE_FRAC,
            "plotted_trajectories": N_TRAJ,
            "exported_fixed_point_counts": counts,
        },
    )


def s9(cache_root, output):
    source = (
        cache_root / "supplementary" / "make_nl_cycle_consistency_node_sweep.cache.pkl"
    )
    data = trusted_pickle(source)
    summary, noise, training = [], [], []
    for index, run_name in enumerate(data["run_names"]):
        common = {
            "run_name": run_name,
            "latent_size": data["latent_sizes"][index],
            "seed": data["seeds"][index],
        }
        summary.append(
            {
                **common,
                "val_r2": data["val_r2"][index],
                "linear_cycle_consistency": data["linear_cc"][index],
                "best_epoch": data["best_epochs"][index],
            }
        )
        noise.extend(
            {**common, "noise_std": std, "r2": r2}
            for std, r2 in zip(data["noise_stds"], data["r2_per_noise"][index])
        )
        training.extend(
            {
                **common,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            for epoch, (train_loss, validation_loss) in enumerate(
                zip(data["train_curves"][index], data["val_curves"][index])
            )
        )
    common_fields = ["run_name", "latent_size", "seed"]
    rows_csv(
        output / "s9" / "model_summary.csv",
        common_fields + ["val_r2", "linear_cycle_consistency", "best_epoch"],
        summary,
    )
    rows_csv(
        output / "s9" / "noise_curves.csv",
        common_fields + ["noise_std", "r2"],
        noise,
    )
    rows_csv(
        output / "s9" / "inverse_training_curves.csv",
        common_fields + ["epoch", "train_loss", "validation_loss"],
        training,
    )


def compiled_metrics(cache_root, output):
    source_dir = cache_root / "supplementary" / "compiled_metrics_cache"
    for label in ("NBFF", "MultiTask", "RandomTarget", "PCM", "CDM"):
        source = source_dir / f"{label}_compiled_metrics.pkl"
        data = trusted_pickle(source)
        columns_csv(output / "s11-s15" / f"{label}_metrics.csv", data["metrics"])
        json_file(
            output / "s11-s15" / f"{label}_metadata.json",
            {
                "source_cache": source.relative_to(cache_root).as_posix(),
                "task_label": data.get("task_label", label),
                "task_trained_lyapunov_mean": data.get("lex_tt_mean0"),
            },
        )


def supporting_tables(source_dir, output):
    if source_dir is None:
        return
    destination = output / "supporting_tables"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("S*_Table.csv")):
        shutil.copy2(source, destination / source.name)


def checksums(output):
    checksum_path = output / "SHA256SUMS.txt"
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(output).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    figure5(args.cache_root, args.output)
    figure6(args.cache_root, args.output)
    s8(args.cache_root, args.tt_dir, args.output)
    s9(args.cache_root, args.output)
    compiled_metrics(args.cache_root, args.output)
    supporting_tables(args.si_dir, args.output)
    for name in (
        "DATA_LICENSE.md",
        "DATA_MANIFEST.tsv",
        "README.md",
        "requirements-paper.txt",
        "constraints-paper.txt",
    ):
        shutil.copy2(
            ROOT / "paper_reproduction" / name,
            args.output / name,
        )
    shutil.copy2(ROOT / "DATA_LICENSE", args.output / "DATA_LICENSE")
    checksums(args.output)
    print(f"Wrote portable publication data to {args.output}")


if __name__ == "__main__":
    main()
