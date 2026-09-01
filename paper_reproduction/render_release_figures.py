#!/usr/bin/env python3
"""Render publication figures from the portable CtDToolkit data deposit.

This program intentionally accepts only the CSV, JSON, and NPZ files written by
``export_release_data.py``, ``export_large_cache_data.py``, and
``export_figure4_data.py``. It never reads trained models or pickle caches.

The accepted manuscript contains manually assembled schematic panels in Figures
4 and 6. For those figures this program renders the deposited numerical panels;
the external schematic is represented by the same placeholder used by the
analysis script. Figure 4's learning-progression panel is emitted separately
from its B--D latent-trajectory grid.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = REPO_ROOT / "examples" / "figures"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Extracted root of the portable publication-data deposit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        help=("Subset of fig4 fig5 fig6 s4 s5 s8 s9 s11-s15 s16, or all " "(default)."),
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Do not verify SHA256SUMS.txt before plotting.",
    )
    return parser.parse_args(argv)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_columns(path: Path, string_columns=()) -> dict[str, np.ndarray]:
    rows = read_rows(path)
    if not rows:
        return {}
    output = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in string_columns:
            output[key] = np.asarray(values)
        else:
            output[key] = np.asarray(values, dtype=float)
    return output


def save_figure(fig, output_dir: Path, stem: str, dpi: int, exts=("pdf", "png")):
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in exts:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        written.append(path)
    plt.close(fig)
    return written


def verify_checksums(root: Path) -> int:
    checksum_file = root / "SHA256SUMS.txt"
    if not checksum_file.is_file():
        raise FileNotFoundError(checksum_file)
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def import_module_safely(name: str, path: Path):
    old_argv = sys.argv
    old_path = list(sys.path)
    sys.argv = [str(path)]
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


def exec_source_before(path: Path, marker: str, name: str):
    source = path.read_text(encoding="utf-8")
    if marker not in source:
        raise RuntimeError(f"Marker not found in {path}: {marker}")
    namespace = {"__file__": str(path), "__name__": name}
    old_argv = sys.argv
    old_path = list(sys.path)
    sys.argv = [str(path)]
    sys.path.insert(0, str(path.parent))
    try:
        exec(compile(source.split(marker, 1)[0], str(path), "exec"), namespace)
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    return namespace


def extract_functions(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    missing = names - {node.name for node in functions}
    if missing:
        raise RuntimeError(f"Missing functions in {path}: {sorted(missing)}")
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)


def clean_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def render_figure4(root: Path, output: Path, dpi: int):
    data_dir = root / "figure4"
    with np.load(data_dir / "panel_a_learning_progression.npz") as data:
        epochs = data["epochs"]
        inputs = data["inputs"]
        outputs = data["outputs"]
    fig, axes = plt.subplots(len(epochs), 2, figsize=(6.5, 7.2), sharex=True)
    for row, epoch in enumerate(epochs):
        axes[row, 0].plot(inputs[row], lw=0.8)
        axes[row, 1].plot(outputs[row], lw=0.8)
        axes[row, 0].set_ylabel(f"epoch {int(epoch)}")
        for ax in axes[row]:
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Task inputs")
    axes[0, 1].set_title("TT controlled outputs")
    axes[-1, 0].set_xlabel("Time bin")
    axes[-1, 1].set_xlabel("Time bin")
    fig.tight_layout()
    written = save_figure(fig, output, "Figure4_A_learning_progression", dpi)

    with np.load(data_dir / "panel_b_3bff_trajectories.npz") as data:
        b_names = data["model_names"].astype(str)
        b_traj = data["trajectories"]
        b_ranges = data["shared_axis_ranges"]
    c_rows = read_rows(data_dir / "panel_c_multitask_trajectories.csv")
    c_by_model = {}
    for model in b_names:
        model_rows = [row for row in c_rows if row["model"] == model]
        trials = []
        bins = []
        for trial in sorted({int(row["trial_index"]) for row in model_rows}):
            rows = sorted(
                (row for row in model_rows if int(row["trial_index"]) == trial),
                key=lambda row: int(row["time_index"]),
            )
            trials.append(
                np.asarray([[float(row[f"pc{k}"]) for k in (1, 2, 3)] for row in rows])
            )
            bins.append(int(rows[0]["response_angle_bin"]))
        c_by_model[model] = (trials, bins)
    with np.load(data_dir / "panel_d_random_target_trajectories.npz") as data:
        d_names = data["model_names"].astype(str)
        d_bins = data["reach_angle_bins"]
        d_traj = data["trajectories"]

    fig = plt.figure(figsize=(12, 10))
    for col, model in enumerate(b_names):
        ax = fig.add_subplot(3, 4, col + 1, projection="3d")
        for trial, color in ((2, "lightgray"), (1, "gray"), (0, "black")):
            xyz = b_traj[col, trial]
            ax.plot(*xyz.T, color=color, linewidth=2.5)
        centers = b_ranges.mean(axis=1)
        halves = (b_ranges[:, 1] - b_ranges[:, 0]) * 0.5 * 0.65
        ax.set_xlim(centers[0] - halves[0], centers[0] + halves[0])
        ax.set_ylim(centers[1] - halves[1], centers[1] + halves[1])
        ax.set_zlim(centers[2] - halves[2], centers[2] + halves[2])
        ax.view_init(45, 45)
        ax.set_title(model)
        clean_3d(ax)
        ax = fig.add_subplot(3, 4, 4 + col + 1, projection="3d")
        trials, bins = c_by_model[model]
        for xyz, color_bin in zip(trials, bins):
            ax.plot(*xyz.T, color=plt.cm.jet(color_bin / 8), alpha=0.5)
        clean_3d(ax)
        ax = fig.add_subplot(3, 4, 8 + col + 1, projection="3d")
        d_index = list(d_names).index(model)
        for xyz, color_bin in zip(d_traj[d_index], d_bins):
            ax.plot(*xyz.T, color=plt.cm.jet(color_bin / 8), alpha=0.5)
        ax.view_init(-32, 128)
        clean_3d(ax)
    for row, label in enumerate(("3BFF", "MultiTask (MemoryPro)", "RandomTarget")):
        fig.text(0.02, 1 - (row + 0.5) / 3, label, rotation=90, va="center")
    fig.tight_layout(rect=(0.04, 0, 1, 1))
    written += save_figure(fig, output, "Figure4_BCD_latent_trajectories", dpi)
    return written


def render_figure5(root: Path, output: Path, dpi: int):
    script = FIGURE_ROOT / "Fig5Metrics" / "make_figure5_reconstruction_simplicity.py"
    ns = exec_source_before(script, 'if __name__ == "__main__":', "release_figure5")
    data_dir = root / "figure5"
    metadata = read_json(data_dir / "metadata.json")
    metrics = numeric_columns(data_dir / "metrics.csv", ("run_name", "group"))
    metric_rows = {}
    for latent in ns["CALLOUT_LATENTS"]:
        indices = np.where(metrics["latent_size"].astype(int) == latent)[0]
        metric_rows[latent] = int(
            indices[metadata["cache_config"]["latent_indices"][str(latent)]]
        )
    rec_csv = numeric_columns(data_dir / "panel_b_reconstruction.csv")
    reconstruction = {
        "trial": metadata["cache_config"]["trial"],
        "neuron": metadata["cache_config"]["neuron"],
        "time": rec_csv["time_bin"],
        "true": rec_csv["true_rate"],
        "spikes": rec_csv["spike_count"],
    }
    for latent in (2, 8):
        reconstruction[latent] = {
            "pred": rec_csv[f"node_{latent}_predicted_rate"],
            **metadata["reconstruction_annotations"][str(latent)],
        }
    simp_csv = numeric_columns(data_dir / "panel_e_simplicity.csv")
    simplicity = {
        "time": simp_csv["time_bin"],
        "pc_index": metadata["cache_config"]["pc_index"],
        "demean": metadata["cache_config"].get("demean", False),
    }
    for latent in (8, 32):
        simplicity[latent] = {
            key: simp_csv[f"node_{latent}_{key}"]
            for key in ("actual", "state_pred", "cycle_pred")
        }
        simplicity[latent].update(metadata["simplicity_annotations"][str(latent)])
    payload = {
        "config": metadata["cache_config"],
        "metrics": metrics,
        "metric_rows": metric_rows,
        "reconstruction": reconstruction,
        "simplicity": simplicity,
        "noiseless": metadata["cache_config"].get("noiseless", False),
        "demean": metadata["cache_config"].get("demean", False),
    }
    paths = ns["make_figure"](payload, output, dpi)
    plt.close("all")
    return list(paths)


def plot_projected_fps(ax, fps, trajectories, scale=1.0, **_):
    for trajectory in trajectories:
        ax.plot(*trajectory.T, color="k", linewidth=0.5 * scale, alpha=0.55)
    stable = fps.xstar[fps.is_stable]
    unstable = fps.xstar[~fps.is_stable]
    if len(stable):
        ax.scatter(*stable.T, c="g", marker="o", s=30 * scale * scale)
    if len(unstable):
        ax.scatter(
            *unstable.T,
            c="r",
            marker="x",
            s=40 * scale * scale,
            linewidths=1.5 * scale,
        )


def render_figure6(root: Path, output: Path, dpi: int):
    script = FIGURE_ROOT / "Fig6InputInf" / "make_figure6_combined.py"
    ns = {
        "np": np,
        "plt": plt,
        "gridspec": gridspec,
        "Path": Path,
        "Line2D": Line2D,
        "GOOD_COLOR": "tab:orange",
        "BAD_COLOR": "tab:cyan",
        "Q_THRESH_GOOD": 1e-5,
        "Q_THRESH_BAD": 1e-5,
        "NUM_TRAJ": 10,
        "VIEW_BAD": (30, -10),
        "VIEW_GOOD": (30, 0),
        "STATE_PALETTE": ["#90EE90", "#2ca02c", "#006400"],
        "EFFECTIVE_PALETTE": ["#A0A0A0", "#606060", "#202020"],
        "INFERRED_PALETTE": ["#FFA0A0", "#d62728", "#8B0000"],
    }
    extract_functions(script, {"_clean_3d", "_format_kl_label", "make_figure"}, ns)
    ns["_plot_fps_on_ax"] = plot_projected_fps
    data_dir = root / "figure6"
    metadata = read_json(data_dir / "metadata.json")
    metrics = numeric_columns(data_dir / "metrics.csv", ("run_name", "group"))
    run_names = metrics["run_name"].astype(str)
    kl_scales = np.asarray(
        [float(name.split("co_kl_")[1].rsplit("_", 1)[0]) for name in run_names]
    )
    traces = numeric_columns(data_dir / "panel_b_input_traces.csv")

    def channels(prefix):
        return np.column_stack([traces[f"{prefix}_{i}"] for i in (1, 2, 3)])[None]

    payload = {
        "metrics": metrics,
        "kl_co_scales": kl_scales,
        "best_inp": int(np.where(run_names == metadata["good_model_run_name"])[0][0]),
        "bad_inp": int(np.where(run_names == metadata["bad_model_run_name"])[0][0]),
        "controlled": channels("controlled"),
        "true_inputs": channels("true_input"),
        "effective": channels("effective_input"),
        "ineffective": channels("ineffective_input"),
        "rot_best": channels("good_inferred_rotated"),
        "rot_bad": channels("bad_inferred_rotated"),
    }
    for label, key in (("best", "good"), ("bad", "bad")):
        with np.load(data_dir / f"panel_{key}_fixed_points.npz") as data:
            payload[f"latents_{label}"] = data["trajectories"]
            payload[f"fps_{label}"] = SimpleNamespace(
                xstar=data["fixed_points"],
                qstar=np.zeros(len(data["fixed_points"])),
                is_stable=data["is_stable"],
            )
    path = ns["make_figure"](payload, 0, output)
    plt.close("all")
    return [path, output / "figure6_AF_combined.png"]


def reconstruct_flat(rows, system: str):
    selected = [row for row in rows if row["system"] == system]
    neurons = sorted({int(row["neuron"]) for row in selected})
    result = {}
    for stimulus in ("A", "B"):
        stim_rows = [row for row in selected if row["stimulus"] == stimulus]
        samples = sorted({int(row["sample_index"]) for row in stim_rows})
        matrix = np.full((len(samples), max(neurons) + 1), np.nan)
        phase = np.zeros(len(samples))
        sample_to_index = {sample: index for index, sample in enumerate(samples)}
        for row in stim_rows:
            index = sample_to_index[int(row["sample_index"])]
            matrix[index, int(row["neuron"])] = float(row["normalized_activity"])
            phase[index] = float(row["phase"])
        result[stimulus.lower()] = matrix
        result[f"theta_{stimulus.lower()}"] = phase
    return result


def render_s4(root: Path, output: Path, dpi: int):
    module = import_module_safely(
        "release_s4_module",
        FIGURE_ROOT / "supplementary" / "make_phase_coded_memory_figure.py",
    )
    data_dir = root / "s4"
    metadata = read_json(data_dir / "metadata.json")
    unit_rows = read_rows(data_dir / "single_unit_activity.csv")
    with np.load(data_dir / "pca_trajectories.npz") as pca, np.load(
        data_dir / "neural_data_trial.npz"
    ) as neural:
        fig = plt.figure(figsize=(10.0, 6.1), constrained_layout=False)
        gs = GridSpec(
            2,
            6,
            figure=fig,
            height_ratios=[0.75, 1.0],
            hspace=0.34,
            wspace=0.62,
        )
        for label, title, spec in (
            ("A", "Task schematic (external artwork)", gs[0, :2]),
            ("B", "Example TT inputs and outputs (assembled panel)", gs[0, 2:]),
        ):
            ax = fig.add_subplot(spec)
            module.panel_label(ax, label)
            ax.text(0.5, 0.5, title, ha="center", va="center", color="0.45")
            ax.set_axis_off()

        ax_c = fig.add_subplot(gs[1, :2], projection="3d")
        module.panel_label(ax_c, "C")
        module.plot_pca_panel(
            ax_c,
            pca["tt_latents"],
            pca["extras"],
            pca["is_stim_a"],
            pca["is_stim_b"],
            "TT Latents",
            module.TT_GREEN,
            metadata["max_pca_trials"],
        )
        ax_d = fig.add_subplot(gs[1, 2:4])
        module.plot_unit_grid(
            ax_d,
            reconstruct_flat(unit_rows, "tt"),
            module.TT_NEURONS,
            "D",
            module.TT_GREEN,
            1,
        )
        ax_d.set_title("Example Single Unit Activity", pad=3)
        ax_e = fig.add_subplot(gs[1, 4:])
        module.plot_neural_panel(
            ax_e,
            neural["true_rates"][None],
            neural["spikes"][None],
            0,
            inf_rates=None,
        )
    fig.tight_layout()
    return save_figure(fig, output, "S4_Fig_deposit_reconstruction", dpi)


def render_s5(root: Path, output: Path, dpi: int):
    script = FIGURE_ROOT / "supplementary" / "make_chaotic_delayed_memory_figure.py"
    source = script.read_text(encoding="utf-8")
    prefix = source.split("_cached_payload = _read_cached_payload", 1)[0]
    panel_start = source.index("def panel_task_schematic")
    panel_stop = source.index("# --------------------------- Compose figure")
    ns = {"__file__": str(script), "__name__": "release_s5"}
    old_path = list(sys.path)
    sys.path.insert(0, str(script.parent))
    try:
        exec(
            compile(
                prefix + "\n" + source[panel_start:panel_stop], str(script), "exec"
            ),
            ns,
        )
    finally:
        sys.path[:] = old_path
    data_dir = root / "s5"
    metadata = read_json(data_dir / "metadata.json")
    with np.load(data_dir / "task_and_output_trial.npz") as task, np.load(
        data_dir / "task_trained_dynamics.npz"
    ) as dynamics:
        n_types = len(dynamics["cue_pair_trial_indices"])
        n_trials = n_types + 1
        time = task["inputs"].shape[0]
        example = n_trials - 1
        inputs = np.zeros((n_trials, time, task["inputs"].shape[1]), dtype=float)
        targets = np.zeros((n_trials, time, task["targets"].shape[1]), dtype=float)
        controlled = np.zeros_like(targets)
        extra = np.zeros((n_trials, task["phase_metadata"].shape[0]), dtype=float)
        inputs[example], targets[example], controlled[example] = (
            task["inputs"],
            task["targets"],
            task["model_output"],
        )
        extra[:n_types] = dynamics["cue_pair_metadata"]
        extra[example] = task["phase_metadata"]
        latents = np.zeros((n_trials, time, 3), dtype=float)
        latents[:n_types] = dynamics["cue_pair_latent_pcs"]
        latents[example] = dynamics["baseline_ic_latent_pcs"]
        pert_out = np.zeros((len(task["perturbed_outputs"]), n_trials, time, 1))
        pert_out[:, example] = task["perturbed_outputs"]
        pert_pc = np.zeros(
            (len(dynamics["perturbed_ic_latent_pcs"]), n_trials, time, 3)
        )
        pert_pc[:, example] = dynamics["perturbed_ic_latent_pcs"]
        base_h0 = np.zeros((n_trials, 3))
        base_h0[example] = dynamics["baseline_initial_pc"]
        pert_h0 = np.zeros((len(dynamics["perturbed_initial_pcs"]), n_trials, 3))
        pert_h0[:, example] = dynamics["perturbed_initial_pcs"]
        latent_delta = np.zeros(
            (len(dynamics["latent_perturbation_norm"]), n_trials, time)
        )
        output_delta = np.zeros_like(latent_delta)
        latent_delta[:, example] = dynamics["latent_perturbation_norm"]
        output_delta[:, example] = dynamics["output_perturbation_norm"]
    ns.update(
        inputs_np=inputs,
        targets_np=targets,
        controlled_np=controlled,
        extra_np=extra,
        trial_idx=example,
        pert_out_np=pert_out,
        latents_pc=latents,
        ic_trial_idx=example,
        pert_pc=pert_pc,
        base_h0_pc=base_h0,
        pert_h0_pc=pert_h0,
        latent_delta=latent_delta,
        output_delta=output_delta,
        TT_LYAP_MAX=metadata["task_trained_lyapunov_max"],
        TT_LYAP_MAX_STD=metadata["task_trained_lyapunov_max_std"],
        dd_data={},
        dd_message="not used in S5 TT figure",
    )
    ns["configure_plotting"]()
    fig = plt.figure(figsize=(14.5, 9.6), constrained_layout=False)
    outer = GridSpec(
        4,
        4,
        figure=fig,
        height_ratios=[0.58, 0.82, 0.92, 1.25],
        hspace=0.7,
        wspace=0.45,
    )
    ax_task = fig.add_subplot(outer[0, :2])
    ax_io = fig.add_subplot(outer[1, :2], sharex=ax_task)
    ns["panel_task_schematic"](ax_task, title="Task structure", title_kwargs={})
    ns["panel_io"](ax_io, title="Example inputs and target output")
    ax_lat = fig.add_subplot(outer[0:3, 2:])
    ns["panel_four_latent_trajectories"](
        ax_lat, title="TT latent trajectories (one trial per type)"
    )
    ax_perf = fig.add_subplot(outer[2, :2], sharex=ax_task)
    ns["panel_unperturbed"](ax_perf, title="Perturbed trajectories in output space")
    ax_ic2 = fig.add_subplot(outer[3, 0])
    ax_ic3 = fig.add_subplot(outer[3, 1], projection="3d")
    ax_growth = fig.add_subplot(outer[3, 2:])
    ns["panel_tt_pc_perturbations"](
        ax_ic2, ax_ic3, title="Perturbed trajectories in top 2 PCs"
    )
    ns["panel_tt_perturbation_growth"](
        ax_growth, title="Perturbation expansion over time"
    )
    fig.align_ylabels()
    for ax, letter in (
        (ax_task, "A"),
        (ax_lat, "B"),
        (ax_perf, "C"),
        (ax_ic2, "D"),
        (ax_growth, "E"),
    ):
        ns["add_panel_label"](fig, ax, letter)
    return save_figure(fig, output, "S5_Fig", dpi)


def render_s8(root: Path, output: Path, dpi: int):
    script = FIGURE_ROOT / "supplementary" / "make_figure_dsa.py"
    ns = exec_source_before(script, "dsa_payload = build_dsa_payload", "release_s8")
    data_dir = root / "s8"
    rows = read_rows(data_dir / "dsa_matrices.csv")
    dsa = {}
    for family in ("NODE", "GRU"):
        family_rows = [row for row in rows if row["family"] == family]
        labels = list(dict.fromkeys(row["row_model"] for row in family_rows))
        matrix = np.zeros((len(labels), len(labels)))
        index = {label: i for i, label in enumerate(labels)}
        for row in family_rows:
            matrix[index[row["row_model"]], index[row["column_model"]]] = float(
                row["dissimilarity"]
            )
        dsa[family] = {"labels": labels, "matrix": matrix}
    fps = {}
    for model in ("NODE8", "NODE64", "GRU8", "GRU64"):
        with np.load(data_dir / f"{model.lower()}_fixed_points.npz") as data:
            fps[model] = {
                "fps": SimpleNamespace(
                    xstar=data["fixed_points"],
                    qstar=np.zeros(len(data["fixed_points"])),
                    is_stable=data["is_stable"],
                ),
                "latents": data["trajectories"],
            }
    ns["_plot_fps_on_ax"] = plot_projected_fps
    ns["MANUSCRIPT_FIG_PATH"] = output / "S8_Fig.pdf"
    ns["make_figure"](dsa, fps, output, dpi)
    source_pdf = output / "figureDSA_ABCD.pdf"
    source_png = output / "figureDSA_ABCD.png"
    shutil.copy2(source_pdf, output / "S8_Fig.pdf")
    shutil.copy2(source_png, output / "S8_Fig.png")
    plt.close("all")
    return [output / "S8_Fig.pdf", output / "S8_Fig.png"]


def render_s9(root: Path, output: Path, dpi: int):
    module = import_module_safely(
        "release_s9_module",
        FIGURE_ROOT / "supplementary" / "make_nl_cycle_consistency_node_sweep.py",
    )
    data_dir = root / "s9"
    summary = read_rows(data_dir / "model_summary.csv")
    noise = read_rows(data_dir / "noise_curves.csv")
    training = read_rows(data_dir / "inverse_training_curves.csv")
    run_names = [row["run_name"] for row in summary]
    payload = {
        "run_names": np.asarray(run_names),
        "latent_sizes": np.asarray([int(row["latent_size"]) for row in summary]),
        "seeds": np.asarray([int(row["seed"]) for row in summary]),
        "val_r2": np.asarray([float(row["val_r2"]) for row in summary]),
        "linear_cc": np.asarray(
            [float(row["linear_cycle_consistency"]) for row in summary]
        ),
        "best_epochs": np.asarray([int(row["best_epoch"]) for row in summary]),
    }
    noise_stds = sorted({float(row["noise_std"]) for row in noise})
    payload["noise_stds"] = np.asarray(noise_stds)
    payload["r2_per_noise"] = np.asarray(
        [
            [
                float(
                    next(
                        row["r2"]
                        for row in noise
                        if row["run_name"] == run and float(row["noise_std"]) == level
                    )
                )
                for level in noise_stds
            ]
            for run in run_names
        ]
    )
    payload["train_curves"] = np.asarray(
        [
            np.asarray(
                [
                    float(row["train_loss"])
                    for row in sorted(
                        (r for r in training if r["run_name"] == run),
                        key=lambda r: int(r["epoch"]),
                    )
                ]
            )
            for run in run_names
        ],
        dtype=object,
    )
    payload["val_curves"] = np.asarray(
        [
            np.asarray(
                [
                    float(row["validation_loss"])
                    for row in sorted(
                        (r for r in training if r["run_name"] == run),
                        key=lambda r: int(r["epoch"]),
                    )
                ]
            )
            for run in run_names
        ],
        dtype=object,
    )
    args = Namespace(noise_scatter_ind=2, task="3bff")
    fig = module.build_figure(payload, args)
    return save_figure(fig, output, "S9_Fig", dpi, exts=("pdf", "svg", "png"))


def parse_lyapunov(value: str):
    return np.fromstring(value.strip().strip("[]"), sep=" ")


def render_s11_s15(root: Path, output: Path, dpi: int):
    module = import_module_safely(
        "release_compiled_metrics_module",
        FIGURE_ROOT / "supplementary" / "make_compiled_metrics_vs_latent_size.py",
    )
    written = []
    data_dir = root / "s11-s15"
    tasks = (
        ("NBFF", "nbff", "S11_Fig"),
        ("MultiTask", "multitask", "S12_Fig"),
        ("RandomTarget", "randomtarget", "S13_Fig"),
        ("PCM", "pcm", "S14_Fig"),
        ("CDM", "cdm", "S15_Fig"),
    )
    for label, key, manuscript_stem in tasks:
        rows = read_rows(data_dir / f"{label}_metrics.csv")
        metrics = {}
        for column in rows[0]:
            if column in ("run_name", "group"):
                metrics[column] = [row[column] for row in rows]
            elif column == "lyapunov":
                metrics[column] = [parse_lyapunov(row[column]) for row in rows]
            else:
                metrics[column] = [float(row[column]) for row in rows]
        metadata = read_json(data_dir / f"{label}_metadata.json")
        payload = {
            "metrics": metrics,
            "lex_tt_mean0": metadata["task_trained_lyapunov_mean"],
        }
        paths = module.plot_task_metrics(
            payload, module.TASKS[key], output, exts=("pdf", "png"), show=False
        )
        for path in paths:
            destination = output / f"{manuscript_stem}{path.suffix}"
            shutil.copy2(path, destination)
            written.append(destination)
    return written


def render_s16(root: Path, output: Path, dpi: int):
    module = import_module_safely(
        "make_phase_coded_memory_simple",
        FIGURE_ROOT / "supplementary" / "make_phase_coded_memory_simple.py",
    )
    data_dir = root / "s16"
    metadata = read_json(data_dir / "metadata.json")
    with np.load(data_dir / "pca_trajectories.npz") as pca, np.load(
        data_dir / "firing_rate_rasters.npz"
    ) as rasters:
        payload = {
            "extras": pca["extras"],
            "inds_a": pca["is_stim_a"],
            "inds_b": pca["is_stim_b"],
            "true_rates": rasters["ground_truth"][None],
            "gt": {
                "rates_pca": pca["ground_truth_rate_pcs"],
                "lats_pca": pca["ground_truth_latent_pcs"],
            },
        }
        for label, key in (("dd8", "dd8"), ("dd", "dd16"), ("dd128", "dd128")):
            payload[label] = {
                "inf_rates": rasters[key][None],
                "rates_pca": pca[f"{key}_rate_pcs"],
                "lats_pca": pca[f"{key}_latent_pcs"],
                **metadata["model_metrics"][key],
            }
    args = Namespace(trial=0, max_pca_trials=metadata["max_pca_trials"])
    fig = module.build_figure(payload, args)
    return save_figure(fig, output, "S16_Fig", dpi)


RENDERERS = {
    "fig4": render_figure4,
    "fig5": render_figure5,
    "fig6": render_figure6,
    "s4": render_s4,
    "s5": render_s5,
    "s8": render_s8,
    "s9": render_s9,
    "s11-s15": render_s11_s15,
    "s16": render_s16,
}


def main(argv=None):
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not args.skip_checksums:
        count = verify_checksums(data_root)
        print(f"Verified {count} deposited-file checksums.")
    requested = list(RENDERERS) if args.figures == ["all"] else args.figures
    unknown = set(requested) - set(RENDERERS)
    if unknown:
        raise ValueError(f"Unknown figure groups: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in requested:
        print(f"Rendering {name} from portable data ...", flush=True)
        paths = RENDERERS[name](data_root, output, args.dpi)
        results[name] = [str(Path(path).resolve()) for path in paths]
    report = {
        "data_root": str(data_root),
        "rendered_groups": requested,
        "files": results,
        "note": (
            "Figures 4 and 6 contain external schematic artwork in the accepted "
            "manuscript; this command regenerates their deposited numerical panels."
        ),
    }
    report_path = output / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
