"""Generate the PhaseCodedMemory supplemental figure.

This script is a runnable, notebook-friendly replacement for the exploratory
``FigurePhaseCodedMemory.ipynb`` notebook. It leaves panels A and B blank by
default and generates manuscript-oriented panels C-H from the saved TT and LFADS
analyses.
"""

# %%
from __future__ import annotations

import argparse
import os
import pickle
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "libs" / "DSA", REPO_ROOT / "libs" / "lfads-jslds"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import dotenv
except ImportError:
    dotenv = types.ModuleType("dotenv")
    dotenv.find_dotenv = lambda *args, **kwargs: str(REPO_ROOT / ".env")
    dotenv.load_dotenv = lambda *args, **kwargs: False
    sys.modules["dotenv"] = dotenv

if "DSA" not in sys.modules:
    dsa_stub = types.ModuleType("DSA")

    class _UnavailableDSA:
        def __init__(self, *args, **kwargs):
            raise ImportError("DSA metrics require optional kooplearn dependencies.")

    dsa_stub.DSA = _UnavailableDSA
    sys.modules["DSA"] = dsa_stub

try:
    import lightning_fabric.utilities.data as _lf_data

    if not hasattr(_lf_data, "AttributeDict"):

        class AttributeDict(dict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        _lf_data.AttributeDict = AttributeDict
except ImportError:
    pass

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.init = lambda *args, **kwargs: None
    wandb_stub.log = lambda *args, **kwargs: None
    wandb_stub.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub

try:
    import packaging.version as _packaging_version

    pkg_vendor = types.ModuleType("pkg_resources._vendor")
    pkg_packaging = types.ModuleType("pkg_resources._vendor.packaging")
    pkg_packaging.version = _packaging_version
    sys.modules.setdefault("pkg_resources._vendor", pkg_vendor)
    sys.modules.setdefault("pkg_resources._vendor.packaging", pkg_packaging)
    sys.modules.setdefault(
        "pkg_resources._vendor.packaging.version", _packaging_version
    )
except ImportError:
    pass

try:
    import pytorch_lightning.utilities.model_helpers as _pl_model_helpers

    if not hasattr(_pl_model_helpers, "_ModuleMode"):

        class _ModuleMode:
            pass

        _pl_model_helpers._ModuleMode = _ModuleMode
except ImportError:
    pass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.decomposition import PCA

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

# %%
STIM_A = "#8f2f8f"
STIM_B = "#5dcde0"
TT_GREEN = "#3bbf78"
DD_RED = "#ff3b3f"
HIST_TT = "#72b58e"
HIST_DD = "#ff8b8f"
HIST_TT_GREY = "#9a9a9a"
MEDIAN_TT_LINE = "#404040"
MEDIAN_DD_LINE = DD_RED

DEFAULT_DD_SUBPATH = (
    "20260528_PCM_LFADS_RateBiasInit_DimSweep_NewScale/"
    "prefix=tt_PhaseCodedMemory_max_epochs=200_gen_dim=16_seed=0"
)
DEFAULT_DD128_SUBPATH = (
    "20260528_PCM_LFADS_RateBiasInit_DimSweep_NewScale/"
    "prefix=tt_PhaseCodedMemory_max_epochs=200_gen_dim=128_seed=0"
)
DEFAULT_DD8_SUBPATH = (
    "20260528_PCM_LFADS_RateBiasInit_DimSweep_NewScale/"
    "prefix=tt_PhaseCodedMemory_max_epochs=200_gen_dim=8_seed=0"
)
TT_NEURONS = [1, 2, 3, 10, 11, 14]
DD_NEURONS = [1, 3, 4, 9, 10, 12]
DD128_NEURONS = [1, 3, 4, 9, 10, 12]
DD8_NEURONS = [1, 3, 4, 9, 10, 12]

# Additional DD models beyond the primary 16D model (whose arrays live under the
# unsuffixed keys lats_dd_pca / flat_dd / mean_delta_dd). Each entry is stored
# under the keys lats_<prefix>_pca / flat_<prefix> / mean_delta_<prefix>.
EXTRA_DD_MODELS = (
    {
        "prefix": "dd8",
        "path_attr": "dd8_path",
        "subpath_attr": "dd8_subpath",
        "dim_label": "8D",
    },
    {
        "prefix": "dd128",
        "path_attr": "dd128_path",
        "subpath_attr": "dd128_subpath",
        "dim_label": "128D",
    },
)
SUPPLEMENTARY_OUTPUT_DIR = (
    REPO_ROOT / "examples" / "figures" / "supplementary" / "outputs"
)
MANUSCRIPT_FIGS_DIR = REPO_ROOT / "manuscript" / "figs"

METRIC_BAR_KEYS = ("state_r2", "rate_r2", "cycle_con", "co-bps")
METRIC_BAR_LABELS = {
    "state_r2": "State R²",
    "rate_r2": "Rate R²",
    "cycle_con": "Cycle Con.",
    "co-bps": "co-bps",
}
METRIC_BAR_SPEC = {
    "state_r2": {},
    "rate_r2": {},
    "cycle_con": {"variance_threshold": 0.01},
    "co-bps": {},
}


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in {
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        }
    except ImportError:
        return False


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SUPPLEMENTARY_OUTPUT_DIR)
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=MANUSCRIPT_FIGS_DIR,
        help="Manuscript figs directory where the DD figure is copied so it loads automatically.",
    )
    parser.add_argument(
        "--cache", type=Path, default=Path(__file__).with_suffix(".cache.pkl")
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute extracted arrays instead of using the cache.",
    )
    parser.add_argument("--phase", default="val", choices=["train", "val", "all"])
    parser.add_argument("--tt-path", type=Path, default=None)
    parser.add_argument("--dd-path", type=Path, default=None)
    parser.add_argument("--dd-subpath", default=DEFAULT_DD_SUBPATH)
    parser.add_argument("--dd128-path", type=Path, default=None)
    parser.add_argument("--dd128-subpath", default=DEFAULT_DD128_SUBPATH)
    parser.add_argument("--dd8-path", type=Path, default=None)
    parser.add_argument("--dd8-subpath", default=DEFAULT_DD8_SUBPATH)
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial index for neural rates/spiking panel.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-pca-trials", type=int, default=240)
    parser.add_argument("--scatter-step", type=int, default=50)
    parser.add_argument(
        "--show-placeholders", action=argparse.BooleanOptionalAction, default=False
    )
    if argv is None and _in_notebook():
        argv = []
    return parser.parse_args(argv)


def as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_home_dir() -> Path:
    dotenv.load_dotenv(dotenv.find_dotenv())
    home = os.environ.get("HOME_DIR")
    if home is None:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("HOME_DIR"):
                    home = line.split("=", 1)[1].strip()
                    break
    if home is None:
        home = str(REPO_ROOT) + os.sep
    resolved = Path(home).expanduser().resolve()
    os.environ["HOME_DIR"] = str(resolved) + os.sep
    return resolved


def default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    home = get_home_dir()
    tt_path = (
        args.tt_path or home / "content/trained_models/task-trained/tt_PhaseCodedMemory"
    )
    if args.tt_path is None and not tt_path.exists():
        pretrained_tt = REPO_ROOT / "pretrained/PCM_NoisyGRU_Final"
        if pretrained_tt.exists():
            tt_path = pretrained_tt
    dd_path = args.dd_path or tt_path / args.dd_subpath
    return tt_path, dd_path


def extra_dd_path(args: argparse.Namespace, tt_path: Path, spec: dict) -> Path:
    """Resolve the directory for an EXTRA_DD_MODELS entry (override or subpath)."""
    override = getattr(args, spec["path_attr"])
    return override or tt_path / getattr(args, spec["subpath_attr"])


# %%
def compute_theta(signal: np.ndarray) -> np.ndarray:
    """Return phase angles in [0, 2*pi) for a sinusoidal reference signal."""
    signal = np.clip(signal, -1.0, 1.0)
    ds = np.gradient(signal, axis=1)
    principal = np.arcsin(signal)
    theta = np.empty_like(principal)

    q1 = (signal >= 0) & (ds >= 0)
    q2 = (signal >= 0) & (ds < 0)
    q3 = (signal < 0) & (ds < 0)
    q4 = (signal < 0) & (ds >= 0)
    theta[q1] = principal[q1]
    theta[q2] = np.pi - principal[q2]
    theta[q3] = np.pi - principal[q3]
    theta[q4] = 2 * np.pi + principal[q4]
    return theta


def pca_trajectories(latents: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=3)
    flat = latents.reshape(-1, latents.shape[-1])
    return pca.fit_transform(flat).reshape(latents.shape[0], latents.shape[1], 3)


def flatten_poststim(
    latents: np.ndarray,
    theta: np.ndarray,
    extras: np.ndarray,
    inds_a: np.ndarray,
    inds_b: np.ndarray,
) -> dict[str, np.ndarray]:
    flat_all, theta_all = [], []
    flat_a, flat_b = [], []
    theta_a, theta_b = [], []
    for trial in range(latents.shape[0]):
        start = int(extras[trial, 1])
        trial_lats = latents[trial, start:, :]
        trial_theta = theta[trial, start:]
        flat_all.append(trial_lats)
        theta_all.append(trial_theta)
        if inds_a[trial]:
            flat_a.append(trial_lats)
            theta_a.append(trial_theta)
        elif inds_b[trial]:
            flat_b.append(trial_lats)
            theta_b.append(trial_theta)
    return {
        "all": np.concatenate(flat_all, axis=0),
        "a": np.concatenate(flat_a, axis=0),
        "b": np.concatenate(flat_b, axis=0),
        "theta_all": np.concatenate(theta_all, axis=0),
        "theta_a": np.concatenate(theta_a, axis=0),
        "theta_b": np.concatenate(theta_b, axis=0),
    }


def normalize_by_all(flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = flat["all"].mean(axis=0)
    std = flat["all"].std(axis=0)
    std[std == 0] = 1.0
    return {
        "all": (flat["all"] - mean) / std,
        "a": (flat["a"] - mean) / std,
        "b": (flat["b"] - mean) / std,
        "theta_all": flat["theta_all"],
        "theta_a": flat["theta_a"],
        "theta_b": flat["theta_b"],
    }


def compute_dd_model_arrays(
    dd_path, phase, theta, extras, inds_a, inds_b, run_name
) -> dict:
    """Load a DD analysis and return its PCA trajectories, post-stim flat dict,
    and per-unit mean A-minus-B activity difference."""
    an = Analysis_DD.create(
        run_name=run_name, filepath=str(dd_path) + os.sep, model_type="LFADS"
    )
    lats = as_numpy(an.get_latents(phase=phase))
    flat = normalize_by_all(flatten_poststim(lats, theta, extras, inds_a, inds_b))
    return {
        "lats_pca": pca_trajectories(lats),
        "flat": flat,
        "mean_delta": flat["a"].mean(axis=0) - flat["b"].mean(axis=0),
    }


def load_or_compute_payload(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.cache.exists() and not args.force:
        with args.cache.open("rb") as f:
            cached = pickle.load(f)
        required_extra = [f"flat_{spec['prefix']}" for spec in EXTRA_DD_MODELS]
        if (
            "bar_metrics" in cached
            and "inf_rates" in cached
            and "rates_pca" in cached
            and all(k in cached for k in required_extra)
        ):
            return cached
        base_keys = (
            "bar_metrics",
            "inf_rates",
            "rates_pca",
            "theta",
            "extras",
            "inds_a",
            "inds_b",
        )
        missing = [s for s in EXTRA_DD_MODELS if f"flat_{s['prefix']}" not in cached]
        if all(k in cached for k in base_keys) and missing:
            # Only some extra-model additions are missing. Augment in place by
            # loading just those analyses and reusing the cached theta/extras/inds,
            # rather than recomputing the entire (multi-GB) payload from scratch.
            print(
                f"Augmenting cache with {[s['dim_label'] for s in missing]} DD model arrays.",
                flush=True,
            )
            tt_path, _ = default_paths(args)
            for spec in missing:
                dd_path = extra_dd_path(args, tt_path, spec)
                if not dd_path.exists():
                    raise FileNotFoundError(
                        f"Could not find {spec['dim_label']} DD analysis at {dd_path}. "
                        f"Set --{spec['prefix']}-path or --{spec['prefix']}-subpath."
                    )
                arrays = compute_dd_model_arrays(
                    dd_path,
                    args.phase,
                    cached["theta"],
                    cached["extras"],
                    cached["inds_a"],
                    cached["inds_b"],
                    run_name=spec["prefix"].upper(),
                )
                cached[f"lats_{spec['prefix']}_pca"] = arrays["lats_pca"]
                cached[f"flat_{spec['prefix']}"] = arrays["flat"]
                cached[f"mean_delta_{spec['prefix']}"] = arrays["mean_delta"]
            with args.cache.open("wb") as f:
                pickle.dump(cached, f)
            return cached
        print("Cache is missing new payload keys; recomputing.", flush=True)

    tt_path, dd_path = default_paths(args)
    if not tt_path.exists():
        raise FileNotFoundError(
            f"Could not find TT analysis at {tt_path}. Set --tt-path or HOME_DIR."
        )
    if not dd_path.exists():
        raise FileNotFoundError(
            f"Could not find DD analysis at {dd_path}. Set --dd-path or --dd-subpath."
        )

    an_tt = Analysis_TT(
        run_name="TT", filepath=str(tt_path) + os.sep, use_train_dm=True
    )
    an_dd = Analysis_DD.create(
        run_name="DD", filepath=str(dd_path) + os.sep, model_type="LFADS"
    )

    _, noiseless_inputs, _ = an_tt.get_model_inputs_noiseless(phase=args.phase)
    inputs_nl = as_numpy(noiseless_inputs)
    extras = as_numpy(an_tt.get_extra_inputs(phase=args.phase))
    lats_tt = as_numpy(an_tt.get_latents(phase=args.phase))
    lats_dd = as_numpy(an_dd.get_latents(phase=args.phase))
    true_rates = as_numpy(an_dd.get_true_rates(phase=args.phase))
    spikes = as_numpy(an_dd.get_spiking(phase=args.phase))
    inf_rates = as_numpy(an_dd.get_rates(phase=args.phase))

    inds_a = inputs_nl[:, :, 1].sum(axis=1) > 0
    inds_b = inputs_nl[:, :, 2].sum(axis=1) > 0
    theta = compute_theta(inputs_nl[:, :, 0])

    flat_tt = normalize_by_all(flatten_poststim(lats_tt, theta, extras, inds_a, inds_b))
    flat_dd = normalize_by_all(flatten_poststim(lats_dd, theta, extras, inds_a, inds_b))

    bar_metrics = compute_bar_metrics(an_tt, an_dd)

    payload = {
        "inputs_nl": inputs_nl,
        "extras": extras,
        "inds_a": inds_a,
        "inds_b": inds_b,
        "theta": theta,
        "lats_tt_pca": pca_trajectories(lats_tt),
        "lats_dd_pca": pca_trajectories(lats_dd),
        "rates_pca": pca_trajectories(true_rates),
        "flat_tt": flat_tt,
        "flat_dd": flat_dd,
        "mean_delta_tt": flat_tt["a"].mean(axis=0) - flat_tt["b"].mean(axis=0),
        "mean_delta_dd": flat_dd["a"].mean(axis=0) - flat_dd["b"].mean(axis=0),
        "true_rates": true_rates,
        "spikes": spikes,
        "inf_rates": inf_rates,
        "bar_metrics": bar_metrics,
    }
    for spec in EXTRA_DD_MODELS:
        dd_path_x = extra_dd_path(args, tt_path, spec)
        if not dd_path_x.exists():
            raise FileNotFoundError(
                f"Could not find {spec['dim_label']} DD analysis at {dd_path_x}. "
                f"Set --{spec['prefix']}-path or --{spec['prefix']}-subpath."
            )
        arrays = compute_dd_model_arrays(
            dd_path_x,
            args.phase,
            theta,
            extras,
            inds_a,
            inds_b,
            run_name=spec["prefix"].upper(),
        )
        payload[f"lats_{spec['prefix']}_pca"] = arrays["lats_pca"]
        payload[f"flat_{spec['prefix']}"] = arrays["flat"]
        payload[f"mean_delta_{spec['prefix']}"] = arrays["mean_delta"]

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as f:
        pickle.dump(payload, f)
    return payload


def compute_bar_metrics(an_tt, an_dd) -> dict[str, float]:
    """Compute state R², rate R², cycle consistency, and co-bps for the TT/DD pair."""
    comparison = Comparison(comparison_tag="PCM_bar")
    comparison.load_analysis(an_tt, group="TT", reference_analysis=True)
    comparison.load_analysis(an_dd, group="DD")
    metrics = comparison.compute_metrics(metric_dict_list=dict(METRIC_BAR_SPEC))
    out: dict[str, float] = {}
    for key in METRIC_BAR_KEYS:
        values = metrics.get(key, [])
        out[key] = float(values[0]) if values else float("nan")
    return out


# %%
def setup_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
        }
    )


def panel_label(ax, label: str):
    text_func = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_func(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="normal",
        ha="left",
        va="top",
    )


def place_row_panel_labels(fig, axes_labels, dy=0.012, dx=-0.01):
    """Stamp panel labels for a single row of axes at a common figure height.

    Each label is placed at its own panel's left edge (in figure coordinates),
    so all labels share the same vertical position. The common height is taken
    above the highest plot element in the row -- including axis titles, whose
    rendered extents are measured -- so the labels never intersect the plots.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    positions = [ax.get_position() for ax, _ in axes_labels]
    tops = []
    for (ax, _), pos in zip(axes_labels, positions):
        top = pos.y1
        title = ax.title
        if title.get_text():
            bbox = title.get_window_extent(renderer=renderer)
            top = max(top, inv.transform((0.0, bbox.y1))[1])
        tops.append(top)
    y = min(max(tops) + dy, 0.999)
    for (ax, label), pos in zip(axes_labels, positions):
        fig.text(
            max(pos.x0 + dx, 0.0),
            y,
            label,
            fontsize=13,
            fontweight="normal",
            ha="left",
            va="bottom",
        )


def style_phase_axis(ax, show_xlabel=False, show_ylabel=False):
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    if show_xlabel:
        ax.set_xlabel("Phase", labelpad=0)
    else:
        ax.set_xticklabels([])
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_unit_grid(
    parent_ax,
    flat: dict[str, np.ndarray],
    neurons: list[int],
    title: str,
    title_color: str,
    step: int,
):
    panel_label(parent_ax, title[0])
    parent_ax.axis("off")
    if title.startswith("C"):
        parent_ax.set_title("Example Single Unit Activity", pad=3)
    else:
        parent_ax.set_title("Example " + r"$\bf{DD}$" + "-Inferred Latents", pad=3)

    gs = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=parent_ax.get_subplotspec(), wspace=0.22, hspace=0.24
    )
    idx = np.arange(0, min(flat["a"].shape[0], flat["b"].shape[0]), step)
    for i, neuron in enumerate(neurons):
        ax = parent_ax.figure.add_subplot(gs[i // 3, i % 3])
        n = min(neuron, flat["a"].shape[1] - 1, flat["b"].shape[1] - 1)
        ax.scatter(
            flat["theta_a"][idx],
            flat["a"][idx, n],
            s=1.2,
            color=STIM_A,
            alpha=0.18,
            linewidths=0,
        )
        ax.scatter(
            flat["theta_b"][idx],
            flat["b"][idx, n],
            s=1.2,
            color=STIM_B,
            alpha=0.18,
            linewidths=0,
        )
        ax.text(
            0.92,
            0.8,
            str(neuron),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6,
        )
        style_phase_axis(ax, show_xlabel=i >= 3, show_ylabel=False)


def plot_pca_panel(
    ax,
    lats_pca: np.ndarray,
    extras: np.ndarray,
    inds_a: np.ndarray,
    inds_b: np.ndarray,
    title: str,
    title_color: str,
    max_trials: int,
    show_pc_labels: bool = True,
):
    selected = np.arange(lats_pca.shape[0])
    if selected.size > max_trials:
        selected = np.linspace(0, selected.size - 1, max_trials).astype(int)
    for trial in selected:
        color = STIM_A if inds_a[trial] else STIM_B if inds_b[trial] else "0.3"
        start = int(extras[trial, 1])
        ax.plot(
            lats_pca[trial, start:, 0],
            lats_pca[trial, start:, 1],
            lats_pca[trial, start:, 2],
            color=color,
            alpha=0.18,
            linewidth=0.45,
        )
    ax.set_title(title, pad=1, color=title_color or "black")
    if show_pc_labels:
        ax.set_xlabel("PC1", labelpad=-8)
        ax.set_ylabel("PC2", labelpad=-8)
        ax.set_zlabel("PC3", labelpad=-8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=20, azim=-55)
    ax.set_box_aspect((1.0, 0.85, 0.7))


def plot_neural_panel(
    parent_ax,
    rates: np.ndarray,
    spikes: np.ndarray,
    trial: int,
    inf_rates: np.ndarray | None = None,
):
    panel_label(parent_ax, "E")
    parent_ax.axis("off")
    trial = int(np.clip(trial, 0, rates.shape[0] - 1))
    shapes = [rates.shape[-1], spikes.shape[-1]]
    times = [rates.shape[1], spikes.shape[1]]
    if inf_rates is not None:
        shapes.append(inf_rates.shape[-1])
        times.append(inf_rates.shape[1])
    n_neurons = min(500, *shapes)
    t = min(times)
    rates_img = rates[trial, :t, :n_neurons].T
    if np.nanmean(rates_img) > 0:
        rates_img = rates_img / np.nanmean(rates_img) * 2.0
    spikes_img = spikes[trial, :t, :n_neurons].T

    panels = [("Ground-Truth Rates", rates_img, False)]
    if inf_rates is not None:
        inf_img = inf_rates[trial, :t, :n_neurons].T
        if np.nanmean(inf_img) > 0:
            inf_img = inf_img / np.nanmean(inf_img) * 2.0
        panels.append(("DD-Inferred Rates", inf_img, False))
    panels.append(("Simulated Spiking", spikes_img > 0, True))

    gs = GridSpecFromSubplotSpec(
        len(panels), 1, subplot_spec=parent_ax.get_subplotspec(), hspace=0.35
    )
    axes = []
    for i, (title, img, is_spikes) in enumerate(panels):
        ax = parent_ax.figure.add_subplot(gs[i, 0])
        ax.imshow(
            img, aspect="auto", interpolation="nearest", cmap="viridis", origin="lower"
        )
        ax.set_title(title, pad=1)
        ax.tick_params(length=2, pad=1)
        ax.spines[["top", "right"]].set_visible(False)
        axes.append(ax)

    for ax in axes[:-1]:
        ax.set_xticks([])
    axes[-1].set_xlabel("Time (ms)", labelpad=0)
    axes[0].set_ylabel("Neurons")
    axes[-1].set_yticks([0, n_neurons - 1])
    axes[-1].set_yticklabels(["1", str(n_neurons)])


def plot_hist_panel(
    ax,
    delta_tt: np.ndarray,
    delta_dd: np.ndarray,
    dd_label: str = "DD",
    bins=None,
    tt_color: str = HIST_TT,
    show_medians: bool = False,
):
    panel_label(ax, "H")
    if bins is None:
        bins = np.linspace(0, max(np.percentile(np.abs(delta_dd), 99), 1.0), 55)
    abs_tt = np.abs(delta_tt)
    abs_dd = np.abs(delta_dd)
    ax.hist(abs_tt, bins=bins, density=True, color=tt_color, alpha=0.85, label="TT")
    ax.hist(abs_dd, bins=bins, density=True, color=HIST_DD, alpha=0.85, label=dd_label)
    if show_medians:
        # Vertical line at the median of each distribution.
        tt_line = MEDIAN_TT_LINE if tt_color == HIST_TT_GREY else tt_color
        ax.axvline(
            np.median(abs_tt), color=tt_line, linestyle="--", linewidth=1.1, zorder=5
        )
        ax.axvline(
            np.median(abs_dd),
            color=MEDIAN_DD_LINE,
            linestyle="--",
            linewidth=1.1,
            zorder=5,
        )
    ax.set_ylabel("Probability Density")
    ax.set_xlabel("norm. abs. $\Delta$ hidden unit act")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)


def plot_metrics_bar(ax, bar_metrics: dict[str, float]):
    panel_label(ax, "I")
    keys = list(METRIC_BAR_KEYS)
    values = [bar_metrics.get(k, float("nan")) for k in keys]
    labels = [METRIC_BAR_LABELS[k] for k in keys]
    xs = np.arange(len(keys))
    ax.bar(xs, values, color=DD_RED, width=0.65, edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("DD Reconstruction Metrics", pad=2)
    ax.axhline(0, color="0.5", linewidth=0.5)
    ymax = max([v for v in values if np.isfinite(v)] + [1.0])
    ax.set_ylim(min(0.0, *[v for v in values if np.isfinite(v)] or [0.0]), ymax * 1.12)
    for x, v in zip(xs, values):
        if np.isfinite(v):
            ax.text(
                x, v + ymax * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6
            )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2, pad=1)


def save_figure(fig, output_dir: Path, stem: str, dpi: int, exts=("pdf", "svg")):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)


def build_figure(payload: dict[str, np.ndarray], args: argparse.Namespace):
    setup_matplotlib()
    fig = plt.figure(figsize=(6.7, 10.4), constrained_layout=False)
    gs = GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[1.0, 1.05, 1.05, 1.05, 0.85],
        wspace=0.22,
        hspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    for label, ax in (("A", ax_a), ("B", ax_b)):
        panel_label(ax, label)
        ax.axis("off")
        if args.show_placeholders:
            ax.text(
                0.5,
                0.5,
                "schematic panel",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.55",
            )

    ax_c = fig.add_subplot(gs[1, 0])
    plot_unit_grid(
        ax_c, payload["flat_tt"], TT_NEURONS, "C", TT_GREEN, args.scatter_step
    )

    ax_d = fig.add_subplot(gs[1, 1], projection="3d")
    panel_label(ax_d, "D")
    plot_pca_panel(
        ax_d,
        payload["lats_tt_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "TT Latents",
        TT_GREEN,
        args.max_pca_trials,
    )

    ax_e = fig.add_subplot(gs[2, 0])
    plot_neural_panel(
        ax_e,
        payload["true_rates"],
        payload["spikes"],
        args.trial,
        inf_rates=payload.get("inf_rates"),
    )

    ax_f = fig.add_subplot(gs[2, 1], projection="3d")
    panel_label(ax_f, "F")
    plot_pca_panel(
        ax_f,
        payload["lats_dd_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "DD-Inferred Latents",
        DD_RED,
        args.max_pca_trials,
    )

    ax_g = fig.add_subplot(gs[3, 0])
    plot_unit_grid(ax_g, payload["flat_dd"], DD_NEURONS, "G", DD_RED, args.scatter_step)

    ax_h = fig.add_subplot(gs[3, 1])
    plot_hist_panel(ax_h, payload["mean_delta_tt"], payload["mean_delta_dd"])

    ax_i = fig.add_subplot(gs[4, 0])
    plot_metrics_bar(ax_i, payload["bar_metrics"])

    ax_j = fig.add_subplot(gs[4, 1], projection="3d")
    panel_label(ax_j, "J")
    plot_pca_panel(
        ax_j,
        payload["rates_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "Ground-Truth Rate PCs",
        "black",
        args.max_pca_trials,
    )

    return fig


def save_panel_exports(payload: dict[str, np.ndarray], args: argparse.Namespace):
    setup_matplotlib()
    panels = {}

    fig_c = plt.figure(figsize=(3.2, 1.55))
    ax_c = fig_c.add_subplot(111)
    plot_unit_grid(
        ax_c, payload["flat_tt"], TT_NEURONS, "C", TT_GREEN, args.scatter_step
    )
    panels["panel_C_single_units_TT"] = fig_c

    fig_d = plt.figure(figsize=(3.2, 2.2))
    ax_d = fig_d.add_subplot(111, projection="3d")
    panel_label(ax_d, "D")
    plot_pca_panel(
        ax_d,
        payload["lats_tt_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "TT Latents",
        TT_GREEN,
        args.max_pca_trials,
    )
    panels["panel_D_latents_TT"] = fig_d

    fig_e = plt.figure(figsize=(3.2, 2.9))
    ax_e = fig_e.add_subplot(111)
    plot_neural_panel(
        ax_e,
        payload["true_rates"],
        payload["spikes"],
        args.trial,
        inf_rates=payload.get("inf_rates"),
    )
    panels["panel_E_neural_data"] = fig_e

    fig_f = plt.figure(figsize=(3.2, 2.2))
    ax_f = fig_f.add_subplot(111, projection="3d")
    panel_label(ax_f, "F")
    plot_pca_panel(
        ax_f,
        payload["lats_dd_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "DD-Inferred Latents",
        DD_RED,
        args.max_pca_trials,
    )
    panels["panel_F_latents_DD"] = fig_f

    fig_g = plt.figure(figsize=(3.2, 1.55))
    ax_g = fig_g.add_subplot(111)
    plot_unit_grid(ax_g, payload["flat_dd"], DD_NEURONS, "G", DD_RED, args.scatter_step)
    panels["panel_G_single_units_DD"] = fig_g

    fig_h = plt.figure(figsize=(3.2, 2.0))
    ax_h = fig_h.add_subplot(111)
    plot_hist_panel(ax_h, payload["mean_delta_tt"], payload["mean_delta_dd"])
    panels["panel_H_delta_hist"] = fig_h

    fig_i = plt.figure(figsize=(3.2, 2.0))
    ax_i = fig_i.add_subplot(111)
    plot_metrics_bar(ax_i, payload["bar_metrics"])
    panels["panel_I_dd_metrics_bar"] = fig_i

    fig_j = plt.figure(figsize=(3.2, 2.2))
    ax_j = fig_j.add_subplot(111, projection="3d")
    panel_label(ax_j, "J")
    plot_pca_panel(
        ax_j,
        payload["rates_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "Ground-Truth Rate PCs",
        "black",
        args.max_pca_trials,
    )
    panels["panel_J_rates_pca"] = fig_j

    for stem, fig in panels.items():
        save_figure(fig, args.output_dir, stem, args.dpi)
        plt.close(fig)


def build_tt_figure(payload: dict[str, np.ndarray], args: argparse.Namespace):
    """TT model + simulated data only (panels A-E).

    Layout: row 0 = A, B schematics (2 cols). Row 1 = C (TT latents PCA),
    D (single-unit activity), E (rates + spikes), all on a single row.
    """
    setup_matplotlib()
    fig = plt.figure(figsize=(6.7, 3.9), constrained_layout=False)
    gs = GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1.0, 1.4],
        wspace=0.7,
        hspace=0.35,
    )

    # Row 0: schematics span 3 cols each.
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    for label, ax in (("A", ax_a), ("B", ax_b)):
        panel_label(ax, label)
        ax.axis("off")
        if args.show_placeholders:
            ax.text(
                0.5,
                0.5,
                "schematic panel",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.55",
            )

    # Row 1: C = TT latents, D = single units, E = rates + spikes (2 cols each).
    ax_c = fig.add_subplot(gs[1, 0:2], projection="3d")
    panel_label(ax_c, "C")
    plot_pca_panel(
        ax_c,
        payload["lats_tt_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        "TT Latents",
        TT_GREEN,
        args.max_pca_trials,
    )

    ax_d = fig.add_subplot(gs[1, 2:4])
    plot_unit_grid(
        ax_d, payload["flat_tt"], TT_NEURONS, "D", TT_GREEN, args.scatter_step
    )
    # plot_unit_grid sets the title based on the first character of `title`;
    # restore the TT single-unit subtitle for the new D panel.
    ax_d.set_title("Example Single Unit Activity", pad=3)

    ax_e = fig.add_subplot(gs[1, 4:6])
    plot_neural_panel(
        ax_e,
        payload["true_rates"],
        payload["spikes"],
        args.trial,
        inf_rates=None,
    )

    return fig


def _plot_dd_model_row(
    fig,
    axes,
    payload,
    model,
    neurons,
    dim_label,
    labels,
    args,
    hist_bins,
    show_pc_labels=True,
):
    """Plot one DD-model row: latents PCA, single-unit grid, and Δ-activity hist.

    ``axes`` is (ax_pca, ax_units, ax_hist); ``model`` provides the per-model
    arrays; ``labels`` are the three panel letters for this row. The TT model is
    always the histogram reference (grey, with a median line), while the DD curve
    reflects ``model``. ``hist_bins`` is shared across rows for a common scale.
    ``show_pc_labels`` keeps the PC1/PC2/PC3 axis labels on the PCA panel (used
    only for the top row so the stacked PCA column is labelled once).
    """
    ax_pca, ax_units, ax_hist = axes
    plot_pca_panel(
        ax_pca,
        model["lats_pca"],
        payload["extras"],
        payload["inds_a"],
        payload["inds_b"],
        f"{dim_label} DD-Inferred Latents",
        DD_RED,
        args.max_pca_trials,
        show_pc_labels=show_pc_labels,
    )

    plot_unit_grid(
        ax_units, model["flat"], neurons, labels[1], DD_RED, args.scatter_step
    )
    # plot_unit_grid sets the fallback title "Example DD-Inferred Latents" for any
    # non-C label; for the split DD figure this panel is single-unit activity.
    ax_units.set_title(f"Example {dim_label} DD Single Unit Activity", pad=3)

    plot_hist_panel(
        ax_hist,
        payload["mean_delta_tt"],
        model["mean_delta"],
        dd_label=f"DD ({dim_label})",
        bins=hist_bins,
        tt_color=HIST_TT_GREY,
        show_medians=True,
    )

    # plot_pca_panel / plot_unit_grid / plot_hist_panel each stamp their own panel
    # label at axes-relative positions that don't line up across 2D and 3D axes.
    # Strip those and place this row's labels at a single, shared figure height.
    for ax in axes:
        for txt in list(ax.texts):
            txt.remove()
    place_row_panel_labels(fig, list(zip(axes, labels)))


def build_dd_figure(payload: dict[str, np.ndarray], args: argparse.Namespace):
    """DD model validation, three rows: 8D, 16D, and 128D models from the sweep.

    Row 0 (A-C): 8D model. Row 1 (D-F): 16D model. Row 2 (G-I): 128D model.
    Each row shows DD-inferred latents (PCA), example single-unit activity, and
    the distribution of stimulus-conditioned activity differences vs. the TT model.
    The right-most Δ-activity histograms share a common scale and mark the median
    of the (grey) TT distribution and of each DD distribution with a dashed line.
    """
    setup_matplotlib()
    fig = plt.figure(figsize=(6.7, 7.0), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig, wspace=0.4, hspace=0.6)

    rows = [
        {
            "lats_pca": payload["lats_dd8_pca"],
            "flat": payload["flat_dd8"],
            "mean_delta": payload["mean_delta_dd8"],
            "neurons": DD8_NEURONS,
            "dim_label": "8D",
            "labels": ("A", "B", "C"),
        },
        {
            "lats_pca": payload["lats_dd_pca"],
            "flat": payload["flat_dd"],
            "mean_delta": payload["mean_delta_dd"],
            "neurons": DD_NEURONS,
            "dim_label": "16D",
            "labels": ("D", "E", "F"),
        },
        {
            "lats_pca": payload["lats_dd128_pca"],
            "flat": payload["flat_dd128"],
            "mean_delta": payload["mean_delta_dd128"],
            "neurons": DD128_NEURONS,
            "dim_label": "128D",
            "labels": ("G", "H", "I"),
        },
    ]

    # Shared histogram bins so the right-most panels are on the same scale. The
    # upper edge covers the 99th percentile of the TT and every DD distribution.
    all_deltas = [np.abs(payload["mean_delta_tt"])] + [
        np.abs(r["mean_delta"]) for r in rows
    ]
    hist_upper = max([np.percentile(d, 99) for d in all_deltas] + [1.0])
    hist_bins = np.linspace(0, hist_upper, 55)

    hist_axes = []
    for r, row in enumerate(rows):
        ax_pca = fig.add_subplot(gs[r, 0], projection="3d")
        ax_units = fig.add_subplot(gs[r, 1])
        ax_hist = fig.add_subplot(gs[r, 2])
        _plot_dd_model_row(
            fig,
            (ax_pca, ax_units, ax_hist),
            payload,
            row,
            row["neurons"],
            row["dim_label"],
            row["labels"],
            args,
            hist_bins,
            show_pc_labels=(r == 0),
        )
        hist_axes.append(ax_hist)

    # Put every histogram on identical x/y limits (same scale across rows).
    ymax = max(ax.get_ylim()[1] for ax in hist_axes)
    for ax in hist_axes:
        ax.set_xlim(0, hist_upper)
        ax.set_ylim(0, ymax)

    return fig


# %%
def main(argv=None):
    args = parse_args(argv)
    payload = load_or_compute_payload(args)

    fig = build_figure(payload, args)
    save_figure(fig, args.output_dir, "FigureS3_PhaseCodedMemory", args.dpi)
    plt.close(fig)

    fig_tt = build_tt_figure(payload, args)
    save_figure(
        fig_tt, args.output_dir, "CyclicDatasetV3_TT", args.dpi, exts=("png", "pdf")
    )
    plt.close(fig_tt)

    fig_dd = build_dd_figure(payload, args)
    save_figure(
        fig_dd, args.output_dir, "CyclicDatasetV3_DD", args.dpi, exts=("png", "pdf")
    )
    # Also drop the DD figure straight into the manuscript figs dir so the
    # \includegraphics{figs/CyclicDatasetV3_DD.png} picks it up automatically.
    # (The TT figure is intentionally NOT copied here: it has manually inserted
    # components in the manuscript version.)
    save_figure(
        fig_dd, args.manuscript_dir, "CyclicDatasetV3_DD", args.dpi, exts=("png", "pdf")
    )
    plt.close(fig_dd)

    save_panel_exports(payload, args)
    print(f"Saved PhaseCodedMemory figure outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
