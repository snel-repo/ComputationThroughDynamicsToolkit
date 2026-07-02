# %%
"""Paper-quality summary figure for the Chaotic Delayed Matching task.

Run this file as a script or cell-by-cell in an editor that supports ``# %%``.
The local benchmark/config name is ``ChaoticDelayedMatching``; figure text uses
"Chaotic Delayed Matching" by default because this is the manuscript-facing name.
"""

import argparse
import importlib
import os
import pickle
import re
import shutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.metrics import compute_jacobians

# %%
# --------------------------- User parameters ---------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_NAME_FOR_FIGURE = "Chaotic Delayed Matching"
TT_RUN_PATH = (
    REPO_ROOT
    / "content"
    / "trained_models"
    / "task-trained"
    / "tt_ChaoticDelayedMatching"
)
# Use the same DD-NODE sweep that backs the compiled-metrics figure
# (FigureS_CDM_metrics_vs_latent_size); the example fit shown here is the
# seed=0, latent_size=32 run from that sweep.
DD_NODE_SWEEP_PATH = TT_RUN_PATH / "20260529_CDM_NODE_DimSweep_Final"
DD_NODE_EXAMPLE_RUN_PATH = (
    DD_NODE_SWEEP_PATH
    / "prefix=tt_ChaoticDelayedMatching_max_epochs=200_latent_size=32_seed=1"
)
# Save next to the other supplementary figure scripts (e.g. PhaseCodedMemory,
# NL cycle-consistency, 3BFF) rather than in the repo-wide content/figures.
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_STEM = "chaotic_delayed_matching_summary"

# Manuscript integration: --move-to-manuscript copies the standalone TT/DD PDFs
# into manuscript/figs/ (where main.tex includes them by name -- figs/
# ChaoticDelayedMatching_TT.pdf and figs/ChaoticDelayedMatching_DD.pdf) and then
# rebuilds the PDF with latexmk (config in manuscript/.latexmkrc).
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
MANUSCRIPT_FIGS_DIR = MANUSCRIPT_DIR / "figs"
# Figures included by main.tex; their filenames must match the \includegraphics
# calls exactly so each copy drops straight into place.
MANUSCRIPT_FIGURES = (
    "ChaoticDelayedMatching_TT.pdf",
    "ChaoticDelayedMatching_DD.pdf",
)

PHASE = "all"
EXAMPLE_TRIAL = (
    "nonmatch"  # int trial index, or "nonmatch"/"match" for automatic selection.
)
RASTER_TRIAL = 2
DD_NODE_PHASE = PHASE
DD_NODE_TRIAL = (
    "same_as_panel_b"  # int trial index in DD_NODE_PHASE, or "same_as_panel_b".
)
N_RASTER_NEURONS = 48
N_LATENT_TRAJECTORY_TRIALS = 24
N_IC_PERTURBATIONS = 16
IC_PERTURB_SCALE = 0.5
# IC perturbation seeding, applied consistently to BOTH the TT and DD models:
#   "pc12"     -- ring in the plane of each model's top two latent PCs (fair,
#                 apples-to-apples between TT and DD).
#   "lyapunov" -- a tight cone around each model's leading Lyapunov vector (the
#                 direction the maximal Lyapunov exponent describes), so the
#                 perturbation-growth panel and the lambda_max bar measure the
#                 same thing. Falls back to "pc12" if the leading vector can't be
#                 computed.
#   "random"   -- isotropic random directions in the full latent space.
IC_PERTURB_MODE = "pc12"
IC_PERTURB_LYAP_JITTER = 0.1  # cone half-width (frac of unit dir) in "lyapunov" mode
IC_PERTURB_LYAP_NITER = 50  # power-iteration steps for the leading Lyapunov vector
IC_TRIAL = 0
RANDOM_SEED = 11
# Fraction of trials used to estimate the maximal Lyapunov exponent. Matches the
# compiled-metrics figure (make_compiled_metrics_vs_latent_size.py), which uses
# a 10% subsample for the CDM/NODE Lyapunov pass.
LYAP_SUBSET_FRAC = 0.1
TIME_BIN_MS = (
    1.0  # ChaoticDelayedMatching uses one model step per ms in the default configs.
)

SAVE_FIGURE = True
SHOW_FIGURE = True
# When run as a script, --move-to-manuscript flips this on (parsed just below).
# For cell-by-cell execution argparse is skipped, so set it True by hand here if
# you want the copy + latexmk rebuild to run from the editor.
MOVE_TO_MANUSCRIPT = True

# --- Figure-data cache -------------------------------------------------------
# The slow part of this script is loading the TT/DD analyses and running the
# model rollouts; the plotting itself is fast. We pickle the computed payload
# (the arrays the panels consume) so that re-runs where only the plotting code
# or styling changed skip the model work entirely. The cache stores a signature
# of the compute-affecting parameters below and is rebuilt automatically when
# any of them change; pass --recompute to force a rebuild, --no-cache to ignore
# the cache for one run.
CACHE_PATH = Path(__file__).resolve().with_suffix(".cache.pkl")
USE_CACHE = True  # read and write the cached compute payload
RECOMPUTE = False  # force a fresh compute even if a valid cache exists

if "ipykernel" not in sys.modules and __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Build the ChaoticDelayedMatching supplementary figures."
    )
    _parser.add_argument(
        "--move-to-manuscript",
        action="store_true",
        help=(
            "After saving, copy ChaoticDelayedMatching_TT.pdf and "
            "ChaoticDelayedMatching_DD.pdf into manuscript/figs/ (overwriting the "
            "included copies) and rebuild the manuscript PDF with latexmk."
        ),
    )
    _parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute the figure data from the models instead of using the cache.",
    )
    _parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the figure-data cache for this run.",
    )
    # parse_known_args so an odd launch context can't crash the figure build.
    _args, _ = _parser.parse_known_args()
    MOVE_TO_MANUSCRIPT = _args.move_to_manuscript
    RECOMPUTE = _args.recompute
    USE_CACHE = not _args.no_cache


def copy_to_manuscript(
    written: list[Path], figs_dir: Path = MANUSCRIPT_FIGS_DIR
) -> list[Path]:
    """Copy generated figure files into the manuscript ``figs/`` directory.

    ``main.tex`` includes these by exact filename
    (``figs/ChaoticDelayedMatching_TT.pdf`` and ``..._DD.pdf``), so each file
    drops straight into place and overwrites the previously included copy.
    Returns the destinations actually written (empty if the manuscript figs
    directory is missing).
    """
    if not figs_dir.is_dir():
        print(f"Skipped manuscript copy (missing {figs_dir})")
        return []
    copied: list[Path] = []
    for src in written:
        if not src.exists():
            warnings.warn(f"Expected figure {src} not found; skipped copy.")
            continue
        dst = figs_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"  copied {src.name} -> {dst}")
    return copied


def rebuild_manuscript(manuscript_dir: Path = MANUSCRIPT_DIR) -> None:
    """Rebuild the manuscript PDF with ``latexmk`` (config in ``.latexmkrc``).

    Warns rather than raising if ``latexmk`` is missing or the build fails, so a
    figure run isn't lost just because the LaTeX toolchain isn't available here.
    """
    if not (manuscript_dir / "main.tex").exists():
        print(f"Skipped manuscript rebuild (no main.tex in {manuscript_dir})")
        return
    print(f"Rebuilding manuscript with latexmk in {manuscript_dir}")
    try:
        subprocess.run(["latexmk"], cwd=str(manuscript_dir), check=True)
    except FileNotFoundError:
        warnings.warn("latexmk not found on PATH; skipped manuscript rebuild.")
    except subprocess.CalledProcessError as exc:
        warnings.warn(
            f"latexmk failed (exit {exc.returncode}); manuscript not rebuilt."
        )


PHASE_DEFS = [
    ("Cue 1", "cue1_on", "cue1_off", "#4C78A8"),
    ("Delay 1", "delay1_on", "delay1_off", "#F58518"),
    ("Cue 2", "cue2_on", "cue2_off", "#B279A2"),
    ("Delay 2", "delay2_on", "delay2_off", "#E45756"),
    ("Response", "resp_on", "resp_off", "#54A24B"),
]
# %%

# --------------------------- Setup helpers ---------------------------


def install_compatibility_stubs():
    """Small import shims used by older saved pickles."""

    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *args, **kwargs: False
        dotenv_stub.find_dotenv = lambda *args, **kwargs: ""
        sys.modules["dotenv"] = dotenv_stub

    if "DSA" not in sys.modules:
        dsa_module = types.ModuleType("DSA")
        dsa_stats = types.ModuleType("DSA.stats")
        dsa_stats.dsa_bw_data_splits = lambda *args, **kwargs: None
        dsa_stats.dsa_to_id = lambda *args, **kwargs: None
        dsa_module.DSA = object
        sys.modules["DSA"] = dsa_module
        sys.modules["DSA.stats"] = dsa_stats

    if "geotorch" not in sys.modules:
        geotorch_stub = types.ModuleType("geotorch")
        geotorch_stub.orthogonal = lambda *args, **kwargs: None
        sys.modules["geotorch"] = geotorch_stub

    if "wandb" not in sys.modules:
        wandb_stub = types.ModuleType("wandb")

        class Image:
            def __init__(self, *args, **kwargs):
                pass

        wandb_stub.Image = Image
        wandb_stub.init = lambda *args, **kwargs: None
        wandb_stub.log = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb_stub

    try:
        import jaraco.context
        import jaraco.functools
        import jaraco.text
        import packaging
        import packaging.markers
        import packaging.requirements
        import packaging.specifiers
        import packaging.utils
        import packaging.version
        import pkg_resources  # noqa: F401

        vendor_stub = sys.modules.setdefault(
            "pkg_resources._vendor", types.ModuleType("pkg_resources._vendor")
        )
        vendor_stub.jaraco = jaraco
        vendor_stub.packaging = packaging
        sys.modules["pkg_resources._vendor.jaraco"] = jaraco
        sys.modules["pkg_resources._vendor.jaraco.context"] = jaraco.context
        sys.modules["pkg_resources._vendor.jaraco.functools"] = jaraco.functools
        sys.modules["pkg_resources._vendor.jaraco.text"] = jaraco.text
        sys.modules["pkg_resources._vendor.packaging"] = packaging
        sys.modules["pkg_resources._vendor.packaging.markers"] = packaging.markers
        sys.modules["pkg_resources._vendor.packaging.requirements"] = (
            packaging.requirements
        )
        sys.modules["pkg_resources._vendor.packaging.specifiers"] = packaging.specifiers
        sys.modules["pkg_resources._vendor.packaging.utils"] = packaging.utils
        sys.modules["pkg_resources._vendor.packaging.version"] = packaging.version
    except Exception:
        pass

    try:
        import lightning_fabric.utilities.data as lightning_data

        if not hasattr(lightning_data, "AttributeDict"):

            class AttributeDict(dict):
                __getattr__ = dict.get
                __setattr__ = dict.__setitem__

            lightning_data.AttributeDict = AttributeDict
    except Exception:
        pass

    try:
        import pytorch_lightning.utilities.model_helpers as model_helpers

        if not hasattr(model_helpers, "_ModuleMode"):

            class _ModuleMode:
                def __init__(self, *args, **kwargs):
                    pass

            model_helpers._ModuleMode = _ModuleMode
    except Exception:
        pass

    legacy_aliases = {
        "ctd.data_modeling.models.SAE.dyn_models_GRU": (
            "ctd.data_modeling.models.SAE.dyn_models_gru"
        ),
    }
    for legacy_name, target_name in legacy_aliases.items():
        if legacy_name not in sys.modules:
            try:
                sys.modules[legacy_name] = importlib.import_module(target_name)
            except Exception:
                pass


def configure_plotting():
    # We intentionally prefer Arial and fall back to Liberation Sans (a
    # metric-compatible Arial stand-in) when Arial is not installed. Matplotlib
    # emits a "Font family 'Arial' not found" warning on every fallback lookup;
    # silence that one logger so the deliberate fallback does not spam the log.
    import logging

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            # Prefer Arial; Liberation Sans is a metric-compatible stand-in used
            # when Arial is not installed, so the rendered figure still looks
            # like Arial. DejaVu Sans is the final fallback.
            "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


# Panel-marker styling. Letters are drawn as standalone bold 12 pt Arial text at
# the top-left corner of each panel's axes. Because matplotlib axes positions
# (``get_position``) are determined by the GridSpec and exclude tick/axis labels,
# every panel in the same GridSpec column shares the same left edge, so placing
# each letter at ``x0`` automatically aligns the markers vertically.
PANEL_LABEL_FONTSIZE = 12
PANEL_LABEL_DX = -0.052  # figure-fraction offset left of the axes' left edge
PANEL_LABEL_DY = 0.006  # figure-fraction offset above the axes' top edge


def add_panel_label(fig, ax, letter, dx=PANEL_LABEL_DX, dy=PANEL_LABEL_DY):
    """Draw a bold 12 pt panel marker at the top-left corner of ``ax``."""
    bbox = ax.get_position()
    # Inherit the configured font.family chain (Arial -> Liberation Sans ->
    # DejaVu Sans) rather than hard-coding "Arial", which would bypass the
    # fallback and render the markers in a different font than the body text.
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        letter,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def response_window_cols(extra_tensor):
    if extra_tensor.shape[1] >= 13:
        return 8, 9
    if extra_tensor.shape[1] >= 11:
        return 6, 7
    raise ValueError(f"Unsupported extra shape: {tuple(extra_tensor.shape)}")


def get_extra_indices(extra_tensor):
    if extra_tensor.shape[1] >= 13:
        return {
            "cue1_on": 0,
            "cue1_off": 1,
            "delay1_on": 2,
            "delay1_off": 3,
            "cue2_on": 4,
            "cue2_off": 5,
            "delay2_on": 6,
            "delay2_off": 7,
            "resp_on": 8,
            "resp_off": 9,
            "cue1_id": 10,
            "cue2_id": 11,
            "nonmatch": 12,
        }
    return {
        "cue1_on": 0,
        "cue1_off": 1,
        "delay1_on": 2,
        "delay1_off": 3,
        "cue2_on": 4,
        "cue2_off": 5,
        "delay2_on": None,
        "delay2_off": None,
        "resp_on": 6,
        "resp_off": 7,
        "cue1_id": None,
        "cue2_id": None,
        "nonmatch": None,
    }


def time_axis(n_time):
    return np.arange(n_time) * TIME_BIN_MS


def rate_r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return np.nan
    return 1.0 - ss_res / ss_tot


def variance_weighted_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_dims = y_true.shape[-1]
    return r2_score(
        y_true.reshape(-1, n_dims),
        y_pred.reshape(-1, n_dims),
        multioutput="variance_weighted",
    )


def describe_heatmap_panel(name, raw_array, selected_array, x_axis, y_axis):
    print(f"{name} heatmap raw shape: {raw_array.shape}")
    print(f"{name} heatmap selected shape passed to plotting: {selected_array.shape}")
    print(f"{name} heatmap current axis assignment: x={x_axis}, y={y_axis}")


def plot_time_neuron_heatmap(ax, time_by_neuron, cmap, vmin=None, vmax=None):
    """Plot an array shaped (time, neurons) with time on x and neurons on y."""
    time_by_neuron = np.asarray(time_by_neuron)
    n_time, n_neurons = time_by_neuron.shape
    t_edges = np.arange(n_time + 1) * TIME_BIN_MS
    neuron_edges = np.arange(n_neurons + 1) - 0.5
    mesh = ax.pcolormesh(
        t_edges,
        neuron_edges,
        time_by_neuron.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set_xlim(t_edges[0], t_edges[-1])
    ax.set_ylim(n_neurons - 0.5, -0.5)
    return mesh


def time_xlabel():
    if TIME_BIN_MS == 1.0:
        return "time (ms)"
    return f"time (ms; {TIME_BIN_MS:g} ms/bin)"


def add_phase_spans(ax, extra_row, alpha=0.12, axis="x"):
    idx = get_extra_indices(extra_row[None, :])
    for _, start_key, stop_key, color in PHASE_DEFS:
        if idx[start_key] is None:
            continue
        start = int(extra_row[idx[start_key]]) * TIME_BIN_MS
        stop = int(extra_row[idx[stop_key]]) * TIME_BIN_MS
        if axis == "y":
            ax.axhspan(start, stop, color=color, alpha=alpha, lw=0)
        else:
            ax.axvspan(start, stop, color=color, alpha=alpha, lw=0)


def trial_phase_windows(extra_row):
    idx = get_extra_indices(extra_row[None, :])
    windows = []
    for label, start_key, stop_key, color in PHASE_DEFS:
        if idx[start_key] is None:
            continue
        start = int(extra_row[idx[start_key]]) * TIME_BIN_MS
        stop = int(extra_row[idx[stop_key]]) * TIME_BIN_MS
        windows.append((label, start, stop, color))
    return windows


def phase_color_for_time(time_value, windows, default="0.72"):
    for _, start, stop, color in windows:
        if start <= time_value < stop:
            return color
    return default


def plot_phase_colored_trajectory(
    ax, pc_trial, extra_row, lw=1.8, alpha=1.0, zorder=2, linestyle="solid"
):
    t = time_axis(pc_trial.shape[0])
    points = pc_trial[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    windows = trial_phase_windows(extra_row)
    segment_times = 0.5 * (t[:-1] + t[1:])
    colors = [phase_color_for_time(time_value, windows) for time_value in segment_times]
    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=lw,
        alpha=alpha,
        zorder=zorder,
        linestyles=linestyle,
    )
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


# Linestyle key for the four (cue1, cue2) trial types in ChaoticDelayedMatching.
CUE_PAIR_LINESTYLE = {
    (0, 0): ("solid", "A-A (match)"),
    (1, 1): ((0, (4, 1.5)), "B-B (match)"),
    (0, 1): ((0, (1.5, 1.5)), "A-B (non-match)"),
    (1, 0): ((0, (3, 1.5, 1, 1.5)), "B-A (non-match)"),
}

# End-point marker shape per cue-pair type. Used in the TT latent-trajectory
# panel, where the lines are already phase-colored along their length (so a
# per-type linestyle is hard to read); instead the marker at the end of each
# trajectory encodes the trial type.
CUE_PAIR_ENDMARKER = {
    (0, 0): ("o", "A-A (match)"),
    (1, 1): ("s", "B-B (match)"),
    (0, 1): ("^", "A-B (non-match)"),
    (1, 0): ("D", "B-A (non-match)"),
}


def cue_pair_for_trial(extra_row):
    """Return (cue1_id, cue2_id) for one trial, or None if metadata missing."""
    idx = get_extra_indices(np.atleast_2d(extra_row))
    c1, c2 = idx.get("cue1_id"), idx.get("cue2_id")
    if c1 is None or c2 is None or extra_row.shape[0] <= max(c1, c2):
        return None
    return int(extra_row[c1]), int(extra_row[c2])


def choose_neurons_by_variance(array, n_neurons):
    flat_var = np.nanvar(array, axis=(0, 1))
    n_neurons = min(int(n_neurons), array.shape[-1])
    return np.argsort(flat_var)[::-1][:n_neurons]


def rollout_model(model, input_tensor, initial_hidden):
    outputs = []
    latents = []
    hidden = initial_hidden
    for t in range(input_tensor.shape[1]):
        output, hidden = model(input_tensor[:, t, :], hidden)
        outputs.append(output)
        latents.append(hidden)
    return torch.stack(outputs, dim=1), torch.stack(latents, dim=1)


def model_rates(model, latent_tensor):
    if hasattr(model, "_rates"):
        return model._rates(latent_tensor)
    if hasattr(model, "act"):
        return model.act(latent_tensor)
    return latent_tensor


def simulate_spikes_from_rates(rate_like, seed=0, min_rate=0.02, max_rate=0.55):
    rate_like = np.asarray(rate_like)
    lo = np.nanpercentile(rate_like, 1)
    hi = np.nanpercentile(rate_like, 99)
    denom = max(hi - lo, 1e-8)
    norm = np.clip((rate_like - lo) / denom, 0.0, 1.0)
    poisson_rate = min_rate + (max_rate - min_rate) * norm
    rng = np.random.default_rng(seed)
    spikes = rng.poisson(poisson_rate).astype(np.float32)
    return spikes, poisson_rate


def make_ic_perturbations(
    hidden0,
    pca,
    n_perturbations,
    scale,
    mode,
    generator,
    leading_dir=None,
    lyap_jitter=0.0,
):
    """IC perturbations (n_perturbations, n_trials, D), each of norm ``scale``.

    Seeding modes:
      "pc12"     -- evenly spaced ring in the plane of the top two latent PCs.
      "lyapunov" -- a tight cone around the leading Lyapunov vector
                    ``leading_dir`` (the direction that grows fastest over the
                    trajectory); ``lyap_jitter`` sets the cone half-width as a
                    fraction of the unit direction.
      anything else -- isotropic random directions.
    """
    if mode == "pc12":
        pc_dirs = torch.as_tensor(
            pca.components_[:2], dtype=hidden0.dtype, device=hidden0.device
        )
        theta = torch.linspace(
            0, 2 * torch.pi, n_perturbations + 1, device=hidden0.device
        )[:-1]
        delta = (
            torch.cos(theta)[:, None] * pc_dirs[0]
            + torch.sin(theta)[:, None] * pc_dirs[1]
        )
        delta = delta[:, None, :].expand(n_perturbations, hidden0.shape[0], -1).clone()
    elif mode == "lyapunov":
        if leading_dir is None:
            raise ValueError(
                "make_ic_perturbations: 'lyapunov' mode requires leading_dir"
            )
        v = torch.as_tensor(
            leading_dir, dtype=hidden0.dtype, device=hidden0.device
        ).reshape(-1)
        v = v / torch.linalg.norm(v).clamp_min(1e-12)
        jitter = torch.randn(
            n_perturbations,
            hidden0.shape[-1],
            generator=generator,
            device=hidden0.device,
            dtype=hidden0.dtype,
        )
        dirs = v[None, :] + float(lyap_jitter) * jitter
        delta = dirs[:, None, :].expand(n_perturbations, hidden0.shape[0], -1).clone()
    else:
        delta = torch.randn(
            n_perturbations,
            hidden0.shape[0],
            hidden0.shape[-1],
            generator=generator,
            device=hidden0.device,
        )
    delta = (
        scale * delta / torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
    )
    return delta


def leading_growth_direction(jacobians, n_iter=50, seed=0):
    """Unit IC-space direction that grows fastest over a trajectory.

    Power iteration for the top right-singular vector of the linearized
    propagator ``M = J_{T-1} ... J_0`` -- i.e. the finite-time leading Lyapunov
    vector at ``t=0``, the direction the maximal Lyapunov exponent actually
    describes. Per-step renormalization keeps the iteration stable under
    (near-)chaotic growth. ``jacobians`` is ``(T, D, D)``, the discrete state
    Jacobians along one trajectory.
    """
    T, D, _ = jacobians.shape
    gen = torch.Generator(device=jacobians.device).manual_seed(int(seed))
    v = torch.randn(D, generator=gen, device=jacobians.device, dtype=jacobians.dtype)
    v = v / torch.linalg.norm(v).clamp_min(1e-12)
    for _ in range(int(n_iter)):
        w = v
        for t in range(T):  # forward: direction of M v
            w = jacobians[t] @ w
            w = w / torch.linalg.norm(w).clamp_min(1e-12)
        for t in range(T - 1, -1, -1):  # backward: direction of M^T (M v)
            w = jacobians[t].transpose(-1, -2) @ w
            w = w / torch.linalg.norm(w).clamp_min(1e-12)
        v = w
    return v


def _single_trial_state_jacobians(z_all, u_all, cell, trial):
    """Discrete state Jacobians ``(T, D, D)`` along one trajectory."""
    idx = int(trial)
    Jz, _, _ = compute_jacobians(
        z=z_all[idx : idx + 1], u=u_all[idx : idx + 1], f=cell, num_trials=None
    )
    return Jz[0]


def tt_leading_lyapunov_vector(analysis, phase, trial, n_iter=50, seed=0):
    """Leading Lyapunov vector at the IC for a task-trained model (latent space).

    Mirrors the input construction in ``Analysis_TT.compute_lyapunov_exp`` so the
    direction matches the exponent reported in the figure.
    """
    outputs = analysis.get_model_outputs(phase=phase)
    latents = outputs["latents"]
    states = outputs.get("states") if isinstance(outputs, dict) else None
    inputs = analysis.get_model_inputs(phase=phase)[1]
    combined = torch.cat([states, inputs], dim=-1) if states is not None else inputs
    model = analysis.wrapper.model
    cell = model.generator if hasattr(model, "generator") else model.cell
    Jz = _single_trial_state_jacobians(latents, combined, cell, trial)
    return leading_growth_direction(Jz, n_iter=n_iter, seed=seed)


def dd_leading_lyapunov_vector(analysis, phase, trial, n_iter=50, seed=0):
    """Leading Lyapunov vector at the IC for a data-trained model (latent space).

    Mirrors the input construction in ``Analysis_DD.compute_lyapunov_exp``.
    """
    latents = analysis.get_latents(phase=phase)
    inputs = analysis.get_inputs(phase=phase)
    cell = analysis.get_dynamics_model()
    Jz = _single_trial_state_jacobians(latents, inputs, cell, trial)
    return leading_growth_direction(Jz, n_iter=n_iter, seed=seed)


def path_label(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def latent_dim_from_model_or_path(analysis, path):
    for attr_path in (
        ("latent_size",),
        ("hparams", "latent_size"),
        ("hparams", "model", "latent_size"),
    ):
        value = analysis.model
        for attr in attr_path:
            if isinstance(value, dict):
                value = value.get(attr)
            else:
                value = getattr(value, attr, None)
            if value is None:
                break
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    match = re.search(r"(?:model_)?latent_size[=,_-](\d+)", path.name)
    if match:
        return int(match.group(1))
    return np.nan


def pad_pc(pc_array):
    if pc_array.shape[-1] >= 3:
        return pc_array
    pad_width = [(0, 0)] * pc_array.ndim
    pad_width[-1] = (0, 3 - pc_array.shape[-1])
    return np.pad(pc_array, pad_width, mode="constant")


def load_dd_node_analysis(path):
    from ctd.comparison.analysis.dd.dd import Analysis_DD

    required = [path / "model.pkl", path / "datamodule.pkl"]
    if not all(p.exists() for p in required):
        return None, f"DD-NODE run not found:\n{path_label(path)}"
    try:
        analysis = Analysis_DD.create(
            run_name=path.name, filepath=f"{path}/", model_type="SAE"
        )
    except Exception as exc:
        return None, f"Could not load DD-NODE run:\n{type(exc).__name__}: {exc}"
    analysis.model.eval()
    return analysis, None


install_compatibility_stubs()
configure_plotting()
os.environ.setdefault("HOME_DIR", str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
torch.set_grad_enabled(False)


# %%
# --------------------------- Load analyses -----------------------


def compute_payload():
    """Load the analyses and run the model rollouts the figure needs.

    Returns a dict of exactly the module-level values the plotting code
    below consumes. This is the expensive part of the script (model
    unpickling, inference, perturbation rollouts, Lyapunov estimation) and
    is what the on-disk cache stores. Helper functions defined inside are
    used only here.
    """
    missing = [
        name
        for name in ("model.pkl", "datamodule_sim.pkl", "simulator.pkl")
        if not (TT_RUN_PATH / name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {TT_RUN_PATH}")

    tt_analysis = Analysis_TT(run_name=TT_RUN_PATH.name, filepath=f"{TT_RUN_PATH}/")
    tt_analysis.wrapper.eval()
    if hasattr(tt_analysis.model, "noise_level"):
        tt_analysis.model.noise_level = 0.0

    tt_out = tt_analysis.get_model_outputs(phase=PHASE)
    ics, inputs, targets = tt_analysis.get_model_inputs(phase=PHASE)
    extra = tt_analysis.get_extra_inputs(phase=PHASE)
    controlled = tt_out["controlled"]
    latents = tt_out["latents"]

    inputs_np = to_numpy(inputs)
    targets_np = to_numpy(targets)
    extra_np = to_numpy(extra)
    controlled_np = to_numpy(controlled)
    latents_np = to_numpy(latents)

    latent_pca = PCA(n_components=3)
    latents_pc = latent_pca.fit_transform(
        latents_np.reshape(-1, latents_np.shape[-1])
    ).reshape(latents_np.shape[0], latents_np.shape[1], 3)

    tt_rate_like = to_numpy(
        model_rates(
            tt_analysis.model,
            latents.to(device=next(tt_analysis.model.parameters()).device),
        )
    )
    sim_spikes, sim_rates = simulate_spikes_from_rates(tt_rate_like, seed=RANDOM_SEED)

    # --------------------------- Derived data ---------------------------
    def phase_slice(n_trials, phase):
        split = int(0.8 * n_trials)
        if phase == "train":
            return slice(0, split)
        if phase == "val":
            return slice(split, n_trials)
        if phase == "all":
            return slice(0, n_trials)
        raise ValueError(f"Unknown phase: {phase}")

    def select_trial(extra_array, selector, default=0):
        if isinstance(selector, str):
            if extra_array.shape[1] < 13:
                return int(default)
            nonmatch = extra_array[:, 12].astype(bool)
            if selector.lower() == "nonmatch":
                candidates = np.flatnonzero(nonmatch)
            elif selector.lower() == "match":
                candidates = np.flatnonzero(~nonmatch)
            else:
                raise ValueError(f"Unknown trial selector: {selector}")
            if len(candidates) == 0:
                return int(default)
            return int(candidates[0])
        return int(np.clip(int(selector), 0, extra_array.shape[0] - 1))

    resp_on_col, resp_off_col = response_window_cols(extra)
    trial_idx = select_trial(extra_np, EXAMPLE_TRIAL)
    ic_trial_idx = select_trial(extra_np, IC_TRIAL)

    rng = np.random.default_rng(RANDOM_SEED)
    device = next(tt_analysis.model.parameters()).device
    plot_inputs = inputs.to(device)

    # In "lyapunov" mode, seed the TT perturbations along the model's leading
    # Lyapunov vector for the example trial; fall back to "pc12" if unavailable.
    tt_leading_dir = None
    tt_seed_mode = IC_PERTURB_MODE
    if IC_PERTURB_MODE == "lyapunov":
        try:
            tt_leading_dir = tt_leading_lyapunov_vector(
                tt_analysis,
                PHASE,
                ic_trial_idx,
                n_iter=int(IC_PERTURB_LYAP_NITER),
                seed=RANDOM_SEED,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"TT leading Lyapunov vector unavailable "
                f"({type(exc).__name__}: {exc}); falling back to pc12 seeding."
            )
            tt_seed_mode = "pc12"

    with torch.no_grad():
        torch.manual_seed(RANDOM_SEED)
        if hasattr(tt_analysis.model, "init_hidden"):
            hidden0 = tt_analysis.model.init_hidden(plot_inputs.shape[0]).to(device)
        else:
            hidden0 = torch.zeros(
                plot_inputs.shape[0], tt_analysis.model.latent_size, device=device
            )

        perturb_generator = torch.Generator(device=device)
        perturb_generator.manual_seed(RANDOM_SEED)
        d_hidden0 = make_ic_perturbations(
            hidden0,
            latent_pca,
            int(N_IC_PERTURBATIONS),
            float(IC_PERTURB_SCALE),
            tt_seed_mode,
            perturb_generator,
            leading_dir=tt_leading_dir,
            lyap_jitter=float(IC_PERTURB_LYAP_JITTER),
        )
        ensemble_inputs = (
            plot_inputs.unsqueeze(0)
            .expand(N_IC_PERTURBATIONS, -1, -1, -1)
            .reshape(
                N_IC_PERTURBATIONS * plot_inputs.shape[0],
                plot_inputs.shape[1],
                plot_inputs.shape[2],
            )
        )
        ensemble_h0 = (hidden0.unsqueeze(0) + d_hidden0).reshape(
            N_IC_PERTURBATIONS * hidden0.shape[0],
            hidden0.shape[-1],
        )
        pert_out_flat, pert_lat_flat = rollout_model(
            tt_analysis.model, ensemble_inputs, ensemble_h0
        )

    pert_out = pert_out_flat.reshape(
        N_IC_PERTURBATIONS, plot_inputs.shape[0], *pert_out_flat.shape[1:]
    )
    pert_lat = pert_lat_flat.reshape(
        N_IC_PERTURBATIONS, plot_inputs.shape[0], *pert_lat_flat.shape[1:]
    )
    pert_out_np = to_numpy(pert_out)
    pert_lat_np = to_numpy(pert_lat)
    pert_pc = latent_pca.transform(
        pert_lat_np.reshape(-1, pert_lat_np.shape[-1])
    ).reshape(
        N_IC_PERTURBATIONS,
        plot_inputs.shape[0],
        plot_inputs.shape[1],
        3,
    )
    pert_h0_np = to_numpy(hidden0.unsqueeze(0) + d_hidden0)
    base_h0_pc = latent_pca.transform(to_numpy(hidden0))
    pert_h0_pc = latent_pca.transform(
        pert_h0_np.reshape(-1, pert_h0_np.shape[-1])
    ).reshape(
        N_IC_PERTURBATIONS,
        plot_inputs.shape[0],
        3,
    )
    latent_delta = np.linalg.norm(pert_lat_np - latents_np[None, :, :, :], axis=-1)
    output_delta = np.linalg.norm(pert_out_np - controlled_np[None, :, :, :], axis=-1)

    def compute_dd_summary(tt_latents_for_dd, dd_latents, true_rates, pred_rates):
        n_rate_trials = min(true_rates.shape[0], pred_rates.shape[0])
        n_rate_time = min(true_rates.shape[1], pred_rates.shape[1])
        n_rate_neurons = min(true_rates.shape[2], pred_rates.shape[2])
        rate_r2 = variance_weighted_r2(
            true_rates[:n_rate_trials, :n_rate_time, :n_rate_neurons],
            pred_rates[:n_rate_trials, :n_rate_time, :n_rate_neurons],
        )

        n_pair_trials = min(tt_latents_for_dd.shape[0], dd_latents.shape[0])
        n_pair_time = min(tt_latents_for_dd.shape[1], dd_latents.shape[1])
        tt_pair = tt_latents_for_dd[:n_pair_trials, :n_pair_time]
        dd_pair = dd_latents[:n_pair_trials, :n_pair_time]
        if n_pair_trials >= 4:
            state_split = int(0.8 * n_pair_trials)
            state_split = min(max(state_split, 1), n_pair_trials - 1)
            state_reg = LinearRegression()
            state_reg.fit(
                tt_pair[:state_split].reshape(-1, tt_pair.shape[-1]),
                dd_pair[:state_split].reshape(-1, dd_pair.shape[-1]),
            )
            dd_state_pred = state_reg.predict(
                tt_pair[state_split:].reshape(-1, tt_pair.shape[-1])
            ).reshape(dd_pair[state_split:].shape)
            state_r2 = variance_weighted_r2(dd_pair[state_split:], dd_state_pred)
        else:
            state_r2 = np.nan
        return float(rate_r2), float(state_r2)

    def prepare_dd_node_data(path):
        dd_analysis, message = load_dd_node_analysis(path)
        if dd_analysis is None:
            return {}, message
        try:
            dd_spikes = to_numpy(dd_analysis.get_spiking(phase=DD_NODE_PHASE))
            dd_true_rates = to_numpy(dd_analysis.get_true_rates(phase=DD_NODE_PHASE))
            dd_rates = to_numpy(dd_analysis.get_rates(phase=DD_NODE_PHASE))
            print(f"DD-NODE predicted firing rates shape: {dd_rates.shape}")
            dd_latents = to_numpy(dd_analysis.get_latents(phase=DD_NODE_PHASE))
            dd_inputs = to_numpy(dd_analysis.get_inputs(phase=DD_NODE_PHASE))

            tt_dd_slice = phase_slice(latents_np.shape[0], DD_NODE_PHASE)
            tt_latents_for_dd = latents_np[tt_dd_slice]
            extra_for_dd = extra_np[tt_dd_slice]
            if DD_NODE_TRIAL == "same_as_panel_b":
                phase_start = tt_dd_slice.start or 0
                dd_trial_idx = int(
                    np.clip(trial_idx - phase_start, 0, dd_spikes.shape[0] - 1)
                )
            else:
                dd_trial_idx = int(
                    np.clip(int(DD_NODE_TRIAL), 0, dd_spikes.shape[0] - 1)
                )

            n_pair_trials = min(tt_latents_for_dd.shape[0], dd_latents.shape[0])
            n_pair_time = min(tt_latents_for_dd.shape[1], dd_latents.shape[1])
            tt_pair = tt_latents_for_dd[:n_pair_trials, :n_pair_time]
            dd_pair = dd_latents[:n_pair_trials, :n_pair_time]
            tt_design = tt_pair.reshape(-1, tt_pair.shape[-1])
            tt_design = np.c_[tt_design, np.ones(tt_design.shape[0])]
            dd_target = dd_pair.reshape(-1, dd_pair.shape[-1])
            tt_to_dd, *_ = np.linalg.lstsq(tt_design, dd_target, rcond=None)

            n_pc = min(3, dd_latents.shape[-1])
            dd_pca = PCA(n_components=n_pc)
            dd_pc = dd_pca.fit_transform(
                dd_latents.reshape(-1, dd_latents.shape[-1])
            ).reshape(dd_latents.shape[0], dd_latents.shape[1], n_pc)
            tt_aligned = (
                np.c_[
                    tt_latents_for_dd.reshape(-1, tt_latents_for_dd.shape[-1]),
                    np.ones(tt_latents_for_dd.shape[0] * tt_latents_for_dd.shape[1]),
                ]
                @ tt_to_dd
            ).reshape(
                tt_latents_for_dd.shape[0],
                tt_latents_for_dd.shape[1],
                dd_latents.shape[-1],
            )
            tt_aligned_pc = dd_pca.transform(
                tt_aligned.reshape(-1, tt_aligned.shape[-1])
            ).reshape(tt_aligned.shape[0], tt_aligned.shape[1], n_pc)

            rate_r2, state_r2 = compute_dd_summary(
                tt_latents_for_dd, dd_latents, dd_true_rates, dd_rates
            )
            latent_dim = latent_dim_from_model_or_path(dd_analysis, path)
            metrics = {
                "path": path,
                "latent_dim": latent_dim,
                "rate_r2": rate_r2,
                "state_r2": state_r2,
            }
            print("DD-NODE example metrics:", metrics)

            # ---- DD-NODE IC perturbations (analogous to TT panel C). ----
            pert_payload = {}
            try:
                pert_payload = compute_dd_ic_perturbations(
                    dd_analysis,
                    dd_pca,
                    ic_trial_idx,
                    phase_start=(tt_dd_slice.start or 0),
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[chaotic figure] DD IC perturbations unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )

            # ---- DD-NODE maximal Lyapunov exponent. ----
            try:
                les_mean, les_std = dd_analysis.compute_lyapunov_exp(
                    phase=DD_NODE_PHASE, subset_frac=LYAP_SUBSET_FRAC
                )
                # Report the maximal exponent and the std AT that exponent's
                # position, not max(std) over the whole spectrum (which would
                # borrow the trial-to-trial spread of a different exponent).
                les_mean_np = to_numpy(les_mean)
                les_std_np = to_numpy(les_std)
                dd_max_idx = int(les_mean_np.argmax())
                metrics["lyap_max_dd"] = float(les_mean_np[dd_max_idx])
                metrics["lyap_max_dd_std"] = float(les_std_np[dd_max_idx])
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[chaotic figure] DD Lyapunov exponent unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                metrics["lyap_max_dd"] = float("nan")
                metrics["lyap_max_dd_std"] = float("nan")

            return {
                "spikes": dd_spikes,
                "true_rates": dd_true_rates,
                "rates": dd_rates,
                "latents_pc": pad_pc(dd_pc),
                "tt_aligned_pc": pad_pc(tt_aligned_pc),
                "inputs": dd_inputs,
                "extra": extra_for_dd,
                "trial_idx": dd_trial_idx,
                "phase": DD_NODE_PHASE,
                "metrics": metrics,
                "dd_pca": dd_pca,
                "perturbations": pert_payload,
            }, None
        except Exception as exc:
            return {}, f"Could not load DD-NODE outputs:\n{type(exc).__name__}: {exc}"

    def compute_dd_ic_perturbations(
        dd_analysis, dd_pca, ic_trial_idx_global, phase_start
    ):
        """Roll out the DD-NODE decoder from a few random IC perturbations.

        Returns a dict with the unperturbed and perturbed DD latents (in PC space)
        for one example trial and the trial-averaged perturbation-norm growth curve
        on a log scale. Mirrors the TT-side computation that backs panel C of the
        TT figure.
        """
        model = dd_analysis.model
        device = next(model.parameters()).device

        # Standard initial conditions from the encoder on real spike inputs.
        dd_spiking, dd_inputs = dd_analysis.get_model_inputs(phase=DD_NODE_PHASE)
        dd_spiking = dd_spiking.to(device)
        dd_inputs = dd_inputs.to(device)
        with torch.no_grad():
            _, h_n = model.encoder(dd_spiking[:, : model.hparams.encoder_window, :])
            h_n = torch.cat([*h_n], -1)
            ic = model.ic_linear(model.dropout(h_n))
            baseline_lat, _ = model.decoder(dd_inputs, ic)

        n_trials = ic.shape[0]
        ic_trial_local = int(
            np.clip(ic_trial_idx_global - phase_start, 0, n_trials - 1)
        )
        latent_size = ic.shape[-1]

        # IC perturbations, seeded per IC_PERTURB_MODE (matched to the TT side):
        # a ring in the DD top-2 PC plane ("pc12"), a tight cone around the DD
        # leading Lyapunov vector ("lyapunov"), or isotropic random directions.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(RANDOM_SEED))
        n_pert = int(N_IC_PERTURBATIONS)
        dd_seed_mode = IC_PERTURB_MODE
        dd_leading_dir = None
        if IC_PERTURB_MODE == "lyapunov":
            try:
                dd_leading_dir = dd_leading_lyapunov_vector(
                    dd_analysis,
                    DD_NODE_PHASE,
                    ic_trial_local,
                    n_iter=int(IC_PERTURB_LYAP_NITER),
                    seed=RANDOM_SEED,
                ).to(device=device, dtype=ic.dtype)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[chaotic figure] DD leading Lyapunov vector unavailable "
                    f"({type(exc).__name__}: {exc}); falling back to pc12 seeding."
                )
                dd_seed_mode = "pc12"

        if dd_seed_mode == "pc12":
            pc = torch.as_tensor(dd_pca.components_[:2], dtype=ic.dtype, device=device)
            theta = torch.linspace(0, 2 * torch.pi, n_pert + 1, device=device)[:-1]
            dirs = torch.cos(theta)[:, None] * pc[0] + torch.sin(theta)[:, None] * pc[1]
            delta = dirs[:, None, :].expand(n_pert, n_trials, latent_size).clone()
        elif dd_seed_mode == "lyapunov":
            v = dd_leading_dir.reshape(-1)
            v = v / torch.linalg.norm(v).clamp_min(1e-12)
            jitter = torch.randn(
                n_pert, latent_size, generator=gen, device=device, dtype=ic.dtype
            )
            dirs = v[None, :] + float(IC_PERTURB_LYAP_JITTER) * jitter
            delta = dirs[:, None, :].expand(n_pert, n_trials, latent_size).clone()
        else:
            delta = torch.randn(
                n_pert, n_trials, latent_size, generator=gen, device=device
            )
        delta = (
            float(IC_PERTURB_SCALE)
            * delta
            / torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
        )

        pert_ic = (ic.unsqueeze(0) + delta).reshape(n_pert * n_trials, latent_size)
        pert_inputs = (
            dd_inputs.unsqueeze(0)
            .expand(n_pert, -1, -1, -1)
            .reshape(n_pert * n_trials, dd_inputs.shape[1], dd_inputs.shape[2])
        )
        with torch.no_grad():
            pert_lat, _ = model.decoder(pert_inputs, pert_ic)
        pert_lat = pert_lat.reshape(n_pert, n_trials, dd_inputs.shape[1], latent_size)

        baseline_np = to_numpy(baseline_lat)
        pert_np = to_numpy(pert_lat)
        delta_norm = np.linalg.norm(pert_np - baseline_np[None, :, :, :], axis=-1)

        baseline_pc = dd_pca.transform(baseline_np.reshape(-1, latent_size)).reshape(
            baseline_np.shape[0], baseline_np.shape[1], dd_pca.n_components_
        )
        pert_pc = dd_pca.transform(pert_np.reshape(-1, latent_size)).reshape(
            pert_np.shape[0], pert_np.shape[1], pert_np.shape[2], dd_pca.n_components_
        )

        return {
            "baseline_pc": pad_pc(baseline_pc),
            "pert_pc": pad_pc(pert_pc),
            "delta_norm": delta_norm,
            "trial_idx": ic_trial_local,
            "n_perturbations": n_pert,
        }

    dd_data, dd_message = prepare_dd_node_data(DD_NODE_EXAMPLE_RUN_PATH)

    # Maximal Lyapunov exponent of the TT model (for DD-vs-TT comparison panel).
    try:
        tt_les_mean, tt_les_std = tt_analysis.compute_lyapunov_exp(
            phase=PHASE, subset_frac=LYAP_SUBSET_FRAC
        )
        # Report the maximal exponent and the std AT that exponent's position,
        # not max(std) over the whole spectrum (the leading exponent is well
        # constrained even though deeper, strongly-contracting directions have
        # much larger trial-to-trial spread).
        tt_les_mean_np = to_numpy(tt_les_mean)
        tt_les_std_np = to_numpy(tt_les_std)
        tt_max_idx = int(tt_les_mean_np.argmax())
        TT_LYAP_MAX = float(tt_les_mean_np[tt_max_idx])
        TT_LYAP_MAX_STD = float(tt_les_std_np[tt_max_idx])
    except Exception as exc:  # noqa: BLE001
        print(
            f"[chaotic figure] TT Lyapunov exponent unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
        TT_LYAP_MAX = float("nan")
        TT_LYAP_MAX_STD = float("nan")

    return {
        "inputs": inputs,
        "targets_np": targets_np,
        "extra": extra,
        "extra_np": extra_np,
        "inputs_np": inputs_np,
        "controlled_np": controlled_np,
        "latents": latents,
        "latents_pc": latents_pc,
        "sim_spikes": sim_spikes,
        "sim_rates": sim_rates,
        "trial_idx": trial_idx,
        "ic_trial_idx": ic_trial_idx,
        "pert_out_np": pert_out_np,
        "pert_pc": pert_pc,
        "base_h0_pc": base_h0_pc,
        "pert_h0_pc": pert_h0_pc,
        "latent_delta": latent_delta,
        "output_delta": output_delta,
        "dd_data": dd_data,
        "dd_message": dd_message,
        "TT_LYAP_MAX": TT_LYAP_MAX,
        "TT_LYAP_MAX_STD": TT_LYAP_MAX_STD,
    }


# --------------------------- Cache load-or-compute ---------------------------
# Compute-affecting parameters; changing any of these invalidates the cache so
# a stale payload is never silently reused.
_CACHE_SIGNATURE = {
    "phase": PHASE,
    "example_trial": EXAMPLE_TRIAL,
    "dd_node_phase": DD_NODE_PHASE,
    "dd_node_trial": DD_NODE_TRIAL,
    "n_ic_perturbations": int(N_IC_PERTURBATIONS),
    "ic_perturb_scale": float(IC_PERTURB_SCALE),
    "ic_perturb_mode": IC_PERTURB_MODE,
    "ic_perturb_lyap_jitter": float(IC_PERTURB_LYAP_JITTER),
    "ic_perturb_lyap_niter": int(IC_PERTURB_LYAP_NITER),
    "ic_trial": IC_TRIAL,
    "random_seed": int(RANDOM_SEED),
    "lyap_subset_frac": float(LYAP_SUBSET_FRAC),
    "tt_run_path": str(TT_RUN_PATH),
    "dd_run_path": str(DD_NODE_EXAMPLE_RUN_PATH),
}
_PAYLOAD_KEYS = (
    "inputs",
    "targets_np",
    "extra",
    "extra_np",
    "inputs_np",
    "controlled_np",
    "latents",
    "latents_pc",
    "sim_spikes",
    "sim_rates",
    "trial_idx",
    "ic_trial_idx",
    "pert_out_np",
    "pert_pc",
    "base_h0_pc",
    "pert_h0_pc",
    "latent_delta",
    "output_delta",
    "dd_data",
    "dd_message",
    "TT_LYAP_MAX",
    "TT_LYAP_MAX_STD",
)


def _read_cached_payload(path):
    """Return the cached payload if usable, else None (triggering recompute)."""
    if not (USE_CACHE and not RECOMPUTE and path.exists()):
        return None
    try:
        with open(path, "rb") as fh:
            cached = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] ignoring unreadable cache ({exc})")
        return None
    if cached.get("__signature__") != _CACHE_SIGNATURE:
        print("[cache] parameters changed since cache was written; recomputing.")
        return None
    if not all(k in cached for k in _PAYLOAD_KEYS):
        print("[cache] cache is missing expected keys; recomputing.")
        return None
    return cached


def _write_cached_payload(path, payload):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[cache] wrote figure data to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] could not write cache ({exc})")


_cached_payload = _read_cached_payload(CACHE_PATH)
if _cached_payload is not None:
    print(f"[cache] loaded figure data from {CACHE_PATH} (use --recompute to refresh).")
    _payload = _cached_payload
else:
    _payload = compute_payload()
    _payload["__signature__"] = _CACHE_SIGNATURE
    if USE_CACHE:
        _write_cached_payload(CACHE_PATH, _payload)

# Expose the payload as module globals so the plotting code below is unchanged.
globals().update({k: _payload[k] for k in _PAYLOAD_KEYS})


# %%
# --------------------------- Plot panel functions ---------------------------
def panel_task_schematic(ax, title="A  Task structure", title_kwargs=None):
    ax.set_title(title, loc="left", **(title_kwargs or {"fontweight": "bold"}))
    t = time_axis(inputs_np.shape[1])
    ax.set_xlim(t[0], inputs_np.shape[1] * TIME_BIN_MS)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for label, start, stop, color in trial_phase_windows(extra_np[trial_idx]):
        ax.add_patch(
            Rectangle(
                (start, 0.26),
                max(stop - start, TIME_BIN_MS),
                0.46,
                facecolor=color,
                alpha=0.22,
                edgecolor=color,
                lw=1.0,
            )
        )
        ax.text(
            0.5 * (start + stop),
            0.49,
            label,
            ha="center",
            va="center",
            color="0.15",
            fontsize=7,
        )
    ax.text(t[0], 0.06, "Match: A-A/B-B -> -1", color="0.25", ha="left", va="bottom")
    ax.text(
        t[-1], 0.06, "Non-match: A-B/B-A -> +1", color="0.25", ha="right", va="bottom"
    )
    ax.tick_params(axis="x", labelbottom=False)


def panel_io(ax, title=None):
    cue_text = ""
    if extra_np.shape[1] >= 13:
        cue1 = int(extra_np[trial_idx, 10])
        cue2 = int(extra_np[trial_idx, 11])
        trial_type = "non-match" if int(extra_np[trial_idx, 12]) else "match"
        cue_text = f" | trial {trial_idx}: {cue1}-{cue2} {trial_type}"
    ax.set_title(title or f"Inputs and target output{cue_text}", loc="left")
    t = time_axis(inputs_np.shape[1])
    ax.plot(t, inputs_np[trial_idx, :, 0], color="#4C78A8", lw=1.2, label="Cue A")
    ax.plot(t, inputs_np[trial_idx, :, 1], color="#B279A2", lw=1.2, label="Cue B")
    ax.plot(t, targets_np[trial_idx, :, 0], color="0.15", lw=1.4, label="Target")
    add_phase_spans(ax, extra_np[trial_idx], alpha=0.08)
    ax.set_ylabel("amplitude")
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(frameon=False, loc="upper right")


def panel_unperturbed(
    ax_trace, title="B  Trial output (baseline + IC-perturbed rollouts)"
):
    t = time_axis(inputs_np.shape[1])
    ax_trace.set_title(
        title,
        loc="left",
        fontweight="bold",
    )
    add_phase_spans(ax_trace, extra_np[trial_idx], alpha=0.08)
    pert_alpha = min(0.55, max(0.08, 2.5 / max(N_IC_PERTURBATIONS, 1)))
    for k in range(N_IC_PERTURBATIONS):
        ax_trace.plot(
            t,
            pert_out_np[k, trial_idx, :, 0],
            color="#D62728",
            lw=0.75,
            alpha=pert_alpha,
            label="perturbed ICs" if k == 0 else None,
            zorder=2,
        )
    ax_trace.plot(
        t, targets_np[trial_idx, :, 0], color="0.1", lw=1.5, label="target", zorder=4
    )
    ax_trace.plot(
        t,
        controlled_np[trial_idx, :, 0],
        color="#1F77B4",
        lw=1.3,
        label="model",
        zorder=5,
    )
    ax_trace.set_ylabel("output")
    ax_trace.set_xlabel(time_xlabel())
    ax_trace.legend(frameon=False, loc="upper right")


def panel_many_latent_trajectories(ax):
    n_trials = min(int(N_LATENT_TRAJECTORY_TRIALS), latents_pc.shape[0])
    # Stratify the trial sample so all four (cue1, cue2) pairs are represented:
    # uniformly pick from each cue-pair bucket present in the dataset, then top
    # up to n_trials with a uniform sweep across all trials.
    rng_strat = np.random.default_rng(RANDOM_SEED + 1)
    pair_buckets = {}
    for i in range(latents_pc.shape[0]):
        pair = cue_pair_for_trial(extra_np[i])
        if pair is None:
            continue
        pair_buckets.setdefault(pair, []).append(i)
    chosen = []
    if pair_buckets:
        per_bucket = max(1, n_trials // max(len(pair_buckets), 1))
        for pair, bucket in pair_buckets.items():
            if len(bucket) <= per_bucket:
                chosen.extend(bucket)
            else:
                chosen.extend(rng_strat.choice(bucket, size=per_bucket, replace=False))
    sweep = np.linspace(0, latents_pc.shape[0] - 1, n_trials, dtype=int)
    trial_choices = np.unique(np.r_[chosen, sweep, ic_trial_idx]).astype(int)
    line_alpha = min(0.6, max(0.16, 4.5 / max(len(trial_choices), 1)))
    used_pairs = set()
    for trial in trial_choices:
        if trial == ic_trial_idx:
            continue
        pair = cue_pair_for_trial(extra_np[trial])
        ls_entry = CUE_PAIR_LINESTYLE.get(pair)
        linestyle = ls_entry[0] if ls_entry is not None else "solid"
        if pair is not None:
            used_pairs.add(pair)
        plot_phase_colored_trajectory(
            ax,
            latents_pc[trial],
            extra_np[trial],
            lw=0.95,
            alpha=line_alpha,
            zorder=1,
            linestyle=linestyle,
        )
        ax.scatter(
            latents_pc[trial, 0, 0],
            latents_pc[trial, 0, 1],
            color="0.15",
            s=5,
            alpha=0.35,
            zorder=2,
        )
        ax.scatter(
            latents_pc[trial, -1, 0],
            latents_pc[trial, -1, 1],
            color="0.15",
            marker="x",
            s=8,
            alpha=0.35,
            zorder=2,
        )

    ic_pair = cue_pair_for_trial(extra_np[ic_trial_idx])
    ic_ls_entry = CUE_PAIR_LINESTYLE.get(ic_pair)
    ic_linestyle = ic_ls_entry[0] if ic_ls_entry is not None else "solid"
    if ic_pair is not None:
        used_pairs.add(ic_pair)
    plot_phase_colored_trajectory(
        ax,
        latents_pc[ic_trial_idx],
        extra_np[ic_trial_idx],
        lw=2.6,
        alpha=0.95,
        zorder=4,
        linestyle=ic_linestyle,
    )
    ax.scatter(
        latents_pc[ic_trial_idx, 0, 0],
        latents_pc[ic_trial_idx, 0, 1],
        color="0.05",
        s=18,
        alpha=0.9,
        zorder=5,
    )
    ax.scatter(
        latents_pc[ic_trial_idx, -1, 0],
        latents_pc[ic_trial_idx, -1, 1],
        color="0.05",
        marker="x",
        s=28,
        alpha=0.9,
        zorder=5,
    )
    for label, _, _, color in trial_phase_windows(extra_np[trial_idx]):
        ax.plot([], [], color=color, lw=1.8, label=label)
    # Cue-pair linestyle key (in the canonical order A-A, A-B, B-A, B-B).
    for pair in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        if pair not in used_pairs:
            continue
        linestyle, label = CUE_PAIR_LINESTYLE[pair]
        ax.plot(
            [],
            [],
            color="0.25",
            lw=1.4,
            linestyle=linestyle,
            label=label,
        )
    ax.plot([], [], color="0.15", marker="o", linestyle="None", ms=3, label="start")
    ax.plot([], [], color="0.15", marker="x", linestyle="None", ms=4, label="end")
    ax.plot([], [], color="0.05", lw=2.6, label="perturbation trial")
    ax.set_title(
        f"Task-trained latent trajectories | {len(trial_choices)} trials", loc="left"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="best", ncols=2)


def panel_ic_perturbations(ax_pc2, ax_pc3, ax_growth):
    t = time_axis(inputs_np.shape[1])
    alpha = min(0.7, max(0.08, 3.0 / max(N_IC_PERTURBATIONS, 1)))
    ax_pc2.set_title(
        "D  Initial-condition perturbations", loc="left", fontweight="bold"
    )
    ax_pc2.plot(
        latents_pc[ic_trial_idx, :, 0],
        latents_pc[ic_trial_idx, :, 1],
        color="0.15",
        lw=2.4,
        label="baseline",
    )
    ax_pc2.scatter(
        base_h0_pc[ic_trial_idx, 0], base_h0_pc[ic_trial_idx, 1], color="0.15", s=18
    )
    for k in range(N_IC_PERTURBATIONS):
        label = f"{N_IC_PERTURBATIONS} perturbed ICs" if k == 0 else None
        ax_pc2.plot(
            pert_pc[k, ic_trial_idx, :, 0],
            pert_pc[k, ic_trial_idx, :, 1],
            color="#D62728",
            lw=0.8,
            alpha=alpha,
            label=label,
        )
        ax_pc2.scatter(
            pert_h0_pc[k, ic_trial_idx, 0],
            pert_h0_pc[k, ic_trial_idx, 1],
            color="#D62728",
            s=8,
            alpha=alpha,
        )
    ax_pc2.set_xlabel("PC1")
    ax_pc2.set_ylabel("PC2")
    ax_pc2.legend(frameon=False, loc="best")

    ax_pc3.plot(
        latents_pc[ic_trial_idx, :, 0],
        latents_pc[ic_trial_idx, :, 1],
        latents_pc[ic_trial_idx, :, 2],
        color="0.15",
        lw=2.1,
    )
    for k in range(N_IC_PERTURBATIONS):
        ax_pc3.plot(
            pert_pc[k, ic_trial_idx, :, 0],
            pert_pc[k, ic_trial_idx, :, 1],
            pert_pc[k, ic_trial_idx, :, 2],
            color="#D62728",
            lw=0.7,
            alpha=alpha,
        )
    ax_pc3.set_title("top 3 PCs")
    ax_pc3.set_xlabel("PC1")
    ax_pc3.set_ylabel("PC2")
    ax_pc3.set_zlabel("PC3")

    for k in range(N_IC_PERTURBATIONS):
        ax_growth.plot(
            t, latent_delta[k, ic_trial_idx], color="#9467BD", lw=0.6, alpha=alpha
        )
    ax_growth.plot(
        t,
        latent_delta[:, ic_trial_idx].mean(axis=0),
        color="#9467BD",
        lw=1.8,
        label="mean latent",
    )
    ax_growth.set_yscale("log")
    ax_growth.set_xlabel(time_xlabel())
    ax_growth.set_ylabel("norm")
    ax_growth.set_title("perturbation growth")
    ax_growth.legend(frameon=False, loc="best")


def panel_four_latent_trajectories(ax, title="TT latent trajectories"):
    """TT latent trajectories: one example trial from each (cue1, cue2) type.

    Plots exactly four trajectories (A-A, A-B, B-A, B-B) in the top two latent
    PCs, phase-colored along the trajectory. All lines are solid; the trial
    type is encoded by the marker shape at the end of each trajectory (the
    phase coloring makes per-type linestyles hard to read).
    """
    # First trial encountered for each of the four cue-pair types.
    trials_by_pair = {}
    for i in range(latents_pc.shape[0]):
        pair = cue_pair_for_trial(extra_np[i])
        if pair in CUE_PAIR_LINESTYLE and pair not in trials_by_pair:
            trials_by_pair[pair] = i
        if len(trials_by_pair) == 4:
            break

    for pair in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        if pair not in trials_by_pair:
            continue
        trial = trials_by_pair[pair]
        end_marker = CUE_PAIR_ENDMARKER[pair][0]
        plot_phase_colored_trajectory(
            ax,
            latents_pc[trial],
            extra_np[trial],
            lw=1.8,
            alpha=0.95,
            zorder=3,
            linestyle="solid",
        )
        # Start: a small neutral dot, identical for every trial. End: the
        # per-trial-type marker shape, which is what distinguishes the cue pairs.
        ax.scatter(
            latents_pc[trial, 0, 0],
            latents_pc[trial, 0, 1],
            color="0.55",
            s=14,
            zorder=4,
        )
        ax.scatter(
            latents_pc[trial, -1, 0],
            latents_pc[trial, -1, 1],
            color="0.15",
            marker=end_marker,
            s=46,
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )

    # Legend: phase colors, then the four cue-pair end markers, then the start.
    legend_ref = next(iter(trials_by_pair.values()), trial_idx)
    for label, _, _, color in trial_phase_windows(extra_np[legend_ref]):
        ax.plot([], [], color=color, lw=1.8, label=label)
    for pair in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        if pair not in trials_by_pair:
            continue
        end_marker, label = CUE_PAIR_ENDMARKER[pair]
        ax.plot(
            [],
            [],
            color="0.15",
            marker=end_marker,
            linestyle="None",
            ms=6,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
    ax.plot([], [], color="0.55", marker="o", linestyle="None", ms=4, label="start")

    ax.set_title(title, loc="left")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="best", ncols=2)


def panel_tt_pc_perturbations(ax_pc2, ax_pc3, title="Perturbed trajectories in PCs"):
    """Initial-condition perturbations in the top 2 PCs (left) and top 3 (right)."""
    alpha = min(0.7, max(0.08, 3.0 / max(N_IC_PERTURBATIONS, 1)))
    ax_pc2.set_title(title, loc="left")
    ax_pc2.plot(
        latents_pc[ic_trial_idx, :, 0],
        latents_pc[ic_trial_idx, :, 1],
        color="0.15",
        lw=2.4,
        label="baseline",
    )
    ax_pc2.scatter(
        base_h0_pc[ic_trial_idx, 0], base_h0_pc[ic_trial_idx, 1], color="0.15", s=18
    )
    for k in range(N_IC_PERTURBATIONS):
        label = f"{N_IC_PERTURBATIONS} perturbed ICs" if k == 0 else None
        ax_pc2.plot(
            pert_pc[k, ic_trial_idx, :, 0],
            pert_pc[k, ic_trial_idx, :, 1],
            color="#D62728",
            lw=0.8,
            alpha=alpha,
            label=label,
        )
        ax_pc2.scatter(
            pert_h0_pc[k, ic_trial_idx, 0],
            pert_h0_pc[k, ic_trial_idx, 1],
            color="#D62728",
            s=8,
            alpha=alpha,
        )
    ax_pc2.set_xlabel("PC1")
    ax_pc2.set_ylabel("PC2")
    ax_pc2.legend(frameon=False, loc="best")

    ax_pc3.plot(
        latents_pc[ic_trial_idx, :, 0],
        latents_pc[ic_trial_idx, :, 1],
        latents_pc[ic_trial_idx, :, 2],
        color="0.15",
        lw=2.1,
    )
    for k in range(N_IC_PERTURBATIONS):
        ax_pc3.plot(
            pert_pc[k, ic_trial_idx, :, 0],
            pert_pc[k, ic_trial_idx, :, 1],
            pert_pc[k, ic_trial_idx, :, 2],
            color="#D62728",
            lw=0.7,
            alpha=alpha,
        )
    ax_pc3.set_title("top 3 PCs")
    ax_pc3.set_xlabel("PC1")
    ax_pc3.set_ylabel("PC2")
    ax_pc3.set_zlabel("PC3")


def panel_tt_perturbation_growth(ax_growth, title="Perturbation expansion over time"):
    """Trial-averaged growth of the IC perturbation in latent and output space.

    Each space's curves are divided by their own initial (t=0) mean norm so both
    the latent and output traces start at 1.0. Latent and output norms live on
    different scales, so this puts them on a common "relative magnitude" axis and
    the log-slope reflects how fast the perturbation grows or decays in each
    space rather than its absolute size.
    """
    t = time_axis(inputs_np.shape[1])
    alpha = min(0.7, max(0.08, 3.0 / max(N_IC_PERTURBATIONS, 1)))
    ax_growth.set_title(title, loc="left")
    latent_mean = latent_delta[:, ic_trial_idx].mean(axis=0)
    output_mean = output_delta[:, ic_trial_idx].mean(axis=0)
    latent_ref = max(float(latent_mean[0]), 1e-12)
    output_ref = max(float(output_mean[0]), 1e-12)
    for k in range(N_IC_PERTURBATIONS):
        ax_growth.plot(
            t,
            latent_delta[k, ic_trial_idx] / latent_ref,
            color="#9467BD",
            lw=0.6,
            alpha=alpha,
        )
    ax_growth.plot(
        t,
        latent_mean / latent_ref,
        color="#9467BD",
        lw=1.8,
        label="mean latent",
    )
    for k in range(N_IC_PERTURBATIONS):
        ax_growth.plot(
            t,
            output_delta[k, ic_trial_idx] / output_ref,
            color="#1F77B4",
            lw=0.6,
            alpha=alpha,
        )
    ax_growth.plot(
        t,
        output_mean / output_ref,
        color="#1F77B4",
        lw=1.8,
        label="mean output",
    )
    ax_growth.set_yscale("log")
    ax_growth.set_ylim(1e-2, 1e2)
    ax_growth.set_xlabel(time_xlabel())
    ax_growth.set_ylabel("relative perturbation norm")
    ax_growth.legend(frameon=False, loc="best")


def panel_raster(ax_spikes, ax_rates, ax_pred_rates):
    ax_spikes.set_title("C  DD-NODE data-trained fit", loc="left", fontweight="bold")
    if dd_data:
        spikes_for_panel = dd_data["spikes"]
        true_rates_for_panel = dd_data["true_rates"]
        pred_rates = dd_data["rates"]
        extra_for_panel = dd_data["extra"]
    else:
        spikes_for_panel = sim_spikes
        true_rates_for_panel = sim_rates
        pred_rates = None
        extra_for_panel = extra_np

    trial = min(RASTER_TRIAL, spikes_for_panel.shape[0] - 1)
    if pred_rates is not None:
        n_heatmap_neurons = min(true_rates_for_panel.shape[-1], pred_rates.shape[-1])
        neuron_source = true_rates_for_panel[:, :, :n_heatmap_neurons]
    else:
        n_heatmap_neurons = true_rates_for_panel.shape[-1]
        neuron_source = true_rates_for_panel
    neurons = choose_neurons_by_variance(neuron_source, N_RASTER_NEURONS)
    sim_spikes_selected = np.take(spikes_for_panel[trial], neurons, axis=1)
    describe_heatmap_panel(
        "Simulated spikes",
        spikes_for_panel,
        sim_spikes_selected,
        x_axis="selected dim 0 = time",
        y_axis="selected dim 1 = neurons",
    )
    plot_time_neuron_heatmap(
        ax_spikes,
        sim_spikes_selected,
        cmap="Greys",
        vmin=0,
    )
    add_phase_spans(ax_spikes, extra_for_panel[trial], alpha=0.08)
    ax_spikes.set_xlabel(time_xlabel())
    ax_spikes.set_ylabel("neuron")
    sim_rates_selected = np.take(true_rates_for_panel[trial], neurons, axis=1)

    pred_rates_selected = None
    if pred_rates is not None:
        pred_trial = min(trial, pred_rates.shape[0] - 1)
        pred_neurons = neurons
        pred_rates_selected = np.take(pred_rates[pred_trial], pred_neurons, axis=1)
        rate_vmin = np.nanpercentile(
            np.concatenate([sim_rates_selected.ravel(), pred_rates_selected.ravel()]), 1
        )
        rate_vmax = np.nanpercentile(
            np.concatenate([sim_rates_selected.ravel(), pred_rates_selected.ravel()]),
            99,
        )
    else:
        pred_trial = trial
        rate_vmin = np.nanpercentile(sim_rates_selected, 1)
        rate_vmax = np.nanpercentile(sim_rates_selected, 99)

    describe_heatmap_panel(
        "Simulated rates",
        true_rates_for_panel,
        sim_rates_selected,
        x_axis="selected dim 0 = time",
        y_axis="selected dim 1 = neurons",
    )
    plot_time_neuron_heatmap(
        ax_rates,
        sim_rates_selected,
        cmap="magma",
        vmin=rate_vmin,
        vmax=rate_vmax,
    )
    add_phase_spans(ax_rates, extra_for_panel[trial], alpha=0.08)
    ax_rates.set_title("simulated rates")
    if dd_data:
        print("Panel C sim rates source: dd_data['true_rates']")
    else:
        print("Panel C sim rates source: sim_rates")
    ax_rates.set_xlabel(time_xlabel())
    ax_rates.set_ylabel("neuron")

    if pred_rates_selected is None:
        ax_pred_rates.text(
            0.5,
            0.5,
            dd_message or "DD-NODE rates unavailable",
            ha="center",
            va="center",
            wrap=True,
        )
        ax_pred_rates.set_axis_off()
        return
    describe_heatmap_panel(
        "DD-NODE predicted rates",
        pred_rates,
        pred_rates_selected,
        x_axis="selected dim 0 = time",
        y_axis="selected dim 1 = neurons",
    )
    plot_time_neuron_heatmap(
        ax_pred_rates,
        pred_rates_selected,
        cmap="magma",
        vmin=rate_vmin,
        vmax=rate_vmax,
    )
    add_phase_spans(ax_pred_rates, extra_for_panel[pred_trial], alpha=0.08)
    ax_pred_rates.set_title("DD-NODE predicted rates")
    print(
        "Rate heatmap neuron indices shared by simulated/predicted:", neurons.tolist()
    )
    print(
        f"Shared rate heatmap color scale: vmin={rate_vmin:.4g}, vmax={rate_vmax:.4g}"
    )
    ax_pred_rates.set_xlabel(time_xlabel())
    ax_pred_rates.set_ylabel("neuron")


def panel_dd_node(ax_rates, ax_pc):
    ax_rates.set_title("example rate traces")
    if not dd_data:
        ax_rates.text(
            0.5,
            0.5,
            dd_message or "DD-NODE outputs unavailable",
            ha="center",
            va="center",
            wrap=True,
        )
        ax_rates.set_axis_off()
        ax_pc.set_axis_off()
        return

    trial = dd_data["trial_idx"]
    true_rates = dd_data["true_rates"]
    pred_rates = dd_data["rates"]
    neurons = choose_neurons_by_variance(true_rates, 4)
    t = time_axis(true_rates.shape[1])
    offsets = np.arange(len(neurons)) * np.nanpercentile(true_rates[:, :, neurons], 95)
    r2_values = []
    for j, neuron in enumerate(neurons):
        true_trace = true_rates[trial, :, neuron]
        pred_trace = pred_rates[trial, :, neuron]
        r2 = rate_r2_score(true_trace, pred_trace)
        r2_values.append(r2)
        ax_rates.plot(
            t,
            true_trace + offsets[j],
            color="0.15",
            lw=1.2,
            label="true" if j == 0 else None,
        )
        ax_rates.plot(
            t,
            pred_trace + offsets[j],
            color="#D62728",
            lw=1.0,
            alpha=0.85,
            label="DD-NODE" if j == 0 else None,
        )
        ax_rates.text(
            t[-1] + 0.015 * (t[-1] - t[0]),
            offsets[j] + np.nanmedian(true_trace),
            f"n{int(neuron)} R2={r2:.2f}",
            va="center",
            ha="left",
            fontsize=7,
            color="0.2",
            clip_on=False,
        )
    ax_rates.set_xlim(t[0], t[-1] * 1.18)
    print(
        "Panel C example neuron rate R2:",
        {int(n): float(r) for n, r in zip(neurons, r2_values)},
    )
    ax_rates.set_xlabel(time_xlabel())
    ax_rates.set_ylabel("rate + offset")
    ax_rates.legend(frameon=False, loc="upper right")

    pc = dd_data["latents_pc"]
    tt_pc = dd_data["tt_aligned_pc"]
    ax_pc.plot(
        tt_pc[trial, :, 0],
        tt_pc[trial, :, 1],
        tt_pc[trial, :, 2],
        color="0.15",
        lw=1.2,
        label="TT aligned",
    )
    ax_pc.plot(
        pc[trial, :, 0],
        pc[trial, :, 1],
        pc[trial, :, 2],
        color="#D62728",
        lw=1.2,
        label="DD-NODE",
    )
    ax_pc.scatter(
        pc[trial, 0, 0], pc[trial, 0, 1], pc[trial, 0, 2], color="#D62728", s=16
    )
    ax_pc.scatter(
        pc[trial, -1, 0],
        pc[trial, -1, 1],
        pc[trial, -1, 2],
        color="#D62728",
        marker="x",
        s=20,
    )
    ax_pc.scatter(
        tt_pc[trial, 0, 0], tt_pc[trial, 0, 1], tt_pc[trial, 0, 2], color="0.15", s=16
    )
    ax_pc.scatter(
        tt_pc[trial, -1, 0],
        tt_pc[trial, -1, 1],
        tt_pc[trial, -1, 2],
        color="0.15",
        marker="x",
        s=20,
    )
    ax_pc.set_title(f"DD-NODE latent trajectory | {dd_data['phase']} trial {trial}")
    ax_pc.legend(frameon=False, loc="best")
    ax_pc.set_xlabel("PC1")
    ax_pc.set_ylabel("PC2")
    ax_pc.set_zlabel("PC3")


def panel_dd_metrics(ax):
    """Bar chart of Rate R^2 and State R^2 for the example DD-NODE fit."""
    ax.set_title("B  Reconstruction metrics", loc="left", fontweight="bold")
    if not dd_data:
        ax.text(0.5, 0.5, "DD-NODE outputs unavailable", ha="center", va="center")
        ax.set_axis_off()
        return
    metrics = dd_data.get("metrics", {}) or {}
    labels = [r"Rate $R^2$", r"State $R^2$"]
    values = [
        metrics.get("rate_r2", float("nan")),
        metrics.get("state_r2", float("nan")),
    ]
    xs = np.arange(len(labels))
    colors = ["#4C78A8", "#54A24B"]
    ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="0.75", linewidth=0.5, linestyle="--")
    for x, v in zip(xs, values):
        if np.isfinite(v):
            ax.text(
                x, min(v + 0.03, 1.02), f"{v:.2f}", ha="center", va="bottom", fontsize=8
            )
    ax.spines[["top", "right"]].set_visible(False)


def panel_dd_lyap(ax):
    """Bar chart comparing maximal Lyapunov exponent for TT and DD-NODE."""
    ax.set_title("C  Maximal Lyapunov exponent", loc="left", fontweight="bold")
    metrics = (dd_data or {}).get("metrics", {}) or {}
    tt_val = TT_LYAP_MAX
    tt_err = TT_LYAP_MAX_STD if np.isfinite(TT_LYAP_MAX_STD) else 0.0
    dd_val = metrics.get("lyap_max_dd", float("nan"))
    dd_err = metrics.get("lyap_max_dd_std", float("nan"))
    dd_err = dd_err if np.isfinite(dd_err) else 0.0
    labels = ["TT", "DD-NODE"]
    values = [tt_val, dd_val]
    errs = [tt_err, dd_err]
    colors = ["0.25", "#D62728"]
    xs = np.arange(len(labels))
    bars = ax.bar(
        xs,
        values,
        yerr=errs,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.7,
        capsize=4,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\lambda_{\max}$ (nats / bin)")
    ax.axhline(0.0, color="0.75", linewidth=0.5)
    finite_vals = [v for v in values if np.isfinite(v)]
    if finite_vals:
        ymin = min(0.0, min(finite_vals) * 1.2)
        ymax = max(0.05, max(finite_vals) * 1.4)
        ax.set_ylim(ymin, ymax)
    for x, v, e in zip(xs, values, errs):
        if np.isfinite(v):
            ax.text(
                x,
                v + (0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.spines[["top", "right"]].set_visible(False)


def panel_dd_ic_perturbations(ax_pc3, ax_growth):
    """DD-NODE IC perturbations: top 3 PCs + perturbation-norm growth.

    Mirrors panel C of the TT figure (initial-condition perturbations) but
    applied to the data-trained NODE: random unit-norm perturbations of the
    encoder-derived ICs are rolled out through the DD-NODE decoder with the
    same input sequence, then projected onto the DD-NODE PC basis.
    """
    pert = (dd_data or {}).get("perturbations", {}) or {}
    if not pert:
        for ax in (ax_pc3, ax_growth):
            ax.text(
                0.5,
                0.5,
                "DD-NODE IC perturbations unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        return

    baseline_pc = pert["baseline_pc"]
    pert_pc = pert["pert_pc"]
    delta_norm = pert["delta_norm"]
    trial = int(pert["trial_idx"])
    n_pert = int(pert["n_perturbations"])
    alpha = min(0.7, max(0.08, 3.0 / max(n_pert, 1)))

    ax_pc3.set_title("D  DD-NODE IC perturbations", loc="left", fontweight="bold")
    ax_pc3.plot(
        baseline_pc[trial, :, 0],
        baseline_pc[trial, :, 1],
        baseline_pc[trial, :, 2],
        color="0.15",
        lw=2.0,
        label="baseline",
    )
    for k in range(n_pert):
        ax_pc3.plot(
            pert_pc[k, trial, :, 0],
            pert_pc[k, trial, :, 1],
            pert_pc[k, trial, :, 2],
            color="#D62728",
            lw=0.8,
            alpha=alpha,
            label=f"{n_pert} perturbed ICs" if k == 0 else None,
        )
    ax_pc3.scatter(
        baseline_pc[trial, 0, 0],
        baseline_pc[trial, 0, 1],
        baseline_pc[trial, 0, 2],
        color="0.15",
        s=18,
    )
    ax_pc3.scatter(
        baseline_pc[trial, -1, 0],
        baseline_pc[trial, -1, 1],
        baseline_pc[trial, -1, 2],
        color="0.15",
        marker="x",
        s=22,
    )
    ax_pc3.set_xlabel("PC1")
    ax_pc3.set_ylabel("PC2")
    ax_pc3.set_zlabel("PC3")
    ax_pc3.legend(frameon=False, loc="best", fontsize=7)

    # Compare DD-NODE vs TT sensitivity to IC perturbations on the same axis.
    # Each curve is divided by its own initial perturbation norm so both start
    # at 1.0 ("same relative magnitude"); the log-slope then reflects how fast
    # nearby trajectories diverge, independent of the (different) latent scales
    # of the TT and DD-NODE state spaces.
    t_dd = time_axis(delta_norm.shape[-1])
    dd_mean = np.maximum(delta_norm[:, trial].mean(axis=0), 1e-12)
    dd_ref = dd_mean[0]
    ax_growth.set_title("perturbation growth (DD vs TT)", loc="left")
    for k in range(n_pert):
        ax_growth.plot(
            t_dd,
            np.maximum(delta_norm[k, trial], 1e-12) / dd_ref,
            color="#D62728",
            lw=0.6,
            alpha=alpha,
        )
    ax_growth.plot(
        t_dd,
        dd_mean / dd_ref,
        color="#D62728",
        lw=1.8,
        label="DD-NODE (mean)",
    )
    # Overlay the TT-side perturbation growth for the matched example trial,
    # normalized the same way so the two start together and only their growth
    # rates are compared.
    if latent_delta is not None and latent_delta.size:
        tt_mean = np.maximum(latent_delta[:, ic_trial_idx].mean(axis=0), 1e-12)
        ax_growth.plot(
            time_axis(tt_mean.shape[0]),
            tt_mean / tt_mean[0],
            color="0.15",
            lw=1.8,
            ls="--",
            label="TT (mean)",
        )
    ax_growth.set_yscale("log")
    ax_growth.set_xlabel(time_xlabel())
    ax_growth.set_ylabel("relative perturbation norm")
    ax_growth.spines[["top", "right"]].set_visible(False)
    ax_growth.legend(frameon=False, loc="best", fontsize=7)


# %%
# --------------------------- Compose figure ---------------------------
fig = plt.figure(figsize=(14.5, 12.6), constrained_layout=False)
outer = GridSpec(
    5,
    4,
    figure=fig,
    height_ratios=[0.58, 0.82, 0.92, 1.25, 1.35],
    hspace=0.62,
    wspace=0.45,
)

ax_task = fig.add_subplot(outer[0, :2])
ax_io = fig.add_subplot(outer[1, :2], sharex=ax_task)
ax_perf = fig.add_subplot(outer[2, :2], sharex=ax_task)
ax_many_latents = fig.add_subplot(outer[0:3, 2:])
panel_task_schematic(ax_task)
panel_io(ax_io)
panel_unperturbed(ax_perf)
panel_many_latent_trajectories(ax_many_latents)

ax_ic2 = fig.add_subplot(outer[3, 0])
ax_ic3 = fig.add_subplot(outer[3, 1], projection="3d")
ax_icg = fig.add_subplot(outer[3, 2:])
panel_ic_perturbations(ax_ic2, ax_ic3, ax_icg)

bottom = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[4, :], wspace=0.5)
ax_raster = fig.add_subplot(bottom[0, 0])
ax_true_rates = fig.add_subplot(bottom[0, 1])
ax_pred_heatmap = fig.add_subplot(bottom[0, 2])
ax_dd_rates = fig.add_subplot(bottom[0, 3])
ax_dd_pc = fig.add_subplot(bottom[0, 4], projection="3d")
panel_raster(ax_raster, ax_true_rates, ax_pred_heatmap)
panel_dd_node(ax_dd_rates, ax_dd_pc)

fig.suptitle(
    TASK_NAME_FOR_FIGURE, x=0.02, y=0.985, ha="left", fontsize=13, fontweight="bold"
)
fig.align_ylabels()

if SAVE_FIGURE:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path.relative_to(REPO_ROOT)}")
    print(f"Saved {pdf_path.relative_to(REPO_ROOT)}")


# %%
# --------------------------- TT-only split figure ---------------------------
# Panels (markers drawn as standalone bold 12 pt Arial text, vertically aligned):
#   A  Task structure and example inputs (schematic + cue/target traces)
#   B  TT latent trajectories -- one example trial per type (AA, AB, BA, BB)
#   C  Perturbed trajectories in output space
#   D  Perturbed trajectories in top 2 PCs (left) and top 3 PCs (right)
#   E  Perturbation expansion over time
fig_tt = plt.figure(figsize=(14.5, 9.6), constrained_layout=False)
outer_tt = GridSpec(
    4,
    4,
    figure=fig_tt,
    height_ratios=[0.58, 0.82, 0.92, 1.25],
    hspace=0.7,
    wspace=0.45,
)

# Panel A: task schematic (top) + example inputs (below), left two columns.
ax_task_tt = fig_tt.add_subplot(outer_tt[0, :2])
ax_io_tt = fig_tt.add_subplot(outer_tt[1, :2], sharex=ax_task_tt)
panel_task_schematic(ax_task_tt, title="Task structure", title_kwargs={})
panel_io(ax_io_tt, title="Example inputs and target output")

# Panel B: TT latent trajectories (one example trial per type), spanning the
# top three rows on the right.
ax_latents_tt = fig_tt.add_subplot(outer_tt[0:3, 2:])
panel_four_latent_trajectories(
    ax_latents_tt, title="TT latent trajectories (one trial per type)"
)

# Panel C: perturbed trajectories in output space, back in the first column.
ax_perf_tt = fig_tt.add_subplot(outer_tt[2, :2], sharex=ax_task_tt)
panel_unperturbed(ax_perf_tt, title="Perturbed trajectories in output space")

# Panels D and E share the bottom row: D = IC-perturbed trajectories in the top
# 2 PCs (left) and top 3 PCs (middle); E = perturbation expansion over time.
ax_ic2_tt = fig_tt.add_subplot(outer_tt[3, 0])
ax_ic3_tt = fig_tt.add_subplot(outer_tt[3, 1], projection="3d")
ax_icg_tt = fig_tt.add_subplot(outer_tt[3, 2:])
panel_tt_pc_perturbations(
    ax_ic2_tt, ax_ic3_tt, title="Perturbed trajectories in top 2 PCs"
)
panel_tt_perturbation_growth(ax_icg_tt, title="Perturbation expansion over time")

# The LaTeX caption provides the figure title; the standalone split figures
# omit the in-figure suptitle to avoid colliding with the top-panel headers.
fig_tt.align_ylabels()

# Standalone bold 12 pt panel markers, placed after layout so each sits at the
# top-left corner of its panel. Same-column panels share a left edge, so the
# markers are aligned vertically.
add_panel_label(fig_tt, ax_task_tt, "A")
add_panel_label(fig_tt, ax_latents_tt, "B")
add_panel_label(fig_tt, ax_perf_tt, "C")
add_panel_label(fig_tt, ax_ic2_tt, "D")
add_panel_label(fig_tt, ax_icg_tt, "E")


# %%
# --------------------------- DD-only split figure ---------------------------
# Row 0: A (DD-NODE data-trained fit -- raster, true rates, predicted rates,
# example traces, aligned latents). Row 1: B (Rate/State R^2 metrics),
# C (maximal Lyapunov exponent vs TT), D (DD-NODE IC perturbations: PC view
# + perturbation growth), mirroring TT panel C.
fig_dd = plt.figure(figsize=(14.5, 6.4), constrained_layout=False)
outer_dd = GridSpec(
    2,
    10,
    figure=fig_dd,
    height_ratios=[1.0, 1.05],
    wspace=1.05,
    hspace=0.55,
)
# Row 0: panel A (5 sub-axes spanning all 10 cols, 2 cols each).
ax_raster_dd = fig_dd.add_subplot(outer_dd[0, 0:2])
ax_true_rates_dd = fig_dd.add_subplot(outer_dd[0, 2:4])
ax_pred_heatmap_dd = fig_dd.add_subplot(outer_dd[0, 4:6])
ax_dd_rates_dd = fig_dd.add_subplot(outer_dd[0, 6:8])
ax_dd_pc_dd = fig_dd.add_subplot(outer_dd[0, 8:10], projection="3d")
panel_raster(ax_raster_dd, ax_true_rates_dd, ax_pred_heatmap_dd)
panel_dd_node(ax_dd_rates_dd, ax_dd_pc_dd)
# Relabel old panel "C" -> "A" for the standalone DD figure.
ax_raster_dd.set_title("A  DD-NODE data-trained fit", loc="left", fontweight="bold")

# Row 1: panels B (metrics), C (Lyapunov DD vs TT), D (IC perturbations
# 3D PCs + growth-norm curve).
ax_metrics_dd = fig_dd.add_subplot(outer_dd[1, 0:2])
ax_lyap_dd = fig_dd.add_subplot(outer_dd[1, 2:4])
ax_pert_pc_dd = fig_dd.add_subplot(outer_dd[1, 4:7], projection="3d")
ax_pert_growth_dd = fig_dd.add_subplot(outer_dd[1, 7:10])
panel_dd_metrics(ax_metrics_dd)
panel_dd_lyap(ax_lyap_dd)
panel_dd_ic_perturbations(ax_pert_pc_dd, ax_pert_growth_dd)

# The LaTeX caption provides the figure title; the standalone split figures
# omit the in-figure suptitle to avoid colliding with the top-panel headers.
fig_dd.align_ylabels()

if SAVE_FIGURE:
    tt_png = OUTPUT_DIR / "ChaoticDelayedMatching_TT.png"
    tt_pdf = OUTPUT_DIR / "ChaoticDelayedMatching_TT.pdf"
    dd_png = OUTPUT_DIR / "ChaoticDelayedMatching_DD.png"
    dd_pdf = OUTPUT_DIR / "ChaoticDelayedMatching_DD.pdf"
    fig_tt.savefig(tt_png, bbox_inches="tight")
    fig_tt.savefig(tt_pdf, bbox_inches="tight")
    fig_dd.savefig(dd_png, bbox_inches="tight")
    fig_dd.savefig(dd_pdf, bbox_inches="tight")
    print(f"Saved {tt_png.relative_to(REPO_ROOT)}")
    print(f"Saved {tt_pdf.relative_to(REPO_ROOT)}")
    print(f"Saved {dd_png.relative_to(REPO_ROOT)}")
    print(f"Saved {dd_pdf.relative_to(REPO_ROOT)}")

if MOVE_TO_MANUSCRIPT:
    if not SAVE_FIGURE:
        warnings.warn(
            "MOVE_TO_MANUSCRIPT requested but SAVE_FIGURE is False; "
            "nothing to copy into the manuscript."
        )
    else:
        copied = copy_to_manuscript([OUTPUT_DIR / name for name in MANUSCRIPT_FIGURES])
        if copied:
            rebuild_manuscript()

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)
    plt.close(fig_tt)
    plt.close(fig_dd)
