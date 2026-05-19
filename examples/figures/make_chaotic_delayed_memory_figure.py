# %%
"""Paper-quality summary figure for the Chaotic Delayed Memory task.

Run this file as a script or cell-by-cell in an editor that supports ``# %%``.
The local benchmark/config name is ``ChaoticDelayedMatching``; figure text uses
"Chaotic Delayed Memory" by default because this is the manuscript-facing name.
"""

import importlib
import os
import sys
import types
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

# %%
# --------------------------- User parameters ---------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_NAME_FOR_FIGURE = "Chaotic Delayed Memory"
TT_RUN_PATH = (
    REPO_ROOT
    / "content"
    / "trained_models"
    / "task-trained"
    / "tt_ChaoticDelayedMatching"
)
LFADS_RUN_PATH = (
    TT_RUN_PATH
    / "20260320_ChaoticDelayedMatching_LFADS_CB"
    / "prefix=tt_ChaoticDelayedMatching_max_epochs=1000"
)
OUTPUT_DIR = REPO_ROOT / "content" / "figures"
OUTPUT_STEM = "chaotic_delayed_memory_summary"

PHASE = "all"
EXAMPLE_TRIAL = (
    "nonmatch"  # int trial index, or "nonmatch"/"match" for automatic selection.
)
RASTER_TRIAL = 2
LFADS_PHASE = PHASE
LFADS_TRIAL = "same_as_panel_b"  # int trial index in LFADS_PHASE, or "same_as_panel_b".
N_RASTER_NEURONS = 48
N_IC_PERTURBATIONS = 16
IC_PERTURB_SCALE = 0.5
IC_PERTURB_MODE = "pc12"  # "pc12" or "random"
IC_TRIAL = 0
RANDOM_SEED = 11
TIME_BIN_MS = (
    1.0  # ChaoticDelayedMatching uses one model step per ms in the default configs.
)

SAVE_FIGURE = True
SHOW_FIGURE = True

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
        sys.modules[
            "pkg_resources._vendor.packaging.requirements"
        ] = packaging.requirements
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
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
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


def plot_phase_colored_trajectory(ax, pc_trial, extra_row, lw=1.8):
    t = time_axis(pc_trial.shape[0])
    points = pc_trial[:, :2].reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    windows = trial_phase_windows(extra_row)
    segment_times = 0.5 * (t[:-1] + t[1:])
    colors = [phase_color_for_time(time_value, windows) for time_value in segment_times]
    lc = LineCollection(segments, colors=colors, linewidths=lw)
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


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


def make_ic_perturbations(hidden0, pca, n_perturbations, scale, mode, generator):
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


def load_lfads_analysis(path):
    from ctd.comparison.analysis.dd.dd import Analysis_DD

    required = [path / "model.pkl", path / "datamodule.pkl"]
    if not all(p.exists() for p in required):
        return None, f"LFADS run not found:\n{path.relative_to(REPO_ROOT)}"
    try:
        analysis = Analysis_DD.create(
            run_name=path.name, filepath=f"{path}/", model_type="LFADS"
        )
    except Exception as exc:
        return None, f"Could not load LFADS run:\n{type(exc).__name__}: {exc}"
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

lfads_analysis, lfads_message = load_lfads_analysis(LFADS_RUN_PATH)


# %%
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
        IC_PERTURB_MODE,
        perturb_generator,
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
pert_pc = latent_pca.transform(pert_lat_np.reshape(-1, pert_lat_np.shape[-1])).reshape(
    N_IC_PERTURBATIONS,
    plot_inputs.shape[0],
    plot_inputs.shape[1],
    3,
)
pert_h0_np = to_numpy(hidden0.unsqueeze(0) + d_hidden0)
base_h0_pc = latent_pca.transform(to_numpy(hidden0))
pert_h0_pc = latent_pca.transform(pert_h0_np.reshape(-1, pert_h0_np.shape[-1])).reshape(
    N_IC_PERTURBATIONS,
    plot_inputs.shape[0],
    3,
)
latent_delta = np.linalg.norm(pert_lat_np - latents_np[None, :, :, :], axis=-1)
output_delta = np.linalg.norm(pert_out_np - controlled_np[None, :, :, :], axis=-1)

lfads_data = {}
fit_metrics = {}
if lfads_analysis is not None:
    try:
        lfads_spikes = to_numpy(lfads_analysis.get_spiking(phase=LFADS_PHASE))
        lfads_true_rates = to_numpy(lfads_analysis.get_true_rates(phase=LFADS_PHASE))
        lfads_rates = to_numpy(lfads_analysis.get_rates(phase=LFADS_PHASE))
        print(f"LFADS predicted firing rates shape: {lfads_rates.shape}")
        lfads_latents = to_numpy(lfads_analysis.get_latents(phase=LFADS_PHASE))
        lfads_inputs = to_numpy(lfads_analysis.get_inputs(phase=LFADS_PHASE))

        tt_lfads_slice = phase_slice(latents_np.shape[0], LFADS_PHASE)
        tt_latents_for_lfads = latents_np[tt_lfads_slice]
        if LFADS_TRIAL == "same_as_panel_b":
            phase_start = tt_lfads_slice.start or 0
            lfads_trial_idx = int(
                np.clip(trial_idx - phase_start, 0, lfads_spikes.shape[0] - 1)
            )
        else:
            lfads_trial_idx = int(
                np.clip(int(LFADS_TRIAL), 0, lfads_spikes.shape[0] - 1)
            )

        n_pair_trials = min(tt_latents_for_lfads.shape[0], lfads_latents.shape[0])
        n_pair_time = min(tt_latents_for_lfads.shape[1], lfads_latents.shape[1])
        tt_pair = tt_latents_for_lfads[:n_pair_trials, :n_pair_time]
        lfads_pair = lfads_latents[:n_pair_trials, :n_pair_time]
        tt_design = tt_pair.reshape(-1, tt_pair.shape[-1])
        tt_design = np.c_[tt_design, np.ones(tt_design.shape[0])]
        lfads_target = lfads_pair.reshape(-1, lfads_pair.shape[-1])
        tt_to_lfads, *_ = np.linalg.lstsq(tt_design, lfads_target, rcond=None)

        lfads_pca = PCA(n_components=3)
        lfads_pc = lfads_pca.fit_transform(
            lfads_latents.reshape(-1, lfads_latents.shape[-1])
        ).reshape(lfads_latents.shape[0], lfads_latents.shape[1], 3)
        tt_aligned = (
            np.c_[
                tt_latents_for_lfads.reshape(-1, tt_latents_for_lfads.shape[-1]),
                np.ones(tt_latents_for_lfads.shape[0] * tt_latents_for_lfads.shape[1]),
            ]
            @ tt_to_lfads
        ).reshape(
            tt_latents_for_lfads.shape[0],
            tt_latents_for_lfads.shape[1],
            lfads_latents.shape[-1],
        )
        tt_aligned_pc = lfads_pca.transform(
            tt_aligned.reshape(-1, tt_aligned.shape[-1])
        ).reshape(tt_aligned.shape[0], tt_aligned.shape[1], 3)

        n_rate_trials = min(lfads_true_rates.shape[0], lfads_rates.shape[0])
        n_rate_time = min(lfads_true_rates.shape[1], lfads_rates.shape[1])
        n_rate_neurons = min(lfads_true_rates.shape[2], lfads_rates.shape[2])
        rate_r2 = variance_weighted_r2(
            lfads_true_rates[:n_rate_trials, :n_rate_time, :n_rate_neurons],
            lfads_rates[:n_rate_trials, :n_rate_time, :n_rate_neurons],
        )

        if n_pair_trials >= 4:
            state_split = int(0.8 * n_pair_trials)
            state_split = min(max(state_split, 1), n_pair_trials - 1)
            state_reg = LinearRegression()
            state_reg.fit(
                tt_pair[:state_split].reshape(-1, tt_pair.shape[-1]),
                lfads_pair[:state_split].reshape(-1, lfads_pair.shape[-1]),
            )
            lfads_state_pred = state_reg.predict(
                tt_pair[state_split:].reshape(-1, tt_pair.shape[-1])
            ).reshape(lfads_pair[state_split:].shape)
            state_r2 = variance_weighted_r2(lfads_pair[state_split:], lfads_state_pred)
        else:
            state_r2 = np.nan

        fit_metrics = {
            "Rate R2": float(rate_r2),
            "State R2": float(state_r2),
        }
        print("LFADS summary metrics:", fit_metrics)

        lfads_data = {
            "spikes": lfads_spikes,
            "true_rates": lfads_true_rates,
            "rates": lfads_rates,
            "latents_pc": lfads_pc,
            "tt_aligned_pc": tt_aligned_pc,
            "inputs": lfads_inputs,
            "trial_idx": lfads_trial_idx,
            "phase": LFADS_PHASE,
        }
    except Exception as exc:
        lfads_message = f"Could not load LFADS outputs:\n{type(exc).__name__}: {exc}"


# %%
# --------------------------- Plot panel functions ---------------------------
def panel_task_schematic(ax):
    ax.set_title("A  Task structure", loc="left", fontweight="bold")
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


def panel_io(ax):
    cue_text = ""
    if extra_np.shape[1] >= 13:
        cue1 = int(extra_np[trial_idx, 10])
        cue2 = int(extra_np[trial_idx, 11])
        trial_type = "non-match" if int(extra_np[trial_idx, 12]) else "match"
        cue_text = f" | trial {trial_idx}: {cue1}-{cue2} {trial_type}"
    ax.set_title(f"Inputs and target output{cue_text}", loc="left")
    t = time_axis(inputs_np.shape[1])
    ax.plot(t, inputs_np[trial_idx, :, 0], color="#4C78A8", lw=1.2, label="Cue A")
    ax.plot(t, inputs_np[trial_idx, :, 1], color="#B279A2", lw=1.2, label="Cue B")
    ax.plot(t, targets_np[trial_idx, :, 0], color="0.15", lw=1.4, label="Target")
    add_phase_spans(ax, extra_np[trial_idx], alpha=0.08)
    ax.set_ylabel("amplitude")
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(frameon=False, loc="upper right")


def panel_unperturbed(ax_trace, ax_pc):
    t = time_axis(inputs_np.shape[1])
    ax_trace.set_title(
        "B  Unperturbed trial performance", loc="left", fontweight="bold"
    )
    ax_trace.plot(t, targets_np[trial_idx, :, 0], color="0.1", lw=1.5, label="target")
    ax_trace.plot(
        t, controlled_np[trial_idx, :, 0], color="#1F77B4", lw=1.2, label="model"
    )
    add_phase_spans(ax_trace, extra_np[trial_idx], alpha=0.08)
    ax_trace.set_ylabel("output")
    ax_trace.set_xlabel(time_xlabel())
    ax_trace.legend(frameon=False, loc="upper right")

    cue1 = int(extra_np[trial_idx, 10]) if extra_np.shape[1] >= 12 else -1
    cue2 = int(extra_np[trial_idx, 11]) if extra_np.shape[1] >= 12 else -1
    plot_phase_colored_trajectory(ax_pc, latents_pc[trial_idx], extra_np[trial_idx])
    ax_pc.scatter(
        latents_pc[trial_idx, 0, 0],
        latents_pc[trial_idx, 0, 1],
        color="0.2",
        s=12,
        label="start",
        zorder=3,
    )
    ax_pc.scatter(
        latents_pc[trial_idx, -1, 0],
        latents_pc[trial_idx, -1, 1],
        color="0.2",
        marker="x",
        s=22,
        label="end",
        zorder=3,
    )
    for label, _, _, color in trial_phase_windows(extra_np[trial_idx]):
        ax_pc.plot([], [], color=color, lw=1.8, label=label)
    ax_pc.set_title(f"latent trajectory | cue {cue1}-{cue2}")
    ax_pc.set_xlabel("PC1")
    ax_pc.set_ylabel("PC2")
    ax_pc.legend(frameon=False, loc="best")


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
        lw=1.5,
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
        lw=1.3,
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
            t, output_delta[k, ic_trial_idx], color="#2CA02C", lw=0.6, alpha=alpha
        )
    ax_growth.plot(
        t,
        latent_delta[:, ic_trial_idx].mean(axis=0),
        color="#9467BD",
        lw=1.8,
        label="mean latent",
    )
    ax_growth.plot(
        t,
        output_delta[:, ic_trial_idx].mean(axis=0),
        color="#2CA02C",
        lw=1.5,
        label="mean output",
    )
    ax_growth.set_yscale("log")
    ax_growth.set_xlabel(time_xlabel())
    ax_growth.set_ylabel("norm")
    ax_growth.set_title("perturbation growth")
    ax_growth.legend(frameon=False, loc="best")


def panel_raster(ax_spikes, ax_rates, ax_pred_rates):
    ax_spikes.set_title("E  Simulated spiking data", loc="left", fontweight="bold")
    if lfads_data:
        spikes_for_panel = lfads_data["spikes"]
        true_rates_for_panel = lfads_data["true_rates"]
        pred_rates = lfads_data["rates"]
    else:
        spikes_for_panel = sim_spikes
        true_rates_for_panel = sim_rates
        pred_rates = None

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
    add_phase_spans(ax_spikes, extra_np[trial], alpha=0.08)
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
    add_phase_spans(ax_rates, extra_np[trial], alpha=0.08)
    ax_rates.set_title("simulated rates")
    if lfads_data:
        print("Panel E sim rates source: lfads_data['true_rates']")
    else:
        print("Panel E sim rates source: sim_rates")
    ax_rates.set_xlabel(time_xlabel())
    ax_rates.set_ylabel("neuron")

    if pred_rates_selected is None:
        ax_pred_rates.text(
            0.5,
            0.5,
            lfads_message or "LFADS rates unavailable",
            ha="center",
            va="center",
        )
        ax_pred_rates.set_axis_off()
        return
    describe_heatmap_panel(
        "LFADS predicted rates",
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
    add_phase_spans(ax_pred_rates, extra_np[pred_trial], alpha=0.08)
    ax_pred_rates.set_title("LFADS predicted rates")
    print(
        "Rate heatmap neuron indices shared by simulated/predicted:", neurons.tolist()
    )
    print(
        f"Shared rate heatmap color scale: vmin={rate_vmin:.4g}, vmax={rate_vmax:.4g}"
    )
    ax_pred_rates.set_xlabel(time_xlabel())
    ax_pred_rates.set_ylabel("neuron")


def panel_fit_metrics(ax):
    ax.set_title("C  Model fit metrics", loc="left", fontweight="bold")
    if not fit_metrics:
        ax.text(
            0.5,
            0.5,
            lfads_message or "LFADS metrics unavailable",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_axis_off()
        return
    labels = list(fit_metrics.keys())
    values = np.array([fit_metrics[label] for label in labels], dtype=float)
    x = np.arange(len(labels))
    colors = ["#D62728", "#4C78A8"]
    bars = ax.bar(x, values, color=colors[: len(labels)], width=0.62)
    ax.axhline(0, color="0.25", lw=0.7)
    ax.set_ylim(0, max(1.0, np.nanmax(values) * 1.12))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R2")
    for bar, value in zip(bars, values):
        label = "nan" if np.isnan(value) else f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.02 if np.isnan(value) else value + 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def panel_lfads(ax_rates, ax_pc):
    ax_rates.set_title("F  LFADS data-trained fit", loc="left", fontweight="bold")
    if not lfads_data:
        ax_rates.text(
            0.5,
            0.5,
            lfads_message or "LFADS outputs unavailable",
            ha="center",
            va="center",
            wrap=True,
        )
        ax_rates.set_axis_off()
        ax_pc.set_axis_off()
        return

    trial = lfads_data["trial_idx"]
    true_rates = lfads_data["true_rates"]
    pred_rates = lfads_data["rates"]
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
            label="LFADS" if j == 0 else None,
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
        "Panel E example neuron rate R2:",
        {int(n): float(r) for n, r in zip(neurons, r2_values)},
    )
    ax_rates.set_xlabel(time_xlabel())
    ax_rates.set_ylabel("rate + offset")
    ax_rates.legend(frameon=False, loc="upper right")

    pc = lfads_data["latents_pc"]
    tt_pc = lfads_data["tt_aligned_pc"]
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
        label="LFADS",
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
    ax_pc.set_title(f"LFADS latent trajectory | {lfads_data['phase']} trial {trial}")
    ax_pc.legend(frameon=False, loc="best")
    ax_pc.set_xlabel("PC1")
    ax_pc.set_ylabel("PC2")
    ax_pc.set_zlabel("PC3")


# %%
# --------------------------- Compose figure ---------------------------
fig = plt.figure(figsize=(14.5, 12.0), constrained_layout=False)
outer = GridSpec(
    5,
    4,
    figure=fig,
    height_ratios=[0.58, 0.82, 0.92, 1.25, 1.2],
    hspace=0.62,
    wspace=0.45,
)

ax_task = fig.add_subplot(outer[0, :2])
ax_io = fig.add_subplot(outer[1, :2], sharex=ax_task)
ax_perf = fig.add_subplot(outer[2, :2], sharex=ax_task)
ax_perf_pc = fig.add_subplot(outer[1:3, 2:])
ax_metrics = fig.add_subplot(outer[0, 2:])
panel_task_schematic(ax_task)
panel_io(ax_io)
panel_unperturbed(ax_perf, ax_perf_pc)
panel_fit_metrics(ax_metrics)

ax_ic2 = fig.add_subplot(outer[3, 0])
ax_ic3 = fig.add_subplot(outer[3, 1], projection="3d")
ax_icg = fig.add_subplot(outer[3, 2:])
panel_ic_perturbations(ax_ic2, ax_ic3, ax_icg)

bottom = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[4, :], wspace=0.45)
ax_raster = fig.add_subplot(bottom[0, 0])
ax_true_rates = fig.add_subplot(bottom[0, 1])
ax_pred_heatmap = fig.add_subplot(bottom[0, 2])
ax_lfads_rates = fig.add_subplot(bottom[0, 3])
ax_lfads_pc = fig.add_subplot(bottom[0, 4], projection="3d")
panel_raster(ax_raster, ax_true_rates, ax_pred_heatmap)
panel_lfads(ax_lfads_rates, ax_lfads_pc)

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

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)
