"""Generate manuscript Figure 5 reconstruction/simplicity panels for 3BFF NODE sweep.

This is a clean, scriptable replacement for the exploratory Fig5Metrics notebooks.
Panels A and D are intentionally left as schematic placeholders.
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

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison
from ctd.data_modeling.extensions.evaluation import bits_per_spike

try:
    import dotenv
except ImportError:
    dotenv = None


COLORS = {
    2: "#2ca25f",
    8: "#2b6cb0",
    64: "#e68613",
    "truth": "#111111",
    "pred_8": "#9ecae1",
    "pred_64": "#f6c176",
}
LATENT_CMAP = "turbo"
LATENT_COLOR_DIMS = np.array([2, 3, 5, 8, 16, 32, 64])
LATENT_PASTEL_MIX = 0.0
LATENT_CMAP_TRIM = (
    0.10,
    0.95,
)  # matches examples/pick_latent_colormap.py turbo override

# Manuscript-wide palette: {D: "#hex"} produced by
# examples/pick_latent_colormap.py. Resolved at import time; if absent we fall
# back to sampling LATENT_CMAP directly so the script still runs unchanged.
LATENT_PALETTE_PATH = REPO_ROOT / "latent_palette.json"
_LATENT_PALETTE_CACHE: dict[int, tuple[float, float, float, float]] | None = None


def _load_latent_palette() -> dict[int, tuple[float, float, float, float]] | None:
    """Load {D: rgba} from LATENT_PALETTE_PATH; cache the result.

    Returns None if the file is missing or unreadable, in which case callers
    fall back to sampling LATENT_CMAP. Keys are coerced to int (JSON stores
    them as strings).
    """
    global _LATENT_PALETTE_CACHE
    if _LATENT_PALETTE_CACHE is not None:
        return _LATENT_PALETTE_CACHE
    if not LATENT_PALETTE_PATH.exists():
        return None
    try:
        raw = json.loads(LATENT_PALETTE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: could not read {LATENT_PALETTE_PATH}: {exc}; "
            "falling back to {LATENT_CMAP} sampling."
        )
        return None
    _LATENT_PALETTE_CACHE = {int(k): to_rgba(v) for k, v in raw.items()}
    return _LATENT_PALETTE_CACHE


def latent_dim_color(latent_size):
    """Return the manuscript color for ``latent_size``.

    Prefers an exact lookup in the JSON palette. For latent sizes not in the
    JSON, log2-interpolates between the two nearest palette entries so the
    color stays consistent with the saved mapping. Falls back to direct
    LATENT_CMAP sampling if no palette file is present.
    """
    palette = _load_latent_palette()
    if palette is not None:
        if int(latent_size) in palette:
            rgba = np.array(palette[int(latent_size)], dtype=float)
        else:
            keys = np.array(sorted(palette.keys()), dtype=float)
            log_target = np.log2(float(latent_size))
            log_keys = np.log2(keys)
            if log_target <= log_keys[0]:
                rgba = np.array(palette[int(keys[0])], dtype=float)
            elif log_target >= log_keys[-1]:
                rgba = np.array(palette[int(keys[-1])], dtype=float)
            else:
                hi = int(np.searchsorted(log_keys, log_target))
                lo = hi - 1
                t = (log_target - log_keys[lo]) / (log_keys[hi] - log_keys[lo])
                c_lo = np.array(palette[int(keys[lo])], dtype=float)
                c_hi = np.array(palette[int(keys[hi])], dtype=float)
                rgba = (1 - t) * c_lo + t * c_hi
    else:
        cmap = plt.get_cmap(LATENT_CMAP)
        log_dims = np.log2(LATENT_COLOR_DIMS)
        pos = (np.log2(latent_size) - log_dims.min()) / (
            log_dims.max() - log_dims.min()
        )
        pos = LATENT_CMAP_TRIM[0] + pos * (LATENT_CMAP_TRIM[1] - LATENT_CMAP_TRIM[0])
        rgba = np.array(cmap(pos))
    rgba[:3] = rgba[:3] + (1.0 - rgba[:3]) * LATENT_PASTEL_MIX
    return tuple(rgba)


def light_latent_dim_color(latent_size, amount=0.55):
    rgba = np.array(latent_dim_color(latent_size))
    rgba[:3] = rgba[:3] + (1.0 - rgba[:3]) * amount
    return tuple(rgba)


# Which NODE seed (0-indexed within the sorted sweep) to use for the
# representative models. Set to the index of the model you want to feature.
RECONSTRUCTION_LATENTS = (2, 8)
SIMPLICITY_LATENTS = (8, 32)
CALLOUT_LATENTS = tuple(sorted(set(RECONSTRUCTION_LATENTS + SIMPLICITY_LATENTS)))
LATENT_INDICES = {
    2: 0,
    8: 1,
    32: 0,
}

# If True, the figure is regenerated with noise-free external inputs (the TT
# model and each DD model are re-evaluated using the task's clean inputs in
# place of the noisy training inputs). Equivalent to passing ``--noiseless``
# on the command line; the CLI flag overrides this when set.
NOISELESS_FLAG = False

# If True, ignore the cached payload and recompute everything (metric table,
# reconstruction examples, simplicity examples). Equivalent to passing
# ``--force``; the CLI flag overrides this when set. Use this when you've
# changed LATENT_INDICES, model paths, or analysis logic and want a fresh run.
FORCE_RECOMPUTE = False

# If True, subtract each trial's own mean (per PC, along time) from the actual,
# state-R², and cycle-Con traces shown in panel E (and in the panel-E candidate
# browser). R² annotations are recomputed on the per-trial-demeaned series, so
# they reflect how well the *shape* of each PC is predicted, ignoring per-trial
# DC offsets. Equivalent to passing ``--demean``; the CLI flag overrides this.
DEMEAN_FLAG = False


# %%
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in {
            "ZMQInteractiveShell",
            "TerminalInteractiveShell",
        }
    except ImportError:
        return False


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "outputs"
    )
    parser.add_argument(
        "--cache", type=Path, default=Path(__file__).with_suffix(".cache.pkl")
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=FORCE_RECOMPUTE,
        help=(
            "Recompute metrics and examples instead of loading the cache. "
            "Defaults to the module-level FORCE_RECOMPUTE constant; pass "
            "--no-force to override when the constant is True."
        ),
    )
    parser.add_argument("--trial", type=int, default=7)
    parser.add_argument("--neuron", type=int, default=0)
    parser.add_argument(
        "--timepoints",
        type=int,
        default=100,
        help="Default number of timepoints shown in panels B and E.",
    )
    parser.add_argument(
        "--panel-b-timepoints",
        type=int,
        default=None,
        help="Override timepoints shown in panel B.",
    )
    parser.add_argument(
        "--panel-e-timepoints",
        type=int,
        default=None,
        help="Override timepoints shown in panel E.",
    )
    parser.add_argument(
        "--pc-index", type=int, default=7, help="Zero-indexed PC to show in panel E."
    )
    parser.add_argument(
        "--noiseless",
        action=argparse.BooleanOptionalAction,
        default=NOISELESS_FLAG,
        help=(
            "Re-run the TT and DD models with the task's noise-free external "
            "inputs instead of the noisy training inputs. Saves to a separate "
            "'_noiseless' file and annotates the figure. Defaults to the "
            "module-level NOISELESS_FLAG constant; pass --no-noiseless to "
            "force-disable when the constant is True."
        ),
    )
    parser.add_argument(
        "--demean",
        action=argparse.BooleanOptionalAction,
        default=DEMEAN_FLAG,
        help=(
            "Subtract each trial's own mean (per PC) from the panel-E actual / "
            "state / cycle traces before plotting, and recompute the panel-E "
            "R² annotations on the per-trial-demeaned series. Highlights how "
            "well the shape of each PC is predicted, ignoring DC offsets."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    if argv is None and _in_notebook():
        argv = []
    return parser.parse_args(argv)


def _demean_per_trial(arr: np.ndarray) -> np.ndarray:
    """Subtract the per-trial time-mean of an (n_trials, n_time, ...) array."""
    return arr - arr.mean(axis=1, keepdims=True)


def as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_home_dir() -> Path:
    if dotenv is not None:
        dotenv.load_dotenv(dotenv.find_dotenv())
    home = os.environ.get("HOME_DIR")
    if home is None:
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip().startswith("HOME_DIR"):
                    home = line.split("=", 1)[1].strip()
                    break
    if home is None:
        home = str(Path(__file__).resolve().parents[3]) + os.sep
    resolved = Path(home).expanduser().resolve()
    os.environ["HOME_DIR"] = str(resolved) + os.sep
    return resolved


def load_comparison() -> Comparison:
    home_dir = get_home_dir()
    path_tt = home_dir / "content/trained_models/task-trained/tt_3bff"
    # path_node_sweep = path_tt / "20250207_3BFF_NODE_sweep"
    path_node_sweep = path_tt / "20260520_NBFF_SAE_NODE_DimSweep"
    if not path_node_sweep.exists():
        raise FileNotFoundError(
            f"Could not find NODE sweep at {path_node_sweep}. "
            "Set HOME_DIR to the directory containing content/trained_models."
        )

    comparison = Comparison(comparison_tag="Figure5_3BFF_NODE_ReconSimplicity")
    comparison.load_analysis(
        Analysis_TT(run_name="TT", filepath=str(path_tt) + os.sep),
        reference_analysis=True,
        group="TT",
    )

    subfolders = sorted([p for p in path_node_sweep.iterdir() if p.is_dir()])
    for subfolder in subfolders:
        latent_size = subfolder.name.split("latent_size=")[1].split("_")[0]
        analysis = Analysis_DD.create(
            run_name=f"NODE{latent_size}",
            filepath=str(subfolder) + os.sep,
            model_type="SAE",
        )
        comparison.load_analysis(analysis, group=f"NODE{latent_size}")
    comparison.regroup()
    # Force deterministic forward passes. Analysis_DD_SAE.get_model_outputs
    # calls self.model(...) and Analysis_TT.get_model_outputs calls
    # self.wrapper(...), neither of which wraps the forward in
    # .eval()/no_grad. Models left in train mode produce slightly different
    # outputs on every call (dropout, sample-based ICs). The main panel E
    # and the candidates path invoke get_latents / get_rates in different
    # orders; that non-determinism is enough to flip signs or reorder
    # low-variance PCs and to perturb the TT latents used by state_pred —
    # exactly the "doesn't resemble" mismatch the debug figure shows.
    # eval() on both .model and .wrapper makes both code paths see the same
    # model output.
    for analysis in comparison.analyses:
        for attr in ("model", "wrapper"):
            obj = getattr(analysis, attr, None)
            if obj is None:
                continue
            if hasattr(obj, "eval"):
                obj.eval()
            # task_wrapper.forward injects torch.randn_like(input) * dynamic_noise
            # on every timestep. That's gaussian noise drawn fresh each call, not
            # dropout, so .eval() does NOT suppress it — every get_latents
            # produces different TT latents. Zero it out for analysis so the
            # main panel E and the candidates path see identical TT outputs.
            if hasattr(obj, "dynamic_noise"):
                try:
                    obj.dynamic_noise = 0
                except (AttributeError, TypeError):
                    pass
            # RNN/GRU/etc. TT models inject Gaussian noise in two more places:
            #   - init_hidden: torch.randn_like(init_h) * latent_ic_var
            #   - forward:     torch.randn_like(drive)  * noise_level (per step)
            # Both are drawn fresh on every call regardless of .eval(). NODE
            # is already deterministic, but any RNN-family TT model needs these
            # zeroed for analysis to be reproducible. See
            # ctd/task_modeling/model/rnn.py for the exact spots.
            inner = getattr(obj, "model", None)
            if inner is not None:
                for noise_attr in ("latent_ic_var", "noise_level"):
                    if hasattr(inner, noise_attr):
                        try:
                            setattr(inner, noise_attr, 0)
                        except (AttributeError, TypeError):
                            pass
    return comparison


def representative_indices(comparison: Comparison) -> dict[int, int]:
    matches: dict[int, list[int]] = {latent: [] for latent in CALLOUT_LATENTS}
    for i, analysis in enumerate(comparison.analyses):
        if not analysis.run_name.startswith("NODE"):
            continue
        try:
            latent = int(analysis.run_name.replace("NODE", ""))
        except ValueError:
            continue
        if latent in matches:
            matches[latent].append(i)
    reps: dict[int, int] = {}
    for latent, idx_list in matches.items():
        if not idx_list:
            continue
        pick = LATENT_INDICES.get(latent, 0)
        if not (0 <= pick < len(idx_list)):
            raise RuntimeError(
                f"LATENT_INDICES[{latent}]={pick} out of range; "
                f"{len(idx_list)} NODE{latent} runs available."
            )
        reps[latent] = idx_list[pick]
    missing = set(CALLOUT_LATENTS) - set(reps)
    if missing:
        raise RuntimeError(f"Missing representative NODE runs: {sorted(missing)}")
    return reps


def compute_metric_table(comparison: Comparison) -> dict[str, np.ndarray]:
    metric_dict = {
        "state_r2": {},
        "rate_r2": {},
        "cycle_con": {"variance_threshold": 0.01},
        "co-bps": {},
    }
    metrics = comparison.compute_metrics(metric_dict_list=metric_dict)
    latent_sizes = np.array(
        [int(name.replace("NODE", "")) for name in metrics["run_name"]]
    )
    metrics["latent_size"] = latent_sizes
    for key in ("state_r2", "rate_r2", "cycle_con", "co-bps"):
        metrics[key] = np.asarray(metrics[key], dtype=float)
    return metrics


def find_metric_row(
    metrics: dict[str, np.ndarray], comparison: Comparison, analysis_idx: int
) -> int:
    node_seen = -1
    for i, analysis in enumerate(comparison.analyses):
        if i == comparison.ref_ind:
            continue
        node_seen += 1
        if i == analysis_idx:
            return node_seen
    raise ValueError(f"Could not map analysis index {analysis_idx} to metrics row.")


def neuron_metrics(rates, true_rates, spikes, neuron: int) -> tuple[float, float]:
    pred = rates[..., neuron].reshape(-1)
    true = true_rates[..., neuron].reshape(-1)
    spike = spikes[..., neuron : neuron + 1]
    bps = bits_per_spike(rates[..., neuron : neuron + 1], spike)
    return float(bps), float(r2_score(true, pred))


def compute_reconstruction_examples(
    comparison, reps, metrics, trial, neuron, timepoints
):
    target_rep = reps[RECONSTRUCTION_LATENTS[0]]
    true_rates = as_numpy(comparison.analyses[target_rep].get_true_rates(phase="val"))
    spikes = as_numpy(comparison.analyses[target_rep].get_spiking(phase="val"))
    t = np.arange(min(timepoints, true_rates.shape[1]))
    trial = min(trial, true_rates.shape[0] - 1)
    neuron = min(neuron, true_rates.shape[2] - 1)

    examples = {
        "trial": trial,
        "neuron": neuron,
        "time": t,
        "true": true_rates[trial, t, neuron],
        "spikes": spikes[trial, t, neuron],
    }
    for latent in RECONSTRUCTION_LATENTS:
        analysis = comparison.analyses[reps[latent]]
        rates = as_numpy(analysis.get_rates(phase="val"))
        bps, r2 = neuron_metrics(rates, true_rates, spikes, neuron)
        examples[latent] = {
            "pred": rates[trial, t, neuron],
            "co_bps": bps,
            "rate_r2": r2,
        }
    return examples


def reconstruct_latents(
    readout: LinearRegression, log_rates_pca: np.ndarray, variance_threshold=0.01
) -> np.ndarray:
    w_hat = readout.coef_
    b_hat = readout.intercept_
    centered = log_rates_pca - b_hat
    u, sigma, vt = np.linalg.svd(w_hat, full_matrices=False)
    normalized = (sigma**2) / np.sum(sigma**2)
    n_components = np.searchsorted(np.cumsum(normalized), 1 - variance_threshold) + 1
    n_components = min(n_components, len(sigma))
    w_pinv = (
        vt[:n_components, :].T
        @ np.diag(1 / sigma[:n_components])
        @ u[:, :n_components].T
    )
    return centered @ w_pinv.T


def compute_simplicity_examples(
    comparison, reps, trial, timepoints, pc_index, demean=False
):
    true_train = as_numpy(
        comparison.analyses[comparison.ref_ind].get_latents(phase="train")
    )
    true_val = as_numpy(
        comparison.analyses[comparison.ref_ind].get_latents(phase="val")
    )
    trial = min(trial, true_val.shape[0] - 1)
    out = {"trial": trial, "pc_index": pc_index, "time": None, "demean": bool(demean)}

    for latent in SIMPLICITY_LATENTS:
        analysis = comparison.analyses[reps[latent]]
        lats_train = as_numpy(analysis.get_latents(phase="train"))
        lats_val = as_numpy(analysis.get_latents(phase="val"))
        rates_train = np.log(
            np.clip(as_numpy(analysis.get_rates(phase="train")), 1e-9, None)
        )
        rates_val = np.log(
            np.clip(as_numpy(analysis.get_rates(phase="val")), 1e-9, None)
        )

        pca_lats = PCA()
        lats_train_flat = pca_lats.fit_transform(
            lats_train.reshape(-1, lats_train.shape[-1])
        )
        lats_val_flat = pca_lats.transform(lats_val.reshape(-1, lats_val.shape[-1]))

        pca_rates = PCA()
        rates_train_flat = pca_rates.fit_transform(
            rates_train.reshape(-1, rates_train.shape[-1])
        )
        rates_val_flat = pca_rates.transform(rates_val.reshape(-1, rates_val.shape[-1]))

        lm_state = LinearRegression().fit(
            true_train.reshape(-1, true_train.shape[-1]), lats_train_flat
        )
        pred_state_flat = lm_state.predict(true_val.reshape(-1, true_val.shape[-1]))

        lm_readout = LinearRegression().fit(lats_train_flat, rates_train_flat)
        pred_log_rates_flat = lm_readout.predict(lats_val_flat)
        pred_cycle_flat = reconstruct_latents(lm_readout, pred_log_rates_flat)

        n_trials, n_time = lats_val.shape[:2]
        dim = min(pc_index, lats_val_flat.shape[1] - 1)
        t = np.arange(min(timepoints, n_time))
        if out["time"] is None:
            out["time"] = t

        actual_full = lats_val_flat.reshape(n_trials, n_time, -1)
        state_full = pred_state_flat.reshape(n_trials, n_time, -1)
        cycle_full = pred_cycle_flat.reshape(n_trials, n_time, -1)
        if demean:
            actual_full = _demean_per_trial(actual_full)
            state_full = _demean_per_trial(state_full)
            cycle_full = _demean_per_trial(cycle_full)

        actual = actual_full[trial, t, dim]
        pred_state = state_full[trial, t, dim]
        pred_cycle = cycle_full[trial, t, dim]

        actual_flat = actual_full[:, :, dim].reshape(-1)
        state_flat = state_full[:, :, dim].reshape(-1)
        cycle_flat = cycle_full[:, :, dim].reshape(-1)

        out[latent] = {
            "actual": actual,
            "state_pred": pred_state,
            "cycle_pred": pred_cycle,
            "state_r2": float(r2_score(actual_flat, state_flat)),
            "cycle_r2": float(r2_score(actual_flat, cycle_flat)),
        }
    return out


def _patch_analysis_clean(
    analysis,
    lat_train,
    lat_val,
    rates_train=None,
    rates_val=None,
    true_rates_train=None,
    true_rates_val=None,
    spikes_train=None,
    spikes_val=None,
):
    """Override an analysis's getters to return pre-computed noise-free outputs.

    Returns an ``undo()`` callable that restores the original methods. The
    Comparison's metric routines call ``get_latents``/``get_rates``/
    ``get_model_outputs``/``get_true_rates``/``get_spiking`` on each analysis —
    replacing them here is enough to make the whole metric table reflect the
    noise-free model evaluations plus the simulator-regenerated targets.
    """
    original = {}

    def _patch(name, fn):
        original[name] = getattr(analysis, name)
        setattr(analysis, name, fn)

    def get_latents(phase="all"):
        if phase == "train":
            return lat_train
        if phase == "val":
            return lat_val
        return torch.cat([lat_train, lat_val], dim=0)

    _patch("get_latents", get_latents)

    if rates_train is not None:

        def get_rates(phase="all"):
            if phase == "train":
                return rates_train
            if phase == "val":
                return rates_val
            return torch.cat([rates_train, rates_val], dim=0)

        def get_model_outputs(phase="all"):
            if phase == "train":
                return rates_train, lat_train
            if phase == "val":
                return rates_val, lat_val
            return (
                torch.cat([rates_train, rates_val], dim=0),
                torch.cat([lat_train, lat_val], dim=0),
            )

        _patch("get_rates", get_rates)
        _patch("get_model_outputs", get_model_outputs)

    if true_rates_train is not None:

        def get_true_rates(phase="all"):
            if phase == "train":
                return true_rates_train
            if phase == "val":
                return true_rates_val
            return torch.cat([true_rates_train, true_rates_val], dim=0)

        _patch("get_true_rates", get_true_rates)

    if spikes_train is not None:

        def get_spiking(phase="all"):
            if phase == "train":
                return spikes_train
            if phase == "val":
                return spikes_val
            return torch.cat([spikes_train, spikes_val], dim=0)

        _patch("get_spiking", get_spiking)

    def undo():
        for name, fn in original.items():
            setattr(analysis, name, fn)

    return undo


def simulate_clean_targets(tt_analysis):
    """Run the TT simulator once with noise-free task inputs.

    Returns a dict with ``rates_train/val`` (the simulator's clean firing
    rates, i.e. ``activity``) and ``spikes_train/val`` (a Poisson sample from
    those clean rates) split at the original TT train/val boundary. Returns
    ``None`` if no simulator is loaded or the datamodule lacks noise-free
    inputs.
    """
    simulator = getattr(tt_analysis, "simulator", None)
    datamodule = getattr(tt_analysis, "datamodule", None)
    if simulator is None or datamodule is None:
        return None
    if not (hasattr(datamodule, "train_ds") and hasattr(datamodule, "valid_ds")):
        return None
    if len(datamodule.train_ds.tensors) < 8 or len(datamodule.valid_ds.tensors) < 8:
        # No clean inputs (tensors[7]) available in this datamodule.
        return None

    def _clean_ds(ds):
        t = list(ds.tensors)
        t[1] = t[7]
        return TensorDataset(*t)

    orig_train = datamodule.train_ds
    orig_valid = datamodule.valid_ds
    datamodule.train_ds = _clean_ds(orig_train)
    datamodule.valid_ds = _clean_ds(orig_valid)
    try:
        with torch.no_grad():
            sim = simulator.generate_simulated_data(
                tt_analysis.wrapper, datamodule, seed=0
            )
    finally:
        datamodule.train_ds = orig_train
        datamodule.valid_ds = orig_valid

    activity = torch.as_tensor(np.asarray(sim["activity"]), dtype=torch.float32)
    spikes = torch.as_tensor(np.asarray(sim["data"]), dtype=torch.float32)
    n_train = len(orig_train)
    return {
        "rates_train": activity[:n_train],
        "rates_val": activity[n_train:],
        "spikes_train": spikes[:n_train],
        "spikes_val": spikes[n_train:],
    }


def apply_noiseless_overrides(comparison: Comparison):
    """Patch every analysis in ``comparison`` to use noise-free external inputs.

    Steps:

    1. The reference TT analysis's ``get_latents`` is replaced with the
       noiseless version, so any consumer that asks for "true latents" gets
       the trajectory the model produces under clean inputs.
    2. The TT simulator is run once with the datamodule's noise-free inputs
       (``tensors[7]``) swapped into ``tensors[1]``. The resulting firing
       rates and Poisson spike samples are used as the noise-free targets for
       every DD analysis (``get_true_rates`` / ``get_spiking``), so Rate R²,
       co-BPS, and Cycle-Con are all evaluated against the clean simulator
       output rather than the original noisy-input simulation.
    3. Each DD model is re-run with its own spiking encoder data but with the
       clean inputs substituted on the recurrent path; the resulting
       ``rates``/``latents`` replace ``get_rates``/``get_latents``/
       ``get_model_outputs``.

    Returns a list of ``undo()`` callables.
    """
    tt = comparison.analyses[comparison.ref_ind]
    clean_inputs = {
        phase: tt.get_true_inputs(phase=phase).detach() for phase in ("train", "val")
    }
    tt_clean_lat = {
        phase: tt.get_latents_noiseless(phase=phase).detach()
        for phase in ("train", "val")
    }
    undos = [_patch_analysis_clean(tt, tt_clean_lat["train"], tt_clean_lat["val"])]

    clean_targets = simulate_clean_targets(tt)
    if clean_targets is None:
        print(
            "WARNING: could not simulate clean targets (no simulator.pkl, or no "
            "noise-free inputs in the TT datamodule). Rate R² / co-BPS will "
            "compare clean DD predictions against the original noisy targets."
        )
    else:
        print(
            f"  simulated clean targets: rates={tuple(clean_targets['rates_train'].shape)} "
            f"+ {tuple(clean_targets['rates_val'].shape)}"
        )

    for i, analysis in enumerate(comparison.analyses):
        if i == comparison.ref_ind:
            continue
        if not hasattr(analysis, "model") or not hasattr(analysis, "get_model_inputs"):
            continue
        device = getattr(analysis.model, "device", torch.device("cpu"))
        rates_per_phase = {}
        latents_per_phase = {}
        for phase in ("train", "val"):
            spikes, _ = analysis.get_model_inputs(phase=phase)
            spikes = spikes.to(device)
            ci = clean_inputs[phase].to(device)
            with torch.no_grad():
                log_rates, latents = analysis.model(spikes, ci)
            rates_per_phase[phase] = torch.exp(log_rates).detach()
            latents_per_phase[phase] = latents.detach()

        true_rates_kwargs = {}
        if clean_targets is not None:
            existing_true = analysis.get_true_rates(phase="val")
            if tuple(existing_true.shape) == tuple(
                clean_targets["rates_val"].shape
            ) and tuple(analysis.get_spiking(phase="val").shape) == tuple(
                clean_targets["spikes_val"].shape
            ):
                true_rates_kwargs = {
                    "true_rates_train": clean_targets["rates_train"].to(
                        rates_per_phase["train"].device
                    ),
                    "true_rates_val": clean_targets["rates_val"].to(
                        rates_per_phase["val"].device
                    ),
                    "spikes_train": clean_targets["spikes_train"].to(
                        rates_per_phase["train"].device
                    ),
                    "spikes_val": clean_targets["spikes_val"].to(
                        rates_per_phase["val"].device
                    ),
                }
            else:
                print(
                    f"  WARNING: clean-target shape doesn't match "
                    f"{analysis.run_name} datamodule shape; leaving "
                    "true_rates/spikes for this analysis unchanged."
                )

        undos.append(
            _patch_analysis_clean(
                analysis,
                latents_per_phase["train"],
                latents_per_phase["val"],
                rates_per_phase["train"],
                rates_per_phase["val"],
                **true_rates_kwargs,
            )
        )
    return undos


def build_payload(
    cache: Path,
    force: bool,
    trial: int,
    neuron: int,
    panel_b_timepoints: int,
    panel_e_timepoints: int,
    pc_index: int,
    noiseless: bool = False,
    demean: bool = False,
):
    config = {
        "trial": trial,
        "neuron": neuron,
        "panel_b_timepoints": panel_b_timepoints,
        "panel_e_timepoints": panel_e_timepoints,
        "pc_index": pc_index,
        "latent_indices": dict(LATENT_INDICES),
        "reconstruction_latents": tuple(RECONSTRUCTION_LATENTS),
        "simplicity_latents": tuple(SIMPLICITY_LATENTS),
        "noiseless": bool(noiseless),
        "demean": bool(demean),
    }
    if cache.exists() and not force:
        with cache.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("config") == config:
            return cached

    torch.manual_seed(42)
    np.random.seed(42)
    comparison = load_comparison()
    reps = representative_indices(comparison)
    undos: list = []
    if noiseless:
        undos = apply_noiseless_overrides(comparison)
    try:
        metrics = compute_metric_table(comparison)
        metric_rows = {
            latent: find_metric_row(metrics, comparison, idx)
            for latent, idx in reps.items()
        }
        payload = {
            "config": config,
            "metrics": metrics,
            "metric_rows": metric_rows,
            "reconstruction": compute_reconstruction_examples(
                comparison, reps, metrics, trial, neuron, panel_b_timepoints
            ),
            "simplicity": compute_simplicity_examples(
                comparison, reps, trial, panel_e_timepoints, pc_index, demean=demean
            ),
            "noiseless": bool(noiseless),
            "demean": bool(demean),
        }
    finally:
        for undo in undos:
            undo()
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(payload, f)
    return payload


def set_panel_label(ax, label):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def clean_axis(ax, keep_bottom=False):
    ax.spines[["top", "right", "left"]].set_visible(False)
    if not keep_bottom:
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
    ax.set_yticks([])


def _limits_including_callouts(limits, values, pad_fraction=0.06):
    if limits is None:
        return None
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return limits
    low, high = float(limits[0]), float(limits[1])
    low = min(low, float(values.min()))
    high = max(high, float(values.max()))
    span = high - low
    pad = pad_fraction * (span if span > 0 else 1.0)
    return low - pad, high + pad


def scatter_metrics(
    ax,
    metrics,
    x_key,
    y_key,
    metric_rows,
    xlabel,
    ylabel,
    xlim=None,
    ylim=None,
    ticks_minmax=True,
    xticks=None,
    yticks=None,
    minor_xticks=None,
    minor_yticks=None,
    tick_decimals=None,
    note=None,
    include_callouts_in_limits=True,
):
    latent = metrics["latent_size"]
    sizes = 20 + 8 * np.log2(latent)
    callout_rows = set(metric_rows.values())
    callout_rows_list = [
        metric_rows[lat] for lat in CALLOUT_LATENTS if lat in metric_rows
    ]
    non_callout = np.array([i not in callout_rows for i in range(len(latent))])
    point_colors = [latent_dim_color(value) for value in latent[non_callout]]
    ax.scatter(
        metrics[x_key][non_callout],
        metrics[y_key][non_callout],
        s=sizes[non_callout],
        c=point_colors,
        alpha=0.7,
        linewidth=1.25,
        marker="x",
    )
    for lat in CALLOUT_LATENTS:
        row = metric_rows[lat]
        ax.scatter(
            metrics[x_key][row],
            metrics[y_key][row],
            s=70,
            c=[latent_dim_color(lat)],
            alpha=1.0,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if include_callouts_in_limits and callout_rows_list:
        xlim = _limits_including_callouts(xlim, metrics[x_key][callout_rows_list])
        ylim = _limits_including_callouts(ylim, metrics[y_key][callout_rows_list])
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_tick(t, tick_decimals) for t in xticks])
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([format_tick(t, tick_decimals) for t in yticks])
    if ticks_minmax and xticks is None and yticks is None:
        x_ticks = [*ax.get_xlim()]
        y_ticks = [*ax.get_ylim()]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([format_tick(t, tick_decimals) for t in x_ticks])
        ax.set_yticklabels([format_tick(t, tick_decimals) for t in y_ticks])
    if minor_xticks is not None:
        ax.set_xticks(minor_xticks, minor=True)
    if minor_yticks is not None:
        ax.set_yticks(minor_yticks, minor=True)
    ax.tick_params(axis="both", which="minor", length=3, width=0.6)
    if note is not None:
        ax.text(
            0.05, 0.95, note, transform=ax.transAxes, ha="left", va="top", fontsize=8
        )
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def add_latent_size_legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=latent_dim_color(latent),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.2,
            linestyle="none",
        )
        for latent in LATENT_COLOR_DIMS
    ]
    labels = [
        str(latent) if latent in CALLOUT_LATENTS else "" for latent in LATENT_COLOR_DIMS
    ]
    ax.legend(
        handles,
        labels,
        title="Latent size",
        loc="lower right",
        bbox_to_anchor=(1.05, 0.03),
        ncol=len(LATENT_COLOR_DIMS),
        frameon=False,
        fontsize=6.5,
        title_fontsize=7,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.45,
        borderpad=0.1,
    )


def format_tick(value, decimals=None):
    if decimals is not None:
        return f"{value:.{decimals}f}"
    return f"{value:.3f}" if abs(value) < 0.2 else f"{value:.2f}"


def draw_placeholder(ax, title, labels):
    ax.set_axis_off()
    ax.text(0.5, 0.64, title, ha="center", va="center", fontsize=10, fontweight="bold")
    y = 0.43
    for text in labels:
        ax.text(0.5, y, text, ha="center", va="center", fontsize=8)
        y -= 0.11


def draw_panel_b(fig, outer, data):
    gs = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer, height_ratios=[1, 1, 0.22], hspace=0.05
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax, latent in zip(axes[:2], RECONSTRUCTION_LATENTS):
        ax.plot(data["time"], data["true"], color=COLORS["truth"], lw=1.1)
        ax.plot(
            data["time"], data[latent]["pred"], color=latent_dim_color(latent), lw=1.1
        )
        clean_axis(ax)
        ax.text(
            0.98,
            0.88,
            f"co-BPS {data[latent]['co_bps']:.2f}\nRate R$^2$ {data[latent]['rate_r2']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=latent_dim_color(latent),
        )
    spike_t = data["time"][data["spikes"] > 0]
    axes[2].eventplot(
        spike_t, colors="black", lineoffsets=0.35, linelengths=0.55, linewidths=0.6
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Time")
    clean_axis(axes[2], keep_bottom=True)
    axes[2].spines["bottom"].set_visible(True)
    return axes[0]


def draw_panel_e(fig, outer, data):
    gs = GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer, height_ratios=[0.32, 1, 1], hspace=0.34, wspace=0.15
    )
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    title_suffix = " (per-trial demeaned)" if data.get("demean") else ""
    title_ax.text(
        0.5,
        0.72,
        f"Predicting Data-Driven PC {data['pc_index'] + 1}{title_suffix}",
        ha="center",
        va="center",
        fontsize=9,
    )
    first_ax = None
    for row, latent in enumerate(SIMPLICITY_LATENTS, start=1):
        for col, key in enumerate(("state_pred", "cycle_pred")):
            ax = fig.add_subplot(gs[row, col])
            if first_ax is None:
                first_ax = ax
            ax.plot(
                data["time"],
                data[latent]["actual"],
                color=latent_dim_color(latent),
                lw=1.0,
            )
            ax.plot(
                data["time"],
                data[latent][key],
                color=light_latent_dim_color(latent),
                lw=1.0,
            )
            clean_axis(ax)
            metric = data[latent]["state_r2" if key == "state_pred" else "cycle_r2"]
            ax.text(
                0.05, 0.87, f"R$^2$ {metric:.2f}", transform=ax.transAxes, fontsize=7
            )
            if row == 1:
                ax.set_title("State R$^2$" if col == 0 else "Cycle-Con", fontsize=8)
            if col == 0:
                ax.text(
                    -0.05,
                    0.5,
                    f"{latent}D",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=latent_dim_color(latent),
                )
    return first_ax


def draw_panel_i(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Simplicity")
    ax.set_ylabel("Reconstruction")
    points = {2: (0.78, 0.28), 8: (0.78, 0.82), 64: (0.25, 0.82)}
    for lat, (x, y) in points.items():
        ax.scatter(
            x,
            y,
            s=110,
            c=[latent_dim_color(lat)],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    ax.text(
        points[64][0],
        points[64][1] + 0.08,
        "Invented\nfeatures",
        ha="center",
        fontsize=8,
        color=latent_dim_color(64),
    )
    ax.text(
        points[2][0],
        points[2][1] - 0.13,
        "Underfitting",
        ha="center",
        fontsize=8,
        color=latent_dim_color(2),
    )
    opt = points[8]
    for lat in (2, 64):
        ax.annotate(
            "",
            xy=points[lat],
            xytext=opt,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9, linestyle="--"),
        )
    ax.annotate(
        "",
        xy=(0.98, -0.02),
        xytext=(0.82, -0.02),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.annotate(
        "",
        xy=(-0.02, 0.98),
        xytext=(-0.02, 0.82),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xticks([])
    ax.set_yticks([])


# %%
def make_figure(payload, output_dir: Path, dpi: int):
    metrics = payload["metrics"]
    metric_rows = payload["metric_rows"]
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(9.2, 8.8), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig, wspace=0.42, hspace=0.48)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_placeholder(ax_a, "Reconstruction Metrics", ["Rate R$^2$", "co-BPS"])
    set_panel_label(ax_a, "A")

    ax_b_anchor = draw_panel_b(fig, gs[0, 1], payload["reconstruction"])
    set_panel_label(ax_b_anchor, "B")

    noiseless = bool(payload.get("noiseless", False))
    ax_c = fig.add_subplot(gs[0, 2])
    scatter_metrics(
        ax_c,
        metrics,
        "co-bps",
        "rate_r2",
        metric_rows,
        "co-BPS",
        "Rate R$^2$",
        tick_decimals=2,
        **(
            {}
            if noiseless
            else dict(
                xlim=(0.07, 0.13),
                ylim=(0.5, 1.05),
                xticks=[0.07, 0.13],
                yticks=[0.5, 1.0],
                include_callouts_in_limits=False,
            )
        ),
    )
    add_latent_size_legend(ax_c)
    set_panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 0])
    draw_placeholder(ax_d, "Simplicity Metrics", ["State R$^2$", "Cycle-Con"])
    set_panel_label(ax_d, "D")

    ax_e_anchor = draw_panel_e(fig, gs[1, 1], payload["simplicity"])
    set_panel_label(ax_e_anchor, "E")

    ax_f = fig.add_subplot(gs[1, 2])
    scatter_metrics(
        ax_f,
        metrics,
        "cycle_con",
        "state_r2",
        metric_rows,
        "Cycle-Con",
        "State R$^2$",
        tick_decimals=1,
        **(
            {}
            if noiseless
            else dict(
                xlim=(0.5, 1.05),
                ylim=(0.5, 1.05),
                xticks=[0.5, 1.0],
                yticks=[0.5, 1.0],
                include_callouts_in_limits=False,
            )
        ),
    )
    set_panel_label(ax_f, "F")

    ax_g = fig.add_subplot(gs[2, 0])
    scatter_metrics(
        ax_g,
        metrics,
        "state_r2",
        "rate_r2",
        metric_rows,
        "State R$^2$",
        "Rate R$^2$",
        tick_decimals=1,
        note="Using Ground-Truth",
        **(
            {}
            if noiseless
            else dict(
                xlim=(0.5, 1.05),
                ylim=(0.5, 1.05),
                xticks=[0.5, 1.0],
                yticks=[0.5, 1.0],
                include_callouts_in_limits=False,
            )
        ),
    )
    set_panel_label(ax_g, "G")

    ax_h = fig.add_subplot(gs[2, 1])
    scatter_metrics(
        ax_h,
        metrics,
        "cycle_con",
        "co-bps",
        metric_rows,
        "Cycle-Con",
        "co-BPS",
        tick_decimals=2,
        note="Without Ground-Truth",
        **(
            {}
            if noiseless
            else dict(
                xlim=(0.5, 1.05),
                ylim=(0.07, 0.13),
                xticks=[0.5, 1.0],
                yticks=[0.07, 0.13],
                include_callouts_in_limits=False,
            )
        ),
    )
    set_panel_label(ax_h, "H")

    ax_i = fig.add_subplot(gs[2, 2])
    draw_panel_i(ax_i)
    set_panel_label(ax_i, "I")

    noiseless = bool(payload.get("noiseless", False))
    if noiseless:
        fig.suptitle(
            "Noise-free external inputs",
            fontsize=10,
            fontweight="bold",
            color="#b22222",
            y=0.995,
        )
        fig.text(
            0.99,
            0.99,
            "TT + DD evaluated with noise-free task inputs",
            ha="right",
            va="top",
            fontsize=8,
            color="#b22222",
            style="italic",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if noiseless:
        suffix += "_noiseless"
    if payload.get("demean"):
        suffix += "_demean"
    pdf_path = output_dir / f"figure5{suffix}.pdf"
    png_path = output_dir / f"figure5{suffix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    return pdf_path, png_path


# %%
def main():
    args = parse_args()
    panel_b_timepoints = args.panel_b_timepoints or args.timepoints
    panel_e_timepoints = args.panel_e_timepoints or args.timepoints
    cache = args.cache
    default_cache = Path(__file__).with_suffix(".cache.pkl")
    if cache == default_cache:
        suffix = ""
        if args.noiseless:
            suffix += "_noiseless"
        if args.demean:
            suffix += "_demean"
        if suffix:
            cache = cache.with_name(cache.stem + suffix + cache.suffix)
    payload = build_payload(
        cache,
        args.force,
        args.trial,
        args.neuron,
        panel_b_timepoints,
        panel_e_timepoints,
        args.pc_index,
        noiseless=args.noiseless,
        demean=args.demean,
    )
    pdf_path, png_path = make_figure(payload, args.output_dir, args.dpi)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()


# %%
# --------------------------- Interactive snippet browser ---------------------------
# Run the cells below interactively (e.g. an IDE that recognizes ``# %%``) to
# preview many candidate snippets at once, find ones you like, and then plug the
# chosen indices into the CLI args ``--trial``, ``--neuron``, ``--pc-index``.
#
# The full unsliced arrays for the three representative NODE models are cached
# in BROWSE_CACHE so subsequent browse calls are fast — only the first call (or
# a call with a different LATENT_INDICES) pays the load cost.

BROWSE_CACHE = Path(__file__).with_suffix(".browse_cache.pkl")


def _browse_cache_path(noiseless: bool) -> Path:
    if noiseless:
        return BROWSE_CACHE.with_name(
            BROWSE_CACHE.stem + "_noiseless" + BROWSE_CACHE.suffix
        )
    return BROWSE_CACHE


def load_browse_payload(force: bool = False, noiseless: bool | None = None) -> dict:
    """Load (or build) the full-array cache used by ``browse_panel_b/e``.

    When ``noiseless`` is True the TT model is re-evaluated with the task's
    clean inputs, the simulator generates clean firing rates/spikes, and the DD
    models are re-run with clean inputs substituted on the recurrent path —
    so the browser shows the same data the noiseless figure uses. ``None``
    defaults to the module-level ``NOISELESS_FLAG``.
    """
    if noiseless is None:
        noiseless = NOISELESS_FLAG
    cache_path = _browse_cache_path(noiseless)
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        if (
            cached.get("latent_indices") == dict(LATENT_INDICES)
            and tuple(cached.get("reconstruction_latents", ()))
            == tuple(RECONSTRUCTION_LATENTS)
            and tuple(cached.get("simplicity_latents", ())) == tuple(SIMPLICITY_LATENTS)
            and cached.get("noiseless") == bool(noiseless)
        ):
            return cached
    torch.manual_seed(42)
    np.random.seed(42)
    comparison = load_comparison()
    reps = representative_indices(comparison)
    undos: list = []
    if noiseless:
        undos = apply_noiseless_overrides(comparison)
    try:
        target_rep = reps[RECONSTRUCTION_LATENTS[0]]
        true_rates_val = as_numpy(
            comparison.analyses[target_rep].get_true_rates(phase="val")
        )
        spikes_val = as_numpy(comparison.analyses[target_rep].get_spiking(phase="val"))
        true_lat_train = as_numpy(
            comparison.analyses[comparison.ref_ind].get_latents(phase="train")
        )
        true_lat_val = as_numpy(
            comparison.analyses[comparison.ref_ind].get_latents(phase="val")
        )
        per_latent: dict[int, dict] = {}
        for latent in CALLOUT_LATENTS:
            a = comparison.analyses[reps[latent]]
            rates_train = as_numpy(a.get_rates(phase="train"))
            rates_val = as_numpy(a.get_rates(phase="val"))
            per_latent[latent] = {
                "rates_val": rates_val,
                "latents_train": as_numpy(a.get_latents(phase="train")),
                "latents_val": as_numpy(a.get_latents(phase="val")),
                "log_rates_train": np.log(np.clip(rates_train, 1e-9, None)),
                "log_rates_val": np.log(np.clip(rates_val, 1e-9, None)),
            }
    finally:
        for undo in undos:
            undo()
    payload = {
        "latent_indices": dict(LATENT_INDICES),
        "reconstruction_latents": tuple(RECONSTRUCTION_LATENTS),
        "simplicity_latents": tuple(SIMPLICITY_LATENTS),
        "noiseless": bool(noiseless),
        "true_rates_val": true_rates_val,
        "spikes_val": spikes_val,
        "true_latents_train": true_lat_train,
        "true_latents_val": true_lat_val,
        "per_latent": per_latent,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)
    return payload


def _neuron_summary(bp, latent, neuron):
    true_full = bp["true_rates_val"][..., neuron]
    pred_full = bp["per_latent"][latent]["rates_val"][..., neuron]
    spike = bp["spikes_val"][..., neuron : neuron + 1]
    bps = bits_per_spike(
        bp["per_latent"][latent]["rates_val"][..., neuron : neuron + 1], spike
    )
    r2 = r2_score(true_full.reshape(-1), pred_full.reshape(-1))
    return float(bps), float(r2)


def browse_panel_b(
    trials=None,
    neurons=None,
    timepoints: int = 100,
    latents=RECONSTRUCTION_LATENTS,
    force: bool = False,
    noiseless: bool | None = None,
    title: str = "Panel B candidates (rows=trials, cols=neurons)",
):
    """Grid-preview candidate (trial, neuron) pairs for panel B.

    Each subplot overlays the true rate (black) with the model predictions for
    each requested latent. Per-neuron co-BPS and rate R² (computed across the
    full val set) are annotated so you can spot well-fit neurons quickly.
    ``noiseless`` defaults to ``NOISELESS_FLAG`` and controls whether the
    underlying data is the noise-free version (simulator-regenerated targets
    + clean-input DD predictions).
    """
    bp = load_browse_payload(force=force, noiseless=noiseless)
    true_rates = bp["true_rates_val"]
    spikes = bp["spikes_val"]
    n_trials, n_time, n_neurons = true_rates.shape
    if trials is None:
        trials = list(range(min(6, n_trials)))
    if neurons is None:
        neurons = list(range(min(6, n_neurons)))
    trials = list(trials)
    neurons = list(neurons)
    timepoints = min(timepoints, n_time)
    t = np.arange(timepoints)

    summaries = {
        latent: {n: _neuron_summary(bp, latent, n) for n in neurons}
        for latent in latents
    }

    n_rows, n_cols = len(trials), len(neurons)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.9 * n_cols, 1.4 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for r, tr in enumerate(trials):
        for c, n in enumerate(neurons):
            ax = axes[r, c]
            ax.plot(t, true_rates[tr, :timepoints, n], color=COLORS["truth"], lw=0.9)
            for latent in latents:
                pred = bp["per_latent"][latent]["rates_val"][tr, :timepoints, n]
                ax.plot(t, pred, color=latent_dim_color(latent), lw=0.8, alpha=0.85)
            spk = np.flatnonzero(spikes[tr, :timepoints, n] > 0)
            if spk.size:
                ymin, ymax = ax.get_ylim()
                span = ymax - ymin
                ax.vlines(
                    spk,
                    ymin + 0.02 * span,
                    ymin + 0.14 * span,
                    color="black",
                    linewidth=0.5,
                )
                ax.set_ylim(ymin, ymax)
            text = "\n".join(
                f"{latent}D bps={summaries[latent][n][0]:.2f} r2={summaries[latent][n][1]:.2f}"
                for latent in latents
            )
            ax.set_title(f"trial={tr} neuron={n}", fontsize=7)
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                ha="left",
            )
            ax.tick_params(labelsize=6)
            ax.set_yticks([])
    if bp.get("noiseless"):
        title = title + "  [noiseless]"
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def _fit_panel_e(bp, latent, demean: bool = False):
    """Fit the PCA + state/cycle regressions used by panel E for one latent size.

    When ``demean`` is True the returned actual/state/cycle arrays have each
    trial's per-PC time-mean subtracted, so downstream R² and plots reflect
    shape-only fits.
    """
    lat_train = bp["per_latent"][latent]["latents_train"]
    lat_val = bp["per_latent"][latent]["latents_val"]
    rates_train = bp["per_latent"][latent]["log_rates_train"]
    rates_val = bp["per_latent"][latent]["log_rates_val"]
    true_train = bp["true_latents_train"]
    true_val = bp["true_latents_val"]

    pca_lats = PCA()
    train_flat = pca_lats.fit_transform(lat_train.reshape(-1, lat_train.shape[-1]))
    val_flat = pca_lats.transform(lat_val.reshape(-1, lat_val.shape[-1]))

    pca_rates = PCA()
    rates_train_flat = pca_rates.fit_transform(
        rates_train.reshape(-1, rates_train.shape[-1])
    )
    rates_val_flat = pca_rates.transform(rates_val.reshape(-1, rates_val.shape[-1]))

    lm_state = LinearRegression().fit(
        true_train.reshape(-1, true_train.shape[-1]), train_flat
    )
    state_pred_flat = lm_state.predict(true_val.reshape(-1, true_val.shape[-1]))

    lm_readout = LinearRegression().fit(train_flat, rates_train_flat)
    pred_log_rates_flat = lm_readout.predict(val_flat)
    cycle_pred_flat = reconstruct_latents(lm_readout, pred_log_rates_flat)

    n_tr, n_tm = lat_val.shape[:2]
    actual = val_flat.reshape(n_tr, n_tm, -1)
    state_pred = state_pred_flat.reshape(n_tr, n_tm, -1)
    cycle_pred = cycle_pred_flat.reshape(n_tr, n_tm, -1)
    if demean:
        actual = _demean_per_trial(actual)
        state_pred = _demean_per_trial(state_pred)
        cycle_pred = _demean_per_trial(cycle_pred)
    return {
        "actual": actual,
        "state_pred": state_pred,
        "cycle_pred": cycle_pred,
        "n_pcs": val_flat.shape[1],
    }


def browse_panel_e(
    trials=None,
    pc_indices=None,
    timepoints: int = 100,
    latents=(8, 32),
    force: bool = False,
    noiseless: bool | None = None,
    demean: bool | None = None,
    title: str = "Panel E candidates (rows=trials, cols=PCs)",
):
    """Grid-preview candidate (trial, pc_index) pairs for panel E.

    Each subplot shows the data-driven PC trace (solid) with the state-R² and
    cycle-Con predictions (dashed/dotted) for each requested latent. Whole-PC
    R²s are annotated, so a snippet where both methods do well on a single PC
    is easy to spot. ``noiseless`` defaults to ``NOISELESS_FLAG``. When
    ``demean`` is True (defaults to ``DEMEAN_FLAG``) each trace has its own
    per-trial mean subtracted and the annotated R²s are recomputed on the
    demeaned series, so shape fits are scored independently of DC offset.
    """
    if demean is None:
        demean = DEMEAN_FLAG
    bp = load_browse_payload(force=force, noiseless=noiseless)
    if trials is None:
        trials = list(range(min(6, bp["true_latents_val"].shape[0])))
    trials = list(trials)
    fits = {latent: _fit_panel_e(bp, latent, demean=demean) for latent in latents}
    max_pcs = max(fits[latent]["n_pcs"] for latent in latents)
    if pc_indices is None:
        pc_indices = list(range(min(6, max_pcs)))
    pc_indices = list(pc_indices)

    timepoints = min(timepoints, bp["true_latents_val"].shape[1])
    t = np.arange(timepoints)

    pc_summary = {}
    for latent in latents:
        fit = fits[latent]
        pc_summary[latent] = {}
        for pc in pc_indices:
            dim = min(pc, fit["n_pcs"] - 1)
            actual = fit["actual"][:, :, dim].reshape(-1)
            state_r2 = r2_score(actual, fit["state_pred"][:, :, dim].reshape(-1))
            cycle_r2 = r2_score(actual, fit["cycle_pred"][:, :, dim].reshape(-1))
            pc_summary[latent][pc] = (float(state_r2), float(cycle_r2))

    n_rows, n_cols = len(trials), len(pc_indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.9 * n_cols, 1.4 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for r, tr in enumerate(trials):
        for c, pc in enumerate(pc_indices):
            ax = axes[r, c]
            text_lines = []
            for latent in latents:
                fit = fits[latent]
                dim = min(pc, fit["n_pcs"] - 1)
                ax.plot(
                    t,
                    fit["actual"][tr, :timepoints, dim],
                    color=latent_dim_color(latent),
                    lw=0.9,
                )
                ax.plot(
                    t,
                    fit["state_pred"][tr, :timepoints, dim],
                    color=light_latent_dim_color(latent),
                    lw=0.7,
                    linestyle="--",
                )
                ax.plot(
                    t,
                    fit["cycle_pred"][tr, :timepoints, dim],
                    color=light_latent_dim_color(latent),
                    lw=0.7,
                    linestyle=":",
                )
                state_r2, cycle_r2 = pc_summary[latent][pc]
                text_lines.append(f"{latent}D s={state_r2:.2f} c={cycle_r2:.2f}")
            ax.set_title(f"trial={tr} pc={pc}", fontsize=7)
            ax.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                fontsize=6,
                va="top",
                ha="left",
            )
            ax.tick_params(labelsize=6)
            ax.set_yticks([])
    suptitle = title + "  (— actual, -- state, ·· cycle)"
    if bp.get("noiseless"):
        suptitle += "  [noiseless]"
    if demean:
        suptitle += "  [per-trial demeaned]"
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    return fig


# %%
# Example: preview a coarse grid. Edit the ranges to zoom in on promising
# (trial, neuron) / (trial, pc) combos. The figure itself is not modified by
# these cells; once you've chosen, update --trial / --neuron / --pc-index.
if True:
    browse_panel_b(trials=range(8), neurons=range(8), timepoints=100)
    browse_panel_e(trials=range(8), pc_indices=range(8), timepoints=100)
    plt.show()

# %%
# --------------------------- Panel-E bias diagnostic ----------------------
# A reviewer flagged Fig. 5E as showing an unexpected mean offset between
# the data-driven PC and its affine prediction. With the affine map fit on the
# flattened training set, the *dataset-wide* residual mean is ~0; but residuals
# from slow, trial-specific components can still produce a local-mean offset in
# any 100-bin snippet. This helper makes the structure explicit:
#
#   1. Reports global mean(pred) - mean(actual) on val (sanity check that the
#      affine fit isn't actually biased on the full set).
#   2. Plots a per-trial histogram of mean(actual - pred) for the chosen PC.
#   3. Plots a few *full* trials (not snippets) with actual + state_pred so you
#      can see whether the snippet view exaggerates a slow residual lobe.
#
# Run interactively after running the main pipeline (so the comparison + reps
# are loaded), or pass an existing payload via ``payload`` if you have one.


def diagnose_panel_e_bias(
    latent: int = 64,
    pc_index: int | None = None,
    n_trial_examples: int = 4,
    noiseless: bool | None = None,
    show: bool = True,
):
    """Diagnose whether panel E's apparent mean offset is from snippet-view of
    slow residuals (rather than a biased affine fit). Calls ``plt.show()`` by
    default; pass ``show=False`` to suppress."""
    if noiseless is None:
        noiseless = NOISELESS_FLAG
    if pc_index is None:
        pc_index = parse_args([]).pc_index

    torch.manual_seed(42)
    np.random.seed(42)
    comparison = load_comparison()
    reps = representative_indices(comparison)
    undos: list = []
    if noiseless:
        undos = apply_noiseless_overrides(comparison)
    try:
        true_train = as_numpy(
            comparison.analyses[comparison.ref_ind].get_latents(phase="train")
        )
        true_val = as_numpy(
            comparison.analyses[comparison.ref_ind].get_latents(phase="val")
        )
        a = comparison.analyses[reps[latent]]
        lats_train = as_numpy(a.get_latents(phase="train"))
        lats_val = as_numpy(a.get_latents(phase="val"))
    finally:
        for undo in undos:
            undo()

    pca_lats = PCA()
    train_flat = pca_lats.fit_transform(lats_train.reshape(-1, lats_train.shape[-1]))
    val_flat = pca_lats.transform(lats_val.reshape(-1, lats_val.shape[-1]))
    lm = LinearRegression().fit(
        true_train.reshape(-1, true_train.shape[-1]), train_flat
    )
    pred_val_flat = lm.predict(true_val.reshape(-1, true_val.shape[-1]))

    dim = int(np.clip(pc_index, 0, val_flat.shape[1] - 1))
    n_trials, n_time = lats_val.shape[:2]
    actual = val_flat.reshape(n_trials, n_time, -1)[:, :, dim]
    pred = pred_val_flat.reshape(n_trials, n_time, -1)[:, :, dim]
    residual = actual - pred

    overall_bias = float(residual.mean())
    overall_actual_mean = float(actual.mean())
    overall_pred_mean = float(pred.mean())
    per_trial_mean = residual.mean(axis=1)
    state_r2_global = float(r2_score(val_flat[:, dim], pred_val_flat[:, dim]))

    print(
        f"--- panel-E bias diagnostic (NODE{latent}, PC{dim + 1}, noiseless={noiseless}) ---"
    )
    print(f"  global mean(actual) = {overall_actual_mean: .5f}")
    print(f"  global mean(pred)   = {overall_pred_mean: .5f}")
    print(f"  global mean(actual - pred) = {overall_bias: .5f}  (should be ~0)")
    print(f"  state-R² on val      = {state_r2_global: .4f}")
    print(
        "  per-trial mean(actual - pred): "
        f"mean={per_trial_mean.mean(): .4f}, "
        f"std={per_trial_mean.std(): .4f}, "
        f"range=[{per_trial_mean.min(): .4f}, {per_trial_mean.max(): .4f}]"
    )
    print(
        "  → if per-trial std is large relative to the PC range, the snippet "
        "you see is just landing on one residual lobe, not a biased fit."
    )

    fig = plt.figure(figsize=(11, 1.4 * n_trial_examples + 2.2))
    gs = GridSpec(
        n_trial_examples + 1,
        1,
        figure=fig,
        hspace=0.6,
        height_ratios=[1.0] * n_trial_examples + [1.4],
    )

    # Pick the most-extreme trials by |per-trial residual mean| — those are the
    # ones a casual viewer would call "biased".
    rank = np.argsort(np.abs(per_trial_mean))[::-1][:n_trial_examples]
    for row, trial in enumerate(rank):
        ax = fig.add_subplot(gs[row, 0])
        t_full = np.arange(n_time)
        ax.plot(
            t_full,
            actual[trial],
            color=latent_dim_color(latent),
            lw=1.0,
            label="actual",
        )
        ax.plot(
            t_full,
            pred[trial],
            color=light_latent_dim_color(latent),
            lw=1.0,
            label="affine pred",
        )
        ax.axhline(0, color="0.75", lw=0.6, alpha=0.6)
        ax.axvspan(0, 100, color="0.85", alpha=0.35, lw=0)
        ax.set_title(
            f"trial {trial}: mean(actual - pred) over full trial = "
            f"{per_trial_mean[trial]:+.3f}    "
            f"(grey band = first 100-bin snippet)",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=7)

    ax_hist = fig.add_subplot(gs[-1, 0])
    ax_hist.hist(
        per_trial_mean,
        bins=40,
        color="0.4",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_hist.axvline(0, color="black", lw=1.0)
    ax_hist.axvline(
        overall_bias,
        color="#E45756",
        lw=1.2,
        label=f"set-wide mean = {overall_bias:+.4f}",
    )
    ax_hist.set_xlabel("per-trial mean(actual - pred)")
    ax_hist.set_ylabel("# trials")
    ax_hist.set_title(
        f"Per-trial residual means (NODE{latent}, PC{dim + 1}) — width here = "
        "how much offset a single-trial snippet can show",
        fontsize=9,
    )
    ax_hist.legend(frameon=False, fontsize=8)
    ax_hist.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Panel E bias diagnostic  (NODE{latent}, PC{dim + 1}, "
        f"noiseless={noiseless})",
        fontsize=10,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig, {
        "per_trial_mean": per_trial_mean,
        "global_bias": overall_bias,
        "state_r2": state_r2_global,
    }


# %%
# Convenience cell — run after the main pipeline to actually render the
# diagnostic. Edit `latent` / `pc_index` to inspect different panel-E choices.
if False:
    diagnose_panel_e_bias(latent=64, pc_index=7)


# %%
# --------------------------- Level-shift evidence figure -------------------
# Targeted single-panel figure for the claim: "DD PC <pc> contains per-trial
# level shifts that aren't predictable from the TT latents." The validation
# set is concatenated trial-by-trial onto one x-axis with vertical separators
# and trials sorted by signed per-trial mean(actual − pred). The DD actual
# trace climbs in DC across trials while the TT→DD affine prediction sits
# roughly flat — the level shift is visible directly.


def make_level_shift_figure(
    latent: int = 32,
    pc_index: int = 7,
    n_trials_show: int = 12,
    noiseless: bool | None = None,
    save: bool = False,
    output_dir: Path | None = None,
    show: bool = True,
):
    """Single-panel strip showing per-trial DC offsets in DD PC ``pc_index``.

    Trials are sorted by signed per-trial mean(actual − pred); ``n_trials_show``
    of them (sampled uniformly along that order so the full DC gradient is
    visible) are concatenated onto one x-axis with vertical separators.
    """
    if noiseless is None:
        noiseless = NOISELESS_FLAG
    bp = load_browse_payload(noiseless=noiseless)
    fit = _fit_panel_e(bp, latent, demean=False)
    dim = int(np.clip(pc_index, 0, fit["n_pcs"] - 1))
    actual = fit["actual"][:, :, dim]
    pred = fit["state_pred"][:, :, dim]
    n_trials, n_time = actual.shape

    residual_mean = actual.mean(axis=1) - pred.mean(axis=1)
    r2_raw = float(r2_score(actual.reshape(-1), pred.reshape(-1)))

    order_by_offset = np.argsort(residual_mean)
    if n_trials_show >= n_trials:
        strip_trials = order_by_offset
    else:
        idx = np.linspace(0, n_trials - 1, n_trials_show).round().astype(int)
        strip_trials = order_by_offset[idx]

    color_actual = latent_dim_color(latent)
    color_pred = light_latent_dim_color(latent)

    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    x = 0
    for k, tr in enumerate(strip_trials):
        t = np.arange(x, x + n_time)
        ax.plot(t, actual[tr], color=color_actual, lw=0.9)
        ax.plot(t, pred[tr], color=color_pred, lw=0.9, linestyle="--")
        if k > 0:
            ax.axvline(x, color="0.85", lw=0.5)
        x += n_time
    ax.set_xlim(0, x)
    ax.set_xticks([n_time * (k + 0.5) for k in range(len(strip_trials))])
    ax.set_xticklabels(
        [f"t{tr}\nΔ={residual_mean[tr]:+.2f}" for tr in strip_trials], fontsize=6
    )
    ax.set_xlabel("trials sorted by per-trial mean(actual − pred)  →")
    ax.set_ylabel(f"PC {dim + 1}")
    ax.spines[["top", "right"]].set_visible(False)
    title = (
        f"Per-trial level shifts in NODE{latent} PC {dim + 1} "
        f"(R² over all val = {r2_raw:.2f})"
    )
    if noiseless:
        title += "  [noiseless]"
    ax.set_title(title, fontsize=10)
    ax.legend(
        [
            Line2D([0], [0], color=color_actual, lw=1.2),
            Line2D([0], [0], color=color_pred, lw=1.2, linestyle="--"),
        ],
        ["DD actual", "TT→DD affine pred"],
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()

    if save:
        out = output_dir or (Path(__file__).parent / "outputs")
        out.mkdir(parents=True, exist_ok=True)
        suffix = "_noiseless" if noiseless else ""
        png = out / f"level_shifts_NODE{latent}_PC{dim + 1}{suffix}.png"
        pdf = out / f"level_shifts_NODE{latent}_PC{dim + 1}{suffix}.pdf"
        fig.savefig(png, bbox_inches="tight", dpi=200)
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Saved {png}")
        print(f"Saved {pdf}")

    if show:
        plt.show()
    return fig


# %%
# Convenience cell — render the level-shift figure for NODE32, PC 8.
if True:
    make_level_shift_figure(latent=32, pc_index=7, n_trials_show=12, save=True)


# %%
# --------------------------- Panel-E vs candidates side-by-side debug ------
# Drop-in figure to diff what compute_simplicity_examples (i.e. the main
# figure's panel E) shows against what _fit_panel_e / browse_panel_e
# (the candidates plot) shows. Same (trial, pc, latent) on both sides, plus a
# third "overlay" column so any divergence is obvious. Pulls (trial, pc,
# demean, noiseless) directly from a chosen figure cache so the two sides are
# guaranteed to be configured identically.


def debug_compare_main_vs_candidates(
    figure_cache: Path | None = None,
    save: bool = False,
    output_dir: Path | None = None,
    show: bool = True,
):
    """Side-by-side debug: main panel E (cache) vs candidates path (re-fit).

    The function loads the figure cache, then re-runs the candidates path
    (``_fit_panel_e`` via ``load_browse_payload``) with exactly the same
    ``trial``, ``pc_index``, ``demean``, and ``noiseless`` settings. For each
    (latent, prediction) pair it draws three subplots: the cached main-figure
    trace, the freshly fit candidates trace, and the two overlaid. Each title
    shows ``max|Δ|`` between the two arrays so numerical mismatches stand out.
    """
    if figure_cache is None:
        script = Path(__file__).resolve()
        candidates = sorted(script.parent.glob(script.stem + "*.cache.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"No figure cache found next to {script}. Run the figure first."
            )
        figure_cache = max(candidates, key=lambda p: p.stat().st_mtime)
    figure_cache = Path(figure_cache)
    with figure_cache.open("rb") as f:
        payload = pickle.load(f)

    cfg = payload.get("config", {})
    sim = payload["simplicity"]
    trial = int(sim["trial"])
    pc_index = int(sim["pc_index"])
    demean = bool(sim.get("demean", cfg.get("demean", False)))
    noiseless = bool(payload.get("noiseless", cfg.get("noiseless", False)))
    simplicity_latents = tuple(cfg.get("simplicity_latents", SIMPLICITY_LATENTS))

    bp = load_browse_payload(force=True, noiseless=noiseless)
    fits = {lat: _fit_panel_e(bp, lat, demean=demean) for lat in simplicity_latents}

    # Determinism probe: call get_latents twice on a fresh comparison and see
    # whether the model itself returns the same numbers. If max|Δ| > 0 here,
    # the analysis model is in train mode (or otherwise stochastic) and that
    # alone explains why the main panel E and the candidates path disagree:
    # each path calls get_latents / get_rates in a different order, so each
    # sees a different forward pass with different dropout draws / IC samples.
    torch.manual_seed(42)
    np.random.seed(42)
    probe_cmp = load_comparison()
    probe_reps = representative_indices(probe_cmp)
    print("--- model-determinism probe ---")
    # TT drives true_train/true_val (and thus state_pred); each DD drives
    # latents/rates (and thus cycle_pred + actual). Probe both.
    tt = probe_cmp.analyses[probe_cmp.ref_ind]
    tt_obj = getattr(tt, "wrapper", None) or getattr(tt, "model", None)
    tt_training = (
        bool(getattr(tt_obj, "training", False)) if tt_obj is not None else None
    )
    tt_dnoise = (
        getattr(tt_obj, "dynamic_noise", "<missing>") if tt_obj is not None else None
    )
    tt_env_dnoise = (
        getattr(getattr(tt_obj, "task_env", None), "dynamic_noise", "<missing>")
        if tt_obj is not None
        else None
    )
    tt_init_hidden = (
        "init_hidden"
        if hasattr(getattr(tt_obj, "model", None), "init_hidden")
        else (
            "init_hidden_from_ic"
            if hasattr(getattr(tt_obj, "model", None), "init_hidden_from_ic")
            else "zeros"
        )
    )
    tt_inner = getattr(tt_obj, "model", None)
    tt_ic_var = getattr(tt_inner, "latent_ic_var", "<missing>")
    tt_noise_level = getattr(tt_inner, "noise_level", "<missing>")
    print(
        f"  TT     wrapper.training={tt_training}  "
        f"wrapper.dynamic_noise={tt_dnoise}  "
        f"task_env.dynamic_noise={tt_env_dnoise}  "
        f"hidden_init={tt_init_hidden}  "
        f"model.latent_ic_var={tt_ic_var}  "
        f"model.noise_level={tt_noise_level}"
    )
    tt_x1 = as_numpy(tt.get_latents(phase="val"))
    tt_x2 = as_numpy(tt.get_latents(phase="val"))
    tt_d = float(np.abs(tt_x1 - tt_x2).max())
    # Also probe by directly hitting the wrapper with a fixed input — this
    # isolates the wrapper.forward call from any get_model_inputs side effects.
    if tt_obj is not None:
        ics, inputs, _ = tt.get_model_inputs(phase="val")
        inputs_to_env = tt.get_inputs_to_env(phase="val")
        with torch.no_grad():
            o1 = tt_obj(ics, inputs, inputs_to_env)["latents"]
            o2 = tt_obj(ics, inputs, inputs_to_env)["latents"]
        tt_d_direct = float((o1 - o2).abs().max().item())
        print(f"  TT direct-forward max|Δ|={tt_d_direct:.2e}")
    print(
        f"  TT     max|Δ(get_latents x2)|={tt_d:.2e}  "
        f"{'DETERMINISTIC' if tt_d == 0 else 'STOCHASTIC → state_pred mismatch'}"
    )
    for lat in simplicity_latents:
        a = probe_cmp.analyses[probe_reps[lat]]
        model = getattr(a, "model", None)
        is_training = (
            bool(getattr(model, "training", False)) if model is not None else None
        )
        x1 = as_numpy(a.get_latents(phase="val"))
        x2 = as_numpy(a.get_latents(phase="val"))
        d = float(np.abs(x1 - x2).max())
        print(
            f"  NODE{lat:>2d}  model.training={is_training}  "
            f"max|Δ(get_latents x2)|={d:.2e}  "
            f"{'DETERMINISTIC' if d == 0 else 'STOCHASTIC → cycle/actual mismatch'}"
        )

    # Raw-input comparison: rebuild the comparison a third time and check
    # whether get_latents on the TT and DD reps returns the same numbers as
    # the browse-payload cache from this very session. This isolates whether
    # the mismatch is upstream of PCA (model output diverges between two
    # load_comparison calls) or downstream (PCA / LinearRegression).
    torch.manual_seed(42)
    np.random.seed(42)
    raw_cmp = load_comparison()
    raw_reps = representative_indices(raw_cmp)
    raw_tt_val = as_numpy(raw_cmp.analyses[raw_cmp.ref_ind].get_latents(phase="val"))
    raw_tt_train = as_numpy(
        raw_cmp.analyses[raw_cmp.ref_ind].get_latents(phase="train")
    )
    print("--- raw-latents reproducibility (load_comparison #2 vs #3) ---")
    bp_tt_train_d = float(np.abs(bp["true_latents_train"] - raw_tt_train).max())
    bp_tt_val_d = float(np.abs(bp["true_latents_val"] - raw_tt_val).max())
    print(f"  TT train max|Δ|={bp_tt_train_d:.2e}    TT val max|Δ|={bp_tt_val_d:.2e}")
    for lat in simplicity_latents:
        a = raw_cmp.analyses[raw_reps[lat]]
        train_d = float(
            np.abs(
                bp["per_latent"][lat]["latents_train"]
                - as_numpy(a.get_latents(phase="train"))
            ).max()
        )
        val_d = float(
            np.abs(
                bp["per_latent"][lat]["latents_val"]
                - as_numpy(a.get_latents(phase="val"))
            ).max()
        )
        print(f"  NODE{lat} train max|Δ|={train_d:.2e}    val max|Δ|={val_d:.2e}")

    rows = [
        (lat, key)
        for lat in simplicity_latents
        for key in ("actual", "state_pred", "cycle_pred")
    ]
    n_rows = len(rows)
    print("--- overlay deltas (main panel E cache vs candidates re-fit) ---")
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(11.5, 1.8 * n_rows),
        sharex=True,
        squeeze=False,
    )
    for r, (lat, key) in enumerate(rows):
        ax_main, ax_cand, ax_over = axes[r]
        fig_arr = np.asarray(sim[lat][key])
        n_show = len(fig_arr)
        t = np.arange(n_show)
        dim = min(pc_index, fits[lat]["n_pcs"] - 1)
        cand_arr = fits[lat][key][trial, :n_show, dim]
        delta = float(np.abs(fig_arr - cand_arr).max())
        match = np.allclose(fig_arr, cand_arr)

        c_main = latent_dim_color(lat)
        c_cand = light_latent_dim_color(lat)
        ax_main.plot(t, fig_arr, color=c_main, lw=1.2)
        ax_cand.plot(t, cand_arr, color=c_main, lw=1.2)
        ax_over.plot(t, fig_arr, color=c_main, lw=1.6, label="main (cache)")
        ax_over.plot(
            t, cand_arr, color=c_cand, lw=1.0, linestyle="--", label="candidates"
        )

        for ax in (ax_main, ax_cand, ax_over):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        ax_main.set_ylabel(f"NODE{lat}\n{key}", fontsize=8)
        ax_main.set_title("main panel E (cache)", fontsize=8)
        ax_cand.set_title("candidates (_fit_panel_e)", fontsize=8)
        ax_over.set_title(
            f"overlay  max|Δ|={delta:.2e}  {'OK' if match else 'MISMATCH'}",
            fontsize=8,
            color="black" if match else "#b22222",
        )
        print(
            f"  NODE{lat:>2d} {key:11s}  max|Δ|={delta:.2e}  "
            f"{'OK' if match else 'MISMATCH'}"
        )
        if r == 0:
            ax_over.legend(frameon=False, fontsize=7, loc="upper right")

    fig.suptitle(
        f"Panel E vs candidates  "
        f"trial={trial}  pc={pc_index}  demean={demean}  noiseless={noiseless}  "
        f"cache={figure_cache.name}",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    if save:
        out = output_dir or (Path(__file__).parent / "outputs")
        out.mkdir(parents=True, exist_ok=True)
        suffix = "_noiseless" if noiseless else ""
        suffix += "_demean" if demean else ""
        png = out / f"debug_panelE_vs_candidates{suffix}.png"
        fig.savefig(png, bbox_inches="tight", dpi=200)
        print(f"Saved {png}")

    if show:
        plt.show()
    return fig


# %%
# Convenience cell — drop-in side-by-side debug. Picks the most-recently
# modified figure cache automatically. Pass ``figure_cache=...`` to target a
# specific run (e.g. the demean or noiseless variant).
if True:
    debug_compare_main_vs_candidates(save=True)


# %%
