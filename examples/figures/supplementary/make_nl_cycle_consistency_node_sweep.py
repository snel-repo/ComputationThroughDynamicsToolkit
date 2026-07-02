"""Generate the supplementary figure for nonlinear cycle consistency on a NODE
latent-size sweep.

For each NODE data-driven (SAE) model in a sweep (varying ``latent_size`` and
``seed``), we fit a small MLP encoder ``f: log-rates -> TT latents`` on the
training split and evaluate its variance-weighted R^2 on validation under
increasing amounts of Gaussian noise injected in log-rate space. The figure
plots:

* Panel A: NL cycle-consistency R^2 vs. noise std, grouped/colored by latent
  size.
* Panel B: NL cycle-consistency R^2 (at a chosen noise std) vs. *linear* cycle
  consistency R^2, with the unity line, colored by latent size.
* Panel C: NL cycle-consistency R^2 vs. latent size at the clean (noise=0) and
  high-noise endpoints, with per-latent-size seed means and error bars.

Results are cached as a pickle in this directory so re-runs are cheap; pass
``--force`` to recompute.
"""

# %%
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
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
    import pytorch_lightning.utilities.model_helpers as _pl_model_helpers

    if not hasattr(_pl_model_helpers, "_ModuleMode"):

        class _ModuleMode:
            pass

        _pl_model_helpers._ModuleMode = _ModuleMode
except ImportError:
    pass

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.metrics import compute_nl_cycle_consistency, get_cycle_consistency

# %%
DEFAULT_NODE_SUBPATH = "20260520_NBFF_SAE_NODE_DimSweep"
DEFAULT_PRETRAINED_TT = REPO_ROOT / "pretrained" / "20241017_NBFF_NoisyGRU_NewFinal"
DEFAULT_NOISE_STDS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5)
SUPPLEMENTARY_OUTPUT_DIR = (
    REPO_ROOT / "examples" / "figures" / "supplementary" / "outputs"
)

# Manuscript-wide latent-size color mapping, kept in sync with
# examples/figures/Fig5Metrics/make_figure5_reconstruction_simplicity.py and
# make_compiled_metrics_vs_latent_size.py so a given latent size renders in the
# same color across every figure -- even when sweeps cover different latent sizes.
LATENT_CMAP = "turbo"
LATENT_COLOR_DIMS = np.array([2, 3, 5, 8, 16, 32, 64])
LATENT_PASTEL_MIX = 0.0
LATENT_CMAP_TRIM = (0.10, 0.95)
LATENT_PALETTE_PATH = REPO_ROOT / "latent_palette.json"
_LATENT_PALETTE_CACHE: dict[int, tuple[float, float, float, float]] | None = None


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
        "--cache",
        type=Path,
        default=Path(__file__).with_suffix(".cache.pkl"),
        help="Pickle of per-run metric arrays; re-used unless --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute metrics instead of using the cache.",
    )
    parser.add_argument(
        "--tt-path",
        type=Path,
        default=None,
        help="TT model dir (contains model.pkl, datamodule_sim.pkl).",
    )
    parser.add_argument(
        "--node-sweep-path",
        type=Path,
        default=None,
        help="Sweep dir containing one subfolder per NODE run (default: tt_3bff/"
        + DEFAULT_NODE_SUBPATH
        + ").",
    )
    parser.add_argument(
        "--task",
        default="3bff",
        help="Task tag (used for default paths and output names).",
    )
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=list(DEFAULT_NOISE_STDS),
        help="Noise standard deviations (in train log-rate std units).",
    )
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the MLP encoder fit in compute_nl_cycle_consistency.",
    )
    parser.add_argument(
        "--noise-scatter-ind",
        type=int,
        default=2,
        help="Index into --noise-stds used for the scatter (panel B).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of NODE runs evaluated.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force a torch device (default: cuda if available).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    if argv is None and _in_notebook():
        argv = []
    return parser.parse_args(argv)


# %%
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
    tt_dir = home / "content" / "trained_models" / "task-trained" / f"tt_{args.task}"
    tt_path = args.tt_path or tt_dir
    if args.tt_path is None and not (tt_path / "model.pkl").exists():
        if DEFAULT_PRETRAINED_TT.exists():
            tt_path = DEFAULT_PRETRAINED_TT
    sweep_path = args.node_sweep_path or (tt_dir / DEFAULT_NODE_SUBPATH)
    return tt_path, sweep_path


def parse_latent_seed(name: str) -> tuple[int | None, int | None]:
    lat_match = re.search(r"latent_size[=_-](\d+)", name)
    seed_match = re.search(r"seed[=_-](\d+)", name)
    latent = int(lat_match.group(1)) if lat_match else None
    seed = int(seed_match.group(1)) if seed_match else None
    return latent, seed


def discover_node_runs(sweep_path: Path) -> list[Path]:
    if not sweep_path.exists():
        raise FileNotFoundError(f"NODE sweep dir does not exist: {sweep_path}")
    subdirs = [p for p in sorted(sweep_path.iterdir()) if p.is_dir()]
    runs = [
        p
        for p in subdirs
        if (p / "model.pkl").exists() and (p / "datamodule.pkl").exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No NODE runs (with model.pkl + datamodule.pkl) under {sweep_path}"
        )
    return runs


def compute_run_metrics(
    args: argparse.Namespace,
    tt_path: Path,
    run_paths: list[Path],
) -> dict[str, np.ndarray]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    an_tt = Analysis_TT(run_name=f"tt_{args.task}", filepath=str(tt_path) + os.sep)
    an_tt.wrapper.eval()

    runs = run_paths if args.max_runs is None else run_paths[: args.max_runs]
    noise_stds = list(args.noise_stds)

    latent_sizes: list[int] = []
    seeds: list[int] = []
    run_names: list[str] = []
    val_r2: list[float] = []
    linear_cc: list[float] = []
    r2_per_noise: list[list[float]] = []
    train_curves: list[np.ndarray] = []
    val_curves: list[np.ndarray] = []
    best_epochs: list[int] = []

    for i, run_path in enumerate(runs):
        latent, seed = parse_latent_seed(run_path.name)
        if latent is None:
            print(f"[skip] {run_path.name}: could not parse latent_size")
            continue
        run_names.append(run_path.name)
        latent_sizes.append(latent)
        seeds.append(seed if seed is not None else -1)

        print(
            f"\n[{i+1}/{len(runs)}] latent_size={latent} seed={seed} -- {run_path.name}"
        )
        try:
            an_dd = Analysis_DD.create(
                run_name=f"NODE{latent}_seed{seed}",
                filepath=str(run_path) + os.sep,
                model_type="SAE",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [load fail] {type(exc).__name__}: {exc}")
            val_r2.append(np.nan)
            linear_cc.append(np.nan)
            r2_per_noise.append([np.nan] * len(noise_stds))
            train_curves.append(np.array([], dtype=float))
            val_curves.append(np.array([], dtype=float))
            best_epochs.append(-1)
            continue

        try:
            inf_rates_train, inf_latents_train = an_dd.get_model_outputs(phase="train")
            inf_rates_val, inf_latents_val = an_dd.get_model_outputs(phase="val")

            inf_latents_train_np = as_numpy(inf_latents_train)
            inf_rates_train_np = as_numpy(inf_rates_train)
            inf_latents_val_np = as_numpy(inf_latents_val)
            inf_rates_val_np = as_numpy(inf_rates_val)

            nl_cc = compute_nl_cycle_consistency(
                inf_latents_train=inf_latents_train_np,
                inf_rates_train=inf_rates_train_np,
                inf_latents_val=inf_latents_val_np,
                inf_rates_val=inf_rates_val_np,
                hidden_sizes=tuple(args.mlp_hidden),
                lr=args.lr if hasattr(args, "lr") else 1e-3,
                max_epochs=args.max_epochs,
                patience=args.patience,
                noise_stds=tuple(noise_stds),
                seed=args.seed,
                device=device,
            )
            lin_cc = get_cycle_consistency(
                inf_latents_train=inf_latents_train_np,
                inf_rates_train=inf_rates_train_np,
                inf_latents_val=inf_latents_val_np,
                inf_rates_val=inf_rates_val_np,
            )

            val_r2.append(float(nl_cc["val_r2"]))
            linear_cc.append(float(lin_cc))
            r2_per_noise.append([float(v) for v in nl_cc["r2_per_noise"]])
            train_curves.append(np.asarray(nl_cc.get("train_losses", []), dtype=float))
            val_curves.append(np.asarray(nl_cc.get("val_losses", []), dtype=float))
            best_epochs.append(int(nl_cc.get("best_epoch", -1)))
            print(f"  val_r2={val_r2[-1]:.3f} | linear_cc={linear_cc[-1]:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [metric fail] {type(exc).__name__}: {exc}")
            val_r2.append(np.nan)
            linear_cc.append(np.nan)
            r2_per_noise.append([np.nan] * len(noise_stds))
            train_curves.append(np.array([], dtype=float))
            val_curves.append(np.array([], dtype=float))
            best_epochs.append(-1)

    return {
        "run_names": np.array(run_names),
        "latent_sizes": np.array(latent_sizes, dtype=int),
        "seeds": np.array(seeds, dtype=int),
        "val_r2": np.array(val_r2, dtype=float),
        "linear_cc": np.array(linear_cc, dtype=float),
        "r2_per_noise": np.array(r2_per_noise, dtype=float),
        "noise_stds": np.array(noise_stds, dtype=float),
        "train_curves": np.array(train_curves, dtype=object),
        "val_curves": np.array(val_curves, dtype=object),
        "best_epochs": np.array(best_epochs, dtype=int),
    }


def load_or_compute_payload(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.cache.exists() and not args.force:
        with args.cache.open("rb") as f:
            cached = pickle.load(f)
        noise_match = list(cached.get("noise_stds", [])) == list(args.noise_stds)
        has_curves = "train_curves" in cached and "val_curves" in cached
        if noise_match and has_curves:
            print(f"Loaded cached metrics from {args.cache}")
            return cached
        if not noise_match:
            print("Cache noise_stds do not match args.noise_stds; recomputing.")
        else:
            print("Cache is missing training curves; recomputing.")

    tt_path, sweep_path = default_paths(args)
    if not (tt_path / "model.pkl").exists():
        raise FileNotFoundError(
            f"Could not find TT model.pkl under {tt_path}. Set --tt-path or HOME_DIR."
        )
    runs = discover_node_runs(sweep_path)
    print(f"Found {len(runs)} NODE runs under {sweep_path}")

    payload = compute_run_metrics(args, tt_path, runs)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved metric cache to {args.cache}")
    return payload


# %%
def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.18,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )


def _load_latent_palette() -> dict[int, tuple[float, float, float, float]] | None:
    """Load {D: rgba} from LATENT_PALETTE_PATH; cache the result.

    Returns None if the file is missing or unreadable, in which case callers
    fall back to sampling LATENT_CMAP. Mirrors the loader in the Figure 5 and
    compiled-metrics scripts so all figures pull from the same manuscript palette.
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
            f"falling back to {LATENT_CMAP} sampling."
        )
        return None
    _LATENT_PALETTE_CACHE = {int(k): to_rgba(v) for k, v in raw.items()}
    return _LATENT_PALETTE_CACHE


def latent_dim_color(latent_size: int) -> tuple[float, float, float, float]:
    """Return the manuscript color for ``latent_size`` (matches Figure 5).

    Prefers an exact lookup in the JSON palette; for sizes not in the palette,
    log2-interpolates between the two nearest entries. Falls back to sampling
    LATENT_CMAP over the fixed log2 range of LATENT_COLOR_DIMS when no palette
    file is present, so the color of a given latent size is stable across tasks
    regardless of which sizes a particular sweep contains.
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


def colors_by_latent(
    latent_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    unique = np.unique(latent_sizes[latent_sizes > 0])
    color_map = {int(lat): np.array(latent_dim_color(int(lat))) for lat in unique}
    colors = np.array(
        [color_map.get(int(lat), (0.5, 0.5, 0.5, 1.0)) for lat in latent_sizes]
    )
    return unique, colors, color_map


def _latent_sorted_legend(
    ax,
    color_map: dict[int, np.ndarray],
    extra_handles=None,
    extra_labels=None,
    **legend_kwargs,
) -> None:
    handles = []
    labels = []
    for lat in sorted(color_map.keys()):
        handles.append(plt.Line2D([], [], color=color_map[lat], lw=1.4))
        labels.append(f"latent={lat}")
    if extra_handles and extra_labels:
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    ax.legend(handles, labels, **legend_kwargs)


def draw_panel_a(ax, payload: dict[str, np.ndarray]) -> None:
    panel_label(ax, "A")
    noise_stds = payload["noise_stds"]
    r2_per_noise = payload["r2_per_noise"]
    latent_sizes = payload["latent_sizes"]
    _, _, color_map = colors_by_latent(latent_sizes)

    for i in range(r2_per_noise.shape[0]):
        lat = int(latent_sizes[i])
        if not np.all(np.isfinite(r2_per_noise[i])):
            continue
        ax.plot(
            noise_stds,
            r2_per_noise[i],
            marker="o",
            ms=3,
            lw=1.0,
            alpha=0.85,
            color=color_map.get(lat, (0.5, 0.5, 0.5, 1.0)),
        )
    ax.axhline(0.0, color="0.7", lw=0.6, ls=":")
    ax.set_xlabel("Noise std (in train log-rate std units)")
    ax.set_ylabel("NL cycle-consistency R$^2$")
    ax.set_title("Nonlinear cycle consistency vs. noise level")
    ax.set_ylim(-0.2, 1.05)
    _latent_sorted_legend(
        ax, color_map, frameon=False, loc="lower left", ncols=2, handlelength=1.2
    )


def draw_panel_b(ax, payload: dict[str, np.ndarray], noise_ind: int) -> None:
    panel_label(ax, "B")
    noise_stds = payload["noise_stds"]
    r2_per_noise = payload["r2_per_noise"]
    linear_cc = payload["linear_cc"]
    latent_sizes = payload["latent_sizes"]
    noise_ind = int(np.clip(noise_ind, 0, r2_per_noise.shape[1] - 1))
    _, colors, color_map = colors_by_latent(latent_sizes)

    finite = np.isfinite(linear_cc) & np.isfinite(r2_per_noise[:, noise_ind])
    ax.scatter(
        linear_cc[finite],
        r2_per_noise[finite, noise_ind],
        c=colors[finite],
        s=26,
        edgecolor="0.2",
        linewidth=0.3,
        alpha=0.9,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("Linear cycle-consistency R$^2$")
    ax.set_ylabel(f"NL cycle-consistency R$^2$ (noise={noise_stds[noise_ind]:g})")
    ax.set_title("Nonlinear vs. linear cycle consistency")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=color_map[lat], ms=5)
        for lat in sorted(color_map.keys())
    ]
    labels = [f"latent={lat}" for lat in sorted(color_map.keys())]
    ax.legend(
        handles, labels, frameon=False, loc="lower right", ncols=2, handlelength=1.2
    )


def draw_panel_c(ax, payload: dict[str, np.ndarray], noise_ind: int) -> None:
    panel_label(ax, "C")
    latent_sizes = payload["latent_sizes"]
    r2_per_noise = payload["r2_per_noise"]
    noise_stds = payload["noise_stds"]
    noise_ind = int(np.clip(noise_ind, 0, r2_per_noise.shape[1] - 1))

    unique = np.unique(latent_sizes[latent_sizes > 0])
    mean_clean = []
    err_clean = []
    mean_noisy = []
    err_noisy = []
    for lat in unique:
        mask = latent_sizes == lat
        clean_vals = r2_per_noise[mask, 0]
        noisy_vals = r2_per_noise[mask, noise_ind]
        clean_vals = clean_vals[np.isfinite(clean_vals)]
        noisy_vals = noisy_vals[np.isfinite(noisy_vals)]
        mean_clean.append(np.mean(clean_vals) if clean_vals.size else np.nan)
        err_clean.append(np.std(clean_vals) if clean_vals.size else 0.0)
        mean_noisy.append(np.mean(noisy_vals) if noisy_vals.size else np.nan)
        err_noisy.append(np.std(noisy_vals) if noisy_vals.size else 0.0)

    ax.errorbar(
        unique,
        mean_clean,
        yerr=err_clean,
        marker="o",
        ms=4,
        lw=1.2,
        capsize=2.5,
        color="#1f77b4",
        label="noise = 0",
    )
    ax.errorbar(
        unique,
        mean_noisy,
        yerr=err_noisy,
        marker="s",
        ms=4,
        lw=1.2,
        capsize=2.5,
        color="#d62728",
        label=f"noise = {noise_stds[noise_ind]:g}",
    )
    ax.set_xscale("log")
    ax.set_xticks(unique)
    ax.set_xticklabels([str(int(u)) for u in unique])
    ax.set_xlabel("NODE latent size")
    ax.set_ylabel("NL cycle-consistency R$^2$ (mean$\\pm$std over seeds)")
    ax.set_title("Cycle consistency vs. latent dimensionality")
    ax.set_ylim(-0.2, 1.05)
    ax.legend(frameon=False, loc="lower left")


def draw_panel_d(ax_train, ax_val, payload: dict[str, np.ndarray]) -> None:
    panel_label(ax_train, "C")
    panel_label(ax_val, "D")
    latent_sizes = payload["latent_sizes"]
    train_curves = payload.get("train_curves")
    val_curves = payload.get("val_curves")
    best_epochs = payload.get("best_epochs")
    if train_curves is None or val_curves is None:
        for ax in (ax_train, ax_val):
            ax.text(
                0.5,
                0.5,
                "training curves not in cache",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
            ax.set_axis_off()
        return

    _, _, color_map = colors_by_latent(latent_sizes)

    plotted_any = False
    for i in range(len(train_curves)):
        tr = np.asarray(train_curves[i], dtype=float)
        va = np.asarray(val_curves[i], dtype=float)
        if tr.size == 0 or va.size == 0:
            continue
        lat = int(latent_sizes[i])
        color = color_map.get(lat, (0.5, 0.5, 0.5, 1.0))
        epochs = np.arange(1, tr.size + 1)
        ax_train.plot(epochs, tr, color=color, lw=0.9, alpha=0.85)
        ax_val.plot(epochs, va, color=color, lw=0.9, alpha=0.85)
        if best_epochs is not None:
            be = int(best_epochs[i])
            if 0 <= be < va.size:
                ax_val.plot(
                    [be + 1],
                    [va[be]],
                    marker="o",
                    ms=3.0,
                    color=color,
                    mec="0.15",
                    mew=0.4,
                    alpha=0.95,
                )
        plotted_any = True

    if not plotted_any:
        for ax in (ax_train, ax_val):
            ax.text(
                0.5,
                0.5,
                "no training curves recorded",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.4",
            )
            ax.set_axis_off()
        return

    for ax in (ax_train, ax_val):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
    ax_train.set_ylabel("MSE loss (standardized Z)")
    ax_train.set_title("Train loss")
    ax_val.set_title("Val loss")

    # Share y-limits between train and val so divergence is visually comparable.
    lo = min(ax_train.get_ylim()[0], ax_val.get_ylim()[0])
    hi = max(ax_train.get_ylim()[1], ax_val.get_ylim()[1])
    ax_train.set_ylim(lo, hi)
    ax_val.set_ylim(lo, hi)
    ax_val.tick_params(labelleft=False)

    _latent_sorted_legend(
        ax_val,
        color_map,
        frameon=False,
        loc="upper right",
        ncols=1,
        handlelength=1.2,
        fontsize=6,
    )


def build_figure(
    payload: dict[str, np.ndarray], args: argparse.Namespace
) -> plt.Figure:
    setup_matplotlib()
    fig = plt.figure(figsize=(10.0, 8.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_d_train = fig.add_subplot(gs[1, 0])
    ax_d_val = fig.add_subplot(gs[1, 1])
    draw_panel_a(ax_a, payload)
    draw_panel_b(ax_b, payload, args.noise_scatter_ind)
    draw_panel_d(ax_d_train, ax_d_val, payload)
    fig.suptitle(
        f"Nonlinear cycle consistency across NODE latent sizes ({args.task})",
        x=0.02,
        y=1.00,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)


# %%
def main(argv=None) -> None:
    args = parse_args(argv)
    payload = load_or_compute_payload(args)
    fig = build_figure(payload, args)
    stem = f"FigureS_NL_CycleCon_NODE_Sweep_{args.task}"
    save_figure(fig, args.output_dir, stem, args.dpi)
    plt.close(fig)
    print(f"Saved NL cycle-consistency figure outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
