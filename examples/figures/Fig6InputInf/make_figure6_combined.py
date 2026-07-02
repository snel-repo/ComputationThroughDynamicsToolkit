"""Generate Figure 6 (input-inference) combined panels A-F.

Scriptable replacement for ``Figure6_Combined.ipynb``. Designed to be run cell-
by-cell in the VSCode/PyCharm interactive Python window (each ``# %%`` block is
a cell), or end-to-end from the command line.

Expensive computations (loading the LFADS sweep, ``compute_metrics``,
``find_fixed_points``) are cached to ``make_figure6_combined.cache.pkl`` so
that re-running the script only repeats the plotting code. Pass ``--force`` to
invalidate the cache.
"""

# %%
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "libs" / "DSA", REPO_ROOT / "libs" / "lfads-jslds"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dotenv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

dotenv.load_dotenv(dotenv.find_dotenv())


# Module-level toggles; the CLI flags override these when set.
FORCE_RECOMPUTE_METRICS = False
FORCE_RECOMPUTE_FPS = True
TRIAL_INDEX = 1
GOOD_COLOR = "tab:orange"
BAD_COLOR = "tab:cyan"

# Match FigInputInfFPFinding.ipynb: hardcoded full-list indices into
# ``comparison.analyses`` (after ``regroup()``). The notebook uses these
# instead of argmax(input_r2)/argmax(co-bps), so the FP results — and
# Panel B traces / C-D highlights — line up across the figure family.
BEST_MODEL_IND = 5
BAD_MODEL_IND = 0

# Match the notebook's plot_fps default — find_fixed_points seeded at 0 so
# the optimizer's random initial states reproduce.
FPS_SEED = 0
FPS_N_INITS = 1024

# Render-only thresholds (match FigInputInfFPFinding.ipynb).
Q_THRESH_GOOD = 1e-5
Q_THRESH_BAD = 1e-5
NUM_TRAJ = 10
VIEW_BAD = (30, -10)  # (elev, azim)
VIEW_GOOD = (30, 0)


# %%
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "outputs"
    )
    parser.add_argument(
        "--metrics-cache",
        type=Path,
        default=Path(__file__).parent / "make_figure6_combined.metrics.pkl",
        help="Cache for the comparison-derived data (metrics, traces, model selection).",
    )
    parser.add_argument(
        "--fps-cache",
        type=Path,
        default=Path(__file__).parent / "make_figure6_combined.fps.pkl",
        help="Cache for the fixed-point payload (slow find_fixed_points calls).",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="CLI convenience: recompute BOTH caches (overrides both module toggles).",
    )
    parser.add_argument(
        "--force-metrics",
        action=argparse.BooleanOptionalAction,
        default=FORCE_RECOMPUTE_METRICS,
        help="Recompute the metrics cache only (FPs still cached).",
    )
    parser.add_argument(
        "--force-fps",
        action=argparse.BooleanOptionalAction,
        default=FORCE_RECOMPUTE_FPS,
        help="Recompute the FPs cache only (metrics still cached).",
    )
    parser.add_argument("--trial", type=int, default=TRIAL_INDEX)
    parser.add_argument(
        "--best-model-ind",
        type=int,
        default=BEST_MODEL_IND,
        help="Index into comparison.analyses (full list, after regroup) for the 'good' model.",
    )
    parser.add_argument(
        "--bad-model-ind",
        type=int,
        default=BAD_MODEL_IND,
        help="Index into comparison.analyses (full list, after regroup) for the 'bad' model.",
    )
    parser.add_argument(
        "--fps-seed",
        type=int,
        default=FPS_SEED,
        help="Seed passed to find_fixed_points (matches plot_fps default of 0).",
    )
    parser.add_argument(
        "--fps-n-inits",
        type=int,
        default=FPS_N_INITS,
        help="n_inits passed to find_fixed_points (matches fig3's standard FP finder).",
    )
    return parser.parse_args(argv)


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


if _in_notebook():
    args = parse_args([])
else:
    args = parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
force_metrics = args.force or args.force_metrics
force_fps = args.force or args.force_fps
print(f"Metrics cache: {args.metrics_cache}  force={force_metrics}")
print(f"FPs cache:     {args.fps_cache}  force={force_fps}")
print(f"Output:        {args.output_dir}")


# %%
def effective_inputs(inputs: np.ndarray) -> np.ndarray:
    """Return only the +/-1 impulses that actually flip the corresponding bit."""
    B, T, N = inputs.shape
    states = np.zeros((B, N), dtype=int)
    effective = np.zeros_like(inputs)
    for t in range(T):
        x_t = inputs[:, t, :]
        eff_pos = (states == 0) & (x_t == 1)
        eff_neg = (states == 1) & (x_t == -1)
        e_slice = effective[:, t, :]
        e_slice[eff_pos] = 1
        e_slice[eff_neg] = -1
        states[eff_pos] = 1
        states[eff_neg] = 0
    return effective


def ineffective_inputs(inputs: np.ndarray) -> np.ndarray:
    """Return the +/-1 impulses that do NOT flip the bit (redundant inputs)."""
    B, T, N = inputs.shape
    states = np.zeros((B, N), dtype=int)
    ineffective = np.zeros_like(inputs)
    for t in range(T):
        x_t = inputs[:, t, :]
        ineff_pos = (states == 1) & (x_t == 1)
        ineff_neg = (states == 0) & (x_t == -1)
        i_slice = ineffective[:, t, :]
        i_slice[ineff_pos] = 1
        i_slice[ineff_neg] = -1
        flip_pos = (states == 0) & (x_t == 1)
        flip_neg = (states == 1) & (x_t == -1)
        states[flip_pos] = 1
        states[flip_neg] = 0
    return ineffective


# %%
def load_comparison():
    """Construct the TT + LFADS-sweep comparison. Not cached (holds live torch models)."""
    HOME_DIR = os.environ["HOME_DIR"]
    path_TT = HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
    path_LFADS = path_TT + "20241004_NBFF_InputInf_Replication/"

    an_TT = Analysis_TT(run_name="TT", filepath=path_TT)
    comparison = Comparison(comparison_tag="Input_InfTest_Sweep_Random")
    comparison.load_analysis(an_TT, reference_analysis=True)

    subfolders = [f.path for f in os.scandir(path_LFADS) if f.is_dir()]
    n_models = len(subfolders)
    last_seed = 0
    for i, sub in enumerate(subfolders):
        final_folder = sub.split("/")[-1]
        sub = sub + "/"
        print(f"\rLoading {i + 1}/{n_models} LFADS models", end="", flush=True)
        co_kl = float(final_folder.split("kl_co_scale=")[1].split("_")[0])
        seed = int(final_folder.split("seed=")[1].split("_")[0])
        last_seed = seed
        an_dd = Analysis_DD.create(
            run_name=f"LFADS_co_kl_{co_kl}_{seed}",
            filepath=sub,
            model_type="LFADS",
        )
        comparison.load_analysis(an_dd, group=f"CO_KL_{co_kl}")
    print()  # newline after the in-place "Loading X/N" updates
    comparison.regroup()
    return an_TT, comparison, last_seed


# %%
# Stage 1 — cheap-ish: load the comparison, compute metrics, cache every DD
# model's inferred inputs. Model-selection-agnostic: changing which model is
# "good" or "bad" later does NOT invalidate this cache. Written to disk before
# the FP optimization starts.
def build_metrics_payload(force: bool, cache_path: Path):
    """Return (metrics_payload, live_handles_or_None).

    On cache hit returns (payload, None). On miss loads the sweep, computes
    metrics + per-model inferred inputs, writes the cache, and returns
    (payload, (comparison, seed, dd_analyses)) so the FP stage can reuse the
    already-loaded analyses without re-loading the sweep.
    """
    required_keys = {
        "metrics",
        "run_names",
        "full_analyses_info",
        "all_inferred_inputs",
        "true_inputs",
        "controlled",
        "effective",
        "ineffective",
        "kl_co_scales",
        "seed",
    }
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        missing = required_keys - set(payload)
        if not missing:
            print(f"Loaded metrics cache from {cache_path}")
            return payload, None
        print(
            f"Metrics cache at {cache_path} is missing keys {sorted(missing)} "
            "(probably written by an older script version) — rebuilding."
        )

    print("Metrics cache miss — loading LFADS sweep and computing metrics...")
    an_TT, comparison, seed = load_comparison()

    metric_dict_list = {
        "rate_r2": {},
        "state_r2": {},
        "input_r2": {},
        "cycle_con": {"variance_threshold": 0.01},
        "co-bps": {},
    }
    metrics = comparison.compute_metrics(metric_dict_list=metric_dict_list)

    dd_analyses = [a for a in comparison.analyses if isinstance(a, Analysis_DD)]
    run_names = [a.run_name for a in dd_analyses]
    kl_co_scales = np.array(
        [float(rn.split("co_kl_")[1].split("_")[0]) for rn in run_names]
    )

    # Record the full analyses-list layout so the selection cell can print a
    # complete index table without needing the live comparison object.
    full_analyses_info = [
        {"index": i, "run_name": a.run_name, "type": type(a).__name__}
        for i, a in enumerate(comparison.analyses)
    ]

    true_inputs = an_TT.get_true_inputs(phase="val").detach().cpu().numpy()
    out_dict = an_TT.get_model_outputs(phase="val")
    controlled = out_dict["controlled"].detach().cpu().numpy()

    print("Caching inferred inputs for every DD model...")
    all_inferred_inputs = {
        a.run_name: a.get_inferred_inputs(phase="val").detach().cpu().numpy()
        for a in dd_analyses
    }

    eff_mat = effective_inputs(true_inputs)
    ineff_mat = ineffective_inputs(true_inputs)

    payload = {
        "metrics": metrics,
        "kl_co_scales": kl_co_scales,
        "run_names": run_names,
        "full_analyses_info": full_analyses_info,
        "seed": seed,
        "controlled": controlled,
        "true_inputs": true_inputs,
        "effective": eff_mat,
        "ineffective": ineff_mat,
        "all_inferred_inputs": all_inferred_inputs,
    }

    # Write the cache NOW so a long/interrupted FP run still leaves the
    # metrics work saved to disk.
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved metrics cache to {cache_path}")

    return payload, (comparison, seed, dd_analyses)


metrics_payload, _live = build_metrics_payload(force_metrics, args.metrics_cache)


# %%
# ---------------- Pick the "good" and "bad" model by index ----------------
# Indices are into ``comparison.analyses`` (the full list, including the TT
# reference). The table below shows every loaded analysis and (for DD models)
# its key metrics. Adjust BEST_MODEL_IND / BAD_MODEL_IND at the top of the
# file — or pass ``--best-model-ind`` / ``--bad-model-ind`` on the CLI — and
# re-run from this cell. Changing the selection does NOT invalidate the
# metrics cache; the FPs cache is keyed by run name and refreshes as needed.


def _print_model_table(payload: dict) -> None:
    metrics = payload["metrics"]
    run_names = payload["run_names"]
    dd_metric_by_run = {
        rn: {
            "input_r2": float(metrics["input_r2"][i]),
            "co-bps": float(metrics["co-bps"][i]),
            "cycle_con": float(metrics["cycle_con"][i]),
            "rate_r2": float(metrics["rate_r2"][i]),
        }
        for i, rn in enumerate(run_names)
    }
    header = f"{'idx':>3}  {'type':<14}  {'run_name':<42}  {'input_r2':>9}  {'co-bps':>8}  {'cycle':>7}"
    print(header)
    print("-" * len(header))
    for entry in payload["full_analyses_info"]:
        m = dd_metric_by_run.get(entry["run_name"])
        if m is None:
            metric_str = f"{'-':>9}  {'-':>8}  {'-':>7}"
        else:
            metric_str = (
                f"{m['input_r2']:>9.3f}  {m['co-bps']:>8.3f}  {m['cycle_con']:>7.3f}"
            )
        print(
            f"{entry['index']:>3}  {entry['type']:<14}  {entry['run_name']:<42}  {metric_str}"
        )


_print_model_table(metrics_payload)


def _resolve_selected_models(payload: dict, best_ind: int, bad_ind: int):
    """Validate the requested indices and return their run names."""
    info = payload["full_analyses_info"]
    if not (0 <= best_ind < len(info)) or not (0 <= bad_ind < len(info)):
        raise IndexError(
            f"best/bad index out of range; comparison.analyses has {len(info)} entries."
        )
    best_entry = info[best_ind]
    bad_entry = info[bad_ind]
    for label, entry in (("good", best_entry), ("bad", bad_entry)):
        if entry["run_name"] not in payload["all_inferred_inputs"]:
            raise RuntimeError(
                f"{label} model at index {entry['index']} ({entry['run_name']}) "
                f"is a {entry['type']}, which has no inferred inputs. "
                f"Pick a different index (rows shown as 'Analysis_DD' above)."
            )
    return best_entry["run_name"], bad_entry["run_name"]


def _rotate_to_true(inferred: np.ndarray, true_inputs: np.ndarray) -> np.ndarray:
    """Linear-regress inferred inputs onto true inputs and return the rotated traces."""
    flat_inf = inferred.reshape(-1, inferred.shape[-1])
    flat_true = true_inputs.reshape(-1, true_inputs.shape[-1])
    reg = LinearRegression().fit(flat_inf, flat_true)
    return reg.predict(flat_inf).reshape(true_inputs.shape)


best_run_name, bad_run_name = _resolve_selected_models(
    metrics_payload, args.best_model_ind, args.bad_model_ind
)
best_inp_metric_ind = metrics_payload["run_names"].index(best_run_name)
bad_inp_metric_ind = metrics_payload["run_names"].index(bad_run_name)
rot_best = _rotate_to_true(
    metrics_payload["all_inferred_inputs"][best_run_name],
    metrics_payload["true_inputs"],
)
rot_bad = _rotate_to_true(
    metrics_payload["all_inferred_inputs"][bad_run_name],
    metrics_payload["true_inputs"],
)
print(f"\nGood model (analyses[{args.best_model_ind}]): {best_run_name}")
print(f"Bad  model (analyses[{args.bad_model_ind}]):  {bad_run_name}")


# %%
# Stage 2 — slow: find fixed points for the chosen good/bad models. The cache
# stores results PER RUN NAME, so picking a different index in the selection
# cell only triggers FP recomputation if that specific model hasn't been
# fixed-point-searched yet — every previously-searched model stays cached.
def build_fps_payload(
    force: bool,
    cache_path: Path,
    metrics_payload: dict,
    live_handles,
    fps_seed: int,
    best_run_name: str,
    bad_run_name: str,
    *,
    n_inits: int,
):
    fp_cache_by_run: dict[str, dict] = {}
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            fp_cache_by_run = pickle.load(f)
        print(
            f"Loaded FPs cache from {cache_path} (contains {len(fp_cache_by_run)} model(s))"
        )

    missing = [
        name for name in (best_run_name, bad_run_name) if name not in fp_cache_by_run
    ]
    if missing:
        print(f"FPs cache miss for: {missing}")
        if live_handles is None:
            print("(re-loading LFADS sweep to recover the analysis objects)")
            _, comparison, _ = load_comparison()
            dd_analyses = [a for a in comparison.analyses if isinstance(a, Analysis_DD)]
        else:
            comparison, _seed_live, dd_analyses = live_handles

        run_names_now = [a.run_name for a in dd_analyses]
        if run_names_now != metrics_payload["run_names"]:
            raise RuntimeError(
                "Run-name order changed since the metrics cache was written. "
                "Re-run with --force-metrics."
            )
        analyses_by_run = {a.run_name: a for a in dd_analyses}

        for run_name in missing:
            analysis = analyses_by_run[run_name]
            print(f"Finding fixed points for {run_name}...")
            inputs = analysis.get_inferred_inputs(phase="val")
            mean_in = torch.mean(inputs, dim=0).mean(dim=0).detach()
            fps = analysis.compute_FPs(
                inputs=mean_in,
                n_inits=n_inits,
                device="cpu",
                seed=fps_seed,
                compute_jacobians=True,
            )
            _, lats = analysis.get_model_outputs(phase="val")
            fp_cache_by_run[run_name] = {
                "fps": fps,
                "latents": lats.detach().cpu().numpy(),
                "fps_seed": fps_seed,
                "n_inits": n_inits,
            }

        with cache_path.open("wb") as f:
            pickle.dump(fp_cache_by_run, f)
        print(f"Saved FPs cache to {cache_path}")

    payload = {
        "fps_best": fp_cache_by_run[best_run_name]["fps"],
        "fps_bad": fp_cache_by_run[bad_run_name]["fps"],
        "latents_best": fp_cache_by_run[best_run_name]["latents"],
        "latents_bad": fp_cache_by_run[bad_run_name]["latents"],
    }
    return payload


fps_payload = build_fps_payload(
    force_fps,
    args.fps_cache,
    metrics_payload,
    _live,
    args.fps_seed,
    best_run_name,
    bad_run_name,
    n_inits=args.fps_n_inits,
)
payload = {
    **metrics_payload,
    **fps_payload,
    "rot_best": rot_best,
    "rot_bad": rot_bad,
    "best_inp": best_inp_metric_ind,
    "bad_inp": bad_inp_metric_ind,
    "best_model_ind": args.best_model_ind,
    "bad_model_ind": args.bad_model_ind,
}


# %%
def _clean_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _plot_fps_on_ax(ax, fps, latents, q_thresh=1e-6, n_traj=NUM_TRAJ, scale=1.0):
    """Plot PCA-projected latent trajectories and FPs onto a 3D axes.

    ``scale`` multiplies line widths and marker sizes; pass 1.5 to make
    everything 50% bigger.
    """
    x_star = fps.xstar
    q_star = fps.qstar
    keep = q_star < q_thresh
    x_star = x_star[keep]
    stability = fps.is_stable[keep]
    pca = PCA(n_components=3)
    lats_flat = latents.reshape(-1, latents.shape[-1])
    lats_pca = pca.fit_transform(lats_flat).reshape(
        latents.shape[0], latents.shape[1], 3
    )
    x_pca = pca.transform(x_star)
    for i in range(min(n_traj, lats_pca.shape[0])):
        ax.plot(
            lats_pca[i, :, 0],
            lats_pca[i, :, 1],
            lats_pca[i, :, 2],
            linewidth=0.5 * scale,
        )
    stable = x_pca[stability]
    unstable = x_pca[~stability]
    ax.scatter(
        stable[:, 0],
        stable[:, 1],
        stable[:, 2],
        c="g",
        marker="o",
        s=30 * scale * scale,
    )
    ax.scatter(
        unstable[:, 0],
        unstable[:, 1],
        unstable[:, 2],
        c="r",
        marker="x",
        s=40 * scale * scale,
        linewidths=1.5 * scale,
    )


# Per-bit color palettes for Panel B traces.
STATE_PALETTE = ["#90EE90", "#2ca02c", "#006400"]  # light / mid / dark green
EFFECTIVE_PALETTE = ["#A0A0A0", "#606060", "#202020"]  # light / mid / dark grey
INFERRED_PALETTE = ["#FFA0A0", "#d62728", "#8B0000"]  # light / mid / dark red


def _format_kl_label(kl: float) -> str:
    """Render a KL coefficient as a compact Xe-Y string (e.g. '1e-2', '3e-1')."""
    if kl == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(kl))))
    mantissa = kl / (10**exp)
    if abs(mantissa - round(mantissa)) < 1e-6:
        mantissa_str = f"{int(round(mantissa))}"
    else:
        mantissa_str = f"{mantissa:.1f}"
    if exp == 0:
        return mantissa_str
    return f"{mantissa_str}e{exp}"


# %%
def make_figure(payload: dict, trial_num: int, output_dir: Path) -> Path:
    metrics = payload["metrics"]
    kl_co_scales = np.asarray(payload["kl_co_scales"])
    best_inp = payload["best_inp"]
    bad_inp = payload["bad_inp"]

    fig = plt.figure(figsize=(16, 12))
    outer = gridspec.GridSpec(
        nrows=1,
        ncols=3,
        width_ratios=[1, 1.05, 1.05],
        wspace=0.28,
        figure=fig,
    )

    # ---- Column 0: A (1/5) on top, B (4/5) below ----
    col0 = gridspec.GridSpecFromSubplotSpec(
        nrows=5,
        ncols=1,
        subplot_spec=outer[0, 0],
        hspace=0.35,
    )

    # Panel A — placeholder (top 1/5)
    ax_A = fig.add_subplot(col0[0, 0])
    ax_A.text(
        0.5,
        0.5,
        "A - Inference schematic\n(Illustrator)",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax_A.transAxes,
    )
    ax_A.set_xticks([])
    ax_A.set_yticks([])
    for spine in ax_A.spines.values():
        spine.set_linestyle((0, (3, 3)))
        spine.set_edgecolor("0.6")
    ax_A.set_title("A", loc="left", fontweight="bold", fontsize=14)

    # Panel B — 5 stacked traces in bottom 4/5 (state, true, effective, good-inf, bad-inf)
    inner_B = gridspec.GridSpecFromSubplotSpec(
        nrows=5, ncols=1, subplot_spec=col0[1:5, 0], hspace=0.4
    )
    b_titles = [
        "State",
        "True inputs",
        "Effective inp",
        "Inferred (good)",
        "Inferred (bad)",
    ]
    b_data = [
        payload["controlled"][trial_num],
        payload["true_inputs"][trial_num],
        payload["effective"][trial_num],
        payload["rot_best"][trial_num],
        payload["rot_bad"][trial_num],
    ]
    b_palettes = [
        STATE_PALETTE,
        EFFECTIVE_PALETTE,
        EFFECTIVE_PALETTE,
        INFERRED_PALETTE,
        INFERRED_PALETTE,
    ]
    for i, (data, title, palette) in enumerate(zip(b_data, b_titles, b_palettes)):
        ax = fig.add_subplot(inner_B[i, 0])
        n_ch = data.shape[-1]
        for k in range(n_ch):
            color = palette[k % len(palette)]
            ax.plot(data[:, k], color=color, linewidth=1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(
            title, fontsize=9, rotation=0, ha="right", va="center", labelpad=4
        )
        if i == 0:
            ax.set_title("B", loc="left", fontweight="bold", fontsize=14)

    # ---- Helper for the C/D scatters with discrete KL legend ----
    def _scatter_with_kl_legend(ax, y_vals, y_label, title_letter):
        from matplotlib.lines import Line2D

        unique_kls = np.sort(np.unique(kl_co_scales))
        # Darker dot = larger KL (more regularisation). Span [0.15, 0.85]
        # of the greyscale ramp so the lightest sample is still visible on
        # white and the darkest still distinguishable from the black edge.
        log_kls = np.log10(np.where(unique_kls > 0, unique_kls, np.nan))
        lo, hi = np.nanmin(log_kls), np.nanmax(log_kls)

        def _shade_for(kl_value: float) -> tuple:
            ln = np.log10(kl_value) if kl_value > 0 else lo
            t = 0.0 if hi == lo else (ln - lo) / (hi - lo)
            grey = 0.85 - 0.7 * t  # 0.85 = light, 0.15 = dark
            return (grey, grey, grey)

        kl_to_color = {float(kl): _shade_for(float(kl)) for kl in unique_kls}
        point_colors = [kl_to_color[float(kl)] for kl in kl_co_scales]
        ax.scatter(
            metrics["input_r2"],
            y_vals,
            c=point_colors,
            edgecolor="0.25",
            s=45,
            linewidth=0.6,
        )
        ax.scatter(
            metrics["input_r2"][best_inp],
            y_vals[best_inp],
            c=GOOD_COLOR,
            s=110,
            edgecolor="k",
            zorder=5,
            label="good",
        )
        ax.scatter(
            metrics["input_r2"][bad_inp],
            y_vals[bad_inp],
            c=BAD_COLOR,
            s=110,
            edgecolor="k",
            zorder=5,
            label="bad",
        )
        ax.set_xlabel("Input R$^2$")
        ax.set_ylabel(y_label)
        ax.set_title(title_letter, loc="left", fontweight="bold", fontsize=14)
        ax.set_box_aspect(1)

        # Strip the upper / right spines and trim each axis to two tick
        # labels (lo / hi) so the panels read as clean comparison plots.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        x_lo, x_hi = float(np.min(metrics["input_r2"])), float(
            np.max(metrics["input_r2"])
        )
        y_lo, y_hi = float(np.min(y_vals)), float(np.max(y_vals))
        ax.set_xticks([x_lo, x_hi])
        ax.set_yticks([y_lo, y_hi])
        ax.set_xticklabels([f"{x_lo:.2f}", f"{x_hi:.2f}"])
        ax.set_yticklabels([f"{y_lo:.2f}", f"{y_hi:.2f}"])
        ax.tick_params(axis="both", which="both", length=3, labelsize=8)

        # Discrete legend with Xe-Y entries for each KL value (sorted small→large
        # so darker dots stack downward, matching what's in the scatter).
        handles = []
        for kl in unique_kls:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=kl_to_color[float(kl)],
                    markeredgecolor="0.25",
                    markersize=7,
                    label=f"KL={_format_kl_label(float(kl))}",
                )
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=GOOD_COLOR,
                markeredgecolor="k",
                markersize=9,
                label="good",
            )
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=BAD_COLOR,
                markeredgecolor="k",
                markersize=9,
                label="bad",
            )
        )
        ax.legend(
            handles=handles,
            fontsize=7,
            loc="best",
            frameon=False,
            handletextpad=0.4,
            labelspacing=0.3,
        )

    # ---- Column 1: C (top, square) + E (bottom, larger 3D) ----
    col1 = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=outer[0, 1],
        height_ratios=[1, 1.8],
        hspace=0.25,
    )
    ax_C = fig.add_subplot(col1[0, 0])
    _scatter_with_kl_legend(ax_C, np.asarray(metrics["co-bps"]), "co-BPS", "C")

    ax_E = fig.add_subplot(col1[1, 0], projection="3d")
    _plot_fps_on_ax(
        ax_E,
        payload["fps_bad"],
        payload["latents_bad"],
        q_thresh=Q_THRESH_BAD,
        scale=1.5,
    )
    _clean_3d(ax_E)
    ax_E.view_init(elev=VIEW_BAD[0], azim=VIEW_BAD[1])
    ax_E.set_title(
        "E - bad model FPs", loc="left", fontweight="bold", fontsize=12, color=BAD_COLOR
    )

    # ---- Column 2: D (top, square) + F (bottom, larger 3D) ----
    col2 = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=1,
        subplot_spec=outer[0, 2],
        height_ratios=[1, 1.8],
        hspace=0.25,
    )
    ax_D = fig.add_subplot(col2[0, 0])
    _scatter_with_kl_legend(
        ax_D, np.asarray(metrics["cycle_con"]), "Cycle consistency", "D"
    )

    ax_F = fig.add_subplot(col2[1, 0], projection="3d")
    _plot_fps_on_ax(
        ax_F,
        payload["fps_best"],
        payload["latents_best"],
        q_thresh=Q_THRESH_GOOD,
        scale=1.5,
    )
    _clean_3d(ax_F)
    ax_F.view_init(elev=VIEW_GOOD[0], azim=VIEW_GOOD[1])
    ax_F.set_title(
        "F - good model FPs",
        loc="left",
        fontweight="bold",
        fontsize=12,
        color=GOOD_COLOR,
    )

    pdf_path = output_dir / "figure6_AF_combined.pdf"
    png_path = output_dir / "figure6_AF_combined.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return pdf_path


fig_path = make_figure(payload, args.trial, args.output_dir)

if _in_notebook():
    plt.show()
