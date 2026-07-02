"""Generate the DSA figure (panels A-G).

Layout — 3 columns:
  Column 1: A) Scatter of State R2 vs Rate R2 for data-trained models fit to
     3BFF. Color encodes latent size (manuscript palette); marker encodes model
     family (NODE = circle, GRU = triangle).
  Column 2 (NODE): B) DSA cross-comparison matrix (TT + NODE across latent
     sizes); C) DSA-to-TT vs latent size; D) NODE8 inferred fixed points +
     latent trajectories.
  Column 3 (GRU): E/F/G — same three rows for the GRU sweep.

DSA is a dynamical *dissimilarity*: the matrix diagonal is 0 and lower
TT-distance means the model's dynamics more faithfully match the task-trained
system. Set ``GRU_SWEEP_DIR`` to the GRU sweep folder name; while it is None the
GRU column renders labeled placeholders.

Designed to run cell-by-cell in the VSCode/PyCharm interactive window (each
``# %%`` block is a cell) or end-to-end from the command line. Expensive work
(metrics, DSA fits, fixed-point optimization) is cached to ``*.cache.pkl`` next
to this file; pass ``--force`` to invalidate every cache, or the per-stage
``--force-*`` flags to refresh just one.
"""

# %%
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# Destination for the manuscript build: main.tex includes this exact path as the
# supplementary DSA figure (\includegraphics{figs/FigureS_DSA_NODE_GRU.pdf}), so
# the main panel is written here directly — no manual copy step.
MANUSCRIPT_FIG_PATH = REPO_ROOT / "manuscript" / "figs" / "FigureS_DSA_NODE_GRU.pdf"
for path in (REPO_ROOT, REPO_ROOT / "libs" / "DSA", REPO_ROOT / "libs" / "lfads-jslds"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dotenv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
# Save text as embedded (TrueType) glyphs rather than vector outlines, so the
# Arial text in the PDF/PS/SVG stays real, selectable font rather than paths.
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# Manuscript-wide font hierarchy for the DSA figure: 12 pt bold panel letters,
# 6 pt floor for the smallest elements (axis/tick labels, legends, colorbar),
# 8 pt for column titles and row labels in between.
PANEL_LETTER_PT = 12
HEADER_PT = 8
SMALL_PT = 6
plt.rcParams.update(
    {
        "font.size": SMALL_PT,
        "axes.titlesize": HEADER_PT,
        "axes.labelsize": SMALL_PT,
        "xtick.labelsize": SMALL_PT,
        "ytick.labelsize": SMALL_PT,
        "legend.fontsize": SMALL_PT,
    }
)

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

dotenv.load_dotenv(dotenv.find_dotenv())


# ----------------------------------------------------------------------------
# Configuration — edit these when running cell-by-cell.
# ----------------------------------------------------------------------------
# Root holding the task-trained model folders.
TRAINED_MODELS_ROOT = (
    Path(os.environ.get("HOME_DIR", str(REPO_ROOT)))
    / "content"
    / "trained_models"
    / "task-trained"
)
TT_DIR = "tt_3bff"  # task-trained reference (loaded directly, holds model.pkl)

# NODE dimensionality sweep (relative to TRAINED_MODELS_ROOT / TT_DIR). Each
# run folder is ``prefix=..._latent_size=N_..._seed=S`` with model.pkl +
# datamodule.pkl inside.
NODE_SWEEP_DIR = "20260520_NBFF_SAE_NODE_DimSweep"

# GRU dimensionality sweep — PLACEHOLDER. Set this to the GRU sweep folder name
# (same layout as the NODE sweep) once those models finish training. While it
# is None, panels A/C and the GRU64 FP subplot render labeled placeholders.
# An older GRU sweep exists at "20250124_NBFF_GRU_Sweep" if you want to wire one
# in for testing, but the manuscript GRU models are still training.
GRU_SWEEP_DIR = "20260527_NBFF_GRU_RNN_Sweep"

# Latent sizes to include in the DSA matrices (B/C) and the per-size selections.
LATENT_SIZES = [2, 3, 5, 8, 16, 32, 64]
# Seed used to pick the single representative model per latent size (B/C/D).
REP_SEED = 0
# Seeds included in the Panel-A scatter cloud (all available by default).
SCATTER_SEEDS = [0, 1, 2, 3, 4]

# Inferred fixed points are found for every DD model (all latent sizes) and laid
# out as a grid: one row per family (NODE, GRU), one column per latent size.
FP_SIZES = list(LATENT_SIZES)
FP_FAMILIES = ["NODE", "GRU"]
# Built from FP_FAMILIES × FP_SIZES as (label, family, latent_size).
FP_MODELS = [(f"{fam}{size}", fam, size) for fam in FP_FAMILIES for size in FP_SIZES]
# Shared 3D camera angle (elev, azim) for every FP panel — mirrors Figure 6's
# view_init so the landscapes are aligned and visually comparable.
FP_VIEW = (30, 30)
# FP cleanup for the visualization (applied in the model's latent space, both
# thresholds are fractions of the trajectory cloud's RMS radius):
#  - drop FPs whose nearest inferred-trajectory point is farther than this
#    (spurious fixed points the optimizer found off the data manifold),
FP_TRAJ_DIST_FRAC = 0.80
#  - merge FPs that lie within this of each other into a single point.
FP_MERGE_DIST_FRAC = 0.02
# Align every FP/trajectory panel into a common view by linearly projecting each
# model's latents onto the true 3-bit task state (the TT model's `controlled`
# output, corners at +/-1). All models then share the same (bit1, bit2, bit3)
# axes, so the fixed FP_VIEW camera shows the same orientation. Falls back to
# per-model PCA if the target can't be built. Set False to use raw PCA.
FP_ALIGN_TO_STATE = True
# Fixed-point marker colors — match Figure 6 (matplotlib default green/red).
FP_STABLE_COLOR = "g"
FP_UNSTABLE_COLOR = "r"

# DSA fit parameters (mirror Comparison.compare_dynamics_DSA defaults).
DSA_N_DELAYS = 20
# Rank of the reduced DMD operator. Must be <= the smallest delay-embedded
# dimension (n_delays * min_latent_dim) so every system yields an operator of
# the SAME size — DSA compares operators directly and this build does not
# zero-pad mismatched shapes. With n_delays=20 and the smallest latent dim 2,
# the floor is 40, so 30 leaves margin while staying expressive.
DSA_RANK = 30
DSA_DELAY_INTERVAL = 1
DSA_ITERS = 1500
DSA_LR = 0.005
DSA_PERCENT_DATA = 1.0  # fraction of val trials per model fed to DSA
DSA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed-point finder parameters (mirror Fig 6).
FPS_SEED = 0
FPS_N_INITS = 1024
Q_THRESH = 1e-4
NUM_TRAJ = 10

# Marker per model family for Panel A.
FAMILY_MARKER = {"NODE": "o", "GRU": "^"}

LATENT_CMAP = "turbo"
LATENT_COLOR_DIMS = np.array([2, 3, 5, 8, 16, 32, 64])
LATENT_CMAP_TRIM = (0.10, 0.95)
# Resolve the palette exactly like Figure 5 (REPO_ROOT/latent_palette.json) so
# the latent-size colors match it byte-for-byte. That file is currently absent,
# so both Figure 5 and this script fall back to sampling LATENT_CMAP ("turbo")
# over LATENT_COLOR_DIMS with LATENT_CMAP_TRIM — and if it is ever added, both
# pick it up identically. (Do NOT use a local FigDSA palette: the old copy held
# viridis values that did not match Figure 5's rendered turbo colors.)
LATENT_PALETTE_PATH = REPO_ROOT / "latent_palette.json"
_LATENT_PALETTE_CACHE: dict[int, tuple] | None = None


# %%
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    parser.add_argument(
        "--scatter-cache",
        type=Path,
        default=here / "make_figure_dsa.scatter.pkl",
        help="Cache for Panel A state/rate R2 scatter data.",
    )
    parser.add_argument(
        "--dsa-cache",
        type=Path,
        default=here / "make_figure_dsa.dsa.pkl",
        help="Cache for Panel B/C DSA matrices.",
    )
    parser.add_argument(
        "--fps-cache",
        type=Path,
        default=here / "make_figure_dsa.fps.pkl",
        help="Cache for Panel D fixed-point payloads (keyed per model).",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute ALL caches.",
    )
    parser.add_argument(
        "--force-scatter", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--force-dsa", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--force-fps", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dpi", type=int, default=200)
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
force_scatter = args.force or args.force_scatter
force_dsa = args.force or args.force_dsa
force_fps = args.force or args.force_fps


# %%
# ----------------------------------------------------------------------------
# Manuscript latent-size color palette (matches Figure 5).
# ----------------------------------------------------------------------------
def _load_latent_palette() -> dict[int, tuple] | None:
    global _LATENT_PALETTE_CACHE
    if _LATENT_PALETTE_CACHE is not None:
        return _LATENT_PALETTE_CACHE
    if not LATENT_PALETTE_PATH.exists():
        return None
    try:
        raw = json.loads(LATENT_PALETTE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    _LATENT_PALETTE_CACHE = {int(k): to_rgba(v) for k, v in raw.items()}
    return _LATENT_PALETTE_CACHE


def latent_dim_color(latent_size: int) -> tuple:
    """Manuscript color for ``latent_size`` (log2-interp between palette keys)."""
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
                rgba = (1 - t) * np.array(
                    palette[int(keys[lo])], dtype=float
                ) + t * np.array(palette[int(keys[hi])], dtype=float)
    else:
        cmap = plt.get_cmap(LATENT_CMAP)
        log_dims = np.log2(LATENT_COLOR_DIMS)
        pos = (np.log2(latent_size) - log_dims.min()) / (
            log_dims.max() - log_dims.min()
        )
        pos = LATENT_CMAP_TRIM[0] + pos * (LATENT_CMAP_TRIM[1] - LATENT_CMAP_TRIM[0])
        rgba = np.array(cmap(pos))
    return tuple(rgba)


# %%
# ----------------------------------------------------------------------------
# Sweep-folder lookup.
# ----------------------------------------------------------------------------
def sweep_path(sweep_dir: str | None) -> Path | None:
    if sweep_dir is None:
        return None
    return TRAINED_MODELS_ROOT / TT_DIR / sweep_dir


def find_run_folder(sweep_dir: str | None, latent_size: int, seed: int) -> Path | None:
    """Locate the ``latent_size=N ... seed=S`` run folder with model.pkl inside."""
    root = sweep_path(sweep_dir)
    if root is None or not root.exists():
        return None
    matches = [
        Path(p)
        for p in glob.glob(str(root / f"*latent_size={latent_size}_*seed={seed}"))
        if (Path(p) / "model.pkl").exists() and (Path(p) / "datamodule.pkl").exists()
    ]

    # Guard against substring collisions (e.g. latent_size=2 vs 32) via exact
    # token match on the parsed fields.
    def _exact(p: Path) -> bool:
        name = p.name
        return f"latent_size={latent_size}_" in name and name.endswith(f"seed={seed}")

    matches = [p for p in matches if _exact(p)]
    return matches[0] if matches else None


def load_node(latent_size: int, seed: int) -> Analysis_DD | None:
    folder = find_run_folder(NODE_SWEEP_DIR, latent_size, seed)
    if folder is None:
        return None
    return Analysis_DD.create(
        run_name=f"NODE{latent_size}", filepath=f"{folder}/", model_type="SAE"
    )


def load_gru(latent_size: int, seed: int) -> Analysis_DD | None:
    folder = find_run_folder(GRU_SWEEP_DIR, latent_size, seed)
    if folder is None:
        return None
    return Analysis_DD.create(
        run_name=f"GRU{latent_size}", filepath=f"{folder}/", model_type="SAE"
    )


def load_tt() -> Analysis_TT:
    return Analysis_TT(run_name="TT", filepath=f"{TRAINED_MODELS_ROOT / TT_DIR}/")


# %%
# ----------------------------------------------------------------------------
# Panel A — State R2 vs Rate R2 scatter.
# ----------------------------------------------------------------------------
def _scatter_points_for_family(family: str, loader, an_tt: Analysis_TT) -> list[dict]:
    points: list[dict] = []
    for latent_size in LATENT_SIZES:
        for seed in SCATTER_SEEDS:
            analysis = loader(latent_size, seed)
            if analysis is None:
                continue
            comparison = Comparison(comparison_tag=f"FigDSA_scatter_{family}")
            comparison.load_analysis(an_tt, reference_analysis=True, group="TT")
            comparison.load_analysis(analysis, group=f"{family}{latent_size}")
            m = comparison.compute_metrics(
                metric_dict_list={"state_r2": {}, "rate_r2": {}}
            )
            if not m["state_r2"]:
                continue
            points.append(
                {
                    "family": family,
                    "latent_size": latent_size,
                    "seed": seed,
                    "state_r2": float(np.asarray(m["state_r2"][0]).reshape(-1)[0]),
                    "rate_r2": float(np.asarray(m["rate_r2"][0]).reshape(-1)[0]),
                }
            )
            print(
                f"  [{family}{latent_size} seed{seed}] "
                f"state_r2={points[-1]['state_r2']:.3f} "
                f"rate_r2={points[-1]['rate_r2']:.3f}",
                flush=True,
            )
            del comparison, analysis
            gc_collect()
    return points


def gc_collect():
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_scatter_payload(force: bool, cache_path: Path) -> dict:
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        print(
            f"Loaded scatter cache from {cache_path} "
            f"({len(payload['points'])} points)"
        )
        return payload

    print("Scatter cache miss — computing state/rate R2 per model...")
    an_tt = load_tt()
    points: list[dict] = []
    print("NODE models:")
    points += _scatter_points_for_family("NODE", load_node, an_tt)
    if GRU_SWEEP_DIR is not None:
        print("GRU models:")
        points += _scatter_points_for_family("GRU", load_gru, an_tt)
    else:
        print("GRU sweep not set — Panel A will show NODE only (GRU placeholder).")

    payload = {"points": points, "gru_available": GRU_SWEEP_DIR is not None}
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved scatter cache to {cache_path}")
    return payload


# %%
# ----------------------------------------------------------------------------
# Panels B/C — DSA matrices.
# ----------------------------------------------------------------------------
def _dsa_matrix(analyses: list, labels: list[str]) -> np.ndarray:
    """Fit DSA over a list of analyses; return the pairwise similarity matrix."""
    from DSA import DSA

    latent_list = []
    for analysis in analyses:
        latents = analysis.get_latents(phase="val").detach().cpu().numpy()
        n_keep = max(1, int(DSA_PERCENT_DATA * latents.shape[0]))
        latent_list.append(latents[:n_keep])
    dsa = DSA(
        latent_list,
        n_delays=DSA_N_DELAYS,
        rank=DSA_RANK,
        delay_interval=DSA_DELAY_INTERVAL,
        verbose=True,
        device=DSA_DEVICE,
        iters=DSA_ITERS,
        lr=DSA_LR,
    )
    similarities = np.asarray(dsa.fit_score())
    print(f"  DSA matrix {similarities.shape} for labels {labels}")
    return similarities


def _build_one_dsa(family: str, loader) -> dict | None:
    """TT + one representative model per latent size -> DSA matrix + labels."""
    an_tt = load_tt()
    analyses = [an_tt]
    labels = ["TT"]
    for latent_size in LATENT_SIZES:
        analysis = loader(latent_size, REP_SEED)
        if analysis is None:
            print(f"  {family}{latent_size} (seed {REP_SEED}) not found — skipping.")
            continue
        analyses.append(analysis)
        labels.append(f"{family}{latent_size}")
    if len(analyses) < 2:
        return None
    matrix = _dsa_matrix(analyses, labels)
    return {"matrix": matrix, "labels": labels, "family": family}


def build_dsa_payload(force: bool, cache_path: Path) -> dict:
    payload: dict = {}
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        print(f"Loaded DSA cache from {cache_path} (keys: {list(payload)})")

    if "NODE" not in payload:
        print("Computing NODE DSA matrix (Panel B)...")
        t0 = time.perf_counter()
        node = _build_one_dsa("NODE", load_node)
        if node is not None:
            payload["NODE"] = node
        print(f"  NODE DSA done in {time.perf_counter() - t0:.1f}s")
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)

    if GRU_SWEEP_DIR is not None and "GRU" not in payload:
        print("Computing GRU DSA matrix (Panel C)...")
        t0 = time.perf_counter()
        gru = _build_one_dsa("GRU", load_gru)
        if gru is not None:
            payload["GRU"] = gru
        print(f"  GRU DSA done in {time.perf_counter() - t0:.1f}s")
        with cache_path.open("wb") as f:
            pickle.dump(payload, f)

    return payload


# %%
# ----------------------------------------------------------------------------
# Panel D — fixed points + latent trajectories.
# ----------------------------------------------------------------------------
def _compute_fps_for_model(
    label: str, family: str, latent_size: int | None
) -> dict | None:
    if family == "TT":
        an = load_tt()
        fps = an.compute_FPs(
            n_inits=FPS_N_INITS,
            seed=FPS_SEED,
            device="cpu",
            compute_jacobians=True,
        )
        lats = an.get_latents_noiseless(phase="val").detach().cpu().numpy()
        return {"fps": fps, "latents": lats, "label": label}

    loader = load_node if family == "NODE" else load_gru
    an = loader(latent_size, REP_SEED)
    if an is None:
        return None
    # SAE/NODE models take the task's external inputs (the 3BFF flip pulses).
    # Find fixed points with the input held fully at ZERO (no drive), rather
    # than at the mean input as Fig 6 did. The zero-input vector is sized to the
    # model's input channels.
    _, inputs = an.get_model_inputs(phase="val")
    zero_in = torch.zeros(inputs.shape[-1], dtype=torch.float32)
    fps = an.compute_FPs(
        inputs=zero_in,
        n_inits=FPS_N_INITS,
        seed=FPS_SEED,
        device="cpu",
        compute_jacobians=True,
    )
    _, lats = an.get_model_outputs(phase="val")
    return {"fps": fps, "latents": lats.detach().cpu().numpy(), "label": label}


def build_fps_payload(force: bool, cache_path: Path) -> dict:
    cache: dict[str, dict] = {}
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
        print(f"Loaded FPs cache from {cache_path} (models: {list(cache)})")

    for label, family, latent_size in FP_MODELS:
        if label in cache:
            continue
        if family == "GRU" and GRU_SWEEP_DIR is None:
            continue  # placeholder
        print(f"Finding fixed points for {label}...")
        t0 = time.perf_counter()
        result = _compute_fps_for_model(label, family, latent_size)
        if result is not None:
            cache[label] = result
            print(f"  {label} FPs done in {time.perf_counter() - t0:.1f}s")
        with cache_path.open("wb") as f:
            pickle.dump(cache, f)
    return cache


# %%
# ----------------------------------------------------------------------------
# Plot helpers.
# ----------------------------------------------------------------------------
def _clean_3d(ax):
    # Hide the panes outright (they were the white wireframe box showing on top
    # of the tinted panel background). Keep the axis lines present but fully
    # transparent — set_visible(False) on a 3D axis line makes get_tightbbox
    # return empty and crashes savefig(bbox_inches="tight") when a title is set.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


_ALIGN_TARGET_CACHE = None
_ALIGN_TARGET_DONE = False


def _get_align_target():
    """Return the true 3-bit task state for the val set as a flat (N, 3) array,
    used as a common projection target so every model's FP panel shares one
    orientation. Cached; returns None (per-model PCA fallback) on any failure."""
    global _ALIGN_TARGET_CACHE, _ALIGN_TARGET_DONE
    if _ALIGN_TARGET_DONE:
        return _ALIGN_TARGET_CACHE
    target = None
    try:
        tt = load_tt()
        out = tt.get_model_outputs(phase="val")
        controlled = out.get("controlled") if isinstance(out, dict) else None
        if controlled is not None:
            arr = controlled.detach().cpu().numpy()
            target = arr.reshape(-1, arr.shape[-1])
            print(
                f"[FP align] aligning FP panels to true {target.shape[-1]}-bit "
                f"task state ({target.shape[0]} timepoints)"
            )
    except Exception as exc:  # noqa: BLE001 — alignment is best-effort
        print(
            f"[FP align] could not build alignment target ({exc}); "
            "falling back to per-model PCA"
        )
        target = None
    _ALIGN_TARGET_CACHE = target
    _ALIGN_TARGET_DONE = True
    return target


def _filter_fps_near_traj(x_star, stability, traj_points, dist_frac):
    """Drop FPs whose nearest trajectory point is farther than ``dist_frac`` of
    the trajectory cloud's RMS radius (removes optimizer-found points off the
    data manifold). Distances are computed in the full latent space."""
    if not len(x_star):
        return x_star, stability
    center = traj_points.mean(axis=0)
    rms_radius = np.sqrt(np.mean(np.sum((traj_points - center) ** 2, axis=1)))
    thresh = dist_frac * rms_radius
    # Nearest-trajectory distance per FP (loop over the small set of FPs).
    nearest = np.array(
        [np.min(np.linalg.norm(traj_points - x, axis=1)) for x in x_star]
    )
    keep = nearest <= thresh
    return x_star[keep], stability[keep]


def _merge_close_fps(x_star, stability, merge_dist):
    """Greedily merge FPs within ``merge_dist`` of a cluster seed into a single
    point (centroid); the merged point is stable if a majority of its members
    are. Keeps the visualization from stacking many near-identical markers."""
    if not len(x_star):
        return x_star, stability
    clusters: list[list[int]] = []
    for i, x in enumerate(x_star):
        for c in clusters:
            if np.linalg.norm(x - x_star[c[0]]) <= merge_dist:
                c.append(i)
                break
        else:
            clusters.append([i])
    merged_x = np.array([x_star[c].mean(axis=0) for c in clusters])
    merged_stab = np.array([stability[c].mean() >= 0.5 for c in clusters])
    return merged_x, merged_stab


def _plot_fps_on_ax(ax, fps, latents, q_thresh=Q_THRESH, n_traj=NUM_TRAJ, scale=1.0):
    x_star = fps.xstar
    q_star = fps.qstar
    keep = q_star < q_thresh
    x_star = x_star[keep]
    stability = fps.is_stable[keep]

    # --- Clean up the FP set in the full latent space (before PCA) ---
    lats_full = latents.reshape(-1, latents.shape[-1])
    # Subsample trajectory points for the nearest-distance test (speed).
    if lats_full.shape[0] > 4000:
        idx = np.linspace(0, lats_full.shape[0] - 1, 4000).astype(int)
        traj_points = lats_full[idx]
    else:
        traj_points = lats_full
    center = traj_points.mean(axis=0)
    rms_radius = np.sqrt(np.mean(np.sum((traj_points - center) ** 2, axis=1)))
    x_star, stability = _filter_fps_near_traj(
        x_star, stability, traj_points, FP_TRAJ_DIST_FRAC
    )
    x_star, stability = _merge_close_fps(
        x_star, stability, FP_MERGE_DIST_FRAC * rms_radius
    )

    def _pad3(arr):
        if arr.shape[-1] >= 3:
            return arr
        pad = np.zeros((*arr.shape[:-1], 3 - arr.shape[-1]), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=-1)

    # Project to 3D for plotting. Preferred: a shared embedding — linearly map
    # this model's latents onto the true 3-bit task state so every panel lands
    # in the same (bit1, bit2, bit3) frame. Fall back to per-model PCA when the
    # alignment target is unavailable or its trial count doesn't match.
    align_target = _get_align_target() if FP_ALIGN_TO_STATE else None
    if align_target is not None and align_target.shape[0] == lats_full.shape[0]:
        reg = LinearRegression().fit(lats_full, align_target)
        lats_proj = _pad3(reg.predict(lats_full))
        x_pca = _pad3(reg.predict(x_star)) if len(x_star) else np.empty((0, 3))
    else:
        # Low-dimensional models (e.g. the 2D NODE) have fewer than 3 latent
        # dims, so cap PCA components and zero-pad the projection out to 3.
        n_comp = min(3, latents.shape[-1])
        pca = PCA(n_components=n_comp)
        lats_proj = _pad3(pca.fit_transform(lats_full))
        x_pca = _pad3(pca.transform(x_star)) if len(x_star) else np.empty((0, 3))
    lats_pca = lats_proj.reshape(latents.shape[0], latents.shape[1], 3)
    # Trajectories: thin, semi-transparent black so the FP markers read on top.
    for i in range(min(n_traj, lats_pca.shape[0])):
        ax.plot(
            lats_pca[i, :, 0],
            lats_pca[i, :, 1],
            lats_pca[i, :, 2],
            color="k",
            linewidth=0.45 * scale,
            alpha=0.45,
            zorder=1,
        )
    if len(x_pca):
        stable = x_pca[stability]
        unstable = x_pca[~stability]
        # Stable FPs: large green dots; no depth-shading so they stay vivid
        # against the dense black trajectories (Figure-6 green/red, no halo).
        m = scale * FP_MARKER_SCALE  # FP markers shrunk to ~30% (linear)
        ax.scatter(
            stable[:, 0],
            stable[:, 1],
            stable[:, 2],
            c=FP_STABLE_COLOR,
            marker="o",
            s=35 * m * m,
            depthshade=False,
            zorder=10,
        )
        ax.scatter(
            unstable[:, 0],
            unstable[:, 1],
            unstable[:, 2],
            c=FP_UNSTABLE_COLOR,
            marker="x",
            s=27.5 * m * m,
            linewidths=1.4 * m,
            depthshade=False,
            zorder=11,
        )


def _placeholder(ax, text):
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
        color="0.4",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linestyle((0, (3, 3)))
        spine.set_edgecolor("0.6")


def _plot_dsa_matrix(ax, matrix, labels, title, vmin=None, vmax=None):
    # Stored order is [TT, <smallest latent> .. <largest latent>]. Reverse only
    # the rows so the displayed matrix has TT in the bottom row and latent size
    # increasing upward, while columns stay in stored order (TT at left, latent
    # size increasing rightward). Self-comparisons (zeros) run along the
    # anti-diagonal. ``vmin``/``vmax`` set a shared color scale across matrices.
    matrix = np.asarray(matrix)[::-1, :]
    row_labels = list(labels)[::-1]
    col_labels = list(labels)
    cmap = plt.get_cmap("viridis_r")
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=SMALL_PT)
    ax.set_yticklabels(row_labels, fontsize=SMALL_PT)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Small centered sub-label (e.g. "NODE"/"GRU"); the bold panel letter is
    # drawn separately in make_figure so all letters align on one vertical line.
    if title:
        ax.set_title(title, loc="center", fontweight="bold", fontsize=HEADER_PT)

    # Box the TT row's DSA(TT, model) cells — these are the values plotted in
    # panel C. Exclude the TT-vs-TT self-comparison (the leftmost, TT, column).
    n = len(row_labels)
    tt_row = row_labels.index("TT") if "TT" in row_labels else n - 1
    tt_col = col_labels.index("TT") if "TT" in col_labels else 0
    ax.add_patch(
        Rectangle(
            (tt_col + 0.5, tt_row - 0.5),
            n - 1,
            1.0,
            fill=False,
            edgecolor="crimson",
            linewidth=2.5,
            zorder=5,
            clip_on=False,
        )
    )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=SMALL_PT)
    cbar.set_label("DSA dissimilarity", fontsize=SMALL_PT)


def _dsa_to_tt(entry: dict) -> tuple[list[int], list[float]]:
    """From a DSA payload entry, pull DSA(TT, model) vs latent size.

    The matrix' first row/col is the TT reference, so ``matrix[0, k]`` is the
    dissimilarity between TT and the model in ``labels[k]``. Returns the latent
    sizes and their TT-dissimilarities, sorted by latent size.
    """
    matrix = np.asarray(entry["matrix"])
    labels = entry["labels"]
    tt_idx = labels.index("TT")
    sizes, vals = [], []
    for k, label in enumerate(labels):
        if k == tt_idx:
            continue
        size = int("".join(ch for ch in label if ch.isdigit()))
        sizes.append(size)
        vals.append(float(matrix[tt_idx, k]))
    order = np.argsort(sizes)
    return [sizes[i] for i in order], [vals[i] for i in order]


FAMILY_LINESTYLE = {"NODE": "-", "GRU": "--"}


def _plot_dsa_vs_latent_on_ax(
    ax,
    dsa: dict,
    families=("NODE", "GRU"),
    title="DSA to TT vs latent size",
    show_legend=True,
    ylim=None,
):
    """Line+marker plot of DSA(TT, model) against latent size, per model family.

    ``families`` restricts which model families are drawn so a per-column panel
    can show NODE-only or GRU-only.
    """
    plotted = False
    for family in families:
        if family not in dsa:
            continue
        marker = FAMILY_MARKER.get(family, "o")
        linestyle = FAMILY_LINESTYLE.get(family, "-")
        sizes, vals = _dsa_to_tt(dsa[family])
        if not sizes:
            continue
        ax.plot(sizes, vals, linestyle=linestyle, color="0.5", linewidth=1.2, zorder=1)
        for s, v in zip(sizes, vals):
            ax.scatter(
                s,
                v,
                color=latent_dim_color(s),
                marker=marker,
                s=24,
                edgecolor="0.25",
                linewidth=0.5,
                zorder=3,
            )
        plotted = True
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent size", fontsize=SMALL_PT)
    ax.set_ylabel("DSA dissimilarity to TT", fontsize=SMALL_PT)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=PANEL_LETTER_PT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    if plotted:
        all_sizes = sorted(
            {s for fam in families if fam in dsa for s in _dsa_to_tt(dsa[fam])[0]}
        )
        ax.set_xticks(all_sizes)
        ax.set_xticklabels([str(s) for s in all_sizes], fontsize=SMALL_PT)
        ax.minorticks_off()
    fam_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKER.get(f, "o"),
            linestyle=FAMILY_LINESTYLE.get(f, "-"),
            color="0.5",
            markerfacecolor="0.6",
            markeredgecolor="0.2",
            markersize=5,
            label=f,
        )
        for f in families
        if f in dsa
    ]
    if show_legend and fam_handles:
        # Park the Model key in the empty space above the curve's dip, centered
        # over latent_size = 16 (x in data coords, y in axes fraction).
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.legend(
            handles=fam_handles,
            fontsize=SMALL_PT,
            loc="upper center",
            bbox_to_anchor=(16, 1.0),
            bbox_transform=trans,
            frameon=False,
            title="Model",
            title_fontsize=SMALL_PT,
        )


def _dsa_to_tt_ylim(dsa: dict, families=("NODE", "GRU"), pad=0.05):
    """Shared y-limits across DSA-vs-latent panels so columns are comparable."""
    vals = [v for fam in families if fam in dsa for v in _dsa_to_tt(dsa[fam])[1]]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1.0
    return (lo - pad * span, hi + pad * span)


def make_dsa_vs_latent_figure(dsa: dict, output_dir: Path, dpi: int) -> Path | None:
    """Standalone version of the DSA-to-TT vs latent-size plot."""
    if "NODE" not in dsa and "GRU" not in dsa:
        print("No DSA matrices available — skipping DSA-vs-latent figure.")
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_dsa_vs_latent_on_ax(ax, dsa, title="DSA to TT vs latent size")
    fig.tight_layout()
    pdf_path = output_dir / "figureDSA_dsa_to_tt_vs_latent.pdf"
    png_path = output_dir / "figureDSA_dsa_to_tt_vs_latent.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return pdf_path


# %%
# ----------------------------------------------------------------------------
# Assemble the figure.
# ----------------------------------------------------------------------------
def _draw_scatter(ax, scatter):
    """Panel A — State R2 vs Rate R2, color=latent size, marker=model family."""
    points = scatter["points"]
    seen_sizes, seen_families = set(), set()
    for p in points:
        ax.scatter(
            p["state_r2"],
            p["rate_r2"],
            color=latent_dim_color(p["latent_size"]),
            marker=FAMILY_MARKER.get(p["family"], "o"),
            s=55,
            edgecolor="0.25",
            linewidth=0.5,
            zorder=3,
        )
        seen_sizes.add(p["latent_size"])
        seen_families.add(p["family"])
    ax.set_xlabel("State R$^2$", fontsize=10)
    ax.set_ylabel("Rate R$^2$", fontsize=10)
    ax.set_title("A", loc="left", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=latent_dim_color(s),
            markeredgecolor="0.25",
            label=f"D={s}",
        )
        for s in sorted(seen_sizes)
    ]
    fam_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKER[f],
            linestyle="",
            markersize=8,
            markerfacecolor="0.6",
            markeredgecolor="0.2",
            label=f,
        )
        for f in ["NODE", "GRU"]
        if f in seen_families
    ]
    leg1 = ax.legend(
        handles=size_handles,
        fontsize=7,
        loc="center left",
        frameon=False,
        title="Latent size",
        title_fontsize=8,
        ncol=2,
        handletextpad=0.2,
        labelspacing=0.3,
        bbox_to_anchor=(0.0, 0.4),
    )
    ax.add_artist(leg1)
    if fam_handles:
        ax.legend(
            handles=fam_handles,
            fontsize=8,
            loc="lower left",
            frameon=False,
            title="Model",
            title_fontsize=8,
        )
    if not scatter.get("gru_available", False):
        ax.text(
            0.02,
            0.02,
            "GRU: awaiting trained path",
            transform=ax.transAxes,
            fontsize=6,
            color="0.5",
            ha="left",
            va="bottom",
        )


def _draw_one_fp(
    fig,
    subspec,
    fps,
    fp_label,
    col_title=None,
    row_label=None,
    panel_letter=None,
    scale=1.0,
):
    """Draw one inferred-FP landscape into ``subspec``, aligned like Figure 6.

    Every cell shares the same camera angle (``FP_VIEW``) and a cubic box aspect
    so the landscapes can be compared at a glance. ``col_title`` (latent size)
    is shown above the top row, ``row_label`` (family) is the vertical label at
    the left of each row, and ``panel_letter`` is a horizontal bold letter drawn
    top-left (the figure-panel tag). ``scale`` enlarges the trajectory/marker
    sizes for bigger cells.
    """
    if fp_label in fps:
        ax = fig.add_subplot(subspec, projection="3d")
        _plot_fps_on_ax(ax, fps[fp_label]["fps"], fps[fp_label]["latents"], scale=scale)
        _clean_3d(ax)
        ax.view_init(elev=FP_VIEW[0], azim=FP_VIEW[1])
        # zoom > 1 enlarges the data inside the cube so it fills more of the
        # cell, letting the FP row be shorter without shrinking the blobs.
        ax.set_box_aspect((1, 1, 1), zoom=FP_BOX_ZOOM)
    else:
        ax = fig.add_subplot(subspec)
        _placeholder(ax, f"{fp_label}\nawaiting path")
    text2d = getattr(ax, "text2D", ax.text)
    if col_title is not None:
        ax.set_title(col_title, fontsize=HEADER_PT, fontweight="bold")
    if row_label is not None:
        # Vertical family label at the left of the row.
        text2d(
            -0.12,
            0.5,
            row_label,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=HEADER_PT,
            fontweight="bold",
        )
    if panel_letter is not None:
        # Horizontal panel tag, top-left (kept upright, unlike the row label).
        text2d(
            -0.12,
            1.08,
            panel_letter,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=PANEL_LETTER_PT,
            fontweight="bold",
        )
    return ax


def _dsa_vs_r2_points(dsa: dict, scatter: dict) -> list[dict]:
    """Pair each model's DSA-to-TT (from the matrix' TT row, seed REP_SEED) with
    its State/Rate R2 (from the scatter cache, same seed)."""
    r2_by = {
        (p["family"], p["latent_size"], p["seed"]): p for p in scatter.get("points", [])
    }
    points: list[dict] = []
    for family in ("NODE", "GRU"):
        if family not in dsa:
            continue
        sizes, vals = _dsa_to_tt(dsa[family])
        for size, dsa_val in zip(sizes, vals):
            rec = r2_by.get((family, size, REP_SEED))
            if rec is None:
                continue
            points.append(
                {
                    "family": family,
                    "latent_size": size,
                    "dsa": dsa_val,
                    "state_r2": rec["state_r2"],
                    "rate_r2": rec["rate_r2"],
                }
            )
    return points


def _scatter_r2_vs_dsa(ax, points, r2_key, xlabel, title, show_legend=False):
    """Scatter of a reconstruction metric (x) vs DSA dissimilarity to TT (y).
    Color = latent size (Fig 5 palette); marker = model family."""
    seen_sizes, seen_families = set(), set()
    for p in points:
        ax.scatter(
            p[r2_key],
            p["dsa"],
            color=latent_dim_color(p["latent_size"]),
            marker=FAMILY_MARKER.get(p["family"], "o"),
            s=70,
            edgecolor="0.25",
            linewidth=0.5,
            zorder=3,
        )
        seen_sizes.add(p["latent_size"])
        seen_families.add(p["family"])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("DSA dissimilarity to TT", fontsize=10)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)
    if show_legend:
        size_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=latent_dim_color(s),
                markeredgecolor="0.25",
                label=f"D={s}",
            )
            for s in sorted(seen_sizes)
        ]
        fam_handles = [
            Line2D(
                [0],
                [0],
                marker=FAMILY_MARKER[f],
                linestyle="",
                markersize=8,
                markerfacecolor="0.6",
                markeredgecolor="0.2",
                label=f,
            )
            for f in ("NODE", "GRU")
            if f in seen_families
        ]
        leg1 = ax.legend(
            handles=size_handles,
            fontsize=6.5,
            loc="upper right",
            frameon=False,
            title="Latent size",
            title_fontsize=7,
            ncol=2,
            handletextpad=0.2,
            labelspacing=0.25,
        )
        ax.add_artist(leg1)
        if fam_handles:
            ax.legend(
                handles=fam_handles,
                fontsize=7.5,
                loc="lower right",
                frameon=False,
                title="Model",
                title_fontsize=7,
            )


# Latent sizes whose inferred-FP landscapes go in the main figure's bottom row.
MAIN_FP_SIZES = [8, 64]

# How much to enlarge the 3D FP data inside its cubic cell (panel C). >1 fills
# the otherwise-empty top/bottom margins so the FP row can be made shorter.
FP_BOX_ZOOM = 1.22

# Linear shrink factor for the FP markers (stable dots / unstable x's) in panel
# C — they read much too large at full size, so draw them at ~30%.
FP_MARKER_SCALE = 0.3


def make_figure(dsa, fps, output_dir: Path, dpi: int) -> Path:
    # Render the whole figure (canvas + fonts) at 60% of its native size so it
    # shrinks uniformly without changing any layout proportions. Helpers read
    # the module-level point sizes, so scale those + the matching rcParams for
    # the duration of this function and restore them afterwards (other figures
    # keep their native sizes).
    s = 0.6
    global PANEL_LETTER_PT, HEADER_PT, SMALL_PT
    _saved_pt = (PANEL_LETTER_PT, HEADER_PT, SMALL_PT)
    _rc_keys = (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    )
    _saved_rc = {k: plt.rcParams[k] for k in _rc_keys}
    PANEL_LETTER_PT, HEADER_PT, SMALL_PT = (v * s for v in _saved_pt)
    plt.rcParams.update(
        {
            "font.size": SMALL_PT,
            "axes.titlesize": HEADER_PT,
            "axes.labelsize": SMALL_PT,
            "xtick.labelsize": SMALL_PT,
            "ytick.labelsize": SMALL_PT,
            "legend.fontsize": SMALL_PT,
        }
    )
    try:
        return _make_figure_body(dsa, fps, output_dir, dpi, s)
    finally:
        PANEL_LETTER_PT, HEADER_PT, SMALL_PT = _saved_pt
        plt.rcParams.update(_saved_rc)


def _make_figure_body(dsa, fps, output_dir: Path, dpi: int, s: float) -> Path:
    # 6.5 in wide (manuscript text width). Height trimmed from the original
    # 12:17 proportions: the FP row (panel C) is the tallest and carries a lot
    # of empty margin, so it gets a smaller share and the rows sit closer.
    # `s` scales the whole canvas.
    fig = plt.figure(figsize=(6.5 * s, 6.5 * 13.5 / 12 * s))
    outer = gridspec.GridSpec(
        3,
        1,
        height_ratios=[0.85, 0.55, 1.45],
        hspace=0.3,
        figure=fig,
    )

    # ---- Row 1: DSA matrices (NODE, GRU) side by side, shared color scale ----
    mats = [np.asarray(dsa[f]["matrix"]) for f in ("NODE", "GRU") if f in dsa]
    dsa_vmin = float(min(m.min() for m in mats)) if mats else None
    dsa_vmax = float(max(m.max() for m in mats)) if mats else None

    mat_row = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        wspace=0.18,
    )
    ax_nmat = fig.add_subplot(mat_row[0, 0])
    if "NODE" in dsa:
        _plot_dsa_matrix(
            ax_nmat,
            dsa["NODE"]["matrix"],
            dsa["NODE"]["labels"],
            "NODE",
            vmin=dsa_vmin,
            vmax=dsa_vmax,
        )
    else:
        _placeholder(ax_nmat, "DSA: TT vs NODE")
        ax_nmat.set_title("NODE", loc="center", fontweight="bold", fontsize=HEADER_PT)

    ax_gmat = fig.add_subplot(mat_row[0, 1])
    if "GRU" in dsa:
        _plot_dsa_matrix(
            ax_gmat,
            dsa["GRU"]["matrix"],
            dsa["GRU"]["labels"],
            "GRU",
            vmin=dsa_vmin,
            vmax=dsa_vmax,
        )
    else:
        _placeholder(ax_gmat, "DSA: TT vs GRU\nawaiting trained path")
        ax_gmat.set_title("GRU", loc="center", fontweight="bold", fontsize=HEADER_PT)

    # ---- Row 2: combined NODE + GRU DSA-to-TT vs latent size. Slightly inset,
    # with more padding on the left than the right. ----
    line_row = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=outer[1],
        width_ratios=[0.9, 7.4, 0.55],
    )
    ax_line = fig.add_subplot(line_row[0, 1])
    line_ylim = _dsa_to_tt_ylim(dsa, families=("NODE", "GRU"))
    _plot_dsa_vs_latent_on_ax(
        ax_line,
        dsa,
        families=("NODE", "GRU"),
        title="",
        ylim=line_ylim,
    )

    # Colored highlight bands in C at the callout sizes — each FP column below
    # carries the SAME tint as its band (see the background fill set per cell).
    HL_ALPHA = 0.22
    for size in MAIN_FP_SIZES:
        ax_line.axvspan(
            size / 1.18,
            size * 1.18,
            lw=0,
            zorder=0,
            color=latent_dim_color(size),
            alpha=HL_ALPHA,
        )

    # ---- Row 3: inferred-FP landscapes — rows = model family (NODE, GRU),
    # columns = latent size (8D, 64D). State-aligned + filtered/merged. ----
    # One clean tinted block per latent-size column (matches the bands in C),
    # drawn in a backing axes so the cells can sit close (negative spacing)
    # without the per-column tints overlapping/darkening one another.
    ncol = len(MAIN_FP_SIZES)
    bg_ax = fig.add_subplot(outer[2])
    bg_ax.set_axis_off()
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.set_zorder(0)
    # Inset the tint blocks from the left so they clear the NODE/GRU row labels,
    # and leave a gap between columns so the two blocks don't touch.
    BG_LEFT_PAD = 0.07
    BG_GAP = 0.05
    avail = 1.0 - BG_LEFT_PAD
    block_w = (avail - BG_GAP * (ncol - 1)) / ncol
    for c, size in enumerate(MAIN_FP_SIZES):
        bg = list(to_rgba(latent_dim_color(size)))
        bg[3] = HL_ALPHA
        x0 = BG_LEFT_PAD + c * (block_w + BG_GAP)
        bg_ax.add_patch(
            Rectangle((x0, 0), block_w, 1, facecolor=tuple(bg), edgecolor="none")
        )

    fp_grid = gridspec.GridSpecFromSubplotSpec(
        len(FP_FAMILIES),
        ncol,
        subplot_spec=outer[2],
        wspace=-0.08,
        hspace=-0.12,
    )
    for r, family in enumerate(FP_FAMILIES):
        for c, size in enumerate(MAIN_FP_SIZES):
            ax_fp = _draw_one_fp(
                fig,
                fp_grid[r, c],
                fps,
                f"{family}{size}",
                col_title=(f"D={size}" if r == 0 else None),
                row_label=(family if c == 0 else None),
                panel_letter=None,
                scale=1.4,
            )
            ax_fp.patch.set_visible(False)  # show the column block behind
            ax_fp.set_zorder(1)

    # Stability legend (proxy handles), tucked just beneath the colored blocks.
    stab_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=10 * s,
            markerfacecolor=FP_STABLE_COLOR,
            markeredgecolor=FP_STABLE_COLOR,
            label="Stable fixed point",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            markersize=10 * s,
            markeredgecolor=FP_UNSTABLE_COLOR,
            markeredgewidth=2,
            label="Unstable fixed point",
        ),
        Line2D([0], [0], color="k", lw=1.6, alpha=0.6, label="Latent trajectory"),
    ]
    fp_bottom = outer[2].get_position(fig).y0
    fig.legend(
        handles=stab_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, fp_bottom + 0.005),
        fontsize=SMALL_PT,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    # Panel letters (A: matrices, B: DSA-vs-latent, C: fixed points) drawn at a
    # single figure-x so all three sit on the same vertical line, each at the top
    # of its row. The outer GridSpec rows all span the full width, so their left
    # edge (x0) is shared.
    letter_x = outer[0].get_position(fig).x0
    for spec, letter in ((outer[0], "A"), (outer[1], "B"), (outer[2], "C")):
        fig.text(
            letter_x,
            spec.get_position(fig).y1,
            letter,
            fontsize=PANEL_LETTER_PT,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    pdf_path = output_dir / "figureDSA_ABCD.pdf"
    png_path = output_dir / "figureDSA_ABCD.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    # Also drop the PDF straight into the manuscript so the build picks it up
    # without a manual copy — main.tex includes it as MANUSCRIPT_FIG_PATH's name.
    if MANUSCRIPT_FIG_PATH.parent.is_dir():
        fig.savefig(MANUSCRIPT_FIG_PATH, bbox_inches="tight")
        print(f"Wrote {MANUSCRIPT_FIG_PATH}")
    else:
        print(f"Skipped manuscript copy (missing {MANUSCRIPT_FIG_PATH.parent})")
    return pdf_path


def make_fp_figure(fps, output_dir: Path, dpi: int) -> Path:
    """Standalone fixed-point figure: a grid of inferred-FP landscapes,
    rows = model family (NODE, GRU), columns = latent size."""
    n_sizes = len(FP_SIZES)
    fig = plt.figure(figsize=(max(14, 2.1 * n_sizes), 5.5))
    grid = gridspec.GridSpec(
        len(FP_FAMILIES),
        n_sizes,
        hspace=0.15,
        wspace=0.05,
        figure=fig,
    )
    for r, family in enumerate(FP_FAMILIES):
        for c, size in enumerate(FP_SIZES):
            _draw_one_fp(
                fig,
                grid[r, c],
                fps,
                f"{family}{size}",
                col_title=(f"D={size}" if r == 0 else None),
                row_label=(family if c == 0 else None),
            )
    fig.suptitle(
        "Inferred fixed points + latent trajectories "
        "(rows: model family, columns: latent size)",
        fontsize=12,
        fontweight="bold",
        x=0.5,
        y=1.02,
    )
    pdf_path = output_dir / "figureDSA_fixed_points.pdf"
    png_path = output_dir / "figureDSA_fixed_points.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    return pdf_path


# %%
dsa_payload = build_dsa_payload(force_dsa, args.dsa_cache)
fps_payload = build_fps_payload(force_fps, args.fps_cache)

# Main figure: DSA matrices + DSA-vs-latent line + 8D/64D FP landscapes
# (rows = NODE/GRU, columns = latent size).
fig_path = make_figure(dsa_payload, fps_payload, args.output_dir, args.dpi)
# Full FP grid (all latent sizes) in its own figure.
fp_fig_path = make_fp_figure(fps_payload, args.output_dir, args.dpi)
dsa_latent_path = make_dsa_vs_latent_figure(dsa_payload, args.output_dir, args.dpi)

if _in_notebook():
    plt.show()
