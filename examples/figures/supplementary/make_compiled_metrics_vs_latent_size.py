#!/usr/bin/env python
# %%
"""Supplementary: six TT-vs-NODE metrics by latent size, one figure per task.

Mirrors the six-panel latent-size summary from the
``Figure1_*_TTGRULyap.ipynb`` exploratory notebooks. Emits **one supplementary
figure per dataset** (NBFF, MultiTask, RandomTarget, PCM, CDM) under
``examples/figures/supplementary/outputs/compiled_metrics/`` as
``FigureS_<Task>_metrics_vs_latent_size.{pdf,png}``.

Per-task metric caches live next to the script as
``compiled_metrics_cache/<Task>_compiled_metrics.pkl``. Each task is handled
independently: a complete cache is reused, a missing cache is computed from
scratch, and a cache that predates Lyapunov (has the other metrics but no
``lyapunov`` column) is topped up with just the Lyapunov pass. Pass
``--recompute`` to ignore caches entirely.

Pass ``--move-to-manuscript`` to also copy each task's figure files into
``manuscript/figs/`` (where ``main.tex`` includes them by name) and rebuild the
manuscript PDF with ``latexmk``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

# Headless/CLI runs use the non-interactive Agg backend, but when running
# cell-by-cell in VS Code/Jupyter we leave the interactive backend in place so
# figures render inline for inspection (see plot_task_metrics' show path).
if "ipykernel" not in sys.modules:
    mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "compiled_metrics"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "compiled_metrics_cache"
DEFAULT_EXTS = ("pdf", "png")

# Manuscript integration: --move-to-manuscript copies each task's generated
# figure files into MANUSCRIPT_FIGS_DIR (main.tex includes them by the exact
# output_stem name, e.g. figs/FigureS_NBFF_metrics_vs_latent_size.png) and then
# rebuilds the PDF with latexmk (config in manuscript/.latexmkrc).
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
MANUSCRIPT_FIGS_DIR = MANUSCRIPT_DIR / "figs"

# Manuscript styling for the supplementary panels.
FIG_WIDTH_IN = 6.5
FIG_HEIGHT_IN = 4.0
AXIS_LABEL_PT = 6
TICK_LABEL_PT = 6
COLUMN_HEADER_PT = 8
PANEL_LETTER_PT = 12
MARKER_SIZE = 10
# Default y-limits for the reconstruction R2 panels when a task doesn't override
# them via TaskConfig.rate_r2_ylim / state_r2_ylim.
R2_DEFAULT_YLIM = (0.0, 1.1)
# Horizontal position (axes fraction) of the y-axis labels. Nudged toward the
# axis so each label nestles in the empty vertical gap between the top and
# bottom y-tick labels. More negative = further left.
YLABEL_X = -0.06
COLUMN_HEADERS = ("Reconstruction", "Simplicity", "Dyn. Systems ID")
PANEL_LETTERS = ("A", "B", "C")

# Manuscript-wide latent-size color mapping, kept in sync with
# examples/figures/Fig5Metrics/make_figure5_reconstruction_simplicity.py so a
# given latent size renders in the same color across every figure — even when
# the sweep covers different latent sizes from task to task.
LATENT_CMAP = "turbo"
LATENT_COLOR_DIMS = np.array([2, 3, 5, 8, 16, 32, 64])
LATENT_PASTEL_MIX = 0.0
LATENT_CMAP_TRIM = (0.10, 0.95)
LATENT_PALETTE_PATH = REPO_ROOT / "latent_palette.json"
_LATENT_PALETTE_CACHE: dict[int, tuple[float, float, float, float]] | None = None
DEFAULT_SWEEP_DIRS: dict[str, str | None] = {
    # Sweep dirs are only read on a cache miss; tasks with a complete cache reuse
    # it and never touch these paths. Set to None to fall back to auto-discovery
    # under the task's TT folder.
    "NBFF": "20260520_NBFF_SAE_NODE_DimSweep",
    "MultiTask": "20250814_MultiTask_NODE_Sweep",
    "RandomTarget": "20250821_RandomTarget_LatentSweep_Smaller",
    "PCM": "20260528_PCM_LFADS_RateBiasInit_DimSweep_NewScale",
    "CDM": "20260529_CDM_NODE_DimSweep_Final",
}
LATENT_RE = re.compile(r"latent_size=(\d+)")
# LFADS dim sweeps encode the generator (latent) size as ``gen_dim=N`` in their
# run-folder names rather than ``latent_size=`` / ``NODE``.
GEN_DIM_RE = re.compile(r"gen_dim=(\d+)")
# Group labels are ``<model_label><size>`` (e.g. NODE16, LFADS16); parse the
# trailing size back out so sort_metrics_by_latent works for either model type.
MODEL_LABEL_RE = re.compile(r"(?:NODE|LFADS|GRU|RNN)(\d+)")

METRIC_DICT_LIST = {
    "state_r2": {},
    "rate_r2": {},
    "cycle_con": {"variance_threshold": 0.01},
    "co-bps": {},
    "wasserstein_geometry": {"distance_metric": "cdf_wasserstein"},
    "lyapunov": {},
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# %%
# Edit these values when running the file cell-by-cell with VS Code/Jupyter.
CELL_TASKS = ["NBFF", "MultiTask", "RandomTarget", "PCM", "CDM"]
CELL_HOME_DIR: str | None = None
CELL_TRAINED_MODELS_ROOT: str | None = None
CELL_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CELL_CACHE_DIR = DEFAULT_CACHE_DIR
CELL_SWEEP_DIRS = DEFAULT_SWEEP_DIRS.copy()
CELL_RECOMPUTE = False
CELL_SKIP_REFERENCE_LYAP = False
# Reuse a cache's already-computed metrics and add only the Lyapunov column
# (plus the TT reference line). Set True to extend a cache built without it.
CELL_AUGMENT_LYAPUNOV = False
CELL_STRICT = False
# Copy each task's figures into manuscript/figs/ and rebuild with latexmk.
CELL_MOVE_TO_MANUSCRIPT = True


# %%
@dataclass(frozen=True)
class TaskConfig:
    key: str
    label: str
    tt_dir: str
    sweep_dir: str | None
    comparison_tag: str
    output_stem: str
    co_bps_ylim: tuple[float, float] | None = None
    # Per-task y-limits for the unit-capped panels (reconstruction R2 and
    # cycle-consistency). None falls back to the shared default (R2_DEFAULT_YLIM).
    rate_r2_ylim: tuple[float, float] | None = None
    state_r2_ylim: tuple[float, float] | None = None
    cycle_con_ylim: tuple[float, float] | None = None
    compute_lyapunov: bool = True
    lyapunov_kwargs: dict | None = None
    # Data-trained model family for this sweep. Passed straight through to
    # Analysis_DD.create; supported values are "SAE", "LFADS", and "External".
    model_type: str = "SAE"
    # Prefix used to name/group each loaded run (e.g. "NODE16", "LFADS16").
    # Defaults to "NODE" for the SAE NODE sweeps; set to "LFADS" for LFADS.
    model_label: str = "NODE"


TASKS = {
    "nbff": TaskConfig(
        key="nbff",
        label="NBFF",
        tt_dir="tt_3bff",
        sweep_dir=DEFAULT_SWEEP_DIRS["NBFF"],
        comparison_tag="Figure1NBFF_TTGRU_Sweep",
        output_stem="FigureS_NBFF_metrics_vs_latent_size",
        co_bps_ylim=(0.05, 0.135),
    ),
    "multitask": TaskConfig(
        key="multitask",
        label="MultiTask",
        tt_dir="tt_MultiTask",
        sweep_dir=DEFAULT_SWEEP_DIRS["MultiTask"],
        comparison_tag="MultiTask_DDNODE_Sweep",
        output_stem="FigureS_MultiTask_metrics_vs_latent_size",
        co_bps_ylim=(0.0, 0.15),
        compute_lyapunov=True,
        lyapunov_kwargs={"subset_frac": 0.1},
    ),
    "randomtarget": TaskConfig(
        key="randomtarget",
        label="RandomTarget",
        tt_dir="tt_RandomTarget",
        sweep_dir=DEFAULT_SWEEP_DIRS["RandomTarget"],
        comparison_tag="Figure1RandomTarget_TTGRU_Sweep",
        output_stem="FigureS_RandomTarget_metrics_vs_latent_size",
        co_bps_ylim=(0.0, 0.05),
    ),
    "pcm": TaskConfig(
        key="pcm",
        label="PCM",
        tt_dir="tt_PhaseCodedMemory",
        sweep_dir=DEFAULT_SWEEP_DIRS["PCM"],
        comparison_tag="Figure1PCM_TTGRU_Sweep",
        output_stem="FigureS_PCM_metrics_vs_latent_size",
        co_bps_ylim=(0.0, 0.02),
        compute_lyapunov=True,
        lyapunov_kwargs={"subset_frac": 0.1},
        model_type="LFADS",
        model_label="LFADS",
    ),
    "cdm": TaskConfig(
        key="cdm",
        label="CDM",
        tt_dir="tt_ChaoticDelayedMatching",
        sweep_dir=DEFAULT_SWEEP_DIRS["CDM"],
        comparison_tag="Figure1CDM_TTGRU_Sweep",
        output_stem="FigureS_CDM_metrics_vs_latent_size",
        co_bps_ylim=(0.0, 1.0),
        compute_lyapunov=True,
        lyapunov_kwargs={"subset_frac": 0.1},
    ),
}

ALIASES = {
    "3bff": "nbff",
    "nbff": "nbff",
    "multitask": "multitask",
    "multi": "multitask",
    "randomtarget": "randomtarget",
    "random": "randomtarget",
    "rt": "randomtarget",
    "pcm": "pcm",
    "phasecodedmemory": "pcm",
    "phase_coded_memory": "pcm",
    "cdm": "cdm",
    "chaoticdelayedmatching": "cdm",
    "chaotic_delayed_matching": "cdm",
}


# %%
def normalize_task_name(name: str) -> str:
    key = name.strip().replace("-", "_").replace(" ", "_").lower()
    compact = key.replace("_", "")
    return ALIASES.get(key, ALIASES.get(compact, compact))


def analysis_path(path: Path) -> str:
    return f"{path}/"


def extract_latent_size(text: str) -> int | None:
    for pattern in (LATENT_RE, GEN_DIM_RE, MODEL_LABEL_RE):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def discover_home_dir(cli_home_dir: str | None) -> Path:
    if cli_home_dir:
        return Path(cli_home_dir).expanduser().resolve()
    env_home_dir = os.environ.get("HOME_DIR")
    if env_home_dir:
        return Path(env_home_dir).expanduser().resolve()
    return REPO_ROOT


def parse_sweep_overrides(
    items: list[str] | dict[str, str | Path | None],
) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    if isinstance(items, dict):
        iterator = items.items()
    else:
        iterator = []
        for item in items:
            if "=" not in item:
                raise ValueError(f"Expected --sweep-dir TASK=PATH, got {item!r}")
            iterator.append(item.split("=", 1))

    for task_name, sweep_dir in iterator:
        if sweep_dir in (None, ""):
            continue
        task_key = normalize_task_name(str(task_name))
        if task_key not in TASKS:
            raise ValueError(f"Unknown task in --sweep-dir: {task_name!r}")
        overrides[task_key] = Path(sweep_dir).expanduser()
    return overrides


def latent_run_folders(sweep_path: Path) -> list[tuple[int, Path]]:
    folders: list[tuple[int, Path]] = []
    if not sweep_path.exists():
        return folders
    for child in sweep_path.iterdir():
        if not child.is_dir():
            continue
        latent_size = extract_latent_size(child.name)
        if latent_size is None:
            continue
        if (child / "model.pkl").exists() and (child / "datamodule.pkl").exists():
            folders.append((latent_size, child))
    return sorted(folders, key=lambda item: (item[0], item[1].name))


def discover_sweep_path(tt_path: Path) -> Path | None:
    candidates: list[tuple[int, float, Path]] = []
    if not tt_path.exists():
        return None
    for child in tt_path.iterdir():
        if not child.is_dir():
            continue
        runs = latent_run_folders(child)
        if runs:
            candidates.append((len(runs), child.stat().st_mtime, child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def resolve_sweep_path(
    task: TaskConfig, trained_models_root: Path, overrides: dict[str, Path]
) -> Path | None:
    tt_path = trained_models_root / task.tt_dir
    override = overrides.get(task.key)
    if override is not None:
        return override if override.is_absolute() else tt_path / override
    if task.sweep_dir is not None:
        return tt_path / task.sweep_dir
    return discover_sweep_path(tt_path)


# %%
def to_float(value: Any) -> float:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    if array.dtype == object and array.shape:
        return to_float(array.reshape(-1)[0])
    return float(array.reshape(-1)[0])


def first_lyapunov(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        return to_float(value[0])
    try:
        import torch

        if torch.is_tensor(value):
            return to_float(value.reshape(-1)[0])
    except ImportError:
        pass
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def sort_metrics_by_latent(metrics: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    run_names = list(metrics["run_name"])
    latent_sizes = [extract_latent_size(str(name)) for name in run_names]
    if any(size is None for size in latent_sizes):
        missing = [name for name, size in zip(run_names, latent_sizes) if size is None]
        raise ValueError(f"Could not parse latent size from run names: {missing}")

    order = sorted(
        range(len(run_names)), key=lambda idx: (latent_sizes[idx], run_names[idx])
    )
    sorted_metrics: dict[str, Any] = {}
    for key, values in metrics.items():
        if isinstance(values, (list, tuple)) and len(values) == len(order):
            sorted_metrics[key] = [values[idx] for idx in order]
        else:
            sorted_metrics[key] = values
    return sorted_metrics, [int(latent_sizes[idx]) for idx in order]


# %%
def _compute_per_run_metrics(
    task: TaskConfig,
    an_tt: Any,
    runs: list[tuple[int, Path]],
    task_metric_dict_list: dict[str, Any],
) -> dict[str, Any]:
    """Run ``Comparison.compute_metrics`` for each sweep run, one model at a time.

    Returns a metrics dict of parallel lists keyed by metric name (plus
    ``run_name``/``group``), in sweep-run order. Shared by the full-compute path
    and the Lyapunov-only augment path, so both produce row-aligned results.
    """
    from ctd.comparison.analysis.dd.dd import Analysis_DD
    from ctd.comparison.comparison import Comparison

    try:
        import torch
    except ImportError:
        torch = None

    metrics: dict[str, Any] = {"run_name": [], "group": []}
    for metric in task_metric_dict_list.keys():
        metrics[metric] = []

    print(
        f"[{task.label}] starting per-model compute_metrics over {len(runs)} models "
        f"× {len(task_metric_dict_list)} metrics (one DD model at a time)"
    )
    t0 = time.perf_counter()
    n_runs = len(runs)
    for run_idx, (latent_size, run_path) in enumerate(runs, start=1):
        t_run = time.perf_counter()
        elapsed = time.perf_counter() - t0
        if run_idx > 1:
            avg = elapsed / (run_idx - 1)
            eta = avg * (n_runs - run_idx + 1)
            eta_str = f" — avg {avg:.1f}s/run, ETA {eta / 60:.1f} min"
        else:
            eta_str = ""
        run_label = f"{task.model_label}{latent_size}"
        print(
            f"[{task.label}] === Run {run_idx} of {n_runs}: {run_label}"
            f"{eta_str} ==="
        )
        analysis_node = Analysis_DD.create(
            run_name=run_label,
            filepath=analysis_path(run_path),
            model_type=task.model_type,
        )
        print(
            f"[{task.label}]   loaded {run_label} ({run_path.name}) "
            f"in {time.perf_counter() - t_run:.1f}s"
        )

        comparison = Comparison(comparison_tag=task.comparison_tag)
        comparison.load_analysis(an_tt, reference_analysis=True, group="TT")
        comparison.load_analysis(analysis_node, group=run_label)

        t_compute = time.perf_counter()
        single_metrics = comparison.compute_metrics(
            metric_dict_list=task_metric_dict_list
        )
        print(
            f"[{task.label}]   compute_metrics({run_label}) in "
            f"{time.perf_counter() - t_compute:.1f}s"
        )

        for key, values in single_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].extend(values)

        del comparison
        del analysis_node
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[{task.label}]   cleared {run_label} from memory "
            f"(run total: {time.perf_counter() - t_run:.1f}s)"
        )
    print(
        f"[{task.label}] per-model compute_metrics done in "
        f"{time.perf_counter() - t0:.1f}s"
    )
    return metrics


def _augment_cache_with_lyapunov(
    task: TaskConfig,
    trained_models_root: Path,
    sweep_path: Path,
    cache_path: Path,
    cached: dict[str, Any],
    recompute: bool,
    skip_reference_lyap: bool,
) -> dict[str, Any]:
    """Add the Lyapunov metric to an existing cache without recomputing the rest.

    Reuses every already-cached metric (state/rate R², cycle-consistency,
    co-bps, geometry) and computes only the per-run Lyapunov exponent plus the
    TT reference line, then merges and re-saves. Useful after an initial run
    that skipped Lyapunov: the expensive non-Lyapunov metrics are kept as-is.
    """
    if not task.compute_lyapunov:
        raise ValueError(
            f"[{task.label}] augment-lyapunov requested but this task has "
            "compute_lyapunov=False; nothing to add."
        )

    cached_metrics = cached["metrics"]
    if cached_metrics.get("lyapunov") and not recompute:
        print(
            f"[{task.label}] cache already has Lyapunov for "
            f"{len(cached_metrics['lyapunov'])} runs — nothing to do "
            "(pass --recompute to force)."
        )
        return cached

    print(f"[{task.label}] augmenting existing cache with Lyapunov only")
    t_task = time.perf_counter()

    from ctd.comparison.analysis.tt.tt import Analysis_TT

    tt_path = trained_models_root / task.tt_dir
    if not (tt_path / "model.pkl").exists():
        raise FileNotFoundError(
            f"Missing TT model for {task.label}: {tt_path / 'model.pkl'}"
        )

    runs = latent_run_folders(sweep_path)
    if not runs:
        raise FileNotFoundError(
            f"No latent-size {task.model_label} runs found in {sweep_path}"
        )

    t0 = time.perf_counter()
    an_tt = Analysis_TT(run_name="TT", filepath=analysis_path(tt_path))
    print(f"[{task.label}] loaded TT analysis in {time.perf_counter() - t0:.1f}s")

    lyap_kwargs = dict(task.lyapunov_kwargs or {})

    lex_tt_mean0: float | None = cached.get("lex_tt_mean0")
    if not skip_reference_lyap:
        print(f"[{task.label}] computing TT reference Lyapunov exponent")
        t0 = time.perf_counter()
        lex_tt_mean, _ = an_tt.compute_lyapunov_exp(phase="val", **lyap_kwargs)
        lex_tt_mean0 = to_float(lex_tt_mean[0])
        print(f"[{task.label}] TT Lyapunov done in {time.perf_counter() - t0:.1f}s")

    lyap_metrics = _compute_per_run_metrics(
        task, an_tt, runs, {"lyapunov": lyap_kwargs}
    )

    # The augment loop applies the same NaN-skip logic as the original compute,
    # so its run order must line up with the cached rows before we splice the
    # Lyapunov column in. Refuse to merge if they diverge rather than silently
    # mis-aligning a run's Lyapunov value with another run's other metrics.
    if lyap_metrics["run_name"] != cached_metrics.get("run_name"):
        raise RuntimeError(
            f"[{task.label}] Lyapunov run order does not match the cached run "
            "order; cannot safely merge. Re-run with --recompute to rebuild the "
            "cache from scratch.\n"
            f"  cached: {cached_metrics.get('run_name')}\n"
            f"  lyap:   {lyap_metrics['run_name']}"
        )

    cached_metrics["lyapunov"] = lyap_metrics["lyapunov"]
    cached["lex_tt_mean0"] = lex_tt_mean0
    cached.setdefault("metric_dict_list", {})["lyapunov"] = lyap_kwargs

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(cached, handle)
    print(
        f"[{task.label}] Lyapunov augment done in "
        f"{time.perf_counter() - t_task:.1f}s — updated {cache_path}"
    )
    return cached


def load_or_compute_metrics(
    task: TaskConfig,
    trained_models_root: Path,
    sweep_path: Path,
    cache_path: Path,
    recompute: bool,
    skip_reference_lyap: bool,
    augment_lyapunov: bool = False,
) -> dict[str, Any]:
    cached: dict[str, Any] | None = None
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)

    if augment_lyapunov:
        if cached is None:
            raise FileNotFoundError(
                f"[{task.label}] --augment-lyapunov needs an existing cache to "
                f"extend, but none was found at {cache_path}. Run once without it "
                "first (or use --recompute to build everything)."
            )
        return _augment_cache_with_lyapunov(
            task=task,
            trained_models_root=trained_models_root,
            sweep_path=sweep_path,
            cache_path=cache_path,
            cached=cached,
            recompute=recompute,
            skip_reference_lyap=skip_reference_lyap,
        )

    if cached is not None and not recompute:
        # Reuse the cache only when it already holds every metric this task
        # wants. A cache built before Lyapunov was enabled has the (expensive)
        # other metrics but no "lyapunov" column; rather than silently dropping
        # the Lyapunov panel — or redoing everything — fill in just that column
        # via the augment path, reusing what's already cached.
        missing_lyap = task.compute_lyapunov and not cached["metrics"].get("lyapunov")
        if not missing_lyap:
            print(f"[{task.label}] cache hit: {cache_path}")
            return cached
        print(
            f"[{task.label}] cache present but missing Lyapunov — reusing cached "
            "metrics and computing only the Lyapunov column"
        )
        return _augment_cache_with_lyapunov(
            task=task,
            trained_models_root=trained_models_root,
            sweep_path=sweep_path,
            cache_path=cache_path,
            cached=cached,
            recompute=recompute,
            skip_reference_lyap=skip_reference_lyap,
        )

    print(f"[{task.label}] no cache — computing metrics from scratch")
    t_task = time.perf_counter()

    from ctd.comparison.analysis.tt.tt import Analysis_TT

    tt_path = trained_models_root / task.tt_dir
    if not (tt_path / "model.pkl").exists():
        raise FileNotFoundError(
            f"Missing TT model for {task.label}: {tt_path / 'model.pkl'}"
        )

    runs = latent_run_folders(sweep_path)
    if not runs:
        raise FileNotFoundError(
            f"No latent-size {task.model_label} runs found in {sweep_path}"
        )
    print(f"[{task.label}] found {len(runs)} {task.model_label} runs in {sweep_path}")

    t0 = time.perf_counter()
    an_tt = Analysis_TT(run_name="TT", filepath=analysis_path(tt_path))
    print(f"[{task.label}] loaded TT analysis in {time.perf_counter() - t0:.1f}s")

    task_metric_dict_list = dict(METRIC_DICT_LIST)
    if not task.compute_lyapunov:
        task_metric_dict_list.pop("lyapunov", None)
        print(f"[{task.label}] Lyapunov exponent disabled for this task — skipping")
    elif task.lyapunov_kwargs:
        task_metric_dict_list["lyapunov"] = dict(task.lyapunov_kwargs)
        print(f"[{task.label}] Lyapunov kwargs: {task_metric_dict_list['lyapunov']}")

    lex_tt_mean0: float | None = None
    if task.compute_lyapunov and not skip_reference_lyap:
        print(
            f"[{task.label}] computing TT reference Lyapunov exponent (for horizontal line in plot)"
        )
        t0 = time.perf_counter()
        ref_lyap_kwargs = dict(task.lyapunov_kwargs or {})
        lex_tt_mean, _ = an_tt.compute_lyapunov_exp(phase="val", **ref_lyap_kwargs)
        lex_tt_mean0 = to_float(lex_tt_mean[0])
        print(f"[{task.label}] TT Lyapunov done in {time.perf_counter() - t0:.1f}s")

    metrics = _compute_per_run_metrics(task, an_tt, runs, task_metric_dict_list)

    payload = {
        "metrics": metrics,
        "lex_tt_mean0": lex_tt_mean0,
        "task_label": task.label,
        "sweep_path": str(sweep_path),
        "metric_dict_list": task_metric_dict_list,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)
    print(
        f"[{task.label}] total elapsed: {time.perf_counter() - t_task:.1f}s — cached to {cache_path}"
    )
    return payload


# %%
def _load_latent_palette() -> dict[int, tuple[float, float, float, float]] | None:
    """Load {D: rgba} from LATENT_PALETTE_PATH; cache the result.

    Returns None if the file is missing or unreadable, in which case callers
    fall back to sampling LATENT_CMAP. Mirrors the loader in the Figure 5
    script so both pull from the same manuscript palette.
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


def metric_colors(latent_sizes: list[int]) -> list[Any]:
    return [latent_dim_color(size) for size in latent_sizes]


def configure_axis(ax: Any, latent_sizes: list[int]) -> None:
    ax.set_xlabel("Latent Size", fontsize=AXIS_LABEL_PT)
    unique_sizes = sorted(set(latent_sizes))
    # Latent sizes span a wide multiplicative range (e.g. 2..64), so a log2
    # x-scale spaces them evenly. The ticks still carry the real values
    # (3, 5, 8, 16, ...) rather than the underlying log2 positions.
    ax.set_xscale("log", base=2)
    if unique_sizes:
        lo, hi = unique_sizes[0], unique_sizes[-1]
        # Multiplicative padding so the end points aren't pinned to the edges.
        factor = 2**0.15
        ax.set_xlim([lo / factor, hi * factor])
    ax.set_xticks(unique_sizes)
    ax.set_xticklabels([str(s) for s in unique_sizes])
    # Suppress the log scale's auto minor ticks/labels between the real sizes.
    ax.minorticks_off()
    ax.tick_params(axis="both", labelsize=TICK_LABEL_PT)


def style_extreme_yticks(ax: Any, cap_at_one: bool = False) -> None:
    """Keep only the upper and lower y ticks, rounded outward to a clean decimal.

    The ticks are snapped to the nearest "nice" step at the order of magnitude of
    the visible span (e.g. an upper limit of 0.15173 becomes 0.16), expanding the
    limits outward so the data points sit inside with a little room rather than
    pinned to the axis edges.

    When ``cap_at_one`` is True (metrics whose theoretical max is 1.0, such as
    any R2 or cycle-consistency), the top tick is pinned to 1.0 while the axis
    extends slightly past it to 1.05 so points at 1.0 aren't pinned to the edge.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        ax.set_yticks([lo, hi])
        ax.tick_params(axis="both", labelsize=TICK_LABEL_PT)
        return

    exponent = int(np.floor(np.log10(span)))
    step = 10.0**exponent
    # Round the ratio before floor/ceil so float error (e.g. 0.3 / 0.1 == 2.9999...)
    # doesn't snap a clean limit down an extra step.
    lo_tick = np.floor(np.round(lo / step, 6)) * step
    decimals = max(0, -exponent)

    if cap_at_one:
        ax.set_ylim(lo_tick, 1.05)
        ax.set_yticks([lo_tick, 1.0])
        ax.set_yticklabels([f"{lo_tick:.{decimals}f}", f"{1.0:.{decimals}f}"])
        ax.tick_params(axis="both", labelsize=TICK_LABEL_PT)
        return

    hi_tick = np.ceil(np.round(hi / step, 6)) * step
    ax.set_ylim(lo_tick, hi_tick)
    ax.set_yticks([lo_tick, hi_tick])
    ax.set_yticklabels([f"{lo_tick:.{decimals}f}", f"{hi_tick:.{decimals}f}"])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_PT)


def add_column_headers(fig: Any, top_axes: list[Any]) -> None:
    """Draw a category header and panel letter above each column."""
    for ax, header, letter in zip(top_axes, COLUMN_HEADERS, PANEL_LETTERS):
        pos = ax.get_position()
        y = pos.y1 + 0.02
        fig.text(
            (pos.x0 + pos.x1) / 2,
            y,
            header,
            ha="center",
            va="bottom",
            fontsize=COLUMN_HEADER_PT,
            fontweight="bold",
        )
        fig.text(
            pos.x0,
            y,
            letter,
            ha="left",
            va="bottom",
            fontsize=PANEL_LETTER_PT,
            fontweight="bold",
        )


def plot_task_metrics(
    payload: dict[str, Any],
    task: TaskConfig,
    output_dir: Path,
    exts: tuple[str, ...] = DEFAULT_EXTS,
    show: bool = False,
) -> list[Path]:
    metrics, latent_sizes = sort_metrics_by_latent(payload["metrics"])
    colors = metric_colors(latent_sizes)

    fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    axs[0, 0].scatter(
        latent_sizes, [to_float(v) for v in metrics["rate_r2"]], c=colors, s=MARKER_SIZE
    )
    axs[0, 0].set_ylabel("Rate R2", fontsize=AXIS_LABEL_PT)
    axs[0, 0].set_ylim(task.rate_r2_ylim or R2_DEFAULT_YLIM)
    configure_axis(axs[0, 0], latent_sizes)

    axs[0, 1].scatter(
        latent_sizes,
        [to_float(v) for v in metrics["state_r2"]],
        c=colors,
        s=MARKER_SIZE,
    )
    axs[0, 1].set_ylabel("State R2", fontsize=AXIS_LABEL_PT)
    axs[0, 1].set_ylim(task.state_r2_ylim or R2_DEFAULT_YLIM)
    configure_axis(axs[0, 1], latent_sizes)

    axs[0, 2].scatter(
        latent_sizes,
        [to_float(v) for v in metrics["wasserstein_geometry"]],
        c=colors,
        s=MARKER_SIZE,
    )
    axs[0, 2].set_ylabel("Wasserstein Geometry", fontsize=AXIS_LABEL_PT)
    configure_axis(axs[0, 2], latent_sizes)

    axs[1, 0].scatter(
        latent_sizes, [to_float(v) for v in metrics["co-bps"]], c=colors, s=MARKER_SIZE
    )
    axs[1, 0].set_ylabel("Co-BPS", fontsize=AXIS_LABEL_PT)
    if task.co_bps_ylim is not None:
        axs[1, 0].set_ylim(task.co_bps_ylim)
    configure_axis(axs[1, 0], latent_sizes)

    axs[1, 1].scatter(
        latent_sizes,
        [to_float(v) for v in metrics["cycle_con"]],
        c=colors,
        s=MARKER_SIZE,
    )
    axs[1, 1].set_ylabel("Cycle Consistency", fontsize=AXIS_LABEL_PT)
    axs[1, 1].set_ylim(task.cycle_con_ylim or R2_DEFAULT_YLIM)
    configure_axis(axs[1, 1], latent_sizes)

    active_axes = [axs[0, 0], axs[0, 1], axs[0, 2], axs[1, 0], axs[1, 1]]
    # Metrics whose theoretical max is 1.0 (rate R2, state R2, cycle-consistency).
    unit_capped_axes = {axs[0, 0], axs[0, 1], axs[1, 1]}
    lyap_values = metrics.get("lyapunov") or []
    zero_lbl = tt_lbl = None
    tt_lyap: float | None = None
    if lyap_values:
        axs[1, 2].scatter(
            latent_sizes,
            [first_lyapunov(v) for v in lyap_values],
            c=colors,
            s=MARKER_SIZE,
        )
        # x in axes fraction, y in data coords — place each label just past the
        # right edge, centered on its line, so the line stays unobstructed.
        lyap_lbl_trans = axs[1, 2].get_yaxis_transform()
        # Zero line: boundary between contracting (stable) and chaotic dynamics.
        axs[1, 2].axhline(0.0, color="0.6", linestyle=":", linewidth=0.8)
        zero_lbl = axs[1, 2].text(
            1.02,
            0.0,
            r"$\lambda = 0$",
            transform=lyap_lbl_trans,
            ha="left",
            va="center",
            fontsize=TICK_LABEL_PT,
            color="0.4",
            clip_on=False,
        )
        if payload.get("lex_tt_mean0") is not None:
            tt_lyap = payload["lex_tt_mean0"]
            axs[1, 2].axhline(tt_lyap, color="black", linestyle="--", linewidth=0.8)
            tt_lbl = axs[1, 2].text(
                1.02,
                tt_lyap,
                "TT",
                transform=lyap_lbl_trans,
                ha="left",
                va="center",
                fontsize=TICK_LABEL_PT,
                color="black",
                clip_on=False,
            )
        axs[1, 2].set_ylabel("Max Lyapunov Exponent", fontsize=AXIS_LABEL_PT)
        configure_axis(axs[1, 2], latent_sizes)
        active_axes.append(axs[1, 2])
    else:
        axs[1, 2].set_axis_off()
        axs[1, 2].text(
            0.5,
            0.5,
            "Lyapunov disabled",
            ha="center",
            va="center",
            transform=axs[1, 2].transAxes,
            fontsize=TICK_LABEL_PT,
            color="gray",
        )

    fig.suptitle(
        f"{task.label} — {task.model_label} metrics by latent size",
        y=1.06,
        fontsize=COLUMN_HEADER_PT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for ax in active_axes:
        style_extreme_yticks(ax, cap_at_one=ax in unit_capped_axes)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_label_coords(YLABEL_X, 0.5)

    # The TT λ_max can sit almost on the λ=0 line (e.g. CDM, where it's ~-6e-5),
    # so their right-margin labels collide. Now that the y-limits are finalized,
    # check proximity against the final span and, if they're too close, push the
    # two labels apart vertically (the lower line's label below it, the higher
    # line's above it). The lines stay put — only the text is jittered.
    if tt_lbl is not None and zero_lbl is not None and tt_lyap is not None:
        lo, hi = axs[1, 2].get_ylim()
        span = hi - lo
        if span > 0 and abs(tt_lyap - 0.0) < 0.10 * span:
            if tt_lyap < 0.0:
                tt_lbl.set_va("top")  # TT below zero -> label below its line
                zero_lbl.set_va("bottom")  # zero above -> label above its line
            else:
                tt_lbl.set_va("bottom")
                zero_lbl.set_va("top")

    add_column_headers(fig, [axs[0, 0], axs[0, 1], axs[0, 2]])
    written: list[Path] = []
    for ext in exts:
        out_path = output_dir / f"{task.output_stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        written.append(out_path)
    # Display the figure for inspection in interactive sessions; otherwise close
    # it to free memory on headless/CLI runs.
    if show:
        plt.show()
    else:
        plt.close(fig)
    return written


# %%
def copy_to_manuscript(
    written: list[Path], figs_dir: Path = MANUSCRIPT_FIGS_DIR
) -> list[Path]:
    """Copy generated figure files into the manuscript ``figs/`` directory.

    ``main.tex`` includes these by exact filename
    (``figs/FigureS_<Task>_metrics_vs_latent_size.png``), and each task's
    ``output_stem`` already matches that name, so every file drops straight into
    place and overwrites the previously included copy. Returns the destinations
    actually written (empty if the manuscript figs directory is missing).
    """
    if not figs_dir.is_dir():
        print(f"Skipped manuscript copy (missing {figs_dir})")
        return []
    copied: list[Path] = []
    for src in written:
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


# %%
def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create six-panel compiled metric figures versus latent size."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["NBFF", "MultiTask", "RandomTarget", "PCM", "CDM"],
        help="Tasks to process. Defaults to all major datasets.",
    )
    parser.add_argument(
        "--home-dir",
        default=None,
        help="Repository/HOME_DIR root. Defaults to $HOME_DIR, then the repo root.",
    )
    parser.add_argument(
        "--trained-models-root",
        default=None,
        help="Root containing task-trained model folders. Defaults to HOME_DIR/content/trained_models/task-trained.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output PDFs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for per-task metric caches.",
    )
    parser.add_argument(
        "--sweep-dir",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Override a task sweep directory. Relative paths are resolved under that task's TT folder.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore existing metric caches and recompute metrics.",
    )
    parser.add_argument(
        "--skip-reference-lyap",
        action="store_true",
        help="Skip the TT reference Lyapunov line.",
    )
    parser.add_argument(
        "--augment-lyapunov",
        action="store_true",
        help=(
            "Reuse an existing cache's metrics and compute only the Lyapunov "
            "column (plus the TT reference line), then merge and re-save. Use "
            "this to add Lyapunov to a cache built without it, without redoing "
            "the other (slow) metrics. Combine with --recompute to force a "
            "fresh Lyapunov pass even when the cache already has one."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if a requested task cannot be processed.",
    )
    parser.add_argument(
        "--move-to-manuscript",
        action="store_true",
        help=(
            "After plotting, copy each task's generated figure files into "
            "manuscript/figs/ (overwriting the included copies) and rebuild the "
            "manuscript PDF with latexmk."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = make_arg_parser()
    if argv is None and running_in_ipykernel():
        args, _ = parser.parse_known_args(argv)
        return args
    return parser.parse_args(argv)


def running_in_ipykernel() -> bool:
    return "ipykernel" in sys.modules


def configure_paper_font(
    preferred: tuple[str, ...] = ("Arial", "Liberation Sans"),
    fallback: str = "DejaVu Sans",
) -> str:
    """Use the first installed font in ``preferred``, else fall back.

    Setting ``font.family`` to a missing name makes matplotlib emit a
    ``findfont: Font family '<name>' not found`` warning for every text element
    drawn, which floods the log. We check the registered font list once up
    front and only request a font that's actually available.

    "Liberation Sans" is metrically identical to Arial (same glyph advances),
    so it's a clean stand-in on Linux systems that lack the Microsoft font.
    """
    from matplotlib import font_manager

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    if isinstance(preferred, str):
        preferred = (preferred,)
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            mpl.rcParams["font.family"] = [name, fallback]
            return name
    print(f"[font] none of {list(preferred)} installed; using '{fallback}'.")
    mpl.rcParams["font.family"] = [fallback]
    return fallback


def run_from_namespace(args: argparse.Namespace) -> None:
    configure_paper_font()

    home_dir = discover_home_dir(args.home_dir)
    trained_models_root = (
        Path(args.trained_models_root).expanduser().resolve()
        if args.trained_models_root
        else home_dir / "content" / "trained_models" / "task-trained"
    )
    overrides = parse_sweep_overrides(args.sweep_dir)

    print(f"Using trained models root: {trained_models_root}")
    print(f"Writing figures to: {Path(args.output_dir)}")

    task_keys = [normalize_task_name(task_name) for task_name in args.tasks]
    unknown_tasks = [task_name for task_name in task_keys if task_name not in TASKS]
    if unknown_tasks:
        raise ValueError(f"Unknown task(s): {unknown_tasks}")

    move_to_manuscript = getattr(args, "move_to_manuscript", False)
    moved_any = False
    for task_key in task_keys:
        task = TASKS[task_key]
        try:
            sweep_path = resolve_sweep_path(task, trained_models_root, overrides)
            if sweep_path is None:
                raise FileNotFoundError(
                    f"No latent-size sweep found for {task.label}. "
                    f"Use --sweep-dir {task.label}=PATH to specify one."
                )

            print(f"{task.label}: using sweep {sweep_path}")
            cache_path = args.cache_dir / f"{task.label}_compiled_metrics.pkl"
            payload = load_or_compute_metrics(
                task=task,
                trained_models_root=trained_models_root,
                sweep_path=sweep_path,
                cache_path=cache_path,
                recompute=args.recompute,
                skip_reference_lyap=args.skip_reference_lyap,
                augment_lyapunov=getattr(args, "augment_lyapunov", False),
            )
            written = plot_task_metrics(
                payload, task, args.output_dir, show=running_in_ipykernel()
            )
            for out in written:
                print(f"{task.label}: saved {out}")
            if move_to_manuscript:
                copied = copy_to_manuscript(written)
                moved_any = moved_any or bool(copied)
        except Exception as exc:
            if args.strict:
                raise
            warnings.warn(f"{task.label}: skipped ({exc})")

    # Rebuild once, after all requested tasks have been copied in.
    if move_to_manuscript and moved_any:
        rebuild_manuscript()


def cell_args() -> argparse.Namespace:
    return argparse.Namespace(
        tasks=CELL_TASKS,
        home_dir=CELL_HOME_DIR,
        trained_models_root=CELL_TRAINED_MODELS_ROOT,
        output_dir=Path(CELL_OUTPUT_DIR),
        cache_dir=Path(CELL_CACHE_DIR),
        sweep_dir=CELL_SWEEP_DIRS,
        recompute=CELL_RECOMPUTE,
        skip_reference_lyap=CELL_SKIP_REFERENCE_LYAP,
        augment_lyapunov=CELL_AUGMENT_LYAPUNOV,
        strict=CELL_STRICT,
        move_to_manuscript=CELL_MOVE_TO_MANUSCRIPT,
    )


def run_from_cells() -> None:
    run_from_namespace(cell_args())


def main(argv: list[str] | None = None) -> None:
    run_from_namespace(parse_args(argv))


# %%
# Run this cell in VS Code/Jupyter after editing the CELL_* settings above.
if running_in_ipykernel():
    run_from_cells()


# %%
if __name__ == "__main__" and not running_in_ipykernel():
    main()
