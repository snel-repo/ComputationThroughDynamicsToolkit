"""Simplified PhaseCodedMemory DD figure.

One row per DD model (8D / 16D / 128D from the dimension sweep), four columns:

  1. imagesc raster of the ground-truth firing rates
  2. imagesc raster of the model-inferred (predicted) firing rates
  3. top-3 PCs of the response period in *firing-rate* space (StimA/StimB)
  4. top-3 PCs of the response period in *latent* space (StimA/StimB)

This reuses the heavy import boilerplate, path resolution, and a few plotting
helpers from ``make_phase_coded_memory_figure`` (imported as ``base``), but
computes its own (much smaller) payload and writes its own cache so it never
clobbers the main figure's cache.
"""

# %%
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Importing the sibling module installs all the optional-dependency stubs
# (dotenv / DSA / wandb / pkg_resources shims) at import time and gives us the
# shared helpers + constants below.
import make_phase_coded_memory_figure as base  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ctd.comparison.analysis.dd.dd import Analysis_DD  # noqa: E402
from ctd.comparison.analysis.tt.tt import Analysis_TT  # noqa: E402
from ctd.comparison.comparison import Comparison  # noqa: E402

as_numpy = base.as_numpy

# Optional InjLFADS 8D run; appended as an extra row when its directory exists
# (skipped silently otherwise, or via --no-ddinj8).
DEFAULT_DDINJ8_SUBPATH = (
    "20260530_PCM_InjLFADS_DimSweep_Final3/"
    "prefix=tt_PhaseCodedMemory_max_epochs=300_gen_dim=8_seed=0"
)

# Each entry is one figure row. ``subpath`` is resolved under the TT model dir.
# Entries marked "optional" only become a row when their analysis directory
# exists (and --no-<prefix> wasn't passed), so the figure still builds without them.
MODELS = (
    {"prefix": "dd8", "dim_label": "8D", "subpath": base.DEFAULT_DD8_SUBPATH},
    {"prefix": "dd", "dim_label": "16D", "subpath": base.DEFAULT_DD_SUBPATH},
    {"prefix": "dd128", "dim_label": "128D", "subpath": base.DEFAULT_DD128_SUBPATH},
)

OUTPUT_DIR = REPO_ROOT / "examples" / "figures" / "supplementary" / "outputs"
MANUSCRIPT_FIGS_DIR = REPO_ROOT / "manuscript" / "figs"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
        help="Recompute arrays instead of using the cache.",
    )
    parser.add_argument("--phase", default="val", choices=["train", "val", "all"])
    parser.add_argument("--tt-path", type=Path, default=None)
    parser.add_argument("--ddinj8-subpath", default=DEFAULT_DDINJ8_SUBPATH)
    parser.add_argument(
        "--no-ddinj8",
        action="store_true",
        help="Skip the optional InjLFADS 8D row even if its directory exists.",
    )
    parser.add_argument(
        "--trial", type=int, default=0, help="Trial index for the raster panels."
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--max-pca-trials", type=int, default=240)
    if argv is None and base._in_notebook():
        argv = []
    return parser.parse_args(argv)


# %%
def response_pca(data: np.ndarray, extras: np.ndarray) -> np.ndarray:
    """Fit a 3-component PCA on the *response period* only, then transform every
    timepoint so trajectories can be drawn from each trial's response onset.

    ``extras[trial, 1]`` is the response-period start index (post-stim onset),
    matching the convention used throughout the main figure script.
    """
    from sklearn.decomposition import PCA

    pts = [data[i, int(extras[i, 1]) :, :] for i in range(data.shape[0])]
    pca = PCA(n_components=3).fit(np.concatenate(pts, axis=0))
    flat = pca.transform(data.reshape(-1, data.shape[-1]))
    return flat.reshape(data.shape[0], data.shape[1], 3)


def compute_r2(an_tt, an_dd) -> dict[str, float]:
    """Rate R² and State R² for one DD model against the TT reference."""
    comp = Comparison(comparison_tag="PCM_simple")
    comp.load_analysis(an_tt, group="TT", reference_analysis=True)
    comp.load_analysis(an_dd, group="DD")
    metrics = comp.compute_metrics(metric_dict_list={"state_r2": {}, "rate_r2": {}})
    return {
        "rate_r2": (
            float(metrics["rate_r2"][0]) if metrics.get("rate_r2") else float("nan")
        ),
        "state_r2": (
            float(metrics["state_r2"][0]) if metrics.get("state_r2") else float("nan")
        ),
    }


def resolve_tt_path(args: argparse.Namespace) -> Path:
    """Resolve the task-trained model dir (override, HOME_DIR, or pretrained)."""
    home = base.get_home_dir()
    tt_path = (
        args.tt_path or home / "content/trained_models/task-trained/tt_PhaseCodedMemory"
    )
    if args.tt_path is None and not tt_path.exists():
        pretrained_tt = REPO_ROOT / "pretrained/PCM_NoisyGRU_Final"
        if pretrained_tt.exists():
            tt_path = pretrained_tt
    return tt_path


def dd_dir(args: argparse.Namespace, tt_path: Path, spec: dict) -> Path:
    """Directory for one MODELS entry, honoring a ``--<prefix>-subpath`` override."""
    subpath = getattr(args, f"{spec['prefix']}_subpath", None) or spec["subpath"]
    return tt_path / subpath


def should_include(args: argparse.Namespace, tt_path: Path, spec: dict) -> bool:
    """Required models are always included; optional ones only when present on
    disk and not suppressed via ``--no-<prefix>``."""
    if not spec.get("optional"):
        return True
    if getattr(args, f"no_{spec['prefix']}", False):
        return False
    return dd_dir(args, tt_path, spec).exists()


def compute_model_arrays(args, extras, an_tt, an) -> dict:
    """Inferred rates/latents response-period PCs and R² metrics for one DD row."""
    inf_rates = as_numpy(an.get_rates(phase=args.phase))
    lats = as_numpy(an.get_latents(phase=args.phase))
    r2 = compute_r2(an_tt, an)
    return {
        "inf_rates": inf_rates,
        "rates_pca": response_pca(inf_rates, extras),
        "lats_pca": response_pca(lats, extras),
        "rate_r2": r2["rate_r2"],
        "state_r2": r2["state_r2"],
    }


def load_or_compute_payload(args: argparse.Namespace) -> dict:
    tt_path = resolve_tt_path(args)
    models = [m for m in MODELS if should_include(args, tt_path, m)]

    if args.cache.exists() and not args.force:
        with args.cache.open("rb") as f:
            cached = pickle.load(f)
        required = {"true_rates", "extras", "inds_a", "inds_b", "gt"}
        missing = [m for m in models if "rate_r2" not in cached.get(m["prefix"], {})]
        if required.issubset(cached) and not missing:
            return cached
        # If the only gaps are optional rows and the base data is cached, add
        # just those rows in place instead of recomputing the whole payload.
        if (
            required.issubset(cached)
            and missing
            and all(m.get("optional") for m in missing)
        ):
            print(
                f"Augmenting simple cache with {[m['dim_label'] for m in missing]}.",
                flush=True,
            )
            an_tt = Analysis_TT(
                run_name="TT", filepath=str(tt_path) + os.sep, use_train_dm=True
            )
            for spec in missing:
                an = Analysis_DD.create(
                    run_name=spec["prefix"].upper(),
                    filepath=str(dd_dir(args, tt_path, spec)) + os.sep,
                    model_type="LFADS",
                )
                cached[spec["prefix"]] = compute_model_arrays(
                    args, cached["extras"], an_tt, an
                )
            with args.cache.open("wb") as f:
                pickle.dump(cached, f)
            return cached
        print("Cache missing expected keys; recomputing.", flush=True)

    if not tt_path.exists():
        raise FileNotFoundError(
            f"Could not find TT analysis at {tt_path}. Set --tt-path or HOME_DIR."
        )

    an_tt = Analysis_TT(
        run_name="TT", filepath=str(tt_path) + os.sep, use_train_dm=True
    )
    _, noiseless_inputs, _ = an_tt.get_model_inputs_noiseless(phase=args.phase)
    inputs_nl = as_numpy(noiseless_inputs)
    extras = as_numpy(an_tt.get_extra_inputs(phase=args.phase))
    lats_tt = as_numpy(an_tt.get_latents(phase=args.phase))
    inds_a = inputs_nl[:, :, 1].sum(axis=1) > 0
    inds_b = inputs_nl[:, :, 2].sum(axis=1) > 0

    payload: dict = {"extras": extras, "inds_a": inds_a, "inds_b": inds_b}
    true_rates = None
    for spec in models:
        dd_path = dd_dir(args, tt_path, spec)
        if not dd_path.exists():
            raise FileNotFoundError(
                f"Could not find {spec['dim_label']} DD analysis at {dd_path}."
            )
        an = Analysis_DD.create(
            run_name=spec["prefix"].upper(),
            filepath=str(dd_path) + os.sep,
            model_type="LFADS",
        )
        if true_rates is None:
            true_rates = as_numpy(an.get_true_rates(phase=args.phase))
            payload["true_rates"] = true_rates
        payload[spec["prefix"]] = compute_model_arrays(args, extras, an_tt, an)

    # Ground-truth row: PCs of the true rates and of the task-trained latents.
    payload["gt"] = {
        "rates_pca": response_pca(true_rates, extras),
        "lats_pca": response_pca(lats_tt, extras),
    }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("wb") as f:
        pickle.dump(payload, f)
    return payload


# %%
def plot_raster(
    ax,
    img: np.ndarray,
    vmax: float,
    title: str | None,
    show_xlabel: bool,
    show_ylabel: bool,
):
    """imagesc a (neurons x time) firing-rate matrix with a shared color scale."""
    ax.imshow(
        img,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="lower",
        vmin=0.0,
        vmax=vmax,
    )
    if title:
        ax.set_title(title, pad=2)
    ax.tick_params(length=2, pad=1)
    ax.spines[["top", "right"]].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Time (bins)", labelpad=0)
    else:
        ax.set_xticks([])
    if show_ylabel:
        ax.set_ylabel("Neurons", labelpad=1)
        ax.set_yticks([0, img.shape[0] - 1])
        ax.set_yticklabels(["1", str(img.shape[0])])
    else:
        ax.set_yticks([])


def stamp_row_labels(fig, row_labels, x=0.04):
    """Bold 12 pt Arial panel labels (one per row) at the figure's left margin.

    ``row_labels`` is a list of (raster_axes, letter); the letter is placed at
    the top-left of the row, flush to the left edge of the figure.
    """
    fig.canvas.draw()
    for ax, letter in row_labels:
        pos = ax.get_position()
        fig.text(
            x,
            pos.y1,
            letter,
            fontsize=12,
            fontweight="bold",
            family="Arial",
            ha="left",
            va="top",
        )


def build_figure(payload: dict, args: argparse.Namespace):
    base.setup_matplotlib()

    extras = payload["extras"]
    inds_a = payload["inds_a"]
    inds_b = payload["inds_b"]
    true_rates = payload["true_rates"]
    trial = int(np.clip(args.trial, 0, true_rates.shape[0] - 1))

    # Row 0 is ground truth (true rates + TT latents); each subsequent row is a
    # DD model's inferred rates + latents. Columns: raster, rate-space PCs,
    # latent-space PCs.
    rows = [
        {
            "label": "Ground\nTruth",
            "rates": true_rates,
            "rate_title": "Ground-Truth Rates",
            "pcs": payload["gt"],
            "color": "black",
        }
    ]
    for spec in MODELS:
        # Skip optional rows that were suppressed (--no-<prefix>) or whose arrays
        # aren't in the payload (directory absent at compute time).
        if spec.get("optional") and getattr(args, f"no_{spec['prefix']}", False):
            continue
        model = payload.get(spec["prefix"])
        if model is None:
            continue
        rows.append(
            {
                "label": spec["dim_label"],
                "rates": model["inf_rates"],
                "rate_title": "DD-Inferred Rates",
                "pcs": model,
                "color": base.DD_RED,
            }
        )

    # Global color scale across every raster so rows are directly comparable.
    imgs = [row["rates"][trial].T for row in rows]
    vmax = float(np.percentile(np.concatenate(imgs, axis=1), 99)) or 1.0

    # Compact layout: a narrow raster column (col 0) keeps the rasters small,
    # leaving the two 3D PC columns the dominant elements. Panel letters run in
    # row-major order (A-C, D-F, ...).
    fig = plt.figure(figsize=(5.0, 6.4), constrained_layout=False)
    gs = fig.add_gridspec(
        len(rows),
        3,
        width_ratios=[0.42, 1.0, 1.0],
        wspace=0.3,
        hspace=0.32,
        left=0.18,
        right=0.98,
        top=0.92,
        bottom=0.1,
    )
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    row_labels = []

    for r, row in enumerate(rows):
        top = r == 0
        bottom = r == len(rows) - 1
        img = row["rates"][trial].T

        ax_rast = fig.add_subplot(gs[r, 0])
        # Column 1 holds each model's firing-rate raster (ground-truth for the
        # top row, model-inferred for the rest), so the header stays generic.
        plot_raster(ax_rast, img, vmax, "Firing Rates" if top else None, bottom, True)
        ax_rast.text(
            -0.78,
            0.5,
            row["label"],
            transform=ax_rast.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontweight="bold",
            fontsize=8,
        )
        row_labels.append((ax_rast, next(letters)))

        ax_rate = fig.add_subplot(gs[r, 1], projection="3d")
        base.plot_pca_panel(
            ax_rate,
            row["pcs"]["rates_pca"],
            extras,
            inds_a,
            inds_b,
            "Rate-Space PCs" if top else "",
            row["color"],
            args.max_pca_trials,
        )

        ax_lat = fig.add_subplot(gs[r, 2], projection="3d")
        base.plot_pca_panel(
            ax_lat,
            row["pcs"]["lats_pca"],
            extras,
            inds_a,
            inds_b,
            "Latent-Space PCs" if top else "",
            row["color"],
            args.max_pca_trials,
        )

        # Annotate the DD-model rows with Rate R² (rate-space) and State R²
        # (latent-space); the ground-truth row is the reference and has none.
        rate_r2 = row["pcs"].get("rate_r2")
        state_r2 = row["pcs"].get("state_r2")
        if rate_r2 is not None:
            print(
                f"{row['label'].replace(chr(10), ' ')}: Rate R²={rate_r2:.3f}  State R²={state_r2:.3f}",
                flush=True,
            )
            ax_rate.text2D(
                0.02,
                0.96,
                f"Rate $R^2$ = {rate_r2:.2f}",
                transform=ax_rate.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )
            ax_lat.text2D(
                0.02,
                0.96,
                f"State $R^2$ = {state_r2:.2f}",
                transform=ax_lat.transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )

    # StimA / StimB legend.
    handles = [
        plt.Line2D([0], [0], color=base.STIM_A, lw=2, label="Stim A"),
        plt.Line2D([0], [0], color=base.STIM_B, lw=2, label="Stim B"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.58, 0.005),
    )
    stamp_row_labels(fig, row_labels)
    return fig


# %%
def main(argv=None):
    args = parse_args(argv)
    payload = load_or_compute_payload(args)
    fig = build_figure(payload, args)
    base.save_figure(
        fig, args.output_dir, "PhaseCodedMemory_DD", args.dpi, exts=("png", "pdf")
    )
    # Drop the figure straight into the manuscript figs dir so the
    # \includegraphics{figs/PhaseCodedMemory_DD.png} picks it up automatically.
    base.save_figure(
        fig, args.manuscript_dir, "PhaseCodedMemory_DD", args.dpi, exts=("png", "pdf")
    )
    plt.close(fig)
    print(f"Saved PhaseCodedMemory DD figure to {args.output_dir.resolve()}")
    print(f"Copied PhaseCodedMemory DD figure to {args.manuscript_dir.resolve()}")


if __name__ == "__main__":
    main()
