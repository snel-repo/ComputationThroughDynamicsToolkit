"""Generate a supplementary figure showing MultiTask trial behavior."""

# %%
from __future__ import annotations

import argparse
import os
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from ctd.comparison.analysis.tt.tt import Analysis_TT

# %%
TASK_GROUPS = {
    "Response": [
        "DelayAnti",
        "DelayPro",
        "MemoryAnti",
        "MemoryPro",
        "ReactAnti",
        "ReactPro",
    ],
    "Decisionmaking": [
        "ContextIntMod1",
        "ContextIntMod2",
        "ContextIntMultimodal",
        "IntMod1",
        "IntMod2",
    ],
    "Matching": [
        "Match2Sample",
        "MatchCatAnti",
        "MatchCatPro",
        "NonMatch2Sample",
    ],
}
COLUMN_LABELS = ("Input 1", "Input 2", "Model", "Target")
PHASE_PALETTE = (
    "#111111",
    "#2ca25f",
    "#d73027",
    "#1f78b4",
    "#ff7f00",
    "#6a3d9a",
    "#e7298a",
    "#a6761d",
    "#66a61e",
    "#7570b3",
)
DEFAULT_PRETRAINED_PATH = (
    REPO_ROOT / "pretrained" / "20241113_MultiTask_NoisyGRU_Final2"
)


# %%
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "outputs"
    )
    parser.add_argument(
        "--tt-path",
        type=Path,
        default=None,
        help="Path containing model.pkl and datamodule_sim.pkl.",
    )
    parser.add_argument("--phase", default="val", choices=["train", "val", "all"])
    parser.add_argument(
        "--trial-offset",
        type=int,
        default=0,
        help="Which matching trial to use within each task.",
    )
    parser.add_argument("--dpi", type=int, default=300)
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
        home = str(REPO_ROOT)
    resolved = Path(home).expanduser().resolve()
    os.environ["HOME_DIR"] = str(resolved) + os.sep
    return resolved


def default_tt_path() -> Path:
    trained_path = (
        get_home_dir() / "content" / "trained_models" / "task-trained" / "tt_MultiTask"
    )
    if (trained_path / "model.pkl").exists():
        return trained_path
    return DEFAULT_PRETRAINED_PATH


# %%
def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
        }
    )


# %%
def split_metadata(
    datamodule, phase: str
) -> tuple[list[str], list[dict[str, list[int]]]]:
    all_task_names = datamodule.extra_data["task_names"]
    all_phase_dicts = datamodule.extra_data["phase_dict"]
    train_conds = as_numpy(datamodule.train_ds.tensors[3]).astype(int).ravel()
    valid_conds = as_numpy(datamodule.valid_ds.tensors[3]).astype(int).ravel()

    if phase == "train":
        inds = train_conds
    elif phase == "val":
        inds = valid_conds
    else:
        inds = np.concatenate([train_conds, valid_conds])

    task_names = [all_task_names[ind] for ind in inds]
    phase_dicts = [all_phase_dicts[ind] for ind in inds]
    return task_names, phase_dicts


def load_examples(
    tt_path: Path, phase: str, trial_offset: int
) -> dict[str, dict[str, np.ndarray | dict[str, list[int]]]]:
    analysis = Analysis_TT("tt_MultiTask", str(tt_path) + os.sep)
    analysis.wrapper.eval()

    ics, noisy_inputs, targets = analysis.get_model_inputs(phase=phase)
    inputs_to_env = analysis.get_inputs_to_env(phase=phase)
    task_names, phase_dicts = split_metadata(analysis.datamodule, phase)

    with torch.no_grad():
        out_dict = analysis.wrapper(ics, noisy_inputs, inputs_to_env)

    model_outputs = as_numpy(out_dict["controlled"])
    targets = as_numpy(targets)
    noisy_inputs = as_numpy(noisy_inputs)

    examples = {}
    for task_name in [task for tasks in TASK_GROUPS.values() for task in tasks]:
        matches = [i for i, name in enumerate(task_names) if name == task_name]
        if not matches:
            raise ValueError(f"No {phase!r} trial found for {task_name}.")
        idx = matches[trial_offset % len(matches)]
        examples[task_name] = {
            "inputs": noisy_inputs[idx],
            "model": model_outputs[idx],
            "target": targets[idx],
            "phase_dict": phase_dicts[idx],
        }
    return examples


# %%
def build_phase_color_map(examples: dict) -> dict[str, str]:
    seen: list[str] = []
    for ex in examples.values():
        for phase in ex["phase_dict"].keys():
            if phase not in seen:
                seen.append(phase)
    return {
        phase: PHASE_PALETTE[i % len(PHASE_PALETTE)] for i, phase in enumerate(seen)
    }


def plot_phase_trajectory(
    ax,
    xy: np.ndarray,
    phase_dict: dict[str, list[int]],
    phase_colors: dict[str, str],
    emphasize_markers: bool = False,
) -> None:
    markersize = 4.0 if emphasize_markers else 2.6
    for phase, (start, end) in phase_dict.items():
        segment = xy[start:end]
        if segment.size == 0:
            continue
        color = phase_colors[phase]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=color,
            lw=0.9,
            marker="o",
            markersize=markersize,
            markeredgewidth=0,
            solid_capstyle="round",
        )


def style_axis(ax) -> None:
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def plot_task_row(
    axes,
    task_name: str,
    example: dict[str, np.ndarray | dict[str, list[int]]],
    phase_colors: dict[str, str],
) -> None:
    inputs = example["inputs"]
    model = example["model"]
    target = example["target"]
    phase_dict = example["phase_dict"]

    plot_phase_trajectory(axes[0], inputs[:, 1:3], phase_dict, phase_colors)
    plot_phase_trajectory(axes[1], inputs[:, 3:5], phase_dict, phase_colors)
    plot_phase_trajectory(
        axes[2], model[:, 1:3], phase_dict, phase_colors, emphasize_markers=True
    )
    plot_phase_trajectory(
        axes[3], target[:, 1:3], phase_dict, phase_colors, emphasize_markers=True
    )
    label = (
        "ContextInt\nMultimodal" if task_name == "ContextIntMultimodal" else task_name
    )
    axes[0].text(
        -0.18,
        0.5,
        label,
        transform=axes[0].transAxes,
        ha="right",
        va="center",
        fontsize=5.8,
    )
    for ax in axes:
        style_axis(ax)


# %%
def build_figure(
    examples: dict[str, dict[str, np.ndarray | dict[str, list[int]]]],
) -> plt.Figure:
    setup_matplotlib()
    phase_colors = build_phase_color_map(examples)
    max_rows = max(len(tasks) for tasks in TASK_GROUPS.values())
    fig = plt.figure(figsize=(13.6, 6.2), constrained_layout=False)
    outer = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[len(tasks) / max_rows for tasks in TASK_GROUPS.values()],
        wspace=0.17,
    )

    for group_idx, (group_name, tasks) in enumerate(TASK_GROUPS.items()):
        gs = GridSpecFromSubplotSpec(
            max_rows, 4, subplot_spec=outer[group_idx], wspace=0.08, hspace=0.18
        )
        group_axes = []
        for row_idx in range(max_rows):
            row_axes = []
            for col_idx in range(4):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                row_axes.append(ax)
                if row_idx == 0:
                    ax.set_title(COLUMN_LABELS[col_idx], pad=2)
            group_axes.append(row_axes)

        for row_idx, task_name in enumerate(tasks):
            plot_task_row(
                group_axes[row_idx], task_name, examples[task_name], phase_colors
            )
        for row_idx in range(len(tasks), max_rows):
            for ax in group_axes[row_idx]:
                ax.axis("off")

        title_ax = group_axes[0][0]
        title_ax.text(
            2.22,
            1.42,
            group_name,
            transform=title_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            marker="o",
            markersize=4,
            markeredgewidth=0,
            label=phase,
        )
        for phase, color in phase_colors.items()
    ]
    ncol = min(len(handles), 6)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.9, bottom=0.1)
    return fig


# %%
def main(argv=None) -> None:
    args = parse_args(argv)
    tt_path = args.tt_path or default_tt_path()
    if not (tt_path / "model.pkl").exists():
        raise FileNotFoundError(f"Could not find model.pkl in {tt_path}")

    examples = load_examples(tt_path, args.phase, args.trial_offset)
    fig = build_figure(examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            args.output_dir / f"FigureS4_MultiTask_behavior.{ext}",
            dpi=args.dpi,
            bbox_inches="tight",
        )
        print(
            f"Saved figure to {args.output_dir / f'FigureS4_MultiTask_behavior.{ext}'}"
        )
    plt.close(fig)


# %%
if __name__ == "__main__":
    main()
