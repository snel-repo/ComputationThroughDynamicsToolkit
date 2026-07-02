"""Generate supplementary Figure S2: example 3BFF trial I/O traces."""

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

import matplotlib.pyplot as plt
import numpy as np
import torch

from ctd.comparison.analysis.tt.tt import Analysis_TT

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]


BIT_COLORS = ("#2ca25f", "#e68613", "#2b6cb0")
ROW_LABELS = ("Model output", "Target output", "Noisy inputs")
DEFAULT_PRETRAINED_PATH = REPO_ROOT / "pretrained" / "20241017_NBFF_NoisyGRU_NewFinal"


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
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--timepoints", type=int, default=501)
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
        get_home_dir() / "content" / "trained_models" / "task-trained" / "tt_3bff"
    )
    if (trained_path / "model.pkl").exists():
        return trained_path
    return DEFAULT_PRETRAINED_PATH


def load_trial(
    tt_path: Path, phase: str, trial: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    analysis = Analysis_TT("tt_3bff", str(tt_path) + os.sep)
    analysis.wrapper.eval()

    ics, inputs, targets = analysis.get_model_inputs(phase=phase)
    inputs_to_env = analysis.get_inputs_to_env(phase=phase)
    if trial < 0 or trial >= inputs.shape[0]:
        raise IndexError(
            f"Trial {trial} is outside the {phase!r} split with {inputs.shape[0]} trials."
        )

    with torch.no_grad():
        out_dict = analysis.wrapper(ics, inputs, inputs_to_env)

    controlled = as_numpy(out_dict["controlled"])[trial]
    targets = as_numpy(targets)[trial]
    inputs = as_numpy(inputs)[trial]
    return controlled, targets, inputs


def plot_s2(
    controlled: np.ndarray, targets: np.ndarray, inputs: np.ndarray, timepoints: int
) -> plt.Figure:
    traces = (controlled, targets, inputs)
    n_time = min(timepoints, *(trace.shape[0] for trace in traces))
    time = np.arange(n_time)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 3.4), sharex=True)
    for ax, row_label, trace in zip(axes, ROW_LABELS, traces):
        for bit_idx, color in enumerate(BIT_COLORS):
            ax.plot(
                time,
                trace[:n_time, bit_idx],
                color=color,
                lw=1.6,
                label=f"Bit {bit_idx + 1}",
            )
        ax.set_ylabel(row_label, fontsize=9)
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].legend(
        loc="upper right", ncol=3, frameon=False, fontsize=8, handlelength=1.5
    )
    axes[0].set_ylim(-0.1, 1.1)
    axes[1].set_ylim(-0.1, 1.1)
    axes[2].set_ylim(-1.5, 1.5)
    axes[-1].set_xlim(0, 500)
    axes[-1].set_xticks([0, 500])
    axes[-1].set_xlabel("Time")

    fig.tight_layout(pad=0.8, h_pad=0.6)
    return fig


def main(argv=None) -> None:
    args = parse_args(argv)
    tt_path = args.tt_path or default_tt_path()
    if not (tt_path / "model.pkl").exists():
        raise FileNotFoundError(f"Could not find model.pkl in {tt_path}")

    controlled, targets, inputs = load_trial(tt_path, args.phase, args.trial)
    fig = plot_s2(controlled, targets, inputs, args.timepoints)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            args.output_dir / f"FigureS2_3BFF_trial.{ext}",
            dpi=args.dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
