"""Run every figure-generation script (and notebook) for the paper in one shot.

This is the umbrella orchestrator that drives the per-figure scripts and
notebooks listed in ``FIGURE_GENERATION.md``. Each item runs in its own
subprocess so a failure in one figure does not abort the rest; a summary of
successes / failures is printed at the end and a non-zero exit code is returned
if anything failed.

Usage examples
--------------

Run everything (main-text scripts, notebooks, and supplementary figures)::

    python examples/figures/make_all_paper_figures.py

Skip notebook execution (much faster; only the ``make_*.py`` scripts run)::

    python examples/figures/make_all_paper_figures.py --no-notebooks

Run only a subset by name (matched against the registered ``name`` field)::

    python examples/figures/make_all_paper_figures.py --only fig5 fig6 supp_3bff

List what would run without running anything::

    python examples/figures/make_all_paper_figures.py --list
"""

# %%
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = Path(__file__).resolve().parent


@dataclass
class FigureJob:
    """One figure-generation unit."""

    name: str
    kind: str  # "script" or "notebook"
    path: Path
    extra_args: list[str] = field(default_factory=list)
    group: str = "main"  # "main", "supplementary"

    def relative(self) -> str:
        try:
            return str(self.path.relative_to(REPO_ROOT))
        except ValueError:
            return str(self.path)


def build_jobs() -> list[FigureJob]:
    """The canonical list of paper figure jobs, in execution order."""
    jobs: list[FigureJob] = [
        # --- Main text --------------------------------------------------
        FigureJob(
            name="fig1",
            kind="script",
            path=FIGURES_DIR / "Fig1PerfCriteria" / "make_figure1_perf_criteria.py",
            group="main",
        ),
        FigureJob(
            name="fig3",
            kind="script",
            path=FIGURES_DIR
            / "Fig3TaskPerformance"
            / "make_figure3_task_performance.py",
            group="main",
        ),
        FigureJob(
            name="fig4",
            kind="script",
            path=FIGURES_DIR / "Fig4Canonical" / "make_figure4_canonical.py",
            group="main",
        ),
        FigureJob(
            name="fig5",
            kind="script",
            path=FIGURES_DIR
            / "Fig5Metrics"
            / "make_figure5_reconstruction_simplicity.py",
            group="main",
        ),
        FigureJob(
            name="fig6",
            kind="script",
            path=FIGURES_DIR / "Fig6InputInf" / "make_figure6_combined.py",
            group="main",
        ),
        # --- Supplementary ----------------------------------------------
        FigureJob(
            name="supp_3bff",
            kind="script",
            path=FIGURES_DIR / "supplementary" / "make_3bff_trial_fig.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_phase_coded",
            kind="script",
            path=FIGURES_DIR / "supplementary" / "make_phase_coded_memory_figure.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_phase_coded_dd",
            kind="script",
            path=FIGURES_DIR / "supplementary" / "make_phase_coded_memory_simple.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_multitask",
            kind="script",
            path=FIGURES_DIR / "supplementary" / "make_multitask_behavior_figure.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_chaotic_delayed",
            kind="script",
            path=FIGURES_DIR
            / "supplementary"
            / "make_chaotic_delayed_memory_figure.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_nl_cycle_node_sweep",
            kind="script",
            path=FIGURES_DIR
            / "supplementary"
            / "make_nl_cycle_consistency_node_sweep.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_lyapunov_validation",
            kind="script",
            path=FIGURES_DIR / "supplementary" / "lyapunov_spectrum_lorenz.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_dsa",
            kind="script",
            path=FIGURES_DIR / "supplementary" / "make_figure_dsa.py",
            group="supplementary",
        ),
        FigureJob(
            name="supp_metrics_vs_latent_size",
            kind="script",
            path=FIGURES_DIR
            / "supplementary"
            / "make_compiled_metrics_vs_latent_size.py",
            group="supplementary",
        ),
    ]
    return jobs


# %%
def _in_notebook() -> bool:
    """True when running inside an IPython/Jupyter kernel."""
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
    # When running cell-by-cell in a Jupyter kernel, sys.argv contains the
    # kernel's '--f=.../kernel.json' arg, which argparse would reject. Default
    # to an empty argv (= "run everything") in that case.
    if argv is None and _in_notebook():
        argv = []
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Run only jobs whose name matches one of these (e.g. fig5 supp_3bff).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="NAME",
        help="Skip jobs whose name matches one of these.",
    )
    parser.add_argument(
        "--group",
        choices=("all", "main", "supplementary"),
        default="all",
        help="Restrict to a single group.",
    )
    parser.add_argument(
        "--notebooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute notebooks via jupyter nbconvert (default: on).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write notebook execution output back to the source .ipynb (default writes a sibling '*.executed.ipynb').",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-job timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep running remaining jobs after a failure (default: on).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the registered jobs and exit without running anything.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command for each job but do not execute it.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for script jobs (default: current).",
    )
    parser.add_argument(
        "--nbconvert",
        default=None,
        help="Path to jupyter nbconvert (default: 'jupyter' on PATH).",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to every script job (after '--'). Notebook jobs ignore these.",
    )
    parser.add_argument(
        "--mpl-backend",
        default="Agg",
        help=(
            "Matplotlib backend forced onto every job's subprocess via MPLBACKEND. "
            "Default 'Agg' makes plt.show() a no-op and silences the per-figure "
            "Figure(WxH) reprs that matplotlib_inline emits in headless runs. "
            "Pass an interactive backend (e.g. Qt5Agg, TkAgg) to see live windows."
        ),
    )
    return parser.parse_args(argv)


# %%
def filter_jobs(jobs: list[FigureJob], args: argparse.Namespace) -> list[FigureJob]:
    selected = jobs
    if args.group != "all":
        selected = [j for j in selected if j.group == args.group]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {j.name for j in jobs}
        if unknown:
            print(f"warning: --only contains unknown job(s): {sorted(unknown)}")
        selected = [j for j in selected if j.name in wanted]
    if args.skip:
        skip = set(args.skip)
        selected = [j for j in selected if j.name not in skip]
    if not args.notebooks:
        selected = [j for j in selected if j.kind != "notebook"]
    return selected


def build_command(job: FigureJob, args: argparse.Namespace) -> list[str]:
    if job.kind == "script":
        cmd = [args.python, str(job.path), *job.extra_args, *args.extra]
        return cmd
    if job.kind == "notebook":
        nbconvert = args.nbconvert
        if nbconvert:
            base = [nbconvert]
        else:
            jupyter = shutil.which("jupyter")
            if jupyter is None:
                raise RuntimeError(
                    "jupyter is not on PATH; pass --nbconvert /path/to/jupyter or use --no-notebooks."
                )
            base = [jupyter, "nbconvert"]
        cmd = [
            *base,
            "--to",
            "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={args.timeout}",
        ]
        if args.inplace:
            cmd += ["--inplace", str(job.path)]
        else:
            cmd += [
                "--output",
                job.path.stem + ".executed.ipynb",
                "--output-dir",
                str(job.path.parent),
                str(job.path),
            ]
        return cmd
    raise ValueError(f"Unknown job kind: {job.kind}")


# %%
# Where figure scripts plausibly write their outputs. We snapshot mtimes for
# files under these dirs before each job, then report the new/modified ones
# after. This gives every script an automatic "here are your outputs" trailer
# without requiring each script to track its own savefig paths.
_CONTENT_FIGURES = REPO_ROOT / "content" / "figures"


def _output_dirs_for(job: FigureJob) -> list[Path]:
    dirs = []
    if job.kind == "script":
        dirs.append(job.path.parent / "outputs")
    dirs.append(_CONTENT_FIGURES)
    return [d for d in dirs if d.exists() or d == job.path.parent / "outputs"]


def _snapshot_mtimes(dirs: list[Path]) -> dict[Path, float]:
    snap: dict[Path, float] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    snap[p] = p.stat().st_mtime
                except OSError:
                    pass
    return snap


def _new_or_modified_since(
    snapshot: dict[Path, float], dirs: list[Path], started_at: float
) -> list[Path]:
    found: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            prev = snapshot.get(p)
            # New file, or one modified after the job started (with a small
            # epsilon to absorb filesystem mtime quantisation).
            if prev is None or m > max(prev, started_at) - 1e-3:
                # Filter known cache/metadata sidecars so the report is clean.
                if p.suffix in {".pyc"} or "__pycache__" in p.parts:
                    continue
                if "cache" in p.parts or p.name.endswith(".cache.pkl"):
                    continue
                found.append(p)
    return sorted(set(found))


@dataclass
class JobResult:
    job: FigureJob
    returncode: int
    duration_s: float
    skipped: bool = False
    reason: str = ""
    outputs: list[Path] = field(default_factory=list)


def run_job(job: FigureJob, args: argparse.Namespace) -> JobResult:
    if not job.path.exists():
        return JobResult(
            job,
            returncode=127,
            duration_s=0.0,
            skipped=True,
            reason=f"missing: {job.path}",
        )

    cmd = build_command(job, args)
    print("\n" + "=" * 72)
    print(f"[{job.name}] ({job.group}, {job.kind})")
    print(f"  -> {job.relative()}")
    print(f"  $ {' '.join(cmd)}")
    if args.dry_run:
        return JobResult(
            job, returncode=0, duration_s=0.0, skipped=True, reason="dry-run"
        )

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Force matplotlib's non-interactive Agg backend so plt.show() is a no-op
    # and the converted-notebook scripts don't dump `Figure(WxH)` reprs to
    # stdout (matplotlib_inline's display() does this on plt.show() under a
    # plain Python subprocess). Override outright — setdefault was not enough
    # because the parent shell or a matplotlibrc can preset a non-Agg backend,
    # leaving the prints in place. Use --mpl-backend to opt out.
    env["MPLBACKEND"] = args.mpl_backend
    # Silence the matching "Matplotlib is currently using agg, which is a
    # non-GUI backend, so cannot show the figure." UserWarning that pyplot
    # emits on plt.show() under the Agg backend. Skip the filter when the
    # user is explicitly requesting a GUI backend — that warning would not
    # fire there, and we shouldn't silence other legitimate warnings.
    if args.mpl_backend.lower() in {"agg", "pdf", "svg", "ps", "cairo"}:
        agg_warning_filter = "ignore:Matplotlib is currently using agg"
        existing = env.get("PYTHONWARNINGS", "")
        env["PYTHONWARNINGS"] = (
            f"{agg_warning_filter},{existing}" if existing else agg_warning_filter
        )
    repo_str = str(REPO_ROOT)
    if repo_str not in env.get("PYTHONPATH", "").split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [repo_str, env.get("PYTHONPATH", "")])
        )

    # Run each script from its own directory so relative paths like
    # 'outputs/...' resolve next to the script (matches how each figure was
    # developed in its own folder / notebook).
    cwd = job.path.parent if job.kind == "script" else REPO_ROOT

    output_dirs = _output_dirs_for(job)
    pre_snapshot = _snapshot_mtimes(output_dirs)
    started_at = time.time()
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            timeout=args.timeout,
            check=False,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124
        print(f"[{job.name}] timed out after {args.timeout}s")
    duration = time.monotonic() - start

    outputs = _new_or_modified_since(pre_snapshot, output_dirs, started_at)
    if outputs:
        print(f"  wrote {len(outputs)} file(s):")
        for p in outputs:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            print(f"    - {rel}")
    return JobResult(job, returncode=rc, duration_s=duration, outputs=outputs)


# %%
def summarize(results: list[JobResult]) -> int:
    print("\n" + "#" * 72)
    print("Summary")
    print("#" * 72)
    width = max((len(r.job.name) for r in results), default=10)
    n_ok = n_fail = n_skip = 0
    for r in results:
        if r.skipped:
            status = f"SKIP ({r.reason})"
            n_skip += 1
        elif r.returncode == 0:
            status = "OK"
            n_ok += 1
        else:
            status = f"FAIL (rc={r.returncode})"
            n_fail += 1
        n_out = (
            f"  ({len(r.outputs)} file{'s' if len(r.outputs) != 1 else ''})"
            if r.outputs
            else ""
        )
        print(f"  {r.job.name:<{width}}  {r.duration_s:7.1f}s  {status}{n_out}")
    print("-" * 72)
    print(f"  {n_ok} ok, {n_fail} failed, {n_skip} skipped, {len(results)} total")

    all_outputs: list[Path] = [p for r in results for p in r.outputs]
    if all_outputs:
        print("\nGenerated files:")
        for r in results:
            if not r.outputs:
                continue
            print(f"  [{r.job.name}]")
            for p in r.outputs:
                try:
                    rel = p.relative_to(REPO_ROOT)
                except ValueError:
                    rel = p
                print(f"    - {rel}")
    return 0 if n_fail == 0 else 1


# %%
def main(argv=None) -> int:
    args = parse_args(argv)
    jobs = filter_jobs(build_jobs(), args)

    if args.list:
        for j in jobs:
            exists = "ok" if j.path.exists() else "MISSING"
            print(
                f"  [{j.group:<13}] {j.name:<26} {j.kind:<8} ({exists})  {j.relative()}"
            )
        return 0

    if not jobs:
        print("No jobs selected; nothing to do.")
        return 0

    results: list[JobResult] = []
    for job in jobs:
        result = run_job(job, args)
        results.append(result)
        if result.returncode != 0 and not result.skipped and not args.continue_on_error:
            print(
                f"[{job.name}] failed (rc={result.returncode}); aborting (--no-continue-on-error)."
            )
            break

    return summarize(results)


if __name__ == "__main__":
    sys.exit(main())
