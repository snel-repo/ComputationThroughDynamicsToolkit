"""Recompute affected figure metrics from saved models into a fresh run folder.

Run on the integration branch containing the metric and analysis fixes. Defaults
to Figures 5, 6, and the compiled supplementary metric sweeps. Optional jobs
refresh Figure 4, PhaseCodedMemory panels, and nonlinear cycle consistency.
No training, publication-data export, or manuscript rebuild is performed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "examples" / "figures"
SCRIPTS = {
    "fig4": FIGURES / "Fig4Canonical" / "make_figure4_canonical.py",
    "fig5": FIGURES / "Fig5Metrics" / "make_figure5_reconstruction_simplicity.py",
    "fig6": FIGURES / "Fig6InputInf" / "make_figure6_combined.py",
    "compiled": FIGURES / "supplementary" / "make_compiled_metrics_vs_latent_size.py",
    "pcm": FIGURES / "supplementary" / "make_phase_coded_memory_figure.py",
    "nlcycle": FIGURES / "supplementary" / "make_nl_cycle_consistency_node_sweep.py",
}


def paper_source(source, filename):
    """Stop Figure 5 after its main guard, before interactive browser cells."""
    tree = ast.parse(source, filename=str(filename))
    guards = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "__name__ == '__main__'"
    ]
    if len(guards) != 1:
        raise ValueError("Expected one Figure 5 main guard; review the runner.")
    tree.body = tree.body[: guards[0] + 1]
    return tree


def job_arguments(name, directory, tasks, home_dir):
    outputs = directory / "outputs"
    common = ["--output-dir", str(outputs)]
    if name == "fig4":
        return []  # This script writes relative to the job's working directory.
    if name == "fig6":
        return common + [
            "--force",
            "--metrics-cache",
            str(directory / "metrics.pkl"),
            "--fps-cache",
            str(directory / "fixed_points.pkl"),
        ]
    if name == "compiled":
        return common + [
            "--recompute",
            "--strict",
            "--home-dir",
            str(home_dir),
            "--cache-dir",
            str(directory / "cache"),
            "--tasks",
            *tasks,
        ]
    arguments = common + ["--force", "--cache", str(directory / "metrics.pkl")]
    if name == "pcm":
        arguments += ["--manuscript-dir", str(directory / "manuscript_preview")]
    if name == "fig5":
        arguments += ["--no-noiseless", "--no-demean"]
    return arguments


def run_worker(name, arguments, seed, threads):
    # Every job gets a new interpreter and imports this checkout explicitly.
    for path in (ROOT, ROOT / "libs" / "DSA", ROOT / "libs" / "lfads-jslds"):
        sys.path.insert(0, str(path))
    import dotenv
    import numpy as np
    import torch

    # Some serialized-model modules call load_dotenv(override=True). Preserve
    # the explicit data root supplied by the launcher, within this process.
    original_load_dotenv = dotenv.load_dotenv

    def load_dotenv(*args, **kwargs):
        kwargs["override"] = False
        return original_load_dotenv(*args, **kwargs)

    dotenv.load_dotenv = load_dotenv
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    script = SCRIPTS[name]
    sys.argv = [str(script), *arguments]
    source = script.read_text()
    tree = paper_source(source, script) if name == "fig5" else ast.parse(source)
    namespace = {"__name__": "__main__", "__file__": str(script), "__package__": None}
    exec(compile(tree, str(script), "exec"), namespace)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", nargs="+", choices=SCRIPTS, default=["fig5", "fig6", "compiled"]
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["NBFF", "MultiTask", "RandomTarget", "PCM", "CDM"],
        default=["NBFF", "MultiTask", "RandomTarget", "PCM", "CDM"],
    )
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=ROOT,
        help="Root containing content/trained_models; overrides .env HOME_DIR.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="New run directory; must not already exist.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without importing model dependencies or creating files.",
    )
    parser.add_argument("--worker", choices=SCRIPTS, help=argparse.SUPPRESS)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        arguments = args.worker_args
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        run_worker(args.worker, arguments, args.seed, args.threads)
        return 0
    if args.worker_args:
        raise ValueError("Unexpected extra arguments")
    home_dir = args.home_dir.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_root = (
        (args.output_root or ROOT / "outputs" / f"fixed_figures_{stamp}")
        .expanduser()
        .resolve()
    )
    jobs = []
    for name in dict.fromkeys(args.jobs):
        directory = output_root / name
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--seed",
            str(args.seed),
            "--threads",
            str(args.threads),
            "--worker",
            name,
            "--",
            *job_arguments(name, directory, args.tasks, home_dir),
        ]
        jobs.append(
            {
                "name": name,
                "command": command,
                "cwd": str(directory),
                "status": "pending",
            }
        )
        print(f"{name}: {shlex.join(command)}", flush=True)
    print(f"HOME_DIR={home_dir}{os.sep}\nRun directory: {output_root}", flush=True)
    if args.dry_run:
        return 0
    if not (home_dir / "content" / "trained_models" / "task-trained").is_dir():
        raise FileNotFoundError(
            "--home-dir must contain content/trained_models/task-trained"
        )
    # An exclusive new folder prevents accidentally replacing accepted caches.
    output_root.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        HOME_DIR=str(home_dir) + os.sep,
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads),
    )
    versions = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "pytorch-lightning"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "revision": revision,
        "home_dir": str(home_dir),
        "seed": args.seed,
        "threads": args.threads,
        "python": sys.executable,
        "versions": versions,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                *SCRIPTS.values(),
                *sorted((ROOT / "ctd").rglob("*.py")),
            ]
        },
        "jobs": jobs,
    }

    def save_manifest():
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    save_manifest()
    for job in jobs:
        directory = Path(job["cwd"])
        (directory / "outputs").mkdir(parents=True)
        started = time.monotonic()
        job["status"] = "running"
        save_manifest()
        try:
            with (directory / "run.log").open("w") as log:
                result = subprocess.run(
                    job["command"],
                    cwd=directory,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            job["returncode"] = result.returncode
            job["status"] = "passed" if result.returncode == 0 else "failed"
        except KeyboardInterrupt:
            job["status"] = "interrupted"
            raise
        except OSError as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
        finally:
            job["elapsed_seconds"] = time.monotonic() - started
            save_manifest()
        print(f"{job['name']}: {job['status']} — {directory / 'run.log'}", flush=True)
    return int(any(job["status"] != "passed" for job in jobs))


if __name__ == "__main__":
    raise SystemExit(main())
