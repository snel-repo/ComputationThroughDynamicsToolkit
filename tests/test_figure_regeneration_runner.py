import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/figures/regenerate_affected_figures.py"
)
spec = importlib.util.spec_from_file_location("figure_regeneration", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_figure5_worker_excludes_interactive_cells():
    source = """
events = []
def main():
    events.append('paper')
if __name__ == '__main__':
    main()
raise RuntimeError('interactive browser must not execute')
"""
    namespace = {"__name__": "__main__"}
    exec(
        compile(runner.paper_source(source, "figure5.py"), "figure5.py", "exec"),
        namespace,
    )
    assert namespace["events"] == ["paper"]
    # Confirm the current real script still satisfies the worker's contract.
    tree = runner.paper_source(runner.SCRIPTS["fig5"].read_text(), "figure5.py")
    assert isinstance(tree.body[-1], ast.If)


def test_dry_run_creates_no_files(tmp_path):
    output = tmp_path / "new_run"
    assert runner.main(["--dry-run", "--output-root", str(output)]) == 0
    assert not output.exists()


def test_runner_isolates_jobs_logs_failures_and_records_manifest(tmp_path, monkeypatch):
    home = tmp_path / "data"
    (home / "content/trained_models/task-trained").mkdir(parents=True)
    output = tmp_path / "fresh_run"
    calls = []

    def run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="test-revision\n")
        calls.append((command, kwargs))
        assert kwargs["env"]["HOME_DIR"] == str(home) + "/"
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        assert Path(kwargs["cwd"]).is_relative_to(output)
        kwargs["stdout"].write("simulated figure execution\n")
        name = command[command.index("--worker") + 1]
        return subprocess.CompletedProcess(command, 1 if name == "fig6" else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.main(
        ["--home-dir", str(home), "--output-root", str(output), "--tasks", "NBFF"]
    )
    assert result == 1
    assert len(calls) == 3  # A failed job does not prevent later jobs running.
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["revision"] == "test-revision"
    assert [job["status"] for job in manifest["jobs"]] == ["passed", "failed", "passed"]
    assert manifest["source_sha256"]
    for job in manifest["jobs"]:
        directory = Path(job["cwd"])
        assert (directory / "run.log").read_text() == "simulated figure execution\n"
        assert (directory / "outputs").is_dir()
        assert "--force" in job["command"] or "--recompute" in job["command"]
    assert "--strict" in calls[-1][0]
    assert calls[-1][0][-2:] == ["--tasks", "NBFF"]
    with pytest.raises(FileExistsError):
        runner.main(["--home-dir", str(home), "--output-root", str(output)])


def test_optional_jobs_keep_output_paths_local(tmp_path):
    args = runner.job_arguments("pcm", tmp_path, ["PCM"], tmp_path)
    assert args[args.index("--manuscript-dir") + 1] == str(
        tmp_path / "manuscript_preview"
    )
    assert runner.job_arguments("fig4", tmp_path, [], tmp_path) == []
    args = runner.job_arguments("fig6", tmp_path, [], tmp_path)
    assert args[args.index("--fps-cache") + 1] == str(tmp_path / "fixed_points.pkl")
