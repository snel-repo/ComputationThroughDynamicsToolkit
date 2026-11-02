"""Resolve trial lengths using the task's documented metadata schema."""

import torch


def trial_length_phases(phase):
    if phase not in {"train", "val", "all"}:
        raise ValueError(f"Unknown phase {phase!r}; expected 'train', 'val', or 'all'.")
    return ("train", "val") if phase == "all" else (phase,)


def get_trial_length_task(datamodule):
    # SAE's historical `system` defaults to 3BFF even for other tasks, so
    # prefer the dataset prefix saved with the datamodule.
    prefix = datamodule.hparams.get("prefix")
    if prefix is None:
        prefix = getattr(datamodule, "fpath", None)
    if prefix is None:
        prefix = getattr(datamodule, "system", "")
    return str(prefix).rstrip("/").split("/")[-1].lower().removeprefix("tt_")


def resolve_trial_lengths(data, task, extra=None, sequence_length=None):
    """Return integer lengths in the reconstruction window's coordinates.

    Simulator trimming is already applied to the saved sequences and timing
    metadata. LFADS's decoder emits `recon_seq_len` steps; its IC encoder
    segment is not an additional offset to subtract from reconstruction ends.
    """
    n_trials, n_steps = data.shape[:2]
    if sequence_length is not None:
        n_steps = int(sequence_length)
    if n_steps <= 0:
        raise ValueError("Trial lengths require a nonempty reconstruction window.")

    if task == "multitask":
        if extra is None or extra.ndim != 2 or extra.shape != (n_trials, 2):
            raise ValueError(
                "MultiTask requires 'extra' metadata with response start/end "
                "columns for every trial."
            )
        ends = extra[:, 1]
        if not torch.all(torch.isfinite(ends) & (ends > 0) & (ends == ends.round())):
            raise ValueError(
                "MultiTask response ends must be positive integer indices."
            )
        return ends.to(device=data.device, dtype=torch.long).clamp(max=n_steps)

    if task not in {
        "3bff",
        "nbff",
        "randomtarget",
        "phasecodedmemory",
        "chaoticdelayedmatching",
    }:
        raise ValueError(
            f"Unknown trial-length schema for task {task!r}; "
            "specify a supported dataset prefix instead of assuming full-length trials."
        )
    return torch.full((n_trials,), n_steps, dtype=torch.long, device=data.device)
