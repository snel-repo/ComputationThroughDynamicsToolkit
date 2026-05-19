import warnings

import numpy as np
import torch


def _get_dsa_classes():
    try:
        from DSA import DMD, DSA, SimilarityTransformDist
    except ImportError as exc:
        raise ImportError(
            "DSA helpers require the pip package `dsa-metric`. "
            "Install the repository with `pip install -e .` so the "
            "`dsa-metric @ git+https://github.com/mitchellostrow/DSA.git@main` "
            "dependency is available."
        ) from exc
    return DMD, DSA, SimilarityTransformDist


def torch_convert(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def dsa_to_id(
    data,
    rank,
    n_delays,
    delay_interval,
    iters=1000,
    lr=1e-2,
    score_method="angular",
    device="cpu",
):
    """Compute DSA between a dataset's DMD operator and the identity matrix.

    These helpers existed in older DSA releases but were removed from the
    pip-installable ``dsa-metric`` package. Keeping them here preserves the
    CtD analysis API while using DSA's public DMD and similarity classes.
    """
    DMD, _, SimilarityTransformDist = _get_dsa_classes()
    data = torch_convert(data)
    dmd = DMD(
        data,
        n_delays=n_delays,
        delay_interval=delay_interval,
        rank=rank,
        device=device,
    )
    dmd.fit()

    if dmd.rank < rank:
        warnings.warn(
            f"The rank of the DMD model {dmd.rank} is less than the specified "
            f"rank {rank}. Will revert to comparison at that rank.",
            stacklevel=2,
        )

    simdist = SimilarityTransformDist(
        iters=iters, score_method=score_method, lr=lr, device=device
    )
    return simdist.fit_score(dmd.A_v, torch.eye(dmd.rank, device=device))


def dsa_bw_data_splits(
    data,
    rank,
    n_delays,
    delay_interval,
    nsplits=2,
    iters=1000,
    lr=1e-2,
    score_method="angular",
    device="cpu",
    avg=True,
):
    """Compute DSA between splits of one dataset.

    This mirrors the legacy ``DSA.stats.dsa_bw_data_splits`` helper used by
    CtD's hyperparameter sweeps.
    """
    _, DSA, _ = _get_dsa_classes()

    if isinstance(data, list):
        data = np.array(data)

    if data.shape[0] % nsplits:
        data = data[: data.shape[0] - (data.shape[0] % nsplits)]

    data_splits = np.split(data, nsplits, axis=0)
    dsa = DSA(
        data_splits,
        n_delays=n_delays,
        rank=rank,
        delay_interval=delay_interval,
        iters=iters,
        lr=lr,
        score_method=score_method,
        device=device,
    )
    score = dsa.fit_score()
    if avg:
        score = score[np.tril_indices(score.shape[0], k=-1)].mean()

    return score
