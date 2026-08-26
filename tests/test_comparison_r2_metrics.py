import numpy as np
import pytest
import torch

from ctd.comparison.metrics import compute_affine_mapping_r2, compute_aligned_signal_r2


def test_input_r2_maps_inferred_inputs_to_true_inputs():
    train_x = np.linspace(-2.0, 2.0, 101)
    val_x = np.linspace(-1.9, 1.9, 91)

    true_train = train_x[:, None]
    true_val = val_x[:, None]
    inferred_train = np.column_stack((2.0 * train_x + 1.0, train_x**2))
    inferred_val = np.column_stack((2.0 * val_x + 1.0, val_x**2))

    input_r2 = compute_affine_mapping_r2(
        source_train=inferred_train,
        target_train=true_train,
        source_val=inferred_val,
        target_val=true_val,
    )
    reverse_r2 = compute_affine_mapping_r2(
        source_train=true_train,
        target_train=inferred_train,
        source_val=true_val,
        target_val=inferred_val,
    )

    assert input_r2 == pytest.approx(1.0)
    assert reverse_r2 < 0.9


def test_affine_mapping_r2_flattens_batched_tensors():
    source = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    target = 3.0 * source - 2.0

    score = compute_affine_mapping_r2(
        source_train=source,
        target_train=target,
        source_val=source,
        target_val=target,
    )

    assert score == pytest.approx(1.0)


def test_affine_mapping_r2_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError, match="same number of samples"):
        compute_affine_mapping_r2(
            source_train=np.zeros((5, 1)),
            target_train=np.zeros((4, 1)),
            source_val=np.zeros((3, 1)),
            target_val=np.zeros((3, 1)),
        )


def test_aligned_signal_r2_requires_matching_shapes():
    with pytest.raises(ValueError, match="same flattened shape"):
        compute_aligned_signal_r2(
            reference_signal=np.zeros((4, 2)),
            comparison_signal=np.zeros((4, 3)),
        )
