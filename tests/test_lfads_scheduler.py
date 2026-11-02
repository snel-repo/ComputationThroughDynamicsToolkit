from types import SimpleNamespace

import pytest
import torch

from ctd.data_modeling.models.LFADS.lfads import LFADS


def optimizer_model(scheduler_enabled=True):
    # Optimizer configuration needs parameters and hyperparameters, not a
    # complete recurrent model or training dataset.
    parameter = torch.nn.Parameter(torch.ones(1))
    return SimpleNamespace(
        parameters=lambda: [parameter],
        hparams=SimpleNamespace(
            lr_scheduler=scheduler_enabled,
            lr_init=0.01,
            lr_stop=0.002,
            lr_decay=0.5,
            lr_patience=1,
            lr_adam_beta1=0.9,
            lr_adam_beta2=0.999,
            lr_adam_epsilon=1e-8,
            weight_decay=0.01,
        ),
    )


def test_scheduler_uses_supported_api_and_reduces_lr(monkeypatch):
    real_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

    # Match the supported keyword signature in PyTorch >= 2.7, even when
    # running this regression on an older supported PyTorch installation.
    def supported_scheduler(*, optimizer, mode, factor, patience, threshold, min_lr):
        return real_scheduler(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
        )

    monkeypatch.setattr(
        torch.optim.lr_scheduler, "ReduceLROnPlateau", supported_scheduler
    )
    config = LFADS.configure_optimizers(optimizer_model())
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert config["monitor"] == "valid/recon_smth"
    assert scheduler.optimizer is optimizer

    scheduler.step(1.0)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    for _ in range(10):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.002)


def test_disabled_scheduler_returns_optimizer():
    optimizer = LFADS.configure_optimizers(optimizer_model(scheduler_enabled=False))
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
