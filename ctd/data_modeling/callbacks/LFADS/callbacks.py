import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torch.nn.functional import poisson_nll_loss

from ctd.data_modeling.callbacks.metrics import compute_metrics
from ctd.data_modeling.extensions.LFADS.utils import send_batch_to_device
from ctd.task_modeling.callbacks.callbacks import (
    _plot_local_log_growth,
    _plot_local_log_growth_by_trial_phase,
    _plot_lyapunov_histogram,
    _plot_lyapunov_history,
    _plot_perturbation_divergence,
)

# plt.switch_backend("Agg")


def has_image_loggers(loggers):
    """Checks whether any image loggers are available.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers to search.
    """
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            return True
        elif isinstance(logger, pl.loggers.WandbLogger):
            return True
    return False


def log_figure(loggers, name, fig, step):
    """Logs a figure image to all available image loggers.

    Parameters
    ----------
    loggers : obj or list[obj]
        An object or list of loggers
    name : str
        The name to use for the logged figure
    fig : matplotlib.figure.Figure
        The figure to log
    step : int
        The step to associate with the logged figure
    """
    # Save figure image to in-memory buffer
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    image = Image.open(img_buf)
    # Distribute image to all image loggers
    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.TensorBoardLogger):
            logger.experiment.add_figure(name, fig, step)
        elif isinstance(logger, pl.loggers.WandbLogger):
            logger.log_image(name, [image], step)
    img_buf.close()


def _extract_trial_phase_metadata(extra_data):
    if (
        torch.is_tensor(extra_data)
        and extra_data.ndim == 2
        and extra_data.shape[1] >= 11
    ):
        return extra_data.detach().cpu().numpy()
    if isinstance(extra_data, (tuple, list)):
        for idx in (3, -1, 0, 1, 2):
            if -len(extra_data) <= idx < len(extra_data):
                item = extra_data[idx]
                if torch.is_tensor(item) and item.ndim == 2 and item.shape[1] >= 11:
                    return item.detach().cpu().numpy()
    return None


def _get_lfads_generator_cell(pl_module):
    decoder = getattr(pl_module, "decoder", None)
    if decoder is None or not hasattr(decoder, "rnn"):
        return None
    cell = getattr(decoder.rnn, "cell", None)
    if cell is None:
        return None
    return getattr(cell, "gen_cell", None)


def _estimate_lfads_max_lyapunov_exponent(
    gen_cell,
    gen_init,
    gen_inputs,
    n_trials=8,
    warmup_steps=50,
    max_steps=None,
    return_diagnostics=False,
):
    if gen_cell is None:
        return None

    gen_init = gen_init[: min(n_trials, gen_init.shape[0])]
    gen_inputs = gen_inputs[: min(n_trials, gen_inputs.shape[0])]
    if gen_init.shape[0] == 0:
        return None

    n_total_steps = gen_inputs.shape[1]
    warmup_steps = min(int(warmup_steps), max(n_total_steps - 1, 0))
    n_steps = n_total_steps - warmup_steps
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if n_steps <= 0:
        return None

    with torch.no_grad():
        hidden = gen_init.detach().clone()
        for step_idx in range(warmup_steps):
            hidden = gen_cell(gen_inputs[:, step_idx, :], hidden)

    tangent = torch.randn_like(hidden)
    tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(
        1.0e-12
    )

    local_logs = []
    with torch.enable_grad():
        for step_idx in range(warmup_steps, warmup_steps + n_steps):
            hidden = hidden.detach().requires_grad_(True)
            next_hidden = gen_cell(gen_inputs[:, step_idx, :], hidden)
            jvp = torch.autograd.grad(
                outputs=next_hidden,
                inputs=hidden,
                grad_outputs=tangent,
                retain_graph=False,
                create_graph=False,
            )[0]
            step_norm = torch.linalg.norm(jvp, dim=1).clamp_min(1.0e-12)
            local_logs.append(torch.log(step_norm))
            tangent = (jvp / step_norm.unsqueeze(1)).detach()
            hidden = next_hidden.detach()

    local_logs = torch.stack(local_logs, dim=1)
    per_trial = local_logs.mean(dim=1)
    time_series = local_logs.mean(dim=0)
    metrics = {
        "max_lyapunov_estimate": float(per_trial.mean().detach().cpu()),
        "max_lyapunov_std": float(per_trial.std(unbiased=False).detach().cpu()),
        "local_log_growth_mean": float(local_logs.mean().detach().cpu()),
        "time_series": time_series.detach().cpu().numpy(),
        "n_trials": int(gen_init.shape[0]),
        "n_steps": int(n_steps),
        "warmup_steps": int(warmup_steps),
    }
    if return_diagnostics:
        metrics["per_trial_estimates"] = per_trial.detach().cpu().numpy()
        metrics["local_log_growth_trials"] = local_logs.detach().cpu().numpy()
        metrics["time_series_std"] = (
            local_logs.std(dim=0, unbiased=False).detach().cpu().numpy()
        )
    return metrics


def _estimate_lfads_perturbation_divergence(
    gen_cell,
    gen_init,
    gen_inputs,
    n_trials=4,
    warmup_steps=50,
    max_steps=None,
    perturbation_scale=1.0e-5,
):
    if gen_cell is None:
        return None

    gen_init = gen_init[: min(n_trials, gen_init.shape[0])]
    gen_inputs = gen_inputs[: min(n_trials, gen_inputs.shape[0])]
    if gen_init.shape[0] == 0:
        return None

    n_total_steps = gen_inputs.shape[1]
    warmup_steps = min(int(warmup_steps), max(n_total_steps - 1, 0))
    n_steps = n_total_steps - warmup_steps
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if n_steps <= 0:
        return None

    with torch.no_grad():
        hidden = gen_init.detach().clone()
        for step_idx in range(warmup_steps):
            hidden = gen_cell(gen_inputs[:, step_idx, :], hidden)

        eps = torch.randn_like(hidden)
        eps = (
            perturbation_scale
            * eps
            / torch.linalg.norm(eps, dim=1, keepdim=True).clamp_min(1.0e-12)
        )

        hidden_a = hidden.clone()
        hidden_b = hidden + eps
        deltas = []
        for step_idx in range(warmup_steps, warmup_steps + n_steps):
            hidden_a = gen_cell(gen_inputs[:, step_idx, :], hidden_a)
            hidden_b = gen_cell(gen_inputs[:, step_idx, :], hidden_b)
            delta = torch.linalg.norm(hidden_b - hidden_a, dim=1).clamp_min(1.0e-12)
            deltas.append(delta)

    deltas = torch.stack(deltas, dim=1)
    log_deltas = torch.log(deltas)
    x = torch.arange(log_deltas.shape[1], dtype=log_deltas.dtype).unsqueeze(0)
    x = x.to(log_deltas.device)
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = log_deltas - log_deltas.mean(dim=1, keepdim=True)
    denom = (x_centered**2).sum(dim=1).clamp_min(1.0e-12)
    slopes = (x_centered * y_centered).sum(dim=1) / denom

    return {
        "log_delta_trials": log_deltas.detach().cpu().numpy(),
        "mean_log_delta": log_deltas.mean(dim=0).detach().cpu().numpy(),
        "std_log_delta": log_deltas.std(dim=0, unbiased=False).detach().cpu().numpy(),
        "per_trial_slope": slopes.detach().cpu().numpy(),
        "mean_growth_slope": float(slopes.mean().detach().cpu()),
        "perturbation_scale": float(perturbation_scale),
    }


class RasterPlot(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, split="valid", n_samples=3, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 3
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        assert split in ["train", "valid"]
        self.split = split
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get data samples from the dataloaders
        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)
        batch = next(iter(dataloader))
        # Move data to the right device
        batch = send_batch_to_device(batch, pl_module.device)
        # Compute model output
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=False,
        )
        # Discard the extra data - only the SessionBatches are relevant here
        batch = batch[0]
        # Log a few example outputs for each session
        # Convert everything to numpy
        encod_data = batch.encod_data.detach().cpu().numpy()
        recon_data = batch.recon_data.detach().cpu().numpy()
        truth = batch.truth.detach().cpu().numpy()
        means = output.output_params.detach().cpu().numpy()
        inputs = output.gen_inputs.detach().cpu().numpy()
        # Compute data sizes
        _, steps_encod, neur_encod = encod_data.shape
        _, steps_recon, neur_recon = recon_data.shape
        # Decide on how to plot panels
        if np.all(np.isnan(truth)):
            plot_arrays = [recon_data, means, inputs]
            height_ratios = [3, 3, 1]
        else:
            plot_arrays = [recon_data, truth, means, inputs]
            height_ratios = [3, 3, 3, 1]
        # Create subplots
        fig, axes = plt.subplots(
            len(plot_arrays),
            self.n_samples,
            sharex=True,
            sharey="row",
            figsize=(3 * self.n_samples, 10),
            gridspec_kw={"height_ratios": height_ratios},
        )
        for i, ax_col in enumerate(axes.T):
            for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                if j < len(plot_arrays) - 1:
                    ax.imshow(array[i].T, interpolation="none", aspect="auto")
                    ax.vlines(steps_encod, 0, neur_recon, color="orange")
                    ax.hlines(neur_encod, 0, steps_recon, color="orange")
                    ax.set_xlim(0, steps_recon)
                    ax.set_ylim(0, neur_recon)
                else:
                    ax.plot(array[i])
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            f"{self.split}/raster_plot",
            fig,
            trainer.global_step,
        )


class ChaoticDelayedMatchingRasterPlot(pl.Callback):
    """ChaoticDelayedMatching raster plot with true firing-rate panel."""

    def __init__(self, split="valid", n_samples=3, log_every_n_epochs=100):
        assert split in ["train", "valid"]
        self.split = split
        self.n_samples = n_samples
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)

        batch = next(iter(dataloader))
        batch = send_batch_to_device(batch, pl_module.device)
        output = pl_module.predict_step(
            batch=batch,
            batch_ix=None,
            sample_posteriors=False,
        )

        batch_data = batch[0]
        extra_data = batch[1]

        encod_data = batch_data.encod_data.detach().cpu().numpy()
        recon_data = batch_data.recon_data.detach().cpu().numpy()
        truth = batch_data.truth.detach().cpu().numpy()
        means = output.output_params.detach().cpu().numpy()
        inputs = output.gen_inputs.detach().cpu().numpy()
        true_rates = extra_data[2].detach().cpu().numpy()

        _, steps_encod, neur_encod = encod_data.shape
        _, steps_recon, neur_recon = recon_data.shape

        if np.all(np.isnan(truth)):
            plot_arrays = [recon_data, true_rates, means, inputs]
            labels = ["Spikes", "True rates", "Inferred rates", "Inputs"]
            height_ratios = [3, 3, 3, 1]
        else:
            plot_arrays = [recon_data, truth, true_rates, means, inputs]
            labels = ["Spikes", "Truth", "True rates", "Inferred rates", "Inputs"]
            height_ratios = [3, 3, 3, 3, 1]

        fig, axes = plt.subplots(
            len(plot_arrays),
            self.n_samples,
            sharex=True,
            sharey="row",
            figsize=(3 * self.n_samples, 12),
            gridspec_kw={"height_ratios": height_ratios},
        )
        if self.n_samples == 1:
            axes = np.array(axes)[:, None]

        for i, ax_col in enumerate(axes.T):
            for j, (ax, array) in enumerate(zip(ax_col, plot_arrays)):
                if j < len(plot_arrays) - 1:
                    ax.imshow(array[i].T, interpolation="none", aspect="auto")
                    ax.vlines(steps_encod, 0, neur_recon, color="orange")
                    ax.hlines(neur_encod, 0, steps_recon, color="orange")
                    ax.set_xlim(0, steps_recon)
                    ax.set_ylim(0, neur_recon)
                else:
                    ax.plot(array[i])
                if i == 0:
                    ax.set_title(f"Trial {i}")
                if i == 0:
                    ax.set_ylabel(labels[j])

        plt.tight_layout()
        log_figure(
            trainer.loggers,
            f"{self.split}/raster_plot_true_rates",
            fig,
            trainer.global_step,
        )


class TrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        dataloader = trainer.datamodule.train_dataloader()
        # Compute outputs and plot for one session at a time
        latents = []
        for batch in dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, pl_module.device)
            # Perform the forward pass through the model
            output = pl_module.predict_step(batch, None, sample_posteriors=False)
            latents.append(output.gen_states)
        latents = torch.cat(latents).detach().cpu().numpy()
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = latents.shape
        if n_lats > 3:
            latents_flat = latents.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            latents = pca.fit_transform(latents_flat)
            latents = latents.reshape(n_samp, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in latents:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*latents[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*latents[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "trajectory_plot",
            fig,
            trainer.global_step,
        )


class CondAvgTrajectoryPlot(pl.Callback):
    """Plots the top-3 PC's of the latent trajectory for
    all samples in the validation set and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return
        # Get only the validation dataloaders
        train_conds = trainer.datamodule.train_cond_idx
        dataloader = trainer.datamodule.train_dataloader()
        # Compute outputs and plot for one session at a time
        latents = []
        for batch in dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, pl_module.device)
            # Perform the forward pass through the model
            output = pl_module.predict_step(batch, None, sample_posteriors=False)
            latents.append(output.factors)
        latents = torch.cat(latents).detach().cpu().numpy()
        # Find the condition averaged trajectory
        latents_cond_avg = np.empty((0, latents.shape[1], latents.shape[2]))
        num_conds = len(np.unique(train_conds))
        for cond in np.unique(train_conds):
            cond_idx = np.where(train_conds == cond)[0]
            cond_latents = latents[cond_idx]
            cond_latents = cond_latents.mean(0, keepdims=True)
            latents_cond_avg = np.concatenate([latents_cond_avg, cond_latents], 0)
        # Reduce dimensionality if necessary
        n_samp, n_step, n_lats = latents.shape
        if n_lats > 3:
            latents_flat = latents_cond_avg.reshape(-1, n_lats)
            pca = PCA(n_components=3)
            latents_cond_avg = pca.fit_transform(latents_flat)
            latents_cond_avg = latents_cond_avg.reshape(num_conds, n_step, 3)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        else:
            explained_variance = 1.0
        # Create figure and plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in latents_cond_avg:
            ax.plot(*traj.T, alpha=1, linewidth=2)
        ax.scatter(*latents_cond_avg[:, 0, :].T, alpha=1, s=20, c="g")
        ax.scatter(*latents_cond_avg[:, -1, :].T, alpha=1, s=20, c="r")
        ax.set_title(f"explained variance: {explained_variance:.2f}")
        plt.tight_layout()
        # Log the figure
        log_figure(
            trainer.loggers,
            "trajectory_plot_cond_avg",
            fig,
            trainer.global_step,
        )


class DTMetricsCallback(pl.Callback):
    """Plots validation spiking data side-by-side with
    inferred inputs and rates and logs to tensorboard.
    """

    def __init__(self, log_every_n_epochs=100):
        """Initializes the callback.
        Parameters
        ----------
        n_samples : int, optional
            The number of samples to plot, by default 3
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.
        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Check for any image loggers
        if not has_image_loggers(trainer.loggers):
            return

        def bits_per_spike(preds, targets):
            """
            Computes BPS for n_samples x n_timesteps x n_neurons arrays.
            Preds are logrates and targets are binned spike counts.
            """
            nll_model = poisson_nll_loss(preds, targets, full=True, reduction="sum")
            nll_null = poisson_nll_loss(
                torch.mean(targets, dim=(0, 1), keepdim=True),
                targets,
                log_input=False,
                full=True,
                reduction="sum",
            )
            return (nll_null - nll_model) / torch.nansum(targets) / math.log(2)

        # Get data samples from the dataloaders
        dataloader = trainer.datamodule.val_dataloader()
        spikes_list = []
        latents_list = []
        true_latents_list = []
        inputs_list = []
        true_inputs_list = []
        rates_list = []
        true_rates_list = []

        for batch in dataloader:
            # Move data to the right device
            batch = send_batch_to_device(batch, pl_module.device)
            # Compute model output
            output = pl_module.predict_step(
                batch=batch,
                batch_ix=None,
                sample_posteriors=False,
            )
            # Discard the extra data - only the SessionBatches are relevant here
            batch_data = batch[0]
            extra_data = batch[1]
            # Log a few example outputs for each session
            # Convert everything to numpy
            encod_data = batch_data.encod_data.detach().cpu().numpy()
            hi_neurons = encod_data.shape[-1]
            recon_data = batch_data.recon_data.detach().cpu().numpy()
            means = output.output_params.detach().cpu().numpy()
            inputs = output.gen_inputs.detach().cpu().numpy()
            lat = output.gen_states.detach().cpu().numpy()

            spikes_list.append(recon_data)
            latents_list.append(lat)
            inputs_list.append(inputs)
            rates_list.append(means)
            true_inputs_list.append(extra_data[0].detach().cpu().numpy())
            true_latents_list.append(extra_data[1].detach().cpu().numpy())
            true_rates_list.append(extra_data[2].detach().cpu().numpy())

        spikes = np.concatenate(spikes_list, axis=0)
        latents = np.concatenate(latents_list, axis=0)
        inputs = np.concatenate(inputs_list, axis=0)
        rates = np.concatenate(rates_list, axis=0)
        true_inputs = np.concatenate(true_inputs_list, axis=0)
        true_latents = np.concatenate(true_latents_list, axis=0)
        true_rates = np.concatenate(true_rates_list, axis=0)

        spikes = spikes.reshape(-1, spikes.shape[-1])
        latents = latents.reshape(-1, latents.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-1])
        rates = rates.reshape(-1, rates.shape[-1])
        true_inputs = true_inputs.reshape(-1, true_inputs.shape[-1])
        true_latents = true_latents.reshape(-1, true_latents.shape[-1])
        true_rates = true_rates.reshape(-1, true_rates.shape[-1])

        metric_dict = compute_metrics(
            true_rates=true_rates,
            inf_rates=rates,
            true_latents=true_latents,
            inf_latents=latents,
            true_spikes=spikes,
            true_inputs=true_inputs,
            inf_inputs=inputs,
            n_heldin=hi_neurons,
            device=pl_module.device,
        )
        # Log the figure
        pl_module.log_dict(
            {
                **metric_dict,
            }
        )


class ChaoticDelayedMatchingLyapunovCallback(pl.Callback):
    """Log Lyapunov diagnostics for ChaoticDelayedMatching"""

    def __init__(
        self,
        split="valid",
        log_every_n_epochs: int = 10,
        max_trials: int = 8,
        warmup_steps: int = 50,
        max_steps: int = None,
        include_histogram: bool = True,
        include_divergence_plot: bool = True,
        max_plot_trials: int = 8,
        max_divergence_trials: int = 4,
    ):
        assert split in ["train", "valid"]
        self.split = split
        self.log_every_n_epochs = log_every_n_epochs
        self.max_trials = max_trials
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.include_histogram = include_histogram
        self.include_divergence_plot = include_divergence_plot
        self.max_plot_trials = max_plot_trials
        self.max_divergence_trials = max_divergence_trials
        self._epoch_history = []
        self._lyapunov_history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        can_log_images = has_image_loggers(trainer.loggers)

        if self.split == "valid":
            dataloader = trainer.datamodule.val_dataloader()
        else:
            dataloader = trainer.datamodule.train_dataloader(shuffle=False)

        batch = next(iter(dataloader))
        batch = send_batch_to_device(batch, pl_module.device)
        extra = _extract_trial_phase_metadata(batch[1]) if len(batch) > 1 else None
        gen_cell = _get_lfads_generator_cell(pl_module)
        if gen_cell is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        try:
            output = pl_module.predict_step(
                batch=batch,
                batch_ix=None,
                sample_posteriors=False,
            )
            gen_init = output.gen_init
            gen_inputs = output.gen_inputs

            metrics = _estimate_lfads_max_lyapunov_exponent(
                gen_cell=gen_cell,
                gen_init=gen_init,
                gen_inputs=gen_inputs,
                n_trials=self.max_trials,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                return_diagnostics=True,
            )
            if metrics is None:
                return

            time_series = np.asarray(metrics["time_series"])
            time_series_std = np.asarray(metrics.get("time_series_std"))
            per_trial_estimates = np.asarray(metrics.get("per_trial_estimates", []))

            pl_module.log(
                "valid/max_lyapunov_estimate",
                metrics["max_lyapunov_estimate"],
                on_epoch=True,
                prog_bar=True,
            )
            pl_module.log(
                "valid/max_lyapunov_std",
                metrics["max_lyapunov_std"],
                on_epoch=True,
            )
            pl_module.log(
                "valid/max_lyapunov_local_log_growth",
                metrics["local_log_growth_mean"],
                on_epoch=True,
            )
            if time_series.size > 0:
                pl_module.log(
                    "valid/max_lyapunov_positive_time_frac",
                    float(np.mean(time_series > 0.0)),
                    on_epoch=True,
                )
                pl_module.log(
                    "valid/max_lyapunov_trace_min",
                    float(np.min(time_series)),
                    on_epoch=True,
                )
                pl_module.log(
                    "valid/max_lyapunov_trace_max",
                    float(np.max(time_series)),
                    on_epoch=True,
                )

            divergence_metrics = None
            if self.include_divergence_plot:
                divergence_metrics = _estimate_lfads_perturbation_divergence(
                    gen_cell=gen_cell,
                    gen_init=gen_init,
                    gen_inputs=gen_inputs,
                    n_trials=self.max_divergence_trials,
                    warmup_steps=self.warmup_steps,
                    max_steps=self.max_steps,
                )
                if divergence_metrics is not None:
                    pl_module.log(
                        "valid/max_lyapunov_mean_perturbation_slope",
                        divergence_metrics["mean_growth_slope"],
                        on_epoch=True,
                    )

            self._epoch_history.append(int(trainer.current_epoch))
            self._lyapunov_history.append(float(metrics["max_lyapunov_estimate"]))

            if can_log_images:
                local_fig = _plot_local_log_growth(time_series, time_series_std)
                log_figure(
                    trainer.loggers,
                    "valid/chaos/local_log_growth_plot",
                    local_fig,
                    trainer.global_step,
                )

                if extra is not None:
                    phase_fig = _plot_local_log_growth_by_trial_phase(
                        time_series,
                        extra=extra,
                        warmup_steps=metrics["warmup_steps"],
                        std_series=time_series_std,
                    )
                    log_figure(
                        trainer.loggers,
                        "valid/chaos/local_lyapunov_by_trial_phase",
                        phase_fig,
                        trainer.global_step,
                    )

                if self.include_histogram and per_trial_estimates.size > 0:
                    hist_fig = _plot_lyapunov_histogram(per_trial_estimates)
                    log_figure(
                        trainer.loggers,
                        "valid/chaos/lyapunov_histogram",
                        hist_fig,
                        trainer.global_step,
                    )

                history_fig = _plot_lyapunov_history(
                    self._epoch_history, self._lyapunov_history
                )
                log_figure(
                    trainer.loggers,
                    "valid/chaos/lyapunov_vs_epoch",
                    history_fig,
                    trainer.global_step,
                )

                if self.include_divergence_plot and divergence_metrics is not None:
                    trial_curves = np.asarray(divergence_metrics["log_delta_trials"])[
                        : self.max_plot_trials
                    ]
                    div_fig = _plot_perturbation_divergence(
                        np.asarray(divergence_metrics["mean_log_delta"]),
                        np.asarray(divergence_metrics["std_log_delta"]),
                        trial_curves=trial_curves,
                    )
                    log_figure(
                        trainer.loggers,
                        "valid/chaos/perturbation_divergence",
                        div_fig,
                        trainer.global_step,
                    )
        finally:
            pl_module.train(was_training)
