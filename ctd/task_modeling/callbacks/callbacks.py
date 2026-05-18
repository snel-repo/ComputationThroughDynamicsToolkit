import io

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from scipy.signal import hilbert
from sklearn.decomposition import PCA


def phase_difference(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the phase offset between two sinusoids x and y of the same frequency.

    Returns
    -------
    delta : float
        Phase difference in radians, such that y(t) ≈ x(t) shifted by +delta.
    """
    # 1) form analytic signals
    ax = hilbert(x)
    ay = hilbert(y)

    # 2) instantaneous phase and unwrap
    phix = np.unwrap(np.angle(ax))
    phiy = np.unwrap(np.angle(ay))

    # 3) pointwise phase difference
    dphi = phiy - phix

    # 4) circular‐mean to get a single constant offset
    return np.angle(np.mean(np.exp(1j * dphi)))


def compute_phase_from_signal(r_input: np.ndarray) -> np.ndarray:
    """
    Compute the instantaneous phase θ(t) ∈ [0, 2π) from a batch of sinusoidal inputs.

    Args:
        r_input: np.ndarray of shape (B, T), values ≈ sin(θ(t)).

    Returns:
        phase: np.ndarray of shape (B, T), estimated θ(t) in radians ∈ [0, 2π).
    """
    # 1) Clamp to [-1,1] to avoid NaNs from arccos due to numerical noise
    x = np.clip(r_input, -1.0, 1.0)

    # 2) Principal angle from arccos: values in [0, π]
    angle0 = np.arccos(x)  # shape (B, T)

    # 3) Approximate temporal derivative to resolve ± ambiguity
    #    Prepend zeros so dr has same shape as x
    dx = np.diff(x, axis=1)
    dr = np.concatenate(
        [np.zeros((x.shape[0], 1), dtype=x.dtype), dx], axis=1
    )  # (B, T)

    # 4) Where the signal is rising (dr >= 0), phase = angle0; else phase = 2π - angle0
    two_pi = 2 * np.pi
    phase = np.where(dr >= 0, angle0, two_pi - angle0)

    return phase


def fig_to_rgb_array(fig):
    # Convert the figure to a numpy array for TB logging
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        fig_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = fig_data.reshape((int(h), int(w), -1))
    plt.close()
    return im


def get_wandb_logger(loggers):

    logger_list = loggers if isinstance(loggers, list) else [loggers]
    for logger in logger_list:
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            return logger.experiment
    else:
        return None


def _dnms_extra_indices(extra):
    """Support both legacy and updated DNMS extra layouts."""
    n_cols = extra.shape[1]
    if n_cols >= 13:
        return {
            "cue1_on": 0,
            "cue1_off": 1,
            "delay1_on": 2,
            "delay1_off": 3,
            "cue2_on": 4,
            "cue2_off": 5,
            "delay2_on": 6,
            "delay2_off": 7,
            "resp_on": 8,
            "resp_off": 9,
            "cue1_id": 10,
            "cue2_id": 11,
            "nonmatch": 12,
        }
    if n_cols >= 11:
        return {
            "cue1_on": 0,
            "cue1_off": 1,
            "delay1_on": 2,
            "delay1_off": 3,
            "cue2_on": 4,
            "cue2_off": 5,
            "delay2_on": None,
            "delay2_off": None,
            "resp_on": 6,
            "resp_off": 7,
            "cue1_id": 8,
            "cue2_id": 9,
            "nonmatch": 10,
        }
    raise ValueError(f"Unsupported DNMS extra shape: {tuple(extra.shape)}")


def _model_rates(model, latents):
    if hasattr(model, "_rates"):
        return model._rates(latents)
    if hasattr(model, "act"):
        return model.act(latents)
    return latents


def _compute_chaos_metrics(model, inputs, latents, perturbation_scale=1.0e-3):
    """Lightweight chaos diagnostics for tanh recurrent nets.

    Returns summary metrics based on:
    1. Growth of a small hidden-state perturbation under identical inputs.
    2. Mean local Jacobian log spectral norm along validation trajectories.
    """

    if not hasattr(model, "recW"):
        return None

    rec_w = model.recW.weight.detach()
    device = rec_w.device
    inputs = inputs.to(device)
    latents = latents.to(device)

    alpha = float(getattr(model, "alpha", 1.0))
    eye = torch.eye(rec_w.shape[0], device=device, dtype=rec_w.dtype)

    deriv = 1.0 - torch.tanh(latents).pow(2)
    log_sigmas = []
    positive_frac = []
    for trial_idx in range(latents.shape[0]):
        for time_idx in range(latents.shape[1]):
            jac = (1.0 - alpha) * eye + alpha * (
                rec_w * deriv[trial_idx, time_idx].unsqueeze(0)
            )
            sigma = torch.linalg.svdvals(jac)[0].clamp_min(1.0e-8)
            log_sigma = torch.log(sigma)
            log_sigmas.append(log_sigma)
            positive_frac.append((log_sigma > 0).float())

    base_hidden = model.init_hidden(batch_size=inputs.shape[0]).to(device)
    eps = torch.randn_like(base_hidden)
    eps = perturbation_scale * eps / eps.norm(dim=1, keepdim=True).clamp_min(1.0e-8)

    noise_level = float(getattr(model, "noise_level", 0.0))
    model.noise_level = 0.0

    def _roll(hidden0):
        hidden = hidden0.clone()
        traj = []
        for step_idx in range(inputs.shape[1]):
            _, hidden = model(inputs[:, step_idx, :], hidden)
            traj.append(hidden)
        return torch.stack(traj, dim=1)

    with torch.no_grad():
        lat_a = _roll(base_hidden)
        lat_b = _roll(base_hidden + eps)

    model.noise_level = noise_level

    delta = torch.linalg.norm(lat_b - lat_a, dim=-1).clamp_min(1.0e-8)
    growth = torch.log(delta / delta[:, :1].clamp_min(1.0e-8))

    rec_eigs = torch.linalg.eigvals(rec_w)
    rec_radius = torch.max(torch.abs(rec_eigs)).real
    log_sigmas = torch.stack(log_sigmas)
    positive_frac = torch.stack(positive_frac)

    return {
        "rec_spectral_radius": float(rec_radius.detach().cpu()),
        "mean_log_jacobian_norm": float(log_sigmas.mean().detach().cpu()),
        "frac_positive_log_jacobian": float(positive_frac.mean().detach().cpu()),
        "final_mean_log_divergence": float(growth[:, -1].mean().detach().cpu()),
        "max_mean_log_divergence": float(growth.mean(dim=0).max().detach().cpu()),
        "perturbation_scale": float(perturbation_scale),
    }


def _estimate_max_lyapunov_exponent(
    model,
    inputs,
    n_trials=8,
    warmup_steps=50,
    max_steps=None,
    return_diagnostics=False,
):
    """Estimate a finite-time maximum Lyapunov proxy for an input-driven RNN.

    Positive values indicate locally expanding dynamics on average, while
    negative values indicate contraction. This is a local, finite-time proxy,
    not an asymptotic Lyapunov exponent for the autonomous system.
    """

    if not hasattr(model, "init_hidden"):
        return None

    model_device = next(model.parameters()).device
    inputs = inputs[: min(n_trials, inputs.shape[0])].to(model_device)
    if inputs.shape[0] == 0:
        return None

    n_total_steps = inputs.shape[1]
    warmup_steps = min(int(warmup_steps), max(n_total_steps - 1, 0))
    n_steps = n_total_steps - warmup_steps
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if n_steps <= 0:
        return None

    was_training = model.training
    model.eval()
    noise_level = getattr(model, "noise_level", None)
    if noise_level is not None:
        model.noise_level = 0.0

    try:
        with torch.no_grad():
            hidden = model.init_hidden(batch_size=inputs.shape[0]).to(model_device)
            for step_idx in range(warmup_steps):
                _, hidden = model(inputs[:, step_idx, :], hidden)

        tangent = torch.randn_like(hidden)
        tangent = tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(
            1.0e-12
        )

        local_logs = []
        with torch.enable_grad():
            for step_idx in range(warmup_steps, warmup_steps + n_steps):
                hidden = hidden.detach().requires_grad_(True)
                _, next_hidden = model(inputs[:, step_idx, :], hidden)
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
            "n_trials": int(inputs.shape[0]),
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
    finally:
        if noise_level is not None:
            model.noise_level = noise_level
        model.train(was_training)


def _estimate_perturbation_divergence(
    model,
    inputs,
    n_trials=4,
    warmup_steps=50,
    max_steps=None,
    perturbation_scale=1.0e-5,
):
    """Track finite-time divergence of nearby hidden states under shared inputs.

    Upward trajectories suggest local expansion; downward trajectories suggest
    contraction. This is an input-conditioned perturbation diagnostic, not a
    formal global chaos proof.
    """

    if not hasattr(model, "init_hidden"):
        return None

    model_device = next(model.parameters()).device
    inputs = inputs[: min(n_trials, inputs.shape[0])].to(model_device)
    if inputs.shape[0] == 0:
        return None

    n_total_steps = inputs.shape[1]
    warmup_steps = min(int(warmup_steps), max(n_total_steps - 1, 0))
    n_steps = n_total_steps - warmup_steps
    if max_steps is not None:
        n_steps = min(n_steps, int(max_steps))
    if n_steps <= 0:
        return None

    was_training = model.training
    model.eval()
    noise_level = getattr(model, "noise_level", None)
    if noise_level is not None:
        model.noise_level = 0.0

    try:
        with torch.no_grad():
            hidden = model.init_hidden(batch_size=inputs.shape[0]).to(model_device)
            for step_idx in range(warmup_steps):
                _, hidden = model(inputs[:, step_idx, :], hidden)

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
                _, hidden_a = model(inputs[:, step_idx, :], hidden_a)
                _, hidden_b = model(inputs[:, step_idx, :], hidden_b)
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
            "std_log_delta": log_deltas.std(dim=0, unbiased=False)
            .detach()
            .cpu()
            .numpy(),
            "per_trial_slope": slopes.detach().cpu().numpy(),
            "mean_growth_slope": float(slopes.mean().detach().cpu()),
            "perturbation_scale": float(perturbation_scale),
        }
    finally:
        if noise_level is not None:
            model.noise_level = noise_level
        model.train(was_training)


def _plot_local_log_growth(time_series, std_series=None):
    """Plot mean local growth over time; values above zero indicate expansion."""

    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(time_series))
    ax.plot(t, time_series, linewidth=1.8)
    if std_series is not None:
        ax.fill_between(
            t, time_series - std_series, time_series + std_series, alpha=0.2
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Local log growth")
    ax.set_title("Finite-time local growth trace")
    fig.tight_layout()
    return fig


def _plot_local_log_growth_by_trial_phase(
    time_series, extra, warmup_steps=0, std_series=None
):
    """Plot mean local growth with DNMS trial phases overlaid.

    Phase boundaries are summarized across the sampled validation trials using
    median onset/offset times, then shifted by the warmup window used before the
    Lyapunov trace is evaluated.
    """

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = np.arange(len(time_series))
    ax.plot(t, time_series, linewidth=1.8, color="tab:blue")
    if std_series is not None:
        ax.fill_between(
            t, time_series - std_series, time_series + std_series, alpha=0.2
        )
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)

    if extra is not None and len(extra) > 0:
        extra = np.asarray(extra)
        extra_idx = _dnms_extra_indices(extra)
        phase_specs = [
            ("cue1", "cue1_on", "cue1_off", "tab:blue"),
            ("delay1", "delay1_on", "delay1_off", "tab:orange"),
            ("cue2", "cue2_on", "cue2_off", "tab:cyan"),
            ("delay2", "delay2_on", "delay2_off", "tab:red"),
            ("response", "resp_on", "resp_off", "tab:green"),
        ]
        for label, start_key, stop_key, color in phase_specs:
            if extra_idx.get(start_key) is None or extra_idx.get(stop_key) is None:
                continue
            start = int(np.round(np.median(extra[:, extra_idx[start_key]]))) - int(
                warmup_steps
            )
            stop = int(np.round(np.median(extra[:, extra_idx[stop_key]]))) - int(
                warmup_steps
            )
            start = max(0, start)
            stop = min(len(time_series), stop)
            if stop <= start:
                continue
            ax.axvspan(start, stop, color=color, alpha=0.12, label=label)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", ncol=3, fontsize=8)
    ax.set_xlabel("Time step within evaluated trial window")
    ax.set_ylabel("Local log growth")
    ax.set_title("Finite-time Lyapunov proxy by trial phase")
    fig.tight_layout()
    return fig


def _plot_lyapunov_histogram(per_trial_estimates):
    """Plot trialwise Lyapunov proxies; positive mass suggests more expanding trials."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        per_trial_estimates, bins=min(20, max(5, len(per_trial_estimates))), alpha=0.85
    )
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Per-trial Lyapunov estimate")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of finite-time Lyapunov estimates")
    fig.tight_layout()
    return fig


def _plot_lyapunov_history(epochs, values):
    """Plot the evolution of the finite-time Lyapunov proxy over training."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, values, linewidth=1.8)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max Lyapunov estimate")
    ax.set_title("Lyapunov proxy over training")
    fig.tight_layout()
    return fig


def _plot_perturbation_divergence(
    mean_log_delta, std_log_delta=None, trial_curves=None
):
    """Plot log-distance growth for nearby states; rising traces indicate divergence."""

    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(mean_log_delta))
    if trial_curves is not None:
        for curve in trial_curves:
            ax.semilogy(t, np.exp(curve), linewidth=0.9, alpha=0.35)
    ax.semilogy(t, np.exp(mean_log_delta), linewidth=1.8)
    if std_log_delta is not None:
        lower = np.exp(mean_log_delta - std_log_delta)
        upper = np.exp(mean_log_delta + std_log_delta)
        ax.fill_between(t, lower, upper, alpha=0.2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Perturbation distance")
    ax.set_title("Finite-time perturbation divergence")
    fig.tight_layout()
    return fig


def _log_wandb_figure(wandb_logger, key, fig, global_step):
    if wandb_logger is None:
        plt.close(fig)
        return
    try:
        wandb_logger.log({key: wandb.Image(fig), "global_step": global_step})
    finally:
        plt.close(fig)


def _extract_inputs_from_validation_batch(batch, device):
    """Extract inputs from a val batch if standard tuple layout"""

    if (
        isinstance(batch, (tuple, list))
        and len(batch) > 1
        and torch.is_tensor(batch[1])
    ):
        return batch[1].to(device)
    return None


def _extract_extra_from_validation_batch(batch):
    """Extract task timing metadata from a validation batch if available."""

    if (
        isinstance(batch, (tuple, list))
        and len(batch) > 5
        and torch.is_tensor(batch[5])
    ):
        return batch[5].detach().cpu().numpy()
    return None


class StateTransitionCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, plot_n_trials=5):

        self.log_every_n_epochs = log_every_n_epochs
        self.plot_n_trials = plot_n_trials

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        dataloader = trainer.datamodule.val_dataloader()
        ics = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        targets = torch.cat([batch[2] for batch in dataloader]).to(pl_module.device)

        logger = get_wandb_logger(trainer.loggers)
        # Pass the data through the model
        output_dict = pl_module.forward(ics, inputs)
        controlled = output_dict["controlled"]

        # Create plots for different cases
        fig, axes = plt.subplots(
            nrows=3,
            ncols=self.plot_n_trials,
            figsize=(8 * self.plot_n_trials, 6),
            sharex=True,
        )
        for trial_num in range(self.plot_n_trials):
            ax1 = axes[0][trial_num]
            ax2 = axes[1][trial_num]
            ax3 = axes[2][trial_num]

            targets = targets.cpu()
            inputs = inputs.cpu()
            pred_outputs = controlled.cpu()
            n_samples, n_timesteps, n_outputs = targets.shape
            input_labels = trainer.datamodule.input_labels
            output_labels = trainer.datamodule.output_labels

            for i in range(n_outputs):
                ax1.plot(targets[trial_num, :, i], label=output_labels[i])
            ax1.legend(loc="right")
            ax1.set_ylabel("Actual Outputs")

            for i in range(n_outputs):
                ax2.plot(pred_outputs[trial_num, :, i], label=output_labels[i])
            ax2.set_ylabel("Predicted Outputs")
            ax2.legend(loc="right")

            _, _, n_inputs = inputs.shape
            for i in range(n_inputs):
                ax3.plot(inputs[trial_num, :, i], label=input_labels[i])

            ax3.legend(loc="right")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Inputs")

            plt.tight_layout()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        trainer.loggers[0].experiment.add_image(
            "state_plot", im, trainer.global_step, dataformats="HWC"
        )
        logger.log({"state_plot": wandb.Image(fig), "global_step": trainer.global_step})


class TrajectoryPlotOverTimeCallback(pl.Callback):
    def __init__(self, log_every_n_epochs=100, num_trials_to_plot=5, axis_num=0):

        self.log_every_n_epochs = log_every_n_epochs
        self.num_trials_to_plot = num_trials_to_plot
        self.axis_num = axis_num

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get trajectories and model predictions
        dataloader = trainer.datamodule.val_dataloader()
        ics = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        output_dict = pl_module.forward(ics, inputs)
        trajs_out = output_dict["controlled"]
        logger = get_wandb_logger(trainer.loggers)

        # Plot the true and predicted trajectories
        trial_vec = torch.tensor(
            np.random.randint(0, trajs_out.shape[0], self.num_trials_to_plot)
        )
        fig, ax = plt.subplots()
        traj_in = 1  # TODO: Fix this
        t1 = np.linspace(0, 1, len(trial_vec) * trajs_out.shape[1])

        def prep_trajs(x):
            return x[trial_vec, :, self.axis_num].detach().cpu().numpy().flatten()

        ax.plot(t1, prep_trajs(traj_in), "C0", label="Actual Traj")
        ax.plot(t1, np.exp(prep_trajs(trajs_out)), "C1", label="Pred Traj")
        ax.set_xlabel("Time (AU)")
        ax.set_ylabel("Firing (AU)")
        ax.set_title(f"axis {self.axis_num}, {self.num_trials_to_plot} trials")
        ax.legend()
        # Log the plot to tensorboard
        im = fig_to_rgb_array(fig)
        logger.add_image(
            "trajectory_plot_over_time", im, trainer.global_step, dataformats="HWC"
        )


class LatentTrajectoryPlot(pl.Callback):
    def __init__(
        self,
        log_every_n_epochs=10,
    ):
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        logger = get_wandb_logger(trainer.loggers)

        # Get trajectories and model predictions
        train_dataloader = trainer.datamodule.train_dataloader()
        ics_train = torch.cat([batch[0] for batch in train_dataloader]).to(
            pl_module.device
        )
        inputs_train = torch.cat([batch[1] for batch in train_dataloader]).to(
            pl_module.device
        )
        output_dict = pl_module.forward(ics_train, inputs_train)
        lats_train = output_dict["latents"]

        lats_train = lats_train.detach().cpu().numpy()
        if not np.isfinite(lats_train).all():
            pl_module.log(
                "valid/latent_has_nan",
                1.0,
                on_epoch=True,
                prog_bar=True,
            )
            return
        pl_module.log("valid/latent_has_nan", 0.0, on_epoch=True, prog_bar=False)
        n_trials, n_times, n_lat_dim = lats_train.shape
        if n_lat_dim > 3:
            pca1 = PCA(n_components=3)
            lats_train = pca1.fit_transform(lats_train.reshape(-1, n_lat_dim))
            lats_train = lats_train.reshape(n_trials, n_times, 3)
            exp_var = np.sum(pca1.explained_variance_ratio_)
        else:
            exp_var = 1.0

        # Plot trajectories
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for traj in lats_train:
            ax.plot(*traj.T, alpha=0.2, linewidth=0.5)
        ax.scatter(*lats_train[:, 0, :].T, alpha=0.1, s=10, c="g")
        ax.scatter(*lats_train[:, -1, :].T, alpha=0.1, s=10, c="r")
        ax.set_title(f"explained variance: {exp_var:.2f}")
        plt.tight_layout()
        trainer.loggers[0].experiment.add_figure(
            "latent_trajectory", fig, global_step=trainer.global_step
        )
        logger.log(
            {"latent_traj": wandb.Image(fig), "global_step": trainer.global_step}
        )


class ChaoticDelayedMatchingPCTimeCallback(pl.Callback):
    """Plot the top N latent PCs over time, colored by cue pair."""

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        n_pcs: int = 6,
        max_trials_per_condition: int = 20,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.n_pcs = n_pcs
        self.max_trials_per_condition = max_trials_per_condition

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        logger = get_wandb_logger(trainer.loggers)
        dataloader = trainer.datamodule.val_dataloader()
        ics = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        inputs = torch.cat([batch[1] for batch in dataloader]).to(pl_module.device)
        conds = torch.cat([batch[4] for batch in dataloader]).cpu().numpy()

        with torch.no_grad():
            latents = pl_module.forward(ics, inputs)["latents"]

        latents = latents.detach().cpu().numpy()
        if not np.isfinite(latents).all():
            pl_module.log(
                "valid/dnms_pc_time_has_nan", 1.0, on_epoch=True, prog_bar=True
            )
            return
        pl_module.log("valid/dnms_pc_time_has_nan", 0.0, on_epoch=True, prog_bar=False)

        n_trials, n_times, n_lat_dim = latents.shape
        n_pcs = min(self.n_pcs, n_lat_dim)
        pca = PCA(n_components=n_pcs)
        lats_pc = pca.fit_transform(latents.reshape(-1, n_lat_dim))
        lats_pc = lats_pc.reshape(n_trials, n_times, n_pcs)

        cond_labels = {
            (0, 0): ("AA", "tab:blue"),
            (0, 1): ("AB", "tab:orange"),
            (1, 0): ("BA", "tab:green"),
            (1, 1): ("BB", "tab:red"),
        }

        fig, axes = plt.subplots(
            nrows=n_pcs,
            ncols=1,
            figsize=(12, 2.2 * n_pcs),
            sharex=True,
        )
        if n_pcs == 1:
            axes = [axes]

        t = np.arange(n_times)
        for pc_idx, ax in enumerate(axes):
            for cond_key, (label, color) in cond_labels.items():
                mask = (conds[:, 0] == cond_key[0]) & (conds[:, 1] == cond_key[1])
                trial_inds = np.flatnonzero(mask)[: self.max_trials_per_condition]
                for j, trial_idx in enumerate(trial_inds):
                    ax.plot(
                        t,
                        lats_pc[trial_idx, :, pc_idx],
                        color=color,
                        alpha=0.25,
                        linewidth=0.9,
                        label=label if j == 0 and pc_idx == 0 else None,
                    )
            ax.set_ylabel(f"PC{pc_idx + 1}")
            ax.set_title(
                f"PC{pc_idx + 1} | var={pca.explained_variance_ratio_[pc_idx]:.3f}"
            )

        axes[0].legend(loc="upper right", ncol=4)
        axes[-1].set_xlabel("Time")
        fig.suptitle("ChaoticDelayedMatching latent PCs over time by cue pair")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        logger_list = (
            trainer.loggers if isinstance(trainer.loggers, list) else [trainer.loggers]
        )
        for tb_logger in logger_list:
            if hasattr(tb_logger, "experiment") and hasattr(
                tb_logger.experiment, "add_figure"
            ):
                tb_logger.experiment.add_figure(
                    "chaotic_delayed_matching/pc_time",
                    fig,
                    global_step=trainer.global_step,
                )

        if logger is not None:
            logger.log(
                {
                    "chaotic_delayed_matching_pc_time": wandb.Image(fig),
                    "global_step": trainer.global_step,
                }
            )
        plt.close(fig)


class PhaseCodedMemoryCallback(pl.Callback):
    """
    Every `log_every_n_epochs`, plot I/O for one trial and log overall MSE.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 100,
        trial_idx: int = None,
        input_axis: int = 0,
        output_axis: int = 0,
    ):
        """
        Args:
            log_every_n_epochs: how often (in epochs) to run this plot.
            trial_idx: which trial to plot (0-based).
                If None, picks a random one each time.
            input_axis: which input channel to plot.
            output_axis: which output channel to plot.
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.trial_idx = trial_idx
        self.input_axis = input_axis
        self.output_axis = output_axis

    def on_validation_epoch_end(self, trainer, pl_module):
        # only run on the specified epochs
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        # 1) Gather entire validation set
        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))  # one batch

        ics = batch[0].to(pl_module.device)
        inputs = batch[1].to(pl_module.device)
        targets = batch[2].to(pl_module.device)

        # 2) Forward pass
        with torch.no_grad():
            output_dict = pl_module.forward(ics, inputs)
            preds = output_dict["controlled"]  # shape (B, T, D_out)

        # 3) Compute overall MSE performance
        mse = torch.mean((preds - targets) ** 2).item()

        # 4) Select trial to plot
        B, T, _ = preds.shape
        if self.trial_idx is None:
            idx = np.random.randint(0, B)
        else:
            idx = np.clip(self.trial_idx, 0, B - 1)

        inp_seq = inputs[idx, :, self.input_axis].detach().cpu().numpy()
        inp_phase = (
            inputs[idx, :, 1].detach().cpu().numpy()
        )  # assuming phase is in second channel
        inp_phase2 = (
            inputs[idx, :, 2].detach().cpu().numpy()
        )  # assuming second phase channel if exists
        true_seq = targets[idx, :, self.output_axis].detach().cpu().numpy()
        pred_seq = preds[idx, :, self.output_axis].detach().cpu().numpy()

        # 5) Make the figure
        t = np.arange(T)
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
        axs[0].plot(t, inp_seq, label=f"Input ch {self.input_axis}")
        axs[0].set_ylabel("Input")
        axs[0].plot(t, inp_phase, label="Phase", linestyle="--", color="orange")
        if inp_phase2 is not None:
            axs[0].plot(t, inp_phase2, label="Phase 2", linestyle=":", color="green")
        axs[0].legend()

        axs[1].plot(t, true_seq, label=f"True output ch {self.output_axis}")
        axs[1].set_ylabel("Target")
        axs[1].legend()

        axs[2].plot(t, pred_seq, label=f"Pred output ch {self.output_axis}")
        axs[2].set_ylabel("Prediction")
        axs[2].set_xlabel("Time step")
        axs[2].legend()

        plt.suptitle(
            f"Trial {idx} | Val MSE: {mse:.4f} | Epoch {trainer.current_epoch}"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle

        # 6) Log to WandB
        logger = get_wandb_logger(trainer.loggers)

        logger.log(
            {"PhaseCodedMemFig2": wandb.Image(fig), "global_step": trainer.global_step}
        )

        plt.close(fig)


class PhaseVsRateCodingCallback(pl.Callback):
    """
    Every `log_every_n_epochs`, plot I/O for one trial and log overall MSE.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 100,
    ):
        """
        Args:
            log_every_n_epochs: how often (in epochs) to run this plot.
            trial_idx: which trial to plot (0-based).
                If None, picks a random one each time.
            input_axis: which input channel to plot.
            output_axis: which output channel to plot.
        """
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        # only run on the specified epochs
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        # 1) Gather entire validation set
        dataloader = trainer.datamodule.val_dataloader()

        ics = torch.cat([batch[0] for batch in dataloader]).to(pl_module.device)
        extra = torch.cat([batch[5] for batch in dataloader]).to(pl_module.device)
        true_inputs = torch.cat([batch[7] for batch in dataloader]).to(pl_module.device)

        true_inputs_inverse = true_inputs.clone()
        true_inputs_inverse[:, :, 1] = true_inputs[:, :, 2]
        true_inputs_inverse[:, :, 2] = true_inputs[:, :, 1]
        # 2) Forward pass
        with torch.no_grad():
            output_dict = pl_module.forward(ics, true_inputs)
            output_dict_inv = pl_module.forward(ics, true_inputs_inverse)
            lats = output_dict["latents"]
            lats_inv = output_dict_inv["latents"]

        # Generate a mask for just the post-stimulus period
        mask = torch.zeros(
            lats.shape[0], lats.shape[1], dtype=torch.bool, device=lats.device
        )
        for i in range(lats.shape[0]):
            mask[i, extra[i, 1].long() :] = True
        # Zero out the latents before the mask
        # zero out pre‐stim
        n_samples = torch.sum(mask, dim=1).unsqueeze(dim=1)
        lats_zerod = lats.clone()
        lats_zerod[~mask] = 0.0
        lats_inv_zerod = lats_inv.clone()
        lats_inv_zerod[~mask] = 0.0

        # Compute per‐channel mean & std over the post‐stimulus period (mask==True),
        # pooling both original and inverted latents for shared statistics.
        comb_lats = torch.cat([lats_zerod, lats_inv_zerod], dim=0)  # [2*B, T, D]
        comb_mask = torch.cat([mask, mask], dim=0)  # [2*B, T]
        comb_mask_exp = comb_mask.unsqueeze(-1)  # [2*B, T, 1]

        # Mask out pre‐stim values by setting them to NaN
        comb_lats_masked = comb_lats.clone()
        comb_lats_masked = comb_lats.masked_fill(~comb_mask_exp, float("nan"))
        # 2) channel-wise mean over the valid (post-stim) entries
        mean_lats = torch.nanmean(comb_lats_masked, dim=(0, 1), keepdim=True)  # [1,1,D]

        # 3) channel-wise mean of squares
        mean2_lats = torch.nanmean(
            comb_lats_masked**2, dim=(0, 1), keepdim=True
        )  # [1,1,D]

        # 4) var = E[x^2] - (E[x])^2, then sqrt → std
        var_lats = mean2_lats - mean_lats**2
        std_lats = torch.sqrt(var_lats)

        # 5) z-score both original & inverted latents
        lats_norm = (lats_zerod - mean_lats) / std_lats
        lats_inv_norm = (lats_inv_zerod - mean_lats) / std_lats

        mean_lat_trials = torch.sum(lats_norm, dim=1) / n_samples
        mean_lat_trials_inv = torch.sum(lats_inv_norm, dim=1) / n_samples

        mean_lat_trials = mean_lat_trials.detach().cpu().numpy()
        mean_lat_trials_inv = mean_lat_trials_inv.detach().cpu().numpy()

        abs_dif_lats = np.abs(mean_lat_trials - mean_lat_trials_inv)
        mean_abs_dif_lats = np.mean(abs_dif_lats)
        angle_diff_list = []
        for i in range(lats_norm.shape[0]):
            for j in range(lats_norm.shape[2]):
                angle_diff_list.append(
                    phase_difference(
                        lats_norm[i, extra[i, 1].long() :, j].detach().cpu().numpy(),
                        lats_inv_norm[i, extra[i, 1].long() :, j]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
        mean_angle_diff = np.mean(np.abs(angle_diff_list))
        metric_dict = {
            "rate_code_val": mean_abs_dif_lats,
            "phase_code_val": mean_angle_diff,
        }
        # Log the figure
        pl_module.log_dict(
            {
                **metric_dict,
            }
        )


class ChaoticDelayedMatchingIOCallback(pl.Callback):
    """Validation callback for ChaoticDelayedMatching I/O quality."""

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        n_trials_to_plot: int = 4,
        output_axis: int = 0,
        n_neurons_to_plot: int = 24,
        neuron_selection: str = "chaotic",
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.n_trials_to_plot = n_trials_to_plot
        self.output_axis = output_axis
        self.n_neurons_to_plot = n_neurons_to_plot
        self.neuron_selection = neuron_selection

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))

        ics = batch[0].to(pl_module.device)
        inputs = batch[1].to(pl_module.device)
        targets = batch[2].to(pl_module.device)
        extra = batch[5].to(pl_module.device)

        with torch.no_grad():
            out = pl_module.forward(ics, inputs)
        preds = out["controlled"]
        latents = out["latents"]
        extra_idx = _dnms_extra_indices(extra)

        mae = torch.mean(torch.abs(preds - targets)).item()
        pred_sign = torch.sign(preds)
        target_sign = torch.sign(targets)
        response_mask = torch.zeros_like(targets)
        for i in range(preds.shape[0]):
            rs = int(extra[i, extra_idx["resp_on"]].item())
            re = int(extra[i, extra_idx["resp_off"]].item())
            response_mask[i, rs:re, :] = 1.0
        masked = response_mask > 0
        acc = torch.mean((pred_sign[masked] == target_sign[masked]).float()).item()

        pl_module.log("valid/dnms_io_mae", mae, on_epoch=True, prog_bar=True)
        pl_module.log("valid/dnms_resp_acc", acc, on_epoch=True, prog_bar=True)

        bsz, n_time, _ = preds.shape
        n_plot = int(np.clip(self.n_trials_to_plot, 1, bsz))
        t = np.arange(n_time)
        out_ax = int(np.clip(self.output_axis, 0, preds.shape[-1] - 1))
        n_latents = latents.shape[-1]
        n_neurons = int(np.clip(self.n_neurons_to_plot, 1, n_latents))
        readout_idx = getattr(
            getattr(pl_module.model, "readout", None), "neuron_idx", None
        )

        rates = _model_rates(pl_module.model, latents)
        rates_cpu = rates[:n_plot].detach().cpu()
        rate_var = rates_cpu.var(dim=(0, 1))
        if self.neuron_selection == "chaotic":
            rate_diff = torch.diff(rates_cpu, dim=1)
            mean_abs_diff = rate_diff.abs().mean(dim=(0, 1))
            diff_sign = torch.sign(rate_diff)
            sign_flips = (
                (diff_sign[:, 1:, :] != diff_sign[:, :-1, :]).float().mean(dim=(0, 1))
            )
            neuron_score = mean_abs_diff + sign_flips + 0.25 * rate_var
        else:
            neuron_score = rate_var
        neuron_inds = (
            torch.topk(neuron_score, k=n_neurons).indices.cpu().numpy().tolist()
        )
        if readout_idx is not None and readout_idx not in neuron_inds:
            neuron_inds = [int(readout_idx)] + neuron_inds[: n_neurons - 1]
        neuron_inds = np.array(neuron_inds, dtype=int)

        fig, axes = plt.subplots(
            nrows=n_plot,
            ncols=3,
            figsize=(17, 2.9 * n_plot),
            sharex=True,
        )
        if n_plot == 1:
            axes = np.array([axes])

        for i in range(n_plot):
            cue1_on = int(extra[i, extra_idx["cue1_on"]].item())
            cue1_off = int(extra[i, extra_idx["cue1_off"]].item())
            delay1_on = int(extra[i, extra_idx["delay1_on"]].item())
            delay1_off = int(extra[i, extra_idx["delay1_off"]].item())
            cue2_on = int(extra[i, extra_idx["cue2_on"]].item())
            cue2_off = int(extra[i, extra_idx["cue2_off"]].item())
            resp_on = int(extra[i, extra_idx["resp_on"]].item())
            resp_off = int(extra[i, extra_idx["resp_off"]].item())
            cue1_id = int(extra[i, extra_idx["cue1_id"]].item())
            cue2_id = int(extra[i, extra_idx["cue2_id"]].item())
            nonmatch = int(extra[i, extra_idx["nonmatch"]].item())

            ax_out = axes[i, 0]
            ax_in = axes[i, 1]
            ax_rates = axes[i, 2]

            ax_out.plot(
                t,
                targets[i, :, out_ax].detach().cpu().numpy(),
                label="target",
                linewidth=1.8,
            )
            ax_out.plot(
                t,
                preds[i, :, out_ax].detach().cpu().numpy(),
                label="prediction",
                linewidth=1.2,
                alpha=0.9,
            )
            ax_out.set_ylabel(f"trial {i}")
            ax_out.set_title(
                f"Output | cue1={cue1_id} cue2={cue2_id} nonmatch={nonmatch}"
            )
            ax_out.legend(loc="upper right")

            ax_in.plot(t, inputs[i].detach().cpu().numpy(), linewidth=1.0)
            ax_in.set_title("Task inputs")
            ax_in.set_ylabel("input")

            for neuron_idx in neuron_inds:
                trace = rates_cpu[i, :, neuron_idx].numpy()
                if neuron_idx == readout_idx:
                    ax_rates.plot(
                        t,
                        trace,
                        color="black",
                        linewidth=2.4,
                        alpha=0.95,
                        zorder=3,
                        label="output neuron" if i == 0 else None,
                    )
                else:
                    ax_rates.plot(
                        t,
                        trace,
                        color="tab:gray",
                        linewidth=0.9,
                        alpha=0.55,
                        zorder=2,
                    )
            ax_rates.set_title(
                f"Neuron activity ({n_neurons} {self.neuron_selection} units)"
            )
            ax_rates.set_ylabel("rate")
            if i == 0 and readout_idx is not None:
                ax_rates.legend(loc="upper right")

            for ax in (ax_out, ax_in, ax_rates):
                ax.axvspan(cue1_on, cue1_off, color="tab:blue", alpha=0.12)
                ax.axvspan(delay1_on, delay1_off, color="tab:orange", alpha=0.10)
                ax.axvspan(cue2_on, cue2_off, color="tab:purple", alpha=0.10)
                if extra_idx["delay2_on"] is not None:
                    delay2_on = int(extra[i, extra_idx["delay2_on"]].item())
                    delay2_off = int(extra[i, extra_idx["delay2_off"]].item())
                    ax.axvspan(delay2_on, delay2_off, color="tab:red", alpha=0.08)
                ax.axvspan(resp_on, resp_off, color="tab:green", alpha=0.12)

        axes[-1, 0].set_xlabel("Time")
        axes[-1, 1].set_xlabel("Time")
        axes[-1, 2].set_xlabel("Time")
        fig.suptitle(
            f"ChaoticDelayedMatching validation I/O | epoch={trainer.current_epoch} "
            f"| mae={mae:.4e} | resp_acc={acc:.3f}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        logger_list = (
            trainer.loggers if isinstance(trainer.loggers, list) else [trainer.loggers]
        )
        for logger in logger_list:
            if hasattr(logger, "experiment") and hasattr(
                logger.experiment, "add_figure"
            ):
                logger.experiment.add_figure(
                    "chaotic_delayed_matching/io_match",
                    fig,
                    global_step=trainer.global_step,
                )

        wandb_logger = get_wandb_logger(trainer.loggers)
        if wandb_logger is not None:
            wandb_logger.log(
                {
                    "chaotic_delayed_matching_io_match": wandb.Image(fig),
                    "global_step": trainer.global_step,
                }
            )
        plt.close(fig)


class ChaoticDelayedMatchingPerformanceCallback(pl.Callback):
    """Validation metrics for the ChaoticDelayedMatching success criterion."""

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        success_error_threshold: float = 1.0,
        criterion_window: int = 100,
        criterion_successes: int = 95,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.success_error_threshold = success_error_threshold
        self.criterion_window = criterion_window
        self.criterion_successes = criterion_successes

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        dataloader = trainer.datamodule.val_dataloader()
        trial_errors = []

        with torch.no_grad():
            for batch in dataloader:
                ics = batch[0].to(pl_module.device)
                inputs = batch[1].to(pl_module.device)
                targets = batch[2].to(pl_module.device)
                extra = batch[5].to(pl_module.device)
                inputs_to_env = batch[6].to(pl_module.device)
                extra_idx = _dnms_extra_indices(extra)

                preds = pl_module.forward(ics, inputs, inputs_to_env=inputs_to_env)[
                    "controlled"
                ]
                for i in range(preds.shape[0]):
                    rs = int(extra[i, extra_idx["resp_on"]].item())
                    re = int(extra[i, extra_idx["resp_off"]].item())
                    err = torch.mean(
                        torch.abs(preds[i, rs:re, 0] - targets[i, rs:re, 0])
                    )
                    trial_errors.append(float(err.detach().cpu()))

        if len(trial_errors) == 0:
            return

        trial_errors = np.asarray(trial_errors, dtype=np.float32)
        success_trials = trial_errors < self.success_error_threshold
        success_pct = 100.0 * float(success_trials.mean())

        best_window_successes = float(np.sum(success_trials))
        criterion_met = 0.0
        if len(success_trials) >= self.criterion_window:
            window = np.ones(self.criterion_window, dtype=np.float32)
            rolling_successes = np.convolve(
                success_trials.astype(np.float32), window, mode="valid"
            )
            best_window_successes = float(np.max(rolling_successes))
            criterion_met = float(best_window_successes >= self.criterion_successes)

        pl_module.log(
            "valid/dnms_success_pct",
            success_pct,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "valid/dnms_best_100_successes",
            best_window_successes,
            on_epoch=True,
        )
        pl_module.log(
            "valid/dnms_95_of_100_met",
            criterion_met,
            on_epoch=True,
            prog_bar=True,
        )


class ChaoticDelayedMatchingChaosCallback(pl.Callback):
    """Log simple chaos proxies during validation for ChaoticDelayedMatching."""

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        perturbation_scale: float = 1.0e-3,
        max_trials: int = 16,
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.perturbation_scale = perturbation_scale
        self.max_trials = max_trials

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        ics = batch[0].to(pl_module.device)
        inputs = batch[1].to(pl_module.device)

        n_trials = min(self.max_trials, inputs.shape[0])
        ics = ics[:n_trials]
        inputs = inputs[:n_trials]

        with torch.no_grad():
            out = pl_module.forward(ics, inputs)

        metrics = _compute_chaos_metrics(
            model=pl_module.model,
            inputs=inputs,
            latents=out["latents"],
            perturbation_scale=self.perturbation_scale,
        )
        if metrics is None:
            return

        pl_module.log(
            "valid/chaos_rec_radius", metrics["rec_spectral_radius"], on_epoch=True
        )
        pl_module.log(
            "valid/chaos_mean_log_jacobian",
            metrics["mean_log_jacobian_norm"],
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "valid/chaos_frac_expanding",
            metrics["frac_positive_log_jacobian"],
            on_epoch=True,
        )
        pl_module.log(
            "valid/chaos_final_log_divergence",
            metrics["final_mean_log_divergence"],
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "valid/chaos_peak_log_divergence",
            metrics["max_mean_log_divergence"],
            on_epoch=True,
        )


class ChaoticDelayedMatchingLyapunovCallback(pl.Callback):
    """Log finite-time Lyapunov diagnostics for an input-driven TT network.

    Positive values suggest locally expanding dynamics, while negative values
    suggest local contraction. This remains a local, finite-time proxy rather
    than a strict asymptotic Lyapunov exponent.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        max_trials: int = 8,
        warmup_steps: int = 50,
        max_steps: int = None,
        include_histogram: bool = True,
        include_divergence_plot: bool = True,
        max_plot_trials: int = 8,
        max_divergence_trials: int = 4,
    ):
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

        dataloader = trainer.datamodule.val_dataloader()
        batch = next(iter(dataloader))
        inputs = _extract_inputs_from_validation_batch(batch, pl_module.device)
        extra = _extract_extra_from_validation_batch(batch)
        if inputs is None:
            return

        metrics = _estimate_max_lyapunov_exponent(
            model=pl_module.model,
            inputs=inputs,
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
            divergence_metrics = _estimate_perturbation_divergence(
                model=pl_module.model,
                inputs=inputs,
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

        wandb_logger = get_wandb_logger(trainer.loggers)
        if wandb_logger is None:
            return

        local_fig = _plot_local_log_growth(time_series, time_series_std)
        _log_wandb_figure(
            wandb_logger,
            "valid/chaos/local_log_growth_plot",
            local_fig,
            trainer.global_step,
        )

        phase_fig = _plot_local_log_growth_by_trial_phase(
            time_series,
            extra=extra,
            warmup_steps=metrics["warmup_steps"],
            std_series=time_series_std,
        )
        _log_wandb_figure(
            wandb_logger,
            "valid/chaos/local_lyapunov_by_trial_phase",
            phase_fig,
            trainer.global_step,
        )

        if self.include_histogram and per_trial_estimates.size > 0:
            hist_fig = _plot_lyapunov_histogram(per_trial_estimates)
            _log_wandb_figure(
                wandb_logger,
                "valid/chaos/lyapunov_histogram",
                hist_fig,
                trainer.global_step,
            )

        self._epoch_history.append(int(trainer.current_epoch))
        self._lyapunov_history.append(float(metrics["max_lyapunov_estimate"]))
        history_fig = _plot_lyapunov_history(
            self._epoch_history, self._lyapunov_history
        )
        _log_wandb_figure(
            wandb_logger,
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
            _log_wandb_figure(
                wandb_logger,
                "valid/chaos/perturbation_divergence",
                div_fig,
                trainer.global_step,
            )
