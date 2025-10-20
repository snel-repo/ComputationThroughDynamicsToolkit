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
