import io
import os

import imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from sklearn.decomposition import PCA

# Helper functions for the MotorNetVideoGenerationArm callback


def calculate_arm_positions(
    shoulder_angle, elbow_angle, upper_arm_length=1, forearm_length=1
):
    """
    Calculate the positions of the shoulder, elbow, and hand
    given the angles of the shoulder and elbow.

    This function assumes that the shoulder is at the origin (0, 0) and the
    geometry of the arm is the standard 2-segment model.
    """
    # Calculate elbow position relative to shoulder
    elbow_x = np.cos(shoulder_angle) * upper_arm_length
    elbow_y = np.sin(shoulder_angle) * upper_arm_length

    # Calculate hand position relative to elbow, considering the elbow angle
    hand_x = elbow_x + np.cos(shoulder_angle + elbow_angle) * forearm_length
    hand_y = elbow_y + np.sin(shoulder_angle + elbow_angle) * forearm_length

    return np.array([0, 0]), np.array([elbow_x, elbow_y]), np.array([hand_x, hand_y])


def draw_muscle_activation(ax, center, radius, start_angle, end_angle, color):
    """
    Draws an arc to represent muscle activation around a joint.
    """
    arc = patches.Arc(
        center,
        radius * 2,
        radius * 2,
        angle=0,
        theta1=start_angle,
        theta2=end_angle,
        color=color,
        linewidth=3,
    )
    ax.add_patch(arc)


def draw_biarticular_muscle(ax, start_pos, end_pos, isFlexor, color):
    """
    Draws a line to represent biarticular muscle activation, correctly shifted
    to indicate whether it is a flexor or extensor.
    """
    # Calculate the direction vector from start to end
    direction = end_pos - start_pos
    # Calculate the perpendicular vector
    perpendicular = np.array([-direction[1], direction[0]])

    if isFlexor:
        # Rotate the perpendicular vector CCW for flexors to shift the line to the left
        shift_vector = perpendicular * 0.1  # Scale the shift magnitude
    else:
        # Rotate the perpendicular vector CW for extensors
        # to shift the line to the right
        shift_vector = -perpendicular * 0.1  # Scale the shift magnitude

    # Apply the shift
    shifted_start_pos = start_pos + shift_vector
    shifted_end_pos = end_pos + shift_vector

    ax.plot(
        [shifted_start_pos[0], shifted_end_pos[0]],
        [shifted_start_pos[1], shifted_end_pos[1]],
        color=color,
        linewidth=2,
    )


def get_activation_color(activation_level):
    """
    Returns a color based on the activation level using a colormap.
    """
    cmap = plt.get_cmap("rainbow")
    return cmap(activation_level)


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


class LatentTrajectoryPlot(pl.Callback):
    def __init__(
        self,
        log_every_n_epochs=10,
        trim_inds=[0, -1],
    ):
        self.log_every_n_epochs = log_every_n_epochs
        self.trim_inds = trim_inds

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
        ext_inputs_train = torch.cat([batch[6] for batch in train_dataloader]).to(
            pl_module.device
        )

        output_dict = pl_module.forward(
            ics_train, inputs_train, inputs_to_env=ext_inputs_train
        )

        lats_train = output_dict["latents"]

        lats_train = lats_train.detach().cpu().numpy()
        lats_train = lats_train[:, self.trim_inds[0] : self.trim_inds[1], :]

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


class MotorNetVideoGenerationArm(pl.Callback):
    def __init__(self, log_every_n_epochs=100, num_trials_to_plot=10):
        self.log_every_n_epochs = log_every_n_epochs
        self.num_trials_to_plot = num_trials_to_plot

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return

        # Get trajectories, model predictions, and actions
        train_dataloader = trainer.datamodule.train_dataloader()
        batch = next(iter(train_dataloader))
        ics = batch[0].to(pl_module.device)
        inputs = batch[1].to(pl_module.device)
        extra = batch[5].to(pl_module.device)
        targets = batch[2].to(pl_module.device)
        inputs_to_env = batch[6].to(pl_module.device)
        true_inputs = batch[7].to(pl_module.device)

        # Pass data through the model
        output_dict = pl_module.forward(
            ics[: self.num_trials_to_plot, :],
            inputs[: self.num_trials_to_plot, :, :],
            inputs_to_env[: self.num_trials_to_plot, :, :],
        )

        # Get the upper arm and forearm lengths
        upper_arm_length = trainer.model.task_env.effector.skeleton.l1
        forearm_length = trainer.model.task_env.effector.skeleton.l2

        # Get the data to plot
        controlled = output_dict["controlled"]
        joints = output_dict["joints"]
        actions = output_dict["actions"]

        B, T, _ = controlled.shape
        frames = []
        # Loop through the trials and time steps to create the video
        for i in range(self.num_trials_to_plot):
            go_cue = extra[i, 1].detach().cpu().numpy().astype(int)
            targ_on = extra[i, 0].detach().cpu().numpy().astype(int)
            for t in range(T):
                fig, ax = plt.subplots(figsize=(5, 5))

                shoulder_angle = joints[i, t, 0].item()
                elbow_angle = joints[i, t, 1].item()

                shoulder_angle_deg = np.rad2deg(shoulder_angle)
                elbow_angle_deg = np.rad2deg(elbow_angle)

                # Calculate arm segment positions
                shoulder_pos, elbow_pos, hand_pos = calculate_arm_positions(
                    shoulder_angle, elbow_angle, upper_arm_length, forearm_length
                )

                # Draw the arm
                ax.plot(
                    [shoulder_pos[0], elbow_pos[0]],
                    [shoulder_pos[1], elbow_pos[1]],
                    "k-",
                )
                ax.plot([elbow_pos[0], hand_pos[0]], [elbow_pos[1], hand_pos[1]], "k-")

                # Draw monoarticular muscle activations
                sho_mono_radius = 0.1
                elb_mono_radius = 0.05
                for muscle_idx in range(4):
                    activation_level = actions[i, t, muscle_idx].item()
                    color = get_activation_color(activation_level)
                    if muscle_idx == 0:
                        draw_muscle_activation(
                            ax,
                            shoulder_pos,
                            sho_mono_radius,
                            shoulder_angle_deg,
                            180,
                            color,
                        )
                    elif muscle_idx == 1:
                        draw_muscle_activation(
                            ax,
                            shoulder_pos,
                            sho_mono_radius,
                            0,
                            shoulder_angle_deg,
                            color,
                        )
                    elif muscle_idx == 2:
                        draw_muscle_activation(
                            ax,
                            elbow_pos,
                            elb_mono_radius,
                            shoulder_angle_deg + elbow_angle_deg,
                            shoulder_angle_deg + 180,
                            color,
                        )
                    elif muscle_idx == 3:
                        draw_muscle_activation(
                            ax,
                            elbow_pos,
                            elb_mono_radius,
                            shoulder_angle_deg + 180,
                            shoulder_angle_deg + elbow_angle_deg,
                            color,
                        )

                # Biarticular muscle visualization
                for muscle_idx in range(4, 6):
                    activation_level = actions[i, t, muscle_idx].item()
                    color = get_activation_color(activation_level)
                    if muscle_idx == 4:
                        draw_biarticular_muscle(
                            ax, shoulder_pos, elbow_pos, isFlexor=True, color=color
                        )
                    elif muscle_idx == 5:
                        draw_biarticular_muscle(
                            ax, shoulder_pos, elbow_pos, isFlexor=False, color=color
                        )

                # Draw a solid line a y = 0 to represent the ground
                ax.plot([-0.5, 0.5], [0, 0], "k-", linewidth=3)

                # Plot as a green square
                if t >= targ_on and t < go_cue:
                    input_patch = patches.Rectangle(
                        (
                            true_inputs[i, t, 0].item() - 0.05,
                            true_inputs[i, t, 1].item() - 0.05,
                        ),
                        0.1,
                        0.1,
                        color="red",
                    )
                    ax.add_patch(input_patch)

                # Plot target position as a red square
                target = patches.Rectangle(
                    (
                        targets[i, t, 0].item() - 0.05,
                        targets[i, t, 1].item() - 0.05,
                    ),
                    0.1,
                    0.1,
                    color="green",
                )
                ax.add_patch(target)

                # Plot hand position as a black dot
                ax.scatter(
                    controlled[i, t, 0].item(),
                    controlled[i, t, 1].item(),
                    color="black",
                )

                # Draw a dashed line from the start position to the current position
                ax.plot(
                    controlled[i, :t, 0].detach().cpu().numpy(),
                    controlled[i, :t, 1].detach().cpu().numpy(),
                    color="black",
                    linestyle="--",
                )

                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-0.2, 1.2)
                ax.axis("off")  # Hide axes for better visualization

                ax.set_title(f"Trial {i}")

                # Draw a bar to represent the time in the trial
                barStart = [-0.5, -0.1]
                bar_t = t / T
                barEnd = [barStart[0] + bar_t, -0.1]
                ax.plot(
                    [barStart[0], barEnd[0]],
                    [barStart[1], barEnd[1]],
                    "b-",
                    linewidth=3,
                )

                # Plot inputs_to_env as an arrow from the hand position with length 1
                inputs_mag = torch.norm(inputs_to_env[i, t, :]).item()
                if inputs_mag > 0.001:
                    ax.arrow(
                        controlled[i, t, 0].item(),
                        controlled[i, t, 1].item(),
                        inputs_to_env[i, t, 0].item() / (50 * inputs_mag),
                        inputs_to_env[i, t, 1].item() / (50 * inputs_mag),
                        head_width=0.05,
                        head_length=0.1,
                        fc="b",
                        ec="b",
                    )
                ax.set_aspect("equal")

                # Save figure to a numpy array
                fig.canvas.draw()
                img_arr = np.array(fig.canvas.renderer.buffer_rgba())
                frames.append(img_arr)
                plt.close(fig)

        # Save frames to a video using imageio
        video_path = "arm_movement_video.mp4"
        imageio.mimwrite(video_path, frames, fps=100)

        # Log the video to wandb
        wandb.log({"video": wandb.Video(video_path)})

        # Remove the video file
        os.remove(video_path)
