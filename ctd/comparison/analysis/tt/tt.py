import os
import pickle
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
from DSA.stats import dsa_bw_data_splits, dsa_to_id
from sklearn.decomposition import PCA

from ctd.comparison.analysis.analysis import Analysis
from ctd.comparison.fixedpoints import find_fixed_points
from ctd.comparison.metrics import compute_input_lyaps, compute_jacobians, compute_lyaps

dotenv.load_dotenv(override=True)
HOME_DIR = os.getenv("HOME_DIR")


class Analysis_TT(Analysis):
    def __init__(self, run_name, filepath, use_train_dm=False):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath, use_train_dm)
        self.run_hps = None

    def load_wrapper(self, filepath, use_train_dm=False):

        with open(filepath + "model.pkl", "rb") as f:
            self.wrapper = pickle.load(f)
        self.env = self.wrapper.task_env
        self.model = self.wrapper.model
        if use_train_dm:
            with open(filepath + "datamodule_train.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        else:
            with open(filepath + "datamodule_sim.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
                self.datamodule.prepare_data()
                self.datamodule.setup()
        # self.env = self.datamodule.data_env.dataset_name
        # if the simulator exists
        if Path(filepath + "simulator.pkl").exists():
            with open(filepath + "simulator.pkl", "rb") as f:
                self.simulator = pickle.load(f)
        n_train = len(self.datamodule.train_ds)
        n_val = len(self.datamodule.valid_ds)
        self.n_trials = n_train + n_val
        self.train_inds = range(0, int(0.8 * self.n_trials))
        self.valid_inds = range(int(0.8 * self.n_trials), self.n_trials)
        self.trim_inds = self.simulator.trim_inds

    def get_inputs(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_inputs = torch.cat([train_ds.tensors[1], valid_ds.tensors[1]], dim=0)

        if self.trim_inds is not None:
            start = self.trim_inds[0] - 1
            end = self.trim_inds[1]
        else:
            start = 0
            end = tt_inputs.shape[1]

        if phase == "all":
            return tt_inputs[:, start:end, :]
        elif phase == "train":
            return tt_inputs[self.train_inds, start:end, :]
        elif phase == "val":
            return tt_inputs[self.valid_inds, start:end, :]

    def get_true_inputs(self, phase="all"):
        train_ds = self.datamodule.train_ds
        valid_ds = self.datamodule.valid_ds
        tt_inputs = torch.cat([train_ds.tensors[7], valid_ds.tensors[7]], dim=0)
        if phase == "all":
            return tt_inputs
        elif phase == "train":
            return tt_inputs[self.train_inds]
        elif phase == "val":
            return tt_inputs[self.valid_inds]

    def get_inputs_to_env(self, phase="all"):
        if phase == "all":
            train_inputs_to_env = self.datamodule.train_ds.tensors[6]
            valid_inputs_to_env = self.datamodule.valid_ds.tensors[6]
            return torch.cat([train_inputs_to_env, valid_inputs_to_env], dim=0)
        elif phase == "train":
            return self.datamodule.train_ds.tensors[6]
        elif phase == "val":
            return self.datamodule.valid_ds.tensors[6]

    def get_model_inputs(self, phase="all"):

        if phase == "all":
            train_ics = self.datamodule.train_ds.tensors[0]
            train_inputs = self.datamodule.train_ds.tensors[1]
            train_targets = self.datamodule.train_ds.tensors[2]
            valid_ics = self.datamodule.valid_ds.tensors[0]
            valid_inputs = self.datamodule.valid_ds.tensors[1]
            valid_targets = self.datamodule.valid_ds.tensors[2]
            tt_ics = torch.cat([train_ics, valid_ics], dim=0)
            tt_inputs = torch.cat([train_inputs, valid_inputs], dim=0)
            tt_targets = torch.cat([train_targets, valid_targets], dim=0)
            return tt_ics, tt_inputs, tt_targets
        elif phase == "train":
            return (
                self.datamodule.train_ds.tensors[0],
                self.datamodule.train_ds.tensors[1],
                self.datamodule.train_ds.tensors[2],
            )
        elif phase == "val":
            return (
                self.datamodule.valid_ds.tensors[0],
                self.datamodule.valid_ds.tensors[1],
                self.datamodule.valid_ds.tensors[2],
            )

    def get_extra_inputs(self, phase="all"):
        if phase == "all":
            train_extra = self.datamodule.train_ds.tensors[5]
            valid_extra = self.datamodule.valid_ds.tensors[5]
            tt_extra = torch.cat([train_extra, valid_extra], dim=0)
            return tt_extra
        elif phase == "train":
            return self.datamodule.train_ds.tensors[5]
        elif phase == "val":
            return self.datamodule.valid_ds.tensors[5]

    def get_model_inputs_noiseless(self, phase="all"):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase=phase)

        train_noiseless_inputs = self.datamodule.train_ds.tensors[7]
        valid_noiseless_inputs = self.datamodule.valid_ds.tensors[7]
        tt_noiseless_inputs = torch.cat(
            [train_noiseless_inputs, valid_noiseless_inputs], dim=0
        )

        if phase == "all":
            return tt_ics, tt_noiseless_inputs, tt_targets
        elif phase == "train":
            return tt_ics, train_noiseless_inputs, tt_targets
        elif phase == "val":
            return tt_ics, valid_noiseless_inputs, tt_targets

    def get_model_outputs(self, phase="all"):
        inputs_to_env = self.get_inputs_to_env(phase=phase)
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase=phase)
        out_dict = self.wrapper(tt_ics, tt_inputs, inputs_to_env)
        return out_dict

    def get_model_outputs_noiseless(self, phase="all"):
        inputs_to_env = self.get_inputs_to_env(phase=phase)
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs_noiseless(phase=phase)
        out_dict = self.wrapper(tt_ics, tt_inputs, inputs_to_env)
        return out_dict

    def get_latents(self, phase="all"):
        out_dict = self.get_model_outputs(phase=phase)
        if self.trim_inds is not None:
            return out_dict["latents"][:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
        else:
            return out_dict["latents"]

    def get_latents_noiseless(self, phase="all"):
        out_dict = self.get_model_outputs_noiseless(phase=phase)
        if self.trim_inds is not None:
            return out_dict["latents"][:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
        else:
            return out_dict["latents"]

    def get_latents_pca(self, num_PCs=3):
        latents = self.get_latents()
        B, T, N = latents.shape
        latents = latents.reshape(-1, N)
        pca = PCA(n_components=num_PCs)
        latents_pca = pca.fit_transform(latents)
        latents_pca = latents.reshape(B, T, num_PCs)
        return latents_pca, pca

    def plot_trial_latents(self, num_trials=10):
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        pca = PCA(n_components=3)
        lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(num_trials):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        ax.set_title("Task-trained Latent Activity")
        plt.show()

    def plot_trial_io(self, num_trials):
        ics, inputs, targets = self.get_model_inputs()
        out_dict = self.get_model_outputs()
        latents = out_dict["latents"].detach().numpy()
        controlled = out_dict["controlled"].detach().numpy()
        pca = PCA(n_components=3)
        lats_pca = pca.fit_transform(latents.reshape(-1, latents.shape[-1]))
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        fig = plt.figure(figsize=(3 * num_trials, 6))
        for i in range(num_trials):
            ax1 = fig.add_subplot(4, num_trials, i + 1)
            ax1.plot(lats_pca[i, :, 0])
            ax1.plot(lats_pca[i, :, 1])
            ax1.plot(lats_pca[i, :, 2])
            ax1.set_title(f"Trial {i}")
            ax2 = fig.add_subplot(4, num_trials, i + num_trials + 1)
            for j in range(controlled.shape[-1]):
                ax2.plot(controlled[i, :, j])

            ax3 = fig.add_subplot(4, num_trials, i + 2 * num_trials + 1)
            for j in range(targets.shape[-1]):
                ax3.plot(targets[i, :, j])

            ax4 = fig.add_subplot(4, num_trials, i + 3 * num_trials + 1)
            for j in range(inputs.shape[-1]):
                ax4.plot(inputs[i, :, j])
            if i == 0:
                ax1.set_ylabel("Latent Activity")
                ax2.set_ylabel("Controlled")
                ax3.set_ylabel("Targets")
                ax4.set_ylabel("Inputs")
            if i == 4:
                ax1.set_xlabel("Time")
                ax2.set_xlabel("Time")
                ax3.set_xlabel("Time")
                ax4.set_xlabel("Time")
            else:
                ax1.set_xlabel("")
                ax2.set_xlabel("")
                ax3.set_xlabel("")
                ax4.set_xlabel("")
                ax1.set_xticks([])
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])

        plt.suptitle("Task-trained Latent Activity")
        plt.show()

    def compute_FPs(
        self,
        noiseless=True,
        inputs=None,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
    ):
        # Compute latent activity from task trained model
        if inputs is None and noiseless:
            _, inputs, _ = self.get_model_inputs_noiseless()
            latents = self.get_latents_noiseless()
        elif inputs is None and not noiseless:
            _, inputs, _ = self.get_model_inputs()
            latents = self.get_latents()
        else:
            latents = self.get_latents()
        if hasattr(self.wrapper.model, "generator"):
            cell = self.wrapper.model.generator
        else:
            cell = self.wrapper.model.cell
        fps = find_fixed_points(
            model=cell,
            state_trajs=latents,
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
        )
        return fps

    def plot_fps(
        self,
        inputs=None,
        num_traj=10,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cpu",
        seed=0,
        compute_jacobians=True,
        q_thresh=1e-5,
    ):

        latents = self.get_latents(phase="val").detach().numpy()
        fps = self.compute_FPs(
            inputs=inputs,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=learning_rate,
            max_iters=max_iters,
            device=device,
            seed=seed,
            compute_jacobians=compute_jacobians,
        )
        xstar = fps.xstar
        q_vals = fps.qstar
        is_stable = fps.is_stable
        figQs = plt.figure()
        axQs = figQs.add_subplot(111)
        q_flag_temp = q_vals < 1e-15
        q_vals[q_flag_temp] = 1e-15
        axQs.hist(np.log10(q_vals), bins=100)
        axQs.set_title("Q* Histogram")
        axQs.set_xlabel("log10(Q*)")

        colors = np.zeros((xstar.shape[0], 3))
        colors[is_stable, :] = np.array([0, 0.3922, 0])  # darkgreen
        colors[~is_stable, 0] = 1

        q_flag = q_vals < q_thresh
        pca = PCA(n_components=3)
        xstar_pca = pca.fit_transform(xstar)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.transform(lats_flat)
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        # Make a color vector based on stability
        ax.scatter(
            xstar_pca[q_flag, 0],
            xstar_pca[q_flag, 1],
            xstar_pca[q_flag, 2],
            c=colors[q_flag, :],
        )
        for i in range(num_traj):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        # Add legend for stability
        ax.plot([], [], "o", color="red", label="Unstable")
        ax.plot([], [], "o", color="darkgreen", label="Stable")
        ax.legend()
        ax.set_title("tt_Fixed Points")
        plt.show()
        return fps

    def simulate_neural_data(self, subfolder, dataset_path):
        self.simulator.simulate_neural_data(
            self.wrapper,
            self.datamodule,
            self.run_name,
            subfolder,
            dataset_path,
            seed=0,
        )

    def find_DSA_hps(
        self,
        rank_sweep=[10, 20],
        delay_sweep=[1, 5],
    ):
        id_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        splits_comp = np.zeros((len(rank_sweep), len(delay_sweep)))
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        for i, rank in enumerate(rank_sweep):
            for j, delay in enumerate(delay_sweep):
                print(f"Rank: {rank}, Delay: {delay}")
                id_comp[i, j] = dsa_to_id(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
                splits_comp[i, j] = dsa_bw_data_splits(
                    data=latents,
                    rank=rank,
                    n_delays=delay,
                    delay_interval=1,
                )
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(id_comp)
        ax.set_title("ID")
        ax.set_xticks(np.arange(len(delay_sweep)))
        ax.set_yticks(np.arange(len(rank_sweep)))
        ax.set_xticklabels(delay_sweep)
        ax.set_yticklabels(rank_sweep)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        plt.savefig(f"{HOME_DIR}/id_comp.png")
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(splits_comp)
        ax2.set_title("Splits")
        ax2.set_xticks(np.arange(len(delay_sweep)))
        ax2.set_yticks(np.arange(len(rank_sweep)))
        ax2.set_xticklabels(delay_sweep)
        ax2.set_yticklabels(rank_sweep)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.savefig(f"{HOME_DIR}/splits_comp.png")
        return id_comp, splits_comp

    def save_latents(self, filepath):
        latents = self.get_latents().detach().numpy()
        with open(filepath, "wb") as f:
            pickle.dump(latents, f)

    def plot_scree(self, max_pcs=10):
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        pca = PCA(n_components=max_pcs)
        pca.fit(latents)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(range(1, max_pcs + 1), pca.explained_variance_ratio_ * 100, marker="o")
        ax.set_xlabel("PC #")
        ax.set_title("Scree Plot")
        ax.set_ylabel("Explained Variance (%)")
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, max_pcs + 1), np.cumsum(pca.explained_variance_ratio_) * 100)
        ax2.set_xlabel("PC #")
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_ylabel("Explained Variance (%)")
        # Add horiz lines at 50, 90, 95, 99%
        ax2.axhline(y=50, color="r", linestyle="--")
        ax2.axhline(y=90, color="r", linestyle="--")
        ax2.axhline(y=95, color="r", linestyle="--")
        ax2.axhline(y=99, color="r", linestyle="--")
        # Add y ticks
        ax2.set_yticks([50, 90, 95, 99])
        plt.savefig(f"{HOME_DIR}/scree_plot.png")
        return pca.explained_variance_ratio_

    def get_trial_lens(self, phase="val"):
        if self.env.dataset_name != "MultiTask":
            raise NotImplementedError(
                f"get_trial_lens not implemented for '{self.env.dataset_name}'."
            )
        phase_dict = self.datamodule.extra_data["phase_dict"]
        train_inds = self.datamodule.train_ds.tensors[3].detach().numpy().astype(int)
        valid_inds = self.datamodule.valid_ds.tensors[3].detach().numpy().astype(int)
        len_list = []
        if phase == "val":
            inds = valid_inds
        elif phase == "train":
            inds = train_inds
        elif phase == "all":
            inds = np.vstack((train_inds, valid_inds))
        for i in inds:
            len_list.append(phase_dict[i]["response"][1])

        return len_list

    def compute_lyapunov_exp(self, phase="val", n_trials=None, input_lyaps=False):
        # Get the latent activity
        outputs = self.get_model_outputs(phase=phase)
        latents = outputs["latents"]
        states = outputs["states"]
        # Get the inputs
        inputs = self.get_model_inputs(phase=phase)[1]
        if states is not None:
            combined_in = torch.cat([states, inputs], dim=-1)
        else:
            combined_in = inputs
        # Get the flow-field model

        if hasattr(self.wrapper.model, "generator"):
            cell = self.wrapper.model.generator
        else:
            cell = self.wrapper.model.cell
        #
        # Compute the Jacobians
        Jz, Ju, trial_idx = compute_jacobians(
            z=latents,
            u=combined_in,
            f=cell,
            num_trials=n_trials,
        )

        # Compute the Lyapunov exponents
        les = compute_lyaps(
            Js=Jz,
            dt=1,
        )
        if input_lyaps:
            # Compute the input Lyapunov exponents
            les_u = compute_input_lyaps(
                Jz=Jz,
                Ju=Ju,
                dt=1,
            )
            les = torch.cat((les, les_u), dim=-1)
        return les.mean(0), les.std(0)
