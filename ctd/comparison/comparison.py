import matplotlib.pyplot as plt
import numpy as np
import torch
from DSA import DSA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ctd.comparison.metrics import (
    compute_nl_cycle_consistency,
    compute_static_rotation_invariant_mmd,
    get_bps,
    get_cycle_consistency,
    get_signal_r2,
    get_signal_r2_linear,
)


class Comparison:
    def __init__(self, comparison_tag=None):
        self.comparison_tag = comparison_tag
        self.num_analyses = 0
        self.analyses = []
        self.ref_ind = None
        self.groups = []

    def load_analysis(self, analysis, group="None", reference_analysis=False):
        self.analyses.append(analysis)
        self.groups.append(group)
        self.num_analyses += 1
        if self.ref_ind is None and reference_analysis:
            self.ref_ind = self.num_analyses - 1
        elif self.ref_ind is not None and reference_analysis:
            # Throw an error
            raise ValueError("There is already a reference analysis")

    def regroup(self):
        groups = np.array(self.groups)
        # Sort analyses by group
        sorted_inds = np.argsort(groups)
        self.analyses = [self.analyses[i] for i in sorted_inds]
        self.groups = groups[sorted_inds]
        self.ref_ind = np.where(sorted_inds == self.ref_ind)[0][0]

    def compute_metrics(
        self,
        ref_ind=None,
        metric_dict_list={},
    ):
        # Get the rates, latents, and inputs
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        reference_analysis = self.analyses[ref_ind]
        true_lats_train = reference_analysis.get_latents(phase="train")
        true_lats_val = reference_analysis.get_latents(phase="val")

        true_inputs_train = reference_analysis.get_inputs(phase="train")
        true_inputs_val = reference_analysis.get_inputs(phase="val")

        # If the task has unequal trial lengths, trim the trials
        if reference_analysis.env.dataset_name == "MultiTask":
            unequal_trial_lens = True
            trial_lens_train = reference_analysis.get_trial_lens(phase="train")
            trial_lens_val = reference_analysis.get_trial_lens(phase="val")
            true_lats_list_train = []
            true_lats_list_val = []
            true_inputs_list_train = []
            true_inputs_list_val = []
            for i, t_len in enumerate(trial_lens_train):
                true_lats_list_train.append(true_lats_train[i, :t_len, :])
                true_inputs_list_train.append(true_inputs_train[i, :t_len, :])
            for i, v_len in enumerate(trial_lens_val):
                true_lats_list_val.append(true_lats_val[i, :v_len, :])
                true_inputs_list_val.append(true_inputs_val[i, v_len, :])

            true_lats_train = torch.concatenate(true_lats_list_train)
            true_lats_val = torch.concatenate(true_lats_list_val)
            true_inputs_train = torch.concatenate(true_inputs_list_train)
            true_inputs_val = torch.concatenate(true_inputs_list_val)

        else:
            unequal_trial_lens = False

        metrics_dict = {"run_name": [], "group": []}
        for metric in metric_dict_list.keys():
            metrics_dict[metric] = []

        # Iterate through the analyses
        for i in range(self.num_analyses):
            print("")
            print(
                f"Working on {i+1} of {self.num_analyses}: {self.analyses[i].run_name}"
            )

            if i == ref_ind:  # Skip the task-trained network
                continue

            metrics_dict["run_name"].append(self.analyses[i].run_name)
            metrics_dict["group"].append(self.groups[i])

            n_input_neurons = self.analyses[i].get_inputs()[0].shape[-1]
            inf_rates_train, inf_latents_train = self.analyses[i].get_model_outputs(
                phase="train"
            )
            inf_rates_val, inf_latents_val = self.analyses[i].get_model_outputs(
                phase="val"
            )
            true_rates_val = self.analyses[i].get_true_rates(phase="val")
            inp_spikes_val = self.analyses[i].get_spiking(phase="val")

            if unequal_trial_lens:
                inf_rates_train_list = []
                inf_rates_val_list = []
                inf_latents_train_list = []
                inf_latents_val_list = []
                true_rates_val_list = []
                spiking_val_list = []

                for j, t_len in enumerate(trial_lens_train):
                    inf_rates_train_list.append(inf_rates_train[j, :t_len, :])
                    inf_latents_train_list.append(inf_latents_train[j, :t_len, :])
                for j, v_len in enumerate(trial_lens_val):
                    inf_rates_val_list.append(inf_rates_val[j, :v_len, :])
                    inf_latents_val_list.append(inf_latents_val[j, :v_len, :])
                    true_rates_val_list.append(true_rates_val[j, :v_len, :])
                    spiking_val_list.append(inp_spikes_val[j, :v_len, :])

                inf_rates_train = torch.concatenate(inf_rates_train_list)
                inf_latents_train = torch.concatenate(inf_latents_train_list)
                inf_rates_val = torch.concatenate(inf_rates_val_list)
                inf_latents_val = torch.concatenate(inf_latents_val_list)
                true_rates_val = torch.concatenate(true_rates_val_list)
                inp_spikes_val = torch.concatenate(spiking_val_list)

            # Check that latents arenot NaN
            if np.isnan(inf_latents_train.detach().numpy()).any():
                continue
            for j, metric in enumerate(metric_dict_list):
                if metric == "rate_r2":
                    rate_r2 = get_signal_r2(
                        signal_true=true_rates_val,
                        signal_pred=inf_rates_val,
                    )
                    print(f"Rate R2: {rate_r2}")
                    metrics_dict["rate_r2"].append(rate_r2)

                elif metric == "recon_r2":
                    recon_r2 = get_signal_r2_linear(
                        signal_true_train=true_lats_train,
                        signal_pred_train=inf_latents_train,
                        signal_true_val=true_lats_val,
                        signal_pred_val=inf_latents_val,
                    )
                    print(f"Recon R2: {recon_r2}")
                    metrics_dict["recon_r2"].append(recon_r2)

                elif metric == "input_r2":
                    inf_inputs_train = self.analyses[i].get_inferred_inputs(
                        phase="train"
                    )
                    inf_inputs_val = self.analyses[i].get_inferred_inputs(phase="val")
                    if unequal_trial_lens:
                        inf_inputs_train_list = []
                        inf_inputs_val_list = []
                        for i, t_len in enumerate(trial_lens_train):
                            inf_inputs_train_list.append(inf_inputs_train[i, :t_len, :])
                        for i, v_len in enumerate(trial_lens_val):
                            inf_inputs_val_list.append(inf_inputs_val[i, :v_len, :])
                        inf_inputs_train = torch.concatenate(inf_inputs_train_list)
                        inf_inputs_val = torch.concatenate(inf_inputs_val_list)

                    input_r2 = get_signal_r2_linear(
                        signal_true_train=true_inputs_train,
                        signal_pred_train=inf_inputs_train,
                        signal_true_val=true_inputs_val,
                        signal_pred_val=inf_inputs_val,
                    )
                    print(f"Input R2: {input_r2}")
                    metrics_dict["input_r2"].append(input_r2)

                elif metric in ["state_r2"]:
                    state_r2 = get_signal_r2_linear(
                        signal_true_train=true_lats_train,
                        signal_pred_train=inf_latents_train,
                        signal_true_val=true_lats_val,
                        signal_pred_val=inf_latents_val,
                    )
                    print(f"State R2: {state_r2}")
                    metrics_dict["state_r2"].append(state_r2)

                elif metric in ["co-bps"]:
                    inf_rates_co = inf_rates_val[..., n_input_neurons:]
                    spiking_co = inp_spikes_val[..., n_input_neurons:]
                    bps = get_bps(
                        inf_rates=inf_rates_co.detach().numpy(),
                        true_spikes=spiking_co,
                    )
                    print(f"CO-BPS: {bps}")
                    metrics_dict["co-bps"].append(bps)
                elif metric in ["cycle_con"]:
                    cycle_con_var = metric_dict_list[metric].get(
                        "variance_threshold", 0.01
                    )
                    linear_cycle_con = get_cycle_consistency(
                        inf_latents_train=inf_latents_train.detach().numpy(),
                        inf_rates_train=inf_rates_train.detach().numpy(),
                        inf_latents_val=inf_latents_val.detach().numpy(),
                        inf_rates_val=inf_rates_val.detach().numpy(),
                        variance_threshold=cycle_con_var,
                    )
                    print(f"Cycle Consistency R2: {linear_cycle_con}")
                    metrics_dict["cycle_con"].append(linear_cycle_con)
                elif metric in ["nl_cycle_con"]:
                    hidden_sizes = metric_dict_list[metric].get(
                        "mlp_hidden_layers", [64, 64]
                    )
                    lr = metric_dict_list[metric].get("lr", 1e-3)
                    max_epochs = metric_dict_list[metric].get("max_epochs", 1000)
                    patience = metric_dict_list[metric].get("patience", 50)
                    min_delta = metric_dict_list[metric].get("min_delta", 1e-5)
                    noise_stds = metric_dict_list[metric].get(
                        "noise_stds", [0, 0.01, 0.05, 0.1]
                    )
                    weight_decay = metric_dict_list[metric].get("weight_decay", 1e-4)
                    device = metric_dict_list[metric].get(
                        "device", "cuda" if torch.cuda.is_available() else "cpu"
                    )

                    nl_cycle_con = compute_nl_cycle_consistency(
                        inf_latents_train=inf_latents_train.detach().numpy(),
                        inf_rates_train=inf_rates_train.detach().numpy(),
                        inf_latents_val=inf_latents_val.detach().numpy(),
                        inf_rates_val=inf_rates_val.detach().numpy(),
                        hidden_sizes=hidden_sizes,
                        lr=lr,
                        max_epochs=max_epochs,
                        patience=patience,
                        min_delta=min_delta,
                        noise_stds=noise_stds,
                        weight_decay=weight_decay,
                        device=device,
                    )
                    print(f"Nonlinear Cycle Consistency R2: {nl_cycle_con}")
                    metrics_dict["nl_cycle_con"].append(nl_cycle_con)

                elif metric in ["mmd"]:
                    feature_types = metric_dict_list[metric].get(
                        "feature_types", ["pairwise", "spectral", "distances"]
                    )
                    max_pairwise_samples = metric_dict_list[metric].get(
                        "max_pairwise_samples", 1000
                    )
                    mmd = compute_static_rotation_invariant_mmd(
                        x_true=true_lats_val.detach().numpy(),
                        x_pred=inf_latents_val.detach().numpy(),
                        feature_types=feature_types,
                        max_pairwise_samples=max_pairwise_samples,
                    )
                    print(f"MMD: {mmd}")
                    metrics_dict["mmd"].append(mmd)

                elif metric in ["lyapunov"]:
                    lyap_mean, lyap_std = self.analyses[i].compute_lyapunov_exp()
                    lyap_mean = lyap_mean.detach().numpy()
                    lyap_std = lyap_std.detach().numpy()
                    print(
                        f"Lyapunov Exponent: {lyap_mean[0]:.6f} +/- {lyap_std[0]:.6f}"
                    )
                    metrics_dict["lyapunov"].append(lyap_mean)

                else:
                    raise ValueError("Invalid metric")

        return metrics_dict

    def compare_dynamics_DSA(
        self,
        phase="val",
        n_delays=20,
        rank=50,
        delay_interval=1,
        device="cuda",
        percent_data=0.01,
    ):
        latent_list = []
        for analysis in self.analyses:
            latents = analysis.get_latents(phase=phase).detach().numpy()
            # latent_list.append(latents.reshape(-1, latents.shape[-1]))
            latent_list.append(latents[: int(percent_data * latents.shape[0]), :, :])

        dsa = DSA(
            latent_list,
            n_delays=n_delays,
            rank=rank,
            delay_interval=delay_interval,
            verbose=True,
            device=device,
            iters=1500,
            lr=0.005,
        )
        similarities = dsa.fit_score()
        cmap_r = plt.cm.get_cmap("viridis_r")
        fit_mat = similarities
        ij_figure = plt.figure()
        ij_ax = ij_figure.add_subplot(111)
        # Plot it as an image
        ij_ax.imshow(fit_mat, cmap=cmap_r)
        ij_ax.set_title("Dynamical Similarity (DSA)")
        ij_ax.set_xticks(np.arange(self.num_analyses))
        ij_ax.set_yticks(np.arange(self.num_analyses))
        ij_ax.set_xticklabels([analysis.run_name for analysis in self.analyses])
        ij_ax.set_yticklabels([analysis.run_name for analysis in self.analyses])
        # Rotate the tick labels and set their alignment.
        plt.setp(
            ij_ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )

        ij_figure.colorbar(
            plt.imshow(fit_mat, cmap=cmap_r),
            ax=ij_ax,
        )
        # Save as pdf in a different
        plt.savefig(f"{self.comparison_tag}_dsa.pdf")
        return similarities

    def plot_trials(self, num_trials, num_pcs=3):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            pca = PCA()
            latents_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            latents_pca_flat = pca.fit_transform(latents_flat)
            latents_pca = latents_pca_flat.reshape(latents.shape)
            for j in range(num_trials):
                axes[i, j].plot(latents_pca[j, :, :num_pcs])
                if i == 0:
                    axes[i, j].set_title(f"Trial {j}")
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

    def plot_inputs(self, num_trials, num_pcs=3):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            inputs = self.analyses[i].get_inputs().detach().numpy()
            for j in range(num_trials):
                axes[i, j].plot(inputs[j, :, :])
                if i == 0:
                    axes[i, j].set_title(f"Trial {j}")
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

    def plot_trials_controlled_reference(self, num_trials=2, ref_ind=None, num_pcs=3):

        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        out_dict = self.analyses[ref_ind].get_model_outputs()
        ref_lats = out_dict["latents"].detach().numpy()
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)
        ref_lats_pca = ref_lats_pca_flat.reshape(ref_lats.shape)

        fig = plt.figure()
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)

            for j in range(num_trials):
                axes[i, j].plot(latents_pca[j, :, :num_pcs])
                if i == 0:
                    axes[i, j].set_title(f"Trial {j}")
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

    def plot_trials_reference_dims(self, num_trials=2, ref_ind=None, dims=[0, 1, 2, 3]):

        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        ref_lats = (
            self.analyses[ref_ind]
            .get_latents(
                phase="val",
            )
            .detach()
            .numpy()
        )
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)
        ref_lats_pca = ref_lats_pca_flat.reshape(ref_lats.shape)

        fig = plt.figure()
        axes = fig.subplots(num_trials, len(dims), sharey=True, sharex=True)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents(phase="val").detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)

            for j in range(num_trials):
                for k in range(len(dims)):
                    if j == num_trials - 1 and k == len(dims) - 1:
                        axes[j, k].plot(
                            latents_pca[j, :100, dims[k]],
                            label=self.analyses[i].run_name,
                        )
                        axes[j, k].set_xlabel("Time")
                    else:
                        axes[j, k].plot(latents_pca[j, :100, dims[k]])
                        axes[j, k].set_xticks([])

            axes[num_trials - 1, len(dims) - 1].legend()
            plt.suptitle("Predicted TT PC")
            plt.savefig(f"{self.comparison_tag}_predicted_tt_pc.pdf")

    def plot_trials_reference(self, num_trials=2, ref_ind=None, num_pcs=4):

        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        ref_lats = (
            self.analyses[ref_ind]
            .get_latents(
                phase="val",
            )
            .detach()
            .numpy()
        )
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)
        ref_lats_pca = ref_lats_pca_flat.reshape(ref_lats.shape)

        fig = plt.figure()
        axes = fig.subplots(self.num_analyses, num_trials)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents(phase="val").detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)

            for j in range(num_trials):
                axes[i, j].plot(latents_pca[j, :, :num_pcs])
                if i == 0:
                    axes[i, j].set_title(f"Trial {j}")
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

    def plot_trials_3d(self, num_trials):
        # Function to plot one trial from each analysis
        fig = plt.figure()
        # One subplot row per analysis
        # One subplot column per trial
        axes = fig.subplots(1, self.num_analyses, subplot_kw={"projection": "3d"})
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            pca = PCA()
            latents_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            latents_pca_flat = pca.fit_transform(latents_flat)
            latents_pca = latents_pca_flat.reshape(latents.shape)
            for j in range(num_trials):
                axes[i].plot(
                    latents_pca[j, :, 0],
                    latents_pca[j, :, 1],
                    latents_pca[j, :, 2],
                )

            axes[i].set_title(f"{self.analyses[i].run_name}")
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_zlabel("PC3")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_zticks([])

    def plot_trials_3d_reference(
        self, num_trials, ref_ind=None, savePDF=False, angle=None
    ):
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        ref_lats = self.analyses[ref_ind].get_latents().detach().numpy()
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)
        ref_lats_pca = ref_lats_pca_flat.reshape(ref_lats.shape)

        fig = plt.figure(figsize=(15, 5))
        axes = fig.subplots(1, self.num_analyses, subplot_kw={"projection": "3d"})
        # Make an array of axis ranges. Should be [-inf, inf] for three rows
        axis_ranges = np.zeros((3, 2))
        axis_ranges[:, 0] = np.inf
        axis_ranges[:, 1] = -1 * np.inf

        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)
            for j in range(num_trials):
                axes[i].plot(
                    latents_pca[j, :, 0],
                    latents_pca[j, :, 1],
                    latents_pca[j, :, 2],
                )
                # Update the axis ranges
            for k in range(3):
                axis_ranges[k, 0] = np.min(
                    [axis_ranges[k, 0], np.min(latents_pca[:, :, k])]
                )
                axis_ranges[k, 1] = np.max(
                    [axis_ranges[k, 1], np.max(latents_pca[:, :, k])]
                )

            axes[i].set_title(f"{self.analyses[i].run_name}")
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_zlabel("PC3")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_zticks([])
        # Set the axis limits
        for i in range(self.num_analyses):
            axes[i].set_xlim(axis_ranges[0, :])
            axes[i].set_ylim(axis_ranges[1, :])
            axes[i].set_zlim(axis_ranges[2, :])
            if angle is not None:
                axes[i].view_init(angle[0], angle[1])

        if savePDF:
            plt.savefig(f"{self.comparison_tag}_3dLats.pdf")
        return fig

    def plot_neural_preds(self, neuron_list, trial_list):
        # Function to plot the neural predictions
        ref_ind = self.ref_ind

        fig = plt.figure(figsize=(10, 10))
        rates_list = []
        r2_vals = []
        dd_names = []
        for i in range(self.num_analyses):
            if i == ref_ind:
                continue
            rates, latents = self.analyses[i].get_model_outputs(phase="val")
            true_rates = self.analyses[i].get_true_rates(phase="val")
            rates_list.append(rates.detach().numpy())
            r2_vals.append(
                r2_score(
                    true_rates.reshape(-1, true_rates.shape[-1]),
                    rates.detach().numpy().reshape(-1, rates.shape[-1]),
                    multioutput="raw_values",
                )
            )
            dd_names.append(self.analyses[i].run_name)

        rates = np.stack(rates_list)
        axes = fig.subplots(len(neuron_list), len(trial_list))
        for i in range(len(dd_names)):
            for j in range(len(neuron_list)):
                for k in range(len(trial_list)):
                    if i == 0:
                        axes[j, k].plot(
                            true_rates[trial_list[k], :, neuron_list[j]],
                            label="True",
                            color="black",
                        )

                    axes[j, k].plot(
                        rates[i, trial_list[k], :, neuron_list[j]],
                        label=f"{dd_names[i]} Pred",
                    )

                    if i == len(dd_names) - 1:
                        axes[j, k].set_xlabel("Time")
                    else:
                        axes[j, k].set_xticks([])
                    if j == 0:
                        axes[j, k].set_title(f"Trial {trial_list[k]}")
                    if k == 0:
                        axes[j, k].set_ylabel(f"Neuron {neuron_list[j]}")
                        axes[j, k].text(
                            0.5,
                            0.9 - 0.05 * i,
                            f"{dd_names[i]} R2: {r2_vals[i][neuron_list[j]]:.2f}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[j, k].transAxes,
                        )
        axes[0, 0].legend()

        for ax in axes.flat:
            # set ymin to 0
            ax.set_ylim(bottom=0)
        plt.suptitle("Neural Predictions")

        plt.savefig(f"{self.comparison_tag}_neural_preds.pdf")

        return r2_vals

    def compare_performance(self):
        _, _, targets = self.analyses[self.ref_ind].get_model_inputs(phase="val")
        mean_r2 = []
        for i in range(self.num_analyses):
            print(f"Working on {i+1} of {self.num_analyses}")
            latents = self.analyses[i].get_latents(phase="val").detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            targets_flat = targets.reshape(
                targets.shape[0] * targets.shape[1], targets.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, targets_flat)
            pred = reg.predict(lats_flat)
            r2_perf = r2_score(targets_flat, pred, multioutput="raw_values")
            print(f"Performance R2s for {self.analyses[i].run_name} is {r2_perf}")
            mean_r2.append(np.mean(r2_perf))

        # Find the mean in each group
        mean_in_group = []
        for group in np.unique(self.groups):
            group_inds = np.where(self.groups == group)[0]
            mean_in_group.append(np.mean([mean_r2[i] for i in group_inds]))

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(np.unique(self.groups))), mean_in_group)
        ax.set_xticks(np.arange(len(np.unique(self.groups))))
        ax.set_xticklabels(np.unique(self.groups))
        minVal = np.min(mean_in_group) - 0.05
        maxVal = np.max(mean_in_group) + 0.05
        ax.set_ylim([minVal, maxVal])
        ax.set_ylabel("Mean R2 of Performance in group")
        ax.set_title("Performance of data-trained models at task (R2)")

    def visualize_stateR2(self, num_trials=2, ref_ind=None, num_pcs=7):

        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        ref_lats = (
            self.analyses[ref_ind]
            .get_latents(
                phase="val",
            )
            .detach()
            .numpy()
        )
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)

        fig = plt.figure(figsize=(20, 10))
        axes = fig.subplots(self.num_analyses, num_pcs)
        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents(phase="val").detach().numpy()
            pca_DD = PCA()

            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            lats_pca_flat = pca_DD.fit_transform(lats_flat)

            reg = LinearRegression().fit(ref_lats_pca_flat, lats_pca_flat)
            pred_latents_pca_flat = reg.predict(ref_lats_pca_flat)
            r2_scores = r2_score(
                lats_pca_flat, pred_latents_pca_flat, multioutput="raw_values"
            )
            pred_latents_pca = pred_latents_pca_flat.reshape(latents.shape)
            lats_pca = lats_pca_flat.reshape(latents.shape)

            for j in range(num_pcs):
                if j < latents.shape[2]:
                    axes[i, j].plot(
                        pred_latents_pca[0, :, j], c="r", label="Predicted from TT"
                    )
                    axes[i, j].plot(lats_pca[0, :, j], c="k", label="True from DD")
                    axes[i, j].set_title(f"R2: {r2_scores[j]:.2f}")
                else:
                    axes[i, j].plot(
                        np.zeros(lats_pca[0, :, 0].shape), label="Predicted from TT"
                    )
                if i == self.num_analyses - 1:
                    axes[i, j].set_xlabel("Time")
                else:
                    axes[i, j].set_xticks([])

            axes[i, 0].set_ylabel(f"{self.analyses[i].run_name}")

    def plot_rate_state_input_r2(
        self,
        ref_ind=None,
        label_runs=False,
        label_groups=True,
        phase="val",
        plot_dict={},
    ):
        if "save_pdf" in plot_dict:
            save_pdf = plot_dict["save_pdf"]
        else:
            save_pdf = False
        if "ax_lim" in plot_dict:
            ax_lim = plot_dict["ax_lim"]
        else:
            ax_lim = None
        if "marker" in plot_dict:
            marker = plot_dict["marker"]
        else:
            marker = "o"

        # Function to compare the latent activity
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        reference_analysis = self.analyses[ref_ind]
        rate_state_inp_mat = np.zeros((self.num_analyses, 4))
        rate_state_inp_mat[ref_ind, :] = np.nan
        true_lats = reference_analysis.get_latents(phase=phase)
        true_inputs = reference_analysis.get_inputs(phase=phase)
        is_multitask = reference_analysis.env.dataset_name == "MultiTask"
        if is_multitask:
            trial_lens = self.analyses[1].get_trial_lens(phase=phase)
            # Stack the latents to the different trial lengths
            true_lats_stack = []
            for i in range(trial_lens.shape[0]):
                true_lats_stack.append(true_lats[i, : int(trial_lens[i]), :])
            true_lats = torch.vstack(true_lats_stack)

        for i in range(self.num_analyses):
            print(
                f"Working on {i+1} of {self.num_analyses}: {self.analyses[i].run_name}"
            )
            print(f"Group: {self.groups[i]}")
            if i == ref_ind:
                continue
            rates, latents = self.analyses[i].get_model_outputs(phase=phase)
            true_rates = self.analyses[i].get_true_rates(phase=phase)

            rates_stack = []
            latents_stack = []
            true_rates_stack = []
            true_inputs_stack = []

            # If multitask (with different trial lengths)
            if is_multitask:
                trial_lens = self.analyses[i].get_trial_lens(phase=phase)
                # Stack the latents to the different trial lengths
                for j in range(latents.shape[0]):
                    latents_stack.append(latents[j, : int(trial_lens[j]), :])
                    rates_stack.append(rates[j, : int(trial_lens[j]), :])
                    true_rates_stack.append(true_rates[j, : int(trial_lens[j]), :])
                    true_inputs_stack.append(true_inputs[j, : int(trial_lens[j]), :])
                rates = torch.vstack(rates_stack)
                latents = torch.vstack(latents_stack)
                true_rates = torch.vstack(true_rates_stack)
                true_inputs = torch.vstack(true_inputs_stack)

            # Check that latents arenot NaN
            if np.isnan(latents.detach().numpy()).any():
                continue
            inf_inputs = self.analyses[i].get_inferred_inputs(phase=phase)
            # Compute the rate R2 and state R2
            rate_state_inp_mat[i, 0] = get_signal_r2(
                signal_true=true_rates,
                signal_pred=rates,
            )
            rate_state_inp_mat[i, 1] = get_signal_r2_linear(
                signal_true=true_lats,
                signal_pred=latents,
            )
            rate_state_inp_mat[i, 2] = get_signal_r2_linear(
                signal_true=true_inputs,
                signal_pred=inf_inputs,
            )
            rate_state_inp_mat[i, 3] = get_signal_r2_linear(
                signal_true=inf_inputs,
                signal_pred=true_inputs,
            )

            print(f"Rate R2: {rate_state_inp_mat[i, 0]}")
            print(f"State R2: {rate_state_inp_mat[i, 1]}")
            print(f"Input R2 (toInf): {rate_state_inp_mat[i, 2]}")
            print(f"Input R2 (toTrue): {rate_state_inp_mat[i, 3]}")

        # Sort results by the groups
        num_groups = len(np.unique(self.groups))
        colors = plt.cm.get_cmap("tab10", num_groups)
        color_inds = np.array(
            [np.where(np.unique(self.groups) == group)[0][0] for group in self.groups]
        )

        # Plot the results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in range(self.num_analyses):
            if i != ref_ind:
                ax.scatter(
                    rate_state_inp_mat[i, 0],
                    rate_state_inp_mat[i, 1],
                    rate_state_inp_mat[i, 2],
                    color=colors(color_inds[i]),
                    marker=marker,
                )
                if label_runs:
                    ax.text(
                        rate_state_inp_mat[i, 0],
                        rate_state_inp_mat[i, 1],
                        rate_state_inp_mat[i, 2],
                        self.analyses[i].run_name,
                    )
        if label_groups:
            for i in range(num_groups):
                ax.scatter(
                    [],
                    [],
                    color=colors(i),
                    label=np.unique(self.groups)[i],
                    marker=marker,
                )
            ax.legend()
        bool_idx = np.arange(self.num_analyses) != ref_ind
        min_val = np.min(rate_state_inp_mat[bool_idx]) - 0.2
        max_val = np.max(rate_state_inp_mat[bool_idx]) + 0.2
        max_val = np.min([max_val, 1.05])
        min_val = np.max([min_val, -1.05])
        if ax_lim is not None:
            min_val = ax_lim[0]
            max_val = ax_lim[1]
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.plot([min_val, max_val], [min_val, max_val], "k--")
        # Square axis
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Rate R2 ")
        ax.set_ylabel("State R2 ")
        ax.set_title(f"Rate-State R2 ({self.comparison_tag})")
        if save_pdf:
            plt.savefig(f"{self.comparison_tag}_rate_state_inp_r2.pdf")
        return rate_state_inp_mat

    def plot_trials_3d_reference_trialLens(
        self, num_trials, ref_ind=None, savePDF=False, angle=None
    ):
        if ref_ind is None:
            ref_ind = self.ref_ind
        if ref_ind is None and self.ref_ind is None:
            # Throw an error
            raise ValueError("No reference index provided")
        ref_lats = self.analyses[ref_ind].get_latents().detach().numpy()
        trial_lens = self.analyses[ref_ind].get_trial_lens().detach().numpy()
        for i in range(len(trial_lens)):
            ref_lats[i, int(trial_lens[i]) :, :] = np.nan
        pca = PCA()
        ref_lats_flat = ref_lats.reshape(
            ref_lats.shape[0] * ref_lats.shape[1], ref_lats.shape[2]
        )
        ref_lats_pca_flat = pca.fit_transform(ref_lats_flat)
        ref_lats_pca = ref_lats_pca_flat.reshape(ref_lats.shape)

        fig = plt.figure(figsize=(15, 5))
        axes = fig.subplots(1, self.num_analyses, subplot_kw={"projection": "3d"})
        # Make an array of axis ranges. Should be [-inf, inf] for three rows
        axis_ranges = np.zeros((3, 2))
        axis_ranges[:, 0] = np.inf
        axis_ranges[:, 1] = -1 * np.inf

        for i in range(self.num_analyses):
            latents = self.analyses[i].get_latents().detach().numpy()
            lats_flat = latents.reshape(
                latents.shape[0] * latents.shape[1], latents.shape[2]
            )
            reg = LinearRegression().fit(lats_flat, ref_lats_pca_flat)
            latents_pca_flat = reg.predict(lats_flat)
            latents_pca = latents_pca_flat.reshape(ref_lats_pca.shape)
            for j in range(num_trials):
                axes[i].plot(
                    latents_pca[j, :, 0],
                    latents_pca[j, :, 1],
                    latents_pca[j, :, 2],
                )
                # Update the axis ranges
            for k in range(3):
                axis_ranges[k, 0] = np.min(
                    [axis_ranges[k, 0], np.min(latents_pca[:, :, k])]
                )
                axis_ranges[k, 1] = np.max(
                    [axis_ranges[k, 1], np.max(latents_pca[:, :, k])]
                )

            axes[i].set_title(f"{self.analyses[i].run_name}")
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_zlabel("PC3")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_zticks([])
        # Set the axis limits
        for i in range(self.num_analyses):
            axes[i].set_xlim(axis_ranges[0, :])
            axes[i].set_ylim(axis_ranges[1, :])
            axes[i].set_zlim(axis_ranges[2, :])
            if angle is not None:
                axes[i].view_init(angle[0], angle[1])

        if savePDF:
            plt.savefig(f"{self.comparison_tag}_3dLats.pdf")
        return fig
