import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.fixedpoints import find_fixed_points


class Analysis_TT_MultiTask(Analysis_TT):
    def __init__(self, run_name, filepath, use_train_dm=False):
        # initialize superclass
        super().__init__(run_name, filepath)
        self.tt_or_dt = "tt"
        self.load_wrapper(filepath, use_train_dm)
        self.plot_path = (
            "/home/csverst/Github/CtDBenchmark/"
            f"ctd/comparison/plots/{self.run_name}/"
        )

    def get_task_flag(self, task_to_analyze, phase="all"):
        # Compute latent activity from task trained model
        task_list = self.datamodule.extra_data["task_names"]
        phase_dict = self.datamodule.extra_data["phase_dict"]

        train_inds = self.datamodule.train_ds.tensors[3].detach().numpy().astype(int)
        valid_inds = self.datamodule.valid_ds.tensors[3].detach().numpy().astype(int)
        all_inds = np.concatenate([train_inds, valid_inds])
        if phase == "all":
            inds = all_inds
        elif phase == "train":
            inds = train_inds
        elif phase == "val":
            inds = valid_inds

        task_list = [task_list[ind] for ind in inds]
        phase_dict = [phase_dict[ind] for ind in inds]

        task_flag = [task == task_to_analyze for task in task_list]
        phase_dict = [phase_dict[i] for i in range(len(phase_dict)) if task_flag[i]]
        return task_flag, phase_dict

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

    def get_model_outputs_noiseless(self, phase="all"):
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs_noiseless(phase=phase)
        dyn_noise = self.wrapper.dynamic_noise
        self.wrapper.dynamic_noise = 0.0
        out_dict = self.wrapper(tt_ics, tt_inputs, tt_targets)
        self.wrapper.dynamic_noise = dyn_noise
        return out_dict

    def get_data_from_phase(self, phase_task, phase, data):
        data_phase = []
        for i, phase_dict in enumerate(phase_task):
            start_idx = phase_dict[phase][0]
            end_idx = phase_dict[phase][1]
            data_phase.append(data[i, start_idx:end_idx, :])
        return data_phase

    def compute_fps_phase(
        self,
        phases,
        task_to_analyze,
        lr=1e-2,
        noise_scale=0.1,
        max_iters=50000,
        n_inits=2048,
        use_noisy=True,
        compute_jacobians=True,
    ):
        # Compute latent activity from task trained model
        task_flag, phase_task = self.get_task_flag(task_to_analyze, phase="val")
        tt_ics, tt_inputs, tt_targets = self.get_model_inputs(phase="val")
        true_inputs = self.get_true_inputs(phase="val")

        tt_ics = tt_ics[task_flag]
        tt_inputs = tt_inputs[task_flag]
        tt_targets = tt_targets[task_flag]
        true_inputs = true_inputs[task_flag]
        if use_noisy:
            out_dict = self.get_model_outputs(phase="val")
        else:
            out_dict = self.get_model_outputs_noiseless(phase="val")
        latents = out_dict["latents"]
        latents = latents[task_flag]

        lats_phase = []
        inputs_phase = []

        for i, phase_dict in enumerate(phase_task):
            # if phases is a list
            if isinstance(phases, str):
                start_idx = phase_dict[phases][0]
                fin_idx = phase_dict[phases][1]
                lats_phase.append(latents[i, start_idx:fin_idx, :])
                inputs_phase.append(true_inputs[i, start_idx:fin_idx, :])
            elif isinstance(phases, list):
                for phase in phases:
                    start_idx = phase_dict[phase][0]
                    fin_idx = phase_dict[phase][1]
                    lats_phase.append(latents[i, start_idx:fin_idx, :])
                    inputs_phase.append(true_inputs[i, start_idx:fin_idx, :])

        lats_phase = torch.cat(lats_phase)
        inputs_phase = torch.cat(inputs_phase)

        fps = find_fixed_points(
            model=self.wrapper.model.cell,
            state_trajs=lats_phase,
            inputs=inputs_phase,
            n_inits=n_inits,
            noise_scale=noise_scale,
            learning_rate=lr,
            max_iters=max_iters,
            device="cpu",
            seed=0,
            compute_jacobians=compute_jacobians,
        )

        return fps

    def plot_fps_phase_interpolation(
        self,
        task,
        phase1,
        phase2,
        n_interp_steps=10,
        lr=1e-3,
        noise_scale=0.0,
        max_iters_init=5000,
        max_iters=5000,
        n_inits=1024,
        use_noisy=False,
    ):
        task_flag, phase_task = self.get_task_flag(task)

        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        true_inputs = self.get_true_inputs()

        tt_ics = tt_ics[task_flag]
        tt_inputs = tt_inputs[task_flag]
        tt_targets = tt_targets[task_flag]
        true_inputs = true_inputs[task_flag]
        if use_noisy:
            inputs_fp = tt_inputs
        else:
            inputs_fp = true_inputs

        out_dict = self.wrapper(tt_ics, inputs_fp, tt_targets)
        latents = out_dict["latents"]
        # latents = latents[task_flag]
        true_inputs_1 = true_inputs[0, phase_task[0][phase1][0], :]
        true_inputs_2 = true_inputs[0, phase_task[0][phase2][0], :]

        true_inputs_interp = np.zeros((n_interp_steps, true_inputs_1.shape[0]))
        for i in range(true_inputs_1.shape[0]):
            true_inputs_interp[:, i] = np.linspace(
                true_inputs_1[i], true_inputs_2[i], n_interp_steps
            )
        true_inputs_interp = torch.Tensor(true_inputs_interp)

        lats_phase = []

        for i, phase_dict in enumerate(phase_task):
            start_idx = phase_dict[phase1][0]
            fin_idx = phase_dict[phase1][1]
            lats_phase.append(latents[i, start_idx:fin_idx, :])

        lats_phase = torch.cat(lats_phase)

        interp_fps = []
        for i, inputs1 in enumerate(true_inputs_interp):
            print(f"Computing fixed points for interp. {i} of {n_interp_steps}")
            if i == 0:
                max_iters1 = max_iters_init
            else:
                max_iters1 = max_iters
            fps = find_fixed_points(
                model=self.wrapper.model.cell,
                state_trajs=lats_phase,
                inputs=inputs1,
                n_inits=n_inits,
                noise_scale=noise_scale,
                learning_rate=lr,
                max_iters=max_iters1,
                device="cpu",
                seed=0,
            )
            interp_fps.append(fps)
            lats_phase = torch.Tensor(fps.xstar)

        all_fps = []
        for i, fps in enumerate(interp_fps):
            all_fps.append(fps.xstar)
        all_fps = np.stack(all_fps, axis=0)
        all_fps_flat = all_fps.reshape(-1, all_fps.shape[-1])
        pca = PCA(n_components=3)
        pca.fit(all_fps_flat)
        all_fps_pca = pca.transform(all_fps_flat)
        all_fps_pca = all_fps_pca.reshape(all_fps.shape[0], all_fps.shape[1], -1)

        colors = plt.cm.jet(np.linspace(0, 1, n_interp_steps))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i, fps in enumerate(interp_fps):
            ax.scatter(
                all_fps_pca[i, :, 0],
                all_fps_pca[i, :, 1],
                all_fps_pca[i, :, 2],
                c=colors[i],
                s=10,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(f"Fixed Points for {task} {phase1} to {phase2}")
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        plt.savefig(f"fps_{task}_{phase1}_{phase2}.png", dpi=300)
        print(len(interp_fps))
        return interp_fps

    def compute_fps_task_interpolation(
        self,
        task1,
        task2,
        phase,
        n_interp_steps=10,
        lr=1e-3,
        noise_scale=0.0,
        max_iters_init=5000,
        max_iters=5000,
        n_inits=2048,
        use_noisy=False,
    ):
        task_flag_1, phase_task_1 = self.get_task_flag(task1)
        task_flag_2, phase_task_2 = self.get_task_flag(task2)

        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        true_inputs = self.get_true_inputs()

        ics_1 = tt_ics[task_flag_1]
        inputs_1 = tt_inputs[task_flag_1]
        targets_1 = tt_targets[task_flag_1]
        true_inputs_1 = true_inputs[task_flag_1]

        if use_noisy:
            inputs_fp_1 = inputs_1
        else:
            inputs_fp_1 = true_inputs_1

        true_inputs_2 = true_inputs[task_flag_2]

        out_dict_1 = self.wrapper(ics_1, inputs_fp_1, targets_1)
        latents_1 = out_dict_1["latents"]

        true_inputs_1 = true_inputs[0, phase_task_1[0][phase][0], :]
        true_inputs_2 = true_inputs[0, phase_task_2[0][phase][0], :]

        true_inputs_interp = np.zeros((n_interp_steps, true_inputs_1.shape[0]))
        for i in range(true_inputs_1.shape[0]):
            true_inputs_interp[:, i] = np.linspace(
                true_inputs_1[i], true_inputs_2[i], n_interp_steps
            )
        true_inputs_interp = torch.Tensor(true_inputs_interp)

        lats_phase = []

        for i, phase_dict in enumerate(phase_task_1):
            start_idx = phase_dict[phase][0]
            fin_idx = phase_dict[phase][1]
            lats_phase.append(latents_1[i, start_idx:fin_idx, :])

        lats_phase = torch.cat(lats_phase)

        interp_fps = []
        for i, inputs1 in enumerate(true_inputs_interp):
            print(f"Computing fixed points for interp. {i} of {n_interp_steps}")
            if i == 0:
                max_iters1 = max_iters_init
            else:
                max_iters1 = max_iters

            fps = find_fixed_points(
                model=self.wrapper.model.cell,
                state_trajs=lats_phase,
                inputs=inputs1,
                n_inits=n_inits,
                noise_scale=noise_scale,
                learning_rate=lr,
                max_iters=max_iters1,
                device="cpu",
                seed=0,
            )
            interp_fps.append(fps)
            lats_phase = torch.Tensor(fps.xstar)
        return interp_fps

    def plot_latent_traj(
        self, task_to_analyze, phase_for_pca="stim1", plot_fps=True, debug_fp_traj=False
    ):
        # Get the model outputs
        out_dict = self.get_model_output()
        lats = out_dict["latents"]
        outputs = out_dict["controlled"]

        # Get the flag for the task (which trials are the correct task)
        task_flag, phase_task = self.get_task_flag(task_to_analyze)
        phase_names = phase_task[0].keys()
        num_phases = len(phase_names)

        # Get the latents and outputs for the task
        lats_task = lats[task_flag].detach().numpy()
        outputs = outputs[task_flag].detach().numpy()
        B, T, D = lats_task.shape

        # Get the readout matrix
        readout = self.wrapper.model.readout

        # Get the latent activity for the phase we want to do PCA on
        lats_phase = self.get_data_from_phase(phase_task, phase_for_pca, lats_task)
        lats_phase_flat = np.concatenate(lats_phase)

        # Compute PCA on the latents in the phase
        pca = PCA(n_components=3)
        pca.fit(lats_phase_flat)

        # Get the full trial latents in the phase PCA space
        lats_pca = pca.transform(lats_task.reshape(-1, D))
        lats_pca = lats_pca.reshape(B, T, -1)

        # If we want to get Fixed points
        if plot_fps:
            fps = {}
            xstar_pca = []
            fps_out = []
            q_star = []
            # For each phase, compute the fixed points
            for phase_name in phase_names:
                print(f"Computing fixed points for {phase_name}")
                # Returns fps and x_trajs in the original space
                fps[phase_name] = self.compute_fp_phase(
                    phase=phase_name, task_to_analyze=task_to_analyze
                )

                # Transform the fps and x_trajs into the PCA space
                xstar = fps[phase_name].xstar
                xstar_pca.append(pca.transform(xstar))

                fps_out.append(readout(torch.Tensor(xstar)))
                q_star.append(fps[phase_name].qstar)
            xstar_pca = np.stack(xstar_pca, axis=0)
            fps_out = np.stack(fps_out, axis=0)
            fps_mat = np.concatenate((xstar_pca[:, :, :2], fps_out[:, :, 1:2]), axis=2)
            q_star = np.stack(q_star, axis=0)
            # Set values of qstar that are zero to 1e-16 for plotting
            q_star[q_star == 0] = 1e-16
            if True:
                fig = plt.figure(figsize=(10 * num_phases, 10))
                for i, phase in enumerate(phase_names):
                    # Add histogram of qstar
                    ax = fig.add_subplot(1, num_phases, i + 1)
                    ax.hist(np.log10(q_star[i]), bins=50)
                    ax.set_xlabel("qstar")
                    ax.set_ylabel("Count")
                    ax.set_title(f"qstar for {task_to_analyze} {phase}")
                plt.savefig(f"qstar_{task_to_analyze}_{phase}.png", dpi=300)

        # window to only FPs whose q values are less than thresh
        thresh = 1e-5
        for i, phase in enumerate(phase_names):
            q_flag = q_star[i] < thresh
            fps_mat[i, ~q_flag, :] = np.nan
        # combine lats_pca 1 and 2 and output 1
        plot_mat = np.concatenate((lats_pca[:, :, :2], outputs[:, :, 1:2]), axis=2)

        fig = plt.figure(figsize=(10 * num_phases, 10))
        for i, phase in enumerate(phase_names):
            ax = fig.add_subplot(1, num_phases, i + 1, projection="3d")
            for j, phase_dict in enumerate(phase_task):
                start_idx = phase_dict[phase][0]
                end_idx = phase_dict[phase][1]
                ax.plot(
                    plot_mat[j, start_idx:end_idx, 0],
                    plot_mat[j, start_idx:end_idx, 1],
                    plot_mat[j, start_idx:end_idx, 2],
                    c="k",
                )
            if plot_fps:
                ax.scatter(
                    fps_mat[i, :, 0], fps_mat[i, :, 1], fps_mat[i, :, 2], s=10, c="r"
                )
            ax.set_xlabel(f"PC1 ({phase_for_pca})")
            ax.set_ylabel(f"PC2 ({phase_for_pca})")
            ax.set_zlabel("Output")
            ax.set_title(f"Latent Trajectory for {task_to_analyze} {phase}")
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])

        plt.savefig(f"LatentTraj_{task_to_analyze}_{phase}.png", dpi=300)

    def plot_task_trial(self, task, num_trials):
        task_flag_1, phase_task_1 = self.get_task_flag(task)

        tt_ics, tt_inputs, tt_targets = self.get_model_inputs()
        true_inputs = self.get_true_inputs()

        ics = tt_ics[task_flag_1]
        inputs = tt_inputs[task_flag_1]
        targets = tt_targets[task_flag_1]
        true_inputs = true_inputs[task_flag_1]

        ics = ics[:num_trials]
        inputs = inputs[:num_trials]
        targets = targets[:num_trials]
        true_inputs = true_inputs[:num_trials]

        trials_phase = phase_task_1[:num_trials]

        out_dict = self.wrapper(ics, inputs, targets)

        controlled = out_dict["controlled"].detach().numpy()
        inputs = inputs.detach().numpy()
        targets = targets.detach().numpy()
        true_inputs = true_inputs.detach().numpy()

        fig = plt.figure(figsize=(3 * num_trials, 6))
        for i in range(num_trials):
            end_ind = trials_phase[i]["response"][1]
            ax2 = fig.add_subplot(4, num_trials, i + 1)
            ax3 = fig.add_subplot(4, num_trials, i + num_trials + 1)
            ax4 = fig.add_subplot(4, num_trials, i + 2 * num_trials + 1)
            ax5 = fig.add_subplot(4, num_trials, i + 3 * num_trials + 1)

            for phase in trials_phase[i].keys():
                start_ind = trials_phase[i][phase][0]
                end_ind = trials_phase[i][phase][1]
                ax2.axvline(start_ind, c="k", linestyle="--")
                ax2.axvline(end_ind, c="k", linestyle="--")
                ax3.axvline(start_ind, c="k", linestyle="--")
                ax3.axvline(end_ind, c="k", linestyle="--")
                ax4.axvline(start_ind, c="k", linestyle="--")
                ax4.axvline(end_ind, c="k", linestyle="--")
                ax5.axvline(start_ind, c="k", linestyle="--")
                ax5.axvline(end_ind, c="k", linestyle="--")

            for j in range(controlled.shape[-1]):
                ax2.plot(controlled[i, :end_ind, j])

            for j in range(targets.shape[-1]):
                ax3.plot(targets[i, :end_ind, j])

            for j in range(inputs.shape[-1]):
                ax4.plot(inputs[i, :end_ind, j])

            for j in range(true_inputs.shape[-1]):
                ax5.plot(true_inputs[i, :end_ind, j])

            ax2.set_xlim([0, end_ind])
            ax3.set_xlim([0, end_ind])
            ax4.set_xlim([0, end_ind])
            ax5.set_xlim([0, end_ind])

            if i == 0:
                ax2.set_ylabel("Controlled")
                ax3.set_ylabel("Targets")
                ax4.set_ylabel("Inputs")
                ax5.set_ylabel("True Inputs")
            if i == 3:
                ax2.set_xlabel("Time")
                ax3.set_xlabel("Time")
                ax4.set_xlabel("Time")
                ax5.set_xlabel("Time")
            else:
                ax2.set_xlabel("")
                ax3.set_xlabel("")
                ax4.set_xlabel("")
                ax2.set_xticks([])
                ax3.set_xticks([])
                ax4.set_xticks([])

        plt.suptitle(f"TT Multitask {task} Trials")
        plt.show()
