import io
import pickle
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from ctd.comparison.analysis.analysis import Analysis
from ctd.comparison.fixedpoints import find_fixed_points
from ctd.comparison.metrics import compute_jacobians, compute_lyaps
from ctd.data_modeling.extensions.LFADS.utils import send_batch_to_device


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class Analysis_DD(ABC, Analysis):
    @staticmethod
    def create(run_name, filepath, model_type="N/A"):
        if model_type == "SAE":
            return Analysis_DD_SAE(run_name, filepath, model_type)
        elif model_type == "LFADS":
            return Analysis_DD_LFADS(run_name, filepath, model_type)
        elif model_type == "External":
            return Analysis_DD_Ext(run_name, filepath)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def __init__(self, run_name, filepath, model_type):
        self.tt_or_dd = "dd"
        self.run_name = run_name
        self.model_type = model_type
        self.load_wrapper(filepath)

    def load_wrapper(self, filepath):
        if torch.cuda.is_available():
            with open(filepath + "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(filepath + "datamodule.pkl", "rb") as f:
                self.datamodule = pickle.load(f)
        else:
            with open(filepath + "model.pkl", "rb") as f:
                self.model = CPU_Unpickler(f).load()
            with open(filepath + "datamodule.pkl", "rb") as f:
                self.datamodule = CPU_Unpickler(f).load()

    def to_device(self, device):
        self.model.to(device)
        self.datamodule.to(device)

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
            _, inputs = self.get_model_inputs()
            latents = self.get_latents()
        else:
            latents = self.get_latents()
        latents = latents.to(device)
        inputs = inputs.to(device)
        m_device = self.model.device
        fps = find_fixed_points(
            model=self.get_dynamics_model(),
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
        self.model.to(m_device)
        return fps

    def plot_fps(
        self,
        inputs=None,
        num_traj=10,
        n_inits=1024,
        noise_scale=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        device="cuda",
        seed=0,
        compute_jacobians=True,
        q_thresh=1e-5,
    ):

        latents = self.get_model_outputs()[1].detach().cpu().numpy()
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
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        zero_flag = q_vals == 0
        q_vals[zero_flag] = 1e-15
        ax.hist(np.log10(q_vals), bins=100)
        ax.set_xlabel("log10(q)")
        ax.set_ylabel("Count")
        q_flag = q_vals < q_thresh
        pca = PCA(n_components=3)
        xstar_pca = pca.fit_transform(xstar)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.transform(lats_flat)
        lats_pca = lats_pca.reshape(latents.shape[0], latents.shape[1], 3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        # Make a color vector based on stability

        xstar_pca = xstar_pca[q_flag]
        is_stable = is_stable[q_flag]

        ax.scatter(
            xstar_pca[is_stable, 0],
            xstar_pca[is_stable, 1],
            xstar_pca[is_stable, 2],
            c="g",
        )
        ax.scatter(
            xstar_pca[~is_stable, 0],
            xstar_pca[~is_stable, 1],
            xstar_pca[~is_stable, 2],
            c="r",
        )

        for i in range(num_traj):
            ax.plot(
                lats_pca[i, :, 0],
                lats_pca[i, :, 1],
                lats_pca[i, :, 2],
            )
        ax.set_title(f"{self.model_type}_Fixed Points")
        plt.show()
        return fps

    def plot_trial(self, num_trials=10, scatterPlot=True):
        latents = self.get_latents().detach().numpy()
        pca = PCA(n_components=3)
        lats_flat = latents.reshape(-1, latents.shape[-1])
        lats_pca = pca.fit_transform(lats_flat)
        lats_pca = lats_pca.reshape(-1, latents.shape[1], 3)
        if scatterPlot:

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")
            for i in range(num_trials):
                ax.plot(
                    lats_pca[i, :, 0],
                    lats_pca[i, :, 1],
                    lats_pca[i, :, 2],
                )
            ax.set_title(f"{self.model_type}_Trial Latent Activity")
        else:
            fig = plt.figure(figsize=(10, 4 * num_trials))
            for i in range(num_trials):
                ax = fig.add_subplot(num_trials, 1, i + 1)
                ax.plot(lats_pca[i, :, 0])
                ax.plot(lats_pca[i, :, 1])
                ax.plot(lats_pca[i, :, 2])
            ax.set_title(f"{self.model_type}_Trial Latent Activity")

        plt.show()

    def get_inputs(self, phase="val"):
        _, inputs = self.get_model_inputs(phase=phase)

        return inputs

    def plot_rates(self, phase="val", neurons=[0], n_trials=5):
        gru_rates = self.get_rates(phase=phase)
        true_rates = self.get_true_rates(phase=phase)
        trial_lens = self.get_trial_lens(phase=phase)
        rates_stack = []
        true_rates_stack = []
        for i in range(len(trial_lens)):
            rates_stack.append(gru_rates[i][:].detach().cpu().numpy())
            true_rates_stack.append(true_rates[i][:].detach().cpu().numpy())

        fig, ax = plt.subplots(n_trials, len(neurons), figsize=(10, 10))
        for i in range(n_trials):
            for j in range(len(neurons)):
                neuron = neurons[j]
                if i == 0 and j == 0:
                    ax[i, j].plot(
                        rates_stack[i][:, neuron],
                        color="black",
                        label="Estimated Rates",
                    )
                    ax[i, j].plot(
                        true_rates_stack[i][:, neuron],
                        color="black",
                        linestyle="--",
                        label="True Rates",
                    )
                else:
                    ax[i, j].plot(rates_stack[i][:, neuron], color="black")
                    # Restart the color order
                    ax[i, j].plot(
                        true_rates_stack[i][:, neuron], color="black", linestyle="--"
                    )
                if i == 0:
                    ax[i, j].set_title(f"Neuron {neuron}")
        ax[0, 0].legend()
        plt.show()

    def plot_scree(self, max_pcs=10):
        latents = self.get_latents().detach().numpy()
        latents = latents.reshape(-1, latents.shape[-1])
        n_lats = latents.shape[-1]
        high_bound = np.min([n_lats, max_pcs])
        pca = PCA(n_components=high_bound)
        pca.fit(latents)
        exp_var = pca.explained_variance_ratio_
        exp_var_ext = np.zeros(max_pcs)
        exp_var_ext[:high_bound] = exp_var
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.plot(range(1, max_pcs + 1), exp_var_ext * 100, marker="o")
        ax.set_xlabel("PC #")
        ax.set_title("Scree Plot")
        ax.set_ylabel("Explained Variance (%)")
        ax2 = fig.add_subplot(122)
        ax2.plot(range(1, max_pcs + 1), np.cumsum(exp_var_ext) * 100)
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
        ax2.set_ylim(0, 105)
        plt.savefig(f"{self.run_name}_scree_plot.pdf")
        return exp_var_ext

    def compute_lyapunov_exp(self, phase="val", n_trials=None):
        # Get the latent activity
        latents = self.get_latents(phase=phase)
        # Get the inputs
        inputs = self.get_inputs(phase=phase)
        # Get the flow-field model
        cell = self.get_dynamics_model()
        #
        # Compute the Jacobians
        Jz, Ju, trial_idx = compute_jacobians(
            z=latents,
            u=inputs,
            f=cell,
            num_trials=n_trials,
        )

        # Compute the Lyapunov exponents
        les = compute_lyaps(
            Js=Jz,
            dt=1,
        )

        return les.mean(0), les.std(0)

    @abstractmethod
    def get_model_inputs(self):
        pass

    @abstractmethod
    def get_model_outputs(self):
        pass

    @abstractmethod
    def get_latents(self):
        pass

    @abstractmethod
    def get_dynamics_model(self):
        pass

    @abstractmethod
    def get_true_rates(self):
        pass

    @abstractmethod
    def get_rates(self):
        pass

    @abstractmethod
    def get_trial_lens(self):
        pass

    @abstractmethod
    def get_spiking(self):
        pass


class Analysis_DD_SAE(Analysis_DD):
    def get_model_inputs(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            dd_spiking = torch.cat(
                (dd_train_ds.tensors[0], dd_val_ds.tensors[0]), dim=0
            )
            dd_inputs = torch.cat((dd_train_ds.tensors[2], dd_val_ds.tensors[2]), dim=0)
        elif phase == "train":
            dd_spiking = self.datamodule.train_ds.tensors[0]
            dd_inputs = self.datamodule.train_ds.tensors[2]
        elif phase == "val":
            dd_spiking = self.datamodule.valid_ds.tensors[0]
            dd_inputs = self.datamodule.valid_ds.tensors[2]

        return dd_spiking, dd_inputs

    def get_model_outputs(self, phase="all"):
        dd_spiking, dd_inputs = self.get_model_inputs(phase=phase)
        dd_spiking = dd_spiking.to(self.model.device)
        dd_inputs = dd_inputs.to(self.model.device)
        log_rates, latents = self.model(dd_spiking, dd_inputs)
        return torch.exp(log_rates), latents

    def get_latents(self, phase="all"):
        _, latents = self.get_model_outputs(phase=phase)
        return latents

    def get_dynamics_model(self):
        return self.model.decoder.cell

    def get_true_rates(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            rates_train = dd_train_ds.tensors[6]
            rates_val = dd_val_ds.tensors[6]
            true_rates = torch.cat((rates_train, rates_val), dim=0)
        elif phase == "train":
            true_rates = self.datamodule.train_ds.tensors[6]
        elif phase == "val":
            true_rates = self.datamodule.valid_ds.tensors[6]
        return true_rates

    def get_rates(self, phase="all"):
        rates, _ = self.get_model_outputs(phase=phase)
        return rates

    def get_trial_lens(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            trial_lens = torch.cat(
                (dd_train_ds.tensors[3][:, -1], dd_val_ds.tensors[3][:, -1]), dim=0
            )
        elif phase == "train":
            trial_lens = self.datamodule.train_ds.tensors[3][:, -1]
        elif phase == "val":
            trial_lens = self.datamodule.valid_ds.tensors[3][:, -1]
        return trial_lens

    def get_spiking(self, phase="all"):
        if phase == "all":
            dd_train_ds = self.datamodule.train_ds
            dd_val_ds = self.datamodule.valid_ds
            dd_spiking = torch.cat(
                (dd_train_ds.tensors[1], dd_val_ds.tensors[1]), dim=0
            )
        elif phase == "train":
            dd_spiking = self.datamodule.train_ds.tensors[1]
        elif phase == "val":
            dd_spiking = self.datamodule.valid_ds.tensors[1]
        return dd_spiking


class Analysis_DD_LFADS(Analysis_DD):
    def get_trial_lens(self, phase="all"):
        dd_extra = []
        if phase == "all":
            train_dl = self.datamodule.train_dataloader(shuffle=False)
            val_dl = self.datamodule.val_dataloader()
            for batch in train_dl:
                # Move data to the right device
                train_extra = batch[1][3][:, -1]
                dd_extra.append(train_extra)
            for batch in val_dl:
                # Move data to the right device
                val_extra = batch[1][3][:, -1]
                dd_extra.append(val_extra)
        elif phase == "train":
            train_dl = self.datamodule.train_dataloader(shuffle=False)
            for batch in train_dl:
                # Move data to the right device
                train_extra = batch[1][3][:, -1]
                dd_extra.append(train_extra)
        elif phase == "val":
            val_dl = self.datamodule.val_dataloader()
            for batch in val_dl:
                # Move data to the right device
                val_extra = batch[1][3][:, -1]
                dd_extra.append(val_extra)
        dd_extra = torch.cat(dd_extra, dim=0)
        return dd_extra

    def get_model_inputs(self, phase="all"):
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            dd_inputs = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][0]
                inputs_train = batch[0][2]
                dd_spiking.append(spiking_train)
                dd_inputs.append(inputs_train)
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][0]
                inputs_val = batch[0][2]
                dd_spiking.append(spiking_val)
                dd_inputs.append(inputs_val)
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            dd_spiking = []
            dd_inputs = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][0]
                inputs_train = batch[0][2]
                dd_spiking.append(spiking_train)
                dd_inputs.append(inputs_train)
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            dd_inputs = []
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][0]
                inputs_val = batch[0][2]
                dd_spiking.append(spiking_val)
                dd_inputs.append(inputs_val)
        dd_spiking = torch.cat(dd_spiking, dim=0)
        dd_inputs = torch.cat(dd_inputs, dim=0)
        return dd_spiking, dd_inputs

    def get_true_rates(self, phase="all"):
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            dd_rates = []
            for batch in train_ds:
                # Move data to the right device
                rates_train = batch[1][2]
                dd_rates.append(rates_train)
            for batch in val_dataloader:
                # Move data to the right device
                rates_val = batch[1][2]
                dd_rates.append(rates_val)
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            dd_rates = []
            for batch in train_ds:
                # Move data to the right device
                rates_train = batch[1][2]
                dd_rates.append(rates_train)
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            dd_rates = []
            for batch in val_dataloader:
                # Move data to the right device
                rates_val = batch[1][2]
                dd_rates.append(rates_val)
        dd_rates = torch.cat(dd_rates, dim=0)
        return dd_rates

    def get_inferred_inputs(self, phase="all"):
        dd_inf_inputs = []
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()

            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])

            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_inf_inputs.append(output[4])
        dd_inf_inputs = torch.cat(dd_inf_inputs, dim=0)
        return dd_inf_inputs

    def get_model_outputs(self, phase="all"):
        dd_rates = []
        dd_latents = []
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])

            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            for batch in train_ds:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            for batch in val_dataloader:
                # Move data to the right device
                batch = send_batch_to_device(batch, self.model.device)
                # Compute model output
                output = self.model.predict_step(
                    batch=batch,
                    batch_ix=None,
                    sample_posteriors=False,
                )
                dd_rates.append(output[0])
                dd_latents.append(output[6])
        dd_rates = torch.cat(dd_rates, dim=0)
        dd_latents = torch.cat(dd_latents, dim=0)
        return dd_rates, dd_latents

    def get_rates(self, phase="all"):
        rates, _ = self.get_model_outputs(phase=phase)
        return rates

    def get_latents(self, phase="all"):
        rates, latents = self.get_model_outputs(phase=phase)
        return latents

    def get_dynamics_model(self):
        return self.model.decoder.rnn.cell.gen_cell

    def get_spiking(self, phase):
        if phase == "all":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][1]
                dd_spiking.append(spiking_train)
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][1]
                dd_spiking.append(spiking_val)
        elif phase == "train":
            train_ds = self.datamodule.train_dataloader(shuffle=False)
            dd_spiking = []
            for batch in train_ds:
                # Move data to the right device
                spiking_train = batch[0][1]
                dd_spiking.append(spiking_train)
        elif phase == "val":
            val_dataloader = self.datamodule.val_dataloader()
            dd_spiking = []
            for batch in val_dataloader:
                # Move data to the right device
                spiking_val = batch[0][1]
                dd_spiking.append(spiking_val)
        dd_spiking = torch.cat(dd_spiking, dim=0)
        return dd_spiking


class Analysis_DD_Ext(Analysis_DD):
    def __init__(self, run_name, filepath):
        self.tt_or_dd = "dd"
        self.run_name = run_name
        self.filepath = filepath

        self.train_true_rates = None
        self.train_true_latents = None
        self.eval_true_rates = None
        self.eval_true_latents = None

        self.load_data(filepath)

    def load_data(self, filepath):
        with h5py.File(filepath, "r") as h5file:
            # Check the fields
            print(h5file.keys())
            self.eval_rates = torch.Tensor(h5file["eval_rates"][()])
            self.eval_latents = torch.Tensor(h5file["eval_latents"][()])
            self.train_rates = torch.Tensor(h5file["train_rates"][()])
            self.train_latents = torch.Tensor(h5file["train_latents"][()])
            if "fixed_points" in h5file.keys():
                self.fixed_points = torch.Tensor(h5file["fixed_points"][()])
            else:
                self.fixed_points = None

    def get_latents(self, phase="all"):
        if phase == "train":
            return self.train_latents
        elif phase == "val":
            return self.eval_latents
        else:
            full_latents = torch.cat((self.train_latents, self.eval_latents), dim=0)
            return full_latents

    def get_rates(self, phase="all"):
        if phase == "train":
            return self.train_rates
        elif phase == "val":
            return self.eval_rates
        else:
            full_rates = torch.cat((self.train_rates, self.eval_rates), dim=0)
            return full_rates

    def get_true_rates(self, phase="all"):
        if phase == "train":
            return self.train_true_rates
        elif phase == "val":
            return self.eval_true_rates
        else:
            full_true_rates = torch.cat(
                (self.train_true_rates, self.eval_true_rates), dim=0
            )
            return full_true_rates

    def get_model_outputs(self, phase="all"):
        if phase == "train":
            return self.train_rates, self.train_latents
        elif phase == "val":
            return self.eval_rates, self.eval_latents
        else:
            return self.get_rates(), self.get_latents()

    def compute_FPs(self, latents, inputs):
        return None

    def add_true_rates(self, train_true_rates, eval_true_rates):
        self.train_true_rates = train_true_rates
        self.eval_true_rates = eval_true_rates

    def plot_fps(self):
        if self.fixed_points is None:
            print("No fixed points to plot")
            return
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        latents = self.get_latents(phase="val")
        fps = self.fixed_points
        for i in range(100):
            ax.plot(
                latents[i, :, 0],
                latents[i, :, 1],
                latents[i, :, 2],
                c="k",
                linewidth=0.1,
            )
        ax.scatter(fps[:, 0], fps[:, 1], fps[:, 2], c="r")
        ax.set_title(f"Fixed Points: {self.run_name}")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
