import os

import h5py
import numpy as np
import torch


def softplus(x):
    return np.log1p(np.exp(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def solve_softplus_bias(z_col, target, eps=1e-12, tol=1e-6, max_iter=100):
    """
    Solve for b in mean(softplus(z + b)) = target for a single neuron column.
    """
    # initial guess via ratio heuristic
    mean_sp = softplus(z_col).mean()
    b = np.log(np.clip(target / (mean_sp + eps), a_min=eps, a_max=None))

    for _ in range(max_iter):
        shifted = z_col + b
        current = softplus(shifted).mean()
        error = current - target
        if abs(error) < tol:
            break
        grad = sigmoid(shifted).mean()
        if grad < eps:
            grad = eps
        delta = error / grad
        b -= 0.5 * delta  # damped Newton
    return b


def fit_poisson_rates_from_latents(
    latents,  # (B, T, N) numpy array
    target_mean_rates,  # scalar or (N,) desired per-bin rate
    link="exp",  # "exp" or "softplus"
    eps=1e-12,
):
    """
    Returns:
      rates: (B, T, N) after applying link (λ)
      biases: (N,) additive per-neuron bias b
      prelink: (B, T, N) the value z + b before link function
        (so exp(prelink) or softplus(prelink))
    """
    B, T, N = latents.shape

    # Broadcast scalar target to vector if needed
    if np.isscalar(target_mean_rates):
        target_mean_rates = np.full((N,), float(target_mean_rates), dtype=latents.dtype)
    else:
        target_mean_rates = np.asarray(target_mean_rates, dtype=latents.dtype).reshape(
            -1
        )
        if target_mean_rates.size == 1:
            target_mean_rates = np.full(
                (N,), target_mean_rates.item(), dtype=latents.dtype
            )
        elif target_mean_rates.size != N:
            raise ValueError(
                f"t_m_r must be sc or len {N}, got {target_mean_rates.shape}"
            )

    z = latents.reshape(-1, N)  # (M, N)
    prelink = np.zeros_like(z)
    rates = np.zeros_like(z)
    biases = np.zeros(N, dtype=z.dtype)

    if link == "exp":
        mean_expz = np.exp(z).mean(axis=0)  # (N,)
        safe_target = np.clip(target_mean_rates, a_min=eps, a_max=None)
        biases = np.log(safe_target / (mean_expz + eps))
        prelink = z + biases[None, :]  # (M, N)
        rates = np.exp(prelink)
    elif link == "softplus":
        for n in range(N):
            col = z[:, n]
            target = target_mean_rates[n]
            b_n = solve_softplus_bias(col, target, eps=eps)
            biases[n] = b_n
            prelink[:, n] = col + b_n
            rates[:, n] = softplus(prelink[:, n])
    else:
        raise ValueError(f"Unknown link '{link}'; use 'exp' or 'softplus'.")

    rates = rates.reshape(B, T, N)
    prelink = prelink.reshape(B, T, N)
    return rates, biases, prelink


def generate_samples(activity, index_of_dispersion, rng):
    """
    Generate samples from a distribution with a specified
    Fano factor for a given activity tensor.

    Parameters:
        activity (np.ndarray): A positive tensor of shape (B, T, N)
        representing the mean activity.

        index_of_dispersion (float):
        The desired Fano factor (variance-to-mean ratio).
        rng (np.random.Generator): The random number generator.

    Returns:
        np.ndarray: Array of generated samples with the same shape as activity.
    """
    if index_of_dispersion < 1:
        # Underdispersion using Binomial thinning
        variance = activity * index_of_dispersion
        p = variance / activity
        n = (activity / p).astype(int)  # Ensure n is an integer
        samples = rng.binomial(n, p)
    elif index_of_dispersion == 1:
        # Poisson distribution
        samples = rng.poisson(activity)
    else:
        # Overdispersion using Negative Binomial distribution
        p = activity / (activity + (index_of_dispersion - 1) * activity)
        r = activity**2 / ((index_of_dispersion - 1) * activity)
        samples = rng.negative_binomial(r, p)

    return samples.astype(float)


def sigmoidActivation(module, input):
    return 1 / (1 + module.exp(-1 * input))


def apply_data_warp_sigmoid(data):
    warp_functions = [sigmoidActivation, sigmoidActivation, sigmoidActivation]
    firingMax = [2, 2, 2, 2]
    numDims = data.shape[1]

    a = np.array(1)
    dataGen = type(a) == type(data)
    if dataGen:
        module = np
    else:
        module = torch

    for i in range(numDims):

        j = np.mod(i, len(warp_functions) * len(firingMax))
        # print(f'Max firing {firingMax[np.mod(j, len(firingMax))]}
        # warp {warp_functions[int(np.floor((j)/(len(warp_functions)+1)))]}')
        data[:, i] = firingMax[np.mod(j, len(firingMax))] * warp_functions[
            int(np.floor((j) / (len(warp_functions) + 1)))
        ](module, data[:, i])

    return data


class NeuralDataSimulator:
    def __init__(
        self,
        neuron_dict,
        embed_dict,
        noise_dict,
        trim_inds=None,
    ):

        """Initialize the neural data simulator
        Args:
            embed_dict (dict): Dictionary of embedding parameters
                fr_scaling (float): Scaling factor for firing rates
                    (higher values lead to lower firing rates)
                rect_func (str): Nonlinearity for rectification
                    options:
                        - "exp": Exponential
                        - "sigmoid": Sigmoid
                        - "softplus": Softplus
            noise_dict (dict): Dictionary of noise parameters
                obs_noise (str): Observation noise model
                    options:
                        - "poisson": Poisson noise
                        - "pseudoPoisson": Pseudo-Poisson noise
                dispersion (float): Dispersion parameter for pseudo-Poisson noise

            n_neurons (int): Number of neurons to simulate
            trim_inds (tuple): Tuple of start and end indices t
                o trim the data for simulation

        """
        self.neuron_dict = neuron_dict
        self.embed_dict = embed_dict
        self.noise_dict = noise_dict
        self.readout = None
        self.orig_mean = None
        self.orig_std = None
        self.trim_inds = trim_inds
        self.frozen_params = False

    def generate_simulated_data(self, task_trained_model, datamodule, seed):
        coupled = task_trained_model.task_env.coupled_env

        # Step 1: Get trajectories and model predictions
        if hasattr(datamodule, "train_ds"):
            train_ds = datamodule.train_ds
            valid_ds = datamodule.valid_ds

            ics = train_ds.tensors[0]
            inputs = train_ds.tensors[1]
            extra = train_ds.tensors[5]
            inputs_to_env = train_ds.tensors[6]

            ics_val = valid_ds.tensors[0]
            inputs_val = valid_ds.tensors[1]
            extra_val = valid_ds.tensors[5]
            inputs_to_env_val = valid_ds.tensors[6]

            ics = torch.cat((ics, ics_val), dim=0)
            inputs = torch.cat((inputs, inputs_val), dim=0)
            extra = torch.cat((extra, extra_val), dim=0)
            inputs_to_env = torch.cat((inputs_to_env, inputs_to_env_val), dim=0)
        else:
            ics = datamodule.stim_ds.tensors[0]
            inputs = datamodule.stim_ds.tensors[1]
            extra = datamodule.stim_ds.tensors[5]
            inputs_to_env = datamodule.stim_ds.tensors[6]

        output_dict = task_trained_model(ics, inputs, inputs_to_env=inputs_to_env)
        latents = output_dict["latents"]

        n_neurons_heldin = self.neuron_dict["n_neurons_heldin"]
        n_neurons_heldout = self.neuron_dict["n_neurons_heldout"]
        total_neurons = n_neurons_heldin + n_neurons_heldout

        if coupled:
            states = output_dict["states"]
            inputs = torch.concatenate((states, inputs), dim=-1).detach().numpy()
        n_trials, n_times, n_lat_dim = latents.shape
        latents = latents.detach().numpy()

        rng = np.random.default_rng(seed)
        # Step 2: Randomly permute latents (if more neurons than latents)
        if not self.frozen_params:
            num_stacks = np.ceil(total_neurons / n_lat_dim)
            for i in range(int(num_stacks)):
                perm_inds_stack = rng.permutation(n_lat_dim)
                if i == 0:
                    perm_inds = perm_inds_stack
                else:
                    perm_inds = np.concatenate((perm_inds, perm_inds_stack))
            self.perm_inds = perm_inds[:total_neurons]
            # Make the readout matrix
            readout = np.zeros((n_lat_dim, total_neurons))
            for i in range(total_neurons):
                readout[self.perm_inds[i], i] = 1
            self.readout = readout

        latents_perm = latents[:, :, self.perm_inds]
        activity = latents_perm[:, :, :total_neurons]
        perm_neurons = self.perm_inds[:total_neurons]

        if self.trim_inds is not None:
            latents = latents[:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
            inputs = inputs[:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
            activity = activity[:, self.trim_inds[0] - 1 : self.trim_inds[1], :]
            if hasattr(datamodule.data_env, "extra_timing_inds"):
                for ind1 in datamodule.data_env.extra_timing_inds:
                    extra[:, ind1] = extra[:, ind1] - self.trim_inds[0]

        # Get the first n_neurons indices and permute the latents
        n_times_cut = activity.shape[1]
        if (
            "target_mean_rates" in self.embed_dict
            and self.embed_dict["target_mean_rates"] is not None
        ):
            # Fit rates to target mean rates
            _, _, activity = fit_poisson_rates_from_latents(
                latents_perm,
                self.embed_dict["target_mean_rates"],
                link=self.embed_dict["rect_func"],
            )
            self.orig_mean = np.mean(activity, keepdims=True)
            self.orig_std = np.std(activity, keepdims=True)
        # Standardize and record original mean and standard deviations
        else:
            if not self.frozen_params:
                self.orig_mean = np.mean(activity, keepdims=True)
                self.orig_std = np.std(activity, keepdims=True)

            activity = (activity - self.orig_mean) / (
                self.embed_dict["fr_scaling"] * self.orig_std
            )

        # Step 3: Apply rectification to positive "rates"
        if self.embed_dict["rect_func"] == "sigmoid":
            rng = np.random.default_rng(seed)
            scaling_matrix = np.logspace(0.2, 1, (total_neurons))
            activity = activity * scaling_matrix[None, :]
            activity = apply_data_warp_sigmoid(activity)
        elif self.embed_dict["rect_func"] == "exp":
            activity = np.exp(activity)
        elif self.embed_dict["rect_func"] == "softplus":
            activity = np.log(1 + np.exp(activity))

        # Step 4: Sample spiking from noisy distribution
        if self.noise_dict["obs_noise"] == "poisson":
            data = np.random.poisson(activity).astype(float)
        elif self.noise_dict["obs_noise"] == "pseudoPoisson":
            dispersion = self.noise_dict["dispersion"]
            data = generate_samples(activity, dispersion, rng)

        # Step 5: Reshape into data tensor
        latents = latents.reshape(n_trials, n_times_cut, n_lat_dim)
        activity = activity.reshape(n_trials, n_times_cut, total_neurons)
        data = data.reshape(n_trials, n_times_cut, total_neurons)

        self.frozen_params = True

        sim_dict = {
            "data": data,
            "inputs": inputs,
            "activity": activity,
            "latents": latents,
            "extra": extra,
            "perm_neurons": perm_neurons,
            "readout": self.readout,
        }
        return sim_dict

    def simulate_neural_data(
        self, task_trained_model, datamodule, run_tag, subfolder, dataset_path, seed=0
    ):
        """
        Simulate neural data from a task-trained model

        TODO: REVISE

        Args:
            task_trained_model (TODO: dtype):
            datamodule (Union[BasicDataModule, TaskTrainedRNNDataModule]):
            run_tag (str):
            subfolder (str):
            dataset_path (str):
            seed (int): Random seed used in data generation

        Returns:
            data (np.ndarray):
            inputs (np.ndarray):
            activity (np.ndarray):
            latents (np.ndarray):
        """

        # Make a filename based on the system being modeled, the number of neurons,
        # the nonlinearity, the observation noise, the epoch number, the model type,
        # and the seed
        sim_dict = self.generate_simulated_data(task_trained_model, datamodule, seed)

        data = sim_dict["data"]
        inputs = sim_dict["inputs"]
        activity = sim_dict["activity"]
        latents = sim_dict["latents"]
        extra = sim_dict["extra"]
        perm_neurons = sim_dict["perm_neurons"]
        readout = sim_dict["readout"]

        n_trials, n_times, n_lat_dim = latents.shape

        # Step 6: Perform data splits into train/valid
        train_inds = range(0, int(0.8 * n_trials))
        valid_inds = range(int(0.8 * n_trials), n_trials)

        # Generate the filepath
        dd_folder = run_tag

        folder_path = os.path.join(dataset_path, dd_folder, subfolder)
        n_heldin = self.neuron_dict["n_neurons_heldin"]
        n_heldout = self.neuron_dict["n_neurons_heldout"]
        total_neurons = n_heldin + n_heldout
        filename = f"heldin_{n_heldin}_heldout_{n_heldout}"
        if self.embed_dict["rect_func"] not in ["exp"]:
            for key, val in self.embed_dict.items():
                filename += f"_{key}_{val}"

        if self.noise_dict["obs_noise"] not in ["poisson"]:
            for key, val in self.noise_dict.items():
                filename += f"_{key}_{val}"

        filename += f"_seed_{seed}"

        # Make the directory if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        fpath = os.path.join(folder_path, filename + ".h5")

        # Save the trajectories
        with h5py.File(fpath, "w") as h5file:
            h5file.create_dataset(
                "train_encod_data", data=data[train_inds, :, :n_heldin]
            )
            h5file.create_dataset(
                "valid_encod_data", data=data[valid_inds, :, :n_heldin]
            )

            h5file.create_dataset(
                "train_recon_data", data=data[train_inds, :, :total_neurons]
            )
            h5file.create_dataset(
                "valid_recon_data", data=data[valid_inds, :, :total_neurons]
            )

            h5file.create_dataset("train_inputs", data=inputs[train_inds])
            h5file.create_dataset("valid_inputs", data=inputs[valid_inds])

            h5file.create_dataset("train_activity", data=activity[train_inds])
            h5file.create_dataset("valid_activity", data=activity[valid_inds])

            h5file.create_dataset("train_latents", data=latents[train_inds])
            h5file.create_dataset("valid_latents", data=latents[valid_inds])

            h5file.create_dataset("train_extra", data=extra[train_inds])
            h5file.create_dataset("valid_extra", data=extra[valid_inds])

            h5file.create_dataset("train_inds", data=train_inds)
            h5file.create_dataset("valid_inds", data=valid_inds)

            h5file.create_dataset("readout", data=readout)
            h5file.create_dataset("orig_mean", data=self.orig_mean)
            h5file.create_dataset("orig_std", data=self.orig_std)

            h5file.create_dataset("perm_neurons", data=perm_neurons)
        return data, inputs, activity, latents
