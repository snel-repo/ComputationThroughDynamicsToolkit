import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from ctd.task_modeling.model.rnn import ChaoticRateRNN
from ctd.task_modeling.simulator.neural_simulator import (
    apply_data_warp_sigmoid,
    fit_poisson_rates_from_latents,
    generate_samples,
)
from ctd.task_modeling.task_env.chaotic_delayed_matching import ChaoticDelayedMatching
from ctd.task_modeling.task_wrapper.task_wrapper import TaskTrainedWrapper


@dataclass
class DNMSEnvConfig:
    n_timesteps: int = 1050
    noise: float = 0.05
    baseline_range: Tuple[int, int] = (50, 50)
    cue1_range: Tuple[int, int] = (200, 200)
    delay1_range: Tuple[int, int] = (200, 200)
    cue2_range: Tuple[int, int] = (200, 200)
    delay2_range: Tuple[int, int] = (200, 200)
    response_range: Tuple[int, int] = (200, 200)
    cue_scale: float = 1.5


@dataclass
class DNMSTeacherConfig:
    hidden_size: int = 200
    recurrent_gain: float = 1.5
    noise_level: float = 0.0
    latent_ic_var: float = 0.05
    hidden_clip: float = 5.0
    alpha: float = 1.0 / 30.0
    use_bias: bool = False
    learnable_ics: bool = False
    init_hidden_dist: str = "uniform"
    init_hidden_uniform_bound: float = 0.1
    input_trainable: bool = False
    input_init_dist: str = "uniform"
    input_uniform_bound: float = 1.0
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 64
    train_samples: int = 2048
    valid_samples: int = 512
    max_epochs: int = 60
    min_epochs: int = 15
    early_stop_patience: int = 12
    target_accuracy: float = 0.96


@dataclass
class DNMSNeuronConfig:
    n_neurons_heldin: int = 64
    n_neurons_heldout: int = 32
    rect_func: str = "exp"
    fr_scaling: float = 100.0
    target_mean_rates: float = 0.01
    obs_noise: str = "pseudoPoisson"
    dispersion: float = 1.0


@dataclass
class DNMSExportConfig:
    total_samples: int = 4096
    train_frac: float = 0.7
    valid_frac: float = 0.15
    seed: int = 0
    run_tag: str = "ChaoticDelayedMatchingBenchmark"
    subfolder: str = "default"


def default_benchmark_config() -> Dict[str, dict]:
    return {
        "env": asdict(DNMSEnvConfig()),
        "teacher": asdict(DNMSTeacherConfig()),
        "neurons": asdict(DNMSNeuronConfig()),
        "export": asdict(DNMSExportConfig()),
    }


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(array):
    return torch.tensor(array, dtype=torch.float32)


def compute_dnms_accuracy(predictions, targets, extra):
    pred = np.asarray(predictions)[..., 0]
    tgt = np.asarray(targets)[..., 0]
    extra = np.asarray(extra)
    if extra.shape[1] >= 13:
        resp_on_col, resp_off_col = 8, 9
    elif extra.shape[1] >= 11:
        resp_on_col, resp_off_col = 6, 7
    else:
        raise ValueError(f"Unsupported DNMS extra shape: {extra.shape}")

    trial_acc = []
    margins = []
    for i in range(pred.shape[0]):
        resp_on = int(extra[i, resp_on_col])
        resp_off = int(extra[i, resp_off_col])
        pred_mean = float(np.mean(pred[i, resp_on:resp_off]))
        tgt_mean = float(np.mean(tgt[i, resp_on:resp_off]))
        pred_label = 1 if pred_mean >= 0 else -1
        tgt_label = 1 if tgt_mean >= 0 else -1
        trial_acc.append(float(pred_label == tgt_label))
        margins.append(abs(pred_mean))

    return {
        "task_accuracy": float(np.mean(trial_acc)),
        "mean_response_margin": float(np.mean(margins)),
        "trial_accuracy": np.asarray(trial_acc, dtype=np.float32),
        "response_margin": np.asarray(margins, dtype=np.float32),
    }


def split_indices(n_samples, train_frac, valid_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    n_valid = int(valid_frac * n_samples)
    n_train = min(max(n_train, 1), n_samples - 2)
    n_valid = min(max(n_valid, 1), n_samples - n_train - 1)
    train_inds = perm[:n_train]
    valid_inds = perm[n_train : n_train + n_valid]
    test_inds = perm[n_train + n_valid :]
    return train_inds, valid_inds, test_inds


def stack_dataset_for_training(dataset_dict):
    ds = TensorDataset(
        to_tensor(dataset_dict["ics"]),
        to_tensor(dataset_dict["inputs"]),
        to_tensor(dataset_dict["targets"]),
        torch.arange(dataset_dict["ics"].shape[0], dtype=torch.float32),
        to_tensor(dataset_dict["conds"]),
        to_tensor(dataset_dict["extra"]),
        to_tensor(dataset_dict["inputs_to_env"]),
        to_tensor(dataset_dict["true_inputs"]),
    )
    return ds


def rollout_model(wrapper, dataset_dict, batch_size=256):
    ds = stack_dataset_for_training(dataset_dict)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    latents = []
    with torch.no_grad():
        for batch in dl:
            ics = batch[0]
            inputs = batch[1]
            inputs_to_env = batch[6]
            out = wrapper.forward(ics, inputs, inputs_to_env=inputs_to_env)
            preds.append(out["controlled"].cpu())
            latents.append(out["latents"].cpu())
    return torch.cat(preds, dim=0).numpy(), torch.cat(latents, dim=0).numpy()


def simulate_neural_data_from_latents(latents, neuron_cfg: DNMSNeuronConfig, seed: int):
    n_trials, _, latent_dim = latents.shape
    total_neurons = neuron_cfg.n_neurons_heldin + neuron_cfg.n_neurons_heldout
    rng = np.random.default_rng(seed)

    n_stacks = int(np.ceil(total_neurons / latent_dim))
    perm_inds = np.concatenate([rng.permutation(latent_dim) for _ in range(n_stacks)])[
        :total_neurons
    ]

    readout = np.zeros((latent_dim, total_neurons), dtype=np.float32)
    for i in range(total_neurons):
        readout[perm_inds[i], i] = 1.0

    latents_perm = latents[:, :, perm_inds]
    activity = latents_perm.copy()

    if neuron_cfg.target_mean_rates is not None:
        activity, _, _ = fit_poisson_rates_from_latents(
            latents_perm,
            neuron_cfg.target_mean_rates,
            link=neuron_cfg.rect_func,
        )
        orig_mean = np.mean(activity, keepdims=True)
        orig_std = np.std(activity, keepdims=True)
    else:
        orig_mean = np.mean(activity, keepdims=True)
        orig_std = np.std(activity, keepdims=True)
        activity = (activity - orig_mean) / (
            neuron_cfg.fr_scaling * np.clip(orig_std, a_min=1e-6, a_max=None)
        )
        if neuron_cfg.rect_func == "sigmoid":
            activity = apply_data_warp_sigmoid(activity)
        elif neuron_cfg.rect_func == "exp":
            activity = np.exp(activity)
        elif neuron_cfg.rect_func == "softplus":
            activity = np.log1p(np.exp(activity))
        else:
            raise ValueError(f"Unsupported rect_func: {neuron_cfg.rect_func}")

    if neuron_cfg.obs_noise == "poisson":
        data = rng.poisson(activity).astype(np.float32)
    elif neuron_cfg.obs_noise == "pseudoPoisson":
        data = generate_samples(activity, neuron_cfg.dispersion, rng).astype(np.float32)
    else:
        raise ValueError(f"Unsupported obs_noise: {neuron_cfg.obs_noise}")

    return {
        "data": data.astype(np.float32),
        "activity": activity.astype(np.float32),
        "latents": latents.astype(np.float32),
        "perm_neurons": perm_inds.astype(np.int32),
        "readout": readout.astype(np.float32),
        "orig_mean": orig_mean.astype(np.float32),
        "orig_std": orig_std.astype(np.float32),
    }


def compute_initial_condition_sensitivity(
    model,
    inputs,
    hidden0,
    perturbation_scale=1.0e-3,
):
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    hidden_t = torch.tensor(hidden0, dtype=torch.float32)
    eps = torch.randn_like(hidden_t)
    eps = perturbation_scale * eps / eps.norm(dim=1, keepdim=True).clamp_min(1e-8)

    noise_level = getattr(model, "noise_level", 0.0)
    model.noise_level = 0.0

    def _roll(h0):
        hidden = h0.clone()
        latents = []
        for t in range(inputs_t.shape[1]):
            _, hidden = model(inputs_t[:, t, :], hidden)
            latents.append(hidden)
        return torch.stack(latents, dim=1)

    with torch.no_grad():
        lat_a = _roll(hidden_t)
        lat_b = _roll(hidden_t + eps)

    model.noise_level = noise_level

    delta = torch.linalg.norm(lat_b - lat_a, dim=-1).cpu().numpy()
    delta0 = np.maximum(delta[:, :1], 1e-8)
    growth = np.log(np.maximum(delta, 1e-8) / delta0)
    return {
        "mean_log_divergence": growth.mean(axis=0).astype(np.float32),
        "median_log_divergence": np.median(growth, axis=0).astype(np.float32),
        "final_mean_log_divergence": float(growth[:, -1].mean()),
        "perturbation_scale": float(perturbation_scale),
    }


def compute_lyapunov_proxy(model, latents):
    if not hasattr(model, "recW"):
        return {"mean_log_spectral_norm": 0.0, "time_series": np.zeros(1)}

    rec_w = model.recW.weight.detach().cpu().numpy()
    hidden = np.asarray(latents)
    deriv = 1.0 - np.square(np.clip(hidden, -1.0, 1.0))
    log_sigmas = np.zeros(hidden.shape[:2], dtype=np.float32)

    for i in range(hidden.shape[0]):
        for t in range(hidden.shape[1]):
            jac = deriv[i, t][:, None] * rec_w
            sigma = np.linalg.svd(jac, compute_uv=False)[0]
            log_sigmas[i, t] = np.log(np.maximum(sigma, 1e-8))

    return {
        "mean_log_spectral_norm": float(log_sigmas.mean()),
        "max_log_spectral_norm": float(log_sigmas.max()),
        "time_series": log_sigmas.mean(axis=0).astype(np.float32),
    }


def plot_trial_io(dataset_dict, predictions, n_trials=4):
    n_trials = min(n_trials, dataset_dict["inputs"].shape[0])
    t = np.arange(dataset_dict["inputs"].shape[1])
    fig, axes = plt.subplots(n_trials, 2, figsize=(12, 2.8 * n_trials), sharex=True)
    if n_trials == 1:
        axes = np.array([axes])

    for i in range(n_trials):
        extra = dataset_dict["extra"][i]
        axes[i, 0].plot(t, dataset_dict["inputs"][i], linewidth=1.0)
        axes[i, 0].set_title("Inputs")
        axes[i, 1].plot(
            t, dataset_dict["targets"][i, :, 0], label="target", linewidth=1.8
        )
        axes[i, 1].plot(t, predictions[i, :, 0], label="pred", linewidth=1.2)
        axes[i, 1].set_title("Decision output")
        axes[i, 1].legend(loc="upper right")
        for ax in axes[i]:
            ax.axvspan(int(extra[0]), int(extra[1]), color="tab:blue", alpha=0.12)
            ax.axvspan(int(extra[2]), int(extra[3]), color="tab:orange", alpha=0.10)
            ax.axvspan(int(extra[4]), int(extra[5]), color="tab:purple", alpha=0.10)
            ax.axvspan(int(extra[6]), int(extra[7]), color="tab:green", alpha=0.12)
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_condition_trajectories(latents, conds, max_trials_per_condition=10):
    flat = latents.reshape(-1, latents.shape[-1])
    pca = PCA(n_components=min(3, latents.shape[-1]))
    proj = pca.fit_transform(flat).reshape(latents.shape[0], latents.shape[1], -1)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    cond_labels = (conds[:, 0].astype(int) * 2 + conds[:, 1].astype(int)).astype(int)
    names = ["AA", "AB", "BA", "BB"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for label in range(4):
        inds = np.where(cond_labels == label)[0][:max_trials_per_condition]
        for ind in inds:
            curve = proj[ind]
            ax.plot(
                curve[:, 0], curve[:, 1], curve[:, 2], color=colors[label], alpha=0.5
            )
    ax.set_title("Teacher trajectories by cue pair (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(names, loc="upper right")
    fig.tight_layout()
    return fig


def plot_chaos_summary(chaos_metrics):
    t = np.arange(len(chaos_metrics["sensitivity"]["mean_log_divergence"]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, chaos_metrics["sensitivity"]["mean_log_divergence"])
    axes[0].set_ylabel("log divergence")
    axes[0].set_title("Sensitivity to initial conditions")
    axes[1].plot(t, chaos_metrics["lyapunov_proxy"]["time_series"])
    axes[1].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("mean log ||J||_2")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Local Lyapunov proxy")
    fig.tight_layout()
    return fig


class ChaoticDelayedMatchingBenchmark:
    def __init__(
        self,
        env_cfg=None,
        teacher_cfg=None,
        neuron_cfg=None,
        export_cfg=None,
        device=None,
    ):
        self.env_cfg = DNMSEnvConfig(**(env_cfg or {}))
        self.teacher_cfg = DNMSTeacherConfig(**(teacher_cfg or {}))
        self.neuron_cfg = DNMSNeuronConfig(**(neuron_cfg or {}))
        self.export_cfg = DNMSExportConfig(**(export_cfg or {}))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.env = None
        self.model = None
        self.wrapper = None

    def _make_env(self, seed):
        return ChaoticDelayedMatching(seed=seed, **asdict(self.env_cfg))

    def _make_wrapper(self):
        model = ChaoticRateRNN(
            latent_size=self.teacher_cfg.hidden_size,
            recurrent_gain=self.teacher_cfg.recurrent_gain,
            noise_level=self.teacher_cfg.noise_level,
            latent_ic_var=self.teacher_cfg.latent_ic_var,
            hidden_clip=self.teacher_cfg.hidden_clip,
            alpha=self.teacher_cfg.alpha,
            use_bias=self.teacher_cfg.use_bias,
            learnable_ics=self.teacher_cfg.learnable_ics,
            init_hidden_dist=self.teacher_cfg.init_hidden_dist,
            init_hidden_uniform_bound=self.teacher_cfg.init_hidden_uniform_bound,
            input_trainable=self.teacher_cfg.input_trainable,
            input_init_dist=self.teacher_cfg.input_init_dist,
            input_uniform_bound=self.teacher_cfg.input_uniform_bound,
        )
        model.init_model(input_size=self.env.input_dim, output_size=self.env.output_dim)
        wrapper = TaskTrainedWrapper(
            learning_rate=self.teacher_cfg.learning_rate,
            weight_decay=self.teacher_cfg.weight_decay,
        )
        wrapper.set_environment(self.env)
        wrapper.set_model(model)
        wrapper.to(self.device)
        return model, wrapper

    def fit_teacher(self):
        set_global_seed(self.export_cfg.seed)
        self.env = self._make_env(seed=self.export_cfg.seed)
        self.model, self.wrapper = self._make_wrapper()

        train_env = self._make_env(seed=self.export_cfg.seed)
        valid_env = self._make_env(seed=self.export_cfg.seed + 1)
        train_dataset = train_env.generate_dataset(self.teacher_cfg.train_samples)[0]
        valid_dataset = valid_env.generate_dataset(self.teacher_cfg.valid_samples)[0]
        train_ds = stack_dataset_for_training(train_dataset)
        valid_ds = stack_dataset_for_training(valid_dataset)

        train_dl = DataLoader(
            train_ds, batch_size=self.teacher_cfg.batch_size, shuffle=True
        )
        valid_dl = DataLoader(
            valid_ds, batch_size=self.teacher_cfg.batch_size, shuffle=False
        )

        best_state = None
        best_val = np.inf
        best_acc = 0.0
        patience = 0
        history = []

        optim = torch.optim.Adam(
            [p for p in self.wrapper.parameters() if p.requires_grad],
            lr=self.teacher_cfg.learning_rate,
            weight_decay=self.teacher_cfg.weight_decay,
        )

        for epoch in range(self.teacher_cfg.max_epochs):
            train_losses = []
            self.wrapper.train()
            for batch in train_dl:
                batch = [b.to(self.device) for b in batch]
                ics, inputs, targets, _, conds, extra, inputs_to_env, _ = batch
                out = self.wrapper.forward(ics, inputs, inputs_to_env=inputs_to_env)
                loss_dict = {
                    "controlled": out["controlled"],
                    "latents": out["latents"],
                    "actions": out["actions"],
                    "targets": targets,
                    "inputs": inputs,
                    "conds": conds,
                    "extra": extra,
                    "epoch": epoch,
                }
                loss = self.env.loss_func(loss_dict)
                if hasattr(self.model, "model_loss"):
                    loss = loss + self.model.model_loss(loss_dict)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), 1.0)
                optim.step()
                train_losses.append(loss.item())

            self.wrapper.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            all_extra = []
            with torch.no_grad():
                for batch in valid_dl:
                    batch = [b.to(self.device) for b in batch]
                    ics, inputs, targets, _, conds, extra, inputs_to_env, _ = batch
                    out = self.wrapper.forward(ics, inputs, inputs_to_env=inputs_to_env)
                    loss_dict = {
                        "controlled": out["controlled"],
                        "latents": out["latents"],
                        "actions": out["actions"],
                        "targets": targets,
                        "inputs": inputs,
                        "conds": conds,
                        "extra": extra,
                        "epoch": epoch,
                    }
                    loss = self.env.loss_func(loss_dict)
                    if hasattr(self.model, "model_loss"):
                        loss = loss + self.model.model_loss(loss_dict)
                    val_losses.append(loss.item())
                    all_preds.append(out["controlled"].cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                    all_extra.append(extra.cpu().numpy())

            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            all_extra = np.concatenate(all_extra, axis=0)
            metrics = compute_dnms_accuracy(all_preds, all_targets, all_extra)
            mean_train = float(np.mean(train_losses))
            mean_val = float(np.mean(val_losses))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_train,
                    "valid_loss": mean_val,
                    "valid_accuracy": metrics["task_accuracy"],
                }
            )

            improved = mean_val < best_val
            if improved:
                best_val = mean_val
                best_acc = metrics["task_accuracy"]
                best_state = {
                    "model": self.model.state_dict(),
                    "wrapper": self.wrapper.state_dict(),
                }
                patience = 0
            else:
                patience += 1

            if (
                epoch + 1 >= self.teacher_cfg.min_epochs
                and best_acc >= self.teacher_cfg.target_accuracy
                and patience >= 2
            ) or patience >= self.teacher_cfg.early_stop_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.wrapper.load_state_dict(best_state["wrapper"])

        return {
            "best_valid_loss": float(best_val),
            "best_valid_accuracy": float(best_acc),
            "history": history,
        }

    def generate_dataset(self):
        if self.wrapper is None:
            raise RuntimeError("fit_teacher must be called before generate_dataset")

        sim_env = self._make_env(seed=self.export_cfg.seed + 2)
        dataset_dict, extra_dict = sim_env.generate_dataset(
            self.export_cfg.total_samples
        )
        self.wrapper.eval()
        self.wrapper.to(self.device)

        predictions, latents = rollout_model(self.wrapper, dataset_dict)
        task_metrics = compute_dnms_accuracy(
            predictions=predictions,
            targets=dataset_dict["targets"],
            extra=dataset_dict["extra"],
        )
        neural = simulate_neural_data_from_latents(
            latents=latents,
            neuron_cfg=self.neuron_cfg,
            seed=self.export_cfg.seed,
        )

        chaos_trials = min(16, latents.shape[0])
        chaos_metrics = {
            "sensitivity": compute_initial_condition_sensitivity(
                self.model.to("cpu"),
                inputs=dataset_dict["inputs"][:chaos_trials],
                hidden0=self.model.init_hidden(batch_size=chaos_trials)
                .detach()
                .cpu()
                .numpy(),
            ),
            "lyapunov_proxy": compute_lyapunov_proxy(
                self.model.to("cpu"),
                latents=latents[:chaos_trials],
            ),
        }

        return {
            "dataset_dict": dataset_dict,
            "extra_dict": extra_dict,
            "predictions": predictions.astype(np.float32),
            "latents": latents.astype(np.float32),
            "task_metrics": task_metrics,
            "neural": neural,
            "chaos_metrics": chaos_metrics,
        }

    def save(self, results, root_dir=None):
        home_dir = os.environ.get("HOME_DIR", ".")
        root = Path(root_dir or Path(home_dir) / "content" / "datasets" / "dd")
        run_tag = self.export_cfg.run_tag
        if run_tag == "ChaoticDelayedMatchingBenchmark":
            run_tag = f"{datetime.now().strftime('%Y%m%d')}_{run_tag}"
        folder = root / run_tag / self.export_cfg.subfolder
        folder.mkdir(parents=True, exist_ok=True)

        filename = (
            f"heldin_{self.neuron_cfg.n_neurons_heldin}_"
            f"heldout_{self.neuron_cfg.n_neurons_heldout}_"
            f"obs_noise_{self.neuron_cfg.obs_noise}_"
            f"seed_{self.export_cfg.seed}"
        )
        h5_path = folder / f"{filename}.h5"
        teacher_path = folder / "teacher_state_dict.pt"
        metrics_path = folder / "benchmark_metrics.json"
        chaos_path = folder / "chaos_diagnostic.npz"
        config_path = folder / "benchmark_config.json"

        dataset_dict = results["dataset_dict"]
        neural = results["neural"]
        preds = results["predictions"]

        train_inds, valid_inds, test_inds = split_indices(
            n_samples=dataset_dict["ics"].shape[0],
            train_frac=self.export_cfg.train_frac,
            valid_frac=self.export_cfg.valid_frac,
            seed=self.export_cfg.seed,
        )
        splits = {"train": train_inds, "valid": valid_inds, "test": test_inds}

        with h5py.File(h5_path, "w") as h5file:
            for split_name, inds in splits.items():
                h5file.create_dataset(
                    f"{split_name}_encod_data",
                    data=neural["data"][inds, :, : self.neuron_cfg.n_neurons_heldin],
                )
                h5file.create_dataset(
                    f"{split_name}_recon_data",
                    data=neural["data"][inds],
                )
                h5file.create_dataset(
                    f"{split_name}_activity", data=neural["activity"][inds]
                )
                h5file.create_dataset(
                    f"{split_name}_latents", data=neural["latents"][inds]
                )
                h5file.create_dataset(
                    f"{split_name}_inputs", data=dataset_dict["inputs"][inds]
                )
                h5file.create_dataset(
                    f"{split_name}_true_inputs", data=dataset_dict["true_inputs"][inds]
                )
                h5file.create_dataset(
                    f"{split_name}_targets", data=dataset_dict["targets"][inds]
                )
                h5file.create_dataset(f"{split_name}_outputs", data=preds[inds])
                h5file.create_dataset(
                    f"{split_name}_extra", data=dataset_dict["extra"][inds]
                )
                h5file.create_dataset(
                    f"{split_name}_conds", data=dataset_dict["conds"][inds]
                )
                h5file.create_dataset(
                    f"{split_name}_ics", data=dataset_dict["ics"][inds]
                )
                h5file.create_dataset(f"{split_name}_inds", data=inds)

            h5file.create_dataset("readout", data=neural["readout"])
            h5file.create_dataset("orig_mean", data=neural["orig_mean"])
            h5file.create_dataset("orig_std", data=neural["orig_std"])
            h5file.create_dataset("perm_neurons", data=neural["perm_neurons"])

            meta = {
                "dataset_name": "ChaoticDelayedMatching",
                "decision_coding": {"match": -1.0, "nonmatch": 1.0},
                "task_accuracy": results["task_metrics"]["task_accuracy"],
                "mean_response_margin": results["task_metrics"]["mean_response_margin"],
            }
            for key, value in meta.items():
                h5file.attrs[key] = (
                    json.dumps(value) if isinstance(value, dict) else value
                )

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "wrapper_state_dict": self.wrapper.state_dict(),
            },
            teacher_path,
        )

        summary = {
            "task_accuracy": results["task_metrics"]["task_accuracy"],
            "mean_response_margin": results["task_metrics"]["mean_response_margin"],
            "lyapunov_proxy_mean_log_spectral_norm": results["chaos_metrics"][
                "lyapunov_proxy"
            ]["mean_log_spectral_norm"],
            "initial_condition_final_mean_log_divergence": results["chaos_metrics"][
                "sensitivity"
            ]["final_mean_log_divergence"],
            "dataset_file": str(h5_path),
            "teacher_file": str(teacher_path),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        np.savez(
            chaos_path,
            sensitivity_mean_log_divergence=results["chaos_metrics"]["sensitivity"][
                "mean_log_divergence"
            ],
            sensitivity_median_log_divergence=results["chaos_metrics"]["sensitivity"][
                "median_log_divergence"
            ],
            lyapunov_time_series=results["chaos_metrics"]["lyapunov_proxy"][
                "time_series"
            ],
            lyapunov_mean=np.float32(
                results["chaos_metrics"]["lyapunov_proxy"]["mean_log_spectral_norm"]
            ),
            lyapunov_max=np.float32(
                results["chaos_metrics"]["lyapunov_proxy"]["max_log_spectral_norm"]
            ),
        )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                default_benchmark_config()
                | {
                    "env": asdict(self.env_cfg),
                    "teacher": asdict(self.teacher_cfg),
                    "neurons": asdict(self.neuron_cfg),
                    "export": asdict(self.export_cfg),
                },
                f,
                indent=2,
            )

        return {
            "folder": str(folder),
            "dataset_file": str(h5_path),
            "teacher_file": str(teacher_path),
            "metrics_file": str(metrics_path),
            "chaos_file": str(chaos_path),
            "config_file": str(config_path),
        }


__all__ = [
    "ChaoticDelayedMatchingBenchmark",
    "compute_dnms_accuracy",
    "default_benchmark_config",
]
