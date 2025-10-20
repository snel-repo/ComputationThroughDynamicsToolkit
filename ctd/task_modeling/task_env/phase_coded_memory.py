import numpy as np
from gymnasium import spaces

from ctd.task_modeling.task_env.loss_func import PhaseCodedMemoryLoss
from ctd.task_modeling.task_env.task_env import DecoupledEnvironment


class PhaseCodedMemory(DecoupledEnvironment):
    """
    Environment for the two‐cue phase‐coded working memory task from Pals et al. (2024).
    - u(t) is a reference sine wave with
        random initial phase θ and trial‐specific frequency.
    - Trials last total_duration seconds (default 0.8 s),
        discretized into n_timesteps bins.
    - Trials consist of:
      1) Initial input of u(t) for one period of its oscillation.
      2) A 5‐bin pulse on either the in‐phase (channel 1)
          or anti‐phase (channel 2) line.
      3) Remainder: network should emit sin(2π·f·t + θ − offset),
        where offset is 0.2π (in‐phase) or 1.2π (anti‐phase).

    Inputs:
        channel 0: u(t) = sin(2π·f·τ + θ)
        channel 1: in‐phase pulse (5 bins)
        channel 2: anti‐phase pulse (5 bins)
    Output:
        channel 0: target oscillation with phase shift
    """

    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        total_duration: float = 0.8,
        dataset_name: str = "PhaseCodedMemory",
        post_stim_l2_weight: float = 0.0,
        lat_l2_full: bool = False,
        pred_loss_full: bool = False,
    ):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name = dataset_name
        self.n_timesteps = n_timesteps
        self.noise = noise
        self.total_duration = total_duration
        # Input / output dims
        self.input_dim = 3
        self.output_dim = 1
        # Gym spaces
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.output_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1.1, high=1.1, shape=(self.input_dim,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
        )
        self.input_labels = ["Reference", "In-phase cue", "Anti-phase cue"]
        self.output_labels = ["Memory"]
        # internal state
        self.t = 0
        self.theta = 0.0  # initial phase
        self.phase = 0.0  # theta minus offset
        self.phase_set = False
        self.lat_l2_full = lat_l2_full  # whether to apply L2 loss to all lats
        self.pred_loss_full = pred_loss_full  # whether to apply pred loss to all
        # loss and coupling
        self.loss_func = PhaseCodedMemoryLoss(
            post_stim_l2_weight=post_stim_l2_weight,
            lat_l2_full=lat_l2_full,
            pred_loss_full=pred_loss_full,
        )
        self.coupled_env = False

    def set_seed(self, seed: int):
        np.random.seed(seed)

    def reset(self):
        """
        Reset time and phase flag. θ is set in generate_trial.
        """
        self.t = 0
        self.phase_set = False
        return np.zeros(self.output_dim, dtype=np.float32)

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        One timestep. action = [u_t, cue_in, cue_anti].
        - On first nonzero cue: set self.phase = self.theta - offset.
        - Always return sin(2π·f·(t·dt) + self.phase).
        """
        u_t, cue_in, cue_anti = action
        if not self.phase_set:
            if cue_in != 0:
                offset = 0.2 * np.pi
                self.phase = self.theta - offset
                self.phase_set = True
            elif cue_anti != 0:
                offset = 1.2 * np.pi
                self.phase = self.theta - offset
                self.phase_set = True
        # compute output
        out = np.sin(2 * np.pi * self.freq * (self.t * self.dt) + self.phase)
        self.t += 1
        return out.astype(np.float32)

    def generate_trial(self):
        """
        Create one trial with:
          - random θ ∈ [0,2π)
          - sample period T_period ∈ Uniform[0.125,0.25]
          - set freq = 1/T_period
          - dt = total_duration / n_timesteps
          - initial u(t) for one T_period (start until bin = period_bins)
          - 5‐bin pulse on cue line
        Returns: inputs_noisy, outputs, inputs_clean, theta
        """
        # 1) sample initial phase and input frequency
        self.reset()
        self.theta = np.random.uniform(0, 2 * np.pi)
        T_period = np.random.uniform(0.125, 0.25)
        self.freq = 1.0 / T_period
        # 2) time step size
        self.dt = self.total_duration / self.n_timesteps
        # 4) build time vector and reference
        t = np.arange(self.n_timesteps) * self.dt
        ref = np.sin(2 * np.pi * self.freq * t + self.theta)
        sigA_ref = np.sin(
            2 * np.pi * self.freq * t + self.theta - 0.2 * np.pi
        )  # in-phase
        sigB_ref = np.sin(
            2 * np.pi * self.freq * t + self.theta - 1.2 * np.pi
        )  # anti-phase
        # 5) clean inputs array
        inputs_clean = np.zeros((self.n_timesteps, self.input_dim), dtype=np.float32)
        inputs_clean[:, 0] = ref.astype(np.float32)
        # 6) choose cue channel
        stim_type = np.random.choice([0, 1])
        # place 5‐bin pulse starting after one period
        stim_len = np.random.randint(60, 80)  # pulse length between 1 and 5 bins
        start = np.random.randint(60, 120)
        end = min(start + stim_len, self.n_timesteps)

        inputs_clean[start:end, 1 + stim_type] = 1.0
        # 7) simulate outputs
        self.reset()
        outputs = np.zeros((self.n_timesteps, self.output_dim), dtype=np.float32)
        for i in range(self.n_timesteps):
            if i < start:
                outputs[i, 0] = inputs_clean[i, 0]  # before cue, output is reference
            elif i >= start:
                if stim_type == 0:
                    outputs[i, 0] = sigA_ref[i]
                else:
                    outputs[i, 0] = sigB_ref[i]
        # 8) add noise
        inputs_noisy = inputs_clean + np.random.normal(
            loc=0.0, scale=self.noise, size=inputs_clean.shape
        ).astype(np.float32)
        sig_onset = start
        sig_offset = end
        return (
            inputs_noisy,
            outputs,
            inputs_clean,
            self.theta,
            sig_onset,
            sig_offset,
            stim_type,
        )

    def generate_dataset(self, n_samples: int):
        """
        Generate N trials ; returns dataset_dict, extra_dict.
        """
        ics_ds = np.zeros((n_samples, self.output_dim), dtype=np.float32)
        inputs_ds = np.zeros(
            (n_samples, self.n_timesteps, self.input_dim), dtype=np.float32
        )
        targets_ds = np.zeros(
            (n_samples, self.n_timesteps, self.output_dim), dtype=np.float32
        )
        true_inputs_ds = np.zeros_like(inputs_ds)
        conds_ds = np.zeros((n_samples, 1), dtype=np.float32)
        extras_ds = np.zeros((n_samples, 3), dtype=np.float32)
        for i in range(n_samples):
            (
                inp_noisy,
                out,
                inp_clean,
                theta,
                sig_onset,
                sig_offset,
                stim_type,
            ) = self.generate_trial()
            inputs_ds[i] = inp_noisy
            true_inputs_ds[i, :] = inp_clean
            targets_ds[i] = out
            conds_ds[i, 0] = theta
            extras_ds[i, 0] = sig_onset
            extras_ds[i, 1] = sig_offset
            extras_ds[i, 2] = stim_type
        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "targets": targets_ds,
            "conds": conds_ds,
            "extra": extras_ds,
            "true_inputs": true_inputs_ds,
            "inputs_to_env": np.zeros((n_samples, 0)),
        }
        extra_dict = {}
        return dataset_dict, extra_dict
