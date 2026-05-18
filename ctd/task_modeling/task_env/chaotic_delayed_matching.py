import numpy as np
from gymnasium import spaces

from ctd.task_modeling.task_env.loss_func import ChaoticDelayedMatchingLoss
from ctd.task_modeling.task_env.task_env import DecoupledEnvironment


class ChaoticDelayedMatching(DecoupledEnvironment):
    """Delayed non-match-to-sample task with chaotic teacher-friendly dynamics."""

    def __init__(
        self,
        n_timesteps: int,
        noise: float,
        dataset_name: str = "ChaoticDelayedMatching",
        cue_scale: float = 1.5,
        baseline_range=(50, 50),
        cue1_range=(200, 200),
        delay1_range=(200, 200),
        cue2_range=None,
        delay2_range=None,
        response_range=(200, 200),
        seed: int = 0,
    ):
        super().__init__(n_timesteps=n_timesteps, noise=noise)

        self.dataset_name = dataset_name
        self.cue_scale = float(cue_scale)

        self.baseline_range = tuple(baseline_range)
        self.cue1_range = tuple(cue1_range)
        self.delay1_range = tuple(delay1_range)
        self.cue2_range = tuple(cue2_range or cue1_range)
        self.delay2_range = tuple(delay2_range or delay1_range)
        self.response_range = tuple(response_range)

        self.input_dim = 2
        self.output_dim = 1
        self.cue_names = ("A", "B")
        self.coupled_env = False
        self.single_neuron_readout = True
        self.readout_seed = int(seed)
        self.extra_timing_inds = list(range(10))

        self.input_labels = ["Cue A", "Cue B"]
        self.output_labels = ["Non-match (+1) vs Match (-1)"]

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.output_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.input_dim,), dtype=np.float32
        )
        self.context_inputs = spaces.Box(
            low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
        )

        self.loss_func = ChaoticDelayedMatchingLoss()

        self.state = np.zeros(self.output_dim, dtype=np.float32)
        self._time = 0
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = np.zeros(self.output_dim, dtype=np.float32)
        self._time = 0
        return self.state

    def step(self, action):
        self._time += 1
        self.state = np.asarray(action, dtype=np.float32)
        return self.state

    def _randint_from_range(self, v):
        lo, hi = int(v[0]), int(v[1])
        if hi <= lo:
            return lo
        return int(self.rng.integers(lo, hi + 1))

    def _sample_epochs(self):
        T = self.n_timesteps
        baseline_len = self._randint_from_range(self.baseline_range)
        cue1_len = self._randint_from_range(self.cue1_range)
        delay1_len = self._randint_from_range(self.delay1_range)
        cue2_len = self._randint_from_range(self.cue2_range)
        delay2_len = self._randint_from_range(self.delay2_range)
        response_len = self._randint_from_range(self.response_range)

        total = (
            baseline_len + cue1_len + delay1_len + cue2_len + delay2_len + response_len
        )
        if total > T:
            over = total - T
            delay2_len = max(6, delay2_len - over)

        cue1_on = min(baseline_len, T - 6)
        cue1_off = min(cue1_on + cue1_len, T - 5)
        delay1_on = cue1_off
        delay1_off = min(delay1_on + delay1_len, T - 4)
        cue2_on = delay1_off
        cue2_off = min(cue2_on + cue2_len, T - 3)
        delay2_on = cue2_off
        delay2_off = min(delay2_on + delay2_len, T - 2)
        resp_on = delay2_off
        resp_off = min(resp_on + response_len, T)

        if cue1_off <= cue1_on:
            cue1_off = min(cue1_on + 1, T)
        if delay1_off <= delay1_on:
            delay1_off = min(delay1_on + 1, T)
        if cue2_off <= cue2_on:
            cue2_off = min(cue2_on + 1, T)
        if delay2_off <= delay2_on:
            delay2_off = min(delay2_on + 1, T)
        if resp_off <= resp_on:
            resp_on = max(0, T - 2)
            resp_off = T

        epochs = np.zeros(T, dtype=np.int8)
        epochs[cue1_on:cue1_off] = 1
        epochs[delay1_on:delay1_off] = 2
        epochs[cue2_on:cue2_off] = 3
        epochs[delay2_on:delay2_off] = 4
        epochs[resp_on:resp_off] = 5

        return (
            cue1_on,
            cue1_off,
            delay1_on,
            delay1_off,
            cue2_on,
            cue2_off,
            delay2_on,
            delay2_off,
            resp_on,
            resp_off,
            epochs,
        )

    def generate_trial(self):
        cue1_id = int(self.rng.integers(0, 2))
        cue2_id = int(self.rng.integers(0, 2))
        nonmatch = float(cue1_id != cue2_id)
        target_value = 1.0 if nonmatch else -1.0

        (
            cue1_on,
            cue1_off,
            delay1_on,
            delay1_off,
            cue2_on,
            cue2_off,
            delay2_on,
            delay2_off,
            resp_on,
            resp_off,
            epochs,
        ) = self._sample_epochs()

        inputs_clean = np.zeros((self.n_timesteps, self.input_dim), dtype=np.float32)
        inputs_clean[cue1_on:cue1_off, cue1_id] = self.cue_scale
        inputs_clean[cue2_on:cue2_off, cue2_id] = self.cue_scale
        targets = np.zeros((self.n_timesteps, self.output_dim), dtype=np.float32)
        targets[resp_on:resp_off, 0] = target_value

        inputs_noisy = inputs_clean + self.rng.normal(
            loc=0.0, scale=self.noise, size=inputs_clean.shape
        ).astype(np.float32)

        return {
            "cue1_id": cue1_id,
            "cue2_id": cue2_id,
            "nonmatch": nonmatch,
            "target_value": target_value,
            "inputs": inputs_noisy,
            "true_inputs": inputs_clean,
            "targets": targets,
            "epochs": epochs,
            "cue1_on": cue1_on,
            "cue1_off": cue1_off,
            "delay1_on": delay1_on,
            "delay1_off": delay1_off,
            "cue2_on": cue2_on,
            "cue2_off": cue2_off,
            "delay2_on": delay2_on,
            "delay2_off": delay2_off,
            "resp_on": resp_on,
            "resp_off": resp_off,
        }

    def generate_dataset(self, n_samples: int):
        n_samples = int(n_samples)

        ics_ds = np.zeros((n_samples, self.output_dim), dtype=np.float32)
        inputs_ds = np.zeros(
            (n_samples, self.n_timesteps, self.input_dim), dtype=np.float32
        )
        true_inputs_ds = np.zeros_like(inputs_ds)
        targets_ds = np.zeros(
            (n_samples, self.n_timesteps, self.output_dim), dtype=np.float32
        )

        conds_ds = np.zeros((n_samples, 3), dtype=np.float32)
        extra_ds = np.zeros((n_samples, 13), dtype=np.float32)

        cue_pairs = np.zeros((n_samples, 2), dtype=np.int16)
        epoch_labels = np.zeros((n_samples, self.n_timesteps), dtype=np.int8)
        target_response = np.zeros((n_samples,), dtype=np.float32)

        for i in range(n_samples):
            trial = self.generate_trial()

            inputs_ds[i] = trial["inputs"]
            true_inputs_ds[i] = trial["true_inputs"]
            targets_ds[i] = trial["targets"]

            conds_ds[i, 0] = trial["cue1_id"]
            conds_ds[i, 1] = trial["cue2_id"]
            conds_ds[i, 2] = trial["nonmatch"]

            extra_ds[i, 0] = trial["cue1_on"]
            extra_ds[i, 1] = trial["cue1_off"]
            extra_ds[i, 2] = trial["delay1_on"]
            extra_ds[i, 3] = trial["delay1_off"]
            extra_ds[i, 4] = trial["cue2_on"]
            extra_ds[i, 5] = trial["cue2_off"]
            extra_ds[i, 6] = trial["delay2_on"]
            extra_ds[i, 7] = trial["delay2_off"]
            extra_ds[i, 8] = trial["resp_on"]
            extra_ds[i, 9] = trial["resp_off"]
            extra_ds[i, 10] = trial["cue1_id"]
            extra_ds[i, 11] = trial["cue2_id"]
            extra_ds[i, 12] = trial["nonmatch"]

            cue_pairs[i] = [trial["cue1_id"], trial["cue2_id"]]
            epoch_labels[i] = trial["epochs"]
            target_response[i] = trial["target_value"]

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "targets": targets_ds,
            "conds": conds_ds,
            "extra": extra_ds,
            "true_inputs": true_inputs_ds,
            "inputs_to_env": np.zeros(
                (n_samples, self.n_timesteps, 0), dtype=np.float32
            ),
        }

        extra_dict = {
            "cue_pairs": cue_pairs,
            "epoch_labels": epoch_labels,
            "target_response": target_response,
            "metadata": {
                "dataset_name": self.dataset_name,
                "task_definition_only": True,
                "cue_names": list(self.cue_names),
                "decision_coding": {"match": -1.0, "nonmatch": 1.0},
            },
        }
        return dataset_dict, extra_dict


__all__ = ["ChaoticDelayedMatching"]
