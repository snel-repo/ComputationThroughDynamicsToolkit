# %%
import argparse
import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


# %%
def install_compatibility_stubs(repo_root: Path):
    os.environ.setdefault("HOME_DIR", str(repo_root))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *args, **kwargs: False
        dotenv_stub.find_dotenv = lambda *args, **kwargs: ""
        sys.modules["dotenv"] = dotenv_stub

    if "DSA" not in sys.modules:
        dsa_module = types.ModuleType("DSA")

        class DummyDSA:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("DSA is not installed in this environment.")

        dsa_module.DSA = DummyDSA
        dsa_stats = types.ModuleType("DSA.stats")
        dsa_stats.dsa_bw_data_splits = lambda *args, **kwargs: None
        dsa_stats.dsa_to_id = lambda *args, **kwargs: None
        sys.modules["DSA"] = dsa_module
        sys.modules["DSA.stats"] = dsa_stats

    try:
        import lightning_fabric.utilities.data as lightning_data

        if not hasattr(lightning_data, "AttributeDict"):

            class AttributeDict(dict):
                __getattr__ = dict.get
                __setattr__ = dict.__setitem__

            lightning_data.AttributeDict = AttributeDict
    except Exception:
        pass

    legacy_aliases = {
        "ctd.data_modeling.models.SAE.dyn_models_GRU": (
            "ctd.data_modeling.models.SAE.dyn_models_gru"
        ),
    }
    for legacy_name, target_name in legacy_aliases.items():
        if legacy_name not in sys.modules:
            try:
                sys.modules[legacy_name] = importlib.import_module(target_name)
            except Exception:
                pass


# %%
class SyntheticInferredAnalysis:
    def __init__(
        self,
        reference_analysis,
        inferred_dim=32,
        obs_dim=24,
        noise_scale=0.05,
        seed=0,
    ):
        self.run_name = f"synthetic_inferred_{inferred_dim}d"
        self.env = reference_analysis.env
        self._rng = np.random.RandomState(seed)
        self._inputs = {
            phase: reference_analysis.get_inputs(phase=phase).detach().cpu()
            for phase in ["train", "val", "all"]
        }

        reference_latents = {
            phase: reference_analysis.get_latents(phase=phase).detach().cpu()
            for phase in ["train", "val"]
        }

        latent_dim = reference_latents["train"].shape[-1]
        mixing = self._rng.normal(size=(latent_dim, inferred_dim))
        mixing /= np.linalg.norm(mixing, axis=0, keepdims=True) + 1e-8
        readout = self._rng.normal(scale=0.2, size=(inferred_dim, obs_dim))
        bias = self._rng.normal(scale=0.1, size=(obs_dim,))

        self._latents = {}
        self._rates = {}
        self._true_rates = {}
        self._spikes = {}

        for phase, latents in reference_latents.items():
            latents_np = latents.numpy()
            inferred_latents = latents_np @ mixing
            inferred_latents += noise_scale * self._rng.normal(
                size=inferred_latents.shape
            )
            inferred_latents = inferred_latents.astype(np.float32)

            true_latents_proj = latents_np @ mixing
            true_rates = np.log1p(np.exp(true_latents_proj @ readout + bias)) + 1e-3
            pred_rates = np.log1p(np.exp(inferred_latents @ readout + bias)) + 1e-3
            spikes = self._rng.poisson(true_rates).astype(np.float32)

            self._latents[phase] = torch.tensor(inferred_latents)
            self._true_rates[phase] = torch.tensor(true_rates.astype(np.float32))
            self._rates[phase] = torch.tensor(pred_rates.astype(np.float32))
            self._spikes[phase] = torch.tensor(spikes)

        self._inputs["all"] = torch.cat(
            [self._inputs["train"], self._inputs["val"]], dim=0
        )

    def _get_phase(self, store, phase):
        if phase == "all":
            return torch.cat([store["train"], store["val"]], dim=0)
        return store[phase]

    def get_inputs(self, phase="all"):
        return self._get_phase(self._inputs, phase)

    def get_model_outputs(self, phase="all"):
        return self._get_phase(self._rates, phase), self._get_phase(
            self._latents, phase
        )

    def get_true_rates(self, phase="all"):
        return self._get_phase(self._true_rates, phase)

    def get_spiking(self, phase="all"):
        return self._get_phase(self._spikes, phase)


# %%
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run DelayEmbedDistributionMetric through Comparison.compute_metrics on "
            "the packaged tt_3bff benchmark."
        )
    )
    parser.add_argument(
        "--tt-path",
        default="content/trained_models/task-trained/tt_3bff/",
        help="Path to the reference task-trained benchmark directory.",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["wasserstein", "kl"],
        default="wasserstein",
        help="Distributional distance to report in the comparison space.",
    )
    parser.add_argument(
        "--input-source",
        choices=["observations", "spikes", "rates", "latents"],
        default="observations",
        help=(
            "Reference signal defining the common delay-embedding space. "
            "'observations'/'spikes' compare observed spikes to model rates; "
            "'rates' compares true rates to model rates."
        ),
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=16,
        help="Optional PCA dimension for each source before delay embedding.",
    )
    parser.add_argument(
        "--n-delays",
        type=int,
        default=2,
        help="Number of delayed copies to append in the delay embedding.",
    )
    parser.add_argument(
        "--delay-lag",
        type=int,
        default=1,
        help="Lag, in time bins, between successive delay copies.",
    )
    parser.add_argument(
        "--temporal-bin-size",
        type=int,
        default=1,
        help="Optional temporal bin size applied before smoothing.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Optional moving-average smoothing window applied over time.",
    )
    parser.add_argument(
        "--covariance-reg",
        type=float,
        default=1e-5,
        help="Diagonal covariance regularization used in Gaussian fitting.",
    )
    parser.add_argument(
        "--inferred-dim",
        type=int,
        default=32,
        help="Latent dimensionality of the synthetic inferred analysis.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.05,
        help="Noise added to the synthetic inferred latents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the synthetic inferred analysis.",
    )
    return parser


def get_default_args():
    return build_parser().parse_args([])


# %%
def _resolve_repo_root():
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        pass
    for candidate in [Path.cwd(), *Path.cwd().resolve().parents]:
        if (candidate / "ctd").is_dir() and (candidate / "examples").is_dir():
            return candidate
    return Path.cwd().resolve()


def run_demo(args):
    repo_root = _resolve_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    install_compatibility_stubs(repo_root)

    from ctd.comparison.analysis.tt.tt import Analysis_TT
    from ctd.comparison.comparison import Comparison

    tt_path = Path(args.tt_path)
    if not tt_path.is_absolute():
        tt_path = (repo_root / tt_path).resolve()
    if not tt_path.exists():
        raise FileNotFoundError(f"Task-trained benchmark not found: {tt_path}")

    reference_analysis = Analysis_TT(run_name="tt_3bff", filepath=f"{tt_path}/")
    inferred_analysis = SyntheticInferredAnalysis(
        reference_analysis=reference_analysis,
        inferred_dim=args.inferred_dim,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    comparison = Comparison(comparison_tag="delay_embed_distribution_demo")
    comparison.load_analysis(
        reference_analysis,
        group="reference",
        reference_analysis=True,
    )
    comparison.load_analysis(inferred_analysis, group="synthetic")

    metric_cfg = {
        "input_source": args.input_source,
        "distance_metric": args.distance_metric,
        "pca_dim": args.pca_dim,
        "n_delays": args.n_delays,
        "delay_lag": args.delay_lag,
        "temporal_bin_size": args.temporal_bin_size,
        "smoothing_window": args.smoothing_window,
        "covariance_reg": args.covariance_reg,
        "random_state": args.seed,
    }

    metrics = comparison.compute_metrics(
        metric_dict_list={
            "state_r2": {},
            "delay_embed_dist": metric_cfg,
        }
    )

    print("\nComputed metrics:")
    print(f"  state_r2: {metrics['state_r2'][0]:.6f}")
    print(f"  delay_embed_dist: {metrics['delay_embed_dist'][0]:.6f}")

    return metrics


# %%
def _in_interactive_kernel():
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def main():
    parser = build_parser()
    if _in_interactive_kernel():
        # In Jupyter / VSCode cell mode the kernel injects extra argv entries
        # like --f=/run/.../kernel-*.json that argparse can't recognize.
        args, _ = parser.parse_known_args([])
    else:
        args = parser.parse_args()
    return run_demo(args)


if __name__ == "__main__":
    main()


# %%
# Cell-style entry point: running this cell in Jupyter / VSCode interactive
# mode runs the demo with default arguments. Edit `cell_overrides` to tweak.
if _in_interactive_kernel():
    cell_overrides = {}  # e.g. {"distance_metric": "kl", "input_source": "rates"}
    _args = build_parser().parse_args([])
    for _k, _v in cell_overrides.items():
        setattr(_args, _k, _v)
    metrics = run_demo(_args)
