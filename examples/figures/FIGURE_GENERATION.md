# Figure Generation Index

This document tracks how each panel in the CtDToolkit paper figures is produced
by code in this repository.

**Run everything at once.** The umbrella orchestrator
[make_all_paper_figures.py](make_all_paper_figures.py) executes every
main-text figure script/notebook and every supplementary script registered
below. Each item runs in its own subprocess so a single failure does not abort
the rest; a summary table is printed at the end. Common invocations:

```bash
# Run all main-text + supplementary figures (scripts + notebooks)
python examples/figures/make_all_paper_figures.py

# Scripts only (skip nbconvert; much faster)
python examples/figures/make_all_paper_figures.py --no-notebooks

# A subset by name (see --list for available job names)
python examples/figures/make_all_paper_figures.py --only fig5 fig6 supp_3bff

# List jobs without running anything
python examples/figures/make_all_paper_figures.py --list
```

All registered jobs are `# %%`-delimited Python scripts (`make_*.py`). Notebook
execution support is retained in the runner so notebook jobs can still be
mixed in if needed, but every default paper figure is a script.

**Layout convention.** Every figure directory has an `outputs/` subfolder.
Generated artifacts use a `panel<X>_<descriptor>.<ext>` naming scheme so it is
obvious which paper panel each file corresponds to. Stale exploratory PDFs that
don't map to a current panel have been moved under each figure dir's `old/`
folder; nothing currently floats at the top level except notebooks, scripts,
caches (`*.pkl`), and the `outputs/` directory itself.

Paths below are relative to `examples/figures/`.

---

## Figure 1 — CtD framework and failure modes

**Source:** [Fig1PerfCriteria/make_figure1_perf_criteria.py](Fig1PerfCriteria/make_figure1_perf_criteria.py) (auto-generated from the original `Fig1OneBitFlipFlopGen.ipynb`; the `.ipynb` is kept for reference only).

All four cells in this script contribute. Markdown headers (as `# %% [markdown]`
blocks) label each panel; inline `# === PANEL X ===` comments mark each savefig
inside the long cells.

| Panel | Sub-row | Cell | Output (`Fig1PerfCriteria/outputs/`) |
| --- | --- | --- | --- |
| **B (Computation, green)** | inputs `u` and output `p` over time | 0 | `panelB_computation_inputs.pdf` |
| **B (Algorithm, orange)** | state-space flow field with stable/unstable FPs | 0 | `panelB_algorithm_flow_field.pdf` |
| **B (Implementation, blue)** | 3D linear embedding of 1D latent dynamics | 0 | `panelB_implementation_3d_embedding.pdf` |
| **B (Implementation, blue)** | 2-neuron simulated firing rates | 0 | `panelB_implementation_rates.pdf` |
| **F (Underfitting)** | top — true vs. underfit `u`/`z` | 1 | `panelF_underfitting_inputs.pdf` |
| **F (Underfitting)** | bottom — ideal vs. underfit state-space | 1 | `panelF_underfitting_flow_field.pdf` |
| **G (Invented Features)** | top — extra-latent traces | 0 | `panelG_invented_features_rates.pdf` |
| **G (Invented Features)** | bottom — 3D state-space with invented FP "B" | 0 | `panelG_invented_features_3d_embedding.pdf` |
| **H (Dynamical Misattribution)** | top — inferred inputs don't match true | 0 | `panelH_misattribution_rates.pdf` |
| **H (Dynamical Misattribution)** | bottom — ideal vs. poor-input flow field | 0 | `panelH_misattribution_flow_field.pdf` |

Panels A, C, D, E are schematic and were built in Illustrator.

---

## Figure 2 — Schematic

Built in Illustrator. No generating code in this repo.

---

## Figure 3 — Dataset dynamics

**Compiled source:** [Fig3TaskPerformance/make_figure3_task_performance.py](Fig3TaskPerformance/make_figure3_task_performance.py) (auto-generated from `Figure3_Combined.ipynb`).

Each section is preceded by a `# %% [markdown]` header naming the corresponding
paper panel. Original task-specific notebooks are still in the folder for
reference; the MultiTask FP gif is produced by
[make_figure3_multitask_animation.py](Fig3TaskPerformance/make_figure3_multitask_animation.py)
(auto-generated from `Figure3MultiTask.ipynb`).

| Panel | Description | Output (`Fig3TaskPerformance/outputs/`) |
| --- | --- | --- |
| **A** | 3BFF task schematic | (Illustrator) |
| **B** | 3BFF single-trial inputs/outputs | `panelB_3bff_io.pdf` |
| **C** | 3BFF canonical TT latents + cube FPs | `panelC_3bff_fps.pdf` |
| **D** | MultiTask schematic (MemoryPro/Anti) | (Illustrator) |
| **E** | MultiTask MemoryPro single-trial I/O | `panelE_multitask_io.pdf` |
| **F** | MultiTask FP rings (Mem1, Resp phases) | `panelF_multitask_fps_overlay.png`, `panelF_multitask_fps_animation.gif`, `panelF_frames/` |
| **G** | RandomTarget bump-perturbed hand kinematics | `panelG_rt_kinematics.pdf` |
| **H** | RandomTarget single-trial I/O | `panelH_rt_io.pdf` |
| **I** | RandomTarget latents in Pec-projection plane | `panelI_rt_pec_projection.pdf` |

> The MultiTask FP animation (gif + frames) is produced by the
> `make_figure3_multitask_animation.py` script; the combined script only
> generates the static overlay PNG.

---

## Figure 4 — TT/DD pipeline + inferred latents

**Compiled source:** [Fig4Canonical/make_figure4_canonical.py](Fig4Canonical/make_figure4_canonical.py) (auto-generated from `Figure4_Combined.ipynb`).

| Panel | Description | Output (`Fig4Canonical/outputs/`) |
| --- | --- | --- |
| **A** | TT learning progression (epochs 10/50/100/250/500) | `panelA_tt_learning_progression_trial<N>.pdf` |
| **B** | 3BFF latents — TT vs LFADS / GRU / LDS | `panelB_3bff_single_3dlats.pdf` |
| **C** | MultiTask MemoryPro latents (TT vs DD) | `panelC_multitask_latents.pdf`, `panelC_multitask_single_3dlats.pdf`, `panelC_multitask_radial.pdf` |
| **D** | RandomTarget latents colored by reach angle | `panelD_rt_radial.pdf`, `panelD_rt_single_3dlats.pdf` |
| Supporting | per-dataset Rate R² / State R² metric scatter | `supp_3bff_rate_state_r2.pdf`, `supp_multitask_rate_state_r2.pdf`, `supp_rt_rate_state_r2.pdf` |

> The `_single_3dlats.pdf` files are written by `Comparison.plot_trials_3d_reference(savePDF=True)`,
> which always saves into the current working directory. To keep them in
> `outputs/`, the relevant cells wrap the call in a small `_to_outputs()`
> context manager that chdir's to `outputs/` for the duration of the call.

---

## Figure 5 — Reconstruction & simplicity metrics

**Source:** [Fig5Metrics/make_figure5_reconstruction_simplicity.py](Fig5Metrics/make_figure5_reconstruction_simplicity.py)

Run with `python make_figure5_reconstruction_simplicity.py` (add `--noiseless`
for the noise-free variant). Output goes to `Fig5Metrics/outputs/` by default
(override with `--output-dir`).

| Panel | Description | Function | Output (`Fig5Metrics/outputs/`) |
| --- | --- | --- | --- |
| **A** | Schematic of reconstruction metrics | `draw_placeholder(...)` | bundled in `figure5.pdf` |
| **B** | Predicted rate of held-out neuron (NODE-3 vs NODE-8) | `draw_panel_b()` | bundled |
| **C** | Rate R² vs co-BPS scatter | `scatter_metrics(...)` | bundled |
| **D** | Schematic of simplicity metrics | `draw_placeholder(...)` | bundled |
| **E** | Inferred PC8 + linear predictions | `draw_panel_e()` | bundled |
| **F** | State R² vs cycle-con scatter | `scatter_metrics(...)` | bundled |
| **G** | Rate R² vs state R² (ground-truth) | `scatter_metrics(...)` | bundled |
| **H** | co-BPS vs cycle-con (no ground-truth) | `scatter_metrics(...)` | bundled |
| **I** | Schematic underfitting / invented features | `draw_panel_i()` | bundled |

Single bundled file: `figure5.pdf` (or `figure5_noiseless.pdf`).

This script is the reference for our figure-script conventions:
panel-labeled `draw_panel_*` functions, `set_panel_label`, cached payload,
`--noiseless` / `--force` flags.

---

## Figure 6 — Inferred inputs affect inferred dynamics

**Compiled source:** [Fig6InputInf/make_figure6_combined.py](Fig6InputInf/make_figure6_combined.py)

| Panel | Description | Output (`Fig6InputInf/outputs/`) |
| --- | --- | --- |
| **A** | Input-inference architecture schematic | (Illustrator) |
| **B** | True / effective / inferred inputs for one trial (good vs bad) | `panelB_input_traces.pdf`, `panelB_effective_input_traces.pdf` |
| **C** | co-BPS vs Input R² scatter (KL sweep) | `panelCD_metric_scatters.pdf` |
| **D** | cycle-consistency vs Input R² scatter | `panelCD_metric_scatters.pdf` (same figure) |
| **E** | Inferred FPs for "bad-input" model | `fps_Worst Inputs.pdf` (via `plot_model_fps` helper) |
| **F** | Inferred FPs for "good-input" model | `fps_Best Inputs.pdf` (via `plot_model_fps` helper) |
| Supporting (FPFinding) | re-derive FPs with mean inferred inputs | `supp_fpfinding_latents.pdf`, `supp_fpfinding_inputs.pdf` |

> The `plot_model_fps(...)` helper writes `fps_{label}.pdf` into `outputs/`;
> the label is set at the call site (`"Best Inputs"` / `"Worst Inputs"`).

---

## Supplementary figures

Scripts live in [supplementary/](supplementary/) and write into
`supplementary/outputs/`. Each is a runnable, notebook-friendly Python file
following the same conventions as Figure 5 (argparse, `outputs/` default,
optional pickle cache + `--force`).

| Script | Description | Output stems (`supplementary/outputs/`) |
| --- | --- | --- |
| [make_3bff_trial_fig.py](supplementary/make_3bff_trial_fig.py) | Single-trial 3BFF I/O traces (FigS2) | `FigureS2_3BFF_trial.{pdf,svg,png}` |
| [make_phase_coded_memory_figure.py](supplementary/make_phase_coded_memory_figure.py) | PhaseCodedMemory single units + DD/TT latent PCA (FigS3) | `FigureS3_PhaseCodedMemory.{pdf,svg}`, `panel_{C..H}_*.{pdf,svg}` |
| [make_multitask_behavior_figure.py](supplementary/make_multitask_behavior_figure.py) | MultiTask per-task example inputs/outputs (FigS4) | `FigureS4_MultiTask_behavior.{pdf,svg,png}` |
| [make_chaotic_delayed_memory_figure.py](supplementary/make_chaotic_delayed_memory_figure.py) | Chaotic Delayed Matching task summary (task structure, IC perturbations, DD-NODE fits) | `chaotic_delayed_matching_summary.{pdf,png}` (saved to `content/figures/`) |
| [make_nl_cycle_consistency_node_sweep.py](supplementary/make_nl_cycle_consistency_node_sweep.py) | Nonlinear cycle-consistency vs. noise on a NODE latent-size sweep — three panels (curves by latent size, NL vs linear scatter, R² vs latent dim) | `FigureS_NL_CycleCon_NODE_Sweep_<task>.{pdf,svg,png}` |
| [make_compiled_metrics_vs_latent_size.py](supplementary/make_compiled_metrics_vs_latent_size.py) | Six-panel TT-vs-NODE metric sweep (rate R², state R², Wasserstein, co-BPS, linear cycle-con, max Lyapunov) vs. NODE latent size — **one figure per dataset** (NBFF, MultiTask, RandomTarget, PCM, CDM) | `FigureS_<Task>_metrics_vs_latent_size.{pdf,svg,png}` |
| [make_figure_dsa.py](supplementary/make_figure_dsa.py) | DSA and fixed-point comparison for NODE/GRU 3BFF latent-size sweeps, added for reviewer-response DSA clarification | `FigureS_DSA_NODE_GRU.pdf` (copied to `manuscript/figs/`) |

Defaults for `make_nl_cycle_consistency_node_sweep.py` point at the 3BFF NODE
dim-sweep (`tt_3bff/20260520_NBFF_SAE_NODE_DimSweep`); use `--task` together
with `--tt-path` / `--node-sweep-path` to point it at a different sweep.
Metrics (`nl_cycle_con` + linear `cycle_con` per run) are cached next to the
script as `make_nl_cycle_consistency_node_sweep.cache.pkl`; pass `--force` to
recompute.

---

## Conventions

**Python scripts are the canonical source.** Every paper figure ships as a
`make_*.py` script that uses `# %%` cell delimiters so it can be executed
end-to-end (`python make_*.py`) or run cell-by-cell in an IDE that understands
`# %%` (VSCode, PyCharm, Spyder, etc.). The legacy `.ipynb` files are kept in
the figure folders for reference but are no longer the authoritative source —
edit the `.py` files. The umbrella runner
[make_all_paper_figures.py](make_all_paper_figures.py) only invokes the `.py`
versions.

When adding a new figure or panel:

1. **Write a `make_figure<N>_<short_name>.py` script** with `# %%` cells and a
   `draw_panel_X` / `set_panel_label` structure (see Figure 5 for the
   reference). Default `--output-dir` should be `outputs/` next to the script.
2. **Mark each panel with a `# %% [markdown]` header** (`## Panel X —
   description`) above the cell that produces it; use
   `outputs/panel<X>_<descriptor>.<ext>` for every `savefig` call.
3. **If a library call writes to CWD** (e.g. `Comparison.plot_*(savePDF=True)`),
   wrap the call in `_to_outputs()` from CanonicalDatasetPerf — copy that
   helper into the new script.
4. **Register the script** in [make_all_paper_figures.py](make_all_paper_figures.py)
   so it is part of the full-paper run.
5. **Update this index** with the panel → output mapping.
6. **Caches stay flat**, in the figure dir (`*.cache.pkl`). Only the actual
   figure components live in `outputs/`.

### Stale-file policy

PDF / PNG / MP4 outputs are gitignored, so cleanup is a local `rm` away. If a
panel is renamed or removed, the old file just stops being regenerated and can
be deleted. Each figure directory now has an `old/` folder for clearly
abandoned exploratory outputs that we don't want to delete outright.
