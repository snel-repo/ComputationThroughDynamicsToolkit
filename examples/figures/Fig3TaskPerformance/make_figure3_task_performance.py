"""Figure 3 — task performance panels (B, C, E, F, G, H, I).

Run as a script (``python make_figure3_task_performance.py``) or cell-by-cell
in an IDE that supports ``# %%`` cell delimiters.

Panel sources (see ``../FIGURE_GENERATION.md``):

* Panel B — 3BFF single-trial inputs / outputs (``outputs/panelB_3bff_io.pdf``)
* Panel C — 3BFF canonical TT latents + cube fixed points (``outputs/panelC_3bff_fps.pdf``)
* Panel E — MultiTask MemoryPro single-trial I/O (``outputs/panelE_multitask_io.pdf``)
* Panel F — MultiTask MemoryPro fixed-point rings, mem1 + response combined
  (``outputs/panelF_multitask_mem_resp_combined.pdf``)
* Panel G — RandomTarget perturbed-reach kinematics
  (``outputs/panelG_rt_kinematics.pdf``)
* Panel H — RandomTarget single-trial I/O (``outputs/panelH_rt_io.pdf``)
* Panel I — RandomTarget latents in Pec motor-potent plane
  (``outputs/panelI_rt_pec_projection.pdf``)

Panels A (3BFF task schematic) and D (MultiTask schematic) are Illustrator-only.
"""

# %% [markdown]
# ## Shared setup
#
# Load TT analyses for 3BFF, MultiTask, and RandomTarget; pre-compute the
# "val" inputs/outputs used by panels B, E, H.

# %%
import os
import sys as _sys

import matplotlib as _mpl

# Force the non-interactive Agg backend before importing pyplot so plt.show()
# is a no-op and matplotlib_inline's display() doesn't print `Figure(WxH)`
# for every figure when the script is run headlessly. An IDE running this
# cell-by-cell has already imported pyplot, so we skip the switch in that case.
if "matplotlib.pyplot" not in _sys.modules:
    _mpl.use("Agg")

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tasks.tt_MultiTask import Analysis_TT_MultiTask
from ctd.comparison.analysis.tt.tasks.tt_RandomTarget import Analysis_TT_RandomTarget
from ctd.comparison.analysis.tt.tt import Analysis_TT

dotenv.load_dotenv(dotenv.find_dotenv())

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOME_DIR = os.environ["HOME_DIR"]

pathNBFF = HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
an_NBFF = Analysis_TT(run_name="NBFF", filepath=pathNBFF)

pathMT = HOME_DIR + "content/trained_models/task-trained/tt_MultiTask/"
an_MT = Analysis_TT_MultiTask(run_name="MT", filepath=pathMT)

pathRT = HOME_DIR + "content/trained_models/task-trained/tt_RandomTarget/"
an_RT = Analysis_TT_RandomTarget(run_name="RT", filepath=pathRT)

# %%
inputs_nbff = an_NBFF.get_inputs(phase="val")
inputs_mt = an_MT.get_inputs(phase="val")
inputs_rt = an_RT.get_inputs(phase="val")

outputs_nbff = an_NBFF.get_model_outputs(phase="val")
controlled_nbff = outputs_nbff["controlled"].detach().cpu().numpy()

outputs_mt = an_MT.get_model_outputs(phase="val")
controlled_mt = outputs_mt["controlled"].detach().cpu().numpy()

outputs_rt = an_RT.get_model_outputs(phase="val")
controlled_rt = outputs_rt["controlled"].detach().cpu().numpy()


# %% [markdown]
# ## Panel B — 3BFF single-trial inputs / outputs
#
# Produces ``outputs/panelB_3bff_io.pdf``.

# %%
fig, axes = plt.subplots(3, 2, figsize=(8, 5), sharex=True)
colors = ["cyan", "red", "green"]
for i, color in enumerate(colors):
    ax_in = axes[i, 0]
    ax_in.plot(inputs_nbff[0, :, i], color=color)
    ax_out = axes[i, 1]
    ax_out.plot(controlled_nbff[0, :, i], color=color)
    for ax in (ax_in, ax_out):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/panelB_3bff_io.pdf")


# %% [markdown]
# ## Panel C — 3BFF canonical TT latents + cube fixed points
#
# Loads the canonical TT-3BFF model, finds fixed points with ``plot_fps``, and
# renders them on the unit cube alongside latent trajectories.

# %%
pathTT = HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
an_TT = Analysis_TT(run_name="TT", filepath=pathTT, use_train_dm=True)

pathDT = pathTT + "20250130_NBFF_LFADS_Viz/prefix=tt_3bff_max_epochs=500_seed=0/"
an_DT = Analysis_DD.create(run_name="DT", filepath=pathDT, model_type="LFADS")

tt_fps = an_TT.plot_fps(
    inputs=torch.zeros(3),
    learning_rate=1e-3,
    noise_scale=0.0,
    n_inits=2000,
    max_iters=10000,
    device="cpu",
)

q_thresh = 8e-6
q_flag = tt_fps.qstar < q_thresh
stable = tt_fps.is_stable[q_flag]
isStable = stable == 1
unStable = stable == 0
qstar = tt_fps.qstar[q_flag]
x_star = tt_fps.xstar[q_flag, :]

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_star)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    x_pca[isStable, 0],
    x_pca[isStable, 1],
    x_pca[isStable, 2],
    c=qstar[isStable],
    cmap="viridis",
    marker="o",
    label="Stable",
)
ax.scatter(
    x_pca[unStable, 0],
    x_pca[unStable, 1],
    x_pca[unStable, 2],
    c=qstar[unStable],
    cmap="viridis",
    marker="x",
    label="Unstable",
)
ax.view_init(elev=25, azim=-60)
fig.savefig(f"{OUTPUT_DIR}/panelC_3bff_fps.pdf")


# %% [markdown]
# ## Panel E — MultiTask MemoryPro single-trial inputs / outputs
#
# Produces ``outputs/panelE_multitask_io.pdf``.

# %%
trial_num = 1
n_mt_inputs = inputs_mt.shape[-1]
n_mt_outputs = controlled_mt.shape[-1]
task_flag, phase_dict = an_MT.get_task_flag("MemoryPro", phase="val")
inputs_memPro = inputs_mt[task_flag, :, :]
controlled_memPro = controlled_mt[task_flag, :, :]
response_ind = phase_dict[0]["response"][1]
input_labels = an_MT.datamodule.data_env.input_labels

phase_order = ["context", "stim1", "mem1", "response"]
phase_boundaries = [
    phase_dict[0][ph][1]
    for ph in phase_order
    if ph in phase_dict[0] and phase_dict[0][ph][1] < response_ind
]

# Rows: 1 fixation + 4 sensory + 1 task + 1 output = 7 rows
n_sensory = 4
n_rows = 1 + n_sensory + 1 + 1
fig, axes = plt.subplots(n_rows, 1, figsize=(8, 1.2 * n_rows), sharex=True)

# Row 0: Fixation (input 0)
axes[0].plot(inputs_memPro[0, :response_ind, 0], color="k")
axes[0].set_ylabel(input_labels[0], rotation=0, ha="right", va="center")

# Rows 1..4: each sensory input on its own row
for k in range(n_sensory):
    idx = 1 + k
    axes[1 + k].plot(inputs_memPro[0, :response_ind, idx], color="C0")
    axes[1 + k].set_ylabel(input_labels[idx], rotation=0, ha="right", va="center")

# Row 5: all task inputs on the same row
task_row = 1 + n_sensory
for i in range(5, n_mt_inputs):
    axes[task_row].plot(inputs_memPro[0, :response_ind, i], label=input_labels[i])
axes[task_row].set_ylabel("Task inputs", rotation=0, ha="right", va="center")

# Final row: outputs
out_row = task_row + 1
for i in range(n_mt_outputs):
    axes[out_row].plot(controlled_memPro[0, :response_ind, i])
axes[out_row].set_ylabel("Outputs", rotation=0, ha="right", va="center")

for ax in axes:
    for b in phase_boundaries:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/panelE_multitask_io.pdf")


# %% [markdown]
# ## Panel F — MultiTask MemoryPro fixed-point rings (mem1 + response)
#
# Produces ``outputs/panelF_multitask_mem_resp_combined.pdf``.

# %%
task_to_analyze = "MemoryPro"
task_for_pca = "MemoryPro"
phase_for_pca = "stim1"

# Get the model outputs
ics, inputs, targets = an_MT.get_model_inputs_noiseless()
out_dict = an_MT.get_model_outputs_noiseless()
lats = out_dict["latents"]
outputs = out_dict["controlled"]

# Latents used to fit phase-aligned PCA
pca_flag, pca_phase_task = an_MT.get_task_flag(task_for_pca)
lats4pca = lats[pca_flag].detach().numpy()
lats_phase_pca = an_MT.get_data_from_phase(pca_phase_task, phase_for_pca, lats4pca)
lats_phase_pca_flat = np.concatenate(lats_phase_pca)

pca = PCA(n_components=3)
pca.fit(lats_phase_pca_flat)

plot_flag, phase_task = an_MT.get_task_flag(task_to_analyze)
phase_names = ["context", "stim1", "mem1", "response"]
num_phases = len(phase_names)

lats4plot = lats[plot_flag].detach().numpy()
outputs4plot = outputs[plot_flag].detach().numpy()
B, T, D = lats4plot.shape
readout = an_MT.wrapper.model.readout

lats_pca = pca.transform(lats4plot.reshape(-1, D)).reshape(B, T, -1)

# %%
# For each MemoryPro phase, compute the fixed points and project to PCA space.
phase_list = [["context", "mem1"], "stim1", ["context", "mem1"], "response"]
fps = {}
xstar_pca = []
fps_out = []
q_star = []
for i, phase_for_fp in enumerate(phase_list):
    fps[phase_names[i]] = an_MT.compute_fps_phase(
        phases=phase_for_fp,
        task_to_analyze=task_to_analyze,
        noise_scale=0.0,
        lr=5e-3,
        max_iters=6000,
        use_noisy=False,
    )
    xstar = fps[phase_names[i]].xstar
    xstar_pca.append(pca.transform(xstar))
    fps_out.append(readout(torch.Tensor(xstar)).detach().numpy())
    q_star.append(fps[phase_names[i]].qstar)

xstar_pca = np.stack(xstar_pca, axis=0)
fps_out = np.stack(fps_out, axis=0)
fps_mat = np.concatenate((xstar_pca[:, :, :2], fps_out[:, :, 1:2]), axis=2)
q_star = np.stack(q_star, axis=0)
q_star[q_star == 0] = 1e-16

# %%
# Mask FPs that fail the per-phase q threshold.
fps_plot = fps_mat.copy()
threshs = [8e-6, 1e-5, 8e-6, 1e-5]
for i in range(num_phases):
    fps_plot[i, ~(q_star[i] < threshs[i]), :] = np.nan

plot_mat = np.concatenate((lats_pca[:, :, :2], outputs4plot[:, :, 1:2]), axis=2)

# %%
# Combined mem1 + response panel.
mem_idx = phase_names.index("mem1")
resp_idx = phase_names.index("response")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

for j, phase_dict_j in enumerate(phase_task):
    s, e = phase_dict_j["mem1"]
    ax.plot(
        plot_mat[j, s:e, 0],
        plot_mat[j, s:e, 1],
        plot_mat[j, s:e, 2],
        c="k",
        alpha=0.5,
        linewidth=0.8,
    )
    s, e = phase_dict_j["response"]
    ax.plot(
        plot_mat[j, s:e, 0],
        plot_mat[j, s:e, 1],
        plot_mat[j, s:e, 2],
        c="gray",
        alpha=0.5,
        linewidth=0.8,
    )

ax.scatter(
    fps_plot[mem_idx, :, 0],
    fps_plot[mem_idx, :, 1],
    fps_plot[mem_idx, :, 2],
    s=20,
    c="b",
    label="mem1 FPs",
)
ax.scatter(
    fps_plot[resp_idx, :, 0],
    fps_plot[resp_idx, :, 1],
    fps_plot[resp_idx, :, 2],
    s=20,
    c="r",
    label="response FPs",
)
ax.set_xlabel(f"PC1 ({phase_for_pca})")
ax.set_ylabel(f"PC2 ({phase_for_pca})")
ax.set_zlabel("Output")
ax.set_title("mem1 + response FPs (combined)")
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-1.5, 1.5])
ax.view_init(40, 40)
ax.legend()

plt.savefig(f"{OUTPUT_DIR}/panelF_multitask_mem_resp_combined.pdf")


# %% [markdown]
# ## Panel G — RandomTarget perturbed-reach kinematics
#
# Recreates the bump-perturbed evaluation from ``bumpMove.ipynb``: latents and
# hand trajectories are colored by bump magnitude. Produces
# ``outputs/panelG_rt_kinematics.pdf``.

# %%
from motornet.effector import RigidTendonArm26
from motornet.muscle import MujocoHillMuscle

from ctd.task_modeling.datamodule.task_datamodule import TaskDataModule
from ctd.task_modeling.task_env.random_target import RandomTarget

effector = RigidTendonArm26(muscle=MujocoHillMuscle())
an_TT_RT = Analysis_TT_RandomTarget(run_name="TT", filepath=pathRT)
task = RandomTarget(effector=effector, max_ep_duration=1.5)
task.dataset_name = "MoveBump"
dm = TaskDataModule(task, n_samples=1400, batch_size=1000)
dm.set_environment(task, for_sim=True)
dm.prepare_data()
dm.setup()
an_TT_RT.wrapper.set_environment(task)
an_TT_RT.datamodule = dm

window_start = 60
window_end = 85

tt_ics, tt_inputs, tt_targets = an_TT_RT.get_model_inputs()
inputs_to_env = an_TT_RT.get_inputs_to_env()
out_dict_rt = an_TT_RT.wrapper(tt_ics, tt_inputs, inputs_to_env=inputs_to_env)
controlled = out_dict_rt["controlled"]
states = out_dict_rt["states"]
latents = out_dict_rt["latents"].detach().numpy()

bump_lats = latents[:, window_start:window_end, :].reshape(-1, latents.shape[-1])
pca_rt = PCA(n_components=3)
bump_pca = pca_rt.fit_transform(bump_lats).reshape(-1, window_end - window_start, 3)

inputs_bump = inputs_to_env[:, 65, :]
bump_mag = inputs_bump[:, 0]
bump_mag_norm = (bump_mag - bump_mag.min()) / (bump_mag.max() - bump_mag.min())

# %%
# Hand-position trajectories colored by bump magnitude.
ctrl = controlled.detach().numpy()
fig_traj = plt.figure()
ax_traj = fig_traj.add_subplot(111)
for i in range(ctrl.shape[0]):
    ax_traj.plot(
        ctrl[i, 30:150, 0],
        ctrl[i, 30:150, 1],
        linewidth=0.5,
        alpha=0.5,
        color=cm.viridis(bump_mag_norm[i]),
    )
ax_traj.set_xlim(-0.5, 0)
ax_traj.set_ylim(0, 0.5)
ax_traj.set_xlabel("x pos")
ax_traj.set_ylabel("y pos")
fig_traj.savefig(f"{OUTPUT_DIR}/panelG_rt_kinematics.pdf")


# %% [markdown]
# ## Panel H — RandomTarget single-trial inputs / outputs
#
# Produces ``outputs/panelH_rt_io.pdf``.

# %%
actions_rt = outputs_rt["actions"].detach().cpu().numpy()
states_rt = outputs_rt["states"].detach().cpu().numpy()

n_rt_inputs = inputs_rt.shape[-1]
n_rt_outputs = controlled_rt.shape[-1]
n_rt_actions = actions_rt.shape[-1]
n_states_rt = states_rt.shape[-1]

# Context inputs: TargetX (blue), TargetY (orange), GoCue (black) — each on its own row
context_colors = ["tab:blue", "tab:orange", "k"]
context_labels = ["Target X", "Target Y", "Go"]

# Model-input groups for the 14-dim observation:
#   [0:2]   visual (fingertip x, y)
#   [2:8]   muscle lengths (6 muscles)
#   [8:14]  muscle velocities (6 muscles)
visual_slice = slice(0, 2)
mlen_slice = slice(2, 8)
mvel_slice = slice(8, 14)

# Layout: 3 context + 1 outputs + 1 actions + 3 model-input group rows = 8 rows
n_rows = 3 + 1 + 1 + 3
fig = plt.figure(figsize=(12, 1.4 * n_rows))

context_axes = []
for r in range(3):
    ax = fig.add_subplot(n_rows, 1, r + 1)
    ax.plot(inputs_rt[0, :, r], color=context_colors[r])
    ax.set_ylabel(context_labels[r], rotation=0, ha="right", va="center")
    context_axes.append(ax)
context_axes[0].set_title("Context Inputs")

# Controlled outputs (zero-baselined: subtract t=0 value per channel)
ax_out = fig.add_subplot(n_rows, 1, 4)
controlled_rt_zb = controlled_rt[0, :, :] - controlled_rt[0, 0:1, :]
for i in range(n_rt_outputs):
    ax_out.plot(controlled_rt_zb[:, i])
ax_out.set_title("Controlled Outputs (Δ from t=0)")

ax_act = fig.add_subplot(n_rows, 1, 5)
for i in range(n_rt_actions):
    ax_act.plot(actions_rt[0, :, i])
ax_act.set_title("Actions")

states_rt_zb = states_rt[0, :, :] - states_rt[0, 0:1, :]

ax_mlen = fig.add_subplot(n_rows, 1, 6)
for i in range(*mlen_slice.indices(n_states_rt)):
    ax_mlen.plot(states_rt_zb[:, i])
ax_mlen.set_ylabel("Muscle length", rotation=0, ha="right", va="center")
ax_mlen.set_title("Model Inputs (Δ from t=0)")

ax_mvel = fig.add_subplot(n_rows, 1, 7)
for i in range(*mvel_slice.indices(n_states_rt)):
    ax_mvel.plot(states_rt_zb[:, i])
ax_mvel.set_ylabel("Muscle velocity", rotation=0, ha="right", va="center")

ax_vis = fig.add_subplot(n_rows, 1, 8)
for i in range(*visual_slice.indices(n_states_rt)):
    ax_vis.plot(states_rt_zb[:, i])
ax_vis.set_ylabel("Visual", rotation=0, ha="right", va="center")

for ax in fig.axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/panelH_rt_io.pdf")


# %% [markdown]
# ## Panel I — RandomTarget latents in Pec motor-potent plane
#
# x-axis: TT latent dim most correlated with hand x-position.
# y-axis: projection of latents onto Pectoralis readout.
# Color: bump magnitude. Produces ``outputs/panelI_rt_pec_projection.pdf``.

# %%
readout_rt = an_TT_RT.wrapper.model.readout.weight.detach().numpy()
controlled_window = controlled[:, window_start:window_end, :]
x_pos = controlled_window[:, :, 0].detach().numpy()
pec_readout = readout_rt[0, :]

pec_proj = np.dot(bump_lats, pec_readout).reshape(-1, window_end - window_start)

fig_pec = plt.figure(figsize=(5, 5))
ax_pec = fig_pec.add_subplot(111)
for i in range(states.shape[0]):
    ax_pec.plot(
        x_pos[i, :],
        pec_proj[i, :],
        linewidth=0.5,
        color=cm.viridis(bump_mag_norm[i]),
    )
sorted_bump = bump_mag_norm[np.argsort(bump_mag_norm)]
for i in range(0, bump_mag_norm.shape[0], 100):
    ax_pec.scatter(
        -0.3 + 0.00003 * i,
        0.05,
        color=cm.viridis(sorted_bump[i]),
    )
ax_pec.text(-0.305, 0.06, "-20", fontsize=12)
ax_pec.text(-0.272, 0.06, "20", fontsize=12)
ax_pec.set_xlabel("x pos")
ax_pec.set_ylabel("pec proj")
fig_pec.savefig(f"{OUTPUT_DIR}/panelI_rt_pec_projection.pdf")
