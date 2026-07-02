"""Auto-converted from Figure4_Combined.ipynb.

Run as a script (`python make_figure4_canonical.py`) or cell-by-cell in an IDE
that supports the ``# %%`` cell delimiter.
"""

# %% [markdown]
# # Figure 4 — Combined panel notebook
#
# This notebook compiles the panel-producing code for **Figure 4** from:
#
# - `LearningProgress.ipynb` — TT training progression at epochs 10/50/100/250/500 (panel A right block).
# - `CanonicalDatasetPerf.ipynb` — DD-inferred latents for 3BFF / MultiTask / RandomTarget (panels B, C, D).
#
# Each `## Panel X` heading precedes the producing code block. See
# `../FIGURE_GENERATION.md` for the cross-figure index.
#
# > The left/middle thirds of Panel A (TT/Sim/DD pipeline schematic) are
# > Illustrator-only.

# %% [markdown]
# ## Panel A — TT training progression (10/50/100/250/500 epochs)
#
# Produces `TT_10_50_100_250_500_<trial>.pdf`.

# %%
import sys as _sys

import matplotlib as _mpl

# Force the non-interactive Agg backend before importing pyplot so plt.show()
# is a no-op and matplotlib_inline's display() doesn't print `Figure(WxH)`
# for every figure when the script is run headlessly. An IDE running this
# cell-by-cell has already imported pyplot, so we skip the switch in that case.
if "matplotlib.pyplot" not in _sys.modules:
    _mpl.use("Agg")

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

import os

# Import pca
import dotenv
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ctd.comparison.analysis.dd.dd import Analysis_DD
from ctd.comparison.analysis.tt.tt import Analysis_TT
from ctd.comparison.comparison import Comparison

dotenv.load_dotenv(dotenv.find_dotenv())
torch.manual_seed(42)
np.random.seed(42)

# %%


HOME_DIR = os.environ["HOME_DIR"]

pathTT_10 = (
    HOME_DIR
    + "content/trained_models/task-trained/tt_3bff/20241107_NBFF_NoisyGRU_TrainingProcess_10/max_epochs=10_batch_size=1000_seed=0/"
)
pathTT_50 = (
    HOME_DIR
    + "content/trained_models/task-trained/tt_3bff/20241107_NBFF_NoisyGRU_TrainingProcess_50/max_epochs=50_batch_size=1000_seed=0/"
)
pathTT_100 = (
    HOME_DIR
    + "content/trained_models/task-trained/tt_3bff/20241107_NBFF_NoisyGRU_TrainingProcess_100/max_epochs=100_batch_size=1000_seed=0/"
)
pathTT_250 = (
    HOME_DIR
    + "content/trained_models/task-trained/tt_3bff/20241107_NBFF_NoisyGRU_TrainingProcess_250/max_epochs=250_batch_size=1000_seed=0/"
)
pathTT_500 = (
    HOME_DIR
    + "content/trained_models/task-trained/tt_3bff/20241017_NBFF_NoisyGRU_NewFinal/"
)
# pathTT = HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
an_TT_10 = Analysis_TT(run_name="TT", filepath=pathTT_10)
an_TT_50 = Analysis_TT(run_name="TT", filepath=pathTT_50)
an_TT_100 = Analysis_TT(run_name="TT", filepath=pathTT_100)
an_TT_250 = Analysis_TT(run_name="TT", filepath=pathTT_250)
an_TT_500 = Analysis_TT(run_name="TT", filepath=pathTT_500)

# %%
in_10 = an_TT_10.get_inputs(phase="val")
out_10 = an_TT_10.get_model_outputs(phase="val")
in_50 = an_TT_50.get_inputs(phase="val")
out_50 = an_TT_50.get_model_outputs(phase="val")
in_100 = an_TT_100.get_inputs(phase="val")
out_100 = an_TT_100.get_model_outputs(phase="val")
in_250 = an_TT_250.get_inputs(phase="val")
out_250 = an_TT_250.get_model_outputs(phase="val")
in_500 = an_TT_500.get_inputs(phase="val")
out_500 = an_TT_500.get_model_outputs(phase="val")


# %%
trial_ind = 6
fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(421)
ax.plot(in_10[trial_ind, :100, :])
ax.set_xlim(0, 100)
ax2 = fig.add_subplot(522)
ax2.plot(out_10["controlled"][trial_ind, :100, :].detach().cpu().numpy())
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.2, 1.2)

ax3 = fig.add_subplot(523)
ax3.plot(in_50[trial_ind, :100, :])
ax3.set_xlim(0, 100)
ax4 = fig.add_subplot(524)
ax4.plot(out_50["controlled"][trial_ind, :100, :].detach().cpu().numpy())
ax4.set_xlim(0, 100)
ax4.set_ylim(-0.2, 1.2)

ax5 = fig.add_subplot(525)
ax5.plot(in_100[trial_ind, :100, :])
ax5.set_xlim(0, 100)
ax6 = fig.add_subplot(526)
ax6.plot(out_100["controlled"][trial_ind, :100, :].detach().cpu().numpy())
ax6.set_xlim(0, 100)
ax6.set_ylim(-0.2, 1.2)

ax7 = fig.add_subplot(527)
ax7.plot(in_250[trial_ind, :100, :])
ax7.set_xlim(0, 100)
ax8 = fig.add_subplot(528)
ax8.plot(out_250["controlled"][trial_ind, :100, :].detach().cpu().numpy())
ax8.set_xlim(0, 100)
ax8.set_ylim(-0.2, 1.2)

ax9 = fig.add_subplot(529)
ax9.plot(in_500[trial_ind, :100, :])
ax9.set_xlim(0, 100)
ax10 = fig.add_subplot(5, 2, 10)
ax10.plot(out_500["controlled"][trial_ind, :100, :].detach().cpu().numpy())
ax10.set_xlim(0, 100)
ax10.set_ylim(-0.2, 1.2)


# Save a pdf
plt.savefig(f"outputs/panelA_tt_learning_progression_trial{trial_ind}.pdf")

# %% [markdown]
# ## Shared setup — load TT and DD canonical comparisons (3BFF, MultiTask, RT)

# %%


HOME_DIR = os.environ["HOME_DIR"]
pathTT_3BFF = HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
pathTT_MT = HOME_DIR + "content/trained_models/task-trained/tt_MultiTask/"
pathTT_RT = HOME_DIR + "content/trained_models/task-trained/tt_RandomTarget/"

an_TT_3BFF = Analysis_TT(run_name="TT_3BFF", filepath=pathTT_3BFF)
an_TT_MT = Analysis_TT(run_name="TT_MT", filepath=pathTT_MT)
an_TT_RT = Analysis_TT(run_name="TT_RT", filepath=pathTT_RT)

path_GRU_Sweep_3BFF = pathTT_3BFF + "20250211_3BFF_GRU_Viz/"
subfolders_GRU_3BFF = [f.path for f in os.scandir(path_GRU_Sweep_3BFF) if f.is_dir()]

path_LFADS_Sweep_3BFF = pathTT_3BFF + "20250130_NBFF_LFADS_Viz/"
subfolders_LFADS_3BFF = [
    f.path for f in os.scandir(path_LFADS_Sweep_3BFF) if f.is_dir()
]

path_LDS_Sweep_3BFF = pathTT_3BFF + "20250130_NBFF_LDS_Viz/"
subfolders_LDS_3BFF = [f.path for f in os.scandir(path_LDS_Sweep_3BFF) if f.is_dir()]

path_GRU_Sweep_MT = pathTT_MT + "20250211_MT_GRU_RNN_Viz/"
subfolders_GRU_MT = [f.path for f in os.scandir(path_GRU_Sweep_MT) if f.is_dir()]

path_LFADS_Sweep_MT = pathTT_MT + "20250130_MultiTask_LFADS_Viz/"
subfolders_LFADS_MT = [f.path for f in os.scandir(path_LFADS_Sweep_MT) if f.is_dir()]

path_LDS_Sweep_MT = pathTT_MT + "20250131_MultiTask_LDS_Viz/"
subfolders_LDS_MT = [f.path for f in os.scandir(path_LDS_Sweep_MT) if f.is_dir()]

path_GRU_Sweep_RT = pathTT_RT + "20250211_RT_GRU_Viz/"
subfolders_GRU_RT = [f.path for f in os.scandir(path_GRU_Sweep_RT) if f.is_dir()]

path_LFADS_Sweep_RT = pathTT_RT + "20250130_RandomTarget_LFADS_Viz/"
subfolders_LFADS_RT = [f.path for f in os.scandir(path_LFADS_Sweep_RT) if f.is_dir()]

path_LDS_Sweep_RT = pathTT_RT + "20250814_RandomTarget_LDS_Sweep/"
subfolders_LDS_RT = [f.path for f in os.scandir(path_LDS_Sweep_RT) if f.is_dir()]


# %% [markdown]
# ## Panel B — 3BFF latents: TT vs. LFADS / GRU / LDS
#
# DD latents are aligned to TT latents with an optimal affine. Three representative trials shown with increasing opacity.

# %%
comparison_3BFF = Comparison(comparison_tag="3BFF")
comparison_3BFF.load_analysis(an_TT_3BFF, reference_analysis=True, group="TT")

for subfolder in subfolders_GRU_3BFF:
    subfolder = subfolder + "/"
    analysis_GRU = Analysis_DD.create(
        run_name="GRU", filepath=subfolder, model_type="SAE"
    )
    comparison_3BFF.load_analysis(analysis_GRU, group="GRU")

for subfolder in subfolders_LFADS_3BFF:
    subfolder = subfolder + "/"
    analysis_LFADS = Analysis_DD.create(
        run_name="LFADS", filepath=subfolder, model_type="LFADS"
    )
    comparison_3BFF.load_analysis(analysis_LFADS, group="LFADS")

for subfolder in subfolders_LDS_3BFF:
    subfolder = subfolder + "/"
    analysis_LDS = Analysis_DD.create(
        run_name="LDS", filepath=subfolder, model_type="SAE"
    )
    comparison_3BFF.load_analysis(analysis_LDS, group="LDS")

comparison_3BFF.regroup()
# comparison_3BFF.plot_trials(num_trials=2)

# %%
comparison_NBFF_single = Comparison(comparison_tag="NBFF_single")
comparison_NBFF_single.load_analysis(an_TT_3BFF, reference_analysis=True, group="TT")
comparison_NBFF_single.load_analysis(analysis_LFADS, group="LFADS")
comparison_NBFF_single.load_analysis(analysis_GRU, group="GRU")
comparison_NBFF_single.load_analysis(analysis_LDS, group="LDS")

# %%
os.makedirs("outputs", exist_ok=True)


def _clean_3d(ax):
    """Remove grey panes, gridlines, and axis lines from a 3D axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


# Inline replacement for comparison_NBFF_single.plot_trials_3d_reference so we
# can color individual trials. Plotted back-to-front: light grey behind, black in front.
analyses_3bff_panel = comparison_NBFF_single.analyses  # TT, LFADS, GRU, LDS
ref_lats_b = analyses_3bff_panel[0].get_latents().detach().numpy()
ref_lats_b_flat = ref_lats_b.reshape(-1, ref_lats_b.shape[-1])
pca_b = PCA()
ref_lats_b_pca_flat = pca_b.fit_transform(ref_lats_b_flat)

panelB_latents = []
axis_ranges_b = np.array([[np.inf, -np.inf]] * 3, dtype=float)
for an in analyses_3bff_panel:
    lats = an.get_latents().detach().numpy()
    lats_flat = lats.reshape(-1, lats.shape[-1])
    reg = LinearRegression().fit(lats_flat, ref_lats_b_pca_flat)
    lats_pca = reg.predict(lats_flat).reshape(
        ref_lats_b.shape[0], ref_lats_b.shape[1], -1
    )
    panelB_latents.append(lats_pca)
    for k in range(3):
        axis_ranges_b[k, 0] = min(axis_ranges_b[k, 0], np.min(lats_pca[:, :, k]))
        axis_ranges_b[k, 1] = max(axis_ranges_b[k, 1], np.max(lats_pca[:, :, k]))

# Trial 0 = black (front), trial 1 = grey, trial 2 = light grey (back).
trial_specs_3bff = [(2, "lightgray"), (1, "gray"), (0, "black")]
col_titles_b = ["TT", "LFADS", "GRU", "LDS"]
line_width_3bff = 2.5

fig = plt.figure(figsize=(15, 5))
for col, lats_pca in enumerate(panelB_latents):
    ax = fig.add_subplot(1, 4, col + 1, projection="3d")
    for z, (trial_idx, c) in enumerate(trial_specs_3bff):
        ax.plot(
            lats_pca[trial_idx, :, 0],
            lats_pca[trial_idx, :, 1],
            lats_pca[trial_idx, :, 2],
            color=c,
            linewidth=line_width_3bff,
            zorder=z,
        )
    ax.set_xlim(axis_ranges_b[0, :])
    ax.set_ylim(axis_ranges_b[1, :])
    ax.set_zlim(axis_ranges_b[2, :])
    ax.view_init(45, 45)
    ax.set_title(col_titles_b[col])
    _clean_3d(ax)

plt.savefig("outputs/panelB_3bff_3dLats.pdf")

# %% [markdown]
# ## Panel C — MultiTask MemoryPro latents (TT vs. DD)
#
# Projected into the 3D subspace defined by response 1, response 2, and fixation output dimensions. Colored by correct response angle. Produces `MT.pdf`.

# %%
comparison_MT = Comparison(comparison_tag="MultiTask")
comparison_MT.load_analysis(an_TT_MT, reference_analysis=True, group="TT")

for subfolder in subfolders_GRU_MT:
    subfolder = subfolder + "/"
    analysis_GRU = Analysis_DD.create(
        run_name="GRU", filepath=subfolder, model_type="SAE"
    )
    comparison_MT.load_analysis(analysis_GRU, group="GRU")

for subfolder in subfolders_LFADS_MT:
    subfolder = subfolder + "/"
    analysis_LFADS = Analysis_DD.create(
        run_name="LFADS", filepath=subfolder, model_type="LFADS"
    )
    comparison_MT.load_analysis(analysis_LFADS, group="LFADS")

for subfolder in subfolders_LDS_MT:
    subfolder = subfolder + "/"
    analysis_LDS = Analysis_DD.create(
        run_name="LDS", filepath=subfolder, model_type="SAE"
    )
    comparison_MT.load_analysis(analysis_LDS, group="LDS")

comparison_MT.regroup()

# %%
lats_true = an_TT_MT.get_latents(phase="val")
extra = an_TT_MT.get_extra_inputs(phase="val")
noiseless_inputs = an_TT_MT.get_model_inputs_noiseless(phase="val")
lats_lfads = analysis_LFADS.get_latents(phase="val")
lats_gru = analysis_GRU.get_latents(phase="val")
lats_lds = analysis_LDS.get_latents(phase="val")

# %%
input_list = an_TT_MT.env.input_labels
memPro_ind = 6
use_pca = False

memPro_trials = []
for i in range(noiseless_inputs[1].shape[0]):
    if noiseless_inputs[1][i, 0, memPro_ind] == 1:
        memPro_trials.append(i)

resp_start = extra[memPro_trials, 0].detach().numpy()
resp_end = extra[memPro_trials, 1].detach().numpy()
targets = an_TT_MT.get_model_inputs(phase="val")[2]
memProTargs = np.zeros((len(memPro_trials), 2))
for i in range(len(memPro_trials)):
    memProTargs[i, :] = targets[memPro_trials[i], int(resp_end[i]) - 1, 1:]

memProAngs = np.arctan2(memProTargs[:, 1], memProTargs[:, 0])
memProBins = np.digitize(memProAngs, np.linspace(-np.pi, np.pi, 9))

lats_true_memPro = lats_true[memPro_trials]
lats_lfads_memPro = lats_lfads[memPro_trials]
lats_gru_memPro = lats_gru[memPro_trials]
lats_lds_memPro = lats_lds[memPro_trials]

lats_true_memPro_s = []
lats_lfads_memPro_s = []
lats_gru_memPro_s = []
lats_lds_memPro_s = []
for i in range(lats_true_memPro.shape[0]):
    lats_true_memPro_s.append(
        lats_true_memPro[i][int(resp_start[i]) : int(resp_end[i])].detach().numpy()
    )
    lats_lfads_memPro_s.append(
        lats_lfads_memPro[i][int(resp_start[i]) : int(resp_end[i])].detach().numpy()
    )
    lats_gru_memPro_s.append(
        lats_gru_memPro[i][int(resp_start[i]) : int(resp_end[i])].detach().numpy()
    )
    lats_lds_memPro_s.append(
        lats_lds_memPro[i][int(resp_start[i]) : int(resp_end[i])].detach().numpy()
    )

lats_true_memPro_s1 = np.concatenate(lats_true_memPro_s, axis=0)
lats_lfads_memPro_s1 = np.concatenate(lats_lfads_memPro_s, axis=0)
lats_gru_memPro_s1 = np.concatenate(lats_gru_memPro_s, axis=0)
lats_lds_memPro_s1 = np.concatenate(lats_lds_memPro_s, axis=0)

readout = an_TT_MT.model.readout
wt = readout.weight.detach().numpy()
x_wt = wt[1, :]
y_wt = wt[2, :]
resp_wt = wt[0, :]

pca_memPro = PCA(n_components=3)
pca_memPro.fit(lats_true_memPro_s1)
lats_true_memPro_s1_pca = pca_memPro.transform(lats_true_memPro_s1)

x_proj = np.dot(lats_true_memPro_s1, x_wt)
y_proj = np.dot(lats_true_memPro_s1, y_wt)
resp_proj = np.dot(lats_true_memPro_s1, resp_wt)
lats_true_memPro_s1_wts = np.stack((x_proj, y_proj, resp_proj), axis=1)

lr_lfads = LinearRegression()
lr_gru = LinearRegression()
lr_lds = LinearRegression()

if use_pca:
    lr_lfads.fit(lats_lfads_memPro_s1, lats_true_memPro_s1_pca)
    lr_gru.fit(lats_gru_memPro_s1, lats_true_memPro_s1_pca)
    lr_lds.fit(lats_lds_memPro_s1, lats_true_memPro_s1_pca)
else:
    lr_lfads.fit(lats_lfads_memPro_s1, lats_true_memPro_s1_wts)
    lr_gru.fit(lats_gru_memPro_s1, lats_true_memPro_s1_wts)
    lr_lds.fit(lats_lds_memPro_s1, lats_true_memPro_s1_wts)

fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(151, projection="3d")
ax2 = fig.add_subplot(152, projection="3d")
ax3 = fig.add_subplot(153, projection="3d")
ax4 = fig.add_subplot(154, projection="3d")
ax5 = fig.add_subplot(155, projection="polar")
for i in range(50):
    if use_pca:
        true_pca = pca_memPro.transform(lats_true_memPro_s[i])
    else:
        xP = np.dot(lats_true_memPro_s[i], x_wt)
        yP = np.dot(lats_true_memPro_s[i], y_wt)
        respP = np.dot(lats_true_memPro_s[i], resp_wt)
        true_pca = np.stack((xP, yP, respP), axis=1)
    pred_lfads = lr_lfads.predict(lats_lfads_memPro_s[i])
    pred_gru = lr_gru.predict(lats_gru_memPro_s[i])
    pred_lds = lr_lds.predict(lats_lds_memPro_s[i])
    colorBin = memProBins[i]
    color = plt.cm.jet(colorBin / 8)
    ax1.plot(true_pca[:, 0], true_pca[:, 1], true_pca[:, 2], color=color, alpha=0.5)
    ax2.plot(
        pred_lfads[:, 0], pred_lfads[:, 1], pred_lfads[:, 2], color=color, alpha=0.5
    )
    ax3.plot(pred_gru[:, 0], pred_gru[:, 1], pred_gru[:, 2], color=color, alpha=0.5)
    ax4.plot(pred_lds[:, 0], pred_lds[:, 1], pred_lds[:, 2], color=color, alpha=0.5)

# Polar color legend
n_color_bins = 8
ang_edges = np.linspace(-np.pi, np.pi, n_color_bins + 1)
for b in range(n_color_bins):
    ax5.fill_between(
        [ang_edges[b], ang_edges[b + 1]],
        [0, 0],
        [1, 1],
        color=plt.cm.jet(b / n_color_bins),
        alpha=0.5,
    )
ax5.set_rticks([])
ax5.set_xticks([])

ax1.set_title("TT")
ax2.set_title("LFADS")
ax3.set_title("GRU")
ax4.set_title("LDS")

for ax in (ax1, ax2, ax3, ax4):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.savefig("outputs/panelC_multitask_latents.pdf")

# %% [markdown]
# ## Panel D — RandomTarget latents (TT vs. DD) colored by reach angle
#
# Produces `RT_radial.pdf`.

# %%
comparison_RT = Comparison(comparison_tag="RandomTarget")
comparison_RT.load_analysis(an_TT_RT, reference_analysis=True, group="TT")

for subfolder in subfolders_GRU_RT:
    subfolder = subfolder + "/"
    analysis_GRU = Analysis_DD.create(
        run_name="GRU", filepath=subfolder, model_type="SAE"
    )
    comparison_RT.load_analysis(analysis_GRU, group="GRU")

for subfolder in subfolders_LFADS_RT:
    subfolder = subfolder + "/"
    analysis_LFADS = Analysis_DD.create(
        run_name="LFADS", filepath=subfolder, model_type="LFADS"
    )
    comparison_RT.load_analysis(analysis_LFADS, group="LFADS")

for subfolder in subfolders_LDS_RT:
    subfolder = subfolder + "/"
    analysis_LDS = Analysis_DD.create(
        run_name="LDS", filepath=subfolder, model_type="SAE"
    )
    comparison_RT.load_analysis(analysis_LDS, group="LDS")


comparison_RT.regroup()

# %%
lats_true_full = an_TT_RT.get_latents(phase="val").detach().numpy()
lats_lfads_full = analysis_LFADS.get_latents(phase="val").detach().numpy()
lats_gru_full = analysis_GRU.get_latents(phase="val").detach().numpy()
lats_lds_full = analysis_LDS.get_latents(phase="val").detach().numpy()

endpoint_pos = an_TT_RT.get_model_outputs(phase="val")["controlled"].detach().numpy()
inputs_true = an_TT_RT.get_true_inputs(phase="val")
extra_RT = an_TT_RT.get_extra_inputs(phase="val").detach().cpu().numpy().astype(int)
go_cues = extra_RT[:, 1]  # per-trial go cue (catch trials marked with -1)

# Window each trial to the `window_len` samples immediately after its go cue.
window_len = 30
T_total = lats_true_full.shape[1]
valid_trials = np.where((go_cues >= 0) & (go_cues + window_len <= T_total))[0]


def _window(lats):
    return np.stack(
        [lats[i, go_cues[i] - 10 : go_cues[i] + window_len] for i in valid_trials]
    )


lats_true = _window(lats_true_full)
lats_lfads = _window(lats_lfads_full)
lats_gru = _window(lats_gru_full)
lats_lds = _window(lats_lds_full)

# Reach angle: target position is in input dims 0/1 once the target turns on.
# Use a timestep guaranteed to be after target onset (target_on <= 30 in val).
start_pos = endpoint_pos[valid_trials, 0, :]
targ_pos = inputs_true[valid_trials, 40, :2]
reach_ang = np.arctan2(
    targ_pos[:, 1] - start_pos[:, 1], targ_pos[:, 0] - start_pos[:, 0]
)

n_bins = 8
bins = np.linspace(-np.pi, np.pi, n_bins + 1)
reach_ang_bins = np.digitize(reach_ang, bins)

# PCA on TT latents and align DD latents into that PCA space.
lats_true_flat = lats_true.reshape(-1, lats_true.shape[-1])
lats_pca = PCA(n_components=3)
lats_pca.fit(lats_true_flat)
lats_true_pca_flat = lats_pca.transform(lats_true_flat)
lats_true_pca = lats_true_pca_flat.reshape(lats_true.shape[0], lats_true.shape[1], -1)


def _align(lats):
    flat = lats.reshape(-1, lats.shape[-1])
    reg = LinearRegression().fit(flat, lats_true_pca_flat)
    return reg.predict(flat).reshape(lats.shape[0], lats.shape[1], -1)


lats_lfads_to_true = _align(lats_lfads)
lats_gru_to_true = _align(lats_gru)
lats_lds_to_true = _align(lats_lds)

# %%
# Group trial indices by reach-angle bin.
trial_list = [[] for _ in range(n_bins)]
for i in range(len(reach_ang_bins)):
    trial_list[reach_ang_bins[i] - 1].append(i)

# %%
# Plot latents colored by reach angle bin
fig = plt.figure(figsize=(25, 10))
ax1 = fig.add_subplot(1, 5, 1, projection="3d")
ax2 = fig.add_subplot(1, 5, 2, projection="3d")
ax3 = fig.add_subplot(1, 5, 3, projection="3d")
ax4 = fig.add_subplot(1, 5, 4, projection="3d")
ax5 = fig.add_subplot(1, 5, 5, projection="polar")

n_trials_per_bin = 15
for i in range(n_bins):
    color = plt.cm.jet(i / n_bins)
    for j in range(min(n_trials_per_bin, len(trial_list[i]))):
        trial = trial_list[i][j]
        ax1.plot(
            lats_true_pca[trial][:, 0],
            lats_true_pca[trial][:, 1],
            lats_true_pca[trial][:, 2],
            color=color,
            alpha=0.5,
        )
        ax2.plot(
            lats_lfads_to_true[trial][:, 0],
            lats_lfads_to_true[trial][:, 1],
            lats_lfads_to_true[trial][:, 2],
            color=color,
            alpha=0.5,
        )
        ax3.plot(
            lats_gru_to_true[trial][:, 0],
            lats_gru_to_true[trial][:, 1],
            lats_gru_to_true[trial][:, 2],
            color=color,
            alpha=0.5,
        )
        ax4.plot(
            lats_lds_to_true[trial][:, 0],
            lats_lds_to_true[trial][:, 1],
            lats_lds_to_true[trial][:, 2],
            color=color,
            alpha=0.5,
        )

ax1.set_title("TT")
ax2.set_title("LFADS")
ax3.set_title("GRU")
ax4.set_title("LDS")

for ax in (ax1, ax2, ax3, ax4):
    ax.view_init(20, 60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

for bin_num in range(n_bins):
    ax5.fill_between(
        [bins[bin_num], bins[bin_num + 1]],
        [0, 0],
        [1, 1],
        color=plt.cm.jet(bin_num / n_bins),
        alpha=0.5,
    )
ax5.set_rticks([])
ax5.set_xticks([])
plt.savefig("outputs/panelD_rt_radial.pdf")

# %% [markdown]
# ## Combined — Panels B, C, D in a single figure
#
# 3 rows (3BFF, MultiTask MemoryPro, RandomTarget) × 4 columns (TT, LFADS, GRU, LDS).
# Reuses the color coding from the individual panels:
# - Row 1 (3BFF): default matplotlib trial colors (3 trials).
# - Row 2 (MemoryPro): `jet` colormap binned by correct response angle.
# - Row 3 (RandomTarget): `jet` colormap binned by reach angle.
#
# Produces `outputs/figure4_BCD_combined.pdf`.

# %%
os.makedirs("outputs", exist_ok=True)


def _clean_3d(ax):
    """Remove grey panes, gridlines, and axis lines from a 3D axes."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.line.set_color((1, 1, 1, 0))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


col_titles = ["TT", "LFADS", "GRU", "LDS"]
row_titles = ["3BFF", "MultiTask (MemoryPro)", "RandomTarget"]
# Drawn back-to-front so black sits on top.
trial_specs_3bff = [(2, "lightgray"), (1, "gray"), (0, "black")]
line_width_3bff = 2.5
# <1 zooms 3BFF in so the trajectories fill more of the subplot.
zoom_3bff = 0.65

fig = plt.figure(figsize=(16, 12))

# ---------- Row 1: 3BFF (panel B) ----------
analyses_3bff = comparison_NBFF_single.analyses  # TT, LFADS, GRU, LDS (in this order)
ref_lats_3bff = analyses_3bff[0].get_latents().detach().numpy()
pca_3bff = PCA()
ref_lats_3bff_flat = ref_lats_3bff.reshape(-1, ref_lats_3bff.shape[-1])
ref_lats_3bff_pca_flat = pca_3bff.fit_transform(ref_lats_3bff_flat)

row1_latents = []
axis_ranges_r1 = np.array([[np.inf, -np.inf]] * 3, dtype=float)
for an in analyses_3bff:
    lats = an.get_latents().detach().numpy()
    lats_flat = lats.reshape(-1, lats.shape[-1])
    reg = LinearRegression().fit(lats_flat, ref_lats_3bff_pca_flat)
    lats_pca = reg.predict(lats_flat).reshape(
        ref_lats_3bff.shape[0], ref_lats_3bff.shape[1], -1
    )
    row1_latents.append(lats_pca)
    for k in range(3):
        axis_ranges_r1[k, 0] = min(axis_ranges_r1[k, 0], np.min(lats_pca[:, :, k]))
        axis_ranges_r1[k, 1] = max(axis_ranges_r1[k, 1], np.max(lats_pca[:, :, k]))

# Zoom the shared 3BFF limits in toward the center.
centers_r1 = axis_ranges_r1.mean(axis=1, keepdims=True)
half_r1 = (axis_ranges_r1[:, 1:2] - axis_ranges_r1[:, 0:1]) / 2
axis_ranges_r1 = np.hstack(
    [centers_r1 - half_r1 * zoom_3bff, centers_r1 + half_r1 * zoom_3bff]
)

for col, lats_pca in enumerate(row1_latents):
    ax = fig.add_subplot(3, 4, col + 1, projection="3d")
    for z, (trial_idx, c) in enumerate(trial_specs_3bff):
        ax.plot(
            lats_pca[trial_idx, :, 0],
            lats_pca[trial_idx, :, 1],
            lats_pca[trial_idx, :, 2],
            color=c,
            linewidth=line_width_3bff,
            zorder=z,
        )
    ax.set_xlim(axis_ranges_r1[0, :])
    ax.set_ylim(axis_ranges_r1[1, :])
    ax.set_zlim(axis_ranges_r1[2, :])
    ax.view_init(45, 45)
    ax.set_title(col_titles[col])
    _clean_3d(ax)

# ---------- Row 2: MultiTask MemoryPro (panel C) ----------
# Refit memPro regressions locally — panel D overwrites the lr_* names
# with regressors trained on RandomTarget latents (different feature dim).
lats_lfads_memPro_s1 = np.concatenate(lats_lfads_memPro_s, axis=0)
lats_gru_memPro_s1 = np.concatenate(lats_gru_memPro_s, axis=0)
lats_lds_memPro_s1 = np.concatenate(lats_lds_memPro_s, axis=0)

lr_lfads_mp = LinearRegression().fit(lats_lfads_memPro_s1, lats_true_memPro_s1_wts)
lr_gru_mp = LinearRegression().fit(lats_gru_memPro_s1, lats_true_memPro_s1_wts)
lr_lds_mp = LinearRegression().fit(lats_lds_memPro_s1, lats_true_memPro_s1_wts)

n_memPro_trials = min(50, len(lats_true_memPro_s))
tt_proj_list, lfads_proj_list, gru_proj_list, lds_proj_list = [], [], [], []
for i in range(n_memPro_trials):
    xP = np.dot(lats_true_memPro_s[i], x_wt)
    yP = np.dot(lats_true_memPro_s[i], y_wt)
    respP = np.dot(lats_true_memPro_s[i], resp_wt)
    true_proj = np.stack((xP, yP, respP), axis=1)
    tt_proj_list.append(true_proj)
    lfads_proj_list.append(lr_lfads_mp.predict(lats_lfads_memPro_s[i]))
    gru_proj_list.append(lr_gru_mp.predict(lats_gru_memPro_s[i]))
    lds_proj_list.append(lr_lds_mp.predict(lats_lds_memPro_s[i]))

row2_data = [tt_proj_list, lfads_proj_list, gru_proj_list, lds_proj_list]
for col, projs in enumerate(row2_data):
    ax = fig.add_subplot(3, 4, 4 + col + 1, projection="3d")
    for i in range(n_memPro_trials):
        color = plt.cm.jet(memProBins[i] / 8)
        ax.plot(projs[i][:, 0], projs[i][:, 1], projs[i][:, 2], color=color, alpha=0.5)
    _clean_3d(ax)

# ---------- Row 3: RandomTarget (panel D) ----------
row3_data = [lats_true_pca, lats_lfads_to_true, lats_gru_to_true, lats_lds_to_true]
n_trials_rt = 15
for col, lats in enumerate(row3_data):
    ax = fig.add_subplot(3, 4, 8 + col + 1, projection="3d")
    for i in range(n_bins):
        color = plt.cm.jet(i / n_bins)
        for j in range(min(n_trials_rt, len(trial_list[i]))):
            trial = trial_list[i][j]
            ax.plot(
                lats[trial][:, 0],
                lats[trial][:, 1],
                lats[trial][:, 2],
                color=color,
                alpha=0.5,
            )
    ax.view_init(-32, 128)
    _clean_3d(ax)

# Row labels on the left margin
for row, label in enumerate(row_titles):
    y = 1 - (row + 0.5) / 3
    fig.text(0.02, y, label, ha="center", va="center", rotation=90, fontsize=12)

plt.tight_layout(rect=[0.04, 0, 1, 1])
plt.savefig("outputs/figure4_BCD_combined.pdf")
plt.savefig("outputs/figure4_BCD_combined.png", dpi=200)
plt.show()

# %% [markdown]
# ## Interactive — pick a viewing angle for TT RandomTarget
#
# Drag the sliders for elevation / azimuth (and optional roll) until the trajectories look right, then copy the printed `view_init(...)` call into the Panel D and combined cells (replacing the current `view_init(20, 60)`).
