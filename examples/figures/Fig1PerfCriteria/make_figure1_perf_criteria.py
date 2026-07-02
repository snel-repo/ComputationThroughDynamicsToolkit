"""Auto-converted from Fig1OneBitFlipFlopGen.ipynb.

Run as a script (`python make_figure1_perf_criteria.py`) or cell-by-cell in an IDE
that supports the ``# %%`` cell delimiter.
"""

# %% [markdown]
# # Figure 1 — CtD framework and failure modes
#
# This notebook produces the panels for Figure 1 of the CtDToolkit paper.
#
# | Panel | Content | Cell |
# | --- | --- | --- |
# | **B** (Computation / Algorithm / Implementation) | 1-BFF inputs, flow field, 3D embedding, firing rates | Cell 0 |
# | **F** (Underfitting) | Mis-fit input + state-space | Cell 1 |
# | **G** (Invented Features) | Extra-dim 3D embedding + extra-feature traces | Cell 2 |
# | **H** (Dynamical Misattribution) | Misattributed trajectory + inferred-input traces | Cell 3 |
# | (Anim) | Time-evolving flow field video | Cell 4 |
#
# Inline `# === PANEL X ===` comment blocks immediately precede each
# `fig.savefig(...)` so you can `grep` for the panel letter to jump straight to
# the producing block.
#
# See `../FIGURE_GENERATION.md` for the cross-figure index.

# %% [markdown]
# ## Cell 0 — Panel B (Computation / Algorithm / Implementation)
#
# Produces:
# - `panelB_computation_inputs.pdf` — Panel B top row (green, Computation: inputs u, output p)
# - `panelB_algorithm_flow_field.pdf` — Panel B middle row (orange, Algorithm)
# - `panelB_implementation_3d_embedding.pdf`, `panelB_implementation_rates.pdf` — Panel B bottom row (blue, Implementation)

# %%
# Force the non-interactive Agg backend before importing pyplot so plt.show()
# is a no-op and matplotlib_inline's display() doesn't print `Figure(WxH)` for
# every figure when this script is run headlessly. An IDE running cell-by-cell
# has already imported pyplot, so we skip the switch in that case.
import sys as _sys

import matplotlib as _mpl

if "matplotlib.pyplot" not in _sys.modules:
    _mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint
from scipy.optimize import fsolve

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]


cmap = "inferno"
# Get the original 'autumn' colormap
autumn = plt.colormaps["autumn"]

# Create a new colormap by trimming the range of the autumn colormap
# Adjusting the start and stop values trims off the yellow part
colors = autumn(np.linspace(0, 0.75, 256))  # Use values up to 0.85 to reduce yellow

# Create a new colormap with the adjusted range
custom_autumn = LinearSegmentedColormap.from_list("custom_autumn", colors)
cmap = custom_autumn


# Modified system equation with sensitivity parameter s
def system(x, t, I_func, s):
    I = I_func(t)
    dU_dx = 2 * x - 6 * x**2 + 4 * x**3
    dxdt = -s * dU_dx + I
    return dxdt


# Input function with reduced amplitudes
def I_func(t):
    # Parameters for the positive Gaussian pulse
    A_pos = 0.475  # Reduced amplitude of positive pulse
    t_pos = 10.0  # Center time of positive pulse
    sigma_pos = 0.7  # Width of positive pulse

    # Parameters for the negative Gaussian pulse
    A_neg = -0.475  # Reduced amplitude of negative pulse
    t_neg = 60.0  # Center time of negative pulse
    sigma_neg = 0.7  # Width of negative pulse

    # Parameters for the ineffective Gaussian pulse
    A_pos_null = 0.48  # Reduced amplitude of ineffective positive pulse
    t_pos_null = 30.0  # Center time of ineffective positive pulse
    sigma_pos_null = 0.9  # Width of ineffective positive pulse

    # Parameters for the ineffective negative Gaussian pulse
    A_neg_null = -0.48  # Reduced amplitude of ineffective negative pulse
    t_neg_null = 80.0  # Center time of ineffective negative pulse
    sigma_neg_null = 0.5  # Width of ineffective negative pulse

    # Positive Gaussian pulse
    I_pos = A_pos * np.exp(-((t - t_pos) ** 2) / (2 * sigma_pos**2))

    # Negative Gaussian pulse
    I_neg = A_neg * np.exp(-((t - t_neg) ** 2) / (2 * sigma_neg**2))

    # Ineffective positive Gaussian pulse (commented out in total input)
    I_pos_null = A_pos_null * np.exp(-((t - t_pos_null) ** 2) / (2 * sigma_pos_null**2))

    # Ineffective negative Gaussian pulse (commented out in total input)
    I_neg_null = A_neg_null * np.exp(-((t - t_neg_null) ** 2) / (2 * sigma_neg_null**2))

    # Total input (currently only effective pulses)
    I_t = I_pos + I_neg  # + I_pos_null + I_neg_null
    return I_t


# Time vector
t = np.linspace(0, 100, 50000)
solver_options = {"rtol": 1e-10, "atol": 1e-12}

# Initial condition
x0 = 0.0

# Sensitivity parameter
s = 1.0  # Smaller s makes the system more sensitive

# Solving the ODE with the modified system equation
x = odeint(system, x0, t, args=(I_func, s), **solver_options).flatten()

# Input values over time
I_over_time = np.array([I_func(ti) for ti in t])

# Time evolution plot
fig0 = plt.figure(figsize=(12, 6))
ax0 = fig0.add_subplot(111)
ax0.plot(t, x, label="State x(t)")
ax0.scatter(t, I_over_time, c=I_over_time, label="Input I(t)", cmap=cmap, linestyle="-")
ax0.set_xlabel("Time")
ax0.set_ylabel("State x and Input I")
ax0.set_title("Time Evolution of State with Inputs (s = 0.3)")

# === PANEL B (top row, green): inputs u and output p — Computation ===
# Saves: outputs/panelB_computation_inputs.pdf
fig0.savefig("outputs/panelB_computation_inputs.pdf")
plt.show()

# Range of x and I values
stepX = 0.1  # Smaller step for smoother curves
x_vals = np.arange(-0.3, 1.3 + stepX, stepX)
I_vals = np.arange(-0.5, 0.5 + stepX, stepX)

# Round values to nearest 0.001
# x_vals = np.round(x_vals, 3)
# I_vals = np.round(I_vals, 3)

X, I_grid = np.meshgrid(x_vals, I_vals)

# Compute the derivative of U(x)
dU_dx = 2 * X - 6 * X**2 + 4 * X**3

# Compute dx/dt for the grid with sensitivity parameter
dXdt = -s * dU_dx + I_grid

# Since I is not dynamic, its derivative is zero
dIdt = np.zeros_like(dXdt)

# Magnitude used to color the quiver arrows
M = np.abs(dXdt)
dXdt_norm = dXdt

# # Round down the values to zero if they are very small
# dXdt_norm[np.abs(dXdt_norm) < 1e-3] = 0


# Function to compute fixed points for a given I
def fixed_points(I, s):
    # Function to find roots
    def func(x):
        return s * (2 * x - 6 * x**2 + 4 * x**3) - I

    guesses = [-0.5, 0.0, 0.5, 1.0, 1.5]
    roots = []
    for guess in guesses:
        root, info, ier, mesg = fsolve(func, guess, full_output=True)
        if ier == 1 and 0.0 <= root[0] <= 1.0:
            roots.append(root[0])
    return np.unique(roots)


# Compute fixed points over a range of I values
I_values = np.linspace(-1.0, 1.0, 75)
stable_fp_x = []
stable_fp_I = []
unstable_fp_x = []
unstable_fp_I = []

for I_val in I_values:
    x_fps = fixed_points(I_val, s)
    for x_fp in x_fps:
        # Compute second derivative of U(x) at x_fp
        d2U_dx2 = 2 - 12 * x_fp + 12 * x_fp**2
        # Compute the eigenvalue (lambda)
        lambda_ = -s * d2U_dx2
        if lambda_ > 0:
            # Unstable fixed point
            unstable_fp_x.append(x_fp)
            unstable_fp_I.append(I_val)
        else:
            # Stable fixed point
            stable_fp_x.append(x_fp)
            stable_fp_I.append(I_val)

# Plotting the corrected flow field
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.quiver(
    I_grid,
    X,
    np.zeros_like(dXdt_norm),
    dXdt_norm,
    M,
    pivot="mid",
    cmap="jet",
    alpha=0.8,
)
# ax.set_xlabel('State x')
# ax.set_ylabel('Input I')
# ax.set_title(f'Dynamical Flow Field (s = {s:.2f})')

# Plotting the system trajectory
from scipy.interpolate import interp1d

# Create interpolation functions
x_interp = interp1d(t, x, kind="linear")
I_interp = interp1d(t, I_over_time, kind="linear")

# Time points for interpolation
t_fine = np.linspace(0, 100, 1000)
x_fine = x_interp(t_fine)
I_fine = I_interp(t_fine)

# Plotting the trajectory
ax.plot(I_fine, x_fine, "k-", linewidth=2, label="System Trajectory")

# Highlighting the bit turning on and off
on_pulse_idx = (t_fine >= 5) & (t_fine <= 25)
off_pulse_idx = (t_fine >= 55) & (t_fine <= 75)
# ax.plot(I_fine[on_pulse_idx], x_fine[on_pulse_idx], 'g-', linewidth=2, label='Bit Turning On')
# ax.plot(I_fine[off_pulse_idx], x_fine[off_pulse_idx], 'r-', linewidth=2, label='Bit Turning Off')

# Plotting the fixed points with colors according to I
# For stable fixed points
scatter_stable = ax.scatter(
    stable_fp_I,
    stable_fp_x,
    c=stable_fp_I,
    cmap=cmap,
    s=40,
    label="Stable Fixed Points",
)

# For unstable fixed points
scatter_unstable = ax.scatter(
    unstable_fp_I,
    unstable_fp_x,
    c=unstable_fp_I,
    marker="x",
    cmap=cmap,
    s=40,
    label="Unstable Fixed Points",
)

# Adjusting plot aesthetics
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.3, 1.3)
# ax.legend(loc='upper right')
ax.grid(False)

# Remove spines and ticks
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# === PANEL B (middle row, orange): state-space diagram with flow field — Algorithm ===
# Saves: outputs/panelB_algorithm_flow_field.pdf
fig.savefig("outputs/panelB_algorithm_flow_field.pdf")
plt.show()


# Plotting the 3D embedding of the system
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


# Prepare the state projections
# We'll define a simple projection function for the state
def project_state(state):
    # Simple linear embedding: map x into y and z linearly
    n1 = state / 2
    n2 = state
    return n1, n2


# Create interpolation functions
from scipy.interpolate import interp1d

x_interp = interp1d(t, x, kind="linear")
I_interp = interp1d(t, I_over_time, kind="linear")

# Time points for interpolation
t_fine = np.linspace(0, 100, 1000)
x_fine = x_interp(t_fine)
I_fine = I_interp(t_fine)

# Project the state variable
n1_fine, n2_fine = project_state(x_fine)

# Plot the system trajectory in 3D
ax.plot(I_fine, n1_fine, n2_fine, "k-", linewidth=2, label="System Trajectory")

# Plotting the fixed points in 3D
stable_n1, stable_n2 = project_state(np.array(stable_fp_x))
unstable_n1, unstable_n2 = project_state(np.array(unstable_fp_x))

# Stable fixed points
sc_stable = ax.scatter(
    stable_fp_I,
    stable_n1,
    stable_n2,
    c=stable_fp_I,
    cmap=cmap,
    s=40,
    label="Stable Fixed Points",
)

# Unstable fixed points
sc_unstable = ax.scatter(
    unstable_fp_I,
    unstable_n1,
    unstable_n2,
    c=unstable_fp_I,
    cmap=cmap,
    s=40,
    marker="x",
    label="Unstable Fixed Points",
)

# Customize the axes labels
ax.set_xlabel("Input I")
ax.set_ylabel("Neuron 1")
ax.set_zlabel("Neuron 2")
ax.set_title("3D Embedding of the System (s = {:.2f})".format(s))

# Add a colorbar
cbar = fig.colorbar(sc_stable, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Input I")

# Adjust view angle for better visualization
ax.view_init(elev=20, azim=-40)

# Remove gridlines and axes panes for a cleaner look
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)

# set axes equal to avoid distortion
# ax.set_xlim()
# Add a grey plane where the projections live (e.g., z = y plane)
min_n1 = np.min(n1_fine)
max_n1 = np.max(n1_fine)
min_n2 = np.min(n2_fine)
max_n2 = np.max(n2_fine)

x_plane = np.linspace(-0.5, 0.5, 100)
y_plane = np.linspace(min_n1, max_n1, 100)


X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = Y_plane * 2  # Since in linear embedding, z = y

# Plot the plane
ax.plot_surface(X_plane, Y_plane, Z_plane, color="grey", alpha=0.3)

ax.legend()
# === PANEL B (bottom row, blue — left): 3D linear embedding of 1D latent — Implementation ===
# Saves: outputs/panelB_implementation_3d_embedding.pdf
fig.savefig("outputs/panelB_implementation_3d_embedding.pdf")
plt.show()

fig = plt.figure(figsize=(8, 8))
# Plot the neuron firing rates over time

ax = fig.add_subplot(111)
ax.plot(n1_fine, label="Neuron 1")
ax.plot(n2_fine, label="Neuron 2")
# ax.plot(I_fine)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
# === PANEL B (bottom row, blue — right): simulated 2-neuron firing rates ===
# Saves: outputs/panelB_implementation_rates.pdf
fig.savefig("outputs/panelB_implementation_rates.pdf")
plt.show()

# %% [markdown]
# ## Cell 1 — Panel F (Underfitting)
#
# Produces:
# - `inputs_modified.pdf` — Panel F top row (true u vs. underfit ẑ)
# - `dynamical_flow_field_modified.pdf` — Panel F bottom row (ideal vs. underfit flow field)

import matplotlib.pyplot as plt

# %%
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.integrate import odeint
from scipy.optimize import fsolve

# Custom colormap setup
autumn = plt.colormaps["autumn"]
colors = autumn(np.linspace(0, 0.75, 256))
custom_autumn = LinearSegmentedColormap.from_list("custom_autumn", colors)
cmap = custom_autumn


# Modified system equation with sensitivity parameter s
def system(x, t, I_func, s):
    I = I_func(t)
    # Reduce the effect of negative inputs
    if I < 0:
        I = 0.3 * I
    # Consistent derivative of U(x)
    dU_dx = 2 * x - 6 * x**2 + 4 * x**3
    dxdt = -s * dU_dx + I
    return dxdt


# Input function with original amplitudes
def I_func(t):
    # Parameters for the positive Gaussian pulse
    A_pos = 0.475  # Amplitude of positive pulse
    t_pos = 10.0  # Center time of positive pulse
    sigma_pos = 0.7  # Width of positive pulse

    # Parameters for the negative Gaussian pulse
    A_neg = -0.475  # Amplitude of negative pulse
    t_neg = 60.0  # Center time of negative pulse
    sigma_neg = 0.7  # Width of negative pulse

    # Positive Gaussian pulse
    I_pos = A_pos * np.exp(-((t - t_pos) ** 2) / (2 * sigma_pos**2))

    # Negative Gaussian pulse
    I_neg = A_neg * np.exp(-((t - t_neg) ** 2) / (2 * sigma_neg**2))

    # Total input
    I_t = I_pos + I_neg
    return I_t


# Time vector
t = np.linspace(0, 100, 5000)
solver_options = {"rtol": 1e-10, "atol": 1e-12}

# Initial condition
x0 = 0.0

# Sensitivity parameter
s = 1.0  # Sensitivity of the system

# Solving the ODE with the modified system equation
x = odeint(system, x0, t, args=(I_func, s), **solver_options).flatten()

# Input values over time
I_over_time = np.array([I_func(ti) for ti in t])

# Time evolution plot
fig0 = plt.figure(figsize=(12, 6))
ax0 = fig0.add_subplot(111)
ax0.plot(t, x, label="State x(t)")
ax0.scatter(t, I_over_time, c=I_over_time, cmap=cmap, label="Input I(t)")
ax0.set_xlabel("Time")
ax0.set_ylabel("State x and Input I")
ax0.set_title("Time Evolution of State with Inputs (s = 1.0)")
ax0.legend()
# === PANEL F (top): true u vs. underfit ẑ — Underfitting ===
# Saves: outputs/panelF_underfitting_inputs.pdf
fig0.savefig("outputs/panelF_underfitting_inputs.pdf")
plt.show()

# Range of x and I values
stepX = 0.05  # Smaller step for smoother curves
x_vals = np.arange(-0.3, 1.3 + stepX, stepX)
I_vals = np.arange(-0.5, 0.5 + stepX, stepX)
X, I_grid = np.meshgrid(x_vals, I_vals)

# Adjust I_grid for negative inputs as in the system
adjusted_I_grid = np.copy(I_grid)
adjusted_I_grid[adjusted_I_grid < 0] = 0.1 * adjusted_I_grid[adjusted_I_grid < 0]

# Compute the derivative of U(x)
dU_dx = 2 * X - 6 * X**2 + 4 * X**3  # Consistent dU/dx

# Compute dx/dt for the grid with sensitivity parameter
dXdt = -s * dU_dx + adjusted_I_grid


# Function to compute fixed points for a given I
def fixed_points(I, s):
    # Adjust I as in the system
    adjusted_I = I
    if I < 0:
        adjusted_I = 0.3 * I

    def func(x):
        return s * (2 * x - 6 * x**2 + 4 * x**3) - adjusted_I

    guesses = [0.0, 0.5, 1.0]
    roots = []
    for guess in guesses:
        root, info, ier, mesg = fsolve(func, guess, full_output=True)
        if ier == 1 and 0.0 <= root[0] <= 1.0:
            roots.append(root[0])
    return np.unique(roots)


# Compute fixed points over a range of I values
I_values = np.linspace(-0.5, 0.5, 150)
stable_fp_x = []
stable_fp_I = []
unstable_fp_x = []
unstable_fp_I = []

for I_val in I_values:
    x_fps = fixed_points(I_val, s)
    for x_fp in x_fps:
        # Compute second derivative of U(x) at x_fp
        d2U_dx2 = 2 - 12 * x_fp + 12 * x_fp**2  # Consistent d2U/dx2
        # Compute the eigenvalue (lambda)
        lambda_ = -s * d2U_dx2
        if lambda_ > 0:
            # Unstable fixed point
            unstable_fp_x.append(x_fp)
            unstable_fp_I.append(I_val)
        else:
            # Stable fixed point
            stable_fp_x.append(x_fp)
            stable_fp_I.append(I_val)

# Plotting the flow field and system trajectory
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

# Plotting the system trajectory
from scipy.interpolate import interp1d

x_interp = interp1d(t, x, kind="linear")
I_interp = interp1d(t, I_over_time, kind="linear")
t_fine = np.linspace(0, 100, 1000)
x_fine = x_interp(t_fine)
I_fine = I_interp(t_fine)
ax.plot(I_fine, x_fine, "k-", linewidth=2, label="System Trajectory")

# Plotting the fixed points with colors according to I
# For stable fixed points
scatter_stable = ax.scatter(
    stable_fp_I,
    stable_fp_x,
    c=stable_fp_I,
    cmap=cmap,
    s=40,
    label="Stable Fixed Points",
)
# For unstable fixed points
scatter_unstable = ax.scatter(
    unstable_fp_I,
    unstable_fp_x,
    c=unstable_fp_I,
    marker="x",
    cmap=cmap,
    s=40,
    label="Unstable Fixed Points",
)

# Adjusting plot aesthetics
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.1, 1.1)
ax.grid(False)
ax.legend()
ax.set_xlabel("Input I")
ax.set_ylabel("State x")
ax.set_title("Flow Field and System Trajectory")
# === PANEL F (bottom): ideal vs. underfit state-space diagram — Underfitting ===
# Saves: outputs/panelF_underfitting_flow_field.pdf
fig.savefig("outputs/panelF_underfitting_flow_field.pdf")
plt.show()

# %% [markdown]
# ## Cell 2 — Panel G (Invented Features)
#
# Produces:
# - `panelG_invented_features_3d_embedding.pdf` — Panel G (bottom): 3D state-space with an invented extra latent dimension
# - `panelG_invented_features_rates.pdf` — Panel G (top): inputs + extra latent dimension ẑ* over time

import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d


# Re-simulate the original 1-BFF system (matches Panel B), independent of Panel F's state
def system(x, t, I_func, s):
    I = I_func(t)
    dU_dx = 2 * x - 6 * x**2 + 4 * x**3
    return -s * dU_dx + I


def I_func(t):
    I_pos = 0.475 * np.exp(-((t - 10.0) ** 2) / (2 * 0.7**2))
    I_neg = -0.475 * np.exp(-((t - 60.0) ** 2) / (2 * 0.7**2))
    return I_pos + I_neg


t = np.linspace(0, 100, 50000)
x = odeint(system, 0.0, t, args=(I_func, 1.0), rtol=1e-10, atol=1e-12).flatten()
I_over_time = np.array([I_func(ti) for ti in t])

x_interp = interp1d(t, x, kind="linear")
I_interp = interp1d(t, I_over_time, kind="linear")

t_fine = np.linspace(0, 100, 1000)
x_fine = x_interp(t_fine)
I_fine = I_interp(t_fine)


# Invented extra feature: cumulative count of input "events" gated on second pulse
def extra_features2(x_fine):
    dif_x_fine = np.abs(np.gradient(x_fine)) > 0.005
    dif_x_fine[:500] = 0
    return x_fine, 0.01 * np.cumsum(dif_x_fine)


n1_fine, n2_fine = extra_features2(x_fine)

# 3D embedding with the invented extra feature
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(
    I_fine,
    n1_fine,
    n2_fine,
    "-",
    color="orange",
    linewidth=2,
    label="System Trajectory",
)
ax.plot(
    I_fine,
    n1_fine,
    np.zeros_like(n2_fine),
    "k-",
    linewidth=2,
    label="System Trajectory",
)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(elev=40, azim=-40)

# === PANEL G (bottom): 3D state-space with invented feature/extra FP — Invented Features ===
# Saves: outputs/panelG_invented_features_3d_embedding.pdf
fig.savefig("outputs/panelG_invented_features_3d_embedding.pdf")
plt.show()

# Extra-feature traces over time
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(n1_fine, label="Neuron 1")
ax.plot(n2_fine, label="Neuron 2")
ax.plot(I_fine, label="Input I")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# === PANEL G (top): inputs + extra latent dimension ẑ* over time — Invented Features ===
# Saves: outputs/panelG_invented_features_rates.pdf
fig.savefig("outputs/panelG_invented_features_rates.pdf")
plt.show()

# %% [markdown]
# ## Cell 3 — Panel H (Dynamical Misattribution)
#
# Produces:
# - `panelH_misattribution_flow_field.pdf` — Panel H (bottom): ideal vs. poor-input flow field
# - `panelH_misattribution_rates.pdf` — Panel H (top): inferred inputs do not match the true inputs

import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d


# Re-simulate the original 1-BFF system (matches Panel B), independent of Panel F/G state
def system(x, t, I_func, s):
    I = I_func(t)
    dU_dx = 2 * x - 6 * x**2 + 4 * x**3
    return -s * dU_dx + I


def I_func(t):
    I_pos = 0.475 * np.exp(-((t - 10.0) ** 2) / (2 * 0.7**2))
    I_neg = -0.475 * np.exp(-((t - 60.0) ** 2) / (2 * 0.7**2))
    return I_pos + I_neg


t = np.linspace(0, 100, 50000)
x = odeint(system, 0.0, t, args=(I_func, 1.0), rtol=1e-10, atol=1e-12).flatten()

x_interp = interp1d(t, x, kind="linear")
# Misattribution: the "inferred input" is taken from the state itself, not the true I(t)
I_interp = interp1d(t, x, kind="linear")

t_fine = np.linspace(0, 100, 1000)
x_fine = x_interp(t_fine)
I_fine = I_interp(t_fine)

n1_fine = x_fine
n2_fine = x_fine

# Flow-field panel: trajectory of misattributed system
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.plot(I_fine, n1_fine, "-", color="orange", linewidth=2, label="System Trajectory")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# === PANEL H (bottom): ideal vs. poor-input flow field — Dynamical Misattribution ===
# Saves: outputs/panelH_misattribution_flow_field.pdf
fig.savefig("outputs/panelH_misattribution_flow_field.pdf")
plt.show()

# Inferred-input traces over time
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.plot(x_fine, label="Neuron 1")
ax.plot(n2_fine, label="Neuron 2")
ax.plot(I_fine, label="Input I")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# === PANEL H (top): inferred inputs do not match the true inputs — Dynamical Misattribution ===
# Saves: outputs/panelH_misattribution_rates.pdf
fig.savefig("outputs/panelH_misattribution_rates.pdf")
plt.show()
