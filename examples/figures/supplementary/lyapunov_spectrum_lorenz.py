# %%
"""Validate the full Lyapunov-spectrum tooling on the Lorenz system.

Reviewer 1 noted that the stability of a dynamical system is characterized by
its *whole* Lyapunov spectrum rather than by the maximal exponent alone: a
chaotic set can still be an attracting set when the sum of all exponents is
negative. To make that point concrete -- and to demonstrate that CtDToolkit's
spectrum estimator is correct -- this example computes the full Lyapunov
spectrum of the classic Lorenz attractor and compares it to the literature
reference.

For the standard parameters (sigma=10, rho=28, beta=8/3) the spectrum is

    lambda ~ [ +0.906, 0.000, -14.572 ]   (Sprott, 2003; Viswanath, 1998)

so the maximal exponent is positive (chaos) while the sum is strongly negative
(the flow contracts phase-space volume -- an attracting set). The Kaplan-Yorke
dimension is ~ 2.06.

Run as a script or cell-by-cell (``# %%``).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from ctd.comparison.metrics import compute_lyapunov_spectrum

# Reference (continuous-time) Lyapunov spectra from the literature.
REFERENCE_SPECTRA = {
    "lorenz": np.array([0.906, 0.0, -14.572]),
    "rossler": np.array([0.0714, 0.0, -5.3943]),
}


# --------------------------- Vector fields ---------------------------
def lorenz_f(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    dx = sigma * (Y - X)
    dy = X * (rho - Z) - Y
    dz = X * Y - beta * Z
    return torch.stack([dx, dy, dz], dim=-1)


def rossler_f(x, a=0.2, b=0.2, c=5.7):
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    dx = -Y - Z
    dy = X + a * Y
    dz = b + Z * (X - c)
    return torch.stack([dx, dy, dz], dim=-1)


SYSTEMS = {"lorenz": lorenz_f, "rossler": rossler_f}


def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def build_discrete_jacobians_rk4(system_f, x0, T, dt, burn_in_time=20.0, eps=1e-7):
    """Discrete Jacobians of the RK4 step map along each trajectory.

    Returns
    -------
    states : (B, T + 1, n) tensor
    Phi    : (B, T, n, n) tensor  -- d x_{t+1} / d x_t at each step
    """
    dtype = torch.float64
    x = x0.to(dtype=dtype).clone()
    B, n = x.shape

    for _ in range(int(round(burn_in_time / dt))):  # discard the transient
        x = rk4_step(system_f, x, dt)

    states = torch.empty((B, T + 1, n), dtype=dtype)
    Phi = torch.empty((B, T, n, n), dtype=dtype)
    states[:, 0] = x
    eye = torch.eye(n, dtype=dtype)

    for t in range(T):
        if t % 5000 == 0:
            print(f"  Jacobian step {t} / {T}", end="\r")
        x_next = rk4_step(system_f, x, dt)
        # Central finite differences, vectorized over the n input columns.
        delta = torch.clamp(eps * torch.maximum(torch.ones_like(x), x.abs()), min=1e-12)
        x_plus = x[:, None, :] + delta[:, None, :] * eye[None, :, :]
        x_minus = x[:, None, :] - delta[:, None, :] * eye[None, :, :]
        f_plus = rk4_step(system_f, x_plus.reshape(B * n, n), dt).reshape(B, n, n)
        f_minus = rk4_step(system_f, x_minus.reshape(B * n, n), dt).reshape(B, n, n)
        diff = (f_plus - f_minus) / (2.0 * delta.unsqueeze(-1))
        Phi[:, t] = diff.transpose(1, 2).contiguous()  # (B, n_out, n_in)
        states[:, t + 1] = x_next
        x = x_next
    print()
    return states, Phi


def estimate_spectrum(system, B=4, T=120000, dt=1e-3, seed=0):
    """Estimate the Lyapunov spectrum of ``system`` via the toolkit.

    Returns the spectrum and summary together with the post-burn-in state
    trajectories, so the attractor can be plotted alongside the exponents.
    """
    torch.manual_seed(seed)
    system_f = SYSTEMS[system]
    x0 = 2.0 * torch.rand(B, 3, dtype=torch.float64)
    states, Phi = build_discrete_jacobians_rk4(system_f, x0, T=T, dt=dt)
    # The toolkit returns the full spectrum (descending) plus summary stats.
    spectrum, summary = compute_lyapunov_spectrum(Phi, dt=dt, return_summary=True)
    return spectrum, summary, states


def report(system, spectrum, summary):
    spectrum_mean = spectrum.mean(0).cpu().numpy()
    spectrum_std = spectrum.std(0).cpu().numpy()
    ref = REFERENCE_SPECTRA[system]
    print(f"\n=== {system.capitalize()} Lyapunov spectrum ===")
    print(f"{'i':>3} {'estimate':>12} {'+/- std':>10} {'reference':>12}")
    for i, (m, s, r) in enumerate(zip(spectrum_mean, spectrum_std, ref)):
        print(f"{i + 1:>3} {m:>12.4f} {s:>10.4f} {r:>12.4f}")
    print(f"  max exponent (lambda_1) : {summary['max']:+.4f}  (chaos if > 0)")
    print(f"  sum of exponents        : {summary['sum']:+.4f}  (attracting if < 0)")
    print(f"  Kaplan-Yorke dimension  : {summary['kaplan_yorke_dim']:.4f}")
    return spectrum_mean, spectrum_std


def plot_trajectory(ax, system, states, dt, n_plot=40000):
    """Draw the post-burn-in attractor trajectories in 3D (all grey)."""
    traj_all = states.cpu().numpy()
    for traj in traj_all:  # one line per trajectory in the batch
        if traj.shape[0] > n_plot:
            traj = traj[:n_plot]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            color="0.5",
            lw=0.3,
            alpha=0.7,
        )
    ax.set_title(f"{system.capitalize()} attractor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)


# %%
# --------------------------- Run + plot ---------------------------
if __name__ == "__main__":
    systems_to_run = ["lorenz", "rossler"]
    dt = 1e-3
    results = {}
    for system in systems_to_run:
        print(f"\nEstimating {system} spectrum (RK4 + finite-difference Jacobians)...")
        spectrum, summary, states = estimate_spectrum(system, dt=dt)
        results[system] = (spectrum, summary, states)

    ncols = len(systems_to_run)
    fig = plt.figure(figsize=(4.5 * ncols, 8))
    for col, system in enumerate(systems_to_run):
        spectrum, summary, states = results[system]
        mean, std = report(system, spectrum, summary)

        # Top row: the attractor trajectory that the Jacobians were taken along.
        ax_traj = fig.add_subplot(2, ncols, col + 1, projection="3d")
        plot_trajectory(ax_traj, system, states, dt)

        # Bottom row: the estimated Lyapunov spectrum vs. the literature.
        ax = fig.add_subplot(2, ncols, ncols + col + 1)
        idx = np.arange(1, len(mean) + 1)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.errorbar(
            idx, mean, yerr=std, fmt="o-", color="#1F77B4", capsize=3, label="estimate"
        )
        ax.plot(
            idx,
            REFERENCE_SPECTRA[system],
            "x",
            color="#D62728",
            ms=8,
            label="reference",
        )
        ax.set_title(
            f"{system.capitalize()}  "
            r"($\lambda_{\max}$=" + f"{summary['max']:+.2f}, "
            r"$\sum\lambda$=" + f"{summary['sum']:+.2f})"
        )
        ax.set_xlabel("exponent index")
        ax.set_ylabel("Lyapunov exponent")
        ax.set_xticks(idx)
        ax.legend(frameon=False)
    fig.tight_layout()

    # Save figure for inclusion in the manuscript supplementary materials.
    import os

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    out_dir = os.path.join(repo_root, "manuscript", "figs")
    os.makedirs(out_dir, exist_ok=True)
    out_path_pdf = os.path.join(out_dir, "LyapunovSpectrumValidation.pdf")
    out_path_png = os.path.join(out_dir, "LyapunovSpectrumValidation.png")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to:\n  {out_path_pdf}\n  {out_path_png}")
    plt.show()
