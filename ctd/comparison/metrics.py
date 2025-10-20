from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch import vmap  # convenience alias
from torch.func import jacrev

# import tqdm
from tqdm import tqdm

from ctd.data_modeling.callbacks.metrics import bits_per_spike


def get_signal_r2_linear(
    signal_true_train, signal_pred_train, signal_true_val, signal_pred_val
):
    # Function to compare the latent activity
    if len(signal_pred_train.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = signal_pred_train.shape
        signal_pred_train_flat = (
            signal_pred_train.reshape(-1, n_d_pred).detach().numpy()
        )
        signal_pred_val_flat = signal_pred_val.reshape(-1, n_d_pred).detach().numpy()
    else:
        signal_pred_train_flat = signal_pred_train.detach().numpy()
        signal_pred_val_flat = signal_pred_val.detach().numpy()

    if len(signal_true_train.shape) == 3:
        n_b_true, n_t_true, n_d_true = signal_true_train.shape
        signal_true_train_flat = (
            signal_true_train.reshape(-1, n_d_true).detach().numpy()
        )
        signal_true_val_flat = signal_true_val.reshape(-1, n_d_true).detach().numpy()
    else:
        signal_true_train_flat = signal_true_train.detach().numpy()
        signal_true_val_flat = signal_true_val.detach().numpy()

    # Compare the latent activity
    reg = LinearRegression().fit(signal_true_train_flat, signal_pred_train_flat)
    preds = reg.predict(signal_true_val_flat)
    signal_r2_linear = r2_score(
        signal_pred_val_flat, preds, multioutput="variance_weighted"
    )
    return signal_r2_linear


def get_signal_r2(signal_true, signal_pred):
    """
    Function to compare the activity of the different model
    without a linear transformation

    Typically used for comparisons of rates to true rates
    """
    if len(signal_pred.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = signal_pred.shape
        signal_pred_flat = (
            signal_pred.reshape(n_b_pred * n_t_pred, n_d_pred).detach().numpy()
        )
    else:
        signal_pred_flat = signal_pred.detach().numpy()

    if len(signal_true.shape) == 3:
        n_b_true, n_t_true, n_d_true = signal_true.shape
        signal_true_flat = (
            signal_true.reshape(n_b_true * n_t_true, n_d_true).detach().numpy()
        )
    else:
        signal_true_flat = signal_true.detach().numpy()

    signal_r2 = r2_score(
        signal_true_flat, signal_pred_flat, multioutput="variance_weighted"
    )
    return signal_r2


def get_cycle_consistency(
    inf_latents_train,
    inf_rates_train,
    inf_latents_val,
    inf_rates_val,
    variance_threshold=0.01,
):
    """
    Computes the variance-weighted R² score between the original latent variables
    and the reconstructed latent variables after applying
    singular value thresholding during reconstruction.

    Parameters:
    inf_latents_train (numpy.ndarray): Inferred latent variables for training,
        shape can be (n_samples, n_latents) or (n_batches, n_time_steps, n_latents).
    inf_rates_train (numpy.ndarray): Inferred rates for training,
        shape can be (n_samples, n_neurons) or (n_batches, n_time_steps, n_neurons).
    inf_latents_val (numpy.ndarray): Inferred latent variables for validation,
        same shape considerations as training latents.
    inf_rates_val (numpy.ndarray): Inferred rates for validation,
        same shape considerations as training rates.
    variance_threshold (float): Threshold for cumulative variance
        to retain in singular values (e.g., 0.01 for 1%).

    Returns:
    float: Variance-weighted R² score between
        original and reconstructed latent variables.
    """

    def reconstruct_latents(
        lin_reg_model, N_pred, variance_threshold=variance_threshold
    ):
        """
        Reconstructs the latent variables from predicted log-rates using
        the pseudoinverse of the readout matrix, applying singular value thresholding.

        Parameters:
        lin_reg_model (LinearRegression): Trained LinearRegression model
            mapping latents to log-rates.
        N_pred (numpy.ndarray): Predicted log-rates, shape (n_samples, n_neurons).
        variance_threshold (float): Threshold for cumulative variance to retain.

        Returns:
        numpy.ndarray: Reconstructed latent variables, shape (n_samples, n_latents).
        """
        # Extract the estimated readout matrix (coefficients) and intercept
        W_hat = lin_reg_model.coef_  # Shape: (n_neurons, n_latents)
        b_hat = lin_reg_model.intercept_  # Shape: (n_neurons,)

        # Ensure N_pred is a 2D array
        if N_pred.ndim == 1:
            N_pred = N_pred.reshape(-1, 1)

        # Subtract the intercept from the predicted log-rates
        N_centered = N_pred - b_hat  # Shape: (n_samples, n_neurons)

        # Perform SVD on W_hat
        U, Sigma, Vt = np.linalg.svd(
            W_hat, full_matrices=False
        )  # W_hat = U @ diag(Sigma) @ Vt

        # Compute normalized squared singular values (variance explained)
        normalized_variance = (Sigma**2) / np.sum(Sigma**2)

        # Compute cumulative variance
        cumulative_variance = np.cumsum(normalized_variance)

        # Determine number of components to retain to capture desired variance
        num_components = (
            np.searchsorted(cumulative_variance, (1 - variance_threshold)) + 1
        )

        # Ensure num_components does not exceed total number of components
        num_components = min(num_components, len(Sigma))

        # Truncate the singular values and corresponding matrices
        U_trunc = U[:, :num_components]  # Shape: (n_neurons, num_components)
        Sigma_trunc = Sigma[:num_components]  # Shape: (num_components,)
        Vt_trunc = Vt[:num_components, :]  # Shape: (num_components, n_latents)

        # Compute the truncated pseudoinverse
        Sigma_inv_trunc = np.diag(
            1 / Sigma_trunc
        )  # Shape: (num_components, num_components)
        W_pinv_trunc = (
            Vt_trunc.T @ Sigma_inv_trunc @ U_trunc.T
        )  # Shape: (n_latents, n_neurons)

        # Reconstruct the latent variables
        L_hat = N_centered @ W_pinv_trunc.T  # Shape: (n_samples, n_latents)

        return L_hat

    # Flatten training latent variables if necessary
    if len(inf_latents_train.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = inf_latents_train.shape
        inf_latents_train_flat = inf_latents_train.reshape(-1, n_d_pred)
        inf_latents_val_flat = inf_latents_val.reshape(-1, n_d_pred)
    else:
        inf_latents_train_flat = inf_latents_train
        inf_latents_val_flat = inf_latents_val

    # Flatten training rates if necessary
    if len(inf_rates_train.shape) == 3:
        n_b_true, n_t_true, n_d_true = inf_rates_train.shape
        inf_rates_train_flat = inf_rates_train.reshape(-1, n_d_true)
    else:
        inf_rates_train_flat = inf_rates_train

    # Compute log-rates
    inf_logrates_train_flat = np.log(inf_rates_train_flat)
    # inf_logrates_val_flat = np.log(inf_rates_val_flat)

    pca_lats = PCA()
    pca_lats.fit(inf_latents_train_flat)
    inf_latents_train_flat = pca_lats.transform(inf_latents_train_flat)

    pca_logrates = PCA()
    pca_logrates.fit(inf_logrates_train_flat)
    inf_logrates_train_flat = pca_logrates.transform(inf_logrates_train_flat)

    inf_latents_val_flat = pca_lats.transform(inf_latents_val_flat)

    # Fit linear regression model from latent variables to log-rates
    emp_readout = LinearRegression()
    emp_readout.fit(inf_latents_train_flat, inf_logrates_train_flat)

    # Predict log-rates from validation latent variables
    preds = emp_readout.predict(inf_latents_val_flat)

    # Reconstruct latent variables from predicted log-rates
    latent_pred_flat = reconstruct_latents(
        emp_readout, preds, variance_threshold=variance_threshold
    )

    # Compute variance-weighted R² score between original
    # and reconstructed latent variables
    r2 = r2_score(
        inf_latents_val_flat, latent_pred_flat, multioutput="variance_weighted"
    )

    return r2


def get_bps(inf_rates, true_spikes):
    # Flatten training latent variables if necessary
    if len(inf_rates.shape) == 3:
        n_b_pred, n_t_pred, n_d_pred = inf_rates.shape
        inf_rates_flat = inf_rates.reshape(-1, n_d_pred)
    else:
        inf_rates_flat = inf_rates

    # Flatten training rates if necessary
    if len(true_spikes.shape) == 3:
        n_b_true, n_t_true, n_d_true = true_spikes.shape
        true_spikes_flat = true_spikes.reshape(-1, n_d_true)
    else:
        true_spikes_flat = true_spikes
    bps = bits_per_spike(
        torch.tensor(np.log(inf_rates_flat)).float(),
        true_spikes_flat.clone().detach().float(),
    ).item()
    return bps


def compute_jacobians(
    z: torch.Tensor,  # (B, T, D)
    u: torch.Tensor,  # (B, T, U)
    f: torch.nn.Module,  # maps (u_t, z_t) -> z_{t+1} of shape (..., D)
    num_trials: Optional[int] = None,  # how many trials to sample from B
    seed: Optional[int] = None,  # for reproducibility
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust vectorized version using functorch.
    Works whether `f` wants batched or unbatched inputs.

    Returns:
        Jz: (N, T, D, D)  -- ∂f/∂z
        Ju: (N, T, D, U)  -- ∂f/∂u
        trial_idx: (N,)    -- indices of the trials
              from the original batch that were used
    """
    B, T, D = z.shape
    _, _, U = u.shape
    device = z.device

    # Sample trials
    if num_trials is not None and num_trials < B:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        trial_idx = torch.randperm(B, generator=gen, device=device)[:num_trials]
    else:
        trial_idx = torch.arange(B, device=device)

    N = trial_idx.shape[0]
    z_sub = z[trial_idx]  # (N, T, D)
    u_sub = u[trial_idx]  # (N, T, U)

    # Flatten (N, T) -> M = N*T
    z_flat = z_sub.reshape(N * T, D)  # (M, D)
    u_flat = u_sub.reshape(N * T, U)  # (M, U)

    # Stable single-sample wrapper: accepts (U,) and (D,) and returns (D,)
    def f_scalar(u_sample: torch.Tensor, z_sample: torch.Tensor) -> torch.Tensor:
        """
        Robustly call f whether it expects batched inputs or not.
        """
        try:
            out = f(u_sample, z_sample)
        except Exception:
            # try with batch dim
            out = f(u_sample.unsqueeze(0), z_sample.unsqueeze(0))
        # squeeze if extraneous batch dim
        if out.ndim == 2 and out.shape[0] == 1:
            out = out.squeeze(0)
        return out  # should be (..., D) with final D

    # Build jacobian functions
    jac_wrt_z = jacrev(f_scalar, argnums=1)  # ∂f/∂z
    jac_wrt_u = jacrev(f_scalar, argnums=0)  # ∂f/∂u

    # vmap over all flattened samples
    Jz_flat = vmap(jac_wrt_z)(u_flat, z_flat)  # (M, D, D)
    Ju_flat = vmap(jac_wrt_u)(u_flat, z_flat)  # (M, D, U)

    # Reshape back to (N, T, D, D) and (N, T, D, U)
    Jz = Jz_flat.view(N, T, D, D)
    Ju = Ju_flat.view(N, T, D, U)

    return Jz, Ju, trial_idx


def compute_lyaps(Js, dt=1, k=None, verbose=False):
    """
    Compute Lyapunov exponents from a sequence of Jacobian matrices.

    This function computes the Lyapunov exponents using the QR decomposition method,
    which tracks the growth rates of perturbations along different directions.

    Parameters
    ----------
    Js : torch.Tensor
        Sequence of Jacobian matrices of shape
            (n_trajectories, time_steps, n_dims, n_dims)
            or (time_steps, n_dims, n_dims)
            NOTE: must be discrete Jacobians!!
    dt : float, optional
        Time step size, by default 1
    k : int, optional
        Number of Lyapunov exponents to compute, by default None (computes all)
    verbose : bool, optional
        Whether to show progress information, by default False

    Returns
    -------
    torch.Tensor
        Lyapunov exponents sorted in descending order
    """
    squeeze = False
    if len(Js.shape) == 3:
        Js = Js.unsqueeze(0)
        squeeze = True

    T, n, _ = Js.shape[-3], Js.shape[-2], Js.shape[-1]
    old_Q = torch.eye(n, device=Js.device, dtype=Js.dtype)

    if k is None:
        k = n

    old_Q = old_Q[:, :k]
    lexp = torch.zeros(*Js.shape[:-3], k, device=Js.device, dtype=Js.dtype)
    lexp_counts = torch.zeros(*Js.shape[:-3], k, device=Js.device, dtype=Js.dtype)

    for t in tqdm(range(T), disable=not verbose):

        # QR-decomposition of Js[t] * old_Q
        mat_Q, mat_R = torch.linalg.qr(torch.matmul(Js[..., t, :, :], old_Q))

        # force diagonal of R to be positive
        # sign_diag = torch.sign(torch.diag(mat_R))
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)
        sign_diag = torch.sign(diag_R)
        sign_diag[sign_diag == 0] = 1
        sign_diag = torch.diag_embed(sign_diag)

        mat_Q = mat_Q @ sign_diag
        mat_R = sign_diag @ mat_R
        old_Q = mat_Q

        # Successively build sum for Lyapunov exponents
        diag_R = mat_R.diagonal(dim1=-2, dim2=-1)

        # Filter zeros in mat_R (would lead to -infs)
        idx = diag_R > 0
        lexp_i = torch.zeros_like(diag_R, dtype=Js.dtype, device=Js.device)
        lexp_i[idx] = torch.log(diag_R[idx])
        lexp[idx] += lexp_i[idx]
        lexp_counts[idx] += 1
    if squeeze:
        lexp = lexp.squeeze(0)
        lexp_counts = lexp_counts.squeeze(0)

    return torch.flip(
        torch.sort((lexp / lexp_counts) * (1 / dt), axis=-1)[0], dims=[-1]
    )


def compute_input_lyaps(
    Jz,
    Ju,
    dt: float = 1.0,
    k: int = None,
    mode: str = "max",
    verbose: bool = False,
    return_details: bool = False,
):
    """
    Input-to-state Lyapunov-like exponents over a trajectory.

    Parameters
    ----------
    Jz : torch.Tensor
        ∂f/∂x along the trajectory.
        Shape: (n_traj, T, D, D) or (T, D, D)
    Ju : torch.Tensor
        ∂f/∂u along the trajectory.
        Shape: (n_traj, T, D, U) or (T, D, U)
    dt : float
        Time step.
    k : int or None
        Number of exponents to return.
        Default: min(D, U) (or for mode="concat", min(D, U*T)).
    mode : {"max","concat"}
        - "max": take the maximum (over injection time s)
        of per-s exponents from Φ(T,s+1) @ Ju(s).
        - "concat": use the full map L_T = [Φ(T,1)Ju(0) ...
        Φ(T,T)Ju(T-1)] and take its top-k rates.
    verbose : bool
        Print simple progress (no external deps).
    return_details : bool
        If True, also returns a dict with per-s gains/exp details.

    Returns
    -------
    exps : torch.Tensor
        Input-to-state exponents, sorted descending.
        Shape: (n_traj, k) or (k,) if single-trajectory input.
    details : dict (optional)
        For mode="max": {"per_s_exps": (n_traj, T, k), "argmax": indices}
        (NaN for s with zero horizon)
        For mode="concat": {"svals": (n_traj, m)} with m = min(D, U*T)
    """
    # Normalize shapes (add batch dim if needed)
    squeeze = False
    if Jz.ndim == 3:
        Jz = Jz.unsqueeze(0)
        squeeze = True
    if Ju.ndim == 3:
        Ju = Ju.unsqueeze(0)

    assert Jz.ndim == 4 and Ju.ndim == 4, "Jz and Ju must be rank-4 or rank-3 tensors"
    assert (
        Jz.shape[0] == Ju.shape[0]
        and Jz.shape[1] == Ju.shape[1]
        and Jz.shape[2] == Ju.shape[2]
    ), "Batch, time, and state dims must match between Jz and Ju"

    B, T, D, _ = Jz.shape
    _, T2, D2, U = Ju.shape
    assert T == T2 and D == D2

    eps = torch.finfo(Jz.dtype).tiny  # to avoid log(0)

    # Helper: build Φ(T, t) for all t using a backward product
    # Φ(T, T) = I; Φ(T, t) = Jz(T-1) ... Jz(t)
    I1 = torch.eye(D, dtype=Jz.dtype, device=Jz.device).expand(B, D, D)
    Phi_list = [None] * (T + 1)
    Phi = I1.clone()
    Phi_list[T] = I1.clone()
    for t in range(T - 1, -1, -1):
        Phi = Phi @ Jz[:, t]  # batched matmul: (B,D,D)@(B,D,D)->(B,D,D)
        Phi_list[t] = Phi

    if mode not in ("max", "concat"):
        raise ValueError('mode must be "max" or "concat"')

    if mode == "max":
        m = min(D, U)
        if k is None:
            k = m
        k = min(k, m)

        # per-s exponents (B, T, k); NaN for s=T-1 (zero horizon)
        per_s_exps = torch.full(
            (B, T, k), float("nan"), dtype=Jz.dtype, device=Jz.device
        )

        rng = range(T - 1)  # last s has horizon 0
        for s in rng:
            if verbose and (s % max(1, T // 10) == 0):
                print(f"[compute_input_lyaps max] s={s}/{T-2}")
            # M_s = Φ(T, s+1) @ Ju(s)  → (B, D, U)
            M = Phi_list[s + 1] @ Ju[:, s]  # (B,D,D)@(B,D,U)->(B,D,U)
            # Top k singular values per batch
            # svdvals returns descending order
            svals = torch.linalg.svdvals(M)[:, :k]  # (B, k)
            # Convert to rates over horizon (T-1-s)*dt
            horizon = (T - 1 - s) * dt
            exps_s = torch.log(torch.clamp(svals, min=eps)) / horizon
            per_s_exps[:, s, :] = exps_s

        # Aggregate: take the top-k over all s and singular directions
        # Flatten (s, k) then topk
        flat = per_s_exps.reshape(B, -1)
        # ignore NaNs by replacing with -inf
        flat = torch.nan_to_num(flat, nan=-float("inf"))
        topk_vals, topk_idx = torch.topk(flat, k=k, dim=1)
        exps = topk_vals  # (B, k)
        # Sort descending just in case topk didn't
        # return sorted (it does, but keep symmetry)
        exps, _ = torch.sort(exps, dim=1, descending=True)

        if squeeze:
            exps = exps.squeeze(0)
            per_s_exps = per_s_exps.squeeze(0)

        if return_details:
            return exps, {
                "per_s_exps": per_s_exps,  # (B, T, k), NaN where undefined
                "argmax": topk_idx
                if not squeeze
                else topk_idx.squeeze(0),  # indices in flattened (s,k)
            }
        return exps

    else:  # mode == "concat"
        # Build L_T = [Φ(T,1)Ju(0) ... Φ(T,T)Ju(T-1)]  → (B, D, U*T)
        blocks = []
        for s in range(T):
            if verbose and (s % max(1, T // 10) == 0):
                print(f"[compute_input_lyaps concat] block {s+1}/{T}")
            blocks.append(Phi_list[s + 1] @ Ju[:, s])  # (B,D,U)
        L = torch.cat(blocks, dim=-1)  # (B, D, U*T)

        m = min(D, U * T)
        if k is None:
            k = m
        k = min(k, m)

        svals = torch.linalg.svdvals(L)[:, :k]  # (B, k), descending
        exps = torch.log(torch.clamp(svals, min=eps)) / (T * dt)

        if squeeze:
            exps = exps.squeeze(0)
            svals = svals.squeeze(0)

        if return_details:
            return exps, {"svals": svals}
        return exps


def extract_pairwise_distances(X, max_pairs=None, seed=42):
    """
    Extract pairwise distances between data points.

    Args:
        X: array of shape (N, D) - data points
        max_pairs: if not None, randomly sample
                this many pairs to avoid memory issues
        seed: random seed for pair sampling

    Returns:
        distances: array of pairwise distances (rotation invariant)
    """
    N = X.shape[0]

    if max_pairs is None or N * (N - 1) // 2 <= max_pairs:
        # Use all pairs
        distances = pdist(X, metric="euclidean")
    else:
        # Randomly sample pairs WITHOUT creating all pairs first
        rng = np.random.RandomState(seed)

        # Generate random pairs directly
        i_indices = rng.randint(0, N, size=max_pairs)
        j_indices = rng.randint(0, N, size=max_pairs)

        # Ensure i != j (no self-pairs)
        mask = i_indices != j_indices
        while not np.all(mask):
            # Regenerate pairs where i == j
            bad_indices = ~mask
            j_indices[bad_indices] = rng.randint(0, N, size=np.sum(bad_indices))
            mask = i_indices != j_indices

        # Compute distances vectorized
        distances = np.linalg.norm(X[i_indices] - X[j_indices], axis=1)

    return distances


def extract_distance_to_origin(X):
    """
    Extract distances from each point to the origin (centroid).
    This captures the "spread" of the distribution.

    Args:
        X: array of shape (N, D)

    Returns:
        distances: distance of each point to centroid
    """
    # Center the data first
    X_centered = X - np.mean(X, axis=0)

    # Distance from each point to the (now zero) centroid
    distances = np.linalg.norm(X_centered, axis=1)

    return distances


def extract_volume_features(X, n_samples=1000, seed=42):
    """
    Extract features related to the volume occupied by the point cloud.
    Uses random sampling for efficiency.

    Args:
        X: array of shape (N, D)
        n_samples: number of random samples for volume estimation
        seed: random seed

    Returns:
        features: volume-related features
    """
    if X.shape[0] <= 1:
        return np.array([])

    rng = np.random.RandomState(seed)

    # Sample subset for efficiency
    n_points = min(n_samples, X.shape[0])
    indices = rng.choice(X.shape[0], n_points, replace=False)
    X_sample = X[indices]

    # Convex hull volume approximation via determinant of covariance
    X_centered = X_sample - np.mean(X_sample, axis=0)
    cov_matrix = np.cov(X_centered.T)

    # Log determinant (more numerically stable)
    sign, logdet = np.linalg.slogdet(cov_matrix)

    # Return both the log determinant and the effective rank
    eigenvals = np.linalg.eigvals(cov_matrix)
    effective_rank = np.sum(eigenvals > 1e-12 * np.max(eigenvals))

    return np.array([logdet, effective_rank])


def extract_density_features(X, n_bins=10):
    """
    Extract features related to the density distribution along each dimension.

    Args:
        X: array of shape (N, D)
        n_bins: number of bins for histogram

    Returns:
        features: concatenated histogram features for each dimension
    """
    if X.shape[0] <= 1:
        return np.array([])

    features = []

    for d in range(X.shape[1]):
        # Histogram of values in dimension d
        hist, _ = np.histogram(X[:, d], bins=n_bins, density=True)
        features.extend(hist)

    return np.array(features)


def rbf_kernel(X, Y, sigma):
    """
    Compute the RBF (Gaussian) kernel matrix between X and Y.
    """
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    distances = XX + YY - 2 * X.dot(Y.T)
    return np.exp(-distances / (2 * sigma**2))


def compute_mmd_rff_simple(X, Y, sigma=None, n_features=500, seed=0):
    """
    Simple MMD computation using Random Fourier Features.

    Args:
        X: array of shape (N1, D)
        Y: array of shape (N2, D)
        sigma: RBF bandwidth
        n_features: number of random Fourier features
        seed: random seed

    Returns:
        mmd_distance: MMD between X and Y
    """
    rng = np.random.RandomState(seed)
    N1, D = X.shape
    N2, _ = Y.shape

    # Estimate sigma using median heuristic if not provided
    if sigma is None:
        # Sample subset for efficiency
        m1 = min(1000, N1)
        m2 = min(1000, N2)

        combined = np.vstack(
            [
                X[rng.choice(N1, m1 // 2, replace=False)],
                Y[rng.choice(N2, m2 // 2, replace=False)],
            ]
        )

        # Compute pairwise distances
        n_combined = combined.shape[0]
        if n_combined > 1:
            diff = combined[:, None, :] - combined[None, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            triu_indices = np.triu_indices(n_combined, k=1)
            dists = np.sqrt(dists_sq[triu_indices])
            sigma = float(np.median(dists))
            if sigma <= 0:
                sigma = 1.0
        else:
            sigma = 1.0

    # Random Fourier Features
    n_cos_sin = n_features // 2
    W = rng.normal(0, 1.0 / sigma, size=(D, n_cos_sin))
    b = rng.uniform(0, 2 * np.pi, size=n_cos_sin)
    scale = np.sqrt(1.0 / n_cos_sin)

    # Compute features
    def compute_features(data):
        projections = data @ W + b
        cos_features = scale * np.cos(projections)
        sin_features = scale * np.sin(projections)
        return np.concatenate([cos_features.mean(axis=0), sin_features.mean(axis=0)])

    phi_X_mean = compute_features(X)
    phi_Y_mean = compute_features(Y)

    return np.linalg.norm(phi_X_mean - phi_Y_mean)


def compute_static_rotation_invariant_mmd(
    x_true,
    x_pred,
    feature_types=["pairwise", "distances"],
    sigma=None,
    n_features=500,
    seed=0,
    max_pairwise_samples=10000,
):
    """
    Compute MMD between two datasets using rotation-invariant features
    that ignore temporal ordering.

    This treats the neural activity as a static distribution
    of latent states, ignoring the temporal sequence and
    focusing only on distributional statistics.
    """

    # Flatten all trajectories into single point clouds (ignoring temporal structure)
    if x_true.ndim == 3:
        X = x_true.reshape(-1, x_true.shape[-1])  # (B*T, D1)
    else:
        X = x_true

    if x_pred.ndim == 3:
        Y = x_pred.reshape(-1, x_pred.shape[-1])  # (B*T, D2)
    else:
        Y = x_pred

    def _approx_median_pairwise_dist(Z, rng, max_pairs=50000):
        n = Z.shape[0]
        if n < 2:
            return 1.0
        # sample pairs without replacement (cap by max_pairs)
        m = min(max_pairs, n * (n - 1) // 2)
        # sample indices then form pairs
        idx = rng.choice(n, size=min(2 * int(np.sqrt(m)) + 1, n), replace=False)
        A = Z[idx]
        d2 = np.sum((A[:, None, :] - A[None, :, :]) ** 2, axis=-1)
        iu = np.triu_indices_from(d2, k=1)
        d = np.sqrt(d2[iu])
        med = np.median(d) if d.size else 1.0
        return float(med) if med > 1e-12 else 1.0

    def _dataset_scale(Zc, method="median_centroid", rng=None):
        # Zc is centered
        if method == "rms_radius":
            s = np.sqrt(np.mean(np.sum(Zc**2, axis=1)))
        elif method == "median_centroid":  # robust & fast (good default)
            s = np.median(np.linalg.norm(Zc, axis=1))
        elif method == "median_pairwise":  # most geometric, a bit heavier
            s = _approx_median_pairwise_dist(Zc, rng=rng)
        else:
            raise ValueError(f"Unknown scale method: {method}")
        return float(s) if s > 1e-12 else 1.0

    rng = np.random.RandomState(seed)

    # 1) Center (translation invariance)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # 2) Pick a geometric scale per dataset (uniform-scale invariance)
    scale_method = "median_centroid"  # or "rms_radius" / "median_pairwise"
    Sx = _dataset_scale(X, method=scale_method, rng=rng)
    Sy = _dataset_scale(Y, method=scale_method, rng=rng)

    # 3) Rescale
    X = X / Sx
    Y = Y / Sy
    # Extract rotation-invariant features for each dataset
    features_X = []
    features_Y = []
    feature_names = []

    if "pairwise" in feature_types:
        pairwise_X = extract_pairwise_distances(X, max_pairwise_samples, seed)
        pairwise_Y = extract_pairwise_distances(Y, max_pairwise_samples, seed + 1)
        if len(pairwise_X) > 0 and len(pairwise_Y) > 0:
            features_X.append(pairwise_X)
            features_Y.append(pairwise_Y)
            feature_names.append("pairwise")

    if "distances" in feature_types:
        dist_X = extract_distance_to_origin(X)
        dist_Y = extract_distance_to_origin(Y)
        if len(dist_X) > 0 and len(dist_Y) > 0:
            features_X.append(dist_X)
            features_Y.append(dist_Y)
            feature_names.append("distances")

    if not features_X or not features_Y:
        raise ValueError("No valid features could be extracted!")

    # Compute MMD for each feature type separately
    mmd_scores = []

    for fx, fy, name in zip(features_X, features_Y, feature_names):

        # Reshape for MMD computation - treat each feature as a 1D point
        fx_reshaped = fx.reshape(-1, 1)
        fy_reshaped = fy.reshape(-1, 1)

        # Compute MMD for this feature type
        mmd_feat = compute_mmd_rff_simple(
            fx_reshaped,
            fy_reshaped,
            sigma=sigma,
            n_features=n_features,
            seed=seed + len(mmd_scores),
        )
        mmd_scores.append(mmd_feat)
        print(f"  {name} MMD: {mmd_feat:.6f}")

    # Combine MMD scores (simple average)
    final_mmd = np.mean(mmd_scores)
    return final_mmd


def compute_nl_cycle_consistency(
    inf_latents_train,
    inf_rates_train,
    inf_latents_val,
    inf_rates_val,
    hidden_sizes=(256, 256),
    lr=1e-3,
    max_epochs=2000,
    patience=50,
    min_delta=1e-5,
    noise_stds=(0.0, 0.01, 0.05, 0.1, 0.2),  # relative to per-dim train latent std
    weight_decay=0.0,
    seed=0,
    device="cpu",
):
    """
    Learn f: log-rates -> latents with early stopping;
            test robustness by adding Gaussian
    noise in log-rate space (original units),
            then evaluate R^2 of f(logR_noisy) vs true Z.

    Returns
    -------
    results : dict with keys:
        - 'val_r2' : baseline variance-weighted R^2 on clean validation data
        - 'noise_stds' : list of noise multipliers
        - 'r2_per_noise' : list of variance-weighted R^2 under each noise level
        - 'enc_state_dict' : best model state_dict
        - 'scalers' : {'logR_mean','logR_std','Z_mean','Z_std'}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------- shape helpers ---------
    def to_2d(x):
        if x.ndim == 3:
            B, T, D = x.shape
            return x.reshape(B * T, D)
        return x

    Z_tr = to_2d(inf_latents_train).astype(np.float32)
    R_tr = to_2d(inf_rates_train).astype(np.float32)
    Z_va = to_2d(inf_latents_val).astype(np.float32)
    R_va = to_2d(inf_rates_val).astype(np.float32)

    # --------- log-rates (numerical safety) ---------
    eps = 1e-8
    logR_tr = np.log(np.clip(R_tr, eps, None))
    logR_va = np.log(np.clip(R_va, eps, None))

    # --------- scalers (fit on TRAIN only) ---------
    def fit_scaler(X):
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True)
        s[s < 1e-8] = 1.0
        return m, s

    def apply_scaler(X, m, s):
        return (X - m) / s

    logR_m, logR_s = fit_scaler(logR_tr)
    Z_m, Z_s = fit_scaler(Z_tr)

    logR_tr_s = apply_scaler(logR_tr, logR_m, logR_s)
    logR_va_s = apply_scaler(logR_va, logR_m, logR_s)
    Z_tr_s = apply_scaler(Z_tr, Z_m, Z_s)
    Z_va_s = apply_scaler(Z_va, Z_m, Z_s)

    in_dim = logR_tr_s.shape[1]
    out_dim = Z_tr_s.shape[1]

    # --------- model ---------
    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim, hidden):
            super().__init__()
            layers = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers += [nn.Linear(d, out_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    enc = MLP(in_dim, out_dim, hidden_sizes).to(device)

    # --------- early-stopped training ---------
    Xtr = torch.tensor(logR_tr_s, device=device)
    Ytr = torch.tensor(Z_tr_s, device=device)
    Xva = torch.tensor(logR_va_s, device=device)
    Yva = torch.tensor(Z_va_s, device=device)

    crit = nn.MSELoss()
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    no_improve = 0

    for epoch in range(max_epochs):
        print(f"\rEpoch {epoch+1}/{max_epochs}...", end="", flush=True)
        enc.train()
        opt.zero_grad()
        pred = enc(Xtr)
        loss = crit(pred, Ytr)
        loss.backward()
        opt.step()

        enc.eval()
        with torch.no_grad():
            val_pred = enc(Xva)
            val_loss = crit(val_pred, Yva).item()

        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in enc.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        enc.load_state_dict(best_state)

    # --------- baseline validation R^2 (original Z units) ---------
    enc.eval()
    with torch.no_grad():
        Z_va_pred_s = enc(torch.tensor(logR_va_s, device=device)).cpu().numpy()
    Z_va_pred = Z_va_pred_s * Z_s + Z_m
    val_r2 = r2_score(Z_va, Z_va_pred, multioutput="variance_weighted")

    # --------- noise in LOG-RATE space (original logR units) ---------
    r2_per_noise = []
    train_logR_std = logR_s.squeeze(0)  # per-dim std in original logR units

    with torch.no_grad():
        for sigma in noise_stds:
            noise = np.random.randn(*logR_va.shape).astype(np.float32) * (
                sigma * train_logR_std
            )
            logR_va_noisy = logR_va + noise  # add in original logR space
            logR_va_noisy_s = (logR_va_noisy - logR_m) / logR_s

            Z_noisy_pred_s = (
                enc(torch.tensor(logR_va_noisy_s, device=device)).cpu().numpy()
            )
            Z_noisy_pred = Z_noisy_pred_s * Z_s + Z_m

            r2 = r2_score(Z_va, Z_noisy_pred, multioutput="variance_weighted")
            r2_per_noise.append(float(r2))
            print(f"  Noise std {sigma:.3f} -> val R^2 {r2:.4f}")

    return {
        "val_r2": float(val_r2),
        "noise_stds": list(noise_stds),
        "r2_per_noise": r2_per_noise,
        "enc_state_dict": best_state,
        "scalers": {
            "logR_mean": logR_m,
            "logR_std": logR_s,
            "Z_mean": Z_m,
            "Z_std": Z_s,
        },
    }
