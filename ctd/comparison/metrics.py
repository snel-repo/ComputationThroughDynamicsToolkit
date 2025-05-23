import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
