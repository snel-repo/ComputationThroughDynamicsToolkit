import numpy as np
import torch
import torch.nn as nn


class LossFunc:
    def __init__():
        pass

    def __call__(self, loss_dict):
        pass


class RandomTargetLoss(LossFunc):
    def __init__(
        self, position_loss, pos_weight, act_weight, full_trial_epoch: int = 200
    ):
        """Initialize the loss function
        Args:
            position_loss (nn.Module): The loss function to use for the position
            pos_weight (float): The weight to apply to the position loss
            act_weight (float): The weight to apply to the action loss
            full_trial_epoch (int): The number of epochs
            before the full trial is included in the loss"""
        self.position_loss = position_loss
        self.action_loss = nn.MSELoss()
        self.pos_weight = pos_weight
        self.act_weight = act_weight
        self.full_trial_epoch = full_trial_epoch

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        act = loss_dict["actions"]
        epoch = loss_dict["epoch"]
        n_time = pred.shape[1]
        # Gradually increase the percent of the trial to include in the loss
        include_loss = np.ceil(n_time * min(1.0, epoch / self.full_trial_epoch)).astype(
            int
        )
        pos_loss = self.pos_weight * self.position_loss(
            pred[:, :include_loss, :], target[:, :include_loss, :]
        )
        act_loss = self.act_weight * self.action_loss(act, torch.zeros_like(act))
        return pos_loss + act_loss


class NBFFLoss(LossFunc):
    def __init__(self, transition_blind):
        """Initialize the loss function
        Args:
            transition_blind (int): The number of steps to
            ignore the effect of transitions for"""

        self.transition_blind = transition_blind

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]

        # Find where the change in the target is not zero
        # Step 1: Find where the transitions occur (change in value)
        transitions = torch.diff(target, dim=1) != 0

        # Initialize the mask with ones, with one less column (due to diff)
        mask = torch.ones_like(transitions, dtype=torch.float)

        # Step 2: Propagate the effect of transitions for 'transition_blind' steps
        for i in range(1, self.transition_blind + 1):
            # Shift the transition marks to the right to affect subsequent values
            shifted_transitions = torch.cat(
                (torch.zeros_like(transitions[:, :i]), transitions[:, :-i]), dim=1
            )
            mask = mask * (1 - shifted_transitions.float())

        # Step 3: Adjust mask size to match the original target tensor
        # Adding a column of ones at the beginning because diff reduces the size by 1
        final_mask = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)
        final_mask[:, 0:5, :] = 0.0

        loss = nn.MSELoss(reduction="none")(pred, target) * final_mask
        return loss.mean()


class MatchTargetLossMSE(LossFunc):
    def __init__(self):
        pass

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return nn.MSELoss()(pred, target)


class MultiTaskLoss(LossFunc):
    def __init__(self, lat_loss_weight=1e-6):
        self.lat_loss_weight = lat_loss_weight
        pass

    def __call__(self, loss_dict):

        """Calculate the loss"""
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        latents = loss_dict["latents"]
        # action = loss_dict["actions"]
        inputs = loss_dict["inputs"]
        extras = loss_dict["extra"]
        resp_start = extras[:, 0].long()
        resp_end = extras[:, 1].long()
        recon_loss = nn.MSELoss(reduction="none")(pred, target)
        mask = torch.ones_like(recon_loss)
        mask_lats = torch.ones_like(latents)

        # Ignore the first 5 time steps and the time steps after the response
        mask[:, 0:5, :] = 0
        for i in range(inputs.shape[0]):
            mask[i, resp_start[i] : resp_end[i], :] = 5.0
            mask[i, resp_start[i] : resp_start[i] + 5, :] = 0.0
            mask[i, resp_end[i] :, :] = 0.0
            # Mask the latents after the response
            mask_lats[i, resp_end[i] :, :] = 0

        masked_loss = recon_loss * mask
        lats_loss = (
            nn.MSELoss(reduction="none")(latents, torch.zeros_like(latents)) * mask_lats
        )

        total_loss = (
            masked_loss.sum(dim=1).mean()
            + self.lat_loss_weight * lats_loss.sum(dim=1).mean()
        )
        return total_loss


class PhaseCodedMemoryLoss(LossFunc):
    """
    MSE on the output ONLY after the stimulus offset (given in `extra`),
    plus an optional L2 penalty on the latents in that same window.

    loss_dict keys expected:
      - "controlled": (B, T, 1)  model outputs
      - "targets"   : (B, T, 1)  ground truth
      - "extra"     : (B,)       integer offset indices (inclusive)
      - "latents"   : (B, T, L)  OPTIONAL, only needed if latent penalty used
    """

    def __init__(
        self,
        post_stim_l2_weight: float = 0.0,
        eps: float = 1e-8,
        lat_l2_full: bool = False,
        pred_loss_full: bool = False,
    ):
        self.post_stim_l2_weight = post_stim_l2_weight
        self.eps = eps
        self.lat_l2_full = lat_l2_full
        self.pred_loss_full = pred_loss_full

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]  # (B, T, 1)
        target = loss_dict["targets"]  # (B, T, 1)
        signal_ind = loss_dict.get("extra")  # (B,2)
        onset_idx = signal_ind[:, 0] if signal_ind is not None else None
        offset_idx = signal_ind[:, 1] if signal_ind is not None else None

        if offset_idx is None:
            raise ValueError("PhaseCodedMemoryLoss requires 'extra' for offsets.")

        B, T, _ = pred.shape
        device = pred.device

        # ----- Mask for post-offset window -----
        time = torch.arange(T, device=device)  # (T,)
        mask_off = time.unsqueeze(0) >= offset_idx.unsqueeze(1)
        mask_on = time.unsqueeze(0) < onset_idx.unsqueeze(1)  # (B, T)
        mask = mask_off | mask_on  # (B, T)
        mask_full = mask.float().squeeze()  # (B, T, 1)
        mask_post = mask_off.float()  # (B, T)

        # ----- Output MSE in masked window -----
        se = (pred - target).pow(2).squeeze(-1)  # (B, T)
        if self.pred_loss_full:
            # If pred_loss_full is True, use the full prediction window
            mse_per_trial = (se * mask_full).sum(dim=1) / mask_full.sum(
                dim=1
            ).clamp_min(1)
        else:
            mse_per_trial = (se * mask_post).sum(dim=1) / mask_post.sum(
                dim=1
            ).clamp_min(1)
        mse_loss = mse_per_trial.mean()

        # ----- Latent L2  -----
        lat_loss = torch.tensor(0.0, device=device)
        if self.post_stim_l2_weight > 0.0:
            lats = loss_dict.get("latents")
            mean_lats = torch.zeros((B, lats.shape[-1]), device=device)  # (B, L)

            if lats is None:
                raise ValueError("Latent penalty requested but 'latents' missing.")
            # L2 norm per timestep (sqrt of sum of squares across latent dims)
            for i in range(lats.shape[0]):
                mean_lats[i, :] = lats[i, offset_idx[i].long() :].sum(dim=0) / (
                    T - offset_idx[i].long()
                )
            mean_lats_sq = mean_lats**2
            sum_mean_lats_sq = mean_lats_sq.sum(dim=1) / B

            lat_loss = sum_mean_lats_sq.mean()

        return mse_loss + self.post_stim_l2_weight * lat_loss


class MatchTargetLossIncrementalMSE(LossFunc):
    def __init__(self, init_bins=5, group_size=10, epochs_per_group=50):
        """Initialize the loss function
        Args:
            init_bins (int): The number of initial bins to use for the loss"""
        self.init_bins = init_bins
        self.group_size = group_size
        self.epochs_per_group = epochs_per_group

    def __call__(self, loss_dict):
        pred = loss_dict["controlled"]
        target = loss_dict["targets"]
        epoch = loss_dict["epoch"]
        n_time = pred.shape[1]

        # Calculate the number of bins to use
        n_bins = min(
            self.init_bins + self.group_size * (epoch // self.epochs_per_group), n_time
        )
        # Calculate the bin size

        # action = loss_dict["actions"]
        # inputs = loss_dict["inputs"]
        return nn.MSELoss()(pred[:, :n_bins, :], target[:, :n_bins, :])
