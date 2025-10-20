import torch


def compute_l2_penalty(lfads, hps):
    if hps.gen_type == "RNN":
        recurrent_kernels_and_weights = [
            (lfads.encoder.ic_enc.fwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
            (lfads.encoder.ic_enc.bwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
            (lfads.decoder.rnn.cell.gen_cell.weight_hh, hps.l2_gen_scale),
        ]
        for param in lfads.readout.parameters():
            recurrent_kernels_and_weights.append((param, hps.l2_readout_scale))
    elif hps.gen_type == "NODE" or hps.gen_type == "S2S_NODE":
        recurrent_kernels_and_weights = [
            (lfads.encoder.ic_enc.fwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
            (lfads.encoder.ic_enc.bwd_gru.cell.weight_hh, hps.l2_ic_enc_scale),
        ]
        for param in lfads.decoder.rnn.cell.gen_cell.parameters():
            recurrent_kernels_and_weights.append((param, hps.l2_gen_scale))
        for param in lfads.readout.parameters():
            recurrent_kernels_and_weights.append((param, hps.l2_readout_scale))
    if lfads.use_con:
        recurrent_kernels_and_weights.extend(
            [
                (lfads.encoder.ci_enc.fwd_gru.cell.weight_hh, hps.l2_ci_enc_scale),
                (lfads.encoder.ci_enc.bwd_gru.cell.weight_hh, hps.l2_ci_enc_scale),
                (lfads.decoder.rnn.cell.con_cell.weight_hh, hps.l2_con_scale),
            ]
        )
    # Add recurrent penalty
    recurrent_penalty = 0.0
    recurrent_size = 0
    for kernel, weight in recurrent_kernels_and_weights:
        if weight > 0:
            recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
            recurrent_size += kernel.numel()
    recurrent_penalty /= recurrent_size + 1e-8
    # Add recon penalty if applicable
    recon_penalty = 0.0
    if hasattr(lfads.recon, "compute_l2"):
        recon_penalty = lfads.recon.compute_l2()
    return recurrent_penalty + recon_penalty
