def make_data_tag(dm_cfg):
    obs_dim = "" if "obs_dim" not in dm_cfg else dm_cfg.obs_dim
    obs_noise = "" if "obs_noise" not in dm_cfg else dm_cfg.obs_noise
    if "obs_noise_params" in dm_cfg:
        obs_noise_params = ",".join(
            [f"{k}={v}" for k, v in dm_cfg.obs_noise_params.items()]
        )
    else:
        obs_noise_params = ""
    data_tag = (
        f"{dm_cfg.system}{obs_dim}_"
        f"{dm_cfg.n_samples}S_"
        f"{dm_cfg.n_timesteps}T_"
        f"{dm_cfg.pts_per_period}P_"
        f"{dm_cfg.seed}seed"
    )
    if obs_noise:
        data_tag += f"_{obs_noise}{obs_noise_params}"
    return data_tag
