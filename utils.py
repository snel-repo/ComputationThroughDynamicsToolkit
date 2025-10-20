import os
from datetime import datetime
from pathlib import Path


def generate_paths(RUN_DESC: str, TASK: str, MODEL: str):
    # ------------------Data Management --------------------------------

    HOME_DIR = Path(os.environ.get("HOME_DIR"))
    paths = dict(
        tt_datasets=HOME_DIR / "content" / "datasets" / "tt",
        sim_datasets=HOME_DIR / "content" / "datasets" / "sim",
        dt_datasets=HOME_DIR / "content" / "datasets" / "dd",
        trained_models=HOME_DIR / "content" / "trained_models",
    )
    for key, val in paths.items():
        if not val.exists():
            val.mkdir(parents=True)

    DATE_STR = datetime.now().strftime("%Y%m%d")
    RUN_TAG = f"{DATE_STR}_{RUN_DESC}"
    RUN_DIR = HOME_DIR / "content" / "runs" / "task-trained" / RUN_TAG

    # -----------------Default Parameter Sets -----------------------------------
    configs = dict(
        task_wrapper=Path(f"configs/task_wrapper/{TASK}.yaml"),
        env_task=Path(f"configs/env_task/{TASK}.yaml"),
        env_sim=Path(f"configs/env_sim/{TASK}.yaml"),
        datamodule_task=Path(f"configs/datamodule_train/datamodule_{TASK}.yaml"),
        datamodule_sim=Path(f"configs/datamodule_sim/datamodule_{TASK}.yaml"),
        model=Path(f"configs/model/{MODEL}.yaml"),
        simulator=Path(f"configs/simulator/default_{TASK}.yaml"),
        callbacks=Path(f"configs/callbacks/default_{TASK}.yaml"),
        loggers=Path("configs/logger/default.yaml"),
        trainer=Path("configs/trainer/default.yaml"),
    )
    output_dict = {
        "path_dict": paths,
        "RUN_TAG": RUN_TAG,
        "RUN_DIR": RUN_DIR,
        "config_dict": configs,
    }
    return output_dict


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


def flatten(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.
    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.
    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.
    Returns
    -------
    dict
        The flattened dictionary.
    See Also
    --------
    lfads_tf2.utils.unflatten : Performs the opposite of this operation.
    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def trial_function(trial):
    return trial.experiment_tag
