import logging
import os
import pickle
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from gymnasium import Env
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger

from utils import flatten

log = logging.getLogger(__name__)


def train(
    overrides: dict = {},
    config_dict: dict = {},
    run_tag: str = "",
    path_dict: dict = {},
):
    """
    Completes the following sequence of steps:
    1. Instantiate environments
       - task_env is for training
       - sim_env is for simulating the neural data
    2. Instantiate model
       - init_model with the correct input and output sizes
    3. Instantiate task-wrapper
       - Set wrapper environment and model
    4. Instantiate training datamodule
       - Set datamodule environment
    5. Instantiate simulator
    6. Instantiate callbacks
    7. Instantiate loggers
    8. Instantiate trainer
    9. Train model
    10. Save model + datamodule
    11. Instantiate sim datamodule
    12. Simulate neural data
    13. Save simulator + sim datamodule

    Args:
        overrides (dict):
        config_dict (dict):
        run_tag (str):
        path_dict (dict):

    Returns:
        None
    """
    compose_list = config_dict.keys()
    # Convert the overrides dict into a list of override strings
    overrides_list = [f"{k}={v}" for k, v in overrides.items()]

    # Generate a run_name from the overrides
    run_list = []
    for k, v in overrides.items():
        if isinstance(v, float):
            v = "{:.2E}".format(v)
        k_list = k.split(".")
        run_list.append(f"{k_list[-1]}={v}")
    run_name = "_".join(run_list)

    # Compose the configs for all components
    config_all = {}
    for field in compose_list:
        with hydra.initialize(
            config_path=str(config_dict[field].parent), job_name=field
        ):
            # Filter overrides relevant to this field
            field_prefix = f"{field}."
            field_overrides = [
                override
                for override in overrides_list
                if override.startswith(field_prefix)
            ]
            # Remove the field prefix from the overrides
            field_overrides = [
                override[len(field_prefix) :]
                if override.startswith(field_prefix)
                else override
                for override in field_overrides
            ]
            config_all[field] = hydra.compose(
                config_name=config_dict[field].name, overrides=field_overrides
            )

    # Get all of the overrides that start with "params"
    p_overrides = {k: v for k, v in overrides.items() if k.startswith("params.")}
    if "params.seed" in p_overrides:
        pl.seed_everything(overrides["params.seed"], workers=True)
        config_all["datamodule_task"]["seed"] = overrides["params.seed"]
        config_all["datamodule_sim"]["seed"] += overrides["params.seed"]
    else:
        pl.seed_everything(0, workers=True)

    env_overrides = {k: v for k, v in overrides.items() if k.startswith("env_params.")}
    # Set shared params for both the task and sim envs
    for k, v in env_overrides.items():
        config_all["env_task"][k.split(".")[1]] = v
        config_all["env_sim"][k.split(".")[1]] = v

    # Order of operations:

    # 1. Instantiate loggers and log HPs
    # 2. Instantiate environments
    #    - task_env is for training
    #    - sim_env is for simulating the neural data
    # 3. Instantiate model
    #    - init_model with the correct input and output sizes
    # 4. Instantiate task-wrapper
    #    - Set wrapper environment and model
    # 5. Instantiate training datamodule
    #    - Set datamodule environment
    # 6. Instantiate simulator
    # 7. Instantiate callbacks
    # 8. Instantiate trainer
    # 9. Train model
    # 10. Save model + datamodule
    # 11. Instantiate sim datamodule
    # 12. Simulate neural data
    # 13. Save simulator + sim datamodule

    # -----Step 1:------Instantiate loggers and log HPs----------------

    flat_list = flatten(overrides).items()
    run_list = []
    for k, v in flat_list:
        if type(v) == float:
            v = "{:.2E}".format(v)
        k_list = k.split(".")
        run_list.append(f"{k_list[-1]}={v}")
    run_name = "_".join(run_list)

    logger: List[pl.LightningLoggerBase] = []
    if "loggers" in config_all:
        for _, lg_conf in config_all["loggers"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                if lg_conf._target_ == "pytorch_lightning.loggers.WandbLogger":
                    lg_conf["group"] = run_tag
                    lg_conf["name"] = run_name
                    logger.append(hydra.utils.instantiate(lg_conf, log_model=False))
                else:
                    logger.append(hydra.utils.instantiate(lg_conf))

    # --- assemble a pure‐python hyperparam dict from each OmegaConf config ---
    hparams = {}
    for key, cfg in config_all.items():
        # cfg is an OmegaConf object, so this works
        hparams[key] = OmegaConf.to_container(cfg, resolve=True)

    for key, value in overrides.items():
        hparams[key] = value

    # find your WandbLogger and log them
    for lg in logger:
        if isinstance(lg, WandbLogger):
            lg.log_hyperparams(hparams)
            break

    # -----Step 2:----------------Instantiate environments----------------------------
    log.info("Instantiating environment")
    task_env: Env = hydra.utils.instantiate(config_all["env_task"], _convert_="all")

    log.info("Instantiating environment for neural simulation")
    sim_env: Env = hydra.utils.instantiate(config_all["env_sim"], _convert_="all")

    # -----Step 2:-----------------Instantiate model--------------------------------
    log.info(f"Instantiating model <{config_all['model']._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(
        config_all["model"], _convert_="all"
    )
    n_outputs = task_env.action_space.shape[0]
    n_inputs = task_env.observation_space.shape[0] + task_env.context_inputs.shape[0]
    model.init_model(n_inputs, n_outputs)

    # -----Step 3:----------------Instantiate task-wrapper----------------------------
    log.info(f"Instantiating task-wrapper <{config_all['task_wrapper']._target_}")
    task_wrapper: pl.LightningModule = hydra.utils.instantiate(
        config_all["task_wrapper"], _convert_="all"
    )
    task_wrapper.set_environment(task_env)
    task_wrapper.set_model(model)

    # -----Step 4:----------------Instantiate datamodule----------------------------
    log.info("Instantiating datamodule for training")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule_task"], _convert_="all"
    )
    datamodule.set_environment(data_env=task_env)

    # -----Step 5:----------------Instantiate simulator---------------------------
    log.info("Instantiating neural data simulator")
    simulator = hydra.utils.instantiate(config_all["simulator"], _convert_="all")

    # -----Step 6:-----------------Instantiate callbacks---------------------------
    callbacks: List[pl.Callback] = []
    if "callbacks" in config_all:
        for _, cb_conf in config_all["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="all"))

    # ------Step 8:--------------Instantiate trainer---------------------------
    targ_string = config_all["trainer"]._target_
    log.info(f"Instantiating trainer <{targ_string}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config_all["trainer"],
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        _convert_="all",
    )

    # -------Step 9:-----------------Train model---------------------------
    log.info("Training model")
    trainer.fit(model=task_wrapper, datamodule=datamodule)

    # -------Step 10:----------Save model and datamodule---------------------------
    log.info("Saving model and datamodule")
    SAVE_PATH = path_dict["trained_models"] / "task-trained"
    task_wrapper.set_environment(sim_env)

    dir_path = os.path.join(SAVE_PATH, run_tag, run_name, "")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    path1 = os.path.join(SAVE_PATH, run_tag, run_name, "model.pkl")
    with open(path1, "wb") as f:
        pickle.dump(task_wrapper, f)

    path2 = os.path.join(SAVE_PATH, run_tag, run_name, "datamodule_train.pkl")
    with open(path2, "wb") as f:
        pickle.dump(datamodule, f)

    # ------Step 11:------------Instantiate sim datamodule---------------------------
    log.info("Instantiating datamodule for neural simulation")
    sim_datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule_sim"], _convert_="all"
    )
    sim_datamodule.set_environment(
        data_env=sim_env,
        for_sim=True,
    )
    sim_datamodule.prepare_data()
    sim_datamodule.setup()

    task_wrapper.set_environment(sim_env)

    # ------Step 12:------------Simulate neural data---------------------------
    simulator.simulate_neural_data(
        task_trained_model=task_wrapper,
        datamodule=sim_datamodule,
        run_tag=run_tag,
        dataset_path=path_dict["dt_datasets"],
        subfolder=run_name,
        seed=0,
    )

    # ------Step 13:--------Save simulator and sim datamodule------------
    path3 = os.path.join(SAVE_PATH, run_tag, run_name, "simulator.pkl")
    with open(path3, "wb") as f:
        pickle.dump(simulator, f)

    path3 = os.path.join(SAVE_PATH, run_tag, run_name, "datamodule_sim.pkl")
    with open(path3, "wb") as f:
        pickle.dump(sim_datamodule, f)
