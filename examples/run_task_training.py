import os

import ray

# LOCAL_MODE = True  # Set to True to run locally for debugging
# if LOCAL_MODE:
#     ray.init(local_mode=True, num_gpus=0)  # Ensure no GPUs are requested
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""

import shutil
from pathlib import Path

import dotenv
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from ctd.task_modeling.task_training import train
from utils import generate_paths, trial_function

dotenv.load_dotenv(override=True)


# ---------------Options---------------
OVERWRITE = True  # Set to True to overwrite existing run

RUN_DESC = "NBFF_test"  # Description of the run
TASK = "NBFF"  # Task to train on (see configs/task_env for options)
MODEL = "NoisyGRULatentL2"  # Model to train (see configs/model for options)

# ----------------- Parameter Selection -----------------------------------
SEARCH_SPACE = {
    "trainer.max_epochs": tune.choice([100]),
    # "trainer.gradient_clip_val": tune.choice([0.5]),
    # "model.recurrent_gain": tune.grid_search([2.2]),
    # "model.input_trainable": tune.choice([True]),
    "datamodule_task.batch_size": tune.grid_search([256]),
    # 'task_wrapper.weight_decay': tune.choice([1e-5]),
    "params.seed": tune.grid_search([0]),
}

# ------------------Data Management --------------------------------
combo_dict = generate_paths(RUN_DESC, TASK, MODEL)
path_dict = combo_dict["path_dict"]
RUN_TAG = combo_dict["RUN_TAG"]
RUN_DIR = combo_dict["RUN_DIR"]
config_dict = combo_dict["config_dict"]

# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: str,
    config_dict: dict,
):

    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(parents=True)
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    tune.run(
        tune.with_parameters(
            train,
            run_tag=run_tag_in,
            path_dict=path_dict,
            config_dict=config_dict,
        ),
        metric="loss",
        mode="min",
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=4, gpu=0),
        num_samples=1,
        storage_path=str(RUN_DIR),
        search_alg=BasicVariantGenerator(),
        scheduler=FIFOScheduler(),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["loss", "training_iteration"],
            sort_by_metric=True,
        ),
        trial_dirname_creator=trial_function,
    )


if __name__ == "__main__":
    main(
        run_tag_in=RUN_TAG,
        path_dict=path_dict,
        config_dict=config_dict,
    )
