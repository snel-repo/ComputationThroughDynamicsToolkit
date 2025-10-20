import os
import shutil

import dotenv

from ctd.comparison.analysis.tt.tt import Analysis_TT


def copy_folder_contents(src_folder, dest_folder):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate over all files and directories in the source folder
    for item_name in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item_name)
        dest_item = os.path.join(dest_folder, item_name)

        # If it's a directory, copy it recursively
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dest_item)


dotenv.load_dotenv(override=True)
HOME_DIR = os.environ.get("HOME_DIR")
print(HOME_DIR)

tt_3bff_path = HOME_DIR + "pretrained/20241017_NBFF_NoisyGRU_NewFinal/"
tt_MultiTask_path = HOME_DIR + "pretrained/20241113_MultiTask_NoisyGRU_Final2/"
tt_RandomTarget_path = HOME_DIR + "pretrained/20241113_RandomTarget_NoisyGRU_Final2/"
tt_PhaseCodedMemory_path = HOME_DIR + "pretrained/PCM_NoisyGRU_Final/"

tt_3bff = Analysis_TT(run_name="tt_3bff", filepath=tt_3bff_path)
tt_MultiTask = Analysis_TT(run_name="tt_MultiTask", filepath=tt_MultiTask_path)
tt_RandomTarget = Analysis_TT(run_name="tt_RandomTarget", filepath=tt_RandomTarget_path)
tt_PCM = Analysis_TT(run_name="tt_PhaseCodedMemory", filepath=tt_PhaseCodedMemory_path)

# Make copies of the pretrained models to the trained_models folder
# if the folders don't already exist
path_3bff = HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
path_MultiTask = HOME_DIR + "content/trained_models/task-trained/tt_MultiTask/"
path_RandomTarget = HOME_DIR + "content/trained_models/task-trained/tt_RandomTarget/"
path_PhaseCodedMemory = (
    HOME_DIR + "content/trained_models/task-trained/tt_PhaseCodedMemory/"
)

if not os.path.exists(path_3bff):
    copy_folder_contents(
        tt_3bff_path, HOME_DIR + "content/trained_models/task-trained/tt_3bff/"
    )

if not os.path.exists(path_MultiTask):
    copy_folder_contents(
        tt_MultiTask_path,
        HOME_DIR + "content/trained_models/task-trained/tt_MultiTask/",
    )

if not os.path.exists(path_RandomTarget):
    copy_folder_contents(
        tt_RandomTarget_path,
        HOME_DIR + "content/trained_models/task-trained/tt_RandomTarget/",
    )

if not os.path.exists(path_PhaseCodedMemory):
    copy_folder_contents(
        tt_PhaseCodedMemory_path,
        HOME_DIR + "content/trained_models/task-trained/tt_PhaseCodedMemory/",
    )

# Generate simulated datasets
dataset_path = HOME_DIR + "content/datasets/dd/"

tt_3bff.simulate_neural_data(
    subfolder="max_epochs=500 n_samples=1000 latent_size=64 seed=0 learning_rate=0.001",
    dataset_path=dataset_path,
)

mt_subfolder = "max_epochs=500 seed=0"
tt_MultiTask.simulate_neural_data(
    subfolder=mt_subfolder,
    dataset_path=dataset_path,
)

rt_subfolder = (
    "max_epochs=2000 latent_size=128 l2_wt=5e-05 "
    + "proprioception_delay=0.02 vision_delay=0.05 "
    + "n_samples=1100 n_samples=1100 seed=0 learning_rate=0.005"
)
tt_RandomTarget.simulate_neural_data(
    subfolder=rt_subfolder,
    dataset_path=dataset_path,
)

# pcm_subfolder = "max_epochs=50_learning_rate=1.00E-03_weight_decay=1.00E-08_
# latent_size=512_post_stim_l2_weight=20"
# split PCM folder name across lines
pcm_subfolder = (
    "max_epochs=50_learning_rate=1.00E-03_weight_decay=1.00E-08_"
    + "latent_size=512_post_stim_l2_weight=20"
)

tt_PCM.simulate_neural_data(
    subfolder=pcm_subfolder,
    dataset_path=dataset_path,
)
