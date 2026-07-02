import os
import shutil
from pathlib import Path

import dotenv

from ctd.comparison.analysis.tt.tt import Analysis_TT

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_home_dir():
    dotenv.load_dotenv(override=True)
    home_dir = os.environ.get("HOME_DIR")
    if home_dir is None:
        return REPO_ROOT
    return Path(home_dir).expanduser().resolve()


def as_analysis_path(path):
    return str(path) + os.sep


def copy_folder_contents(src_folder, dest_folder):
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    for item_name in os.listdir(src_folder):
        src_item = src_folder / item_name
        dest_item = dest_folder / item_name

        if src_item.is_dir():
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(src_item, dest_item)


HOME_DIR = get_home_dir()
print(f"Using HOME_DIR: {HOME_DIR}")

print("\n=== Phase 1: Loading pretrained task-trained models ===")
tt_3bff_path = HOME_DIR / "pretrained" / "20241017_NBFF_NoisyGRU_NewFinal"
tt_MultiTask_path = HOME_DIR / "pretrained" / "20241113_MultiTask_NoisyGRU_Final2"
tt_RandomTarget_path = HOME_DIR / "pretrained" / "20241113_RandomTarget_NoisyGRU_Final2"
tt_PhaseCodedMemory_path = HOME_DIR / "pretrained" / "PCM_NoisyGRU_Final"
tt_ChaoticDelayedMatching_path = (
    HOME_DIR / "pretrained" / "20260320_ChaoticDelayedMatching_Final"
)

tt_3bff = Analysis_TT(run_name="tt_3bff", filepath=as_analysis_path(tt_3bff_path))
tt_MultiTask = Analysis_TT(
    run_name="tt_MultiTask", filepath=as_analysis_path(tt_MultiTask_path)
)
tt_RandomTarget = Analysis_TT(
    run_name="tt_RandomTarget", filepath=as_analysis_path(tt_RandomTarget_path)
)
tt_PCM = Analysis_TT(
    run_name="tt_PhaseCodedMemory", filepath=as_analysis_path(tt_PhaseCodedMemory_path)
)
tt_CDM = Analysis_TT(
    run_name="tt_ChaoticDelayedMatching",
    filepath=as_analysis_path(tt_ChaoticDelayedMatching_path),
)

print("\n=== Phase 2: Copying pretrained models to the trained_models folder ===")
# Make copies of the pretrained models to the trained_models folder
# if the folders don't already exist
path_3bff = HOME_DIR / "content" / "trained_models" / "task-trained" / "tt_3bff"
path_MultiTask = (
    HOME_DIR / "content" / "trained_models" / "task-trained" / "tt_MultiTask"
)
path_RandomTarget = (
    HOME_DIR / "content" / "trained_models" / "task-trained" / "tt_RandomTarget"
)
path_PhaseCodedMemory = (
    HOME_DIR / "content" / "trained_models" / "task-trained" / "tt_PhaseCodedMemory"
)
path_ChaoticDelayedMatching = (
    HOME_DIR
    / "content"
    / "trained_models"
    / "task-trained"
    / "tt_ChaoticDelayedMatching"
)

if not path_3bff.exists():
    print("Copying tt_3bff pretrained model...")
    copy_folder_contents(tt_3bff_path, path_3bff)
else:
    print("tt_3bff pretrained model already present, skipping copy.")

if not path_MultiTask.exists():
    print("Copying tt_MultiTask pretrained model...")
    copy_folder_contents(tt_MultiTask_path, path_MultiTask)
else:
    print("tt_MultiTask pretrained model already present, skipping copy.")

if not path_RandomTarget.exists():
    print("Copying tt_RandomTarget pretrained model...")
    copy_folder_contents(tt_RandomTarget_path, path_RandomTarget)
else:
    print("tt_RandomTarget pretrained model already present, skipping copy.")

if not path_PhaseCodedMemory.exists():
    print("Copying tt_PhaseCodedMemory pretrained model...")
    copy_folder_contents(tt_PhaseCodedMemory_path, path_PhaseCodedMemory)
else:
    print("tt_PhaseCodedMemory pretrained model already present, skipping copy.")

if not path_ChaoticDelayedMatching.exists():
    print("Copying tt_ChaoticDelayedMatching pretrained model...")
    copy_folder_contents(tt_ChaoticDelayedMatching_path, path_ChaoticDelayedMatching)
else:
    print("tt_ChaoticDelayedMatching pretrained model already present, skipping copy.")

print("\n=== Phase 3: Simulating neural data (dd datasets) ===")
# Generate simulated datasets
dataset_path = HOME_DIR / "content" / "datasets" / "dd"

print("Simulating neural data for tt_3bff...")
tt_3bff.simulate_neural_data(
    subfolder="max_epochs=500 n_samples=1000 latent_size=64 seed=0 learning_rate=0.001",
    dataset_path=dataset_path,
)

print("Simulating neural data for tt_MultiTask...")
mt_subfolder = "max_epochs=500 seed=0"
tt_MultiTask.simulate_neural_data(
    subfolder=mt_subfolder,
    dataset_path=dataset_path,
)

print("Simulating neural data for tt_RandomTarget...")
rt_subfolder = (
    "max_epochs=2000 latent_size=128 l2_wt=5e-05 "
    + "proprioception_delay=0.02 vision_delay=0.05 "
    + "n_samples=1100 n_samples=1100 seed=0 learning_rate=0.005"
)
tt_RandomTarget.simulate_neural_data(
    subfolder=rt_subfolder,
    dataset_path=dataset_path,
)

print("Simulating neural data for tt_PhaseCodedMemory...")
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

print("Simulating neural data for tt_ChaoticDelayedMatching...")
cdm_subfolder = "max_epochs=100_recurrent_gain=2.20E+00_batch_size=256_seed=0"
tt_CDM.simulate_neural_data(
    subfolder=cdm_subfolder,
    dataset_path=dataset_path,
)

print("\n=== Done: all datasets generated ===")
