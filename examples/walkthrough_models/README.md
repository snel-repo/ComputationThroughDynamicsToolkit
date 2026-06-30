# Walkthrough models (git LFS)

Pretrained model files used by [`../CtDToolkit_Walkthrough.ipynb`](../CtDToolkit_Walkthrough.ipynb).
They are a **minimal curated slice** of the full `tt_3bff` results (the complete tree is ~148 GB);
only what the tutorial loads is kept here (~4.8 GB). All `.pkl` files are stored with **git LFS**.

## Getting the files

```bash
git lfs install      # one-time, installs the LFS filters
git lfs pull         # download the actual model files for this repo
```

Without `git lfs pull` you will see small text pointer files instead of the real models, and the
notebook's `Path(...).exists()` checks will pass but loading will fail.

## Layout

```
walkthrough_models/
  tt_3bff/                       # task-trained 3BFF teacher (Sections 2, 4, 7-9 reference)
    model.pkl
    datamodule_sim.pkl
    datamodule_train.pkl
    simulator.pkl
  node_sweep/                    # SAE/NODE latent-size sweep, seed=0 (Sections 6, 7)
    prefix=tt_3bff_latent_size={3,5,8,16,32,64}_max_epochs=1000_seed=0/
      model.pkl
      datamodule.pkl
  inputinf_sweep/                # LFADS input-inference sweep, seed=0 (Sections 8, 9)
    kl_co_scale={0,1.00E-02,...,1.00E-06}_..._seed=0/
      model.pkl
      datamodule.pkl
```

The sub-folder names are parsed by the notebook (`latent_size=` and `kl_co_scale=`), so do not
rename them. Each model folder is self-contained: its `datamodule.pkl` carries the simulated
spikes, so no files outside this directory are required.

## Provenance

Copied from `content/trained_models/task-trained/tt_3bff/` (the canonical NBFF task-trained model
and two of its data-driven sweeps). Only the `seed=0` members of each sweep are included.
