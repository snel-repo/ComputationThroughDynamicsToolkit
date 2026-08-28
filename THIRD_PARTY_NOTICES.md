# Third-party software notices

The repository-level BSD-3-Clause license applies to CtDToolkit-authored code
except where a file or directory carries a different license notice. It does
not replace the licenses of the components listed below.

## `ctd/data_modeling/`

This directory contains software distributed under the MIT License:

- Copyright (c) 2021 Andrew Sedler
- License text: [`ctd/data_modeling/LICENSE`](ctd/data_modeling/LICENSE)

The MIT notice and permission text must be retained with copies or substantial
portions of that software.

## `libs/DSA`

This Git submodule points to Dynamical Similarity Analysis (DSA), distributed
under the MIT License:

- Copyright (c) 2023 Mitchell Ostrow
- Upstream: <https://github.com/mitchellostrow/DSA>
- License text: `libs/DSA/LICENSE` after the submodule is initialized

## `libs/lfads-jslds`

This Git submodule points to `lfads-jslds`, distributed under the MIT License:

- Copyright (c) 2024 David Zoltowski
- Upstream: <https://github.com/davidzoltowski/lfads-jslds>
- License text: `libs/lfads-jslds/LICENSE` after the submodule is initialized

## External dependencies

Packages installed through `requirements.txt`, including MotorNet,
`dsa-metric`, JAX, PyTorch, PyTorch Lightning, Ray, and their transitive
dependencies, remain subject to their respective upstream licenses. They are
not relicensed by CtDToolkit.
