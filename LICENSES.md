# Licensing

CtDToolkit uses separate licenses for software and data so that each can be
reused under terms appropriate to its form.

## Project software and documentation

Unless a file or directory contains a more specific license notice, source
code, configuration files, scripts, and documentation authored for
CtDToolkit are available under the BSD 3-Clause License in [`LICENSE`](LICENSE)
(SPDX identifier: `BSD-3-Clause`).

## Project data and model artifacts

Project-authored datasets and data/model artifacts distributed in
`pretrained/`, `examples/walkthrough_models/`, or a CtDToolkit publication
data release are dedicated to the public domain under CC0 1.0 Universal in
[`DATA_LICENSE`](DATA_LICENSE) (SPDX identifier: `CC0-1.0`). This designation
does not change the license of source code serialized with or required to read
an artifact.

## Third-party material

Files and directories with their own license notices retain those licenses.
In particular:

- `ctd/data_modeling/` contains material under the MIT License identified in
  `ctd/data_modeling/LICENSE`.
- The `libs/DSA` submodule is distributed under its own MIT License.
- The `libs/lfads-jslds` submodule is distributed under its own MIT License.
- External dependencies installed through `requirements.txt` retain their
  respective licenses. Detailed notices and upstream links are provided in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Manuscript files that remain accessible in the repository's earlier Git
history are not part of the CtDToolkit software or data release and are not
licensed by the root `LICENSE` or `DATA_LICENSE` files.
