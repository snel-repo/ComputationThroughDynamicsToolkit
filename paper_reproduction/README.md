# Paper data and reproducibility release

This directory defines the compact public-data package for the accepted
CtDToolkit manuscript. It deliberately separates three things:

1. the public software repository, licensed under BSD-3-Clause;
2. the private manuscript repository, which owns the LaTeX and assembled figures;
3. a DOI-backed publication-data deposit containing the values needed to
   reproduce the quantitative figures, dedicated to the public domain under
   CC0-1.0.

The approximately 2 TB local trained-model tree and approximately 39 GB of
intermediate pickle caches are not appropriate publication artifacts. The
deposit should contain plotted values and compact derived arrays instead.
`DATA_MANIFEST.tsv` records the coverage of every main and supplementary
figure.
The scope and canonical terms of the CC0 dedication are recorded in
[`DATA_LICENSE.md`](DATA_LICENSE.md).


## Build the current portable snapshot

Only run the exporter against trusted pickle caches: loading pickle can execute
code. The resulting snapshot contains CSV, JSON, compressed NumPy files, and a
SHA-256 manifest; it contains no pickle objects or full fixed-point Jacobians.

From the repository root:

```bash
python paper_reproduction/export_release_data.py \
  --cache-root /path/to/CtDBenchmark/examples/figures \
  --tt-dir /path/to/CtDBenchmark/content/trained_models/task-trained/tt_3bff \
  --si-dir /path/to/CtDToolkit-paper/build/supporting_information \
  --output /tmp/ctdtoolkit-publication-data
```

Then add the thin exports from the large, trusted local caches:

```bash
python paper_reproduction/export_large_cache_data.py \
  --cache-root /path/to/CtDBenchmark/examples/figures \
  --output /tmp/ctdtoolkit-publication-data
```

Figure 4 requires Python 3.10+ and the versions pinned by the paper constraints:

```bash
python -m pip install -r paper_reproduction/requirements-paper.txt
python paper_reproduction/export_figure4_data.py \
  --data-root /path/to/CtDBenchmark \
  --output /tmp/ctdtoolkit-publication-data
```

Together the exporters cover Figure 4, Figures 5-6, S4-S5, S8-S9, S11-S16,
and S1-S15 Tables. Figures generated deterministically from code or from the
public Git LFS task-trained artifacts are identified separately in the manifest.

## Remaining publication steps

The quantitative snapshot is complete. A future convenience improvement would
be to let every figure script render directly from the portable deposit files.
Figure 3's task-trained artifacts are already public through Git LFS, but its
figure script now resolves them from the public `pretrained/` directory.

## Release gates

Before making anything public:

- run all three exporters in order and verify `SHA256SUMS.txt`;
- verify that the snapshot contains no absolute local paths and that the
  manifest contains no `pending_thin_export` rows;
- test the documented figure commands from a clean clone with Git LFS and both
  submodules initialized;
- create a versioned software release and archive it with a DOI;
- deposit the portable data snapshot separately, then add both DOI links to the
  manuscript Data Availability statement and repository README.

Do not include `.env` files, absolute local paths, scratch outputs, raw
multi-terabyte training trees, or untrusted pickle caches in either release.
