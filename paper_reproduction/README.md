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
[`DATA_LICENSE.md`](DATA_LICENSE.md), with the complete legal text in
[`DATA_LICENSE`](DATA_LICENSE).

## Persistent identifiers

- Software archive (all versions): [doi:10.5281/zenodo.22236302](https://doi.org/10.5281/zenodo.22236302)
- Publication-data archive: [doi:10.5281/zenodo.22236312](https://doi.org/10.5281/zenodo.22236312)

Machine-readable citation metadata for the publication-data archive are in
[`CITATION.cff`](CITATION.cff).


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

## Render directly from the publication-data deposit

Install the pinned paper environment, extract the data archive, and run the
portable renderer from the software repository root:

```bash
python -m pip install -r paper_reproduction/requirements-paper.txt
python paper_reproduction/render_release_figures.py \
  --data-root /path/to/CtDToolkit-publication-data-v1.0.0 \
  --output-dir /tmp/ctdtoolkit-release-figures
```

The command verifies every entry in `SHA256SUMS.txt` before rendering and reads
no trained models or pickle caches. It renders Figure 4, Figures 5--6, S4--S5,
S8--S9, and S11--S16 from the portable CSV/JSON/NPZ files. Use `--figures` to
select a subset.

The accepted Figure 4, Figure 6, and S4 composites contain external schematic
or manually assembled artwork. The renderer therefore emits their deposited
numerical panels and labels the non-deposited panels explicitly; it does not
claim to recreate unavailable Illustrator source. Figure 3 artifacts remain
public through Git LFS, and its figure script resolves them from the public
`pretrained/` directory.

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
