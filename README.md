# HISTrees (SELERA)

HISTrees is a histogram forest built inside an HNSW index; SELERA is the
filtered-ANN search engine built on it, supporting general predicates
(AND/OR/DNF, categorical, range, heterogeneous) with per-region local
selectivity. This repository accompanies the paper.

## Contents

| Path | What it is |
|---|---|
| `supplement.pdf` | Appendices A–C: proofs of Theorems 3.1 & 4.1 and the incremental-maintenance pseudo-code (Algorithm 4). Pagination continues the paper's. |
| `src/` | SELERA core: HISTrees construction, statistics, and search (C++). |
| `harness/` | Evaluation harness: distance kernels, ground truth, metrics, method drivers. |
| `baselines/` | Baseline integration: pinned upstream sources (ACORN, SeRF, NHQ, Filtered-DiskANN; fetched into `_vendor/`), one persistence patch, and thin re-implementations (post-filter, NaviX-style). See `baselines/README.md`. |
| `scripts/` | Build script and dataset preparation scripts. |
| `configs/` | Dataset descriptors, shared build parameters, baseline settings. |
| `docs/additional-results/` | Results referenced in the paper but omitted for space (see below). |

See `ARCHITECTURE.md` for the code-to-paper module mapping.

## Additional results referenced in the paper

- **Proofs** (footnote 1 of the paper): `supplement.pdf`. Read alongside the
  main paper — its numbering (Theorems 3.1/4.1, Proposition 1, Algorithms,
  references) continues the paper's.
- **arXiv-1M QPS–recall panels** (§6.2, omitted from Figure 5 for space):
  `docs/additional-results/arxiv-pareto/`.

## Build

Requires CMake ≥ 3.24, a C++20 compiler, and vcpkg (manifest mode).

    cd baselines && ./fetch.sh && ./apply_patches.sh && cd ..
    ./scripts/build.sh

## Datasets

Large raw vector files are not stored in git. `scripts/setup_data.sh` links a
local dataset directory into `data/`; `configs/datasets.toml` lists the files
each dataset entry expects. `scripts/convert_morevec.py` converts the MoReVec
corpus into that layout.
