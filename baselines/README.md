# baselines/

Baseline methods used in the paper's evaluation. All methods expose the
unified `Method` interface, with every distance computation routed through
`CountedDistance` so distance counts are comparable across methods.

## Setup

    ./fetch.sh           # clone pinned upstream sources into _vendor/
    ./apply_patches.sh   # apply the persistence patch to SeRF

The ACORN and Filtered-DiskANN adapters additionally expect `libfaiss.a` /
`libdiskann.a` to have been built inside the corresponding `_vendor` trees
(a standard CMake build in `<tree>/build`; `patches/acorn/build_glue/`
provides the CMake configuration used for ACORN's FAISS). The top-level
`scripts/build.sh` then compiles the adapters against the `_vendor/` trees
referenced from `harness/CMakeLists.txt`; adapters whose prebuilt library is
absent are skipped.

## Vendored originals (`_vendor/`, pinned in `manifest.toml`)

- **ACORN** — the original FAISS-based implementation bundled in FANNBench.
  Upstream algorithm source is unmodified; `patches/acorn/build_glue/`
  contains only CMake glue to build `libfaiss.a` for the adapter.
- **SeRF** — bundled in FANNBench. One additive patch
  (`patches/serf/serf_add_load.patch`) adds a `load()` symmetric to the
  existing `write()` so the index can be built once and reloaded; search
  behavior is unchanged.
- **NHQ** (`NHQ-NPG_kgraph`) — integrated with zero source modification.
  Requires OpenBLAS (`apt install libopenblas-dev`).
- **Filtered-DiskANN** — bundled in FANNBench, source unmodified.

## `reimpl/`

Thin re-implementations on the shared HNSW core: post-filtering and the
NaviX-style strategy.
