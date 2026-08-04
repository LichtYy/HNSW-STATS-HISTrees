# Architecture

This file maps the repository layout to the components described in the
accompanying paper. Proofs and the incremental-maintenance pseudo-code are in
`supplement.pdf`.

## Module dependency (compile-time)

```
          vcpkg deps (hnswlib, toml++, nlohmann-json, Catch2)
                        |
        +---------------+----------------+
        |                                |
  hnsw_stats_core (src/hnsw_stats)   hnsw_stats_harness
   hnsw   predicate   region         (harness/{common,distance,metrics,
   stats  cost_model  search          gt,method}) — links core
        \_______________________________ |
                                        \|
                          hnsw_stats_tests (tests/, Catch2)
```

## Code → paper mapping

- `src/hnsw_stats/region`, `src/hnsw_stats/stats` — HISTrees construction:
  region partitioning, per-column histogram forests, the discard test, and
  presence bitmaps (paper §3); incremental maintenance (§5; Algorithm 4 in
  `supplement.pdf`).
- `src/hnsw_stats/cost_model`, `src/hnsw_stats/search` — Selera query
  processing: per-region admission and cost-model-driven traversal (§4).
- `src/hnsw_stats/hnsw`, `src/hnsw_stats/predicate` — HNSW graph core and
  predicate (DNF-atom) evaluation.
- `harness/distance/CountedDistance` — unified distance-computation counting
  for cross-method comparability.
- `harness/gt` — brute-force filtered ground truth, with caching.
- `harness/method` — method drivers: Selera (`OursMethod`) and baseline
  adapters (`acorn_orig`, `serf`, `nhq`, `filtered_diskann`); baseline sources
  are fetched by `baselines/fetch.sh` and patched by
  `baselines/apply_patches.sh`.
- `configs/` — dataset descriptors, shared HNSW build parameters (M, efC),
  and baseline settings. Cross-method comparisons use the same distance
  kernels and threading configuration.

## Interface headers (one class per file)

- core: `hnsw/HnswGraph`, `predicate/Predicate`, `region/RegionBuilder`,
  `stats/SigmaEstimator`, `cost_model/CostModel`, `search/Searcher`.
- harness: `method/Method`, `method/SchemaEvolvable`,
  `distance/CountedDistance`, `metrics/MetricsCollector`, `gt/GTProvider`,
  `common/DatasetLoader`, `common/QueryPredicateGen`.
