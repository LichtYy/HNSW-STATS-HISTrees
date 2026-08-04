# arXiv-1M: QPS–recall panels (supplement to Figure 5)

Figure 5 of the paper reports the QPS–recall predicate matrix on SIFT-1M,
GIST-1M, and Amazon; the arXiv-1M panels were omitted for space and are
provided here as `pareto-arxiv.pdf`: QPS vs. recall@10 on arXiv-1M (d=4096),
K=10, single-thread measurement, over the same seven workloads as Figure 5
(AND-2/4, OR-2/4, MIX, CAT, RANGE).

Notes:

- SELERA's plotted curve is the λ-sweep restricted to points at or above the
  default λ=1 operating point (star), pruned to its pareto frontier; the
  swept baselines vary ef / search-list size, and pre-filter is exact.
- SeRF applies to RANGE only and NHQ to CAT only — the predicate families
  their interfaces express.
- The paper's text notes ACORN as absent at d=4096; the ACORN points here
  were obtained later, by parallelizing its graph construction. Its recall
  collapse on selective predicates is consistent with the behavior reported
  on the other corpora.
