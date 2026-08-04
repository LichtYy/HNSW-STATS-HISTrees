#include "post_filter/PostFilter.hpp"

#include <fstream>

#include <chrono>

namespace hnsw_stats {

void PostFilter::build(const Dataset& ds, const BuildParams& bp) {
  const auto t0 = std::chrono::steady_clock::now();
  dim_ = ds.dim; N_ = ds.N;

  bool loaded = false;
  if (!index_cache_.empty()) { std::ifstream cf(index_cache_, std::ios::binary);
    if (cf.good()) { index_ = std::make_unique<HnswIndex>(index_cache_, ds.dim); loaded = true; } }
  if (!loaded) index_ = std::make_unique<HnswIndex>(ds.base.data(), ds.N, ds.dim, bp.M, bp.efc,
                                                    1);
  if (!loaded && !index_cache_.empty()) index_->save(index_cache_);
  searcher_ = std::make_unique<HnswSearch>(*index_);
  build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

SearchResult PostFilter::search(const Query& q, const Predicate& phi, int ef) {

  const auto cand = searcher_->search(q.data, ef);
  SearchResult r;
  r.n_delta = searcher_->delta();
  r.n_pred_evals = cand.size();
  for (const auto& [d, id] : cand) {
    if (phi.eval(id)) {
      ++r.n_delta_satisfying;
      if (static_cast<int>(r.topk_ids.size()) < K_) r.topk_ids.push_back(id);
    }
  }
  return r;
}

}
