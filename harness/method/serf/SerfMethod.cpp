#include "method/serf/SerfMethod.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <unistd.h>
#include <vector>
using namespace std;
#include "segment_graph_2d.h"
#include "data_wrapper.h"

namespace hnsw_stats {

struct SerfMethod::Impl {
  std::unique_ptr<DataWrapper> dw;
  std::unique_ptr<base_hnsw::L2Space> sp;
  std::unique_ptr<SeRF::IndexSegmentGraph2D> idx;
  BaseIndex::IndexParams iparams;

  std::vector<int> ord, inv;
  int N = 0, dim = 0;
};

SerfMethod::SerfMethod(int K, int range_col, int serf_M, unsigned serf_efc)
    : impl_(new Impl), K_(K), range_col_(range_col), serf_M_(serf_M), serf_efc_(serf_efc) {}
SerfMethod::~SerfMethod() = default;

void SerfMethod::build(const Dataset& ds, const BuildParams& bp) {
  Impl& I = *impl_;
  I.N = (int)ds.N;
  I.dim = ds.dim;

  const std::vector<double>& vals = ds.columns[(size_t)range_col_].values;
  I.ord.resize(I.N);
  std::iota(I.ord.begin(), I.ord.end(), 0);
  std::sort(I.ord.begin(), I.ord.end(), [&](int a, int b) { return vals[a] < vals[b]; });
  I.inv.assign(I.N, 0);
  for (int r = 0; r < I.N; ++r) I.inv[I.ord[r]] = r;

  I.dw.reset(new DataWrapper(1, K_, "hs", I.N));
  I.dw->data_dim = (size_t)I.dim;
  I.dw->is_even_weight = false;
  I.dw->real_keys = false;
  I.dw->nodes.resize(I.N);
  I.dw->nodes_keys.resize(I.N);
  for (int r = 0; r < I.N; ++r) {
    int u = I.ord[r];
    I.dw->nodes[r].assign(ds.base.begin() + (size_t)u * I.dim, ds.base.begin() + (size_t)(u + 1) * I.dim);
    I.dw->nodes_keys[r] = r;
  }

  I.sp.reset(new base_hnsw::L2Space((size_t)I.dim));
  I.idx.reset(new SeRF::IndexSegmentGraph2D(I.sp.get(), I.dw.get()));
  I.iparams = BaseIndex::IndexParams();
  I.iparams.K = (unsigned)std::max(bp.M, serf_M_);
  I.iparams.ef_construction = std::max<unsigned>((unsigned)bp.efc, serf_efc_);
  I.iparams.ef_large_for_pruning = 0;
  I.iparams.ef_max = 500;
  I.iparams.recursion_type = BaseIndex::IndexParams::MAX_POS;

  auto t0 = std::chrono::steady_clock::now();
  bool loaded = false;
  if (!index_cache_.empty()) {
    std::ifstream f(index_cache_, std::ios::binary);
    if (f.good()) { f.close();
      try { I.idx->load(index_cache_); loaded = true; std::fprintf(stderr, "[serf] index cache HIT %s\n", index_cache_.c_str()); }
      catch (...) { loaded = false; std::fprintf(stderr, "[serf] cache load failed, rebuilding\n"); }
    }
  }
  if (!loaded) {
    I.idx->buildIndex(&I.iparams);
    if (!index_cache_.empty()) {
      std::string tmp = index_cache_ + ".tmp." + std::to_string((long)getpid());
      try { I.idx->write(tmp); std::rename(tmp.c_str(), index_cache_.c_str()); std::fprintf(stderr, "[serf] index cached -> %s\n", index_cache_.c_str()); }
      catch (...) { std::fprintf(stderr, "[serf] cache write failed (non-fatal)\n"); }
    }
  }
  build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

SearchResult SerfMethod::search(const Query& q, const Predicate& phi, int ef) {
  Impl& I = *impl_;
  SearchResult r;
  if (!I.idx) return r;

  int lo = I.N, hi = -1;
  for (int u = 0; u < I.N; ++u)
    if (phi.eval((PointId)u)) { int k = I.inv[u]; if (k < lo) lo = k; if (k > hi) hi = k; }
  if (hi < lo) return r;

  BaseIndex::SearchParams sp;
  sp.query_K = (unsigned)K_;
  sp.search_ef = (unsigned)std::max(ef, K_);
  sp.query_range = (unsigned)(hi - lo + 1);
  BaseIndex::SearchInfo si(I.dw.get(), &I.iparams, "SeRF", "hs");
  std::vector<float> qv(q.data, q.data + I.dim);
  std::vector<int> res = I.idx->rangeFilteringSearchInRange(&sp, &si, qv, std::make_pair(lo, hi));

  std::reverse(res.begin(), res.end());
  r.topk_ids.reserve(res.size());
  for (int rk : res) r.topk_ids.push_back((uint32_t)I.ord[rk]);
  r.n_delta = si.total_comparison;
  r.n_pred_evals = 0;
  r.n_struct_visits = si.total_comparison;
  return r;
}

}
