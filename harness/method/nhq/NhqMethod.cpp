#include "method/nhq/NhqMethod.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "efanna2e/index_random.h"
#include "efanna2e/index_graph.h"
#include "efanna2e/util.h"

namespace hnsw_stats {

struct NhqMethod::Impl {
  std::unique_ptr<efanna2e::IndexRandom> init;
  std::unique_ptr<efanna2e::IndexGraph> index;
  float* aligned = nullptr;
  unsigned adim = 0;
  int N = 0, dim = 0;
  std::vector<double> catvals;
  ~Impl() { free(aligned); }
};

NhqMethod::NhqMethod(int K, int cat_col, float weight_search)
    : impl_(new Impl), K_(K), cat_col_(cat_col), weight_search_(weight_search) {}
NhqMethod::~NhqMethod() = default;

void NhqMethod::build(const Dataset& ds, const BuildParams& bp) {
  Impl& I = *impl_;
  I.N = (int)ds.N;
  I.dim = ds.dim;
  I.catvals = ds.columns[(size_t)cat_col_].values;

  float* raw = (float*)malloc((size_t)I.N * I.dim * sizeof(float));
  std::memcpy(raw, ds.base.data(), (size_t)I.N * I.dim * sizeof(float));
  I.adim = (unsigned)I.dim;
  I.aligned = efanna2e::data_align(raw, (unsigned)I.N, I.adim);

  I.init.reset(new efanna2e::IndexRandom(I.adim, I.N));
  I.index.reset(new efanna2e::IndexGraph(I.adim, I.N, efanna2e::L2, I.init.get()));

  efanna2e::Parameters p;
  p.Set<unsigned>("K", 100); p.Set<unsigned>("L", 100); p.Set<unsigned>("iter", 12);
  p.Set<unsigned>("S", 10);  p.Set<unsigned>("R", 300); p.Set<unsigned>("RANGE", 64);
  p.Set<unsigned>("PL", 100); p.Set<float>("B", 0.4f);  p.Set<float>("M", 1.0f);

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < I.N; ++i)
    I.index->AddAllNodeAttributes({std::to_string((long long)I.catvals[i])});
  I.index->Build(I.N, I.aligned, p);
  I.index->OptimizeGraph(I.aligned);
  build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

SearchResult NhqMethod::search(const Query& q, const Predicate& phi, int ef) {
  Impl& I = *impl_;
  SearchResult r;
  if (!I.index) return r;

  long long cat = 0; bool found = false;
  for (int u = 0; u < I.N; ++u) if (phi.eval((PointId)u)) { cat = (long long)I.catvals[u]; found = true; break; }
  if (!found) return r;

  std::vector<float> qv(I.adim, 0.f);
  for (int d = 0; d < I.dim; ++d) qv[d] = q.data[d];
  efanna2e::Parameters sp;

  sp.Set<unsigned>("L_search", (unsigned)std::min(std::max(ef, K_), I.N - 1));
  sp.Set<float>("weight_search", weight_search_);
  std::vector<unsigned> res((size_t)K_, 0);
  I.index->SearchWithOptGraph({std::to_string(cat)}, qv.data(), (size_t)K_, sp, res.data());

  r.topk_ids.assign(res.begin(), res.end());
  r.n_delta = (uint64_t)I.index->GetDistCount();
  r.n_pred_evals = 0;
  r.n_struct_visits = r.n_delta;
  return r;
}

size_t NhqMethod::persistent_bytes() {

  return 0;
}

}
