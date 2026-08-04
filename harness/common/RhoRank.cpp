#include "common/RhoRank.hpp"

#include <algorithm>
#include <cstdlib>
#include <queue>
#include <thread>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

RhoMine rho_mine(const Dataset& ds, const std::vector<Query>& Q,
                 const std::vector<double>& labels, int n_labels, int K, int n_quant) {
  const size_t N = ds.N, NQ = Q.size();
  const int dim = ds.dim;
  RhoMine m;
  m.K = K; m.n_labels = n_labels; m.n_quant = n_quant; m.N = N;
  m.quant.resize(NQ);
  m.lab_gt_ids.resize(NQ * n_labels);
  m.lab_gt_dists.resize(NQ * n_labels);
  m.label_count.assign(n_labels, 0);
  for (size_t u = 0; u < N; ++u) ++m.label_count[(int)labels[u]];

  int nthr = 1;
  if (const char* t = std::getenv("HS_PREP_THREADS")) nthr = std::max(1, atoi(t));
  nthr = (int)std::min<size_t>((size_t)nthr, NQ ? NQ : 1);

  using HP = std::priority_queue<std::pair<float, uint32_t>>;
  auto worker = [&](size_t q0, size_t q1) {
    std::vector<float> dall(N);
    for (size_t q = q0; q < q1; ++q) {
      std::vector<HP> h(n_labels);
      for (size_t u = 0; u < N; ++u) {
        const float d = l2_sqr(Q[q].data, &ds.base[u * dim], dim);
        dall[u] = d;
        HP& hl = h[(int)labels[u]];
        if ((int)hl.size() < K) hl.emplace(d, (uint32_t)u);
        else if (d < hl.top().first) { hl.pop(); hl.emplace(d, (uint32_t)u); }
      }
      std::sort(dall.begin(), dall.end());
      auto& qq = m.quant[q];
      qq.resize(n_quant);
      for (int i = 0; i < n_quant; ++i)
        qq[i] = dall[(size_t)((double)i * (double)(N - 1) / (double)(n_quant - 1))];
      for (int l = 0; l < n_labels; ++l) {
        auto& ids = m.lab_gt_ids[q * n_labels + l];
        auto& dd  = m.lab_gt_dists[q * n_labels + l];
        HP& hl = h[l];
        ids.resize(hl.size()); dd.resize(hl.size());
        for (int i = (int)hl.size() - 1; i >= 0; --i) { ids[i] = hl.top().second; dd[i] = hl.top().first; hl.pop(); }
      }
    }
  };
  if (nthr <= 1) {
    worker(0, NQ);
  } else {
    std::vector<std::thread> th;
    const size_t chunk = (NQ + nthr - 1) / nthr;
    for (int t = 0; t < nthr; ++t) {
      const size_t q0 = (size_t)t * chunk, q1 = std::min(NQ, q0 + chunk);
      if (q0 < q1) th.emplace_back(worker, q0, q1);
    }
    for (auto& x : th) x.join();
  }
  return m;
}

double RhoMine::rank_frac(size_t q, float d) const {
  const auto& qq = quant[q];
  const auto it = std::upper_bound(qq.begin(), qq.end(), d);
  return (double)(it - qq.begin()) / (double)n_quant;
}

void RhoMine::rho_of(size_t q, int l, double& mean, double& mx) const {
  const auto& dd = lab_gt_dists[q * n_labels + l];
  mean = 0; mx = 0;
  if (dd.empty()) { mean = mx = 1.0; return; }
  for (float d : dd) { const double r = rank_frac(q, d); mean += r; if (r > mx) mx = r; }
  mean /= (double)dd.size();
}

}
