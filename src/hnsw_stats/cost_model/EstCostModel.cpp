#include "hnsw_stats/cost_model/EstCostModel.hpp"

#include <algorithm>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>

#include "hnsw_stats/distance/Distance.hpp"
#include "hnsw_stats/search/HnswSearch.hpp"

namespace hnsw_stats {

void EstCostModel::calibrate(const HnswGraph& g,
                             const std::vector<RegionId>& region_of,
                             const Predicate& phi, int dist_samples,
                             int bf_searches, int bf_ef, uint64_t seed) {
  using Clock = std::chrono::steady_clock;
  const size_t N = g.size();
  const int dim = g.dim();
  if (N == 0) return;

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, N - 1);

  std::vector<PointId> as(dist_samples), bs(dist_samples);
  for (int i = 0; i < dist_samples; ++i) { as[i]=(PointId)pick(rng); bs[i]=(PointId)pick(rng); }
  volatile float sink = 0.f;
  auto t0 = Clock::now();
  for (int i = 0; i < dist_samples; ++i) sink += l2_sqr(g.vector(as[i]), g.vector(bs[i]), dim);
  const double dist_secs = std::chrono::duration<double>(Clock::now() - t0).count();

  volatile int psink = 0;
  t0 = Clock::now();
  for (int i = 0; i < dist_samples; ++i) psink += phi.eval(as[i]) ? 1 : 0;
  const double pred_secs = std::chrono::duration<double>(Clock::now() - t0).count();

  std::vector<uint32_t> escratch(N, 1u);
  volatile uint64_t esink = 0;
  size_t edge_count = 0;
  t0 = Clock::now();
  for (int i = 0; i < dist_samples; ++i) {
    const PointId u = (PointId)pick(rng);
    for (PointId e : g.neighbors(u)) { esink += escratch[e]; ++edge_count; }
  }
  const double edge_secs = std::chrono::duration<double>(Clock::now() - t0).count();
  (void)sink; (void)psink; (void)esink;
  c_dist_ = dist_secs / dist_samples;
  c_pred_ = pred_secs / dist_samples;
  c_edge_ = edge_count > 0 ? edge_secs / double(edge_count) : 0.0;

  HnswSearch bf(g);
  double total_visited = 0, total_region_touches = 0;
  std::unordered_set<RegionId> regions;
  for (int s = 0; s < bf_searches; ++s) {
    const float* q = g.vector((PointId)pick(rng));
    (void)bf.search(q, bf_ef);
    const auto& vis = bf.visited();
    regions.clear();
    for (PointId u : vis) regions.insert(region_of[u]);
    total_visited += double(vis.size());
    total_region_touches += double(regions.size());
  }
  m_bar_ = total_region_touches > 0 ? total_visited / total_region_touches : double(bf_ef);
}

}
