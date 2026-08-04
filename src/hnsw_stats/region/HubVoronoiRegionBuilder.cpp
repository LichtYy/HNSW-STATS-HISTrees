#include "hnsw_stats/region/HubVoronoiRegionBuilder.hpp"

#include <limits>
#include <queue>
#include <vector>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

namespace {
constexpr RegionId kUnassigned = std::numeric_limits<RegionId>::max();

struct QEntry {
  float key;
  PointId u;
  RegionId region;
  PointId parent;
};
struct ByKey {
  bool operator()(const QEntry& a, const QEntry& b) const { return a.key > b.key; }
};
}

RegionMap HubVoronoiRegionBuilder::build(const HnswGraph& g) {
  const size_t N = g.size();
  const int dim = g.dim();
  RegionMap rm;
  rm.region_of.assign(N, kUnassigned);
  rm.parent_of.assign(N, 0);
  if (N == 0) { rm.n_regions = 0; return rm; }

  std::vector<PointId> region_anchor;
  for (size_t u = 0; u < N; ++u)
    if (g.node_level(static_cast<PointId>(u)) >= ell_)
      region_anchor.push_back(static_cast<PointId>(u));
  if (region_anchor.empty()) region_anchor.push_back(g.entry_point());

  auto anchor_vec = [&](RegionId r) { return g.vector(region_anchor[r]); };
  auto key_to_anchor = [&](PointId u, RegionId r) {
    return l2_sqr(g.vector(u), anchor_vec(r), dim);
  };

  std::vector<float> best(N, std::numeric_limits<float>::max());
  std::vector<char> settled(N, 0);
  std::priority_queue<QEntry, std::vector<QEntry>, ByKey> pq;

  for (RegionId r = 0; r < region_anchor.size(); ++r) {
    const PointId a = region_anchor[r];
    best[a] = 0.f;
    pq.push({0.f, a, r, a});
  }

  while (!pq.empty()) {
    const QEntry e = pq.top();
    pq.pop();
    if (settled[e.u]) continue;
    settled[e.u] = 1;
    rm.region_of[e.u] = e.region;
    rm.parent_of[e.u] = e.parent;
    for (PointId v : g.neighbors(e.u)) {
      if (settled[v]) continue;
      const float k = key_to_anchor(v, e.region);
      if (k < best[v]) {
        best[v] = k;
        pq.push({k, v, e.region, e.u});
      }
    }
  }

  rm.n_regions = static_cast<RegionId>(region_anchor.size());

  for (size_t u = 0; u < N; ++u) {
    if (rm.region_of[u] == kUnassigned) {
      rm.region_of[u] = rm.n_regions++;
      rm.parent_of[u] = static_cast<PointId>(u);
    }
  }

  rm.multilayer = false;
  rm.lmat_max_nodes.resize(rm.n_regions);
  for (RegionId r = 0; r < rm.n_regions; ++r) rm.lmat_max_nodes[r] = r;
  rm.lmat_max_n.assign(rm.n_regions, 0);
  for (size_t u = 0; u < N; ++u) rm.lmat_max_n[rm.region_of[u]]++;
  return rm;
}

}
