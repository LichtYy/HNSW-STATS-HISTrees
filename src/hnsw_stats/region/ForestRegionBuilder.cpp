#include "hnsw_stats/region/ForestRegionBuilder.hpp"

#include <limits>
#include <vector>

namespace hnsw_stats {

namespace {
constexpr RegionId kUnassigned = std::numeric_limits<RegionId>::max();
}

RegionMap ForestRegionBuilder::build(const HnswGraph& g) {
  const size_t N = g.size();
  RegionMap rm;
  rm.region_of.assign(N, kUnassigned);
  rm.parent_of.assign(N, 0);
  if (N == 0) { rm.n_regions = 0; return rm; }

  for (size_t p = 0; p < N; ++p)
    rm.parent_of[p] = g.father(static_cast<PointId>(p));

  RegionId n_regions = 0;
  for (size_t u = 0; u < N; ++u)
    if (g.node_level(static_cast<PointId>(u)) >= ell_)
      rm.region_of[u] = n_regions++;

  std::vector<PointId> path;
  for (size_t u = 0; u < N; ++u) {
    if (rm.region_of[u] != kUnassigned) continue;
    path.clear();
    PointId a = static_cast<PointId>(u);
    while (rm.region_of[a] == kUnassigned) {
      const PointId f = g.father(a);
      if (f == a) {
        rm.region_of[a] = n_regions++;
        break;
      }
      path.push_back(a);
      a = f;
    }
    const RegionId rid = rm.region_of[a];
    for (PointId x : path) rm.region_of[x] = rid;
  }

  rm.n_regions = n_regions;

  rm.multilayer = false;
  rm.lmat_max_nodes.resize(n_regions);
  for (RegionId r = 0; r < n_regions; ++r) rm.lmat_max_nodes[r] = r;
  rm.lmat_max_n.assign(n_regions, 0);
  for (size_t u = 0; u < N; ++u) rm.lmat_max_n[rm.region_of[u]]++;
  return rm;
}

}
