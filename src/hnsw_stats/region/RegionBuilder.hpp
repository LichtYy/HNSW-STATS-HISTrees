#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/hnsw/HnswGraph.hpp"
#include <vector>

namespace hnsw_stats {

struct RegionMap {
  std::vector<RegionId> region_of;
  std::vector<PointId>  parent_of;
  RegionId n_regions = 0;

  bool multilayer = false;
  std::vector<RegionId> lmat_max_nodes;
  std::vector<uint32_t> lmat_max_n;

  struct LmatTreeMeta { uint32_t budget_local = 0; uint32_t cnt_mod = 0; uint32_t cnt_del = 0; };
  std::vector<LmatTreeMeta> lmat_trees;
};

class RegionBuilder {
 public:
  virtual ~RegionBuilder() = default;
  virtual RegionMap build(const HnswGraph& graph) = 0;
};

}
