#pragma once

#include <cstdint>
#include <vector>

namespace hnsw_stats {

using PointId  = uint32_t;
using RegionId = uint32_t;
using Dist     = float;

struct TopK {
  std::vector<PointId> ids;
  std::vector<Dist>    dists;
};

}
