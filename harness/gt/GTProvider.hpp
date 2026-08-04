#pragma once

#include "hnsw_stats/Types.hpp"
#include <string>
#include <vector>

namespace hnsw_stats {

struct GroundTruth {
  std::vector<PointId> ids;
  std::vector<Dist>    dists;
  int n_satisfying = 0;
};

class GTProvider {
 public:
  virtual ~GTProvider() = default;

  virtual GroundTruth ground_truth(const std::string& dataset_id, uint32_t q_id,
                                   const std::string& predicate_canonical,
                                   int K_max) = 0;
};

}
