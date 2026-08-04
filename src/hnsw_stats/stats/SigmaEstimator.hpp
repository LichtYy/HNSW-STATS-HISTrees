#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

namespace hnsw_stats {

class SigmaEstimator {
 public:
  virtual ~SigmaEstimator() = default;

  virtual double sigma_hat(RegionId region, const Predicate& phi) const = 0;

};

}
