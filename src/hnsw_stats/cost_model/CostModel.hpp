#pragma once

#include "hnsw_stats/Types.hpp"

namespace hnsw_stats {

enum class region_search_kind { FLOOD, BESTFIRST };

struct CostConstants {
  double c_dist = 0.0;
  double c_pred = 0.0;
};

class CostModel {
 public:
  virtual ~CostModel() = default;

  virtual void calibrate() = 0;
  virtual CostConstants constants() const = 0;

  virtual region_search_kind decide(RegionId region, double sigma_hat,
                          size_t n_region) const = 0;
};

}
