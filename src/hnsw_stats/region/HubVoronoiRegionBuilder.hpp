#pragma once

#include "hnsw_stats/region/RegionBuilder.hpp"

namespace hnsw_stats {

class HubVoronoiRegionBuilder final : public RegionBuilder {
 public:
  explicit HubVoronoiRegionBuilder(int level_threshold = 2)
      : ell_(level_threshold < 1 ? 1 : level_threshold) {}

  RegionMap build(const HnswGraph& g) override;

 private:
  int ell_;
};

}
