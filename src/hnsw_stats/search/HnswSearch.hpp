#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/hnsw/HnswGraph.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace hnsw_stats {

class HnswSearch {
 public:
  explicit HnswSearch(const HnswGraph& g)
      : g_(g), stamp_(g.size(), 0), cache_(g.size(), 0.f), added_(g.size(), 0) {}

  std::vector<std::pair<float, PointId>> search(const float* q, int ef);

  uint64_t delta() const { return delta_; }

  const std::vector<PointId>& visited() const { return visited_; }

 private:

  float dist(const float* q, PointId u) {
    if (stamp_[u] == epoch_) return cache_[u];
    stamp_[u] = epoch_;
    const float d = l2_sqr_(q, u);
    cache_[u] = d;
    ++delta_;
    visited_.push_back(u);
    return d;
  }
  float l2_sqr_(const float* q, PointId u) const;

  const HnswGraph& g_;
  std::vector<uint32_t> stamp_;
  std::vector<float> cache_;
  std::vector<uint32_t> added_;
  std::vector<PointId> visited_;
  uint32_t epoch_ = 0;
  uint64_t delta_ = 0;
};

}
