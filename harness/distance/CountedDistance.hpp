#pragma once

#include <cstddef>
#include <cstdint>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

enum class Metric { L2, InnerProduct };

class CountedDistance {
 public:
  explicit CountedDistance(Metric m = Metric::L2) : metric_(m) {}

  float operator()(const float* a, const float* b, int dim) {
    ++count_;
    return raw(a, b, dim);
  }

  uint64_t count() const { return count_; }
  void reset() { count_ = 0; }
  Metric metric() const { return metric_; }

 private:

  float raw(const float* a, const float* b, int dim) const {
    return metric_ == Metric::L2 ? l2_sqr(a, b, dim)
                                 : neg_inner_product(a, b, dim);
  }

  Metric metric_;
  uint64_t count_ = 0;
};

}
