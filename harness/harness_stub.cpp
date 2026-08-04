#include "distance/CountedDistance.hpp"

namespace hnsw_stats {

uint64_t harness_scaffold_ok() {
  CountedDistance d(Metric::L2);
  return d.count();
}
}
