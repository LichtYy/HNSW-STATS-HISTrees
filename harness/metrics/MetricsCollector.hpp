#pragma once

#include <cstdint>
#include <vector>

namespace hnsw_stats {

class MetricsCollector {
 public:

  static double recall_at_k(const std::vector<uint32_t>& got,
                            const std::vector<uint32_t>& gt_topk,
                            int K, int n_satisfying);

};

}
