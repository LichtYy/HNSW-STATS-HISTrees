#include "metrics/MetricsCollector.hpp"

#include <algorithm>
#include <unordered_set>

namespace hnsw_stats {

double MetricsCollector::recall_at_k(const std::vector<uint32_t>& got,
                                     const std::vector<uint32_t>& gt_topk,
                                     int K, int n_satisfying) {

  const int denom = std::min(K, n_satisfying);
  if (denom <= 0) return 1.0;

  const int g = std::min<int>(denom, static_cast<int>(gt_topk.size()));
  std::unordered_set<uint32_t> gt(gt_topk.begin(), gt_topk.begin() + g);

  int hit = 0;
  const int kk = std::min<int>(K, static_cast<int>(got.size()));
  for (int i = 0; i < kk; ++i)
    if (gt.count(got[i])) ++hit;

  return static_cast<double>(hit) / static_cast<double>(denom);
}

}
