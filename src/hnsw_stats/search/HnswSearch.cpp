#include "hnsw_stats/search/HnswSearch.hpp"

#include <algorithm>
#include <queue>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

float HnswSearch::l2_sqr_(const float* q, PointId u) const {
  return l2_sqr(q, g_.vector(u), g_.dim());
}

std::vector<std::pair<float, PointId>> HnswSearch::search(const float* q, int ef) {
  ++epoch_;
  delta_ = 0;
  visited_.clear();
  if (epoch_ == 0) {
    std::fill(stamp_.begin(), stamp_.end(), 0u);
    std::fill(added_.begin(), added_.end(), 0u);
    epoch_ = 1;
  }
  if (ef < 1) ef = 1;
  if (g_.size() == 0) return {};

  PointId cur = g_.entry_point();
  float d_cur = dist(q, cur);
  for (int lc = g_.max_level(); lc > 0; --lc) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (PointId nb : g_.neighbors_at(cur, lc)) {
        const float d = dist(q, nb);
        if (d < d_cur) { d_cur = d; cur = nb; changed = true; }
      }
    }
  }

  using DP = std::pair<float, PointId>;
  std::priority_queue<DP, std::vector<DP>, std::greater<DP>> candidates;

  std::priority_queue<DP> results;

  added_[cur] = epoch_;
  candidates.emplace(d_cur, cur);
  results.emplace(d_cur, cur);
  float worst = d_cur;

  while (!candidates.empty()) {
    const auto [cd, cnode] = candidates.top();
    if (cd > worst && static_cast<int>(results.size()) >= ef) break;
    candidates.pop();
    for (PointId nb : g_.neighbors(cnode)) {
      if (added_[nb] == epoch_) continue;
      added_[nb] = epoch_;
      const float d = dist(q, nb);
      if (static_cast<int>(results.size()) < ef || d < worst) {
        candidates.emplace(d, nb);
        results.emplace(d, nb);
        if (static_cast<int>(results.size()) > ef) results.pop();
        worst = results.top().first;
      }
    }
  }

  std::vector<DP> out;
  out.reserve(results.size());
  while (!results.empty()) { out.push_back(results.top()); results.pop(); }
  std::reverse(out.begin(), out.end());
  return out;
}

}
