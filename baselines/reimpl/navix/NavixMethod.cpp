#include "navix/NavixMethod.hpp"

#include <algorithm>
#include <chrono>
#include <queue>
#include <vector>

#include "hnsw_stats/distance/Distance.hpp"

#include <fstream>

namespace hnsw_stats {

void NavixMethod::build(const Dataset& ds, const BuildParams& bp) {
  using Clock = std::chrono::steady_clock;
  dim_ = ds.dim; N_ = ds.N;
  const auto t0 = Clock::now();
  bool loaded = false;
  if (!index_cache_.empty()) { std::ifstream cf(index_cache_, std::ios::binary);
    if (cf.good()) { base_ = std::make_unique<HnswIndex>(index_cache_, dim_); loaded = true; } }
  if (!loaded) base_ = std::make_unique<HnswIndex>(ds.base.data(), N_, dim_, bp.M, bp.efc, 1);
  if (!loaded && !index_cache_.empty()) base_->save(index_cache_);
  deg_ = 2 * (bp.M > 0 ? bp.M : 16);
  mask_.assign(N_, 0); stamp_.assign(N_, 0); added_.assign(N_, 0); svis_.assign(N_, 0);
  cache_.assign(N_, 0.f);
  build_seconds_ = std::chrono::duration<double>(Clock::now() - t0).count();
}

float NavixMethod::dist(const float* q, PointId u) {
  if (stamp_[u] == epoch_) return cache_[u];
  stamp_[u] = epoch_;
  const float d = l2_sqr(q, base_->vector(u), dim_);
  cache_[u] = d; ++delta_;
  return d;
}

SearchResult NavixMethod::search(const Query& q, const Predicate& phi, int ef) {
  ++epoch_; delta_ = 0; delta_sat_ = 0; struct_ = 0; pred_ = 0;
  n_onehop_ = n_directed_ = n_blind_ = 0;
  if (epoch_ == 0) { std::fill(stamp_.begin(), stamp_.end(), 0u);
                     std::fill(added_.begin(), added_.end(), 0u);
                     std::fill(svis_.begin(), svis_.end(), 0u); epoch_ = 1; }
  if (ef < K_) ef = K_;
  const float* qv = q.data;

  size_t m = 0;
  for (size_t u = 0; u < N_; ++u) { ++pred_; const bool s = phi.eval((PointId)u); mask_[u] = s ? 1 : 0; m += s; }
  SearchResult r;
  r.n_pred_evals = pred_;
  if (m == 0) return r;

  const double M = (double)deg_;
  using DP = std::pair<float, PointId>;
  std::priority_queue<DP, std::vector<DP>, std::greater<DP>> cand;
  std::priority_queue<DP> res;
  float worst = 1e30f;
  auto offer = [&](PointId v) {
    if (added_[v] == epoch_ || !mask_[v]) return;
    added_[v] = epoch_;
    const float d = dist(qv, v); ++delta_sat_;
    if ((int)res.size() < ef || d < worst) {
      cand.emplace(d, v); res.emplace(d, v);
      if ((int)res.size() > ef) res.pop();
      worst = res.top().first;
    }
  };

  auto expand = [&](PointId c) {
    const std::span<const PointId> nb = base_->neighbors(c);
    const size_t deg = nb.size(); if (deg == 0) return;
    size_t sel1 = 0; for (PointId v : nb) { ++struct_; if (mask_[v]) ++sel1; }
    const double sigma_l = (double)sel1 / (double)deg;
    const double esv = sigma_l * (M + 1.0) * M;
    if (sigma_l >= ub_onehop_s_) {

      ++n_onehop_;
      for (PointId v : nb) if (mask_[v]) offer(v);
    } else if (esv >= M * leniency_) {

      ++n_directed_;
      std::vector<DP> ranked; ranked.reserve(deg);
      for (PointId v : nb) { const float d = dist(qv, v); ranked.emplace_back(d, v); if (mask_[v]) offer(v); }
      std::sort(ranked.begin(), ranked.end(), [](const DP& a, const DP& b){ return a.first < b.first; });
      size_t found = 0;
      for (const auto& [d, v] : ranked) {
        if (found >= deg) break;
        for (PointId w : base_->neighbors(v)) { ++struct_;
          if (mask_[w]) { offer(w); ++found; } }
      }
    } else {

      ++n_blind_;
      std::queue<PointId> bq;
      size_t found = 0;
      for (PointId v : nb) {
        if (mask_[v]) { offer(v); ++found; }
        else if (svis_[v] != epoch_) { svis_[v] = epoch_; bq.push(v); }
      }
      while (!bq.empty() && found < deg) {
        const PointId u = bq.front(); bq.pop();
        for (PointId w : base_->neighbors(u)) { ++struct_;
          if (mask_[w]) { offer(w); ++found; }
          else if (svis_[w] != epoch_) { svis_[w] = epoch_; bq.push(w); } }
      }
    }
  };

  PointId cur = base_->entry_point();
  float d_cur = dist(qv, cur);
  for (int lc = base_->max_level(); lc > 0; --lc) {
    bool changed = true;
    while (changed) { changed = false;
      for (PointId nb : base_->neighbors_at(cur, lc)) {
        const float d = dist(qv, nb);
        if (d < d_cur) { d_cur = d; cur = nb; changed = true; }
      } }
  }
  if (mask_[cur]) offer(cur);
  expand(cur);
  while (!cand.empty()) {
    const auto [cd, c] = cand.top(); cand.pop();
    if (cd > worst && (int)res.size() >= ef) break;
    expand(c);
  }

  r.n_delta = delta_; r.n_delta_satisfying = delta_sat_; r.n_struct_visits = struct_; r.n_pred_evals = pred_;
  std::vector<DP> out;
  while (!res.empty()) { out.push_back(res.top()); res.pop(); }
  std::reverse(out.begin(), out.end());
  if ((int)out.size() > K_) out.resize(K_);
  for (auto& [d, id] : out) r.topk_ids.push_back(id);
  return r;
}

}
