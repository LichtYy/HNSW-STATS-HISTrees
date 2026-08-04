#include "hnsw_stats/search/OursSearch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

float OursSearch::l2_sqr_(const float* q, PointId u) const {
  return l2_sqr(q, g_.vector(u), g_.dim());
}

void OursSearch::consider(SearchCtx& ctx, PointId id, float d) {
  if (considered_[id] == epoch_) return; considered_[id] = epoch_; ++delta_sat_;
  if (b_trace_) { b_disc_pos_[id] = (uint32_t)cpred_broad_; b_disc_stamp_[id] = epoch_; b_disc_flood_[id] = b_inflood_ ? 1 : 0; }
  if ((int)ctx.results.size() < ctx.ef) { ctx.results.emplace(d, id); ctx.worst = ctx.results.top().first; }
  else if (d < ctx.worst) { ctx.results.emplace(d, id); ctx.results.pop(); ctx.worst = ctx.results.top().first; }

  if (worsttl_on_ && (int)ctx.results.size() >= ctx.ef) worsttl_.emplace_back(delta_, ctx.worst);
}
bool OursSearch::peval(SearchCtx& ctx, PointId u) {
  if (pstamp_[u] == epoch_) return psat_[u];

  if (transit_skip_) {
    const RegionId ru = rm_.region_of[u];
    if (rc_stamp_[ru] == epoch_ && rc_seldnf_[ru] <= 0.0) { pstamp_[u] = epoch_; psat_[u] = 0; return false; }
  }
  pstamp_[u] = epoch_; const bool s = ctx.phi.eval(u); psat_[u] = s ? 1 : 0; ++pred_evals_;
  if (attr_per_region_) per_region_pred_[rm_.region_of[u]] += 1;
  if (rtouch_trace_on_) { const RegionId ru = rm_.region_of[u]; rtouch_touch_(ru); ++rtouch_rpred_[ru]; }
  return s;
}

std::vector<PointId> OursSearch::search(const float* q, const ColumnPredicate& phi,
                                        int ef, int K) {
  return run(q, phi,
             [&](RegionId r) { return cm_.decide(stats_.sigma_hat(r, phi),
                                                  stats_.region_size(r)); },
             ef, K, BoundaryMode::GATED_PROXY);
}

std::vector<PointId> OursSearch::search_boundary(const float* q, const ColumnPredicate& phi,
                                                 BoundaryMode mode, int ef, int K,
                                                 double sigma_gate, bool force_broad,
                                                 bool force_bestfirst) {
  return run(q, phi,
             [&](RegionId r) {
               if (force_broad) return region_search_kind::FLOOD;
               if (force_bestfirst) return region_search_kind::BESTFIRST;
               return cm_.decide(stats_.sigma_hat(r, phi), stats_.region_size(r));
             },
             ef, K, mode, sigma_gate);
}

std::vector<PointId> OursSearch::search_with_strategy(
    const float* q, const Predicate& phi,
    const std::vector<region_search_kind>& region_strategy, int ef, int K, bool lazy) {
  return run(q, phi, [&](RegionId r) { return region_strategy[r]; }, ef, K,
             lazy ? BoundaryMode::LAZY_PROXY : BoundaryMode::EAGER);
}

std::vector<PointId> OursSearch::run(const float* q, const Predicate& phi,
                                     const std::function<region_search_kind(RegionId)>& decide,
                                     int ef, int K, BoundaryMode mode, double sigma_gate) {
  ++epoch_;
  if (++vepoch8_ == 0) { for (auto& p : pack_) p.vis = 0; vepoch8_ = 1; }
  delta_ = 0; delta_sat_ = 0; expanded_ = 0; broad_regions_ = 0; boundary_points_ = 0;
  expanded_verify_ = 0; cc_sel0_b_find_ = 0; cc_sel0_b_verify_ = 0; cc_sel0_c_verify_ = 0;
  pred_evals_ = 0; struct_visits_ = 0; uniq_proxy_pushes_ = 0; last_result_worst_ = -1.f;
  for (int i = 0; i < 64; ++i) chain_hist_[i] = 0;
  if (snap_) { snap_snaps_.clear(); snap_heap_.clear(); snap_satj_ = 0; snap_jidx_ = 0; }
  if (epoch_ == 0) {
    std::fill(stamp_.begin(), stamp_.end(), 0u);
    for (auto& p : pack_) p.vis = 0;
    std::fill(considered_.begin(), considered_.end(), 0u);
    std::fill(svis_.begin(), svis_.end(), 0u);
    std::fill(pstamp_.begin(), pstamp_.end(), 0u);
    std::fill(chain_stamp_.begin(), chain_stamp_.end(), 0u);
    std::fill(ppush_stamp_.begin(), ppush_stamp_.end(), 0u);
    epoch_ = 1;
  }
  if (ef < K) ef = K;
  if (g_.size() == 0) return {};

  const ColumnPredicate* cpgate =
      (sigma_gate >= 0.0) ? dynamic_cast<const ColumnPredicate*>(&phi) : nullptr;

  using DP = std::pair<float, PointId>;
  std::priority_queue<DP> results;
  float worst = 0.f;
  auto consider = [&](PointId id, float d) {
    if (considered_[id] == epoch_) return;
    considered_[id] = epoch_;
    ++delta_sat_;
    if ((int)results.size() < ef) { results.emplace(d, id); worst = results.top().first; }
    else if (d < worst) { results.emplace(d, id); results.pop(); worst = results.top().first; }
    if (snap_) {
      ++snap_satj_;
      if (snap_heap_.size() < 100) { snap_heap_.push_back(d); std::push_heap(snap_heap_.begin(), snap_heap_.end()); }
      else if (d < snap_heap_.front()) { std::pop_heap(snap_heap_.begin(), snap_heap_.end()); snap_heap_.back() = d; std::push_heap(snap_heap_.begin(), snap_heap_.end()); }
      while (snap_jidx_ < snap_J_.size() && snap_J_[snap_jidx_] == snap_satj_) {
        std::vector<float> s = snap_heap_; std::sort(s.begin(), s.end());
        auto wf = [&](int ef_) -> float { return (int)s.size() >= ef_ ? s[ef_ - 1] : -1.f; };
        snap_snaps_.push_back({snap_satj_, d, wf(10), wf(50), wf(100),
                             uniq_proxy_pushes_, broad_regions_, pred_evals_, delta_});
        ++snap_jidx_;
      }
    }
  };

  auto peval = [&](PointId u) -> bool {
    if (pstamp_[u] == epoch_) return psat_[u];
    pstamp_[u] = epoch_;
    const bool s = phi.eval(u);
    psat_[u] = s ? 1 : 0;
    ++pred_evals_;
    return s;
  };
  auto dist_q = [&](PointId v) { return dist(q, v); };
  auto visit = [&](PointId v) {
    const float d = dist_q(v);
    if (peval(v)) consider(v, d);
    return d;
  };
  auto get_chain = [&](PointId u) -> uint32_t { return chain_stamp_[u] == epoch_ ? chain_[u] : 0; };
  auto set_chain = [&](PointId u, uint32_t d) { chain_[u] = d; chain_stamp_[u] = epoch_; };

  std::priority_queue<DP, std::vector<DP>, std::greater<DP>> UQ;

  auto struct_reach = [&](PointId start, float proxy_key) {
    std::queue<std::pair<PointId, int>> sq;
    if (svis_[start] != epoch_) { svis_[start] = epoch_; sq.push({start, 0}); }
    while (!sq.empty()) {
      const auto [w, h] = sq.front(); sq.pop();
      ++struct_visits_;
      if (h > 0 && peval(w)) {
        const float d = dist_q(w); consider(w, d); UQ.emplace(d, w); continue;
      }
      if (h >= struct_hops_) continue;
      for (PointId nb : g_.neighbors(w)) {
        if (svis_[nb] == epoch_ || pack_[nb].vis == vepoch8_) continue;
        if (sigma_gate >= 0.0 &&
            cpgate && stats_.sigma_hat(rm_.region_of[nb], *cpgate) < sigma_gate) continue;
        svis_[nb] = epoch_; sq.push({nb, h + 1});
      }
    }
    (void)proxy_key;
  };

  auto handle_boundary = [&](PointId v, float key_x, uint32_t seed_chain) {
    if (pack_[v].vis == vepoch8_) return;
    if (sigma_gate >= 0.0 && cpgate &&
        stats_.sigma_hat(rm_.region_of[v], *cpgate) < sigma_gate)
      return;
    switch (mode) {
      case BoundaryMode::EAGER:
        set_chain(v, 0); UQ.emplace(visit(v), v); break;
      case BoundaryMode::LAZY_PROXY:
        if (stamp_[v] != epoch_) { set_chain(v, seed_chain + 1);
          if (ppush_stamp_[v] != epoch_) { ppush_stamp_[v] = epoch_; ++uniq_proxy_pushes_; }
          UQ.emplace(key_x, v); } break;
      case BoundaryMode::GATED_PROXY:
        if (peval(v)) { const float d = dist_q(v); consider(v, d); set_chain(v, 0); UQ.emplace(d, v); }
        else if (stamp_[v] != epoch_) { set_chain(v, seed_chain + 1);
          if (ppush_stamp_[v] != epoch_) { ppush_stamp_[v] = epoch_; ++uniq_proxy_pushes_; }
          UQ.emplace(key_x, v); }
        break;
      case BoundaryMode::GATED_STRUCT:
        if (peval(v)) { const float d = dist_q(v); consider(v, d); UQ.emplace(d, v); }
        else struct_reach(v, key_x);
        break;
    }
  };

  PointId cur = g_.entry_point();
  float d_cur = dist(q, cur);
  if (two_stage_)
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
  const PointId v0 = cur;
  UQ.emplace(visit(v0), v0);

  std::queue<PointId> bfs;
  while (!UQ.empty()) {
    const auto [key_x, x] = UQ.top();
    UQ.pop();
    { const uint32_t cd = get_chain(x); if (cd > 0) chain_hist_[cd < 63 ? cd : 63]++; }
    const RegionId rx = rm_.region_of[x];
    const region_search_kind strat = decide(rx);

    if (strat == region_search_kind::BESTFIRST && stamp_[x] != epoch_) {
      const float dx = dist_q(x);
      if (peval(x)) consider(x, dx);
    }
    if ((int)results.size() >= ef && key_x >= worst) break;
    if (pack_[x].vis == vepoch8_) continue;

    if (strat == region_search_kind::FLOOD) {
      ++broad_regions_;
      const uint32_t seed_chain = get_chain(x);
      bfs.push(x);
      while (!bfs.empty()) {
        const PointId u = bfs.front();
        bfs.pop();
        if (pack_[u].vis == vepoch8_) continue;
        pack_[u].vis = vepoch8_; ++expanded_;
        if (peval(u)) consider(u, dist_q(u));
        for (PointId v : g_.neighbors(u)) {
          if (rm_.region_of[v] == rx) {
            if (pack_[v].vis != vepoch8_) bfs.push(v);
          } else {
            ++boundary_points_;
            handle_boundary(v, key_x, seed_chain);
          }
        }
      }
    } else {
      pack_[x].vis = vepoch8_; ++expanded_;
      for (PointId v : g_.neighbors(x)) UQ.emplace(visit(v), v);
    }
  }

  last_result_worst_ = results.empty() ? -1.f : results.top().first;
  std::vector<DP> out;
  out.reserve(results.size());
  while (!results.empty()) { out.push_back(results.top()); results.pop(); }
  std::reverse(out.begin(), out.end());
  if ((int)out.size() > K) out.resize(K);
  std::vector<PointId> ids;
  ids.reserve(out.size());
  for (auto& [d, id] : out) ids.push_back(id);
  return ids;
}

void OursSearch::build_tadj_() {
  const size_t N = g_.size();
  const size_t R = rm_.n_regions;
  std::vector<std::pair<uint64_t, PointId>> pr;
  pr.reserve(N * 4);
  for (PointId u = 0; u < (PointId)N; ++u) {
    const uint32_t ru = rm_.region_of[u];
    for (PointId v : g_.neighbors(u)) {
      const uint32_t rv = rm_.region_of[v];
      if (rv != ru) pr.emplace_back(((uint64_t)ru << 32) | rv, v);
    }
  }
  std::sort(pr.begin(), pr.end(),
            [](const std::pair<uint64_t, PointId>& a, const std::pair<uint64_t, PointId>& b) { return a.first < b.first; });
  tadj_off_.assign(R + 1, 0);
  tadj_dst_.clear(); tadj_rep_.clear();
  uint64_t last = ~0ull;
  for (const auto& kv : pr) {
    if (kv.first == last) continue;
    last = kv.first;
    const uint32_t ru = (uint32_t)(kv.first >> 32), rv = (uint32_t)kv.first;
    tadj_off_[ru + 1]++; tadj_dst_.push_back(rv); tadj_rep_.push_back(kv.second);
  }
  for (size_t r = 0; r < R; ++r) tadj_off_[r + 1] += tadj_off_[r];
  tvis_region_.assign(R + 1, 0);
  tcomp_of_.assign(R + 1, 0);

  mcsr_off_.assign(R + 1, 0);
  for (PointId u = 0; u < (PointId)N; ++u) mcsr_off_[rm_.region_of[u] + 1]++;
  for (size_t r = 0; r < R; ++r) mcsr_off_[r + 1] += mcsr_off_[r];
  mcsr_perm_.resize(N);
  {
    std::vector<size_t> cur(mcsr_off_.begin(), mcsr_off_.end() - 1);
    for (PointId u = 0; u < (PointId)N; ++u) mcsr_perm_[cur[rm_.region_of[u]]++] = u;
  }
  mb_stamp_.assign(R + 1, 0);
  tadj_built_ = true;
}

std::vector<PointId> OursSearch::knn_search_connected_domain(const float* q, const DNFPredicate& phi,
                                                 int ef, int K, bool enable_es2, bool enable_phase2,
                                                 int case3_mode) {
  ++epoch_;
  if (++vepoch8_ == 0) { for (auto& p : pack_) p.vis = 0; vepoch8_ = 1; }
#ifdef HS_TRAVTAX
  travtax_dec_ = 0;
#endif

  uint32_t rng = (epoch_ * 2654435761u) ^ 0x9e3779b9u;
  delta_ = 0; delta_sat_ = 0; expanded_ = 0; broad_regions_ = 0; boundary_points_ = 0;
  pred_evals_ = 0; struct_visits_ = 0; last_result_worst_ = -1.f;
  cc_phase2_struct_ = 0; cc_phase2_delta_ = 0; cc_regions_touched_ = 0; cc_broad_regions_ = 0;
  expanded_verify_ = 0; cc_sel0_b_find_ = 0; cc_sel0_b_verify_ = 0; cc_sel0_c_verify_ = 0;
  cc_transp_regions_ = 0;
  if (((transparent_ && use_presence_ && stats_.presence_on()) || membership_) && !tadj_built_)
    build_tadj_();
  qstat_descent_delta_ = 0;
  sc_seeds_C_ = 0; sc_seeds_B_ = 0; sc_peak_frontier_ = 0; sc_peak_heapq_ = 0;
  case1_ = 0; case2_ = 0; case3_ = 0; pick_first_ = 0; cm2_spread_ = 0;
  b_regions_ = 0; c_regions_ = 0; es2_fired_ = 0;
  if (epoch_ == 0) {
    std::fill(stamp_.begin(), stamp_.end(), 0u); for (auto& p : pack_) p.vis = 0;
    std::fill(considered_.begin(), considered_.end(), 0u);
    std::fill(pstamp_.begin(), pstamp_.end(), 0u); std::fill(svis2_.begin(), svis2_.end(), 0u);
    std::fill(rtype_stamp_.begin(), rtype_stamp_.end(), 0u);
    std::fill(ent_stamp_.begin(), ent_stamp_.end(), 0u);
    std::fill(sfr_stamp_.begin(), sfr_stamp_.end(), 0u); sfr_tok_ = 0;
    std::fill(absorbed_stamp_.begin(), absorbed_stamp_.end(), 0u);
    std::fill(w_stamp_.begin(), w_stamp_.end(), 0u);
    std::fill(rc_stamp_.begin(), rc_stamp_.end(), 0u);
    std::fill(dnf_tr_fstamp_.begin(), dnf_tr_fstamp_.end(), 0u);
    std::fill(dnf_tr_vstamp_.begin(), dnf_tr_vstamp_.end(), 0u);
    epoch_ = 1;
  }
#ifdef HS_EFSWEEP

  if (ef < K) ef = K;
#else
  ef = K;

#endif
  if (g_.size() == 0) return {};
  const double Cd = cm_.c_dist() > 0 ? cm_.c_dist() : 1.0;
  const double Cp = cm_.c_pred();

  const double M_Ce = dnf_degree_ * dnf_cedge_;
  const double lambda_pred = dnf_lambda_pred_;
  const double lambda_tradeoff = dnf_lambda_tradeoff_;
  const auto& dnf_clauses = phi.clauses();
  const size_t dnf_M = dnf_clauses.size();

  const double UNKNOWN = -1.0;

  const bool memb_on = membership_ && tadj_built_ && force_kind_ == CcForce::AUTO &&
                       dnf_lambda_tradeoff_ > 0.01 && dnf_lambda_tradeoff_ < 100.0;

  const double flood_pt_cost = memb_on ? scan_cost_ : M_Ce;
  if (dnf_trace_on_) dnf_trace_.clear();

  std::priority_queue<DP> results;
  float worst = 0.f;
  SearchCtx ctx{results, worst, q, ef, phi};

  struct RSet {
    std::vector<uint32_t>& st; uint32_t tok = 0; std::vector<RegionId> mem;
    explicit RSet(std::vector<uint32_t>& s) : st(s) {}
    void reset(uint32_t t) { tok = t; mem.clear(); }
    bool insert(RegionId r) { if (st[r] == tok) return false; st[r] = tok; mem.push_back(r); return true; }
    bool contains(RegionId r) const { return st[r] == tok; }
    bool empty() const { return mem.empty(); }
  };

  auto compute_region_cost = [&](RegionId r) {
    if (rc_stamp_[r] == epoch_) return;
    rc_stamp_[r] = epoch_;
    const double n = (double)stats_.region_size(r);
    double prod_all = 1.0;
    double total = 0.0, prefix_all_false = 1.0;
    for (size_t j = 0; j < dnf_M; ++j) {
      double prod = 1.0, e_j = 0.0;
      for (const ColumnPredicate& a : dnf_clauses[j]) {
        e_j += prod;
        prod *= stats_.sigma_hat(r, a);
      }
      const double sel_Cj = prod;
      prod_all *= (1.0 - sel_Cj);
      total += prefix_all_false * e_j;
      prefix_all_false *= (1.0 - sel_Cj);
    }
    double sel_dnf = 1.0 - prod_all;

    if (use_presence_ && stats_.presence_on()) {
      if (stats_.dnf_empty(r, dnf_clauses)) sel_dnf = 0.0;
      else if (sel_dnf <= 0.0) sel_dnf = 1e-12;
    }
    const double C_pred_dnf = Cp * total;
    rc_seldnf_[r]   = sel_dnf;
    rc_cpreddnf_[r] = C_pred_dnf;

    rc_costflood_[r]  = (transit_skip_ && sel_dnf <= 0.0) ? 0.0
                        : n * C_pred_dnf + sel_dnf * n * Cd;
  };

  auto cost_bestfirst_of = [&](RegionId r, double entry_dist) -> double {
    dnf_last_nstar_ = 0.0; dnf_last_pi_ = -1.0;
    const double sel_dnf = rc_seldnf_[r];

    if (sel_dnf <= 0.0 && (!sel0_verify_ || (double)results.size() < (double)K))
      return std::numeric_limits<double>::infinity();
    const double n = (double)stats_.region_size(r);
    double n_star;
    if ((double)results.size() >= (double)K) {
      if (entry_dist < 0.0) {
        n_star = n;
      } else {
        const double w = (double)cur_worst(ctx);

        if (es2aware_ && entry_dist >= w && (es2aware_ >= 2 || sel_dnf <= 0.0)) {
          dnf_last_pi_ = 0.0;
          n_star = 1.0;
        } else {
          double pi = w / entry_dist;
          if (pi > 1.0) pi = 1.0;
          dnf_last_pi_ = pi;
          n_star = n * pi;
        }
      }
    } else {
      n_star = ((double)K - (double)results.size()) / sel_dnf;
      if (n_star > n) n_star = n;
    }
    dnf_last_nstar_ = n_star;

    const bool verify = ((double)results.size() >= (double)K);
    return lambda_tradeoff * n_star * (lambda_pred * rc_cpreddnf_[r] + ((verify || trav_allphase_) ? M_Ce : 0.0) + Cd);
  };

  auto cost_model_1_dnf = [&](PointId entry) -> region_search_kind {
#ifdef HS_TRAVTAX
    ++travtax_dec_;
#endif
    const RegionId r = rm_.region_of[entry];
    if (force_kind_ != CcForce::AUTO) {
      bool isC;
      if (force_kind_ == CcForce::RANDOM) {
        uint64_t h = (uint64_t)r * 0x9E3779B97F4A7C15ull;
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 27;
        isC = (h & 1ull) != 0ull;
      } else {
        isC = (force_kind_ == CcForce::ALL_BF);
      }
      rtype_[r] = isC ? 1 : 0;
      return isC ? region_search_kind::BESTFIRST : region_search_kind::FLOOD;
    }

    if (dnf_lambda_tradeoff_ <= 0.01) return region_search_kind::BESTFIRST;
    if (dnf_lambda_tradeoff_ >= 100.0) return region_search_kind::FLOOD;

    compute_region_cost(r);

    const bool verify1 = ((double)results.size() >= (double)K);
    const double entry_dist = verify1 ? (double)dist_q(ctx, entry) : UNKNOWN;
    const double cbf = cost_bestfirst_of(r, entry_dist);
    const double cflood = rc_costflood_[r] + ((verify1 || trav_allphase_) ? (double)stats_.region_size(r) * flood_pt_cost : 0.0);
    const bool isC = (cbf < cflood);

    if (rc_seldnf_[r] <= 0.0) { if (verify1) { if (isC) ++cc_sel0_c_verify_; else ++cc_sel0_b_verify_; } else ++cc_sel0_b_find_; }
    rtype_[r] = isC ? 1 : 0;
    if (dnf_trace_on_) {
      const bool verify = ((double)results.size() >= (double)K);
      uint32_t& st = verify ? dnf_tr_vstamp_[r] : dnf_tr_fstamp_[r];
      if (st != epoch_) { st = epoch_;
        dnf_trace_.push_back(DnfTraceRow{ (uint32_t)r, (uint8_t)(verify?1:0), (uint8_t)(isC?1:0),
          (uint32_t)results.size(),
          (double)stats_.region_size(r), rc_seldnf_[r], rc_cpreddnf_[r],
          entry_dist, (double)cur_worst(ctx), dnf_last_pi_, dnf_last_nstar_, rc_costflood_[r], cbf }); }
    }
    return isC ? region_search_kind::BESTFIRST : region_search_kind::FLOOD;
  };

  auto mark_entered = [&](RegionId r) {
    if (ent_stamp_[r] == epoch_) return; ent_stamp_[r] = epoch_; ++cc_regions_touched_;
    if (attr_per_region_) { const float w = cur_worst(ctx);
      if (std::isinf(w)) ++w_spread_inf_[r]; else { w_spread_sum_[r] += w; ++w_spread_fin_[r]; }
      entry_rank_sum_[r] += (double)cc_regions_touched_; ++entry_rank_cnt_[r]; }
    if (rtype_[r]) ++c_regions_; else { ++b_regions_; ++cc_broad_regions_; }
  };

  auto cost_model_2_dnf = [&](const std::vector<RegionId>& Bregs, const std::vector<RegionId>& Cregs) -> region_search_kind {
#ifdef HS_TRAVTAX
    ++travtax_dec_;
#endif

    if (dnf_lambda_tradeoff_ <= 0.01) return region_search_kind::BESTFIRST;
    if (dnf_lambda_tradeoff_ >= 100.0) return region_search_kind::FLOOD;
    double cflood = 0, sflood = 0, cbf = 0, sbf = 0;
    const bool verify2 = ((double)results.size() >= (double)K);
    for (RegionId r : Bregs) { compute_region_cost(r);
      sflood += rc_seldnf_[r] * (double)stats_.region_size(r);
      cflood += rc_costflood_[r] + ((verify2 || trav_allphase_) ? (double)stats_.region_size(r) * flood_pt_cost : 0.0); }
    for (RegionId r : Cregs) { compute_region_cost(r);
      sbf += rc_seldnf_[r] * (double)stats_.region_size(r); cbf += cost_bestfirst_of(r, UNKNOWN); }
    const double INF = std::numeric_limits<double>::infinity();
    const double scoreB = cflood > 0 ? sflood / cflood : INF;
    const double scoreC = cbf  > 0 ? sbf  / cbf  : INF;
    return scoreB >= scoreC ? region_search_kind::FLOOD : region_search_kind::BESTFIRST;
  };
  auto regions_of_into = [&](const std::vector<PointId>& pts, std::vector<RegionId>& out) {
    out.clear();
    if (++rof_tok_ == 0) { std::fill(rof_seen_.begin(), rof_seen_.end(), 0u); rof_tok_ = 1; }
    for (PointId p : pts) { RegionId r = rm_.region_of[p]; if (rof_seen_[r] != rof_tok_) { rof_seen_[r] = rof_tok_; out.push_back(r); } }
  };

  auto sfr_begin = [&](const std::vector<RegionId>& regs) {
    if (++sfr_tok_ == 0) { std::fill(sfr_stamp_.begin(), sfr_stamp_.end(), 0u); sfr_tok_ = 1; }
    for (RegionId r : regs) sfr_stamp_[r] = sfr_tok_;
  };
  auto sfr_in  = [&](RegionId r) { return sfr_stamp_[r] == sfr_tok_; };
  auto sfr_add = [&](RegionId r) { sfr_stamp_[r] = sfr_tok_; absorbed_stamp_[r] = epoch_; };

  std::vector<DP> sc_heap;
  std::vector<PointId> sc_ph2;
  std::vector<PointId> sb_bq;
  std::vector<RegionId> gc_Bregs, gc_Cregs;

  auto transp_ok = [&]() -> bool {
    return transparent_ && tadj_built_ && use_presence_ && stats_.presence_on() &&
           force_kind_ == CcForce::AUTO &&
           dnf_lambda_tradeoff_ > 0.01 && dnf_lambda_tradeoff_ < 100.0;

  };
  auto region_empty = [&](RegionId r) -> bool {
    compute_region_cost(r);
    return rc_seldnf_[r] <= 0.0;
  };
  auto transp_emit_or_fallback = [&](RegionId nr, PointId rep, const std::function<void(PointId)>& emit_nonempty) {

    if (pack_[rep].vis == vepoch8_) {
      rep = (PointId)~0u;
      for (size_t j = mcsr_off_[nr]; j < mcsr_off_[nr + 1]; ++j)
        if (pack_[mcsr_perm_[j]].vis != vepoch8_) { rep = mcsr_perm_[j]; break; }
    }
    if (rep != (PointId)~0u) emit_nonempty(rep);
  };
  auto transp_core = [&](PointId e0, const std::function<void(PointId)>& emit_nonempty) {
    if (tcomp_epoch_ != epoch_) { tcomp_epoch_ = epoch_; tcomp_frontier_.clear(); transp_replay_worst_ = std::numeric_limits<float>::max(); transp_replay_cnt_ = 0; }
    const RegionId r0 = pack_[e0].rid;
    if (tvis_region_[r0] == epoch_) return;
    const uint32_t comp = (uint32_t)tcomp_frontier_.size();
    tcomp_frontier_.emplace_back();
    tvis_region_[r0] = epoch_; tcomp_of_[r0] = comp;
    tq_.clear(); tq_.push_back(r0);
    size_t th = 0;
    while (th < tq_.size()) {
      const RegionId r = tq_[th++];
      ++cc_transp_regions_;
      for (size_t i = tadj_off_[r]; i < tadj_off_[r + 1]; ++i) {
        const RegionId nr = tadj_dst_[i];
        if (tvis_region_[nr] == epoch_) continue;
        tvis_region_[nr] = epoch_;
        tcomp_of_[nr] = comp;
        if (region_empty(nr)) tq_.push_back(nr);
        else {
          tcomp_frontier_[comp].emplace_back(nr, tadj_rep_[i]);
          transp_emit_or_fallback(nr, tadj_rep_[i], emit_nonempty);
        }
      }
    }
  };

  auto spread_C = [&](const std::vector<PointId>& seeds, const std::vector<RegionId>& startRegs, bool pure,
                      std::vector<PointId>& C_ABs_new, std::vector<PointId>& C_ACs_new,
                      RSet& C_B_new, RSet& C_C_new) -> bool {
    sfr_begin(startRegs);
    sc_heap.clear(); sc_ph2.clear(); size_t sc_ph2_head = 0;
    auto heap_push = [&](float d, PointId id) { sc_heap.emplace_back(d, id); std::push_heap(sc_heap.begin(), sc_heap.end(), std::greater<DP>()); };
    auto heap_pop  = [&]() { std::pop_heap(sc_heap.begin(), sc_heap.end(), std::greater<DP>()); sc_heap.pop_back(); };
    auto rec_cross = [&](PointId e) {
      ++boundary_points_;
      const RegionId re = pack_[e].rid;

      if (cost_model_1_dnf(e) == region_search_kind::BESTFIRST) {
        if (C_C_new.insert(re)) C_ACs_new.push_back(e); else pack_[e].vis = 0u;
      } else {
        if (C_B_new.insert(re)) C_ABs_new.push_back(e); else pack_[e].vis = 0u;
      }
    };
    sc_seeds_C_ += seeds.size();
    for (PointId s : seeds) { pack_[s].vis = vepoch8_;
      const float d = dist_q(ctx, s); if (peval(ctx, s)) consider(ctx, s, d); heap_push(d, s); }
    while (!sc_heap.empty()) {
      if (sc_heap.size() + (sc_ph2.size() - sc_ph2_head) > sc_peak_heapq_) sc_peak_heapq_ = sc_heap.size() + (sc_ph2.size() - sc_ph2_head);
      const float dv = sc_heap.front().first; const PointId v = sc_heap.front().second; heap_pop();
      if ((int)results.size() >= ef && dv >= worst) {
        if (pure && C_B_new.empty() && C_C_new.empty()) {

          if (tcomp_epoch_ == epoch_ && worst < transp_replay_worst_ && transp_replay_cnt_ < 4) {
            transp_replay_worst_ = worst; ++transp_replay_cnt_;
            for (auto& fr : tcomp_frontier_)
              for (const auto& pr : fr)
                transp_emit_or_fallback(pr.first, pr.second,
                    [&](PointId rep) { pack_[rep].vis = vepoch8_; rec_cross(rep); });
            if (!(C_B_new.empty() && C_C_new.empty())) break;
          }
          es2_fired_ = 1;
          if (enable_es2) return true;
        }

        break;
      }
      const RegionId rv = rm_.region_of[v]; mark_entered(rv); ++expanded_;
      if ((double)results.size() >= (double)K) ++expanded_verify_;
      for (PointId e : g_.neighbors(v)) {
        if (pack_[e].vis == vepoch8_) continue; pack_[e].vis = vepoch8_;
        if (sfr_in(pack_[e].rid)) {
          const float d = dist_q(ctx, e); if (peval(ctx, e)) consider(ctx, e, d);
          if (d < cur_worst(ctx)) heap_push(d, e); else sc_ph2.push_back(e);
        } else if (transp_ok() && region_empty(pack_[e].rid)) {
          transp_core(e, [&](PointId rep) { pack_[rep].vis = vepoch8_; rec_cross(rep); });
        } else if (cost_model_1_dnf(e) == region_search_kind::BESTFIRST && pure && C_B_new.empty()) {

          const float d = dist_q(ctx, e); if (peval(ctx, e)) consider(ctx, e, d);
          if (d < cur_worst(ctx)) { sfr_add(pack_[e].rid); heap_push(d, e); }
          else sc_ph2.push_back(e);
        } else {
          rec_cross(e);
        }
      }
    }

    while (sc_ph2_head < sc_ph2.size()) {
      if ((sc_ph2.size() - sc_ph2_head) > sc_peak_heapq_) sc_peak_heapq_ = (sc_ph2.size() - sc_ph2_head);
      const PointId v = sc_ph2[sc_ph2_head++];
      for (PointId e : g_.neighbors(v)) {
        ++cc_phase2_struct_; ++struct_visits_;
        if (pack_[e].vis == vepoch8_) continue;
        if (sfr_in(pack_[e].rid)) {
          if (svis2_[e] != epoch_) {
            svis2_[e] = epoch_;
            if (!enable_phase2) { pack_[e].vis = vepoch8_; const float d = dist_q(ctx, e); if (peval(ctx, e)) consider(ctx, e, d); }
            sc_ph2.push_back(e);
          }
        } else {
          pack_[e].vis = vepoch8_;
          if (transp_ok() && region_empty(pack_[e].rid))
            transp_core(e, [&](PointId rep) { pack_[rep].vis = vepoch8_; rec_cross(rep); });
          else rec_cross(e);
        }
      }
    }
    return false;
  };

  auto spread_B = [&](const std::vector<PointId>& seeds, const std::vector<RegionId>& Bcur,
                      const std::vector<RegionId>& Ccur, bool pure,
                      std::vector<PointId>& B_ABs_new, std::vector<PointId>& B_ACs_new,
                      RSet& B_B_new, RSet& B_C_new) {
    sfr_begin(Bcur);
    sb_bq.clear(); size_t sb_bq_head = 0;
    auto consider_bfs = [&](PointId e) {
      if (peval(ctx, e)) { const float d = dist_q(ctx, e); if (d < cur_worst(ctx)) consider(ctx, e, d); }
    };
    auto gate_continue = [&](RegionId re) -> bool {
      if (pure && B_C_new.empty()) return true;
      ++cm2_spread_;
      gc_Bregs.clear(); gc_Bregs.insert(gc_Bregs.end(), Bcur.begin(), Bcur.end());
      for (RegionId r : B_B_new.mem) gc_Bregs.push_back(r); gc_Bregs.push_back(re);
      gc_Cregs.clear(); gc_Cregs.insert(gc_Cregs.end(), Ccur.begin(), Ccur.end());
      for (RegionId r : B_C_new.mem) gc_Cregs.push_back(r);
      return cost_model_2_dnf(gc_Bregs, gc_Cregs) == region_search_kind::FLOOD;
    };

    if (memb_on) {
      sc_seeds_B_ += seeds.size();

      for (PointId s : seeds) { pack_[s].vis = vepoch8_; consider_bfs(s); }
      mb_rq_.clear();
      auto sched = [&](RegionId r) {
        if (mb_stamp_[r] == epoch_) return;
        mb_stamp_[r] = epoch_; mb_rq_.push_back(r);
      };

      auto pick_seed = [&](RegionId nr, PointId rep) -> PointId {
        if (pack_[rep].vis != vepoch8_) return rep;
        for (size_t j = mcsr_off_[nr]; j < mcsr_off_[nr + 1]; ++j)
          if (pack_[mcsr_perm_[j]].vis != vepoch8_) return mcsr_perm_[j];
        return (PointId)~0u;
      };

      double mag_sf = 0, mag_cf = 0, mag_sb = 0, mag_cb = 0;
      auto mag_add_B = [&](RegionId r) { compute_region_cost(r);
        const double n2 = (double)stats_.region_size(r);
        mag_sf += rc_seldnf_[r] * n2;
        mag_cf += rc_costflood_[r] + ((trav_allphase_ || (double)results.size() >= (double)K) ? n2 * flood_pt_cost : 0.0); };
      auto mag_add_C = [&](RegionId r) { compute_region_cost(r);
        mag_sb += rc_seldnf_[r] * (double)stats_.region_size(r);
        mag_cb += cost_bestfirst_of(r, UNKNOWN); };
      for (RegionId r : Bcur) mag_add_B(r);
      for (RegionId r : Ccur) mag_add_C(r);
      auto gate_fast = [&](RegionId re) -> bool {
        if (pure && B_C_new.empty()) return true;
        ++cm2_spread_;
        compute_region_cost(re);
        const double n2 = (double)stats_.region_size(re);
        const double sf = mag_sf + rc_seldnf_[re] * n2;
        const double cf = mag_cf + rc_costflood_[re] + ((trav_allphase_ || (double)results.size() >= (double)K) ? n2 * flood_pt_cost : 0.0);
        const double scoreB = cf > 0 ? sf / cf : std::numeric_limits<double>::infinity();
        const double scoreC = mag_cb > 0 ? mag_sb / mag_cb : std::numeric_limits<double>::infinity();
        return scoreB >= scoreC;
      };
      for (RegionId r : Bcur) sched(r);
      size_t mh = 0;
      while (mh < mb_rq_.size()) {
        const RegionId r = mb_rq_[mh++];
        mark_entered(r);
        for (size_t i = mcsr_off_[r]; i < mcsr_off_[r + 1]; ++i) {
          const PointId u = mcsr_perm_[i];
          if (pack_[u].vis == vepoch8_) continue;
          pack_[u].vis = vepoch8_;
          ++expanded_;
          if ((double)results.size() >= (double)K) ++expanded_verify_;
          consider_bfs(u);
        }
        for (size_t i = tadj_off_[r]; i < tadj_off_[r + 1]; ++i) {
          const RegionId nr = tadj_dst_[i];
          if (mb_stamp_[nr] == epoch_) continue;
          ++boundary_points_;
          if (transp_ok() && region_empty(nr)) {
            transp_core(tadj_rep_[i], [&](PointId rp) {
              pack_[rp].vis = vepoch8_;
              if (cost_model_1_dnf(rp) == region_search_kind::FLOOD) {
                if (B_B_new.insert(pack_[rp].rid)) { B_ABs_new.push_back(rp); mag_add_B(pack_[rp].rid); } else pack_[rp].vis = 0u;
              } else {
                if (B_C_new.insert(pack_[rp].rid)) { B_ACs_new.push_back(rp); mag_add_C(pack_[rp].rid); } else pack_[rp].vis = 0u;
              }
            });
            continue;
          }
          const PointId sd = pick_seed(nr, tadj_rep_[i]);
          if (sd == (PointId)~0u) { mb_stamp_[nr] = epoch_; continue; }
          if (cost_model_1_dnf(sd) == region_search_kind::FLOOD) {
            if (gate_fast(nr)) { sched(nr); }
            else { pack_[sd].vis = vepoch8_; if (B_B_new.insert(nr)) { B_ABs_new.push_back(sd); mag_add_B(nr); } else pack_[sd].vis = 0u; }
          } else {
            pack_[sd].vis = vepoch8_; if (B_C_new.insert(nr)) { B_ACs_new.push_back(sd); mag_add_C(nr); } else pack_[sd].vis = 0u;
          }
        }
      }
      return;
    }
    sc_seeds_B_ += seeds.size();
    for (PointId s : seeds) { pack_[s].vis = vepoch8_; consider_bfs(s); sb_bq.push_back(s); }
    while (sb_bq_head < sb_bq.size()) {
      if ((sb_bq.size() - sb_bq_head) > sc_peak_heapq_) sc_peak_heapq_ = (sb_bq.size() - sb_bq_head);
      const PointId v = sb_bq[sb_bq_head++];
      const RegionId rv = rm_.region_of[v]; mark_entered(rv); ++expanded_;
      if ((double)results.size() >= (double)K) ++expanded_verify_;
      for (PointId e : g_.neighbors(v)) {
        if (pack_[e].vis == vepoch8_) continue; pack_[e].vis = vepoch8_;
        if (sfr_in(pack_[e].rid)) {
          consider_bfs(e); sb_bq.push_back(e);
        } else if (transp_ok() && region_empty(pack_[e].rid)) {
          ++boundary_points_;
          transp_core(e, [&](PointId rep) {
            pack_[rep].vis = vepoch8_;
            if (cost_model_1_dnf(rep) == region_search_kind::FLOOD) {
              if (B_B_new.insert(pack_[rep].rid)) B_ABs_new.push_back(rep); else pack_[rep].vis = 0u;
            } else {
              if (B_C_new.insert(pack_[rep].rid)) B_ACs_new.push_back(rep); else pack_[rep].vis = 0u;
            }
          });
        } else if (cost_model_1_dnf(e) == region_search_kind::FLOOD) {
          ++boundary_points_;
          if (gate_continue(pack_[e].rid)) { sfr_add(pack_[e].rid); consider_bfs(e); sb_bq.push_back(e); }
          else { if (B_B_new.insert(pack_[e].rid)) B_ABs_new.push_back(e); else pack_[e].vis = 0u; }
        } else {
          ++boundary_points_;
          if (B_C_new.insert(pack_[e].rid)) B_ACs_new.push_back(e); else pack_[e].vis = 0u;
        }
      }
    }
  };

  const uint64_t pre_descent = delta_;
  PointId cur = g_.entry_point();
  float d_cur = dist(q, cur);
  if (two_stage_)
    for (int lc = g_.max_level(); lc > 0; --lc) {
      bool changed = true;
      while (changed) { changed = false;
        for (PointId nb : g_.neighbors_at(cur, lc)) {
          const float d = dist(q, nb);
          if (d < d_cur) { d_cur = d; cur = nb; changed = true; } } }
    }
  qstat_descent_delta_ = delta_ - pre_descent;
  const PointId v0 = cur;
  pack_[v0].vis = vepoch8_;
  if (peval(ctx, v0)) consider(ctx, v0, d_cur);

  std::vector<PointId> ABs, ACs; std::vector<RegionId> B, C;
  if (cost_model_1_dnf(v0) == region_search_kind::FLOOD) { ABs = {v0}; B = {rm_.region_of[v0]}; }
  else { ACs = {v0}; C = {rm_.region_of[v0]}; }

  std::vector<PointId> B_ABs_new, C_ABs_new, B_ACs_new, C_ACs_new, nABs, nACs;
  RSet B_B_new(rs_bb_), C_B_new(rs_cb_), B_C_new(rs_bc_), C_C_new(rs_cc_);
  while (!ABs.empty() || !ACs.empty()) {
    if (ABs.size() + ACs.size() > sc_peak_frontier_) sc_peak_frontier_ = ABs.size() + ACs.size();
    B_ABs_new.clear(); C_ABs_new.clear(); B_ACs_new.clear(); C_ACs_new.clear(); nABs.clear(); nACs.clear();
    if (++rs_tok_ == 0) { std::fill(rs_bb_.begin(), rs_bb_.end(), 0u); std::fill(rs_cb_.begin(), rs_cb_.end(), 0u);
      std::fill(rs_bc_.begin(), rs_bc_.end(), 0u); std::fill(rs_cc_.begin(), rs_cc_.end(), 0u); rs_tok_ = 1; }
    B_B_new.reset(rs_tok_); C_B_new.reset(rs_tok_); B_C_new.reset(rs_tok_); C_C_new.reset(rs_tok_);
    if (ABs.empty()) {
      ++case1_;
      if (spread_C(ACs, C, true, C_ABs_new, C_ACs_new, C_B_new, C_C_new)) break;
      ABs.swap(C_ABs_new); ACs.swap(C_ACs_new);
    } else if (ACs.empty()) {
      ++case2_;
      spread_B(ABs, B, C, true, B_ABs_new, B_ACs_new, B_B_new, B_C_new);
      ABs.swap(B_ABs_new); ACs.swap(B_ACs_new);
    } else {
      ++case3_; ++pick_first_;
      bool walk_B_first;
      switch (case3_mode) {
        case 1: rng = rng * 1664525u + 1013904223u; walk_B_first = (rng >> 16) & 1u; break;
        case 2: walk_B_first = true;  break;
        case 3: walk_B_first = false; break;
        default: walk_B_first = (cost_model_2_dnf(B, C) == region_search_kind::FLOOD); break;
      }
      if (walk_B_first) {
        spread_B(ABs, B, C, false, B_ABs_new, B_ACs_new, B_B_new, B_C_new);
        nACs.insert(nACs.end(), ACs.begin(), ACs.end()); for (PointId p : B_ACs_new) nACs.push_back(p);
        ABs.swap(B_ABs_new); ACs.swap(nACs);
      } else {
        spread_C(ACs, C, false, C_ABs_new, C_ACs_new, C_B_new, C_C_new);
        nABs.insert(nABs.end(), ABs.begin(), ABs.end()); for (PointId p : C_ABs_new) nABs.push_back(p);
        ABs.swap(nABs); ACs.swap(C_ACs_new);
      }
    }
    regions_of_into(ABs, B); regions_of_into(ACs, C);
  }

  last_result_worst_ = results.empty() ? -1.f : results.top().first;
  std::vector<DP> out; out.reserve(results.size());
  while (!results.empty()) { out.push_back(results.top()); results.pop(); }
  std::reverse(out.begin(), out.end());
  if ((int)out.size() > K) out.resize(K);
  std::vector<PointId> ids; ids.reserve(out.size());
  for (auto& [d, id] : out) ids.push_back(id);
  return ids;
}

}
