#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/cost_model/EstCostModel.hpp"
#include "hnsw_stats/hnsw/HnswGraph.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"
#include "hnsw_stats/predicate/DNFPredicate.hpp"
#include "hnsw_stats/region/RegionBuilder.hpp"
#include "hnsw_stats/stats/RegionHistogramStats.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hnsw_stats {

enum class BoundaryMode { EAGER, LAZY_PROXY, GATED_PROXY, GATED_STRUCT };

enum class BfCostModel { G, F, FG, Gplus,
                         Agraph, Chead, Bthresh };

enum class CcForce { AUTO, ALL_FLOOD, ALL_BF, RANDOM };

enum class BoundaryScheme { A, B };

struct F0Snapshot {
  uint32_t j = 0;
  float d_enc = 0, worst10 = -1, worst50 = -1, worst100 = -1;
  uint64_t uniq_proxy = 0, regions = 0, pred = 0, delta = 0;
};

class OursSearch {
 public:
  OursSearch(const HnswGraph& g, const RegionMap& rm,
             const RegionHistogramStats& stats, const EstCostModel& cm)
      : g_(g), rm_(rm), stats_(stats), cm_(cm),
        stamp_(g.size(), 0), cache_(g.size(), 0.f),
        pack_(g.size()), considered_(g.size(), 0), svis_(g.size(), 0),
        pstamp_(g.size(), 0), psat_(g.size(), 0),
        chain_(g.size(), 0), chain_stamp_(g.size(), 0), ppush_stamp_(g.size(), 0),
        unsatc_(rm.n_regions + 1, 0), unsatc_stamp_(rm.n_regions + 1, 0),
        detax_stamp_(rm.n_regions + 1, 0),
        svis2_(g.size(), 0), rtouch_stamp_(rm.n_regions + 1, 0),
        pend_stamp_(g.size(), 0),
        dr_cache_(rm.n_regions + 1, 0.f), dr_stamp_(rm.n_regions + 1, 0),
        gp_sd_(rm.n_regions + 1, 0.f), gp_rho_(rm.n_regions + 1, 0.f), gp_sw_(rm.n_regions + 1, 0.f),
        gp_fracraw_(rm.n_regions + 1, 0.f), gp_fracclamped_(rm.n_regions + 1, 0.f),
        gp_stamp_(rm.n_regions + 1, 0),
        rtouch_rstamp_(rm.n_regions + 1, 0), rtouch_rdelta_(rm.n_regions + 1, 0),
        rtouch_rcent_(rm.n_regions + 1, 0), rtouch_rpred_(rm.n_regions + 1, 0),
        rtouch_rphase2_(rm.n_regions + 1, 0),
        b_disc_pos_(g.size(), 0), b_disc_stamp_(g.size(), 0), b_disc_flood_(g.size(), 0),
        rtype_stamp_(rm.n_regions + 1, 0), rtype_(rm.n_regions + 1, 0),
        ent_stamp_(rm.n_regions + 1, 0), sfr_stamp_(rm.n_regions + 1, 0),
        rs_bb_(rm.n_regions + 1, 0), rs_cb_(rm.n_regions + 1, 0), rs_bc_(rm.n_regions + 1, 0),
        rs_cc_(rm.n_regions + 1, 0), rof_seen_(rm.n_regions + 1, 0),
        absorbed_stamp_(rm.n_regions + 1, 0),
        per_region_dist_(rm.n_regions + 1, 0), per_region_pred_(rm.n_regions + 1, 0),
        w_stamp_(rm.n_regions + 1, 0),
        w_spread_sum_(rm.n_regions + 1, 0.0),
        w_spread_fin_(rm.n_regions + 1, 0), w_spread_inf_(rm.n_regions + 1, 0),
        entry_rank_sum_(rm.n_regions + 1, 0.0), entry_rank_cnt_(rm.n_regions + 1, 0),
        dnf_tr_fstamp_(rm.n_regions + 1, 0), dnf_tr_vstamp_(rm.n_regions + 1, 0),
        rc_stamp_(rm.n_regions + 1, 0),
        rc_seldnf_(rm.n_regions + 1, 0.0),
        rc_cpreddnf_(rm.n_regions + 1, 0.0), rc_costflood_(rm.n_regions + 1, 0.0),
        dkind_(rm.n_regions + 1, 0), dkind_stamp_(rm.n_regions + 1, 0) {

    if (rm.n_regions >= (1u << 24))
      throw std::runtime_error("OursSearch a2a: n_regions >= 2^24 exceeds 24-bit pack rid");
    for (size_t i = 0; i < pack_.size(); ++i) pack_[i].rid = static_cast<uint32_t>(rm.region_of[i]);
  }

  std::vector<PointId> search(const float* q, const ColumnPredicate& phi,
                              int ef, int K);

  std::vector<PointId> search_boundary(const float* q, const ColumnPredicate& phi,
                                       BoundaryMode mode, int ef, int K,
                                       double sigma_gate = -1.0, bool force_broad = false,
                                       bool force_bestfirst = false);

  uint64_t cc_boundary_distanced() const { return cc_bnd_dist_; }
  uint64_t cc_phase2_struct() const { return cc_phase2_struct_; }
  uint64_t cc_phase2_delta() const { return cc_phase2_delta_; }
  uint64_t cc_regions_touched() const { return cc_regions_touched_; }
  uint64_t cc_broad_regions() const { return cc_broad_regions_; }

  uint64_t case1() const { return case1_; }
  uint64_t case2() const { return case2_; }
  uint64_t case3() const { return case3_; }
  uint64_t pick_first() const { return pick_first_; }
  uint64_t cm2_spread() const { return cm2_spread_; }
  uint64_t b_regions() const { return b_regions_; }
  uint64_t c_regions() const { return c_regions_; }
  uint64_t es2_fired() const { return es2_fired_; }

  void set_force(CcForce f) { force_kind_ = f; }
  void set_attr_per_region(bool b) { attr_per_region_ = b; }
  void reset_per_region() { std::fill(per_region_dist_.begin(), per_region_dist_.end(), 0u);
                               std::fill(per_region_pred_.begin(), per_region_pred_.end(), 0u);

                               std::fill(w_spread_sum_.begin(), w_spread_sum_.end(), 0.0);
                               std::fill(w_spread_fin_.begin(), w_spread_fin_.end(), 0u);
                               std::fill(w_spread_inf_.begin(), w_spread_inf_.end(), 0u);
                               std::fill(entry_rank_sum_.begin(), entry_rank_sum_.end(), 0.0);
                               std::fill(entry_rank_cnt_.begin(), entry_rank_cnt_.end(), 0u); }
  double entry_rank_mean(RegionId r) const { return r < entry_rank_cnt_.size() && entry_rank_cnt_[r] ? entry_rank_sum_[r] / entry_rank_cnt_[r] : -1.0; }
  uint64_t per_region_dist(RegionId r) const { return r < per_region_dist_.size() ? per_region_dist_[r] : 0; }
  uint64_t per_region_pred(RegionId r) const { return r < per_region_pred_.size() ? per_region_pred_[r] : 0; }

  uint8_t region_routed(RegionId r) const { return r < rtype_.size() ? rtype_[r] : 0; }

  void set_worsttl(bool b) { worsttl_on_ = b; }
  void worsttl_clear() { worsttl_.clear(); }
  const std::vector<std::pair<uint64_t, float>>& worsttl() const { return worsttl_; }

  std::vector<PointId> knn_search_connected_domain(const float* q, const DNFPredicate& phi, int ef, int K,
                                     bool enable_es2 = true, bool enable_phase2 = true, int case3_mode = 0);
  void set_dnf_degree(double d) { dnf_degree_ = d; }
  void set_dnf_cedge(double c) { dnf_cedge_ = c; }
  void set_dnf_cedge_flood(double c) { dnf_cedge_flood_ = c; }
  double dnf_cedge() const { return dnf_cedge_; }
  double dnf_cedge_flood() const { return dnf_cedge_flood_; }
  double dnf_degree() const { return dnf_degree_; }

  void set_dnf_lambda_pred(double l) { dnf_lambda_pred_ = l; }
  double dnf_lambda_pred() const { return dnf_lambda_pred_; }

  void set_dnf_lambda_tradeoff(double l) { dnf_lambda_tradeoff_ = l; }
  double dnf_lambda_tradeoff() const { return dnf_lambda_tradeoff_; }

  void set_sel0_verify(bool b) { sel0_verify_ = b; }
  bool sel0_verify() const { return sel0_verify_; }

  void set_es2aware(int m) { es2aware_ = m; }
  int es2aware() const { return es2aware_; }

  void set_transit_skip(bool b) { transit_skip_ = b; }
  bool transit_skip() const { return transit_skip_; }

  void set_use_presence(bool b) { use_presence_ = b; }
  bool use_presence() const { return use_presence_; }

  void set_transparent(bool b) { transparent_ = b; }
  bool transparent() const { return transparent_; }
  void invalidate_tadj() { tadj_built_ = false; }

  void set_membership(bool b) { membership_ = b; }
  bool membership() const { return membership_; }
  void set_scan_cost(double c) { scan_cost_ = c; }
  uint64_t transp_regions() const { return cc_transp_regions_; }

  struct DnfTraceRow { uint32_t region; uint8_t phase , decision ;
    uint32_t card;
    double n, sel_dnf, c_pred_dnf, entry_dist, worst, pi, n_star, cost_flood, cost_bestfirst; };
  void set_dnf_trace(bool b) { dnf_trace_on_ = b; }
  const std::vector<DnfTraceRow>& dnf_trace() const { return dnf_trace_; }

  uint64_t sc_seeds_C() const { return sc_seeds_C_; }
  uint64_t sc_seeds_B() const { return sc_seeds_B_; }
  uint64_t sc_peak_frontier() const { return sc_peak_frontier_; }
  uint64_t sc_peak_heapq() const { return sc_peak_heapq_; }

  double w_spread_mean(RegionId r) const { return r < w_spread_fin_.size() && w_spread_fin_[r] ? w_spread_sum_[r] / w_spread_fin_[r] : -1.0; }
  uint32_t w_spread_fin(RegionId r) const { return r < w_spread_fin_.size() ? w_spread_fin_[r] : 0; }
  uint32_t w_spread_inf(RegionId r) const { return r < w_spread_inf_.size() ? w_spread_inf_[r] : 0; }

  bool last_visited(PointId u) const { return u < pack_.size() && pack_[u].vis == vepoch8_; }

  bool last_distanced(PointId u) const { return u < stamp_.size() && stamp_[u] == epoch_; }
  bool last_considered(PointId u) const { return u < considered_.size() && considered_[u] == epoch_; }
  bool last_svis2(PointId u) const { return u < svis2_.size() && svis2_[u] == epoch_; }

  bool last_region_absorbed(RegionId r) const { return r < absorbed_stamp_.size() && absorbed_stamp_[r] == epoch_; }

  uint64_t qstat_uq_emplace() const { return qstat_uq_emplace_; }
  uint64_t qstat_fifo_push() const { return qstat_fifo_push_; }
  uint64_t qstat_bnd_dist() const { return qstat_bnd_dist_; }
  uint64_t qstat_broad_bnd_dist() const { return qstat_broad_bnd_dist_; }
  uint64_t qstat_term_bfsq_nonempty() const { return qstat_term_bfsq_nonempty_; }
  uint64_t qstat_phase2_struct() const { return cc_phase2_struct_; }
  uint64_t qstat_phase2_delta() const { return cc_phase2_delta_; }
  uint64_t qstat_regions_touched() const { return cc_regions_touched_; }
  uint64_t qstat_broad_regions() const { return cc_broad_regions_; }

  bool qstat_region_touched(RegionId r) const {
    return r < rtouch_stamp_.size() && rtouch_stamp_[r] == epoch_;
  }

  struct RegionTrace { uint32_t region; float sigma; float d_seed; float worst; uint8_t broad;
    uint32_t n_region; float sd; float rho; float sw; float frac_raw; float frac_clamped; };
  void rtrace_set_trace(bool b) { rtrace_trace_on_ = b; }
  const std::vector<RegionTrace>& rtrace_trace() const { return rtrace_trace_; }
  uint64_t centroid_calls() const { return centroid_calls_; }

  float centroid_dist(RegionId r) const { return (r < dr_stamp_.size() && dr_stamp_[r] == epoch_) ? dr_cache_[r] : -1.f; }

  void rtouch_set_trace(bool b) { rtouch_trace_on_ = b; }
  uint32_t rtouch_region_delta(RegionId r) const { return (r < rtouch_rstamp_.size() && rtouch_rstamp_[r] == epoch_) ? rtouch_rdelta_[r] : 0u; }
  uint32_t rtouch_region_cent(RegionId r) const { return (r < rtouch_rstamp_.size() && rtouch_rstamp_[r] == epoch_) ? rtouch_rcent_[r] : 0u; }
  uint32_t rtouch_region_pred(RegionId r) const { return (r < rtouch_rstamp_.size() && rtouch_rstamp_[r] == epoch_) ? rtouch_rpred_[r] : 0u; }
  uint32_t rtouch_region_phase2(RegionId r) const { return (r < rtouch_rstamp_.size() && rtouch_rstamp_[r] == epoch_) ? rtouch_rphase2_[r] : 0u; }

  uint64_t qstat_descent_delta() const { return qstat_descent_delta_; }

  void b_set_trace(bool b) { b_trace_ = b; }
  uint64_t b_cpred_broad() const { return cpred_broad_; }
  uint64_t b_cdist_broad() const { return cdist_broad_; }
  uint64_t b_struct_broad() const { return struct_broad_; }
  uint64_t b_cpred_bf() const { return cpred_bf_; }
  uint64_t b_cdist_bf() const { return cdist_bf_; }

  size_t b_cd_count() const { return b_cd_pos_.size(); }
  const uint32_t* b_cd_pos() const { return b_cd_pos_.data(); }
  const float* b_cd_val() const { return b_cd_val_.data(); }
  const uint64_t* b_cd_struct() const { return b_cd_struct_.data(); }

  uint32_t b_disc_pos(PointId id) const { return (id < b_disc_stamp_.size() && b_disc_stamp_[id] == epoch_) ? b_disc_pos_[id] : 0xFFFFFFFFu; }
  bool b_disc_in_flood(PointId id) const { return id < b_disc_flood_.size() && b_disc_stamp_[id] == epoch_ && b_disc_flood_[id]; }
  const std::vector<PointId>& b_efset() const { return b_efset_; }

  void b4_arm(uint64_t t_final, uint64_t t_final_ef, float worst_k, float worst_ef) {
    b4_mode_ = true; b4_tf_ = t_final; b4_tfef_ = t_final_ef; b4_wk_ = worst_k; b4_wef_ = worst_ef; }
  void b4_disarm() { b4_mode_ = false; }
  uint64_t b4_front_tot_tf() const { return b4_front_tot_tf_; }
  uint64_t b4_front_ge_tf() const { return b4_front_ge_tf_; }
  uint64_t b4_front_tot_tfef() const { return b4_front_tot_tfef_; }
  uint64_t b4_front_ge_tfef() const { return b4_front_ge_tfef_; }

  void rtrace_set_sigma0_mask(const std::vector<uint8_t>* m) { rtrace_sigma0_ = m; }

  std::vector<PointId> search_with_strategy(
      const float* q, const Predicate& phi,
      const std::vector<region_search_kind>& region_strategy, int ef, int K,
      bool lazy = false);

  uint64_t delta() const { return delta_; }
  uint64_t delta_satisfying() const { return delta_sat_; }
  uint64_t expanded() const { return expanded_; }

  uint64_t expanded_verify() const { return expanded_verify_; }
  uint64_t sel0_b_find() const { return cc_sel0_b_find_; }
  uint64_t sel0_b_verify() const { return cc_sel0_b_verify_; }
  uint64_t sel0_c_verify() const { return cc_sel0_c_verify_; }
  uint64_t broad_regions() const { return broad_regions_; }
  uint64_t boundary_points() const { return boundary_points_; }
  uint64_t pred_evals() const { return pred_evals_; }
  uint64_t struct_visits() const { return struct_visits_; }

  const uint64_t* chain_hist() const { return chain_hist_; }
  static constexpr int kChainBuckets = 64;
  uint64_t uniq_proxy_pushes() const { return uniq_proxy_pushes_; }
  float last_result_worst() const { return last_result_worst_; }

  void snap_enable(const std::vector<uint32_t>& J) { snap_ = true; snap_J_ = J;
    std::sort(snap_J_.begin(), snap_J_.end()); snap_J_.erase(std::unique(snap_J_.begin(), snap_J_.end()), snap_J_.end()); }
  void snap_disable() { snap_ = false; }
  const std::vector<F0Snapshot>& snap_snapshots() const { return snap_snaps_; }
  void set_struct_hops(int h) { struct_hops_ = h < 1 ? 1 : h; }
  void set_two_stage(bool b) { two_stage_ = b; }

 private:

  using DP = std::pair<float, PointId>;
  struct SearchCtx {
    std::priority_queue<DP>& results;
    float& worst;
    const float* q;
    int ef;
    const Predicate& phi;

  };

  void consider(SearchCtx& ctx, PointId id, float d);

  bool peval(SearchCtx& ctx, PointId u);

  float dist_q(SearchCtx& ctx, PointId v) { return dist(ctx.q, v); }

  float cur_worst(const SearchCtx& ctx) const {
    return (int)ctx.results.size() >= ctx.ef ? ctx.worst
                                             : std::numeric_limits<float>::infinity();
  }

  float dist(const float* q, PointId u) {
    if (stamp_[u] == epoch_) return cache_[u];
    stamp_[u] = epoch_;
    const float d = l2_sqr_(q, u);
    cache_[u] = d;
    ++delta_;
    if (attr_per_region_) per_region_dist_[rm_.region_of[u]] += 1;
    if (rtouch_trace_on_) { const RegionId ru = rm_.region_of[u]; rtouch_touch_(ru); ++rtouch_rdelta_[ru]; }
    return d;
  }

  inline void rtouch_touch_(RegionId r) {
    if (rtouch_rstamp_[r] != epoch_) { rtouch_rstamp_[r] = epoch_;
      rtouch_rdelta_[r] = 0; rtouch_rcent_[r] = 0; rtouch_rpred_[r] = 0; rtouch_rphase2_[r] = 0; }
  }
  float l2_sqr_(const float* q, PointId u) const;

  std::vector<PointId> run(const float* q, const Predicate& phi,
                           const std::function<region_search_kind(RegionId)>& decide,
                           int ef, int K, BoundaryMode mode, double sigma_gate = -1.0);

  const HnswGraph& g_;
  const RegionMap& rm_;
  const RegionHistogramStats& stats_;
  const EstCostModel& cm_;
  std::vector<uint32_t> stamp_;
  std::vector<float> cache_;

  struct alignas(4) PackEntry { uint32_t rid : 24, vis : 8; };
  std::vector<PackEntry> pack_;
  std::vector<uint32_t> considered_;
  std::vector<uint32_t> svis_;
  std::vector<uint32_t> pstamp_;
  std::vector<uint8_t> psat_;
  std::vector<uint32_t> chain_;
  std::vector<uint32_t> chain_stamp_;
  uint64_t chain_hist_[64] = {0};
  std::vector<uint32_t> ppush_stamp_;
  uint64_t uniq_proxy_pushes_ = 0;
  float last_result_worst_ = -1.f;

  bool snap_ = false; std::vector<uint32_t> snap_J_; std::vector<F0Snapshot> snap_snaps_;
  std::vector<float> snap_heap_; uint32_t snap_satj_ = 0; size_t snap_jidx_ = 0;
  int struct_hops_ = 2;
  bool two_stage_ = true;

  std::vector<uint32_t> unsatc_, unsatc_stamp_, detax_stamp_;
  uint64_t detax_regions_ = 0, cross_flood_ = 0, probe_sat_ = 0;

  std::vector<uint32_t> svis2_;
  std::vector<uint32_t> rtouch_stamp_;
  uint64_t cc_bnd_dist_ = 0, cc_phase2_struct_ = 0, cc_phase2_delta_ = 0;
  uint64_t cc_regions_touched_ = 0, cc_broad_regions_ = 0;

  std::vector<uint32_t> rtype_stamp_;
  std::vector<uint8_t>  rtype_;
  std::vector<uint32_t> ent_stamp_;

  std::vector<uint32_t> sfr_stamp_; uint32_t sfr_tok_ = 0;

  std::vector<uint32_t> rs_bb_, rs_cb_, rs_bc_, rs_cc_, rof_seen_; uint32_t rs_tok_ = 0, rof_tok_ = 0;

  std::vector<uint32_t> absorbed_stamp_;
  uint64_t case1_ = 0, case2_ = 0, case3_ = 0, pick_first_ = 0;
  uint64_t cm2_spread_ = 0;
  uint64_t b_regions_ = 0, c_regions_ = 0, es2_fired_ = 0;

  CcForce force_kind_ = CcForce::AUTO;
  bool attr_per_region_ = false;
  std::vector<uint64_t> per_region_dist_, per_region_pred_;

  uint64_t sc_seeds_C_ = 0, sc_seeds_B_ = 0, sc_peak_frontier_ = 0, sc_peak_heapq_ = 0;

  struct ColoSlot { RegionId region; uint32_t vexp; };

  double dnf_degree_ = 21.0;
  double dnf_cedge_ = 101e-9;
  double dnf_cedge_flood_ = 101e-9;

  double dnf_lambda_pred_ = 1.0;
  double dnf_lambda_tradeoff_ = 1.0;
  bool sel0_verify_ = true;
  bool use_presence_ = true;
  bool transparent_ = true;
  bool membership_ = true;
  double scan_cost_ = 2e-9;
  std::vector<size_t>  mcsr_off_;
  std::vector<PointId> mcsr_perm_;
  std::vector<uint32_t> mb_stamp_;
  std::vector<RegionId> mb_rq_;
  bool tadj_built_ = false;
  std::vector<size_t>  tadj_off_;
  std::vector<RegionId> tadj_dst_;
  std::vector<PointId>  tadj_rep_;
  std::vector<uint32_t> tvis_region_;
  std::vector<RegionId> tq_;

  std::vector<uint32_t> tcomp_of_;
  std::vector<std::vector<std::pair<RegionId, PointId>>> tcomp_frontier_;
  uint32_t tcomp_epoch_ = 0;
  const bool trav_allphase_ = !(std::getenv("HS_TRAV_ALLPHASE") && atoi(std::getenv("HS_TRAV_ALLPHASE")) == 0);

  float transp_replay_worst_ = 0.f;
  uint32_t transp_replay_cnt_ = 0;
  void build_tadj_();
  bool transit_skip_ = true;
  int es2aware_ = 1;
  double dnf_last_nstar_ = 0.0;
  double dnf_last_pi_ = -1.0;
  bool dnf_trace_on_ = false;
  std::vector<DnfTraceRow> dnf_trace_;
  std::vector<uint32_t> dnf_tr_fstamp_, dnf_tr_vstamp_;
  std::vector<uint32_t> rc_stamp_;
  std::vector<double> rc_seldnf_, rc_cpreddnf_, rc_costflood_;

  std::vector<uint8_t> dkind_; std::vector<uint64_t> dkind_stamp_;

  bool worsttl_on_ = false;
  std::vector<std::pair<uint64_t, float>> worsttl_;
  std::vector<uint32_t> w_stamp_;
  std::vector<double>   w_spread_sum_;
  std::vector<uint32_t> w_spread_fin_, w_spread_inf_;

  std::vector<double>   entry_rank_sum_;
  std::vector<uint32_t> entry_rank_cnt_;

  std::vector<uint32_t> pend_stamp_;
  uint64_t qstat_uq_emplace_ = 0, qstat_fifo_push_ = 0, qstat_bnd_dist_ = 0;
  uint64_t qstat_broad_bnd_dist_ = 0, qstat_term_bfsq_nonempty_ = 0;
  bool rtrace_trace_on_ = false;
  std::vector<RegionTrace> rtrace_trace_;
  const std::vector<uint8_t>* rtrace_sigma0_ = nullptr;
  std::vector<float> dr_cache_; std::vector<uint32_t> dr_stamp_;

  std::vector<float> gp_sd_, gp_rho_, gp_sw_, gp_fracraw_, gp_fracclamped_; std::vector<uint32_t> gp_stamp_;
  uint64_t centroid_calls_ = 0;

  bool rtouch_trace_on_ = false;
  std::vector<uint32_t> rtouch_rstamp_, rtouch_rdelta_, rtouch_rcent_, rtouch_rpred_, rtouch_rphase2_;

  uint64_t qstat_descent_delta_ = 0;

  bool b_trace_ = false; bool b_inflood_ = false;
  uint64_t cpred_broad_ = 0, cdist_broad_ = 0, struct_broad_ = 0, cpred_bf_ = 0, cdist_bf_ = 0;
  std::vector<uint32_t> b_disc_pos_, b_disc_stamp_; std::vector<uint8_t> b_disc_flood_;
  std::vector<uint32_t> b_cd_pos_; std::vector<float> b_cd_val_; std::vector<uint64_t> b_cd_struct_;
  std::vector<PointId> b_efset_;

  bool b4_mode_ = false; uint64_t b4_tf_ = 0, b4_tfef_ = 0; float b4_wk_ = 0, b4_wef_ = 0;
  bool b4_done_tf_ = false, b4_done_tfef_ = false;
  uint64_t b4_front_tot_tf_ = 0, b4_front_ge_tf_ = 0, b4_front_tot_tfef_ = 0, b4_front_ge_tfef_ = 0;
  uint32_t epoch_ = 0;
  uint8_t vepoch8_ = 0;

  uint64_t delta_ = 0;
#ifdef HS_TRAVTAX
 public:

  uint64_t travtax_decisions() const { return travtax_dec_; }
 private:
  uint64_t travtax_dec_ = 0;
#endif
  uint64_t delta_sat_ = 0;
  uint64_t expanded_ = 0;
  uint64_t expanded_verify_ = 0;
  uint64_t cc_sel0_b_find_ = 0, cc_sel0_b_verify_ = 0, cc_sel0_c_verify_ = 0;
  uint64_t cc_transp_regions_ = 0;
  uint64_t broad_regions_ = 0;
  uint64_t boundary_points_ = 0;
  uint64_t pred_evals_ = 0;
  uint64_t struct_visits_ = 0;
};

}
