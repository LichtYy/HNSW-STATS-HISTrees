#pragma once

#include "method/Method.hpp"

#include <memory>
#include <string>

#include "common/DatasetLoader.hpp"
#include "hnsw_stats/cost_model/EstCostModel.hpp"
#include "hnsw_stats/hnsw/HnswIndex.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"
#include "hnsw_stats/region/RegionBuilder.hpp"
#include "hnsw_stats/search/OursSearch.hpp"
#include "hnsw_stats/stats/RegionHistogramStats.hpp"

namespace hnsw_stats {

class OursMethod final : public Method {
 public:

  OursMethod(int K, int region_level = 2, int hist_buckets = 32)
      : K_(K), region_level_(region_level), hist_buckets_(hist_buckets) {}

  void build(const Dataset& ds, const BuildParams& bp) override;
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;

  void set_multilayer(bool on, int lmat_min = 1, int lmat_max = -1, size_t budget_bytes = 0,
                      double delta_range = 0.05, double eps_point = 0.10) {
    multilayer_on_ = on; lmat_min_ = lmat_min; lmat_max_ = lmat_max;
    budget_bytes_ = budget_bytes; delta_range_ = delta_range; eps_point_ = eps_point;
  }
  void set_naware_c(double c) { naware_c_ = c; }
  bool multilayer_on() const { return multilayer_on_; }
  RegionId region_boundary_cnt() const { return stats_ ? stats_->region_boundary_cnt() : 0; }
  size_t stats_bytes() const { return stats_ ? stats_->bytes() : 0; }

  SearchResult search_mode(const Query& q, const ColumnPredicate& phi, int ef,
                           BoundaryMode mode, double sigma_gate = -1.0, bool force_broad = false,
                           bool force_bestfirst = false);
  void set_r3(bool b) { cm_->set_r3(b); }
  void set_cpred_ratio(double r) { cm_->set_cpred_ratio(r); }
  void clear_cpred_ratio() { cm_->clear_cpred_ratio(); }
  void set_cpred_override(double v) { cm_->set_cpred_override(v); }
  void clear_cpred_override() { cm_->clear_cpred_override(); }
  double theta_for(size_t n) const { return cm_->theta(n); }

  uint64_t last_broad_regions() const { return searcher_->broad_regions(); }
  uint64_t last_boundary_points() const { return searcher_->boundary_points(); }
  uint64_t last_pred_evals() const { return searcher_->pred_evals(); }
  uint64_t last_struct_visits() const { return searcher_->struct_visits(); }
  const uint64_t* last_chain_hist() const { return searcher_->chain_hist(); }
  uint64_t last_uniq_proxy() const { return searcher_->uniq_proxy_pushes(); }
  float last_result_worst() const { return searcher_->last_result_worst(); }
  void snap_enable(const std::vector<uint32_t>& J) { searcher_->snap_enable(J); }
  void snap_disable() { searcher_->snap_disable(); }
  void set_struct_hops(int h) { searcher_->set_struct_hops(h); }
  void set_two_stage(bool b) { searcher_->set_two_stage(b); }
  uint64_t last_cc_boundary_distanced() const { return searcher_->cc_boundary_distanced(); }
  uint64_t last_cc_phase2_struct() const { return searcher_->cc_phase2_struct(); }
  uint64_t last_cc_phase2_delta() const { return searcher_->cc_phase2_delta(); }
  uint64_t last_cc_regions_touched() const { return searcher_->cc_regions_touched(); }
  uint64_t last_cc_broad_regions() const { return searcher_->cc_broad_regions(); }
  uint64_t last_f9_uq_emplace() const { return searcher_->qstat_uq_emplace(); }
  uint64_t last_f9_fifo_push() const { return searcher_->qstat_fifo_push(); }
  uint64_t last_f9_bnd_dist() const { return searcher_->qstat_bnd_dist(); }
  uint64_t last_f9_broad_bnd_dist() const { return searcher_->qstat_broad_bnd_dist(); }
  uint64_t last_f9_term_bfsq_nonempty() const { return searcher_->qstat_term_bfsq_nonempty(); }
  uint64_t last_f9_phase2_struct() const { return searcher_->qstat_phase2_struct(); }
  uint64_t last_f9_phase2_delta() const { return searcher_->qstat_phase2_delta(); }
  uint64_t last_f9_regions_touched() const { return searcher_->qstat_regions_touched(); }
  uint64_t last_f9_broad_regions() const { return searcher_->qstat_broad_regions(); }
  uint64_t expanded_total() const { return searcher_->expanded(); }

  SearchResult knn_search_connected_domain(const Query& q, const DNFPredicate& phi, int ef,
                             bool enable_es2 = true, bool enable_phase2 = true, int case3_mode = 0) {
    SearchResult r; r.topk_ids = searcher_->knn_search_connected_domain(q.data, phi, ef, K_, enable_es2, enable_phase2, case3_mode);
    r.n_delta = searcher_->delta(); r.n_delta_satisfying = searcher_->delta_satisfying();
    r.n_struct_visits = searcher_->struct_visits(); r.n_pred_evals = searcher_->pred_evals();
    return r;
  }

  SearchResult knn_search_connected_domain_efK(const Query& q, const DNFPredicate& phi, int ef, int K_override,
                             bool enable_es2 = true, bool enable_phase2 = true, int case3_mode = 0) {
    SearchResult r; r.topk_ids = searcher_->knn_search_connected_domain(q.data, phi, ef, K_override, enable_es2, enable_phase2, case3_mode);
    r.n_delta = searcher_->delta(); r.n_delta_satisfying = searcher_->delta_satisfying();
    r.n_struct_visits = searcher_->struct_visits(); r.n_pred_evals = searcher_->pred_evals();
    return r;
  }

  SearchResult knn_search_connected_domain_K(const Query& q, const DNFPredicate& phi, int K_override,
                               bool enable_es2 = true, bool enable_phase2 = true, int case3_mode = 0) {
    SearchResult r; r.topk_ids = searcher_->knn_search_connected_domain(q.data, phi, K_override, K_override, enable_es2, enable_phase2, case3_mode);
    r.n_delta = searcher_->delta(); r.n_delta_satisfying = searcher_->delta_satisfying();
    r.n_struct_visits = searcher_->struct_visits(); r.n_pred_evals = searcher_->pred_evals();
    return r;
  }

  PointId inc_next_id() const { return (PointId)N_; }
  void inc_add(const std::vector<float>& vec, const std::vector<double>& colvals);
  void inc_flush();
  bool inc_update(PointId id, size_t col, double new_val);
  bool inc_delete(PointId id);
  bool inc_rebuild_tree(uint32_t tree_idx) {
    bool grew = stats_->rebuild_tree(rm_, tree_idx); if (grew) rebuild_searcher_(); return grew; }
  uint32_t inc_tree_count() const { return stats_->lmat_tree_count(); }
  RegionHistogramStats& stats_mut() { return *stats_; }

  void set_dnf_degree(double d) { degree_cfg_ = d; searcher_->set_dnf_degree(d); }
  void set_index_cache(std::string p) { index_cache_ = std::move(p); }
  void set_dnf_cedge(double c) { cedge_cfg_ = c; searcher_->set_dnf_cedge(c); }
  void set_dnf_cedge_flood(double c) { cedge_flood_cfg_ = c; searcher_->set_dnf_cedge_flood(c); }
  void set_dnf_lambda_pred(double l) { lambda_pred_cfg_ = l; searcher_->set_dnf_lambda_pred(l); }
  void set_dnf_lambda_tradeoff(double l) { lambda_tradeoff_cfg_ = l; searcher_->set_dnf_lambda_tradeoff(l); }

  void set_sel0_verify(bool b) { sel0_verify_cfg_ = b; searcher_->set_sel0_verify(b); }

  void set_es2aware(int m) { es2aware_cfg_ = m; searcher_->set_es2aware(m); }

  void set_transit_skip(bool b) { transit_skip_cfg_ = b; searcher_->set_transit_skip(b && !hist_staled_); }

  void set_use_presence(bool b) { use_presence_cfg_ = b; searcher_->set_use_presence(b); }

  void set_transparent(bool b) { transparent_cfg_ = b; searcher_->set_transparent(b); }

  void set_membership(bool b) { membership_cfg_ = b; searcher_->set_membership(b); }
  void set_scan_cost(double c) { scan_cost_cfg_ = c; searcher_->set_scan_cost(c); }
  uint64_t last_transp_regions() const { return searcher_->transp_regions(); }
  void set_dnf_trace(bool b) { searcher_->set_dnf_trace(b); }
  const std::vector<OursSearch::DnfTraceRow>& dnf_trace() const { return searcher_->dnf_trace(); }
  double dnf_cedge() const { return searcher_->dnf_cedge(); }
  double dnf_cedge_flood() const { return searcher_->dnf_cedge_flood(); }
  double dnf_degree() const { return searcher_->dnf_degree(); }

  uint64_t sc_seeds_C() const { return searcher_->sc_seeds_C(); }
  uint64_t sc_seeds_B() const { return searcher_->sc_seeds_B(); }
  uint64_t sc_peak_frontier() const { return searcher_->sc_peak_frontier(); }
  uint64_t sc_peak_heapq() const { return searcher_->sc_peak_heapq(); }
  uint64_t case1() const { return searcher_->case1(); }
  uint64_t case2() const { return searcher_->case2(); }
  uint64_t case3() const { return searcher_->case3(); }
  uint64_t pick_first() const { return searcher_->pick_first(); }
  uint64_t cm2_spread() const { return searcher_->cm2_spread(); }
  uint64_t b_regions() const { return searcher_->b_regions(); }
  uint64_t c_regions() const { return searcher_->c_regions(); }
#ifdef HS_TRAVTAX
  uint64_t travtax_decisions() const { return searcher_->travtax_decisions(); }
#endif
  uint64_t es2_fired() const { return searcher_->es2_fired(); }

  void set_force(CcForce f) { searcher_->set_force(f); }
  void set_attr_per_region(bool b) { searcher_->set_attr_per_region(b); }
  void reset_per_region() { searcher_->reset_per_region(); }
  uint64_t per_region_dist(RegionId r) const { return searcher_->per_region_dist(r); }
  uint64_t per_region_pred(RegionId r) const { return searcher_->per_region_pred(r); }
  uint8_t region_routed(RegionId r) const { return searcher_->region_routed(r); }

  void set_worsttl(bool b) { searcher_->set_worsttl(b); }
  void worsttl_clear() { searcher_->worsttl_clear(); }
  const std::vector<std::pair<uint64_t, float>>& worsttl() const { return searcher_->worsttl(); }
  double w_spread_mean(RegionId r) const { return searcher_->w_spread_mean(r); }
  uint32_t w_spread_fin(RegionId r) const { return searcher_->w_spread_fin(r); }
  uint32_t w_spread_inf(RegionId r) const { return searcher_->w_spread_inf(r); }
  double entry_rank_mean(RegionId r) const { return searcher_->entry_rank_mean(r); }
  uint64_t last_expanded() const { return searcher_->expanded(); }

  uint64_t last_expanded_verify() const { return searcher_->expanded_verify(); }
  uint64_t last_sel0_b_find() const { return searcher_->sel0_b_find(); }
  uint64_t last_sel0_b_verify() const { return searcher_->sel0_b_verify(); }
  uint64_t last_sel0_c_verify() const { return searcher_->sel0_c_verify(); }
  bool last_visited(PointId u) const { return searcher_->last_visited(u); }

  bool last_distanced(PointId u) const { return searcher_->last_distanced(u); }
  bool last_considered(PointId u) const { return searcher_->last_considered(u); }
  bool last_svis2(PointId u) const { return searcher_->last_svis2(u); }
  bool last_region_absorbed(RegionId r) const { return searcher_->last_region_absorbed(r); }
  uint64_t last_regions_touched() const { return searcher_->qstat_regions_touched(); }

  bool last_region_touched(RegionId r) const { return searcher_->qstat_region_touched(r); }
  void rtrace_set_trace(bool b) { searcher_->rtrace_set_trace(b); }
  const std::vector<OursSearch::RegionTrace>& rtrace_trace() const { return searcher_->rtrace_trace(); }
  void rtrace_set_sigma0_mask(const std::vector<uint8_t>* m) { searcher_->rtrace_set_sigma0_mask(m); }
  uint64_t last_f11_centroid_calls() const { return searcher_->centroid_calls(); }
  float last_centroid_dist(RegionId r) const { return searcher_->centroid_dist(r); }

  void rtouch_set_trace(bool b) { searcher_->rtouch_set_trace(b); }
  uint32_t rtouch_region_delta(RegionId r) const { return searcher_->rtouch_region_delta(r); }
  uint32_t rtouch_region_cent(RegionId r) const { return searcher_->rtouch_region_cent(r); }
  uint32_t rtouch_region_pred(RegionId r) const { return searcher_->rtouch_region_pred(r); }
  uint32_t rtouch_region_phase2(RegionId r) const { return searcher_->rtouch_region_phase2(r); }

  uint64_t last_f9_descent_delta() const { return searcher_->qstat_descent_delta(); }
  bool last_region_entered(RegionId r) const { return searcher_->qstat_region_touched(r); }
  void b_set_trace(bool b) { searcher_->b_set_trace(b); }
  uint64_t b_cpred_broad() const { return searcher_->b_cpred_broad(); }
  uint64_t b_cdist_broad() const { return searcher_->b_cdist_broad(); }
  uint64_t b_struct_broad() const { return searcher_->b_struct_broad(); }
  uint64_t b_cpred_bf() const { return searcher_->b_cpred_bf(); }
  uint64_t b_cdist_bf() const { return searcher_->b_cdist_bf(); }
  size_t b_cd_count() const { return searcher_->b_cd_count(); }
  const uint32_t* b_cd_pos() const { return searcher_->b_cd_pos(); }
  const float* b_cd_val() const { return searcher_->b_cd_val(); }
  const uint64_t* b_cd_struct() const { return searcher_->b_cd_struct(); }
  uint32_t b_disc_pos(PointId id) const { return searcher_->b_disc_pos(id); }
  bool b_disc_in_flood(PointId id) const { return searcher_->b_disc_in_flood(id); }
  const std::vector<PointId>& b_efset() const { return searcher_->b_efset(); }
  void b4_arm(uint64_t tf, uint64_t tfef, float wk, float wef) { searcher_->b4_arm(tf, tfef, wk, wef); }
  void b4_disarm() { searcher_->b4_disarm(); }
  uint64_t b4_front_tot_tf() const { return searcher_->b4_front_tot_tf(); }
  uint64_t b4_front_ge_tf() const { return searcher_->b4_front_ge_tf(); }
  uint64_t b4_front_tot_tfef() const { return searcher_->b4_front_tot_tfef(); }
  uint64_t b4_front_ge_tfef() const { return searcher_->b4_front_ge_tfef(); }
  float last_result_worst_f9() const { return searcher_->last_result_worst(); }
  double mbar_sigma(double s) const { return cm_->mbar_sigma(s); }
  double rho_hat(RegionId r) const { return cm_->rho_hat(r); }
  double theta_F() const { return cm_->theta_F(); }

  double theta_emp() const { return cm_->theta_emp(); }
  bool gate_use_gplus(double s) const { return cm_->gate_use_gplus(s); }
  bool gt_ready() const { return cm_->gt_ready(); }
  double gt_frac(RegionId r, float w) const { return cm_->gt_frac(r, w); }
  double calib_seconds() const { return calib_seconds_; }
  uint64_t calib_delta() const { return calib_delta_; }
  const std::vector<F0Snapshot>& snap_snapshots() const { return searcher_->snap_snapshots(); }
  uint64_t last_delta_sat() const { return searcher_->delta_satisfying(); }
  double region_sigma_hat(RegionId r, const ColumnPredicate& phi) const { return stats_->sigma_hat(r, phi); }
  const RegionMap& region_map() const { return rm_; }
  const HnswGraph& graph() const { return *index_; }

  const RegionHistogramStats& stats_ref() const { return *stats_; }
  const EstCostModel& cost_model_ref() const { return *cm_; }

  size_t graph_bytes() const { return graph_edges() * sizeof(PointId); }
  size_t vector_bytes() const { return N_ * size_t(dim_) * sizeof(float); }
  size_t region_bytes() const { return 2 * N_ * sizeof(PointId); }
  size_t histogram_bytes() const { return stats_ ? stats_->bytes() : 0; }
  size_t region_size(RegionId r) const { return stats_->region_size(r); }
  region_search_kind route(double sigma, size_t n) const { return cm_->decide(sigma, n); }

  SearchResult search_strategy(const Query& q, const Predicate& phi,
                               const std::vector<region_search_kind>& strat, int ef);
  double build_seconds() const { return build_seconds_; }

  double build_t_graph() const { return build_t_graph_; }
  double build_t_region() const { return build_t_region_; }
  double build_t_calib() const { return build_t_calib_; }
  size_t graph_edges() const { return index_->total_edges(); }
  size_t extra_scalar_bytes() const {
    return 2 * N_ * sizeof(PointId) + (stats_ ? stats_->bytes() : 0);
  }
  size_t index_bytes() const {
    return graph_edges() * sizeof(PointId) + N_ * size_t(dim_) * sizeof(float) +
           extra_scalar_bytes();
  }
  double theta() const { return cm_->theta_repr(); }
  double auto_theta() const { return auto_theta_; }
  double m_bar() const { return cm_->m_bar(); }
  double c_dist() const { return cm_->c_dist(); }
  double c_pred() const { return cm_->c_pred(); }
  void set_theta(double t) { cm_->set_theta(t); }
  void clear_theta_override() { cm_->clear_override(); }
  RegionId n_regions() const { return rm_.n_regions; }

 private:
  void rebuild_searcher_() {
    searcher_ = std::make_unique<OursSearch>(*index_, rm_, *stats_, *cm_);

    if (cm_->c_edge() > 0.0) searcher_->set_dnf_cedge(cm_->c_edge());
    if (cedge_cfg_ >= 0.0)       searcher_->set_dnf_cedge(cedge_cfg_);
    if (cedge_flood_cfg_ >= 0.0) searcher_->set_dnf_cedge_flood(cedge_flood_cfg_);
    if (degree_cfg_ >= 0.0)      searcher_->set_dnf_degree(degree_cfg_);
    searcher_->set_dnf_lambda_pred(lambda_pred_cfg_);
    searcher_->set_dnf_lambda_tradeoff(lambda_tradeoff_cfg_);
    searcher_->set_sel0_verify(sel0_verify_cfg_);
    searcher_->set_es2aware(es2aware_cfg_);
    searcher_->set_transit_skip(transit_skip_cfg_ && !hist_staled_);
    searcher_->set_use_presence(use_presence_cfg_);
    searcher_->set_transparent(transparent_cfg_);
    searcher_->set_membership(membership_cfg_);
    searcher_->set_scan_cost(scan_cost_cfg_);
  }
  bool sel0_verify_cfg_ = true;
  bool transit_skip_cfg_ = true;  bool hist_staled_ = false;
  bool use_presence_cfg_ = true;
  bool transparent_cfg_ = true;
  bool membership_cfg_ = true;
  double scan_cost_cfg_ = 2e-9;
  int es2aware_cfg_ = 1;
  double cedge_cfg_ = -1.0, cedge_flood_cfg_ = -1.0, degree_cfg_ = -1.0;
  double lambda_pred_cfg_ = 1.0, lambda_tradeoff_cfg_ = 1.0;
  std::vector<std::pair<PointId, std::vector<double>>> pending_adds_;

  int K_, region_level_, hist_buckets_;
  bool multilayer_on_ = false;
  int lmat_min_ = 1, lmat_max_ = -1;
  size_t budget_bytes_ = 0;
  double delta_range_ = 0.05, eps_point_ = 0.10;
  double naware_c_ = 1.0;
  double auto_theta_ = 0.0;
  size_t N_ = 0; int dim_ = 0; double build_seconds_ = 0;
  double build_t_graph_ = 0, build_t_region_ = 0, build_t_calib_ = 0;
  double calib_seconds_ = 0; uint64_t calib_delta_ = 0;
  std::unique_ptr<HnswIndex> index_;
  std::string index_cache_;
  RegionMap rm_;
  std::unique_ptr<RegionHistogramStats> stats_;
  std::unique_ptr<EstCostModel> cm_;
  std::unique_ptr<OursSearch> searcher_;
};

}
