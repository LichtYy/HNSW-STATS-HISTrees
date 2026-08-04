#pragma once

#include "hnsw_stats/cost_model/CostModel.hpp"
#include "hnsw_stats/hnsw/HnswGraph.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace hnsw_stats {

class EstCostModel {
 public:

  void calibrate(const HnswGraph& g, const std::vector<RegionId>& region_of,
                 const Predicate& phi, int dist_samples = 100000,
                 int bf_searches = 200, int bf_ef = 50, uint64_t seed = 7);

  CostConstants constants() const { return {c_dist_, c_pred_}; }
  double c_dist() const { return c_dist_; }

  double c_edge() const { return c_edge_; }

  double c_pred() const { return cpred_override_ >= 0.0 ? cpred_override_ : c_pred_; }
  void set_cpred_override(double v) { cpred_override_ = v; }
  void clear_cpred_override() { cpred_override_ = -1.0; }
  double m_bar() const { return m_bar_; }

  double theta(size_t n_region) const {
    if (theta_override_ >= 0.0) return theta_override_;
    if (n_region == 0) return 0.0;
    const double mbn = m_bar_ / double(n_region);
    const double r = (cpred_ratio_ >= 0.0) ? cpred_ratio_
                                            : (c_dist_ > 0 ? c_pred_ / c_dist_ : 0.0);

    double t = r3_ ? (mbn + (mbn - 1.0) * r) : (mbn - r);
    return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  }
  void set_r3(bool b) { r3_ = b; }
  void set_cpred_ratio(double r) { cpred_ratio_ = r; }
  void clear_cpred_ratio() { cpred_ratio_ = -1.0; }

  double theta_repr() const { return theta(repr_n_); }
  void set_repr_region_size(size_t n) { repr_n_ = n ? n : 1; }

  void set_theta(double t) { theta_override_ = t; }
  void clear_override() { theta_override_ = -1.0; }

  region_search_kind decide(double sigma_hat, size_t n_region) const {
    return sigma_hat < theta(n_region) ? region_search_kind::FLOOD : region_search_kind::BESTFIRST;
  }

  void set_mbar_buckets(const double b[5]) { for (int i = 0; i < 5; ++i) mbar_b_[i] = b[i]; mbar_ready_ = true; }
  double mbar_sigma(double sigma) const {
    if (!mbar_ready_) return m_bar_ > 0 ? m_bar_ : 5.0;
    int i = sigma < 0.01 ? 0 : sigma < 0.05 ? 1 : sigma < 0.10 ? 2 : sigma < 0.30 ? 3 : 4;
    return mbar_b_[i] > 0 ? mbar_b_[i] : (m_bar_ > 0 ? m_bar_ : 5.0);
  }

  void set_rho_hat(std::vector<double> r) { rho_hat_ = std::move(r); }
  double rho_hat(RegionId r) const { return (r < rho_hat_.size() && rho_hat_[r] > 0) ? rho_hat_[r] : 1.0; }

  void set_centroids(std::vector<float> c, int d) { centroids_ = std::move(c); cdim_ = d; }
  const float* centroid(RegionId r) const { return centroids_.empty() ? nullptr : &centroids_[(size_t)r * cdim_]; }
  int centroid_dim() const { return cdim_; }

  void set_gtraverse(std::vector<float> frac, std::vector<float> wedges, int nb) {
    gtfrac_ = std::move(frac); gt_wedges_ = std::move(wedges); gt_nb_ = nb;
  }
  bool gt_ready() const { return gt_nb_ > 0 && !gtfrac_.empty(); }
  double gt_frac(RegionId r, float w) const {
    if (gt_nb_ <= 0) return 1.0;
    int b = 0; while (b < gt_nb_ - 1 && w >= gt_wedges_[b + 1]) ++b;
    return gtfrac_[(size_t)r * gt_nb_ + b];
  }

  void set_theta_emp(double t) { theta_emp_ = t; }
  double theta_emp() const { return theta_emp_ >= 0 ? theta_emp_ : 0.5; }

  void set_gate(const uint8_t g[5]) { for (int i = 0; i < 5; ++i) gate_use_gplus_[i] = g[i]; gate_ready_ = true; }

  static constexpr uint32_t CALIB_CACHE_VERSION = 2;
  bool save_calib(std::FILE* f) const {
    if (!f) return false;
    auto w = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    const uint32_t ver = CALIB_CACHE_VERSION;
    if (!w(&ver, 4)) return false;

    const double cc[4] = {c_dist_, c_pred_, c_edge_, m_bar_};
    if (!w(cc, sizeof cc)) return false;
    const uint8_t mbr = mbar_ready_ ? 1 : 0, gtr = gate_ready_ ? 1 : 0;
    if (!w(&mbr, 1) || !w(mbar_b_, sizeof mbar_b_)) return false;
    if (!w(&theta_emp_, sizeof theta_emp_)) return false;
    if (!w(&gtr, 1) || !w(gate_use_gplus_, sizeof gate_use_gplus_)) return false;
    const int32_t nb = gt_nb_; const uint64_t nw = gt_wedges_.size(), nf = gtfrac_.size();
    if (!w(&nb, 4) || !w(&nw, 8) || !w(&nf, 8)) return false;
    if (nw && !w(gt_wedges_.data(), nw * sizeof(float))) return false;
    if (nf && !w(gtfrac_.data(), nf * sizeof(float))) return false;
    return true;
  }

  bool load_calib(std::FILE* f) {
    if (!f) return false;
    auto r = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    uint32_t ver = 0; if (!r(&ver, 4) || ver != CALIB_CACHE_VERSION) return false;
    double cc[4]; if (!r(cc, sizeof cc)) return false;
    uint8_t mbr = 0, gtr = 0; double mb[5]; double th = 0; uint8_t gate[5];
    if (!r(&mbr, 1) || !r(mb, sizeof mb)) return false;
    if (!r(&th, sizeof th)) return false;
    if (!r(&gtr, 1) || !r(gate, sizeof gate)) return false;
    int32_t nb = 0; uint64_t nw = 0, nf = 0;
    if (!r(&nb, 4) || !r(&nw, 8) || !r(&nf, 8)) return false;
    std::vector<float> wedges(nw), frac(nf);
    if (nw && !r(wedges.data(), nw * sizeof(float))) return false;
    if (nf && !r(frac.data(), nf * sizeof(float))) return false;

    c_dist_ = cc[0]; c_pred_ = cc[1]; c_edge_ = cc[2]; m_bar_ = cc[3];
    for (int i = 0; i < 5; ++i) mbar_b_[i] = mb[i]; mbar_ready_ = mbr != 0;
    theta_emp_ = th;
    for (int i = 0; i < 5; ++i) gate_use_gplus_[i] = gate[i]; gate_ready_ = gtr != 0;
    gtfrac_ = std::move(frac); gt_wedges_ = std::move(wedges); gt_nb_ = nb;
    return true;
  }
  bool gate_use_gplus(double sigma) const {
    if (!gate_ready_) return false;
    int i = sigma < 0.01 ? 0 : sigma < 0.05 ? 1 : sigma < 0.10 ? 2 : sigma < 0.30 ? 3 : 4;
    return gate_use_gplus_[i] != 0;
  }

  double theta_F() const {
    const double r = (cpred_ratio_ >= 0.0) ? cpred_ratio_ : (c_dist_ > 0 ? c_pred_ / c_dist_ : 0.0);
    double t = 1.0 - r; return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
  }
  double cpred_over_cdist() const {
    return (cpred_ratio_ >= 0.0) ? cpred_ratio_ : (c_dist_ > 0 ? c_pred_ / c_dist_ : 0.0);
  }

 private:
  double c_dist_ = 1.0, c_pred_ = 0.0, m_bar_ = 0.0, c_edge_ = 0.0;
  double theta_override_ = -1.0;
  double cpred_ratio_ = -1.0;
  double cpred_override_ = -1.0;
  bool r3_ = false;
  size_t repr_n_ = 1;
  double mbar_b_[5] = {0, 0, 0, 0, 0};
  bool mbar_ready_ = false;
  std::vector<double> rho_hat_;
  std::vector<float> centroids_;
  int cdim_ = 0;

  std::vector<float> gtfrac_; std::vector<float> gt_wedges_; int gt_nb_ = 0;
  double theta_emp_ = -1.0;
  uint8_t gate_use_gplus_[5] = {0,0,0,0,0}; bool gate_ready_ = false;
};

}
