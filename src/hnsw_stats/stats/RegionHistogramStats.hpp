#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/hnsw/HnswGraph.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"
#include "hnsw_stats/region/RegionBuilder.hpp"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnsw_stats {

struct ColumnView {
  std::string name;
  bool categorical = false;
  const std::vector<double>* values = nullptr;
};

struct MultiLayerParams {
  int    lmat_min    = 1;
  int    lmat_max    = 4;
  size_t budget_bytes = 0;

  double delta_range = 0.05;
  double naware_c = 0.0;

  double eps_point   = 0.10;
};

class RegionHistogramStats {
 public:

  RegionHistogramStats(const RegionMap& rm, const std::vector<ColumnView>& cols,
                       int buckets = 32);

  RegionHistogramStats(const HnswGraph& g, RegionMap& rm,
                       const std::vector<ColumnView>& cols, int buckets,
                       const MultiLayerParams& ml);

  double sigma_hat(RegionId region, const ColumnPredicate& phi) const;

  size_t region_size(RegionId r) const { return region_size_[r]; }
  RegionId n_regions() const { return n_regions_; }

  bool ml_active() const { return ml_.active; }

  void on_insert(RegionMap& rm, PointId id, const std::vector<double>& colvals);

  bool on_update(RegionMap& rm, PointId id, size_t col, double new_val);

  bool on_delete(RegionMap& rm, PointId id);

  bool rebuild_tree(RegionMap& rm, uint32_t tree_idx);

  RegionId forest_lmat_max_ancestor(PointId v) const;
  uint32_t forest_lmat_max_tree_of(PointId v) const;
  void forest_region_boundary_ancestors(PointId v, std::vector<RegionId>& out) const;
  RegionId bound_regionid_of(PointId v) const {
    return (ml_.active && (size_t)v < ml_.bound_regionid_of.size()) ? ml_.bound_regionid_of[v] : kNoRow; }
  uint32_t lmat_tree_count() const { return (uint32_t)ml_.tree_rootcand.size(); }

  size_t ml_ncols() const { return ml_.col_meta.size(); }
  bool   ml_col_cat(size_t ci) const { return ml_.col_meta[ci].cat; }
  double ml_col_value(size_t ci, PointId p) const { return ml_.col_values[ci][p]; }
  int    ml_col_width(size_t ci) const {
    const auto& m = ml_.col_meta[ci];
    return m.cat ? categorical_.at(m.name).n_codes : numeric_.at(m.name).B; }
  int    ml_bucket(size_t ci, double v) const {
    const auto& m = ml_.col_meta[ci];
    if (m.cat) { auto it = categorical_.at(m.name).code_index.find(v); return it == categorical_.at(m.name).code_index.end() ? -1 : it->second; }
    const auto& c = numeric_.at(m.name); int b = int(std::upper_bound(c.edges.begin(), c.edges.end(), v) - c.edges.begin()) - 1;
    return b < 0 ? 0 : (b >= c.B ? c.B - 1 : b); }
  bool   ml_kept(size_t ci, RegionId node) const {
    const auto& m = ml_.col_meta[ci];
    return m.cat ? (categorical_.at(m.name).row_of[node] != kNoRow) : (numeric_.at(m.name).row_of[node] != kNoRow); }
  uint32_t ml_count(size_t ci, RegionId node, int b) const {
    const auto& m = ml_.col_meta[ci];
    if (m.cat) { const auto& c = categorical_.at(m.name); RegionId r = c.row_of[node]; return r == kNoRow ? 0u : c.counts[(size_t)r * c.n_codes + b]; }
    const auto& c = numeric_.at(m.name); RegionId r = c.row_of[node]; return r == kNoRow ? 0u : c.counts[(size_t)r * c.B + b]; }

  bool presence_on() const { return pres_on_; }

  bool atom_empty(RegionId r, const ColumnPredicate& a) const;

  bool dnf_empty(RegionId r, const std::vector<std::vector<ColumnPredicate>>& clauses) const;

  RegionId region_boundary_cnt() const { return boundary_cnt_; }

  size_t bytes() const {
    size_t b = region_size_.size() * sizeof(uint32_t);
    for (auto& [n, c] : numeric_) {
      b += c.n_kept * (size_t)c.B * sizeof(uint32_t);
      b += c.edges.size() * sizeof(double);
      b += (c.hist_node.size() + c.row_of.size()) * sizeof(RegionId);
    }
    for (auto& [n, c] : categorical_) {
      b += c.n_kept * (size_t)c.n_codes * sizeof(uint32_t);
      b += (c.hist_node.size() + c.row_of.size()) * sizeof(RegionId);
    }
    return b;
  }

 private:
  static constexpr RegionId kNoRow = ~(RegionId)0;

  bool pres_on_ = false;
  std::vector<std::vector<uint64_t>> pres_bits_;
  std::vector<int> pres_stride_;
  std::vector<uint8_t> pres_degraded_;

  std::unordered_map<std::string, int> pres_ci_;
  void pres_build_();
  void pres_set_(size_t ci, RegionId r, int slot) {
    pres_bits_[ci][(size_t)r * pres_stride_[ci] + (slot >> 6)] |= (1ull << (slot & 63));
  }

  struct NumericCol {
    std::vector<double> edges;
    std::vector<RegionId> hist_node;
    std::vector<RegionId> row_of;
    std::vector<uint32_t> counts;
    int B = 0;
    RegionId n_kept = 0;
  };
  struct CategoricalCol {
    std::unordered_map<double, int> code_index;
    std::vector<RegionId> hist_node;
    std::vector<RegionId> row_of;
    std::vector<uint32_t> counts;
    int n_codes = 0;
    RegionId n_kept = 0;
  };

  double cdf(RegionId node, const NumericCol& c, double x) const;

  struct MLState {
    bool active = false;
    const HnswGraph* g = nullptr;
    int lmin = 1, lmax = 4, buckets = 32;
    double delta_range = 0.05, eps_point = 0.10; double naware_c = 0.0;
    std::vector<uint32_t> cand_of;
    std::vector<uint32_t> fca_of;
    std::vector<PointId>  cand_pt;
    std::vector<int>      cand_lv;
    std::vector<uint32_t> parent_cand;
    std::vector<uint32_t> tree_of_cand;
    std::vector<uint32_t> cand_regionid;
    std::vector<uint32_t> tree_rootcand;
    std::vector<RegionId> bound_regionid_of;
    std::vector<RegionId> lmatnode_of_tree;
    std::vector<std::vector<double>> col_values;
    struct ColMeta { std::string name; bool cat; };
    std::vector<ColMeta> col_meta;
  };
  static constexpr uint32_t kInvU = ~(uint32_t)0;

  uint32_t fca_cand_of_(PointId v) const;
  RegionId finest_bound_ancestor_(PointId v) const;
  uint32_t lmat_max_root_cand_(PointId v) const;

  RegionId n_regions_ = 0;
  RegionId boundary_cnt_ = 0;
  std::vector<uint32_t> region_size_;
  std::unordered_map<std::string, NumericCol> numeric_;
  std::unordered_map<std::string, CategoricalCol> categorical_;
  MLState ml_;
};

}
