#pragma once

#include "method/Method.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "common/DatasetLoader.hpp"
#include <string>
#include "hnsw_stats/hnsw/HnswIndex.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

namespace hnsw_stats {

class NavixMethod final : public Method {
 public:

  NavixMethod(int K, double ub_onehop_s = 0.5, double leniency = 3.0)
      : K_(K), ub_onehop_s_(ub_onehop_s), leniency_(leniency) {}

  void build(const Dataset& ds, const BuildParams& bp) override;
  void set_index_cache(std::string p) { index_cache_ = std::move(p); }
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;

  size_t graph_edges() const { return base_->total_edges(); }
  size_t index_bytes() const {
    return graph_edges() * sizeof(uint32_t) + N_ * size_t(dim_) * sizeof(float);
  }
  double build_seconds() const { return build_seconds_; }

  uint64_t n_onehop() const { return n_onehop_; }
  uint64_t n_directed() const { return n_directed_; }
  uint64_t n_blind() const { return n_blind_; }

 private:
  float dist(const float* q, PointId u);

  int K_;
  double ub_onehop_s_, leniency_;
  int dim_ = 0, deg_ = 32;
  size_t N_ = 0;
  double build_seconds_ = 0;
  std::string index_cache_;
  std::unique_ptr<HnswIndex> base_;
  std::vector<uint8_t> mask_;
  std::vector<uint32_t> stamp_, added_, svis_;
  std::vector<float> cache_;
  uint32_t epoch_ = 0;
  uint64_t delta_ = 0, delta_sat_ = 0, struct_ = 0, pred_ = 0;
  uint64_t n_onehop_ = 0, n_directed_ = 0, n_blind_ = 0;
};

}
