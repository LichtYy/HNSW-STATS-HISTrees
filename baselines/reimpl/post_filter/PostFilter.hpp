#pragma once

#include "method/Method.hpp"

#include <memory>

#include "common/DatasetLoader.hpp"
#include <string>
#include "hnsw_stats/hnsw/HnswIndex.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"
#include "hnsw_stats/search/HnswSearch.hpp"

namespace hnsw_stats {

class PostFilter final : public Method {
 public:
  explicit PostFilter(int K) : K_(K) {}

  void build(const Dataset& ds, const BuildParams& bp) override;
  void set_index_cache(std::string p) { index_cache_ = std::move(p); }
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;

  double build_seconds() const { return build_seconds_; }
  size_t graph_edges() const { return index_->total_edges(); }
  size_t index_bytes() const {
    return graph_edges() * sizeof(uint32_t) + N_ * size_t(dim_) * sizeof(float);
  }
  const HnswGraph& graph() const { return *index_; }

 private:
  int K_;
  int dim_ = 0;
  size_t N_ = 0; double build_seconds_ = 0;
  std::string index_cache_;
  std::unique_ptr<HnswIndex> index_;
  std::unique_ptr<HnswSearch> searcher_;
};

}
