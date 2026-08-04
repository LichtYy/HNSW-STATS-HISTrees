#pragma once

#include "method/Method.hpp"
#include "common/DatasetLoader.hpp"
#include <memory>
#include <string>

namespace hnsw_stats {

class NhqMethod final : public Method {
 public:
  NhqMethod(int K, int cat_col = 0, float weight_search = 1e7f);
  ~NhqMethod();
  void build(const Dataset& ds, const BuildParams& bp) override;
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;
  size_t persistent_bytes() override;
  double build_seconds() const { return build_seconds_; }
  void set_index_cache(std::string p) { index_cache_ = std::move(p); }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int K_, cat_col_;
  float weight_search_;
  double build_seconds_ = 0;
  std::string index_cache_;
};

}
