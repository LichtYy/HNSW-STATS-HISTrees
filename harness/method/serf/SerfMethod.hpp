#pragma once

#include "method/Method.hpp"
#include "common/DatasetLoader.hpp"
#include <memory>
#include <string>

namespace hnsw_stats {

class SerfMethod final : public Method {
 public:

  SerfMethod(int K, int range_col = 0, int serf_M = 16, unsigned serf_efc = 100);
  ~SerfMethod();
  void build(const Dataset& ds, const BuildParams& bp) override;
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;
  double build_seconds() const { return build_seconds_; }
  void set_index_cache(std::string p) { index_cache_ = std::move(p); }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int K_, range_col_, serf_M_;
  unsigned serf_efc_;
  double build_seconds_ = 0;
  std::string index_cache_;
};

}
