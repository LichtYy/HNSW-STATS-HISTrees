#pragma once

#include "method/Method.hpp"
#include "common/DatasetLoader.hpp"
#include <memory>

namespace hnsw_stats {

class DiskannMethod final : public Method {
 public:
  DiskannMethod(int K, int cat_col = 0, unsigned R = 32, unsigned L = 100, float alpha = 1.2f);
  ~DiskannMethod();
  void build(const Dataset& ds, const BuildParams& bp) override;
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;
  void set_index_cache(const std::string& path);
  size_t persistent_bytes() override;
  double build_seconds() const { return build_seconds_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int K_, cat_col_;
  unsigned R_, L_;
  float alpha_;
  double build_seconds_ = 0;
};

}
