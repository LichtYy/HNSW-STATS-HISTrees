#pragma once

#include "method/Method.hpp"
#include "common/DatasetLoader.hpp"
#include <memory>

namespace hnsw_stats {

class AcornOrigMethod final : public Method {
 public:

  AcornOrigMethod(int K, int attr_col = 0, int gamma = 12, int m_beta_mult = 2, int threads = 1);
  ~AcornOrigMethod();
  void build(const Dataset& ds, const BuildParams& bp) override;
  SearchResult search(const Query& q, const Predicate& phi, int ef) override;
  size_t persistent_bytes() override;
  double build_seconds() const { return build_seconds_; }
  int gamma() const { return gamma_; }
  int m_beta() const { return m_beta_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int K_, attr_col_, gamma_, m_beta_mult_, threads_;
  int m_beta_ = 0;
  double build_seconds_ = 0;
};

}
