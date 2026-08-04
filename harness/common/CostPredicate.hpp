#pragma once

#include "hnsw_stats/predicate/Predicate.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace hnsw_stats {

class CostPredicate final : public Predicate {
 public:
  CostPredicate(const Predicate& inner, uint64_t spin) : inner_(inner), spin_(spin) {}
  bool eval(PointId u) const override {
    uint64_t s = sink_;
    for (uint64_t i = 0; i < spin_; ++i) s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    sink_ = s;
    return inner_.eval(u);
  }
  std::string canonical() const override { return inner_.canonical() + "|spin" + std::to_string(spin_); }
 private:
  const Predicate& inner_;
  uint64_t spin_;
  mutable volatile uint64_t sink_ = 0;
};

class LambdaPredicate final : public Predicate {
 public:
  LambdaPredicate(std::function<bool(PointId)> fn, std::string name)
      : fn_(std::move(fn)), name_(std::move(name)) {}
  bool eval(PointId u) const override { return fn_(u); }
  std::string canonical() const override { return name_; }
 private:
  std::function<bool(PointId)> fn_;
  std::string name_;
};

}
