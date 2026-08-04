#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/hnsw/HnswGraph.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"

#include <string>
#include <vector>

namespace hnsw_stats {

class DNFPredicate final : public Predicate {
 public:
  using Clause = std::vector<ColumnPredicate>;

  explicit DNFPredicate(std::vector<Clause> clauses, const HnswGraph* graph = nullptr)
      : clauses_(std::move(clauses)), graph_(graph) {}

#ifdef HS_PHYSVAL

  void set_physval_spin(uint64_t s) const { spin_ = s; }
#endif

  bool eval(PointId u) const override {
    if (graph_ && graph_->is_deleted(u)) return false;
    for (const Clause& cj : clauses_) {
      bool all = true;
      for (const ColumnPredicate& a : cj) {
#ifdef HS_PHYSVAL

        for (uint64_t i = 0; i < spin_; ++i) physval_sink_ = physval_sink_ * 6364136223846793005ULL + 1442695040888963407ULL;
#endif
        if (!a.eval(u)) { all = false; break; }
      }
      if (all) return true;
    }
    return false;
  }

  std::string canonical() const override {
    std::string s = "OR(";
    for (size_t j = 0; j < clauses_.size(); ++j) {
      if (j) s += ",";
      s += "AND(";
      for (size_t i = 0; i < clauses_[j].size(); ++i) {
        if (i) s += ",";
        s += clauses_[j][i].canonical();
      }
      s += ")";
    }
    s += ")";
    return s;
  }

  const std::vector<Clause>& clauses() const { return clauses_; }
  size_t num_clauses() const { return clauses_.size(); }

 private:
  std::vector<Clause> clauses_;
  const HnswGraph* graph_ = nullptr;
#ifdef HS_PHYSVAL
  mutable uint64_t spin_ = 0;
  mutable uint64_t physval_sink_ = 0;
#endif
};

}
