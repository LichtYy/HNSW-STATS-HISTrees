#pragma once

#include "common/DatasetLoader.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"

#include <string>
#include <vector>

namespace hnsw_stats {

inline constexpr double kSelectivityBands[] = {
    0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75};

class QuantilePredicates {
 public:
  explicit QuantilePredicates(const ScalarColumn& col);

  ColumnPredicate lt(double sel) const;
  ColumnPredicate gt(double sel) const;
  ColumnPredicate interval(double sel) const;

  std::vector<double> categories() const;
  ColumnPredicate eq(double category) const;

  double selectivity(const ColumnPredicate& p) const;

  const std::string& name() const { return col_.name; }
  bool orderable() const { return col_.orderable; }
  bool categorical() const { return col_.type == "categorical"; }

  double quantile(double p) const;

 private:
  const ScalarColumn& col_;
  std::vector<double> sorted_;
};

}
