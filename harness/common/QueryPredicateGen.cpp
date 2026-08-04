#include "common/QueryPredicateGen.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace hnsw_stats {

QuantilePredicates::QuantilePredicates(const ScalarColumn& col)
    : col_(col), sorted_(col.values) {
  std::sort(sorted_.begin(), sorted_.end());
}

double QuantilePredicates::quantile(double p) const {
  if (sorted_.empty()) return 0.0;
  p = std::clamp(p, 0.0, 1.0);
  size_t idx = static_cast<size_t>(p * sorted_.size());
  if (idx >= sorted_.size()) idx = sorted_.size() - 1;
  return sorted_[idx];
}

ColumnPredicate QuantilePredicates::lt(double sel) const {

  if (sel >= 0.999)
    return ColumnPredicate(col_.name, col_.values, ColumnPredicate::Form::LT,
                           quantile(1.0) + 1.0);
  return ColumnPredicate(col_.name, col_.values, ColumnPredicate::Form::LT,
                         quantile(sel));
}

ColumnPredicate QuantilePredicates::gt(double sel) const {

  if (sel >= 0.999)
    return ColumnPredicate(col_.name, col_.values, ColumnPredicate::Form::GT,
                           quantile(0.0) - 1.0);
  return ColumnPredicate(col_.name, col_.values, ColumnPredicate::Form::GT,
                         quantile(1.0 - sel));
}

ColumnPredicate QuantilePredicates::interval(double sel) const {
  const double lo = quantile(0.5 - sel / 2.0);
  const double hi = quantile(0.5 + sel / 2.0);
  return ColumnPredicate(col_.name, col_.values, ColumnPredicate::Form::INTERVAL,
                         lo, hi);
}

std::vector<double> QuantilePredicates::categories() const {
  std::set<double> s(col_.values.begin(), col_.values.end());
  return {s.begin(), s.end()};
}

ColumnPredicate QuantilePredicates::eq(double category) const {
  return ColumnPredicate(col_.name, col_.values, ColumnPredicate::Form::EQ,
                         category);
}

double QuantilePredicates::selectivity(const ColumnPredicate& p) const {
  size_t cnt = 0;
  const size_t N = col_.values.size();
  for (size_t u = 0; u < N; ++u)
    if (p.eval(static_cast<PointId>(u))) ++cnt;
  return N ? static_cast<double>(cnt) / static_cast<double>(N) : 0.0;
}

}
