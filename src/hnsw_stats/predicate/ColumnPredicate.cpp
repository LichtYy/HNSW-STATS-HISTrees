#include "hnsw_stats/predicate/ColumnPredicate.hpp"

#include <cstdio>

namespace hnsw_stats {

volatile uint64_t g_colpred_spin = 0;

std::string ColumnPredicate::canonical() const {
  char buf[128];
  if (text_bytes_) {
    std::snprintf(buf, sizeof buf, "contains_%s_%s", column_.c_str(), pattern_.c_str());
    return buf;
  }
  switch (form_) {
    case Form::LT:
      std::snprintf(buf, sizeof buf, "%s<%.6g", column_.c_str(), a_);
      break;
    case Form::GT:
      std::snprintf(buf, sizeof buf, "%s>%.6g", column_.c_str(), a_);
      break;
    case Form::INTERVAL:
      std::snprintf(buf, sizeof buf, "%.6g<%s<%.6g", a_, column_.c_str(), b_);
      break;
    case Form::EQ:
      std::snprintf(buf, sizeof buf, "%s=%.6g", column_.c_str(), a_);
      break;
  }
  return buf;
}

}
