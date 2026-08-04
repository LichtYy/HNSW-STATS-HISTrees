#pragma once

#include "hnsw_stats/predicate/Predicate.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hnsw_stats {

extern volatile uint64_t g_colpred_spin;

class ColumnPredicate final : public Predicate {
 public:
  enum class Form { LT, GT, INTERVAL, EQ };

  ColumnPredicate(std::string column, const std::vector<double>& values,
                  Form form, double a, double b = 0.0)
      : column_(std::move(column)), values_(values), form_(form), a_(a), b_(b) {}

  static ColumnPredicate str_contains(std::string proxy_column, const std::vector<double>& proxy_values,
                                      double proxy_code, const char* text_bytes,
                                      const uint64_t* text_offsets, std::string pattern) {
    ColumnPredicate p(std::move(proxy_column), proxy_values, Form::EQ, proxy_code);
    p.text_bytes_ = text_bytes; p.text_offsets_ = text_offsets; p.pattern_ = std::move(pattern);
    return p;
  }

  bool eval(PointId u) const override {
    if (g_colpred_spin) {
      static volatile uint64_t sink = 0; uint64_t s = sink;
      for (uint64_t i = 0; i < g_colpred_spin; ++i) s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      sink = s;
    }
    if (text_bytes_) {
      const char* b = text_bytes_ + text_offsets_[u];
      const size_t len = (size_t)(text_offsets_[u + 1] - text_offsets_[u]);
      if (pattern_.empty() || len < pattern_.size()) return false;
      const char c0 = pattern_[0];
      const size_t m = pattern_.size(), end = len - m;
      for (size_t i = 0; i <= end; ++i)
        if (b[i] == c0 && std::memcmp(b + i, pattern_.data(), m) == 0) return true;
      return false;
    }
    const double v = values_[u];
    switch (form_) {
      case Form::LT:       return v < a_;
      case Form::GT:       return v > a_;
      case Form::INTERVAL: return v > a_ && v < b_;
      case Form::EQ:       return v == a_;
    }
    return false;
  }

  std::string canonical() const override;

  Form form() const { return form_; }
  const std::string& column() const { return column_; }
  double a() const { return a_; }
  double b() const { return b_; }
  size_t size() const { return values_.size(); }
  bool is_text() const { return text_bytes_ != nullptr; }
  const std::string& pattern() const { return pattern_; }

 private:
  std::string column_;
  const std::vector<double>& values_;
  Form form_;
  double a_, b_;

  const char* text_bytes_ = nullptr;
  const uint64_t* text_offsets_ = nullptr;
  std::string pattern_;
};

}
