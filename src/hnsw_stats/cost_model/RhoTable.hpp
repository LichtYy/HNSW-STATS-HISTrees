#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"
#include "hnsw_stats/region/RegionBuilder.hpp"
#include "hnsw_stats/stats/RegionHistogramStats.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnsw_stats {

class RhoTable {
 public:
  RhoTable(const RegionMap& rm, const std::vector<ColumnView>& cols, int buckets = 32) {
    n_regions_ = rm.n_regions;
    const size_t N = rm.region_of.size();
    for (const auto& col : cols) {
      const std::vector<double>& v = *col.values;
      if (col.categorical) {
        Cat cc;
        for (double x : v) cc.code_index.emplace(x, 0);
        int idx = 0;
        for (auto& [code, slot] : cc.code_index) slot = idx++;
        cc.n_codes = idx;
        cc.counts.assign((size_t)n_regions_ * cc.n_codes, 0);
        for (size_t u = 0; u < N; ++u)
          cc.counts[(size_t)rm.region_of[u] * cc.n_codes + cc.code_index[v[u]]]++;
        cat_.emplace(col.name, std::move(cc));
      } else {
        Num nc;
        nc.B = std::max(1, buckets);
        std::vector<double> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        nc.edges.resize(nc.B + 1);
        for (int i = 0; i < nc.B; ++i) nc.edges[i] = sorted[(size_t)i * sorted.size() / nc.B];
        nc.edges[nc.B] = sorted.back() + 1.0;
        for (int i = 1; i <= nc.B; ++i)
          if (nc.edges[i] <= nc.edges[i - 1]) nc.edges[i] = nc.edges[i - 1];
        nc.counts.assign((size_t)n_regions_ * nc.B, 0);
        for (size_t u = 0; u < N; ++u) {
          int b = int(std::upper_bound(nc.edges.begin(), nc.edges.end(), v[u]) - nc.edges.begin()) - 1;
          b = std::clamp(b, 0, nc.B - 1);
          nc.counts[(size_t)rm.region_of[u] * nc.B + b]++;
        }
        num_.emplace(col.name, std::move(nc));
      }
    }
  }

  double rho(const ColumnPredicate& p) const {
    using F = ColumnPredicate::Form;
    std::vector<double> cr((size_t)n_regions_, 0.0);
    if (p.form() == F::EQ) {
      auto it = cat_.find(p.column());
      if (it == cat_.end()) return 0.0;
      const Cat& cc = it->second;
      auto ci = cc.code_index.find(p.a());
      if (ci == cc.code_index.end()) return 0.0;
      for (RegionId r = 0; r < n_regions_; ++r)
        cr[r] = cc.counts[(size_t)r * cc.n_codes + ci->second];
    } else {
      auto it = num_.find(p.column());
      if (it == num_.end()) return 0.0;
      const Num& c = it->second;
      for (int b = 0; b < c.B; ++b) {
        const double lo = c.edges[b], hi = c.edges[b + 1];
        bool covered = false;
        switch (p.form()) {
          case F::LT:       covered = (lo < p.a()); break;
          case F::GT:       covered = (hi > p.a()); break;
          case F::INTERVAL: covered = (hi > p.a() && lo < p.b()); break;
          default: break;
        }
        if (covered)
          for (RegionId r = 0; r < n_regions_; ++r)
            cr[r] += c.counts[(size_t)r * c.B + b];
      }
    }
    return concentration_(cr);
  }

 private:
  struct Num { std::vector<double> edges; std::vector<uint32_t> counts; int B = 0; };
  struct Cat { std::unordered_map<double, int> code_index; std::vector<uint32_t> counts; int n_codes = 0; };

  double concentration_(const std::vector<double>& cr) const {
    double T = 0; for (double x : cr) T += x;
    if (T <= 0 || n_regions_ <= 1) return 0.0;
    double H = 0; for (double x : cr) { const double f = x / T; H += f * f; }
    const double inv = 1.0 / n_regions_;
    double rho = (H - inv) / (1.0 - inv);
    return std::clamp(rho, 0.0, 1.0);
  }

  RegionId n_regions_ = 0;
  std::unordered_map<std::string, Num> num_;
  std::unordered_map<std::string, Cat> cat_;
};

}
