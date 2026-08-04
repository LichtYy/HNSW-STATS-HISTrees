#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include <random>

#include "common/DatasetLoader.hpp"
#include "gt/BruteForceGT.hpp"
#include "metrics/MetricsCollector.hpp"
#include "hnsw_stats/hnsw/HnswIndex.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"
#include "hnsw_stats/search/HnswSearch.hpp"
#include "hnsw_stats/distance/Distance.hpp"

using namespace hnsw_stats;
using Catch::Matchers::WithinAbs;

namespace {

Dataset tiny() {
  Dataset ds;
  ds.id = "tiny";
  ds.N = 5;
  ds.dim = 1;
  ds.base = {0, 1, 2, 3, 4};
  ds.queries = {0};
  return ds;
}
}

TEST_CASE("BruteForceGT: exact filtered top-K + n_satisfying", "[search][gt]") {
  Dataset ds = tiny();
  std::vector<double> col = {1, 1, 0, 1, 1};
  ColumnPredicate eq1("c", col, ColumnPredicate::Form::EQ, 1.0);

  BruteForceGT gt(ds, 3, 1);
  auto res = gt.compute(eq1, "");
  REQUIRE(res.size() == 1);
  const auto& e = res[0];
  REQUIRE(e.n_satisfying == 4);
  REQUIRE(e.ids.size() == 3);
  REQUIRE(e.ids == std::vector<PointId>{0, 1, 3});
  REQUIRE_THAT(e.dists[0], WithinAbs(0.0, 1e-9));
}

TEST_CASE("recall@K: min(K,#sat) denominator + exact match", "[search][recall]") {

  std::vector<uint32_t> gt = {0, 1, 3};
  REQUIRE_THAT(MetricsCollector::recall_at_k({0, 1, 3}, gt, 3, 10), WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(MetricsCollector::recall_at_k({0, 9, 3}, gt, 3, 10), WithinAbs(2.0 / 3, 1e-9));

  std::vector<uint32_t> gt2 = {0, 1};
  REQUIRE_THAT(MetricsCollector::recall_at_k({0, 1}, gt2, 3, 2), WithinAbs(1.0, 1e-9));
  REQUIRE_THAT(MetricsCollector::recall_at_k({0, 7}, gt2, 3, 2), WithinAbs(0.5, 1e-9));

  REQUIRE_THAT(MetricsCollector::recall_at_k({}, {}, 3, 0), WithinAbs(1.0, 1e-9));
}

TEST_CASE("HnswIndex+HnswSearch: base-index ids correct under parallel build",
          "[search][hnsw]") {

  const int n = 2000, dim = 8;
  std::mt19937 rng(7);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> base(static_cast<size_t>(n) * dim);
  for (auto& x : base) x = g(rng);

  HnswIndex idx(base.data(), n, dim, 16, 200, 8);
  HnswSearch searcher(idx);

  int hit = 0, Q = 50;
  for (int q = 0; q < Q; ++q) {
    const float* qv = &base[static_cast<size_t>(q * 37 % n) * dim];

    PointId best = 0; float bestd = 1e30f;
    for (int u = 0; u < n; ++u) {
      const float d = l2_sqr(qv, &base[(size_t)u * dim], dim);
      if (d < bestd) { bestd = d; best = (PointId)u; }
    }
    auto res = searcher.search(qv, 64);
    if (!res.empty() && res[0].second == best) ++hit;

    REQUIRE(idx.vector(best)[0] == base[(size_t)best * dim]);
  }
  REQUIRE(hit >= 48);
}
