#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include "hnsw_stats/hnsw/HnswIndex.hpp"
#include "hnsw_stats/region/ForestRegionBuilder.hpp"
#include "hnsw_stats/region/HubVoronoiRegionBuilder.hpp"

using namespace hnsw_stats;

namespace {
HnswIndex make_index(int n, int dim, unsigned threads) {
  std::mt19937 rng(11);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> base(static_cast<size_t>(n) * dim);
  for (auto& x : base) x = g(rng);
  return HnswIndex(base.data(), n, dim, 16, 200, threads);
}
}

TEST_CASE("region contracts: unique assignment + connectivity", "[region]") {
  const int n = 3000, dim = 8;
  HnswIndex idx = make_index(n, dim, 8);
  HubVoronoiRegionBuilder rb(2);
  RegionMap rm = rb.build(idx);

  REQUIRE(rm.region_of.size() == (size_t)n);
  REQUIRE(rm.parent_of.size() == (size_t)n);
  REQUIRE(rm.n_regions > 1);
  REQUIRE(rm.n_regions < (RegionId)n);

  for (int u = 0; u < n; ++u) {

    REQUIRE(rm.region_of[u] < rm.n_regions);

    REQUIRE(rm.region_of[rm.parent_of[u]] == rm.region_of[u]);
  }

  for (int s = 0; s < n; s += 137) {
    PointId u = (PointId)s;
    const RegionId r = rm.region_of[u];
    int steps = 0;
    while (rm.parent_of[u] != u) {
      u = rm.parent_of[u];
      REQUIRE(rm.region_of[u] == r);
      REQUIRE(++steps <= n);
    }
  }
}

TEST_CASE("forest region contracts: partition + father-chain connectivity", "[region]") {

  const int n = 3000, dim = 8, ell = 2;
  HnswIndex idx = make_index(n, dim, 1);
  ForestRegionBuilder rb(ell);
  RegionMap rm = rb.build(idx);

  REQUIRE(rm.region_of.size() == (size_t)n);
  REQUIRE(rm.parent_of.size() == (size_t)n);
  REQUIRE(rm.n_regions > 1);
  REQUIRE(rm.n_regions < (RegionId)n);

  for (int u = 0; u < n; ++u) {
    const PointId f = rm.parent_of[u];
    REQUIRE(f == idx.father((PointId)u));
    if (f != (PointId)u) REQUIRE(idx.node_level(f) > idx.node_level((PointId)u));
  }

  std::vector<int> pop(rm.n_regions, 0);
  for (int u = 0; u < n; ++u) {
    REQUIRE(rm.region_of[u] < rm.n_regions);
    pop[rm.region_of[u]]++;
  }
  size_t total = 0; for (int c : pop) { REQUIRE(c > 0); total += c; }
  REQUIRE(total == (size_t)n);

  for (int s = 0; s < n; s += 61) {
    PointId u = (PointId)s;
    const RegionId r = rm.region_of[u];
    PointId a = u; int steps = 0;
    while (idx.node_level(a) < ell && rm.parent_of[a] != a) {
      REQUIRE(rm.region_of[a] == r);
      a = rm.parent_of[a];
      REQUIRE(++steps <= n);
    }
    REQUIRE(rm.region_of[a] == r);
  }

  int cross = 0, intra = 0;
  for (int u = 0; u < n; ++u)
    for (PointId v : idx.neighbors((PointId)u))
      (rm.region_of[v] != rm.region_of[(PointId)u] ? cross : intra)++;
  REQUIRE(cross > 0);
  REQUIRE(intra > 0);
}

TEST_CASE("region boundary points exist and granularity grows with ell", "[region]") {
  const int n = 3000, dim = 8;
  HnswIndex idx = make_index(n, dim, 4);

  RegionMap r2 = HubVoronoiRegionBuilder(2).build(idx);
  RegionMap r3 = HubVoronoiRegionBuilder(3).build(idx);

  REQUIRE(r3.n_regions <= r2.n_regions);

  int cross = 0, intra = 0;
  for (int u = 0; u < n; ++u)
    for (PointId v : idx.neighbors((PointId)u))
      (r2.region_of[v] != r2.region_of[u] ? cross : intra)++;
  REQUIRE(cross > 0);
  REQUIRE(intra > 0);
}
