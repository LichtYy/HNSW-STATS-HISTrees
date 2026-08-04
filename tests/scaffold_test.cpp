#include <catch2/catch_test_macros.hpp>

#include "hnsw_stats/version.hpp"
#include "distance/CountedDistance.hpp"

TEST_CASE("version is reported", "[scaffold]") {
  REQUIRE(std::string(hnsw_stats::version()) == "0.0.1");
}

TEST_CASE("CountedDistance counts once per call and resets", "[scaffold][delta]") {
  hnsw_stats::CountedDistance d(hnsw_stats::Metric::L2);
  const float a[3] = {0.f, 0.f, 0.f};
  const float b[3] = {3.f, 4.f, 0.f};

  REQUIRE(d.count() == 0);
  const float dist = d(a, b, 3);
  REQUIRE(dist == 25.f);
  REQUIRE(d.count() == 1);
  (void)d(a, b, 3);
  REQUIRE(d.count() == 2);

  d.reset();
  REQUIRE(d.count() == 0);
}
