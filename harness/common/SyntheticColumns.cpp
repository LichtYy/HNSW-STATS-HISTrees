#include "common/SyntheticColumns.hpp"

#include <cmath>
#include <random>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

std::vector<ScalarColumn> generate_synthetic_columns(const Dataset& ds,
                                                     uint64_t seed) {
  const size_t N = ds.N;
  const int dim = ds.dim;
  std::mt19937_64 rng(seed);

  ScalarColumn u_uniform{"u_uniform", "numeric", true, {}};
  ScalarColumn u_gauss{"u_gauss", "numeric", true, {}};
  ScalarColumn c_cat10{"c_cat10", "categorical", false, {}};
  ScalarColumn anchor_dist{"anchor_dist", "numeric", true, {}};
  u_uniform.values.resize(N);
  u_gauss.values.resize(N);
  c_cat10.values.resize(N);
  anchor_dist.values.resize(N);

  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::normal_distribution<double> norm(0.0, 1.0);
  std::uniform_int_distribution<int> cat(0, 9);

  for (size_t i = 0; i < N; ++i) {
    u_uniform.values[i] = unif(rng);
    u_gauss.values[i] = norm(rng);
    c_cat10.values[i] = static_cast<double>(cat(rng));
  }

  std::vector<double> centroid(dim, 0.0);
  for (size_t i = 0; i < N; ++i)
    for (int j = 0; j < dim; ++j) centroid[j] += ds.base[i * dim + j];
  std::vector<float> centroid_f(dim);
  for (int j = 0; j < dim; ++j) centroid_f[j] = static_cast<float>(centroid[j] / N);
  for (size_t i = 0; i < N; ++i)
    anchor_dist.values[i] = l2_sqr(&ds.base[i * dim], centroid_f.data(), dim);

  return {std::move(u_uniform), std::move(u_gauss), std::move(c_cat10),
          std::move(anchor_dist)};
}

}

namespace hnsw_stats {

void append_extra_synthetic_columns(Dataset& ds, uint64_t seed) {
  const size_t N = ds.N;
  auto mk = [&](const char* name, uint64_t salt, auto gen) {
    ScalarColumn c; c.name = name; c.type = "numeric"; c.orderable = true;
    std::mt19937_64 rng(seed * 1000003ull + salt);
    c.values.resize(N);
    for (size_t i = 0; i < N; ++i) c.values[i] = gen(rng);
    ds.columns.push_back(std::move(c));
  };
  std::uniform_real_distribution<double> U(0.0, 1.0);
  std::normal_distribution<double> G(0.0, 1.0);
  std::exponential_distribution<double> E(1.0);
  mk("u_zipf", 1, [&](std::mt19937_64& r) {
    double u = U(r); if (u < 1e-12) u = 1e-12;
    return std::pow(u, -1.0 / 1.16);
  });
  mk("u_bimodal", 2, [&](std::mt19937_64& r) {
    return (U(r) < 0.5 ? -2.0 : 2.0) + 0.5 * G(r);
  });
  mk("u_uni2", 3, [&](std::mt19937_64& r) { return U(r); });
  mk("u_gauss2", 4, [&](std::mt19937_64& r) { return G(r); });
  mk("u_exp", 5, [&](std::mt19937_64& r) { return E(r); });
  mk("u_uni3", 6, [&](std::mt19937_64& r) { return U(r); });
}

}
