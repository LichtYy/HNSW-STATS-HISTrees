#pragma once

#include "common/DatasetLoader.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace hnsw_stats {

struct RhoMine {
  int K = 10, n_labels = 0, n_quant = 0;
  size_t N = 0;
  std::vector<std::vector<float>> quant;
  std::vector<std::vector<uint32_t>> lab_gt_ids;
  std::vector<std::vector<float>> lab_gt_dists;
  std::vector<size_t> label_count;

  double rank_frac(size_t q, float d) const;
  void rho_of(size_t q, int l, double& mean, double& mx) const;
};

RhoMine rho_mine(const Dataset& ds, const std::vector<Query>& Q,
                 const std::vector<double>& labels, int n_labels, int K, int n_quant = 1024);

static constexpr uint32_t RHO_CACHE_VERSION = 1;
inline bool save_rho(const RhoMine& m, std::FILE* f) {
  if (!f) return false;
  auto w = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
  auto wvv = [&](const auto& vv) {
    uint64_t no = vv.size(); if (!w(&no, 8)) return false;
    for (const auto& v : vv) { uint64_t ni = v.size(); if (!w(&ni, 8)) return false;
      if (ni && !w(v.data(), ni * sizeof(v[0]))) return false; } return true; };
  const uint32_t ver = RHO_CACHE_VERSION;
  const int32_t hdr[3] = {m.K, m.n_labels, m.n_quant}; const uint64_t Nn = m.N;
  if (!w(&ver, 4) || !w(hdr, sizeof hdr) || !w(&Nn, 8)) return false;
  uint64_t nlc = m.label_count.size();
  if (!w(&nlc, 8) || (nlc && !w(m.label_count.data(), nlc * sizeof(size_t)))) return false;
  return wvv(m.quant) && wvv(m.lab_gt_ids) && wvv(m.lab_gt_dists);
}
inline bool load_rho(RhoMine& m, std::FILE* f) {
  if (!f) return false;
  auto r = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
  auto rvv = [&](auto& vv) {
    uint64_t no = 0; if (!r(&no, 8)) return false; vv.resize(no);
    for (auto& v : vv) { uint64_t ni = 0; if (!r(&ni, 8)) return false; v.resize(ni);
      if (ni && !r(v.data(), ni * sizeof(v[0]))) return false; } return true; };
  uint32_t ver = 0; int32_t hdr[3]; uint64_t Nn = 0;
  if (!r(&ver, 4) || ver != RHO_CACHE_VERSION || !r(hdr, sizeof hdr) || !r(&Nn, 8)) return false;
  m.K = hdr[0]; m.n_labels = hdr[1]; m.n_quant = hdr[2]; m.N = Nn;
  uint64_t nlc = 0; if (!r(&nlc, 8)) return false; m.label_count.resize(nlc);
  if (nlc && !r(m.label_count.data(), nlc * sizeof(size_t))) return false;
  return rvv(m.quant) && rvv(m.lab_gt_ids) && rvv(m.lab_gt_dists);
}

}
