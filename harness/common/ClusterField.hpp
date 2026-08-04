#pragma once

#include "common/DatasetLoader.hpp"

#include <cstdint>
#include <vector>

namespace hnsw_stats {

struct ClusterFieldParams {
  int C = 256;
  double gamma = 0.9;
  uint64_t seed = 4242;
  int n_labels = 64;
  size_t text_len = 1000;
  size_t text_len_s = 500;
  size_t fit_sample = 20000;
  int iters = 8;
  bool compact = true;

};

void append_cluster_field(Dataset& ds, const ClusterFieldParams& p);

inline void cf_quad_words(uint32_t c, int out[4]) {
  out[0] =      (int)(c % 6);
  out[1] = 6  + (int)((c / 6) % 6);
  out[2] = 12 + (int)((c / 36) % 6);
  out[3] = 18 + (int)((c / 216) % 6);
}

std::vector<uint32_t> judge_partition(const Dataset& ds, int C, uint64_t seed);

}
