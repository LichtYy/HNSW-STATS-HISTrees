#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hnsw_stats {

struct VecsData {
  std::vector<float> data;
  size_t n = 0;
  int dim = 0;
};

VecsData read_fvecs(const std::string& path, size_t max_rows = 0);

struct IVecsData {
  std::vector<int32_t> data;
  size_t n = 0;
  int dim = 0;
};
IVecsData read_ivecs(const std::string& path, size_t max_rows = 0);

}
