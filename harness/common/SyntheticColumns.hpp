#pragma once

#include "common/DatasetLoader.hpp"
#include <cstdint>
#include <vector>

namespace hnsw_stats {

std::vector<ScalarColumn> generate_synthetic_columns(const Dataset& ds,
                                                     uint64_t seed = 1234);

void append_extra_synthetic_columns(Dataset& ds, uint64_t seed = 5678);

}
