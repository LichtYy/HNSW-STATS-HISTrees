#pragma once

#include "common/DatasetLoader.hpp"

#include <string>
#include <vector>

namespace hnsw_stats {

struct ColumnSpec {
  std::string field;
  std::string type;
};

std::vector<ScalarColumn> load_jsonl_columns(const std::string& path,
                                             const std::vector<ColumnSpec>& specs,
                                             size_t max_rows, unsigned threads);

std::vector<ScalarColumn> load_csv_columns(const std::string& path,
                                           const std::vector<ColumnSpec>& specs,
                                           size_t max_rows, unsigned threads);

}
