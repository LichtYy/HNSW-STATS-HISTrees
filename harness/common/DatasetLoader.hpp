#pragma once

#include "hnsw_stats/Types.hpp"
#include <string>
#include <vector>

namespace hnsw_stats {

struct ScalarColumn {
  std::string name;
  std::string type;
  bool orderable = false;
  std::vector<double> values;
};

struct Query {
  const float* data = nullptr;
};

struct TextColumn {
  std::string name;
  std::vector<uint64_t> offsets;
  std::vector<char> bytes;
};

struct Dataset {
  size_t N = 0;
  int dim = 0;
  std::vector<float> base;
  std::vector<float> queries;
  std::vector<ScalarColumn> columns;
  std::vector<TextColumn> text_columns;
  std::string id;
};

class DatasetLoader {
 public:

  static Dataset load(const std::string& dataset_id, size_t max_base = 0,
                      size_t max_queries = 0);

  static std::string repo_root();
};

}
