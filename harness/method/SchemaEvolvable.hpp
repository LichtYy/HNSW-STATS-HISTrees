#pragma once

#include <cstddef>

namespace hnsw_stats {

struct ScalarColumn;

struct AddColumnCost {
  double build_seconds = 0.0;
  size_t extra_bytes = 0;
};

class SchemaEvolvable {
 public:
  virtual ~SchemaEvolvable() = default;
  virtual AddColumnCost add_column(const ScalarColumn&) = 0;
};

}
