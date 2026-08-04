#pragma once

#include "hnsw_stats/Types.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

namespace hnsw_stats {

class Searcher {
 public:
  virtual ~Searcher() = default;

  virtual TopK search(const float* q, const Predicate& phi, int ef, int K) = 0;
};

}
