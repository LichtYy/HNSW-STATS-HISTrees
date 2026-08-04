#pragma once

#include "hnsw_stats/Types.hpp"

#include <string>

namespace hnsw_stats {

class Predicate {
 public:
  virtual ~Predicate() = default;

  virtual bool eval(PointId u) const = 0;

  virtual std::string canonical() const = 0;

};

}
