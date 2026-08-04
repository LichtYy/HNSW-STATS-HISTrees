#pragma once

#include "hnsw_stats/Types.hpp"
#include <span>

namespace hnsw_stats {

class HnswGraph {
 public:
  virtual ~HnswGraph() = default;

  virtual PointId entry_point() const = 0;
  virtual int max_level() const = 0;
  virtual int node_level(PointId u) const = 0;

  virtual PointId father(PointId u) const { return u; }

  virtual std::span<const PointId> neighbors(PointId u) const = 0;

  virtual std::span<const PointId> neighbors_at(PointId u, int level) const = 0;

  virtual size_t size() const = 0;
  virtual int dim() const = 0;
  virtual const float* vector(PointId u) const = 0;

  virtual bool is_deleted(PointId u) const { return false; }

};

}
