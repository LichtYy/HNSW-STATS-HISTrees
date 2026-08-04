#pragma once

#include "common/DatasetLoader.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

#include <string>
#include <vector>

namespace hnsw_stats {

struct GTEntry {
  std::vector<PointId> ids;
  std::vector<float>   dists;
  int n_satisfying = 0;
};

class BruteForceGT {
 public:
  BruteForceGT(const Dataset& ds, int K_max, unsigned threads)
      : ds_(ds), K_max_(K_max), threads_(threads ? threads : 1) {}

  std::vector<GTEntry> compute(const Predicate& phi, const std::string& cache_dir);

 private:
  std::vector<GTEntry> compute_fresh(const Predicate& phi) const;
  std::string cache_path(const std::string& cache_dir, const Predicate& phi) const;

  const Dataset& ds_;
  int K_max_;
  unsigned threads_;
};

}
