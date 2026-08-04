#pragma once

#include <cstdint>
#include <vector>

namespace hnsw_stats {

struct Dataset;
struct Query;
class Predicate;

struct BuildParams {
  int M = 16;
  int efc = 200;

};

struct SearchResult {
  std::vector<uint32_t> topk_ids;
  uint64_t n_delta = 0;
  uint64_t n_delta_satisfying = 0;

  uint64_t n_struct_visits = 0;
  uint64_t n_pred_evals = 0;
  uint64_t n_probe_evals = 0;
};

class Method {
 public:
  virtual ~Method() = default;
  virtual void build(const Dataset&, const BuildParams&) = 0;
  virtual SearchResult search(const Query&, const Predicate&, int ef) = 0;

  virtual size_t persistent_bytes() { return 0; }
};

}
