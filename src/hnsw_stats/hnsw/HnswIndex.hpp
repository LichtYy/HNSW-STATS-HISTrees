#pragma once

#include "hnsw_stats/hnsw/HnswGraph.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace hnswlib {
template <typename T> class HierarchicalNSW;
template <typename T> class SpaceInterface;
}

namespace hnsw_stats {

class HnswIndex final : public HnswGraph {
 public:

  HnswIndex(const float* base, size_t n, int dim, int M, int efc,
            unsigned threads = 1);

  HnswIndex(const std::string& path, int dim);
  void save(const std::string& path) const;
  ~HnswIndex() override;

  PointId entry_point() const override { return entry_base_; }
  int max_level() const override { return max_level_; }
  int node_level(PointId u) const override { return node_level_[u]; }
  PointId father(PointId u) const override { return father_base_[u]; }
  std::span<const PointId> neighbors(PointId u) const override;
  std::span<const PointId> neighbors_at(PointId u, int level) const override;
  size_t size() const override { return n_; }
  int dim() const override { return dim_; }
  const float* vector(PointId u) const override;

  size_t total_edges() const { return edges_total_; }
  size_t layer0_edges() const { return nbr0_.size(); }

  void add(const float* vec, PointId id);
  void remove(PointId id);
  void reproject();
  bool dirty() const { return dirty_; }
  bool is_deleted(PointId u) const override;

 private:
  void project();

  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_;
  size_t n_;
  int dim_;
  size_t cap_ = 0;
  bool dirty_ = false;
  int max_level_ = 0;
  PointId entry_base_ = 0;
  size_t edges_total_ = 0;

  std::vector<PointId> base_to_internal_;
  std::vector<int> node_level_;
  std::vector<PointId> father_base_;

  std::vector<size_t> off0_;
  std::vector<PointId> nbr0_;

  std::vector<std::unordered_map<PointId, std::vector<PointId>>> upper_;
};

}
