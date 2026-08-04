#include "hnsw_stats/hnsw/HnswIndex.hpp"

#include "hnsw_stats/hnsw/vendor/hnswlib/hnswlib.h"

#include <atomic>
#include <thread>
#include <vector>

namespace hnsw_stats {

HnswIndex::HnswIndex(const float* base, size_t n, int dim, int M, int efc,
                     unsigned threads)
    : n_(n), dim_(dim) {
  space_ = std::make_unique<hnswlib::L2Space>(dim);
  hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), n, M, efc);
  cap_ = n;
  if (n == 0) return;

  hnsw_->addPoint(base, 0);
  const unsigned T = std::max(1u, std::min<unsigned>(threads, (unsigned)n));
  if (T == 1) {
    for (size_t i = 1; i < n; ++i) hnsw_->addPoint(base + i * dim, i);
  } else {
    std::atomic<size_t> next{1};
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < T; ++t)
      pool.emplace_back([&] {
        for (size_t i = next++; i < n; i = next++) hnsw_->addPoint(base + i * dim, i);
      });
    for (auto& th : pool) th.join();
  }
  project();
}

HnswIndex::HnswIndex(const std::string& path, int dim) : dim_(dim) {
  space_ = std::make_unique<hnswlib::L2Space>(dim);
  hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), path);
  n_ = hnsw_->cur_element_count;
  cap_ = hnsw_->max_elements_;
  project();
}

void HnswIndex::save(const std::string& path) const { hnsw_->saveIndex(path); }

HnswIndex::~HnswIndex() = default;

void HnswIndex::project() {
  max_level_ = hnsw_->maxlevel_;
  entry_base_ = static_cast<PointId>(hnsw_->getExternalLabel(hnsw_->enterpoint_node_));

  base_to_internal_.assign(n_, 0);
  node_level_.assign(n_, 0);
  father_base_.assign(n_, 0);
  for (size_t internal = 0; internal < n_; ++internal) {
    const PointId base = static_cast<PointId>(hnsw_->getExternalLabel(internal));
    base_to_internal_[base] = static_cast<PointId>(internal);
    node_level_[base] = hnsw_->element_levels_[internal];

    father_base_[base] =
        static_cast<PointId>(hnsw_->getExternalLabel(hnsw_->father(internal)));
  }

  off0_.assign(n_ + 1, 0);
  for (size_t base = 0; base < n_; ++base) {
    unsigned int* ll = hnsw_->get_linklist0(base_to_internal_[base]);
    off0_[base + 1] = hnsw_->getListCount(ll);
  }
  for (size_t i = 0; i < n_; ++i) off0_[i + 1] += off0_[i];
  nbr0_.resize(off0_[n_]);
  for (size_t base = 0; base < n_; ++base) {
    unsigned int* ll = hnsw_->get_linklist0(base_to_internal_[base]);
    const unsigned int cnt = hnsw_->getListCount(ll);
    const PointId* inb = reinterpret_cast<const PointId*>(ll + 1);
    size_t off = off0_[base];
    for (unsigned int j = 0; j < cnt; ++j)
      nbr0_[off + j] = static_cast<PointId>(hnsw_->getExternalLabel(inb[j]));
  }

  upper_.assign(max_level_ + 1, {});
  for (size_t base = 0; base < n_; ++base) {
    const PointId internal = base_to_internal_[base];
    const int lvl = node_level_[base];
    for (int L = 1; L <= lvl; ++L) {
      unsigned int* ll = hnsw_->get_linklist(internal, L);
      const unsigned int cnt = hnsw_->getListCount(ll);
      const PointId* inb = reinterpret_cast<const PointId*>(ll + 1);
      std::vector<PointId> v(cnt);
      for (unsigned int j = 0; j < cnt; ++j)
        v[j] = static_cast<PointId>(hnsw_->getExternalLabel(inb[j]));
      upper_[L].emplace(static_cast<PointId>(base), std::move(v));
    }
  }

  edges_total_ = nbr0_.size();
  for (const auto& m : upper_)
    for (const auto& [base, v] : m) edges_total_ += v.size();
}

std::span<const PointId> HnswIndex::neighbors(PointId u) const {
  return std::span<const PointId>(nbr0_.data() + off0_[u], off0_[u + 1] - off0_[u]);
}

std::span<const PointId> HnswIndex::neighbors_at(PointId u, int level) const {
  if (level == 0) return neighbors(u);
  const auto& m = upper_[level];
  auto it = m.find(u);
  if (it == m.end()) return {};
  return std::span<const PointId>(it->second.data(), it->second.size());
}

const float* HnswIndex::vector(PointId u) const {
  return reinterpret_cast<const float*>(
      hnsw_->getDataByInternalId(base_to_internal_[u]));
}

void HnswIndex::add(const float* vec, PointId id) {
  if (n_ >= cap_) {
    size_t newcap = std::max<size_t>(cap_ ? cap_ * 2 : 16, (size_t)id + 1);
    hnsw_->resizeIndex(newcap);
    cap_ = newcap;
  }
  hnsw_->addPoint(vec, id);
  if ((size_t)id + 1 > n_) n_ = (size_t)id + 1;
  dirty_ = true;
}

void HnswIndex::remove(PointId id) {
  hnsw_->markDelete(base_to_internal_[id]);
}

void HnswIndex::reproject() {
  if (dirty_) { project(); dirty_ = false; }
}

bool HnswIndex::is_deleted(PointId u) const {
  return hnsw_->isMarkedDeleted(base_to_internal_[u]);
}

}
