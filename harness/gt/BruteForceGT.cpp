#include <filesystem>
#include "gt/BruteForceGT.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <queue>
#include <thread>

#include "hnsw_stats/distance/Distance.hpp"

namespace hnsw_stats {

namespace {
constexpr uint32_t kMagic = 0x47544231;

std::string sanitize(const std::string& s) {

  std::string o;
  for (char c : s) {
    if (c == '<') o += "_lt_";
    else if (c == '>') o += "_gt_";
    else if (c == '=') o += "_eq_";
    else o += (std::isalnum((unsigned char)c) ? c : '_');
  }
  return o;
}
}

std::string BruteForceGT::cache_path(const std::string& cache_dir,
                                     const Predicate& phi) const {

  const char* kern =
#ifdef HNSW_STATS_AVX2_DISTANCE
      "_avx2";
#else
      "_scalar";
#endif
  return cache_dir + "/" + ds_.id + "_N" + std::to_string(ds_.N) + "__" +
         sanitize(phi.canonical()) + "__K" + std::to_string(K_max_) + kern + ".gtbin";
}

std::vector<GTEntry> BruteForceGT::compute_fresh(const Predicate& phi) const {
  const int dim = ds_.dim;
  const size_t N = ds_.N;
  const size_t NQ = ds_.queries.size() / dim;
  std::vector<GTEntry> out(NQ);

  std::vector<uint8_t> sat(N, 0);
  {
    const unsigned T = std::max<unsigned>(1u, threads_);
    std::vector<std::thread> pool;
    const size_t chunk = (N + T - 1) / T;
    for (unsigned t = 0; t < T; ++t) {
      const size_t u0 = t * chunk, u1 = std::min(N, u0 + chunk);
      if (u0 >= u1) break;
      pool.emplace_back([&phi, &sat, u0, u1] {
        for (size_t u = u0; u < u1; ++u) sat[u] = phi.eval(static_cast<PointId>(u)) ? 1 : 0; });
    }
    for (auto& th : pool) th.join();
  }
  int n_sat_total = 0;
  for (size_t u = 0; u < N; ++u) n_sat_total += sat[u];

  auto worker = [&](size_t q0, size_t q1) {

    for (size_t q = q0; q < q1; ++q) {
      const float* qv = &ds_.queries[q * dim];
      std::priority_queue<std::pair<float, PointId>> heap;
      const int n_sat = n_sat_total;
      for (size_t u = 0; u < N; ++u) {
        if (!sat[u]) continue;
        const float d = l2_sqr(&ds_.base[u * dim], qv, dim);
        if (static_cast<int>(heap.size()) < K_max_) {
          heap.emplace(d, static_cast<PointId>(u));
        } else if (d < heap.top().first) {
          heap.pop();
          heap.emplace(d, static_cast<PointId>(u));
        }
      }
      GTEntry& e = out[q];
      e.n_satisfying = n_sat;
      e.ids.resize(heap.size());
      e.dists.resize(heap.size());
      for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) {
        e.ids[i] = heap.top().second;
        e.dists[i] = heap.top().first;
        heap.pop();
      }
    }
  };

  const unsigned T = std::max(1u, std::min<unsigned>(threads_, (unsigned)NQ));
  std::vector<std::thread> pool;
  const size_t chunk = (NQ + T - 1) / T;
  for (unsigned t = 0; t < T; ++t) {
    const size_t q0 = t * chunk, q1 = std::min(NQ, q0 + chunk);
    if (q0 >= q1) break;
    pool.emplace_back(worker, q0, q1);
  }
  for (auto& th : pool) th.join();
  return out;
}

std::vector<GTEntry> BruteForceGT::compute(const Predicate& phi,
                                           const std::string& cache_dir) {
  const size_t NQ = ds_.queries.size() / ds_.dim;
  const std::string path = cache_dir.empty() ? "" : cache_path(cache_dir, phi);

  if (!path.empty()) {
    std::ifstream f(path, std::ios::binary);
    if (f) {
      uint32_t magic = 0;
      int32_t kmax = 0;
      uint64_t nq = 0;
      f.read(reinterpret_cast<char*>(&magic), 4);
      f.read(reinterpret_cast<char*>(&kmax), 4);
      f.read(reinterpret_cast<char*>(&nq), 8);
      if (magic == kMagic && kmax == K_max_ && nq == NQ) {
        std::vector<GTEntry> out(NQ);
        for (size_t q = 0; q < NQ; ++q) {
          int32_t n_sat = 0, m = 0;
          f.read(reinterpret_cast<char*>(&n_sat), 4);
          f.read(reinterpret_cast<char*>(&m), 4);
          out[q].n_satisfying = n_sat;
          out[q].ids.resize(m);
          out[q].dists.resize(m);
          if (m) {
            f.read(reinterpret_cast<char*>(out[q].ids.data()), m * 4);
            f.read(reinterpret_cast<char*>(out[q].dists.data()), m * 4);
          }
        }
        if (f) return out;
      }
    }
  }

  std::vector<GTEntry> out = compute_fresh(phi);

  if (!path.empty()) {
    std::filesystem::create_directories(cache_dir);
    std::ofstream f(path, std::ios::binary);
    if (f) {
      const int32_t kmax = K_max_;
      const uint64_t nq = NQ;
      f.write(reinterpret_cast<const char*>(&kMagic), 4);
      f.write(reinterpret_cast<const char*>(&kmax), 4);
      f.write(reinterpret_cast<const char*>(&nq), 8);
      for (size_t q = 0; q < NQ; ++q) {
        const int32_t n_sat = out[q].n_satisfying;
        const int32_t m = static_cast<int32_t>(out[q].ids.size());
        f.write(reinterpret_cast<const char*>(&n_sat), 4);
        f.write(reinterpret_cast<const char*>(&m), 4);
        if (m) {
          f.write(reinterpret_cast<const char*>(out[q].ids.data()), m * 4);
          f.write(reinterpret_cast<const char*>(out[q].dists.data()), m * 4);
        }
      }
    }
  }
  return out;
}

}
