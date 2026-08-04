#include "method/acorn_orig/AcornOrigMethod.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>
#include <string>

#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <faiss/impl/ACORN.h>

namespace hnsw_stats {

struct AcornOrigMethod::Impl {
  std::unique_ptr<faiss::IndexACORNFlat> index;
  std::vector<int> metadata;
  int N = 0, dim = 0;
};

AcornOrigMethod::AcornOrigMethod(int K, int attr_col, int gamma, int m_beta_mult, int threads)
    : impl_(new Impl), K_(K), attr_col_(attr_col),
      gamma_(gamma < 1 ? 1 : gamma), m_beta_mult_(m_beta_mult < 1 ? 1 : m_beta_mult),
      threads_(threads < 1 ? 1 : threads) {}
AcornOrigMethod::~AcornOrigMethod() = default;

void AcornOrigMethod::build(const Dataset& ds, const BuildParams& bp) {

  int bt = threads_; if (const char* e = std::getenv("HS_BUILD_THREADS")) bt = std::max(1, atoi(e));
  faiss::faiss_omp_set_num_threads(bt);
  Impl& I = *impl_;
  I.N = (int)ds.N; I.dim = ds.dim;

  const auto& col = ds.columns[(size_t)std::min<int>(attr_col_, (int)ds.columns.size() - 1)].values;
  I.metadata.resize(I.N);
  for (int i = 0; i < I.N; ++i) I.metadata[i] = (int)llround(col[i]);
  const int M = bp.M;
  m_beta_ = m_beta_mult_ * M;

  char ckey[512] = {0};
  if (const char* cd = std::getenv("HS_INDEX_CACHE_DIR"))
    std::snprintf(ckey, sizeof ckey, "%s/acorn_%s_N%d_M%d_g%d_mb%d.faiss",
                  cd, ds.id.c_str(), I.N, bp.M, gamma_, m_beta_mult_ * bp.M);
  if (ckey[0]) {
    struct stat st;
    if (stat(ckey, &st) == 0) {
      try {
        faiss::Index* raw = faiss::read_index(ckey);
        auto* af = dynamic_cast<faiss::IndexACORNFlat*>(raw);
        if (af) {
          I.index.reset(af);
          build_seconds_ = 0;
          std::fprintf(stderr, "[acorn] index cache HIT %s\n", ckey);
          return;
        }
        delete raw;
        std::fprintf(stderr, "[acorn] cache file wrong type, rebuilding\n");
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[acorn] cache read failed (%s), rebuilding\n", e.what());
      }
    }
  }
  I.index.reset(new faiss::IndexACORNFlat(I.dim, M, gamma_, I.metadata, m_beta_));
  auto t0 = std::chrono::steady_clock::now();
  I.index->add((faiss::idx_t)I.N, ds.base.data());
  build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (ckey[0]) {
    try {

      std::string tmp = std::string(ckey) + ".tmp." + std::to_string((long)getpid());
      faiss::write_index(I.index.get(), tmp.c_str());
      std::rename(tmp.c_str(), ckey);
      std::fprintf(stderr, "[acorn] index cached -> %s\n", ckey);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[acorn] cache write UNSUPPORTED (%s) — vendored ACORN lacks IO; rebuild cost stands\n", e.what());
    }
  }
}

SearchResult AcornOrigMethod::search(const Query& q, const Predicate& phi, int ef) {
  faiss::faiss_omp_set_num_threads(threads_);
  Impl& I = *impl_;
  SearchResult r;
  if (!I.index) return r;
  I.index->acorn.efSearch = std::max(ef, K_);

  std::vector<char> filter((size_t)I.N);
  for (int u = 0; u < I.N; ++u) filter[u] = phi.eval((PointId)u) ? 1 : 0;

  std::vector<faiss::idx_t> ids((size_t)K_, -1);
  std::vector<float> dists((size_t)K_, 0.f);
  const size_t n3_before = faiss::acorn_stats.n3;
  I.index->search(1, q.data, (faiss::idx_t)K_, dists.data(), ids.data(), filter.data());
  const uint64_t ndelta = (uint64_t)(faiss::acorn_stats.n3 - n3_before);

  for (int j = 0; j < K_; ++j) if (ids[j] >= 0) r.topk_ids.push_back((uint32_t)ids[j]);
  r.n_delta = ndelta;
  r.n_delta_satisfying = ndelta;
  r.n_struct_visits = ndelta;
  r.n_pred_evals = 0;
  return r;
}

size_t AcornOrigMethod::persistent_bytes() {
  if (!impl_ || !impl_->index) return 0;
  const char* fn = "/tmp/idxbytes_acorn.faiss";
  faiss::write_index(impl_->index.get(), fn);
  struct stat st; size_t b = (::stat(fn, &st) == 0) ? (size_t)st.st_size : 0;
  ::remove(fn);
  return b;
}

}
