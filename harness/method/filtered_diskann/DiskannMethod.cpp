#include "method/filtered_diskann/DiskannMethod.hpp"
#include "hnsw_stats/predicate/Predicate.hpp"
#include <sys/stat.h>
#include <cstdio>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "index.h"
#include "parameters.h"
#include "utils.h"

namespace hnsw_stats {

struct DiskannMethod::Impl {
  std::unique_ptr<diskann::Index<float, uint32_t, uint32_t>> index;
  unsigned adim = 0;
  int N = 0, dim = 0;
  std::vector<double> catvals;
  std::string data_bin, label_file, idx_prefix;
  std::string cache_path;
};

DiskannMethod::DiskannMethod(int K, int cat_col, unsigned R, unsigned L, float alpha)
    : impl_(new Impl), K_(K), cat_col_(cat_col), R_(R), L_(L), alpha_(alpha) {}
DiskannMethod::~DiskannMethod() = default;
void DiskannMethod::set_index_cache(const std::string& p) { impl_->cache_path = p; }

static unsigned round_up8(int d) { return (unsigned)((d + 7) / 8 * 8); }

void DiskannMethod::build(const Dataset& ds, const BuildParams& bp) {
  Impl& I = *impl_;
  I.N = (int)ds.N; I.dim = ds.dim; I.adim = round_up8(ds.dim);
  I.catvals = ds.columns[(size_t)cat_col_].values;

  const std::string tag = std::to_string((long long)::getpid());
  I.data_bin = "/tmp/diskann_data_" + tag + ".bin";
  I.label_file = "/tmp/diskann_labels_" + tag + ".txt";
  I.idx_prefix = "/tmp/diskann_idx_" + tag;

  const int bthreads = std::getenv("HS_BUILD_THREADS") ? atoi(std::getenv("HS_BUILD_THREADS")) : 1;
  auto sp = std::make_shared<diskann::IndexSearchParams>(L_, 1u);
  auto wp = std::make_shared<diskann::IndexWriteParameters>(
      diskann::IndexWriteParametersBuilder(L_, R_).with_filter_list_size(L_).with_alpha(alpha_)
          .with_saturate_graph(false).with_num_threads((uint32_t)std::max(1, bthreads)).build());

  if (!I.cache_path.empty()) {
    struct stat st;
    if (::stat((I.cache_path).c_str(), &st) == 0) {
      try {
        I.index.reset(new diskann::Index<float, uint32_t, uint32_t>(
            diskann::L2, (size_t)I.dim, (size_t)I.N, wp, sp, 0, false, false, false,
            false, 0, false, true));
        auto t0 = std::chrono::steady_clock::now();
        I.index->load(I.cache_path.c_str(), (uint32_t)std::max(1, bthreads), (uint32_t)L_);
        build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[diskann] index cache HIT %s (%.1fs load)\n", I.cache_path.c_str(), build_seconds_);
        return;
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[diskann] cache load FAILED (%s) -> rebuild\n", e.what());
        I.index.reset();
      }
    }
  }

  std::vector<float> buf(ds.base.begin(), ds.base.end());
  diskann::save_bin<float>(I.data_bin, buf.data(), (size_t)I.N, (size_t)I.dim);
  {
    FILE* fl = std::fopen(I.label_file.c_str(), "w");
    for (int i = 0; i < I.N; ++i) std::fprintf(fl, "%lld\n", (long long)I.catvals[i]);
    std::fclose(fl);
  }
  I.index.reset(new diskann::Index<float, uint32_t, uint32_t>(
      diskann::L2, (size_t)I.dim, (size_t)I.N, wp, sp, 0, false, false, false, false, 0, false,
      true));
  auto t0 = std::chrono::steady_clock::now();
  I.index->build_filtered_index(I.data_bin.c_str(), I.label_file, (size_t)I.N);
  build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (!I.cache_path.empty()) {
    try { I.index->save(I.cache_path.c_str()); std::fprintf(stderr, "[diskann] index cached -> %s\n", I.cache_path.c_str()); }
    catch (const std::exception& e) { std::fprintf(stderr, "[diskann] cache save failed: %s\n", e.what()); }
  }
  ::remove(I.data_bin.c_str()); ::remove(I.label_file.c_str());
}

SearchResult DiskannMethod::search(const Query& q, const Predicate& phi, int ef) {
  Impl& I = *impl_;
  SearchResult r;
  if (!I.index) return r;
  long long cat = 0; bool found = false;
  for (int u = 0; u < I.N; ++u) if (phi.eval((PointId)u)) { cat = (long long)I.catvals[u]; found = true; break; }
  if (!found) return r;

  std::vector<float> qv(I.adim, 0.f);
  for (int d = 0; d < I.dim; ++d) qv[d] = q.data[d];
  std::vector<uint32_t> ids((size_t)K_, 0);
  std::vector<float> dists((size_t)K_, 0.f);
  unsigned L = (unsigned)std::max(ef, K_);
  auto retval = I.index->search_with_filters(qv.data(), (uint32_t)cat, (size_t)K_, L, ids.data(), dists.data());

  r.topk_ids.assign(ids.begin(), ids.end());
  r.n_delta = (uint64_t)retval.second;
  r.n_pred_evals = 0;
  r.n_struct_visits = (uint64_t)retval.second;
  return r;
}

size_t DiskannMethod::persistent_bytes() {
  if (!impl_ || !impl_->index) return 0;
  Impl& I = *impl_;
  const char* fn = "/tmp/idxbytes_diskann.index";
  try { I.index->save(fn); } catch (...) { return 0; }
  struct stat st; size_t g = (::stat(fn, &st) == 0) ? (size_t)st.st_size : 0;
  ::remove(fn); ::remove((std::string(fn) + ".data").c_str());
  ::remove((std::string(fn) + "_labels.txt").c_str());
  if (g == 0) return 0;
  return g + (size_t)I.N * (size_t)I.dim * sizeof(float);
}

}
