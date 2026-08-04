#include "common/ClusterField.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <thread>

namespace hnsw_stats {

namespace {

struct KMeans {
  std::vector<uint32_t> assign;
  std::vector<std::vector<float>> cent;
};

KMeans kmeans(const std::vector<float>& base, size_t N, int dim, int C, uint64_t seed,
              size_t fit_sample, int iters) {
  std::mt19937_64 rng(seed);
  const size_t S = std::min(N, std::max<size_t>(fit_sample, (size_t)C));
  std::vector<size_t> samp(N);
  std::iota(samp.begin(), samp.end(), 0);
  std::shuffle(samp.begin(), samp.end(), rng);
  samp.resize(S);

  KMeans km;
  km.cent.assign(C, std::vector<float>(dim, 0.f));
  for (int c = 0; c < C; ++c)
    std::copy_n(&base[samp[c % S] * dim], dim, km.cent[c].begin());

  auto nearest = [&](const float* v) {
    int best = 0; double bd = 1e300;
    for (int c = 0; c < C; ++c) {
      double d = 0; const float* ct = km.cent[c].data();
      for (int j = 0; j < dim; ++j) { const double e = v[j] - ct[j]; d += e * e; }
      if (d < bd) { bd = d; best = c; }
    }
    return best;
  };

  std::vector<uint32_t> sa(S, 0);
  int fthr = 1; if (const char* t = std::getenv("HS_PREP_THREADS")) fthr = std::max(1, atoi(t));
  for (int it = 0; it < iters; ++it) {
    if (fthr <= 1) { for (size_t i = 0; i < S; ++i) sa[i] = (uint32_t)nearest(&base[samp[i] * dim]); }
    else {
      std::vector<std::thread> th; const size_t ch = (S + (size_t)fthr - 1) / (size_t)fthr;
      for (int t2 = 0; t2 < fthr; ++t2) { const size_t b = (size_t)t2 * ch, e = std::min(S, b + ch);
        th.emplace_back([&, b, e]{ for (size_t i = b; i < e; ++i) sa[i] = (uint32_t)nearest(&base[samp[i] * dim]); }); }
      for (auto& t3 : th) t3.join();
    }
    std::vector<std::vector<double>> sum(C, std::vector<double>(dim, 0.0));
    std::vector<size_t> cnt(C, 0);
    for (size_t i = 0; i < S; ++i) {
      const float* v = &base[samp[i] * dim];
      auto& s = sum[sa[i]]; ++cnt[sa[i]];
      for (int j = 0; j < dim; ++j) s[j] += v[j];
    }
    for (int c = 0; c < C; ++c) {
      if (!cnt[c]) { std::copy_n(&base[samp[rng() % S] * dim], dim, km.cent[c].begin()); continue; }
      for (int j = 0; j < dim; ++j) km.cent[c][j] = (float)(sum[c][j] / cnt[c]);
    }
  }
  km.assign.resize(N);

  int nthr = 1;
  if (const char* t = std::getenv("HS_PREP_THREADS")) nthr = std::max(1, atoi(t));
  if (nthr <= 1) {
    for (size_t u = 0; u < N; ++u) km.assign[u] = (uint32_t)nearest(&base[u * dim]);
  } else {
    std::vector<std::thread> th;
    const size_t chunk = (N + (size_t)nthr - 1) / (size_t)nthr;
    for (int t2 = 0; t2 < nthr; ++t2) {
      const size_t u0 = (size_t)t2 * chunk, u1 = std::min(N, u0 + chunk);
      if (u0 < u1) th.emplace_back([&, u0, u1] {
        for (size_t u = u0; u < u1; ++u) km.assign[u] = (uint32_t)nearest(&base[u * dim]);
      });
    }
    for (auto& x : th) x.join();
  }
  return km;
}

std::vector<uint32_t> compact_label_map(const std::vector<std::vector<float>>& cent, int C,
                                        int n_labels, uint64_t seed) {
  const int dim = cent.empty() ? 0 : (int)cent[0].size();
  std::mt19937_64 rng(seed);
  std::vector<double> w(n_labels);
  double tot = 0;
  for (int l = 0; l < n_labels; ++l) { w[l] = 1.0 / (l + 1); tot += w[l]; }
  std::vector<int> target(n_labels, 1);
  int used = n_labels;
  for (int l = 0; l < n_labels && used < C; ++l) {
    int extra = (int)std::lround(w[l] / tot * C) - 1;
    extra = std::min(extra, C - used);
    if (extra > 0) { target[l] += extra; used += extra; }
  }
  while (used < C) { target[0]++; used++; }
  auto d2 = [&](int a, int b) {
    double sum = 0;
    for (int j = 0; j < dim; ++j) { const double e = cent[a][j] - cent[b][j]; sum += e * e; }
    return sum;
  };
  std::vector<uint32_t> lab(C, 0);
  std::vector<char> assigned(C, 0);
  for (int l = 0; l < n_labels; ++l) {
    int seedc = -1, tries = 0;
    do { seedc = (int)(rng() % C); } while (assigned[seedc] && ++tries < 8 * C);
    if (assigned[seedc]) { for (int c = 0; c < C; ++c) if (!assigned[c]) { seedc = c; break; } }
    if (seedc < 0 || assigned[seedc]) break;
    assigned[seedc] = 1; lab[seedc] = (uint32_t)l;
    for (int k = 1; k < target[l]; ++k) {
      int best = -1; double bd = 1e300;
      for (int c = 0; c < C; ++c) {
        if (assigned[c]) continue;
        const double d = d2(seedc, c);
        if (d < bd) { bd = d; best = c; }
      }
      if (best < 0) break;
      assigned[best] = 1; lab[best] = (uint32_t)l;
    }
  }
  for (int c = 0; c < C; ++c) if (!assigned[c]) lab[c] = (uint32_t)(rng() % n_labels);
  return lab;
}

std::vector<uint32_t> zipf_label_map(int C, int n_labels, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<uint32_t> order(C);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  std::vector<double> w(n_labels);
  double tot = 0;
  for (int l = 0; l < n_labels; ++l) { w[l] = 1.0 / (l + 1); tot += w[l]; }
  std::vector<uint32_t> lab_of_cluster(C, 0);
  int c = 0;
  double acc = 0;
  for (int l = 0; l < n_labels && c < C; ++l) {
    acc += w[l] / tot;
    int upto = std::max(c + 1, (int)std::lround(acc * C));
    if (l == n_labels - 1) upto = C;
    for (; c < upto && c < C; ++c) lab_of_cluster[order[c]] = (uint32_t)l;
  }
  return lab_of_cluster;
}

void fill_text(TextColumn& tc, size_t N, const std::vector<double>& labels, size_t len,
               uint64_t seed, const std::vector<double>* labels2 = nullptr,
               const std::vector<uint32_t>* quad_clusters = nullptr) {
  static const char AB[] = "abcdefghij lmnopqrstuvwxyz";
  tc.offsets.resize(N + 1);
  tc.bytes.reserve(N * (len + 8));
  for (size_t u = 0; u < N; ++u) {
    tc.offsets[u] = tc.bytes.size();
    uint64_t s = seed ^ (0x9E3779B97F4A7C15ull * (u + 1));
    auto nxt = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(s >> 33); };

    char toks[6][8]; int ntok = 0;
    std::snprintf(toks[ntok++], 8, "kw%03d", (int)labels[u]);
    if (labels2) std::snprintf(toks[ntok++], 8, "qw%03d", (int)(*labels2)[u]);
    if (quad_clusters) {
      int qi[4]; cf_quad_words((*quad_clusters)[u], qi);
      for (int j = 0; j < 4; ++j) std::snprintf(toks[ntok++], 8, "qd%03d", qi[j]);
    }
    const size_t nslot = std::max<size_t>((size_t)ntok, len / 8);
    size_t slot[6];
    for (int t = 0; t < ntok; ++t) {
      slot[t] = nxt() % nslot;
      for (int pj = 0; pj < t; ++pj) if (slot[pj] == slot[t]) { slot[t] = (slot[t] + 1) % nslot; pj = -1; }
    }
    size_t word = 0;
    while (tc.bytes.size() - tc.offsets[u] < len) {
      int emit = -1;
      for (int t = 0; t < ntok; ++t) if (word == slot[t]) { emit = t; break; }
      if (emit >= 0) {
        tc.bytes.push_back(' ');
        tc.bytes.insert(tc.bytes.end(), toks[emit], toks[emit] + 5);
        tc.bytes.push_back(' ');
      } else {
        const int wl = 3 + (int)(nxt() % 6);
        for (int i = 0; i < wl; ++i) tc.bytes.push_back(AB[nxt() % (sizeof(AB) - 1)]);
        tc.bytes.push_back(' ');
      }
      ++word;
    }
    for (int t = 0; t < ntok; ++t) if (word <= slot[t]) {
      tc.bytes.push_back(' ');
      tc.bytes.insert(tc.bytes.end(), toks[t], toks[t] + 5);
      tc.bytes.push_back(' ');
    }
  }
  tc.offsets[N] = tc.bytes.size();
}

}

void append_cluster_field(Dataset& ds, const ClusterFieldParams& p) {
  const size_t N = ds.N;
  const int dim = ds.dim;
  if (N == 0 || dim == 0) return;

  KMeans k1 = kmeans(ds.base, N, dim, p.C, p.seed * 2 + 1, p.fit_sample, p.iters);
  std::vector<uint32_t> map1 = p.compact
      ? compact_label_map(k1.cent, p.C, p.n_labels, p.seed * 2 + 11)
      : zipf_label_map(p.C, p.n_labels, p.seed * 2 + 11);
  ScalarColumn anti;
  anti.name = "cf_anti"; anti.type = "categorical"; anti.orderable = false;
  anti.values.resize(N);
  for (size_t u = 0; u < N; ++u) anti.values[u] = (double)map1[k1.assign[u]];

  KMeans k1b = kmeans(ds.base, N, dim, p.C, p.seed * 2 + 7777, p.fit_sample, p.iters);
  std::vector<uint32_t> map1b = zipf_label_map(p.C, 64, p.seed * 2 + 7778);
  ScalarColumn anti2;
  anti2.name = "cf_anti2"; anti2.type = "categorical"; anti2.orderable = false;
  anti2.values.resize(N);
  for (size_t u = 0; u < N; ++u) anti2.values[u] = (double)map1b[k1b.assign[u]];

  std::vector<uint32_t> quad_cl(N);
  for (size_t u = 0; u < N; ++u) quad_cl[u] = k1.assign[u];
  ScalarColumn quad;
  quad.name = "cf_quad"; quad.type = "categorical"; quad.orderable = false;
  quad.values.resize(N);
  for (size_t u = 0; u < N; ++u) quad.values[u] = (double)k1.assign[u];

  KMeans k2 = kmeans(ds.base, N, dim, p.C, p.seed * 2 + 2, p.fit_sample, p.iters);
  std::vector<uint32_t> map2 = zipf_label_map(p.C, p.n_labels, p.seed * 2 + 22);
  ScalarColumn cat;
  cat.name = "cf_cat"; cat.type = "categorical"; cat.orderable = false;
  cat.values.resize(N);
  {
    std::mt19937_64 rng(p.seed * 2 + 33);
    std::uniform_real_distribution<double> u01(0, 1);
    for (size_t u = 0; u < N; ++u)
      cat.values[u] = (u01(rng) < p.gamma) ? (double)map2[k2.assign[u]]
                                           : (double)(rng() % p.n_labels);
  }

  KMeans k3 = kmeans(ds.base, N, dim, p.C, p.seed * 2 + 3, p.fit_sample, p.iters);
  ScalarColumn rng_col;
  rng_col.name = "cf_range"; rng_col.type = "numeric"; rng_col.orderable = true;
  rng_col.values.resize(N);
  {
    std::mt19937_64 rng(p.seed * 2 + 44);
    std::normal_distribution<double> g01(0, 1);
    std::vector<double> dir(dim);
    for (int j = 0; j < dim; ++j) dir[j] = g01(rng);
    std::vector<double> proj(p.C);
    for (int c = 0; c < p.C; ++c) {
      double s = 0; for (int j = 0; j < dim; ++j) s += dir[j] * k3.cent[c][j];
      proj[c] = s;
    }
    std::vector<int> rk(p.C);
    std::iota(rk.begin(), rk.end(), 0);
    std::sort(rk.begin(), rk.end(), [&](int a, int b) { return proj[a] < proj[b]; });
    std::vector<double> gval(p.C);
    for (int r = 0; r < p.C; ++r) gval[rk[r]] = (double)r / std::max(1, p.C - 1);
    std::uniform_real_distribution<double> u01(0, 1);
    for (size_t u = 0; u < N; ++u)
      rng_col.values[u] = p.gamma * gval[k3.assign[u]] + (1.0 - p.gamma) * u01(rng);
  }

  KMeans k4 = kmeans(ds.base, N, dim, p.C, p.seed * 2 + 4, p.fit_sample, p.iters);
  ScalarColumn arange;
  arange.name = "cf_arange"; arange.type = "numeric"; arange.orderable = true;
  arange.values.resize(N);
  {

    std::mt19937_64 rng(p.seed * 2 + 55);
    const int anchor = (int)(rng() % p.C);
    std::vector<double> proj(p.C);
    for (int c = 0; c < p.C; ++c) {
      double s2 = 0;
      for (int j = 0; j < dim; ++j) { const double e = k4.cent[c][j] - k4.cent[anchor][j]; s2 += e * e; }
      proj[c] = s2;
    }
    std::vector<int> rk(p.C);
    std::iota(rk.begin(), rk.end(), 0);
    std::sort(rk.begin(), rk.end(), [&](int a, int b) { return proj[a] < proj[b]; });
    std::vector<double> gval(p.C);
    for (int r = 0; r < p.C; ++r) gval[rk[r]] = (double)r / std::max(1, p.C - 1);
    for (size_t u = 0; u < N; ++u) arange.values[u] = gval[k4.assign[u]];
  }

  ds.columns.push_back(std::move(anti));
  ds.columns.push_back(std::move(anti2));
  ds.columns.push_back(std::move(quad));
  ds.columns.push_back(std::move(cat));
  ds.columns.push_back(std::move(rng_col));
  ds.columns.push_back(std::move(arange));

  const std::vector<double>& anti_vals  = ds.columns[ds.columns.size() - 6].values;
  const std::vector<double>& anti2_vals = ds.columns[ds.columns.size() - 5].values;
  TextColumn t_long;  t_long.name = "cf_text";
  fill_text(t_long, N, anti_vals, p.text_len, p.seed * 3 + 7, &anti2_vals, &quad_cl);
  TextColumn t_short; t_short.name = "cf_text_s";
  fill_text(t_short, N, anti_vals, p.text_len_s, p.seed * 3 + 8);
  const double pool_mb = (double)(t_long.bytes.size() + t_short.bytes.size()) / 1e6;
  ds.text_columns.push_back(std::move(t_long));
  ds.text_columns.push_back(std::move(t_short));

  std::fprintf(stderr,
               "[cluster_field] C=%d gamma=%.2f seed=%llu labels=%d | +cf_anti/cf_cat/cf_range"
               " +cf_text(%.0fB)/cf_text_s(%.0fB) | text pool %.1f MB\n",
               p.C, p.gamma, (unsigned long long)p.seed, p.n_labels, (double)p.text_len,
               (double)p.text_len_s, pool_mb);
}

std::vector<uint32_t> judge_partition(const Dataset& ds, int C, uint64_t seed) {

  std::string path;
  if (const char* cdir = std::getenv("HS_INDEX_CACHE_DIR")) {
    uint64_t h = 1469598103934665603ULL; auto mix=[&](uint64_t x){h^=x;h*=1099511628211ULL;};
    mix(ds.N); mix((uint64_t)ds.dim); mix((uint64_t)C); mix(seed);
    const size_t step = ds.N/512 ? ds.N/512 : 1;
    for (size_t i=0;i<ds.N;i+=step) mix((uint64_t)(int64_t)(ds.base[i*(size_t)ds.dim]*1e6f));
    char sfx[80]; std::snprintf(sfx,sizeof sfx,"/judge_N%zu_C%d_%016llx.bin",ds.N,C,(unsigned long long)h);
    path = std::string(cdir)+sfx;
    if (std::FILE* f=std::fopen(path.c_str(),"rb")) {
      std::vector<uint32_t> v(ds.N); size_t got=std::fread(v.data(),sizeof(uint32_t),ds.N,f); std::fclose(f);
      if (got==ds.N) { std::fprintf(stderr,"[judge] cache HIT %s\n",path.c_str()); return v; }
    }
  }
  KMeans km = kmeans(ds.base, ds.N, ds.dim, C, seed, 20000, 6);
  if (!path.empty()) if (std::FILE* f=std::fopen(path.c_str(),"wb")) {
    std::fwrite(km.assign.data(),sizeof(uint32_t),ds.N,f); std::fclose(f);
    std::fprintf(stderr,"[judge] cached -> %s\n",path.c_str());
  }
  return std::move(km.assign);
}

}
