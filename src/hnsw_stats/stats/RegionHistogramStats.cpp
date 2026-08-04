#include <cstdlib>
#include "hnsw_stats/stats/RegionHistogramStats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hnsw_stats {

static double kRebuildMult() {
  static const double v = [] {
    const char* e = std::getenv("HS_REBUILD_MULT");
    return e ? std::atof(e) : 0.5;
  }();
  return v;
}

namespace {
inline int bucket_of(const std::vector<double>& edges, int B, double x) {
  int b = int(std::upper_bound(edges.begin(), edges.end(), x) - edges.begin()) - 1;
  return std::clamp(b, 0, B - 1);
}
}

RegionHistogramStats::RegionHistogramStats(const RegionMap& rm,
                                           const std::vector<ColumnView>& cols,
                                           int buckets) {
  n_regions_ = rm.n_regions;
  boundary_cnt_ = n_regions_;
  const size_t N = rm.region_of.size();
  region_size_.assign(n_regions_, 0);
  for (size_t u = 0; u < N; ++u) region_size_[rm.region_of[u]]++;

  std::vector<RegionId> ident(n_regions_);
  std::iota(ident.begin(), ident.end(), (RegionId)0);

  for (const auto& col : cols) {
    const std::vector<double>& v = *col.values;
    if (col.categorical) {
      CategoricalCol cc;
      for (double x : v) cc.code_index.emplace(x, 0);
      int idx = 0;
      for (auto& [code, slot] : cc.code_index) slot = idx++;
      cc.n_codes = idx;
      cc.n_kept = n_regions_;
      cc.hist_node = ident;
      cc.row_of = ident;
      cc.counts.assign((size_t)n_regions_ * cc.n_codes, 0);
      for (size_t u = 0; u < N; ++u)
        cc.counts[(size_t)rm.region_of[u] * cc.n_codes + cc.code_index[v[u]]]++;
      categorical_.emplace(col.name, std::move(cc));
    } else {
      NumericCol nc;
      nc.B = std::max(1, buckets);
      std::vector<double> sorted = v;
      std::sort(sorted.begin(), sorted.end());
      nc.edges.resize(nc.B + 1);
      for (int i = 0; i < nc.B; ++i)
        nc.edges[i] = sorted[(size_t)i * sorted.size() / nc.B];
      nc.edges[nc.B] = sorted.back() + 1.0;
      for (int i = 1; i <= nc.B; ++i)
        if (nc.edges[i] <= nc.edges[i - 1]) nc.edges[i] = nc.edges[i - 1];
      nc.n_kept = n_regions_;
      nc.hist_node = ident;
      nc.row_of = ident;
      nc.counts.assign((size_t)n_regions_ * nc.B, 0);
      for (size_t u = 0; u < N; ++u)
        nc.counts[(size_t)rm.region_of[u] * nc.B + bucket_of(nc.edges, nc.B, v[u])]++;
      numeric_.emplace(col.name, std::move(nc));
    }
  }
}

RegionHistogramStats::RegionHistogramStats(const HnswGraph& g, RegionMap& rm,
                                           const std::vector<ColumnView>& cols,
                                           int buckets, const MultiLayerParams& ml) {
  const size_t N = g.size();
  const int B = std::max(1, buckets);
  const int lmin = std::max(1, ml.lmat_min);
  const uint32_t INV = ~0u;

  auto fa = [&](PointId p) { return g.father(p); };
  auto lv = [&](PointId p) { return g.node_level(p); };

  int lmax = ml.lmat_max;
  {
    const int M = g.max_level();
    std::vector<size_t> at_or_above(M + 2, 0);
    for (PointId p = 0; p < N; ++p) { int l = lv(p); if (l > M) l = M; at_or_above[l]++; }
    for (int L = M - 1; L >= 0; --L) at_or_above[L] += at_or_above[L + 1];
    if (lmax < 0) { lmax = lmin; for (int L = M; L >= lmin; --L) if (at_or_above[L] >= 2) { lmax = L; break; } }
    if (lmax < lmin) lmax = lmin;
  }

  rm.multilayer = true;
  rm.region_of.assign(N, 0);
  rm.parent_of.assign(N, 0);
  for (PointId p = 0; p < N; ++p) rm.parent_of[p] = fa(p);

  std::vector<uint32_t> cand_of(N, INV);
  std::vector<PointId> cand_pt;
  for (PointId p = 0; p < N; ++p) if (lv(p) >= lmin) { cand_of[p] = (uint32_t)cand_pt.size(); cand_pt.push_back(p); }
  const size_t NC = cand_pt.size();

  if (NC == 0) {
    n_regions_ = 1; boundary_cnt_ = 1; region_size_.assign(1, (uint32_t)N);
    rm.n_regions = 1; rm.lmat_max_nodes = {0}; rm.lmat_max_n = {(uint32_t)N};
    rm.lmat_trees = { RegionMap::LmatTreeMeta{~(uint32_t)0, 0u, 0u} };
    return;
  }

  std::vector<int> cand_lv(NC);
  std::vector<uint32_t> parent_cand(NC, INV);
  for (uint32_t c = 0; c < NC; ++c) {
    PointId p = cand_pt[c]; cand_lv[c] = lv(p);
    if (cand_lv[c] < lmax) { PointId f = fa(p); if (f != p) parent_cand[c] = cand_of[f]; }
  }

  std::vector<uint32_t> fca(N, INV), lmax_of(N, INV);
  for (PointId p = 0; p < N; ++p) {
    PointId a = p; int guard = 0;
    while (lv(a) < lmin && fa(a) != a && ++guard < (int)N + 1) a = fa(a);
    fca[p] = (lv(a) >= lmin) ? cand_of[a] : INV;
    PointId b = p; guard = 0;
    while (lv(b) < lmax && fa(b) != b && ++guard < (int)N + 1) b = fa(b);
    lmax_of[p] = (lv(b) >= lmin) ? cand_of[b] : INV;
  }

  std::vector<uint32_t> cand_n(NC, 0), lmax_excl(NC, 0);
  for (PointId p = 0; p < N; ++p) if (lmax_of[p] != INV) lmax_excl[lmax_of[p]]++;

  struct TmpNum { std::vector<double> edges; std::vector<uint32_t> counts; int B; };
  struct TmpCat { std::unordered_map<double,int> code_index; std::vector<uint32_t> counts; int n_codes; };
  std::vector<TmpNum> tnum; std::vector<TmpCat> tcat;
  std::vector<int> col_kind(cols.size(), 0);
  std::vector<int> col_slot(cols.size(), 0);
  std::vector<std::vector<int>> pt_bucket(cols.size());
  for (size_t ci = 0; ci < cols.size(); ++ci) {
    const auto& col = cols[ci]; const std::vector<double>& v = *col.values;
    if (col.categorical) {
      TmpCat t; for (double x : v) t.code_index.emplace(x, 0); int idx = 0;
      for (auto& [code, slot] : t.code_index) slot = idx++; t.n_codes = idx;
      t.counts.assign(NC * (size_t)t.n_codes, 0);
      pt_bucket[ci].resize(N); for (size_t u = 0; u < N; ++u) pt_bucket[ci][u] = t.code_index[v[u]];
      col_kind[ci] = 1; col_slot[ci] = (int)tcat.size(); tcat.push_back(std::move(t));
    } else {
      TmpNum t; t.B = B; std::vector<double> s = v; std::sort(s.begin(), s.end());
      t.edges.resize(B + 1); for (int i = 0; i < B; ++i) t.edges[i] = s[(size_t)i * s.size() / B];
      t.edges[B] = s.back() + 1.0; for (int i = 1; i <= B; ++i) if (t.edges[i] <= t.edges[i-1]) t.edges[i] = t.edges[i-1];
      t.counts.assign(NC * (size_t)B, 0);
      pt_bucket[ci].resize(N); for (size_t u = 0; u < N; ++u) pt_bucket[ci][u] = bucket_of(t.edges, B, v[u]);
      col_kind[ci] = 0; col_slot[ci] = (int)tnum.size(); tnum.push_back(std::move(t));
    }
  }

  for (PointId p = 0; p < N; ++p) {
    for (uint32_t c = fca[p]; c != INV; c = parent_cand[c]) {
      cand_n[c]++;
      for (size_t ci = 0; ci < cols.size(); ++ci) {
        if (col_kind[ci] == 0) { auto& t = tnum[col_slot[ci]]; t.counts[(size_t)c * t.B + pt_bucket[ci][p]]++; }
        else { auto& t = tcat[col_slot[ci]]; t.counts[(size_t)c * t.n_codes + pt_bucket[ci][p]]++; }
      }
    }
  }

  std::vector<std::vector<uint8_t>> kept(cols.size(), std::vector<uint8_t>(NC, 0));
  auto is_root = [&](uint32_t c) { return parent_cand[c] == INV; };
  std::vector<uint32_t> by_level(NC);
  std::iota(by_level.begin(), by_level.end(), 0u);
  std::sort(by_level.begin(), by_level.end(), [&](uint32_t a, uint32_t b){ return cand_lv[a] < cand_lv[b]; });

  std::vector<uint32_t> root_of(NC, INV);
  for (uint32_t c = 0; c < NC; ++c) { uint32_t r = c; while (parent_cand[r] != INV) r = parent_cand[r]; root_of[c] = r; }
  std::vector<uint32_t> treeidx_of_root(NC, INV); std::vector<uint32_t> tree_root;
  for (uint32_t c = 0; c < NC; ++c) if (is_root(c)) { treeidx_of_root[c] = (uint32_t)tree_root.size(); tree_root.push_back(c); }
  const size_t NT = tree_root.size();
  std::vector<uint32_t> tree_of(NC); for (uint32_t c = 0; c < NC; ++c) tree_of[c] = treeidx_of_root[root_of[c]];

  const size_t UNBOUNDED = ~(size_t)0;
  std::vector<size_t> budget_local(NT, UNBOUNDED);
  if (ml.budget_bytes != 0 && NT > 0 && N > 0) {
    size_t assigned = 0, maxtree = 0;
    for (size_t t = 0; t < NT; ++t) {
      budget_local[t] = (size_t)((double)ml.budget_bytes * (double)lmax_excl[tree_root[t]] / (double)N);
      assigned += budget_local[t];
      if (lmax_excl[tree_root[t]] > lmax_excl[tree_root[maxtree]]) maxtree = t;
    }
    if (assigned < ml.budget_bytes) budget_local[maxtree] += (ml.budget_bytes - assigned);
  }

  std::vector<std::vector<double>> errc(cols.size(), std::vector<double>(NC, 0.0));
  std::vector<size_t> rowbytes(cols.size(), 0);
  std::vector<double> thr_c(cols.size(), 0.0);
  for (size_t ci = 0; ci < cols.size(); ++ci) {
    auto& K = kept[ci];
    for (uint32_t c = 0; c < NC; ++c) if (is_root(c)) K[c] = 1;
    const bool is_cat = (col_kind[ci] == 1);
    thr_c[ci] = is_cat ? ml.eps_point : ml.delta_range;
    const int W = is_cat ? tcat[col_slot[ci]].n_codes : B;
    rowbytes[ci] = (size_t)W * sizeof(uint32_t);
    const uint32_t* CT = is_cat ? tcat[col_slot[ci]].counts.data() : tnum[col_slot[ci]].counts.data();
    auto& err = errc[ci];
    for (uint32_t c : by_level) {
      if (is_root(c)) continue;
      const uint32_t pcand = parent_cand[c];
      const double nc = cand_n[c] > 0 ? (double)cand_n[c] : 1.0;
      const double np = cand_n[pcand] > 0 ? (double)cand_n[pcand] : 1.0;
      const uint32_t* rc = CT + (size_t)c * W;
      const uint32_t* rp = CT + (size_t)pcand * W;
      double e = 0.0;
      const double nslack = ml.naware_c > 0 ? ml.naware_c * std::sqrt(1.0 / nc + 1.0 / np) : 0.0;
      if (!is_cat) {
        double cc = 0, cp = 0;
        for (int b = 0; b < W; ++b) { cc += rc[b]; cp += rp[b];
          e = std::max(e, std::max(0.0, std::fabs(cc / nc - cp / np) - nslack)); }
      } else {
        for (int b = 0; b < W; ++b) if (rc[b] > 0) {
          const double fc = rc[b] / nc, fp = rp[b] / np;
          const double bs = ml.naware_c > 0 ? ml.naware_c * std::sqrt(std::max(fc * (1.0 - fc), 1e-12) * (1.0 / nc + 1.0 / np)) : 0.0;
          e = std::max(e, std::max(0.0, std::fabs(fp - fc) - bs) / fc); }
      }
      err[c] = e;
    }
  }

  std::vector<size_t> used(NT, 0);
  for (uint32_t c = 0; c < NC; ++c) if (is_root(c)) { const uint32_t t = tree_of[c];
    for (size_t ci = 0; ci < cols.size(); ++ci) used[t] += rowbytes[ci]; }
  struct Elig { uint32_t c; uint32_t ci; double err; };
  std::vector<std::vector<Elig>> tree_elig(NT);
  for (size_t ci = 0; ci < cols.size(); ++ci)
    for (uint32_t c = 0; c < NC; ++c)
      if (!is_root(c) && errc[ci][c] > thr_c[ci]) tree_elig[tree_of[c]].push_back({c, (uint32_t)ci, errc[ci][c]});
  for (size_t t = 0; t < NT; ++t) {
    auto& E = tree_elig[t];
    std::sort(E.begin(), E.end(), [](const Elig& a, const Elig& b){ return a.err > b.err; });
    for (const Elig& e : E)
      if (budget_local[t] == UNBOUNDED || used[t] + rowbytes[e.ci] <= budget_local[t]) { kept[e.ci][e.c] = 1; used[t] += rowbytes[e.ci]; }
  }

  std::vector<uint32_t> boundid(NC, INV);
  RegionId nb = 0;
  for (uint32_t c = 0; c < NC; ++c) { bool any = false; for (size_t ci = 0; ci < cols.size(); ++ci) if (kept[ci][c]) { any = true; break; }
    if (any) boundid[c] = nb++; }
  if (nb == 0) { for (uint32_t c = 0; c < NC; ++c) if (is_root(c)) boundid[c] = nb++; }
  n_regions_ = nb;
  rm.n_regions = nb;

  auto finest_bound = [&](uint32_t c) -> uint32_t {
    while (c != INV && boundid[c] == INV) c = parent_cand[c];
    return c;
  };

  region_size_.assign(nb, 0);
  for (PointId p = 0; p < N; ++p) {
    uint32_t c = (fca[p] != INV) ? finest_bound(fca[p]) : INV;
    rm.region_of[p] = (c != INV) ? boundid[c] : 0;
  }

  for (uint32_t c = 0; c < NC; ++c) if (boundid[c] != INV) region_size_[boundid[c]] = cand_n[c];
  boundary_cnt_ = nb;

  rm.lmat_max_nodes.clear(); rm.lmat_max_n.clear(); rm.lmat_trees.clear();
  const uint32_t U32MAX = ~(uint32_t)0;
  for (uint32_t c = 0; c < NC; ++c) if (lmax_excl[c] > 0) {
    uint32_t bc = (boundid[c] != INV) ? c : finest_bound(c);
    if (bc != INV && boundid[bc] != INV) {
      rm.lmat_max_nodes.push_back(boundid[bc]); rm.lmat_max_n.push_back(lmax_excl[c]);
      const size_t bl = budget_local[tree_of[c]];
      rm.lmat_trees.push_back(RegionMap::LmatTreeMeta{
          (bl == UNBOUNDED ? U32MAX : (uint32_t)std::min<size_t>(bl, U32MAX)), 0u, 0u });
    }
  }
  if (rm.lmat_max_nodes.empty()) { rm.lmat_max_nodes.push_back(0); rm.lmat_max_n.push_back((uint32_t)N);
    rm.lmat_trees.push_back(RegionMap::LmatTreeMeta{U32MAX, 0u, 0u}); }

  for (size_t ci = 0; ci < cols.size(); ++ci) {
    const auto& col = cols[ci]; const auto& K = kept[ci];

    auto finest_kept = [&](uint32_t c) -> uint32_t { while (c != INV && !K[c]) c = parent_cand[c]; return c; };
    const int W = (col_kind[ci] == 1) ? tcat[col_slot[ci]].n_codes : B;

    std::vector<RegionId> row_of(nb, kNoRow); RegionId nkept = 0;
    for (uint32_t c = 0; c < NC; ++c) if (K[c] && boundid[c] != INV) row_of[boundid[c]] = nkept++;
    std::vector<uint32_t> counts((size_t)nkept * W, 0);
    const uint32_t* CT = (col_kind[ci] == 1) ? tcat[col_slot[ci]].counts.data() : tnum[col_slot[ci]].counts.data();
    for (uint32_t c = 0; c < NC; ++c) if (K[c] && boundid[c] != INV) {
      RegionId row = row_of[boundid[c]]; const uint32_t* src = CT + (size_t)c * W;
      std::copy(src, src + W, counts.begin() + (size_t)row * W);
    }

    std::vector<RegionId> hist_node(nb, 0);
    for (uint32_t c = 0; c < NC; ++c) if (boundid[c] != INV) { uint32_t fk = finest_kept(c); hist_node[boundid[c]] = boundid[fk]; }

    if (col_kind[ci] == 1) {
      CategoricalCol cc; cc.code_index = tcat[col_slot[ci]].code_index; cc.n_codes = W;
      cc.n_kept = nkept; cc.counts = std::move(counts); cc.hist_node = std::move(hist_node); cc.row_of = std::move(row_of);
      categorical_.emplace(col.name, std::move(cc));
    } else {
      NumericCol nc; nc.edges = tnum[col_slot[ci]].edges; nc.B = W;
      nc.n_kept = nkept; nc.counts = std::move(counts); nc.hist_node = std::move(hist_node); nc.row_of = std::move(row_of);
      numeric_.emplace(col.name, std::move(nc));
    }
  }

  ml_.active = true; ml_.g = &g; ml_.lmin = lmin; ml_.lmax = lmax; ml_.buckets = B;
  ml_.delta_range = ml.delta_range; ml_.eps_point = ml.eps_point; ml_.naware_c = ml.naware_c;
  ml_.cand_of = cand_of; ml_.fca_of = fca; ml_.cand_pt = cand_pt; ml_.cand_lv = cand_lv;
  ml_.parent_cand = parent_cand; ml_.tree_of_cand = tree_of; ml_.tree_rootcand = tree_root;
  ml_.cand_regionid.assign(NC, kInvU);
  ml_.bound_regionid_of.assign(N, kNoRow);
  for (uint32_t c = 0; c < NC; ++c) if (boundid[c] != INV) {
    ml_.cand_regionid[c] = boundid[c]; ml_.bound_regionid_of[cand_pt[c]] = boundid[c]; }
  ml_.lmatnode_of_tree.assign(NT, kNoRow);
  for (size_t t = 0; t < NT; ++t) if (boundid[tree_root[t]] != INV) ml_.lmatnode_of_tree[t] = boundid[tree_root[t]];
  ml_.col_values.resize(cols.size()); ml_.col_meta.resize(cols.size());
  for (size_t ci = 0; ci < cols.size(); ++ci) {
    ml_.col_values[ci] = *cols[ci].values;
    ml_.col_meta[ci] = MLState::ColMeta{cols[ci].name, cols[ci].categorical};
  }
  pres_build_();
}

double RegionHistogramStats::cdf(RegionId node, const NumericCol& c, double x) const {
  const uint32_t sz = region_size_[node];
  if (sz == 0) return 0.0;
  const RegionId row = c.row_of[node];
  if (row == kNoRow) return 0.0;
  const uint32_t* r = &c.counts[(size_t)row * c.B];
  if (x <= c.edges[0]) return 0.0;
  if (x >= c.edges[c.B]) return 1.0;
  double below = 0.0;
  for (int b = 0; b < c.B; ++b) {
    if (x >= c.edges[b + 1]) { below += r[b]; }
    else { const double lo = c.edges[b], hi = c.edges[b + 1];
      below += ((hi > lo) ? (x - lo) / (hi - lo) : 0.0) * r[b]; break; }
  }
  return below / sz;
}

double RegionHistogramStats::sigma_hat(RegionId region,
                                       const ColumnPredicate& phi) const {
  using F = ColumnPredicate::Form;
  if (phi.form() == F::EQ) {
    auto it = categorical_.find(phi.column());
    if (it == categorical_.end()) return 0.0;
    const auto& cc = it->second;
    const RegionId node = cc.hist_node[region];
    const uint32_t sz = region_size_[node];
    if (sz == 0) return 0.0;
    const RegionId row = cc.row_of[node];
    if (row == kNoRow) return 0.0;
    auto ci = cc.code_index.find(phi.a());
    if (ci == cc.code_index.end()) return 0.0;
    return double(cc.counts[(size_t)row * cc.n_codes + ci->second]) / sz;
  }
  auto it = numeric_.find(phi.column());
  if (it == numeric_.end()) return 0.0;
  const NumericCol& c = it->second;
  const RegionId node = c.hist_node[region];
  switch (phi.form()) {
    case F::LT:       return cdf(node, c, phi.a());
    case F::GT:       return 1.0 - cdf(node, c, phi.a());
    case F::INTERVAL: return std::max(0.0, cdf(node, c, phi.b()) - cdf(node, c, phi.a()));
    default:          return 0.0;
  }
}

uint32_t RegionHistogramStats::fca_cand_of_(PointId v) const {
  if ((size_t)v < ml_.cand_of.size() && ml_.cand_of[v] != kInvU) return ml_.cand_of[v];
  return ((size_t)v < ml_.fca_of.size()) ? ml_.fca_of[v] : kInvU;
}

RegionId RegionHistogramStats::finest_bound_ancestor_(PointId v) const {
  uint32_t c = fca_cand_of_(v);
  while (c != kInvU && ml_.cand_regionid[c] == kInvU) c = ml_.parent_cand[c];
  return (c != kInvU) ? ml_.cand_regionid[c] : kNoRow;
}

uint32_t RegionHistogramStats::lmat_max_root_cand_(PointId v) const {
  uint32_t c = fca_cand_of_(v);
  if (c == kInvU) return kInvU;
  while (ml_.parent_cand[c] != kInvU) c = ml_.parent_cand[c];
  return c;
}

RegionId RegionHistogramStats::forest_lmat_max_ancestor(PointId v) const {
  const uint32_t c = lmat_max_root_cand_(v);
  return (c != kInvU) ? ml_.cand_regionid[c] : kNoRow;
}

uint32_t RegionHistogramStats::forest_lmat_max_tree_of(PointId v) const {
  const uint32_t c = lmat_max_root_cand_(v);
  return (c != kInvU && c < ml_.tree_of_cand.size()) ? ml_.tree_of_cand[c] : 0;
}

void RegionHistogramStats::forest_region_boundary_ancestors(PointId v, std::vector<RegionId>& out) const {
  out.clear();
  for (uint32_t c = fca_cand_of_(v); c != kInvU; c = ml_.parent_cand[c])
    if (ml_.cand_regionid[c] != kInvU) out.push_back(ml_.cand_regionid[c]);
}

void RegionHistogramStats::on_insert(RegionMap& rm, PointId id, const std::vector<double>& colvals) {
  if (!ml_.active) return;
  for (size_t ci = 0; ci < ml_.col_values.size(); ++ci) {
    if ((size_t)id >= ml_.col_values[ci].size()) ml_.col_values[ci].resize((size_t)id + 1, 0.0);
    ml_.col_values[ci][id] = colvals[ci];
  }
  if ((size_t)id >= ml_.bound_regionid_of.size()) ml_.bound_regionid_of.resize((size_t)id + 1, kNoRow);
  ml_.bound_regionid_of[id] = kNoRow;
  if ((size_t)id >= ml_.cand_of.size()) ml_.cand_of.resize((size_t)id + 1, kInvU);
  ml_.cand_of[id] = kInvU;
  if ((size_t)id >= ml_.fca_of.size()) ml_.fca_of.resize((size_t)id + 1, kInvU);
  { PointId a = id; int guard = 0;
    while ((size_t)a < ml_.cand_of.size() && ml_.cand_of[a] == kInvU && ml_.g->father(a) != a && ++guard <= (int)ml_.g->size()) a = ml_.g->father(a);
    ml_.fca_of[id] = ((size_t)a < ml_.cand_of.size()) ? ml_.cand_of[a] : kInvU; }
  if ((size_t)id >= rm.region_of.size()) rm.region_of.resize((size_t)id + 1, 0);
  const RegionId rstar = finest_bound_ancestor_(id);
  rm.region_of[id] = (rstar != kNoRow) ? rstar : 0;

  std::vector<RegionId> ancs; forest_region_boundary_ancestors(id, ancs);
  for (RegionId r : ancs) region_size_[r]++;
  for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
    const auto& m = ml_.col_meta[ci];
    if (m.cat) { auto& c = categorical_.find(m.name)->second; auto it = c.code_index.find(colvals[ci]);
      if (it == c.code_index.end()) continue; const int b = it->second;
      for (RegionId r : ancs) { const RegionId row = c.row_of[r]; if (row != kNoRow) c.counts[(size_t)row * c.n_codes + b]++; }
    } else { auto& c = numeric_.find(m.name)->second; const int b = bucket_of(c.edges, c.B, colvals[ci]);
      for (RegionId r : ancs) { const RegionId row = c.row_of[r]; if (row != kNoRow) c.counts[(size_t)row * c.B + b]++; }
    }
  }
  const uint32_t t = forest_lmat_max_tree_of(id);
  if (t < rm.lmat_max_n.size()) rm.lmat_max_n[t]++;
  if (pres_on_) {
    const RegionId pr = rm.region_of[id];
    for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
      const int b = ml_bucket(ci, colvals[ci]);
      if (b >= 0) pres_set_(ci, pr, b); else pres_degraded_[ci] = 1;
    }
  }
}

bool RegionHistogramStats::on_update(RegionMap& rm, PointId id, size_t col, double new_val) {
  if (!ml_.active || col >= ml_.col_values.size()) return false;
  ml_.col_values[col][id] = new_val;
  if (pres_on_) {
    const int b = ml_bucket(col, new_val);
    if (b >= 0) pres_set_(col, rm.region_of[id], b);
    else pres_degraded_[col] = 1;
  }
  const uint32_t t = forest_lmat_max_tree_of(id);
  if (t >= rm.lmat_trees.size()) return false;
  rm.lmat_trees[t].cnt_mod++;
  const uint32_t thr = std::max<uint32_t>(16, (uint32_t)(kRebuildMult() * (t < rm.lmat_max_n.size() ? rm.lmat_max_n[t] : 0)));
  if (rm.lmat_trees[t].cnt_mod >= thr) return rebuild_tree(rm, t);
  return false;
}

bool RegionHistogramStats::on_delete(RegionMap& rm, PointId id) {
  if (!ml_.active) return false;
  const RegionId arid = forest_lmat_max_ancestor(id);
  if (arid != kNoRow) {
    for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
      const auto& m = ml_.col_meta[ci]; const double val = ml_.col_values[ci][id];
      if (m.cat) { auto& c = categorical_.find(m.name)->second; auto it = c.code_index.find(val);
        if (it != c.code_index.end()) { const RegionId row = c.row_of[arid];
          if (row != kNoRow && c.counts[(size_t)row * c.n_codes + it->second] > 0) c.counts[(size_t)row * c.n_codes + it->second]--; }
      } else { auto& c = numeric_.find(m.name)->second; const int b = bucket_of(c.edges, c.B, val);
        const RegionId row = c.row_of[arid];
        if (row != kNoRow && c.counts[(size_t)row * c.B + b] > 0) c.counts[(size_t)row * c.B + b]--; }
    }
    if (region_size_[arid] > 0) region_size_[arid]--;
  }
  const uint32_t t = forest_lmat_max_tree_of(id);
  if (t < rm.lmat_max_n.size() && rm.lmat_max_n[t] > 0) rm.lmat_max_n[t]--;
  if (t >= rm.lmat_trees.size()) return false;
  rm.lmat_trees[t].cnt_del++;
  const uint32_t thr = std::max<uint32_t>(16, (uint32_t)(kRebuildMult() * (t < rm.lmat_max_n.size() ? rm.lmat_max_n[t] : 0)));
  if (rm.lmat_trees[t].cnt_del >= thr) return rebuild_tree(rm, t);
  return false;
}

bool RegionHistogramStats::rebuild_tree(RegionMap& rm, uint32_t T) {
  if (!ml_.active || T >= ml_.tree_rootcand.size()) return false;
  const uint32_t INV = kInvU;
  const size_t N = ml_.bound_regionid_of.size();
  const RegionId n_regions_before = n_regions_;

  std::vector<uint32_t> Tc; std::unordered_map<uint32_t,uint32_t> loc;
  for (uint32_t c = 0; c < ml_.cand_pt.size(); ++c) if (ml_.tree_of_cand[c] == T) { loc[c] = (uint32_t)Tc.size(); Tc.push_back(c); }
  const size_t nTc = Tc.size();
  if (nTc == 0) { rm.lmat_trees[T].cnt_mod = rm.lmat_trees[T].cnt_del = 0; return false; }

  std::vector<uint32_t> cand_n(nTc, 0);
  std::vector<std::vector<uint32_t>> tmp(ml_.col_meta.size());
  std::vector<int> Wc(ml_.col_meta.size());
  for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
    Wc[ci] = ml_.col_meta[ci].cat ? categorical_.find(ml_.col_meta[ci].name)->second.n_codes
                                  : numeric_.find(ml_.col_meta[ci].name)->second.B;
    tmp[ci].assign(nTc * (size_t)Wc[ci], 0);
  }
  std::vector<uint32_t> Tpts;
  for (PointId p = 0; p < N; ++p) {
    if (ml_.g->is_deleted(p)) continue;
    if (forest_lmat_max_tree_of(p) != T) continue;
    Tpts.push_back(p);
    uint32_t c = fca_cand_of_(p);
    while (c != INV) {
      auto it = loc.find(c);
      if (it != loc.end()) { const uint32_t li = it->second; cand_n[li]++;
        for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
          const auto& m = ml_.col_meta[ci]; const double val = ml_.col_values[ci][p]; int b;
          if (m.cat) { auto& cc = categorical_.find(m.name)->second; auto cit = cc.code_index.find(val); if (cit == cc.code_index.end()) continue; b = cit->second; }
          else { b = bucket_of(numeric_.find(m.name)->second.edges, Wc[ci], val); }
          tmp[ci][(size_t)li * Wc[ci] + b]++;
        }
      }
      c = ml_.parent_cand[c];
    }
  }

  std::vector<std::vector<uint8_t>> kept(ml_.col_meta.size(), std::vector<uint8_t>(nTc, 0));
  auto is_root_li = [&](uint32_t li){ return ml_.parent_cand[Tc[li]] == INV; };
  for (uint32_t li = 0; li < nTc; ++li) if (is_root_li(li)) for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) kept[ci][li] = 1;
  std::vector<std::vector<double>> errc(ml_.col_meta.size(), std::vector<double>(nTc, 0.0));
  for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
    const bool cat = ml_.col_meta[ci].cat; const int W = Wc[ci];
    for (uint32_t li = 0; li < nTc; ++li) {
      if (is_root_li(li)) continue;
      const uint32_t pc = ml_.parent_cand[Tc[li]]; auto pit = loc.find(pc); if (pit == loc.end()) continue;
      const uint32_t pl = pit->second;
      const double nc = cand_n[li] > 0 ? (double)cand_n[li] : 1.0, np = cand_n[pl] > 0 ? (double)cand_n[pl] : 1.0;
      const uint32_t* rc = &tmp[ci][(size_t)li * W]; const uint32_t* rp = &tmp[ci][(size_t)pl * W];
      double e = 0.0;
      const double nslack2 = ml_.naware_c > 0 ? ml_.naware_c * std::sqrt(1.0 / nc + 1.0 / np) : 0.0;
      if (!cat) { double cc = 0, cp = 0; for (int b = 0; b < W; ++b) { cc += rc[b]; cp += rp[b]; e = std::max(e, std::max(0.0, std::fabs(cc / nc - cp / np) - nslack2)); } }
      else { for (int b = 0; b < W; ++b) if (rc[b] > 0) { const double fc = rc[b] / nc, fp = rp[b] / np;
        const double bs2 = ml_.naware_c > 0 ? ml_.naware_c * std::sqrt(std::max(fc * (1.0 - fc), 1e-12) * (1.0 / nc + 1.0 / np)) : 0.0;
        e = std::max(e, std::max(0.0, std::fabs(fp - fc) - bs2) / fc); } }
      errc[ci][li] = e;
    }
  }
  const uint32_t U32MAX = ~(uint32_t)0;
  size_t budget = rm.lmat_trees[T].budget_local;
  const size_t bl = (budget == U32MAX) ? ~(size_t)0 : budget;
  size_t used = 0;
  for (uint32_t li = 0; li < nTc; ++li) if (is_root_li(li)) for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) used += (size_t)Wc[ci] * sizeof(uint32_t);
  struct E { uint32_t li, ci; double err; };
  std::vector<E> elig;
  for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
    const double thr = ml_.col_meta[ci].cat ? ml_.eps_point : ml_.delta_range;
    for (uint32_t li = 0; li < nTc; ++li) if (!is_root_li(li) && errc[ci][li] > thr) elig.push_back({li, (uint32_t)ci, errc[ci][li]});
  }
  std::sort(elig.begin(), elig.end(), [](const E& a, const E& b){ return a.err > b.err; });
  for (const E& e : elig) { const size_t rb = (size_t)Wc[e.ci] * sizeof(uint32_t);
    if (bl == ~(size_t)0 || used + rb <= bl) { kept[e.ci][e.li] = 1; used += rb; } }

  std::vector<uint8_t> now_bound(nTc, 0);
  for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) for (uint32_t li = 0; li < nTc; ++li) if (kept[ci][li]) now_bound[li] = 1;
  auto grow_region = [&](){ region_size_.push_back(0);
    for (auto& kv : numeric_) { kv.second.row_of.push_back(kNoRow); kv.second.hist_node.push_back(0); }
    for (auto& kv : categorical_) { kv.second.row_of.push_back(kNoRow); kv.second.hist_node.push_back(0); } };
  for (uint32_t li = 0; li < nTc; ++li) {
    const uint32_t c = Tc[li]; RegionId rid = ml_.cand_regionid[c];
    if (now_bound[li]) {
      if (rid == INV) { rid = n_regions_++; grow_region(); ml_.cand_regionid[c] = rid; }
      region_size_[rid] = cand_n[li]; ml_.bound_regionid_of[ml_.cand_pt[c]] = rid;
    } else if (rid != INV) {
      region_size_[rid] = 0; ml_.bound_regionid_of[ml_.cand_pt[c]] = kNoRow; ml_.cand_regionid[c] = INV;
    }
  }
  rm.n_regions = n_regions_;

  for (size_t ci = 0; ci < ml_.col_meta.size(); ++ci) {
    const int W = Wc[ci];
    auto refresh = [&](auto& c){
      for (uint32_t li = 0; li < nTc; ++li) { const RegionId rid = ml_.cand_regionid[Tc[li]]; if (rid == INV) continue;
        if (kept[ci][li]) { RegionId row = c.row_of[rid];
          if (row == kNoRow) { row = c.n_kept++; c.counts.resize((size_t)c.n_kept * W, 0); c.row_of[rid] = row; }
          std::copy(tmp[ci].begin() + (size_t)li * W, tmp[ci].begin() + (size_t)(li + 1) * W, c.counts.begin() + (size_t)row * W);
        } else c.row_of[rid] = kNoRow; }
      for (uint32_t li = 0; li < nTc; ++li) { const RegionId rid = ml_.cand_regionid[Tc[li]]; if (rid == INV) continue;
        uint32_t fk = Tc[li]; while (fk != INV) { auto it = loc.find(fk); if (it != loc.end() && kept[ci][it->second]) break; fk = ml_.parent_cand[fk]; }
        c.hist_node[rid] = (fk != INV) ? ml_.cand_regionid[fk] : rid; }
    };
    if (ml_.col_meta[ci].cat) refresh(categorical_.find(ml_.col_meta[ci].name)->second);
    else refresh(numeric_.find(ml_.col_meta[ci].name)->second);
  }

  for (uint32_t p : Tpts) { const RegionId rs = finest_bound_ancestor_(p); if (rs != kNoRow) rm.region_of[p] = rs; }
  if (T < rm.lmat_max_n.size()) rm.lmat_max_n[T] = (uint32_t)Tpts.size();
  rm.lmat_trees[T].cnt_mod = rm.lmat_trees[T].cnt_del = 0;
  boundary_cnt_ = 0; for (uint32_t c = 0; c < ml_.cand_regionid.size(); ++c) if (ml_.cand_regionid[c] != INV) boundary_cnt_++;
  pres_build_();

  return n_regions_ > n_regions_before;
}

void RegionHistogramStats::pres_build_() {
  if (!ml_.active) { pres_on_ = false; return; }
  const size_t NC2 = ml_.col_meta.size();
  pres_bits_.assign(NC2, {});
  pres_stride_.assign(NC2, 0);
  if (pres_degraded_.size() != NC2) pres_degraded_.assign(NC2, 0);
  pres_ci_.clear();
  for (size_t ci = 0; ci < NC2; ++ci) {
    pres_ci_[ml_.col_meta[ci].name] = (int)ci;
    const int W = ml_col_width(ci);
    pres_stride_[ci] = (W + 63) >> 6;
    pres_bits_[ci].assign((size_t)n_regions_ * pres_stride_[ci], 0ull);
  }
  const size_t N = ml_.col_values.empty() ? 0 : ml_.col_values[0].size();
  for (PointId p = 0; p < (PointId)N; ++p) {
    RegionId r = finest_bound_ancestor_(p);
    if (r == kNoRow) r = 0;

    for (size_t ci = 0; ci < NC2; ++ci) {
      const int b = ml_bucket(ci, ml_.col_values[ci][p]);
      if (b >= 0) pres_set_(ci, r, b);
    }
  }
  pres_on_ = true;
}

bool RegionHistogramStats::atom_empty(RegionId r, const ColumnPredicate& a) const {
  if (!pres_on_) return false;
  auto it = pres_ci_.find(a.column());
  if (it == pres_ci_.end()) return false;
  const size_t ci = (size_t)it->second;
  const int stride = pres_stride_[ci];
  const uint64_t* row = &pres_bits_[ci][(size_t)r * stride];
  auto bit = [&](int b) { return (row[b >> 6] >> (b & 63)) & 1ull; };
  using F = ColumnPredicate::Form;
  if (ml_.col_meta[ci].cat) {
    if (a.form() != F::EQ) return false;
    const auto& cc = categorical_.at(ml_.col_meta[ci].name);
    auto cit = cc.code_index.find(a.a());
    if (cit == cc.code_index.end()) return pres_degraded_[ci] == 0;
    return bit(cit->second) == 0;
  }

  const auto& nc = numeric_.at(ml_.col_meta[ci].name);
  const int W = nc.B;
  auto bkt = [&](double x) {
    int b = (int)(std::upper_bound(nc.edges.begin(), nc.edges.end(), x) - nc.edges.begin()) - 1;
    return b < 0 ? 0 : (b >= W ? W - 1 : b);
  };
  int lo = 0, hi = W - 1;
  switch (a.form()) {
    case F::LT:       hi = bkt(a.a()); break;
    case F::GT:       lo = bkt(a.a()); break;
    case F::INTERVAL: lo = bkt(a.a()); hi = bkt(a.b()); break;
    case F::EQ:       lo = hi = bkt(a.a()); break;
  }
  for (int b = lo; b <= hi; ++b) if (bit(b)) return false;
  return true;
}

bool RegionHistogramStats::dnf_empty(RegionId r,
                                     const std::vector<std::vector<ColumnPredicate>>& clauses) const {
  if (!pres_on_ || clauses.empty()) return false;
  for (const auto& cj : clauses) {
    bool clause_empty = false;
    for (const ColumnPredicate& a : cj)
      if (atom_empty(r, a)) { clause_empty = true; break; }
    if (!clause_empty) return false;
  }
  return true;
}

}
