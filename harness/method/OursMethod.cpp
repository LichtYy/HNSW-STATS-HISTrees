#include "method/OursMethod.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <random>
#include <stdexcept>
#include <fstream>
#include <string>

#include "common/QueryPredicateGen.hpp"
#include "hnsw_stats/distance/Distance.hpp"
#include "hnsw_stats/predicate/ColumnPredicate.hpp"
#include "hnsw_stats/region/ForestRegionBuilder.hpp"
#include "hnsw_stats/region/HubVoronoiRegionBuilder.hpp"

namespace hnsw_stats {

void OursMethod::build(const Dataset& ds, const BuildParams& bp) {
  const auto t0 = std::chrono::steady_clock::now();
  N_ = ds.N; dim_ = ds.dim;

  bool loaded = false;
  if (!index_cache_.empty()) {
    std::ifstream f(index_cache_, std::ios::binary);
    if (f.good()) {
      f.close();
      index_ = std::make_unique<HnswIndex>(index_cache_, ds.dim);
      loaded = true;
    }
  }
  if (!loaded) {
    int bt = 1; if (const char* e = std::getenv("HS_BUILD_THREADS")) bt = std::max(1, atoi(e));
    index_ = std::make_unique<HnswIndex>(ds.base.data(), ds.N, ds.dim, bp.M, bp.efc,
                                         (unsigned)bt);
  }
  if (!loaded && !index_cache_.empty()) index_->save(index_cache_);
  const auto t_after_graph = std::chrono::steady_clock::now();
  build_t_graph_ = std::chrono::duration<double>(t_after_graph - t0).count();

  std::vector<ColumnView> views;
  views.reserve(ds.columns.size());
  for (const auto& c : ds.columns)
    views.push_back(ColumnView{c.name, c.type == "categorical", &c.values});

  if (multilayer_on_) {

    rm_ = RegionMap{};
    MultiLayerParams ml;
    ml.lmat_min = lmat_min_; ml.lmat_max = lmat_max_; ml.budget_bytes = budget_bytes_;
    ml.delta_range = delta_range_; ml.eps_point = eps_point_; ml.naware_c = naware_c_;
    stats_ = std::make_unique<RegionHistogramStats>(*index_, rm_, views, hist_buckets_, ml);
  } else {

    const char* rb_env = std::getenv("OURS_REGION_BUILDER");
    if (rb_env && std::string(rb_env) == "hubvoronoi") {
      HubVoronoiRegionBuilder rb(region_level_);
      rm_ = rb.build(*index_);
    } else {
      ForestRegionBuilder rb(region_level_);
      rm_ = rb.build(*index_);
    }
    stats_ = std::make_unique<RegionHistogramStats>(rm_, views, hist_buckets_);
  }
  const auto t_after_region = std::chrono::steady_clock::now();
  build_t_region_ = std::chrono::duration<double>(t_after_region - t_after_graph).count();

  cm_ = std::make_unique<EstCostModel>();
  const ScalarColumn* numcol = nullptr;
  for (const auto& c : ds.columns) if (c.type != "categorical") { numcol = &c; break; }
  if (numcol) {
    QuantilePredicates qp(*numcol);
    ColumnPredicate rep = qp.lt(0.1);
    cm_->calibrate(*index_, rm_.region_of, rep);
  } else if (!ds.columns.empty()) {
    QuantilePredicates qp(ds.columns[0]);
    ColumnPredicate rep = qp.eq(qp.categories().front());
    cm_->calibrate(*index_, rm_.region_of, rep);
  }
  cm_->set_repr_region_size(rm_.n_regions ? ds.N / rm_.n_regions : 1);
  auto_theta_ = cm_->theta_repr();

  if (rm_.n_regions > 0 && N_ > 0) {
    const int dim = dim_; const size_t N = N_; const RegionId R = rm_.n_regions;

    std::vector<double> cx((size_t)R * dim, 0.0); std::vector<uint64_t> cnt(R, 0);
    for (size_t u = 0; u < N; ++u) { RegionId r = rm_.region_of[u]; const float* v = index_->vector((PointId)u);
      double* a = &cx[(size_t)r * dim]; for (int j = 0; j < dim; ++j) a[j] += v[j]; ++cnt[r]; }
    for (RegionId r = 0; r < R; ++r) if (cnt[r]) for (int j = 0; j < dim; ++j) cx[(size_t)r * dim + j] /= cnt[r];
    std::vector<double> rho(R, 0.0);
    for (size_t u = 0; u < N; ++u) { RegionId r = rm_.region_of[u]; const float* v = index_->vector((PointId)u);
      const double* c = &cx[(size_t)r * dim]; double s = 0; for (int j = 0; j < dim; ++j) { double d = v[j] - c[j]; s += d * d; }
      rho[r] += std::sqrt(s); }
    for (RegionId r = 0; r < R; ++r) if (cnt[r]) rho[r] /= cnt[r];
    cm_->set_rho_hat(rho);
    { std::vector<float> cxf(cx.begin(), cx.end()); cm_->set_centroids(std::move(cxf), dim); }

    bool calib_loaded = false;
    std::string calib_path;
    if (!index_cache_.empty()) {
      uint64_t h = 1469598103934665603ULL;
      auto mix = [&](uint64_t x) { h ^= x; h *= 1099511628211ULL; };
      for (RegionId rr : rm_.region_of) mix((uint64_t)rr);
      mix((uint64_t)rm_.n_regions);
      char sfx[40]; std::snprintf(sfx, sizeof sfx, ".%016llx.calib", (unsigned long long)h);
      calib_path = index_cache_ + sfx;
      if (std::FILE* cf = std::fopen(calib_path.c_str(), "rb")) {
        calib_loaded = cm_->load_calib(cf); std::fclose(cf);
        if (calib_loaded) std::fprintf(stderr, "[ours] calib cache HIT %s\n", calib_path.c_str());
      }
    }
    if (!calib_loaded) {

    double bsum[5] = {0,0,0,0,0}, bn[5] = {0,0,0,0,0};
    std::vector<uint32_t> st(N, 0), vx(N, 0), rcst(R, 0); std::vector<float> ca(N, 0.f);
    std::vector<uint32_t> rc(R, 0); uint32_t ep = 0;
    const int bf_ef = 50;
    auto fbf = [&](const float* qv, const ColumnPredicate& fp) {
      ++ep; if (ep == 0) { std::fill(st.begin(),st.end(),0u); std::fill(vx.begin(),vx.end(),0u); std::fill(rcst.begin(),rcst.end(),0u); ep = 1; }
      auto D = [&](PointId u) { if (st[u]==ep) return ca[u]; st[u]=ep; float d=l2_sqr(qv,index_->vector(u),dim); ca[u]=d;
        RegionId r=rm_.region_of[u]; if (rcst[r]!=ep){rcst[r]=ep;rc[r]=0;} ++rc[r]; return d; };
      float w = 0; int nres = 0; std::priority_queue<float> sat;
      auto consider = [&](float d) { if (nres<bf_ef){sat.push(d);++nres;w=sat.top();} else if(d<w){sat.push(d);sat.pop();w=sat.top();} };
      PointId cur=index_->entry_point(); float dc=D(cur);
      for (int lc=index_->max_level(); lc>0; --lc){bool ch=true;while(ch){ch=false;for(PointId nb:index_->neighbors_at(cur,lc)){float d=D(nb);if(d<dc){dc=d;cur=nb;ch=true;}}}}
      std::priority_queue<std::pair<float,PointId>,std::vector<std::pair<float,PointId>>,std::greater<>> Q; Q.emplace(dc,cur);
      while(!Q.empty()){auto[k,x]=Q.top();Q.pop(); if(vx[x]==ep)continue; if(nres>=bf_ef&&k>=w)break; vx[x]=ep;
        for(PointId v:index_->neighbors(x)){ if(vx[v]==ep)continue; float d=D(v); if(fp.eval(v))consider(d); Q.emplace(d,v);} }
      for(RegionId r=0;r<R;++r) if(rcst[r]==ep){ double sg=stats_->sigma_hat(r,fp);
        int b=sg<0.01?0:sg<0.05?1:sg<0.1?2:sg<0.3?3:4; bsum[b]+=rc[r]; bn[b]+=1; }
    };
    std::mt19937_64 rng(11); std::uniform_int_distribution<size_t> pickq(0, N-1);
    const double sels[] = {0.005,0.02,0.05,0.1,0.2,0.35};
    int budget = std::getenv("HS_CALIB_BUDGET") ? std::atoi(std::getenv("HS_CALIB_BUDGET")) : 180;
    while (budget > 0) {
      for (const auto& c : ds.columns) {
        if (budget <= 0) break;
        QuantilePredicates qp(c);
        if (c.type == "categorical") { auto cats=qp.categories(); if(cats.empty())continue;
          double cd=cats[rng()%cats.size()]; ColumnPredicate fp=qp.eq(cd);
          fbf(index_->vector((PointId)pickq(rng)), fp); --budget;
        } else { double sl=sels[rng()%6]; ColumnPredicate fp=(rng()&1)?qp.lt(sl):qp.gt(sl);
          fbf(index_->vector((PointId)pickq(rng)), fp); --budget; }
      }
    }
    double mb[5]; for (int i=0;i<5;++i) mb[i]= bn[i]>0 ? bsum[i]/bn[i] : 0.0;
    cm_->set_mbar_buckets(mb);

    const auto tcal = std::chrono::steady_clock::now(); uint64_t calib_delta = 0;
    struct Obs { RegionId r; float sig; uint32_t rc; uint32_t mr; float w; float sd2; };
    std::vector<Obs> obs; obs.reserve(200 * 64);
    std::vector<uint32_t> rsat_s(R, 0);
    auto sampleF12 = [&](const float* qv, const ColumnPredicate& fp) {
      ++ep; if (ep == 0) { std::fill(st.begin(),st.end(),0u); std::fill(vx.begin(),vx.end(),0u); std::fill(rcst.begin(),rcst.end(),0u); ep = 1; }
      auto D = [&](PointId u) { if (st[u]==ep) return ca[u]; st[u]=ep; float d=l2_sqr(qv,index_->vector(u),dim); ca[u]=d; ++calib_delta;
        RegionId r=rm_.region_of[u]; if (rcst[r]!=ep){rcst[r]=ep;rc[r]=0;} ++rc[r]; return d; };
      float w=0; int nres=0; std::priority_queue<float> sat;
      auto consider=[&](float d){ if(nres<bf_ef){sat.push(d);++nres;w=sat.top();} else if(d<w){sat.push(d);sat.pop();w=sat.top();} };
      PointId cur=index_->entry_point(); float dc=D(cur);
      for(int lc=index_->max_level();lc>0;--lc){bool ch=true;while(ch){ch=false;for(PointId nb:index_->neighbors_at(cur,lc)){float d=D(nb);if(d<dc){dc=d;cur=nb;ch=true;}}}}
      std::priority_queue<std::pair<float,PointId>,std::vector<std::pair<float,PointId>>,std::greater<>> Q; Q.emplace(dc,cur);
      while(!Q.empty()){auto[k,x]=Q.top();Q.pop(); if(vx[x]==ep)continue; if(nres>=bf_ef&&k>=w)break; vx[x]=ep;
        for(PointId v:index_->neighbors(x)){ if(vx[v]==ep)continue; float d=D(v); if(fp.eval(v))consider(d); Q.emplace(d,v);} }
      if(nres<bf_ef) return;
      std::fill(rsat_s.begin(),rsat_s.end(),0u);
      for(size_t u=0;u<N;++u) if(fp.eval((PointId)u)) rsat_s[rm_.region_of[u]]++;
      for(RegionId r=0;r<R;++r) if(rcst[r]==ep){ double sg=stats_->sigma_hat(r,fp);
        float sd2 = cm_->centroid(r) ? l2_sqr(qv, cm_->centroid(r), dim) : 0.f; ++calib_delta;
        obs.push_back({r,(float)sg,rc[r],rsat_s[r],w,sd2}); }
    };
    { std::mt19937_64 rng2(23); std::uniform_int_distribution<size_t> pickq2(0, N-1);
      const double sels2[]={0.005,0.02,0.05,0.1,0.2,0.35}; int budget2 = std::getenv("HS_CALIB_BUDGET") ? std::atoi(std::getenv("HS_CALIB_BUDGET")) : 180;
      while(budget2>0){ for(const auto&c:ds.columns){ if(budget2<=0)break; QuantilePredicates qp2(c);
        if(c.type=="categorical"){auto cats=qp2.categories();if(cats.empty())continue;double cd=cats[rng2()%cats.size()];ColumnPredicate fp=qp2.eq(cd);sampleF12(index_->vector((PointId)pickq2(rng2)),fp);--budget2;}
        else{double sl=sels2[rng2()%6];ColumnPredicate fp=(rng2()&1)?qp2.lt(sl):qp2.gt(sl);sampleF12(index_->vector((PointId)pickq2(rng2)),fp);--budget2;} } } }
    const double Cd=cm_->c_dist()>0?cm_->c_dist():1.0, Cp=cm_->c_pred();
    auto sbin=[&](double s){return s<0.01?0:s<0.05?1:s<0.10?2:s<0.30?3:4;};

    const int GTNB=8; std::vector<float> wedges(GTNB+1,0.f);
    { std::vector<float> ws; ws.reserve(obs.size()); for(auto&o:obs)ws.push_back(o.w); std::sort(ws.begin(),ws.end());
      if(!ws.empty()){ for(int b=0;b<=GTNB;++b) wedges[b]=ws[std::min(ws.size()-1,(size_t)((double)b/GTNB*ws.size()))]; wedges[0]=0; wedges[GTNB]=ws.back()*1.0001f+1; } }
    auto wbk=[&](float w){int b=0;while(b<GTNB-1&&w>=wedges[b+1])++b;return b;};
    std::vector<double> gsum((size_t)R*GTNB,0),gcnt((size_t)R*GTNB,0),glsum(GTNB,0),glcnt(GTNB,0);
    for(auto&o:obs){ int b=wbk(o.w); double nr=stats_->region_size(o.r); if(nr<=0)continue; double f=(double)o.rc/nr;
      gsum[(size_t)o.r*GTNB+b]+=f; gcnt[(size_t)o.r*GTNB+b]+=1; glsum[b]+=f; glcnt[b]+=1; }
    std::vector<float> gtfrac((size_t)R*GTNB,1.f);
    for(RegionId r=0;r<R;++r)for(int b=0;b<GTNB;++b){ size_t i=(size_t)r*GTNB+b;
      gtfrac[i]= gcnt[i]>0 ? (float)(gsum[i]/gcnt[i]) : (glcnt[b]>0 ? (float)(glsum[b]/glcnt[b]) : 1.f); }
    cm_->set_gtraverse(std::move(gtfrac), wedges, GTNB);

    double bcheap[5]={0,0,0,0,0}, bcnt[5]={0,0,0,0,0};
    for(auto&o:obs){ int bi=sbin(o.sig); bcnt[bi]+=1; if((double)o.rc < (double)o.mr) bcheap[bi]+=1; }
    const double sedge[6]={0.0,0.01,0.05,0.10,0.30,1.0}; double theta_emp=1.0;
    for(int bi=0;bi<5;++bi) if(bcnt[bi]>0 && bcheap[bi]/bcnt[bi]>=0.5){ theta_emp=sedge[bi]; break; }
    cm_->set_theta_emp(theta_emp);

    double cG[5]={0,0,0,0,0}, cGp[5]={0,0,0,0,0};
    for(auto&o:obs){ int bi=sbin(o.sig); double nr=stats_->region_size(o.r); double cBFS=nr*Cp+o.sig*nr*Cd;
      bool gbf = cm_->mbar_sigma(o.sig)*(Cd+Cp) < cBFS;
      double rho=cm_->rho_hat(o.r); if(rho<=1e-9)rho=1.0;
      double sw=std::sqrt(std::max(0.f,o.w)), sd=std::sqrt(std::max(0.f,o.sd2));
      double fr=(sw-(sd-rho))/(2.0*rho); fr=fr<0?0:(fr>1?1:fr); bool gpbf = nr*fr*(Cd+Cp) < cBFS;
      cG[bi]  += gbf  ? (double)o.rc : (double)o.mr;
      cGp[bi] += gpbf ? (double)o.rc : (double)o.mr; }
    uint8_t gate[5]; for(int bi=0;bi<5;++bi) gate[bi] = (cGp[bi] < cG[bi]) ? 1 : 0;
    cm_->set_gate(gate);
    calib_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now()-tcal).count();
    calib_delta_ = calib_delta;

    }
    if (!calib_loaded && !calib_path.empty()) {
      if (std::FILE* cf = std::fopen(calib_path.c_str(), "wb")) {
        if (cm_->save_calib(cf)) std::fprintf(stderr, "[ours] calib cached -> %s\n", calib_path.c_str());
        std::fclose(cf);
      }
    }
  }

  build_t_calib_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_after_region).count();
  searcher_ = std::make_unique<OursSearch>(*index_, rm_, *stats_, *cm_);

  if (cm_->c_edge() > 0.0) searcher_->set_dnf_cedge(cm_->c_edge());
  std::fprintf(stderr, "[calib] c_dist=%.2fns c_pred=%.2fns c_edge=%.3fns (M*Cedge=%.2fns)\n",
               cm_->c_dist()*1e9, cm_->c_pred()*1e9, cm_->c_edge()*1e9,
               searcher_->dnf_degree() * cm_->c_edge() * 1e9);
  build_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void OursMethod::inc_add(const std::vector<float>& vec, const std::vector<double>& colvals) {
  const PointId id = (PointId)N_;
  static_cast<HnswIndex&>(*index_).add(vec.data(), id);
  N_ = id + 1;
  pending_adds_.emplace_back(id, colvals);
}
void OursMethod::inc_flush() {
  if (pending_adds_.empty()) return;
  static_cast<HnswIndex&>(*index_).reproject();
  for (auto& [id, cv] : pending_adds_) stats_->on_insert(rm_, id, cv);
  pending_adds_.clear();
  rebuild_searcher_();
}
bool OursMethod::inc_update(PointId id, size_t col, double new_val) {

  if (!(stats_ && stats_->presence_on() && use_presence_cfg_) && !hist_staled_) {
    hist_staled_ = true; searcher_->set_transit_skip(false);
  }
  const bool grew = stats_->on_update(rm_, id, col, new_val);
  if (grew) rebuild_searcher_();
  else searcher_->invalidate_tadj();
  return grew;
}
bool OursMethod::inc_delete(PointId id) {
  static_cast<HnswIndex&>(*index_).remove(id);
  const bool grew = stats_->on_delete(rm_, id);
  if (grew) rebuild_searcher_();
  else searcher_->invalidate_tadj();
  return grew;
}

SearchResult OursMethod::search(const Query& q, const Predicate& phi, int ef) {
  const auto* cp = dynamic_cast<const ColumnPredicate*>(&phi);
  if (!cp) throw std::runtime_error("OursMethod: only ColumnPredicate supported (single-column) for now");
  SearchResult r;
  r.topk_ids = searcher_->search(q.data, *cp, ef, K_);
  r.n_delta = searcher_->delta();
  r.n_delta_satisfying = searcher_->delta_satisfying();
  r.n_struct_visits = searcher_->struct_visits(); r.n_pred_evals = searcher_->pred_evals();
  return r;
}

SearchResult OursMethod::search_strategy(const Query& q, const Predicate& phi,
                                         const std::vector<region_search_kind>& strat, int ef) {
  SearchResult r;
  r.topk_ids = searcher_->search_with_strategy(q.data, phi, strat, ef, K_);
  r.n_delta = searcher_->delta();
  r.n_delta_satisfying = searcher_->delta_satisfying();
  r.n_struct_visits = searcher_->struct_visits(); r.n_pred_evals = searcher_->pred_evals();
  return r;
}

SearchResult OursMethod::search_mode(const Query& q, const ColumnPredicate& phi, int ef,
                                     BoundaryMode mode, double sigma_gate, bool force_broad,
                                     bool force_bestfirst) {
  SearchResult r;
  r.topk_ids = searcher_->search_boundary(q.data, phi, mode, ef, K_, sigma_gate, force_broad, force_bestfirst);
  r.n_delta = searcher_->delta();
  r.n_delta_satisfying = searcher_->delta_satisfying();
  r.n_struct_visits = searcher_->struct_visits(); r.n_pred_evals = searcher_->pred_evals();
  return r;
}

}
