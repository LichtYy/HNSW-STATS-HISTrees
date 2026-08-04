#include "common/DatasetLoader.hpp"

#include <toml++/toml.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "common/AttributesLoader.hpp"
#include "common/Fvecs.hpp"

namespace fs = std::filesystem;

namespace hnsw_stats {

std::string DatasetLoader::repo_root() {
  fs::path p = fs::current_path();
  for (;; p = p.parent_path()) {
    if (fs::exists(p / "configs" / "datasets.toml")) return p.string();
    if (!p.has_relative_path() && p == p.parent_path()) break;
  }
  throw std::runtime_error(
      "could not locate repo root (configs/datasets.toml) from CWD");
}

Dataset DatasetLoader::load(const std::string& dataset_id, size_t max_base,
                            size_t max_queries) {
  const fs::path root = repo_root();
  const auto cfg = toml::parse_file((root / "configs" / "datasets.toml").string());

  const auto* entry = cfg[dataset_id].as_table();
  if (!entry) throw std::runtime_error("dataset not in datasets.toml: " + dataset_id);

  const auto get_str = [&](const char* k) -> std::string {
    auto v = (*entry)[k].value<std::string>();
    if (!v) throw std::runtime_error("missing key '" + std::string(k) +
                                     "' for dataset " + dataset_id);
    return *v;
  };

  const fs::path dir = root / get_str("path");
  const fs::path base_path = dir / get_str("base_file");
  const fs::path query_path = dir / get_str("query_file");

  VecsData base = read_fvecs(base_path.string(), max_base);
  VecsData query = read_fvecs(query_path.string(), max_queries);
  if (query.dim != base.dim)
    throw std::runtime_error("base/query dim mismatch for " + dataset_id);

  Dataset ds;
  ds.id = dataset_id;
  ds.N = base.n;
  ds.dim = base.dim;
  ds.base = std::move(base.data);
  ds.queries = std::move(query.data);

  if (auto af = (*entry)["attribute_file"].value<std::string>()) {
    std::vector<ColumnSpec> specs;
    if (const auto* arr = (*entry)["columns"].as_array()) {
      for (const auto& el : *arr) {
        if (const auto* t = el.as_table()) {
          ColumnSpec s;
          s.field = (*t)["field"].value_or<std::string>("");
          s.type = (*t)["type"].value_or<std::string>("numeric");
          if (!s.field.empty()) specs.push_back(s);
        }
      }
    }

    if (specs.empty()) {
      if (auto ac = (*entry)["auto_columns"].value<std::string>()) {
        std::ifstream hf((dir / *af).string());
        std::string hdr; std::getline(hf, hdr);
        if (!hdr.empty() && hdr.back() == '\r') hdr.pop_back();
        size_t p = 0;
        while (p <= hdr.size()) { size_t c = hdr.find(',', p); if (c == std::string::npos) c = hdr.size();
          ColumnSpec s; s.field = hdr.substr(p, c - p); s.type = *ac;
          if (!s.field.empty()) specs.push_back(s); p = c + 1; }
      }
    }
    if (!specs.empty()) {
      const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
      const std::string apath = (dir / *af).string();
      ds.columns = (af->size() >= 4 && af->substr(af->size() - 4) == ".csv")
                       ? load_csv_columns(apath, specs, ds.N, threads)
                       : load_jsonl_columns(apath, specs, ds.N, threads);
    }
  }

  auto tp = (*entry)["text_pool_file"].value<std::string>();
  if (const char* e = std::getenv("HS_TEXT_POOL_FILE")) tp = std::string(e);
  if (tp) {
    std::string tname = (*entry)["text_pool_name"].value_or<std::string>("abstract");
    if (const char* e = std::getenv("HS_TEXT_POOL_NAME")) tname = e;
    std::ifstream f((dir / *tp).string(), std::ios::binary);
    if (!f) throw std::runtime_error("text_pool_file missing for " + dataset_id + ": " + *tp);
    uint64_t tn = 0;
    f.read(reinterpret_cast<char*>(&tn), 8);
    if (tn < ds.N) throw std::runtime_error("text pool rows < N for " + dataset_id);
    TextColumn tc;
    tc.name = tname;
    tc.offsets.resize(tn + 1);
    f.read(reinterpret_cast<char*>(tc.offsets.data()), 8 * (tn + 1));
    const uint64_t need = tc.offsets[ds.N];
    tc.bytes.resize(need);
    f.read(tc.bytes.data(), (std::streamsize)need);
    if (!f) throw std::runtime_error("text pool truncated read for " + dataset_id);
    tc.offsets.resize(ds.N + 1);
    ds.text_columns.push_back(std::move(tc));
  }
  auto cfile = (*entry)["code_file"].value<std::string>();
  if (const char* e = std::getenv("HS_CODE_FILE")) cfile = std::string(e);
  if (cfile) {
    std::string cname = (*entry)["code_name"].value_or<std::string>("kwcode");
    if (const char* e = std::getenv("HS_CODE_NAME")) cname = e;
    std::ifstream f((dir / *cfile).string(), std::ios::binary);
    if (!f) throw std::runtime_error("code_file missing for " + dataset_id + ": " + *cfile);
    std::vector<uint16_t> raw(ds.N);
    f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(2 * ds.N));
    if (!f) throw std::runtime_error("code_file truncated read for " + dataset_id);
    ScalarColumn sc;
    sc.name = cname;
    sc.type = "categorical";
    sc.values.assign(raw.begin(), raw.end());
    ds.columns.push_back(std::move(sc));
  }
  return ds;
}

}
