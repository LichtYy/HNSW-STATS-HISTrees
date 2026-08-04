#include "common/AttributesLoader.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>

namespace hnsw_stats {

namespace {
using json = nlohmann::json;

double as_number(const json& v) {
  if (v.is_number()) return v.get<double>();
  if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
  return 0.0;
}

std::string as_category(const json& v) {
  if (v.is_null()) return "<null>";
  if (v.is_string()) return v.get<std::string>();
  if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
  if (v.is_number()) return std::to_string(v.get<long long>());
  if (v.is_array() && !v.empty()) return as_category(v[0]);
  return "<other>";
}
}

std::vector<ScalarColumn> load_jsonl_columns(const std::string& path,
                                             const std::vector<ColumnSpec>& specs,
                                             size_t max_rows, unsigned threads) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open attributes file: " + path);
  std::string buf((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());

  std::vector<std::pair<size_t, size_t>> lines;
  size_t s = 0;
  for (size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] == '\n') {
      lines.emplace_back(s, i);
      s = i + 1;
      if (max_rows && lines.size() >= max_rows) break;
    }
  }
  if (s < buf.size() && (!max_rows || lines.size() < max_rows))
    lines.emplace_back(s, buf.size());
  const size_t n = lines.size();

  std::vector<ScalarColumn> cols(specs.size());
  std::vector<std::vector<std::string>> cat_raw(specs.size());
  for (size_t c = 0; c < specs.size(); ++c) {
    cols[c].name = specs[c].field;
    cols[c].type = specs[c].type;
    cols[c].orderable = (specs[c].type == "numeric");
    cols[c].values.resize(n);
    if (specs[c].type == "categorical") cat_raw[c].resize(n);
  }

  auto worker = [&](size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; ++i) {
      json j = json::parse(buf.data() + lines[i].first,
                           buf.data() + lines[i].second);
      for (size_t c = 0; c < specs.size(); ++c) {
        const json& v = j.contains(specs[c].field) ? j[specs[c].field] : json();
        if (specs[c].type == "numeric") cols[c].values[i] = as_number(v);
        else                            cat_raw[c][i] = as_category(v);
      }
    }
  };

  const unsigned T = std::max(1u, std::min<unsigned>(threads, (unsigned)n));
  std::vector<std::thread> pool;
  const size_t chunk = (n + T - 1) / T;
  for (unsigned t = 0; t < T; ++t) {
    const size_t i0 = t * chunk, i1 = std::min(n, i0 + chunk);
    if (i0 >= i1) break;
    pool.emplace_back(worker, i0, i1);
  }
  for (auto& th : pool) th.join();

  for (size_t c = 0; c < specs.size(); ++c) {
    if (specs[c].type != "categorical") continue;
    std::map<std::string, double> code;
    for (const auto& sname : cat_raw[c]) code.emplace(sname, 0.0);
    double k = 0;
    for (auto& [name, v] : code) v = k++;
    for (size_t i = 0; i < n; ++i) cols[c].values[i] = code[cat_raw[c][i]];
    std::fprintf(stderr, "[attrs] categorical '%s': %zu codes:",
                 specs[c].field.c_str(), code.size());
    for (auto& [name, v] : code)
      std::fprintf(stderr, " %g=%s", v, name.c_str());
    std::fprintf(stderr, "\n");
  }
  return cols;
}

std::vector<ScalarColumn> load_csv_columns(const std::string& path,
                                           const std::vector<ColumnSpec>& specs,
                                           size_t max_rows, unsigned threads) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open csv: " + path);
  std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  std::vector<std::pair<size_t, size_t>> lines;
  size_t s = 0;
  for (size_t i = 0; i < buf.size(); ++i)
    if (buf[i] == '\n') { lines.emplace_back(s, i); s = i + 1; }
  if (s < buf.size()) lines.emplace_back(s, buf.size());
  if (lines.empty()) throw std::runtime_error("empty csv: " + path);

  std::vector<std::string> header;
  { std::string h = buf.substr(lines[0].first, lines[0].second - lines[0].first);
    if (!h.empty() && h.back() == '\r') h.pop_back();
    size_t p = 0; while (p <= h.size()) { size_t c = h.find(',', p);
      if (c == std::string::npos) c = h.size();
      header.push_back(h.substr(p, c - p)); p = c + 1; } }
  std::vector<int> col_idx(specs.size(), -1);
  for (size_t c = 0; c < specs.size(); ++c)
    for (size_t h = 0; h < header.size(); ++h) if (header[h] == specs[c].field) col_idx[c] = (int)h;

  size_t n = lines.size() - 1;
  if (max_rows && max_rows < n) n = max_rows;

  std::vector<ScalarColumn> cols(specs.size());
  std::vector<std::vector<std::string>> cat_raw(specs.size());
  for (size_t c = 0; c < specs.size(); ++c) {
    cols[c].name = specs[c].field; cols[c].type = specs[c].type;
    cols[c].orderable = (specs[c].type == "numeric"); cols[c].values.resize(n);
    if (specs[c].type == "categorical") cat_raw[c].resize(n);
  }

  auto worker = [&](size_t i0, size_t i1) {
    std::vector<std::string> fields;
    for (size_t i = i0; i < i1; ++i) {
      const auto& [ls, le] = lines[i + 1];
      fields.clear(); size_t p = ls;
      while (p <= le) { size_t c = buf.find(',', p); if (c == std::string::npos || c > le) c = le;
        fields.emplace_back(buf.substr(p, c - p)); p = c + 1; }
      for (size_t c = 0; c < specs.size(); ++c) {
        const int ci = col_idx[c]; std::string v = (ci >= 0 && ci < (int)fields.size()) ? fields[ci] : "";
        if (specs[c].type == "numeric") { try { cols[c].values[i] = v.empty() ? 0.0 : std::stod(v); } catch (...) { cols[c].values[i] = 0.0; } }
        else cat_raw[c][i] = v.empty() ? "<null>" : v;
      }
    }
  };
  const unsigned T = std::max(1u, std::min<unsigned>(threads, (unsigned)n));
  std::vector<std::thread> pool; const size_t chunk = (n + T - 1) / T;
  for (unsigned t = 0; t < T; ++t) { const size_t a = t * chunk, b = std::min(n, a + chunk); if (a < b) pool.emplace_back(worker, a, b); }
  for (auto& th : pool) th.join();

  for (size_t c = 0; c < specs.size(); ++c) {
    if (specs[c].type != "categorical") continue;
    std::map<std::string, double> code; for (auto& s2 : cat_raw[c]) code.emplace(s2, 0.0);
    double k = 0; for (auto& [name, v] : code) v = k++;
    for (size_t i = 0; i < n; ++i) cols[c].values[i] = code[cat_raw[c][i]];
  }
  return cols;
}

}
