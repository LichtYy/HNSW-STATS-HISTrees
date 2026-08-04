#include "common/Fvecs.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace hnsw_stats {
namespace {

template <typename T>
void read_vecs(const std::string& path, std::vector<T>& out, size_t& n,
               int& dim, size_t max_rows) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open vecs file: " + path);

  f.seekg(0, std::ios::end);
  const std::streamoff bytes = f.tellg();
  f.seekg(0, std::ios::beg);

  int32_t d0 = 0;
  f.read(reinterpret_cast<char*>(&d0), sizeof(int32_t));
  if (!f || d0 <= 0) throw std::runtime_error("bad vecs header: " + path);
  dim = d0;

  const std::streamoff rec_bytes =
      static_cast<std::streamoff>(sizeof(int32_t)) + d0 * sizeof(T);
  if (bytes % rec_bytes != 0)
    throw std::runtime_error("vecs size not a multiple of record size: " + path);
  size_t total = static_cast<size_t>(bytes / rec_bytes);
  if (max_rows && max_rows < total) total = max_rows;

  n = total;
  out.resize(total * static_cast<size_t>(d0));
  f.seekg(0, std::ios::beg);
  for (size_t i = 0; i < total; ++i) {
    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    if (d != d0) throw std::runtime_error("non-uniform dim in vecs: " + path);
    f.read(reinterpret_cast<char*>(out.data() + i * d0), d0 * sizeof(T));
    if (!f) throw std::runtime_error("truncated vecs file: " + path);
  }
}

}

VecsData read_fvecs(const std::string& path, size_t max_rows) {
  VecsData v;
  read_vecs<float>(path, v.data, v.n, v.dim, max_rows);
  return v;
}

IVecsData read_ivecs(const std::string& path, size_t max_rows) {
  IVecsData v;
  read_vecs<int32_t>(path, v.data, v.n, v.dim, max_rows);
  return v;
}

}
