#pragma once

#include <cstddef>

#if defined(__AVX2__) && !defined(HNSW_STATS_SCALAR_ONLY)
#define HNSW_STATS_AVX2_DISTANCE 1
#include <immintrin.h>
#endif

namespace hnsw_stats {

#ifdef HNSW_STATS_AVX2_DISTANCE

inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif

inline float l2_sqr(const float* a, const float* b, int dim) {
#ifdef HNSW_STATS_AVX2_DISTANCE
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc = _mm256_fmadd_ps(d, d, acc);
  }
  float s = hsum256_ps(acc);
  for (; i < dim; ++i) { const float d = a[i] - b[i]; s += d * d; }
  return s;
#else
  float s = 0.f;
  for (int i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
#endif
}

inline float neg_inner_product(const float* a, const float* b, int dim) {
#ifdef HNSW_STATS_AVX2_DISTANCE
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= dim; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  float ip = hsum256_ps(acc);
  for (; i < dim; ++i) ip += a[i] * b[i];
  return -ip;
#else
  float ip = 0.f;
  for (int i = 0; i < dim; ++i) ip += a[i] * b[i];
  return -ip;
#endif
}

}
