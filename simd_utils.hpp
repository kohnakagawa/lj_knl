#pragma once

#include <x86intrin.h>

void print256(__m256d r) {
  union {
    __m256d r;
    double elem[4];
  } tmp;
  tmp.r = r;
  printf("%.10f %.10f %.10f %.10f\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3]);
}

void print512(__m512d r) {
  union {
    __m512d r;
    double elem[8];
  } tmp;
  tmp.r = r;
  printf("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3],
         tmp.elem[4], tmp.elem[5], tmp.elem[6], tmp.elem[7]);
}

void print256i(__m256i r) {
  union {
    __m256i r;
    int32_t elem[8];
  } tmp;
  tmp.r = r;
  printf("%d %d %d %d %d %d %d %d\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3],
         tmp.elem[4], tmp.elem[5], tmp.elem[6], tmp.elem[7]);
}

static inline __m512d _mm512_load2_m256d(const double* hiaddr,
                                         const double* loaddr) {
  __m512d ret;
  ret = _mm512_castpd256_pd512(_mm256_load_pd(loaddr));
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(hiaddr), 0x1);
  return ret;
}

static inline void _mm512_store2_m256d(double* hiaddr,
                                       double* loaddr,
                                       const __m512d& dat) {
  _mm256_store_pd(loaddr, _mm512_castpd512_pd256(dat));
  _mm256_store_pd(hiaddr, _mm512_extractf64x4_pd(dat, 0x1));
}

static inline void _mm256_xyz_store_pd(double* addr,
                                       const __m256d& dat) {
  _mm256_maskstore_pd(addr,
                      _mm256_set_epi64x(0x0,
                                        0xffffffffffffffff,
                                        0xffffffffffffffff,
                                        0xffffffffffffffff),
                      dat);
}

static inline void _mm512_xyz_store2_m256d(double* hiaddr,
                                           double* loaddr,
                        const __m512d& dat) {
  _mm256_xyz_store_pd(loaddr,
                      _mm512_castpd512_pd256(dat));
  _mm256_xyz_store_pd(hiaddr,
                      _mm512_extractf64x4_pd(dat, 0x1));
}

static inline void transpose_4x4x2(__m512d& va,
                                   __m512d& vb,
                                   __m512d& vc,
                                   __m512d& vd) {
  const auto t_a = _mm512_unpacklo_pd(va, vb);
  const auto t_b = _mm512_unpackhi_pd(va, vb);
  const auto t_c = _mm512_unpacklo_pd(vc, vd);
  const auto t_d = _mm512_unpackhi_pd(vc, vd);

  va = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vb = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vc = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
  vd = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_d);
}

static inline void transpose_4x4x2(const __m512d& va,
                                   const __m512d& vb,
                                   const __m512d& vc,
                                   const __m512d& vd,
                                   __m512d& vx,
                                   __m512d& vy,
                                   __m512d& vz) {
  __m512d t_a = _mm512_unpacklo_pd(va, vb);
  __m512d t_b = _mm512_unpackhi_pd(va, vb);
  __m512d t_c = _mm512_unpacklo_pd(vc, vd);
  __m512d t_d = _mm512_unpackhi_pd(vc, vd);

  vx = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vy = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vz = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
}

static inline void transpose_4x4(const __m256d& va,
                                 const __m256d& vb,
                                 const __m256d& vc,
                                 const __m256d& vd,
                                 __m256d& vx,
                                 __m256d& vy,
                                 __m256d& vz) {
  const auto tmp0 = _mm256_unpacklo_pd(va, vb);
  const auto tmp1 = _mm256_unpackhi_pd(va, vb);
  const auto tmp2 = _mm256_unpacklo_pd(vc, vd);
  const auto tmp3 = _mm256_unpackhi_pd(vc, vd);
  vx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
  vy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
  vz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
}
