#include <iostream>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <random>
#include <cmath>
#include <chrono>
#include <x86intrin.h>
#include <sys/stat.h>
//----------------------------------------------------------------------
const double density = 1.0;
// const double density = 0.5;
const int N = 400000;
#ifdef REACTLESS
const int MAX_PAIRS = 60 * N;
#else
const int MAX_PAIRS = 30 * N;
#endif
const double L = 50.0;
const double dt = 0.001;

#ifdef REACTLESS
const char* pairlist_cache_file_name = "pair_all.dat";
#else
const char* pairlist_cache_file_name = "pair.dat";
#endif

struct double4 {double x, y, z, w;};
typedef double4 Vec;

Vec* q = nullptr;
Vec* p = nullptr;

int particle_number = 0;
int number_of_pairs = 0;
int* number_of_partners = nullptr;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int32_t* pointer = nullptr;
int32_t pointer2[N];
int* sorted_list = nullptr;

const double CUTOFF_LENGTH = 3.0;
const double SEARCH_LENGTH = 3.3;
const double CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
//----------------------------------------------------------------------
void
print256(__m256d r) {
  double *a = (double*)(&r);
  printf("%.10f %.10f %.10f %.10f\n", a[0], a[1], a[2], a[3]);
}
//----------------------------------------------------------------------
void
print512(__m512d r) {
  union {
    __m512d r;
    double elem[8];
  } tmp;
  tmp.r = r;
  printf("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3],
         tmp.elem[4], tmp.elem[5], tmp.elem[6], tmp.elem[7]);
}
//----------------------------------------------------------------------
void
print256i(__m256i r) {
  union {
    __m256i r;
    int32_t elem[8];
  } tmp;
  tmp.r = r;
  printf("%d %d %d %d %d %d %d %d\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3],
         tmp.elem[4], tmp.elem[5], tmp.elem[6], tmp.elem[7]);
}
//----------------------------------------------------------------------
void
add_particle(double x, double y, double z) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<double> ud(0.0, 0.1);
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
  particle_number++;
}
//----------------------------------------------------------------------
void
register_pair(int index1, int index2) {
  int i, j;
#ifdef REACTLESS
  i = index1;
  j = index2;
#else
  if (index1 < index2) {
    i = index1;
    j = index2;
  } else {
    i = index2;
    j = index1;
  }
#endif
  i_particles[number_of_pairs] = i;
  j_particles[number_of_pairs] = j;
  number_of_partners[i]++;
  number_of_pairs++;
}
//----------------------------------------------------------------------
void
sortpair(void){
  const int pn = particle_number;
  int pos = 0;
  pointer[0] = 0;
  for (int i = 0; i < pn - 1; i++) {
    pos += number_of_partners[i];
    pointer[i + 1] = pos;
  }
  for (int i = 0; i < pn; i++) {
    pointer2[i] = 0;
  }
  const int s = number_of_pairs;
  for (int k = 0; k < s; k++) {
    int i = i_particles[k];
    int j = j_particles[k];
    int index = pointer[i] + pointer2[i];
    sorted_list[index] = j;
    pointer2[i] ++;
  }
}
//----------------------------------------------------------------------
void
makepair(void) {
  const double SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    number_of_partners[i] = 0;
  }
#ifdef REACTLESS
  for (int i = 0; i < particle_number; i++) {
    for (int j = 0; j < particle_number; j++) {
      if (i == j) continue;
#else
  for (int i = 0; i < particle_number - 1; i++) {
    for (int j = i + 1; j < particle_number; j++) {
#endif
      const double dx = q[i].x - q[j].x;
      const double dy = q[i].y - q[j].y;
      const double dz = q[i].z - q[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < SL2) {
        register_pair(i, j);
      }
    }
  }
}
//----------------------------------------------------------------------
void
allocate(void) {
  posix_memalign((void**)(&q), 64, sizeof(Vec) * N);
  posix_memalign((void**)(&p), 64, sizeof(Vec) * N);

  posix_memalign((void**)(&number_of_partners), 64, sizeof(int) * N);
  posix_memalign((void**)(&pointer), 64, sizeof(int32_t) * N);
  posix_memalign((void**)(&sorted_list), 64, sizeof(int32_t) * MAX_PAIRS);

  std::fill(number_of_partners,
            number_of_partners + N,
            0);
  std::fill(pointer,
            pointer + N,
            0);
  std::fill(sorted_list,
            sorted_list + MAX_PAIRS,
            0);
}
//----------------------------------------------------------------------
void
deallocate(void) {
  free(q);
  free(p);
  free(number_of_partners);
  free(pointer);
  free(sorted_list);
}
//----------------------------------------------------------------------
void
random_shfl() {
  std::mt19937 mt(10);
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto kp = pointer[i];
    const auto np = number_of_partners[i];
    std::shuffle(&sorted_list[kp], &sorted_list[kp + np], mt);
  }
}
//----------------------------------------------------------------------
void
init(void) {
  const double s = 1.0 / std::pow(density * 0.25, 1.0 / 3.0);
  const double hs = s * 0.5;
  int sx = static_cast<int>(L / s);
  int sy = static_cast<int>(L / s);
  int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        double x = ix*s;
        double y = iy*s;
        double z = iz*s;
        add_particle(x     ,y   ,z);
        add_particle(x     ,y+hs,z+hs);
        add_particle(x+hs  ,y   ,z+hs);
        add_particle(x+hs  ,y+hs,z);
      }
    }
  }
  for (int i = 0; i < particle_number; i++) {
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
  }
}
//----------------------------------------------------------------------
void
force_pair(void){
  const auto nps = number_of_pairs;
  for (int k = 0; k < nps; k++) {
    const int i = i_particles[k];
    const int j = j_particles[k];
    double dx = q[j].x - q[i].x;
    double dy = q[j].y - q[i].y;
    double dz = q[j].z - q[i].z;
    double r2 = (dx * dx + dy * dy + dz * dz);
    if (r2 > CL2) continue;
    double r6 = r2 * r2 * r2;
    double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
    p[i].x += df * dx;
    p[i].y += df * dy;
    p[i].z += df * dz;
    p[j].x -= df * dx;
    p[j].y -= df * dy;
    p[j].z -= df * dz;
  }
}
//----------------------------------------------------------------------
void
force_sorted(void) {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = q[i].x;
    const auto qy_key = q[i].y;
    const auto qz_key = q[i].z;
    const auto np = number_of_partners[i];
    double pfx = 0, pfy = 0, pfz = 0;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
      p[j].x -= df*dx;
      p[j].y -= df*dy;
      p[j].z -= df*dz;
    } // end of k loop
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_next(void) {
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const double qx_key = q[i].x;
    const double qy_key = q[i].y;
    const double qz_key = q[i].z;
    double pfx = 0, pfy = 0, pfz = 0;
    const int kp = pointer[i];
    int ja = sorted_list[kp];
    double dxa = q[ja].x - qx_key;
    double dya = q[ja].y - qy_key;
    double dza = q[ja].z - qz_key;
    double df = 0.0;
    double dxb = 0.0, dyb = 0.0, dzb = 0.0;
    int jb = 0;

    const int np = number_of_partners[i];
    for (int k = kp; k < np + kp; k++) {

      const double dx = dxa;
      const double dy = dya;
      const double dz = dza;
      double r2 = (dx * dx + dy * dy + dz * dz);
      const int j = ja;
      ja = sorted_list[k + 1];
      dxa = q[ja].x - qx_key;
      dya = q[ja].y - qy_key;
      dza = q[ja].z - qz_key;
      if (r2 > CL2)continue;
      pfx += df * dxb;
      pfy += df * dyb;
      pfz += df * dzb;
      p[jb].x -= df * dxb;
      p[jb].y -= df * dyb;
      p[jb].z -= df * dzb;
      const double r6 = r2 * r2 * r2;
      df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      jb = j;
      dxb = dx;
      dyb = dy;
      dzb = dz;
    }

    p[jb].x -= df * dxb;
    p[jb].y -= df * dyb;
    p[jb].z -= df * dzb;

    p[i].x += pfx + df * dxb;
    p[i].y += pfy + df * dyb;
    p[i].z += pfz + df * dzb;
  }
}
//----------------------------------------------------------------------
static inline __m512d
_mm512_load2_m256d(const double* hiaddr,
                   const double* loaddr) {
  __m512d ret;
  ret = _mm512_castpd256_pd512(_mm256_load_pd(loaddr));
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(hiaddr), 0x1);
  return ret;
}
//----------------------------------------------------------------------
static inline void
_mm512_store2_m256d(double* hiaddr,
                    double* loaddr,
                    const __m512d& dat) {
  _mm256_store_pd(loaddr, _mm512_castpd512_pd256(dat));
  _mm256_store_pd(hiaddr, _mm512_extractf64x4_pd(dat, 0x1));
}
//----------------------------------------------------------------------
static inline void
_mm256_xyz_store_pd(double* addr,
                    const __m256d& dat) {
  _mm256_maskstore_pd(addr,
                      _mm256_set_epi64x(0x0,
                                        0xffffffffffffffff,
                                        0xffffffffffffffff,
                                        0xffffffffffffffff),
                      dat);
}

static inline void
_mm512_xyz_store2_m256d(double* hiaddr,
                        double* loaddr,
                        const __m512d& dat) {
  _mm256_xyz_store_pd(loaddr,
                      _mm512_castpd512_pd256(dat));
  _mm256_xyz_store_pd(hiaddr,
                      _mm512_extractf64x4_pd(dat, 0x1));
}
//----------------------------------------------------------------------
static inline void
transpose_4x4x2(__m512d& va,
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
//----------------------------------------------------------------------
static inline void
transpose_4x4x2(const __m512d& va,
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
//----------------------------------------------------------------------
// intrin (with scatter & gather)
void
force_intrin_v1(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(q[i].x);
    const auto vqyi = _mm512_set1_pd(q[i].y);
    const auto vqzi = _mm512_set1_pd(q[i].z);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                            2);

      const auto vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
      const auto vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
      const auto vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

      const auto vdx = _mm512_sub_pd(vqxj, vqxi);
      const auto vdy = _mm512_sub_pd(vqyj, vqyi);
      const auto vdz = _mm512_sub_pd(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_pd(vindex, &p[0].x, 8);
      auto vpyj = _mm512_i32gather_pd(vindex, &p[0].y, 8);
      auto vpzj = _mm512_i32gather_pd(vindex, &p[0].z, 8);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));
      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

      vpxj = _mm512_fnmadd_pd(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz, vpzj);

      _mm512_i32scatter_pd(&p[0].x, vindex, vpxj, 8);
      _mm512_i32scatter_pd(&p[0].y, vindex, vpyj, 8);
      _mm512_i32scatter_pd(&p[0].z, vindex, vpzj, 8);
    } // end of k loop
    p[i].x += _mm512_reduce_add_pd(vpxi);
    p[i].y += _mm512_reduce_add_pd(vpyi);
    p[i].z += _mm512_reduce_add_pd(vpzi);

    // remaining loop
    double pfx = 0.0, pfy = 0.0, pfz = 0.0;
    auto qx_key = q[i].x, qy_key = q[i].y, qz_key = q[i].z;
    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      pfx    += df * dx;
      pfy    += df * dy;
      pfz    += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    } // end of k loop
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin (with scatter & gather) + remove remaining loop
void
force_intrin_v2(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi64(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(q[i].x);
    const auto vqyi = _mm512_set1_pd(q[i].y);
    const auto vqzi = _mm512_set1_pd(q[i].z);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi64(np);
    auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                   3LL, 2LL, 1LL, 0LL);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    for (int k = 0; k < num_loop; k += 8) {
      const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                            2);

      const auto mask = _mm512_cmp_epi64_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);

      const auto vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
      const auto vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
      const auto vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

      const auto vdx = _mm512_sub_pd(vqxj, vqxi);
      const auto vdy = _mm512_sub_pd(vqyj, vqyi);
      const auto vdz = _mm512_sub_pd(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_pd(vindex, &p[0].x, 8);
      auto vpyj = _mm512_i32gather_pd(vindex, &p[0].y, 8);
      auto vpzj = _mm512_i32gather_pd(vindex, &p[0].z, 8);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));

      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

      vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

      vpxj = _mm512_fnmadd_pd(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz, vpzj);

      _mm512_mask_i32scatter_pd(&p[0].x, mask, vindex, vpxj, 8);
      _mm512_mask_i32scatter_pd(&p[0].y, mask, vindex, vpyj, 8);
      _mm512_mask_i32scatter_pd(&p[0].z, mask, vindex, vpzj, 8);

      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
    } // end of k loop

    p[i].x += _mm512_reduce_add_pd(vpxi);
    p[i].y += _mm512_reduce_add_pd(vpyi);
    p[i].z += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin (with scatter & gather) + swp + remove remaining loop
void
force_intrin_v3(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi64(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(q[i].x);
    const auto vqyi = _mm512_set1_pd(q[i].y);
    const auto vqzi = _mm512_set1_pd(q[i].z);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi64(np);
    auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                   3LL, 2LL, 1LL, 0LL);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    // initial force calculation
    // load position
    auto vindex_a = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp])),
                                      2);
    auto mask_a = _mm512_cmp_epi64_mask(vk_idx,
                                        vnp,
                                        _MM_CMPINT_LT);
    auto vqxj = _mm512_i32gather_pd(vindex_a, &q[0].x, 8);
    auto vqyj = _mm512_i32gather_pd(vindex_a, &q[0].y, 8);
    auto vqzj = _mm512_i32gather_pd(vindex_a, &q[0].z, 8);

    // calc distance
    auto vdx_a = _mm512_sub_pd(vqxj, vqxi);
    auto vdy_a = _mm512_sub_pd(vqyj, vqyi);
    auto vdz_a = _mm512_sub_pd(vqzj, vqzi);
    auto vr2 = _mm512_fmadd_pd(vdz_a,
                               vdz_a,
                               _mm512_fmadd_pd(vdy_a,
                                               vdy_a,
                                               _mm512_mul_pd(vdx_a, vdx_a)));

    // calc force norm
    auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

    auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                             _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
    vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                               vzero, vdf);
    vdf = _mm512_mask_blend_pd(mask_a, vzero, vdf);

    for (int k = 8; k < num_loop; k += 8) {
      // write back j particle momentum
      vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_i32gather_pd(vindex_a, &p[0].x, 8);
      auto vpyj = _mm512_i32gather_pd(vindex_a, &p[0].y, 8);
      auto vpzj = _mm512_i32gather_pd(vindex_a, &p[0].z, 8);

      vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
      _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
      _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);

      vk_idx = _mm512_add_epi32(vk_idx, vpitch);

      // load position
      auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                        2);
      auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_i32gather_pd(vindex_b, &q[0].x, 8);
      vqyj = _mm512_i32gather_pd(vindex_b, &q[0].y, 8);
      vqzj = _mm512_i32gather_pd(vindex_b, &q[0].z, 8);

      // calc distance
      auto vdx_b = _mm512_sub_pd(vqxj, vqxi);
      auto vdy_b = _mm512_sub_pd(vqyj, vqyi);
      auto vdz_b = _mm512_sub_pd(vqzj, vqzi);
      vr2 = _mm512_fmadd_pd(vdz_b,
                            vdz_b,
                            _mm512_fmadd_pd(vdy_b,
                                            vdy_b,
                                            _mm512_mul_pd(vdx_b,
                                                          vdx_b)));

      // calc force norm
      vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                          _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);
      vdf = _mm512_mask_blend_pd(mask_b, vzero, vdf);

      // send to next
      vindex_a = vindex_b;
      mask_a   = mask_b;
      vdx_a    = vdx_b;
      vdy_a    = vdy_b;
      vdz_a    = vdz_b;
    } // end of k loop

    // final write back momentum
    // write back j particle momentum
    vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
    vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
    vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

    auto vpxj = _mm512_i32gather_pd(vindex_a, &p[0].x, 8);
    auto vpyj = _mm512_i32gather_pd(vindex_a, &p[0].y, 8);
    auto vpzj = _mm512_i32gather_pd(vindex_a, &p[0].z, 8);

    vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
    vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
    vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

    _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
    _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
    _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);

    // write back i particle momentum
    p[i].x += _mm512_reduce_add_pd(vpxi);
    p[i].y += _mm512_reduce_add_pd(vpyi);
    p[i].z += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin (without scatter & gather)
void
force_intrin_v4(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    auto vqi = _mm512_castpd256_pd512(_mm256_load_pd(&q[i].x));
    vqi = _mm512_insertf64x4(vqi, _mm512_castpd512_pd256(vqi), 0x1);
    auto vpi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto j_b = sorted_list[k + kp + 1], j_a = sorted_list[k + kp    ];
      const auto j_d = sorted_list[k + kp + 3], j_c = sorted_list[k + kp + 2];
      const auto j_f = sorted_list[k + kp + 5], j_e = sorted_list[k + kp + 4];
      const auto j_h = sorted_list[k + kp + 7], j_g = sorted_list[k + kp + 6];

      auto vqj_ba = _mm512_load2_m256d(&q[j_b].x, &q[j_a].x);
      auto vqj_dc = _mm512_load2_m256d(&q[j_d].x, &q[j_c].x);
      auto vqj_fe = _mm512_load2_m256d(&q[j_f].x, &q[j_e].x);
      auto vqj_hg = _mm512_load2_m256d(&q[j_h].x, &q[j_g].x);

      auto vdq_ba = _mm512_sub_pd(vqj_ba, vqi);
      auto vdq_dc = _mm512_sub_pd(vqj_dc, vqi);
      auto vdq_fe = _mm512_sub_pd(vqj_fe, vqi);
      auto vdq_hg = _mm512_sub_pd(vqj_hg, vqi);

      __m512d vdx, vdy, vdz;
      transpose_4x4x2(vdq_ba, vdq_dc, vdq_fe, vdq_hg,
                      vdx, vdy, vdz);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));

      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      auto vdf_ba = _mm512_permutex_pd(vdf, 0x00);
      auto vdf_dc = _mm512_permutex_pd(vdf, 0x55);
      auto vdf_fe = _mm512_permutex_pd(vdf, 0xaa);
      auto vdf_hg = _mm512_permutex_pd(vdf, 0xff);

      vpi = _mm512_fmadd_pd(vdf_ba, vdq_ba, vpi);
      auto vpj_ba = _mm512_load2_m256d(&p[j_b].x, &p[j_a].x);
      vpj_ba = _mm512_fnmadd_pd(vdf_ba, vdq_ba, vpj_ba);
      _mm512_store2_m256d(&p[j_b].x, &p[j_a].x, vpj_ba);

      vpi = _mm512_fmadd_pd(vdf_dc, vdq_dc, vpi);
      auto vpj_dc = _mm512_load2_m256d(&p[j_d].x, &p[j_c].x);
      vpj_dc = _mm512_fnmadd_pd(vdf_dc, vdq_dc, vpj_dc);
      _mm512_store2_m256d(&p[j_d].x, &p[j_c].x, vpj_dc);

      vpi = _mm512_fmadd_pd(vdf_fe, vdq_fe, vpi);
      auto vpj_fe = _mm512_load2_m256d(&p[j_f].x, &p[j_e].x);
      vpj_fe = _mm512_fnmadd_pd(vdf_fe, vdq_fe, vpj_fe);
      _mm512_store2_m256d(&p[j_f].x, &p[j_e].x, vpj_fe);

      vpi = _mm512_fmadd_pd(vdf_hg, vdq_hg, vpi);
      auto vpj_hg = _mm512_load2_m256d(&p[j_h].x, &p[j_g].x);
      vpj_hg = _mm512_fnmadd_pd(vdf_hg, vdq_hg, vpj_hg);
      _mm512_store2_m256d(&p[j_h].x, &p[j_g].x, vpj_hg);
    } // end of k loop
    vpi = _mm512_add_pd(vpi,
                        _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                              vpi));
    vpi = _mm512_add_pd(vpi, _mm512_castpd256_pd512(_mm256_load_pd(&p[i].x)));
    _mm256_store_pd(&p[i].x, _mm512_castpd512_pd256(vpi));

    // remaining loop
    double pfx = 0.0, pfy = 0.0, pfz = 0.0;
    auto qx_key = q[i].x, qy_key = q[i].y, qz_key = q[i].z;
#pragma novector
    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      pfx    += df * dx;
      pfy    += df * dy;
      pfz    += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    } // end of k loop
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v1_reactless(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(q[i].x);
    const auto vqyi = _mm512_set1_pd(q[i].y);
    const auto vqzi = _mm512_set1_pd(q[i].z);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                            2);

      const auto vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
      const auto vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
      const auto vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

      const auto vdx = _mm512_sub_pd(vqxj, vqxi);
      const auto vdy = _mm512_sub_pd(vqyj, vqyi);
      const auto vdz = _mm512_sub_pd(vqzj, vqzi);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));
      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);
    } // end of k loop

    auto pfx = p[i].x + _mm512_reduce_add_pd(vpxi);
    auto pfy = p[i].y + _mm512_reduce_add_pd(vpyi);
    auto pfz = p[i].z + _mm512_reduce_add_pd(vpzi);
    auto qx_key = q[i].x, qy_key = q[i].y, qz_key = q[i].z;
#pragma novector
    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
    } // end of k loop
    p[i].x = pfx;
    p[i].y = pfy;
    p[i].z = pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v2_reactless(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn     = particle_number;

  for (int i = 0; i < pn; i++) {
    auto vqi = _mm512_castpd256_pd512(_mm256_load_pd(&q[i].x));
    vqi = _mm512_insertf64x4(vqi, _mm512_castpd512_pd256(vqi), 0x1);
    auto vpi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto j_a = sorted_list[kp + k    ], j_e = sorted_list[kp + k + 4];
      const auto j_b = sorted_list[kp + k + 1], j_f = sorted_list[kp + k + 5];
      const auto j_c = sorted_list[kp + k + 2], j_g = sorted_list[kp + k + 6];
      const auto j_d = sorted_list[kp + k + 3], j_h = sorted_list[kp + k + 7];

      auto vqj_ea = _mm512_load2_m256d(&q[j_e].x, &q[j_a].x);
      auto vqj_fb = _mm512_load2_m256d(&q[j_f].x, &q[j_b].x);
      auto vqj_gc = _mm512_load2_m256d(&q[j_g].x, &q[j_c].x);
      auto vqj_hd = _mm512_load2_m256d(&q[j_h].x, &q[j_d].x);

      auto vdq_ea = _mm512_sub_pd(vqj_ea, vqi);
      auto vdq_fb = _mm512_sub_pd(vqj_fb, vqi);
      auto vdq_gc = _mm512_sub_pd(vqj_gc, vqi);
      auto vdq_hd = _mm512_sub_pd(vqj_hd, vqi);

      __m512d vdx, vdy, vdz;
      transpose_4x4x2(vdq_ea, vdq_fb, vdq_gc, vdq_hd,
                      vdx, vdy, vdz);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));
      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      auto vdf_ea = _mm512_permutex_pd(vdf, 0x00);
      auto vdf_fb = _mm512_permutex_pd(vdf, 0x55);
      auto vdf_gc = _mm512_permutex_pd(vdf, 0xaa);
      auto vdf_hd = _mm512_permutex_pd(vdf, 0xff);

      vpi = _mm512_fmadd_pd(vdf_ea, vdq_ea, vpi);
      vpi = _mm512_fmadd_pd(vdf_fb, vdq_fb, vpi);
      vpi = _mm512_fmadd_pd(vdf_gc, vdq_gc, vpi);
      vpi = _mm512_fmadd_pd(vdf_hd, vdq_hd, vpi);
    } // end of k loop
    vpi = _mm512_add_pd(vpi,
                        _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                              vpi));
    vpi = _mm512_add_pd(vpi, _mm512_castpd256_pd512(_mm256_load_pd(&p[i].x)));

    _mm256_store_pd(&p[i].x, _mm512_castpd512_pd256(vpi));

    double pfx = p[i].x, pfy = p[i].y, pfz = p[i].z;
#pragma novector
    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (dx * dx + dy * dy + dz * dz);
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
    } // end of k loop
    p[i].x = pfx;
    p[i].y = pfy;
    p[i].z = pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v3_reactless(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi64(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(q[i].x);
    const auto vqyi = _mm512_set1_pd(q[i].y);
    const auto vqzi = _mm512_set1_pd(q[i].z);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi64(np);
    auto vk_idx = _mm512_set_epi64(7, 6, 5, 4,
                                   3, 2, 1, 0);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    for (int k = 0; k < num_loop; k += 8) {
      const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                            2);

      const auto mask = _mm512_cmp_epi64_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);

      const auto vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
      const auto vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
      const auto vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

      const auto vdx = _mm512_sub_pd(vqxj, vqxi);
      const auto vdy = _mm512_sub_pd(vqyj, vqyi);
      const auto vdz = _mm512_sub_pd(vqzj, vqzi);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));
      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

      vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
    } // end of k loop

    p[i].x += _mm512_reduce_add_pd(vpxi);
    p[i].y += _mm512_reduce_add_pd(vpyi);
    p[i].z += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
void
measure(void(*pfunc)(), const char *name) {
  const auto beg = std::chrono::system_clock::now();
  const int LOOP = 100;
  for (int i = 0; i < LOOP; i++) {
    pfunc();
  }
  const auto end = std::chrono::system_clock::now();
  const long dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  fprintf(stderr, "N=%d, %s %ld [ms]\n", particle_number, name, dur);
}
//----------------------------------------------------------------------
void
loadpair(void){
  std::ifstream ifs(pairlist_cache_file_name,std::ios::binary);
  ifs.read((char*)&number_of_pairs,sizeof(int));
  ifs.read((char*)number_of_partners,sizeof(int)*N);
  ifs.read((char*)i_particles,sizeof(int)*MAX_PAIRS);
  ifs.read((char*)j_particles,sizeof(int)*MAX_PAIRS);
}
//----------------------------------------------------------------------
void
savepair(void){
  makepair();
  random_shfl();
  std::ofstream ofs(pairlist_cache_file_name,std::ios::binary);
  ofs.write((char*)&number_of_pairs,sizeof(int));
  ofs.write((char*)number_of_partners,sizeof(int)*N);
  ofs.write((char*)i_particles,sizeof(int)*MAX_PAIRS);
  ofs.write((char*)j_particles,sizeof(int)*MAX_PAIRS);
}
//----------------------------------------------------------------------
void
check(void) {
  for (int i = 0; i < number_of_pairs; i++) {
    assert(sorted_list[i] >= 0 && sorted_list[i] < particle_number);
  }
  for (int i = 0; i < particle_number; i++) {
    assert(number_of_partners[i] >= 0 && number_of_partners[i] < particle_number);
    assert(pointer[i] >= 0 && pointer[i] <= number_of_pairs);
  }
}
//----------------------------------------------------------------------
void
print_result(void){
  for (int i = 0; i < 5; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
  for (int i = particle_number-5; i < particle_number; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
}
//----------------------------------------------------------------------
int
main(void) {
  allocate();
  init();
  struct stat st;
  int ret = stat(pairlist_cache_file_name, &st);
  if (ret == 0) {
    std::cerr << "A pair-file is found. I use it." << std::endl;
    loadpair();
  } else {
    std::cerr << "Make pairlist." << std::endl;
    savepair();
  }
  std::cerr << "Number of pairs: " << number_of_pairs << std::endl;
  sortpair();
  check();
#ifdef PAIR
  measure(&force_pair, "pair");
  print_result();
#elif NEXT
  measure(&force_next, "next");
  print_result();
#elif SORTED
  measure(&force_sorted, "sorted");
  print_result();
#elif defined INTRIN_v1 && defined REACTLESS
  measure(&force_intrin_v1_reactless, "intrin_v1_reactless");
  print_result();
#elif defined INTRIN_v2 && defined REACTLESS
  measure(&force_intrin_v2_reactless, "intrin_v2_reactless");
  print_result();
#elif defined INTRIN_v3 && defined REACTLESS
  measure(&force_intrin_v3_reactless, "intrin_v3_reactless");
  print_result();
#elif INTRIN_v1
  measure(&force_intrin_v1, "intrin_v1");
  print_result();
#elif INTRIN_v2
  measure(&force_intrin_v2, "intrin_v2");
  print_result();
#elif INTRIN_v3
  measure(&force_intrin_v3, "intrin_v3");
  print_result();
#elif INTRIN_v4
  measure(&force_intrin_v4, "intrin_v4");
  print_result();
#endif
  deallocate();
}
//----------------------------------------------------------------------
