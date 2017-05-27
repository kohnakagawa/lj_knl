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
//const double density = 0.5;
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
enum {X = 0, Y, Z, W, PX, PY, PZ, WX};
__attribute__((aligned(64))) double z[N][8];
typedef double v4df __attribute__((vector_size(32)));
typedef double v8df __attribute__((vector_size(64)));

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
copy_to_z(void){
  for(int i=0;i<particle_number;i++){
    z[i][X] = q[i].x;
    z[i][Y] = q[i].y;
    z[i][Z] = q[i].z;
    z[i][PX] = p[i].x;
    z[i][PY] = p[i].y;
    z[i][PZ] = p[i].z;
  }
}
//----------------------------------------------------------------------
void
copy_from_z(void){
  for(int i=0;i<particle_number;i++){
    q[i].x = z[i][X];
    q[i].y = z[i][Y];
    q[i].z = z[i][Z];
    p[i].x = z[i][PX];
    p[i].y = z[i][PY];
    p[i].z = z[i][PZ];
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

      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
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
      // load position
      auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                        2);
      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
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
// intrin (without scatter & gather + swp)
// NOTE: modified from https://github.com/kaityo256/lj_simdstep/blob/master/step6/force.cpp#L742
void
force_intrin_v5(void) {
  const auto pn = particle_number;
  const auto vzero = _mm512_setzero_pd();
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);

  for (int i = 0; i < pn; i++) {
    auto vqi = _mm512_castpd256_pd512(_mm256_load_pd(&q[i].x));
    vqi = _mm512_insertf64x4(vqi, _mm512_castpd512_pd256(vqi), 0x1);
    auto vpf = _mm512_setzero_pd();
    const auto kp = pointer[i];

    auto j_b_s = sorted_list[kp + 1], j_a_s = sorted_list[kp    ];
    auto j_d_s = sorted_list[kp + 3], j_c_s = sorted_list[kp + 2];
    auto j_f_s = sorted_list[kp + 5], j_e_s = sorted_list[kp + 4];
    auto j_h_s = sorted_list[kp + 7], j_g_s = sorted_list[kp + 6];

    auto vqj_ba = _mm512_load2_m256d(&q[j_b_s].x, &q[j_a_s].x);
    auto vqj_dc = _mm512_load2_m256d(&q[j_d_s].x, &q[j_c_s].x);
    auto vqj_fe = _mm512_load2_m256d(&q[j_f_s].x, &q[j_e_s].x);
    auto vqj_hg = _mm512_load2_m256d(&q[j_h_s].x, &q[j_g_s].x);

    auto vdq_ba_s = _mm512_sub_pd(vqj_ba, vqi);
    auto vdq_dc_s = _mm512_sub_pd(vqj_dc, vqi);
    auto vdq_fe_s = _mm512_sub_pd(vqj_fe, vqi);
    auto vdq_hg_s = _mm512_sub_pd(vqj_hg, vqi);

    auto vdf      = _mm512_setzero_pd();

    auto vdq_ba_t = _mm512_setzero_pd();
    auto vdq_dc_t = _mm512_setzero_pd();
    auto vdq_fe_t = _mm512_setzero_pd();
    auto vdq_hg_t = _mm512_setzero_pd();

    int j_b_t = 0, j_a_t = 0;
    int j_d_t = 0, j_c_t = 0;
    int j_f_t = 0, j_e_t = 0;
    int j_h_t = 0, j_g_t = 0;
    const auto np = number_of_partners[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto j_b = j_b_s, j_a = j_a_s;
      const auto j_d = j_d_s, j_c = j_c_s;
      const auto j_f = j_f_s, j_e = j_e_s;
      const auto j_h = j_h_s, j_g = j_g_s;
      auto vdq_ba = vdq_ba_s;
      auto vdq_dc = vdq_dc_s;
      auto vdq_fe = vdq_fe_s;
      auto vdq_hg = vdq_hg_s;

      j_b_s = sorted_list[kp + k + 9 ]; j_a_s = sorted_list[kp + k + 8 ];
      j_d_s = sorted_list[kp + k + 11]; j_c_s = sorted_list[kp + k + 10];
      j_f_s = sorted_list[kp + k + 13]; j_e_s = sorted_list[kp + k + 12];
      j_h_s = sorted_list[kp + k + 15]; j_g_s = sorted_list[kp + k + 14];

      __m512d vdx, vdy, vdz;
      transpose_4x4x2(vdq_ba, vdq_dc, vdq_fe, vdq_hg,
                      vdx, vdy, vdz);

      auto vdf_ba = _mm512_permutex_pd(vdf, 0x00);
      auto vdf_dc = _mm512_permutex_pd(vdf, 0x55);
      auto vdf_fe = _mm512_permutex_pd(vdf, 0xaa);
      auto vdf_hg = _mm512_permutex_pd(vdf, 0xff);

      vqj_ba = _mm512_load2_m256d(&q[j_b_s].x, &q[j_a_s].x);
      vdq_ba_s = _mm512_sub_pd(vqj_ba, vqi);
      vpf = _mm512_fmadd_pd(vdf_ba, vdq_ba_t, vpf);

      auto vpj_ba_t = _mm512_load2_m256d(&p[j_b_t].x, &p[j_a_t].x);
      vpj_ba_t = _mm512_fnmadd_pd(vdf_ba, vdq_ba_t, vpj_ba_t);
      _mm512_store2_m256d(&p[j_b_t].x, &p[j_a_t].x, vpj_ba_t);

      vqj_dc = _mm512_load2_m256d(&q[j_d_s].x, &q[j_c_s].x);
      vdq_dc_s = _mm512_sub_pd(vqj_dc, vqi);
      vpf = _mm512_fmadd_pd(vdf_dc, vdq_dc_t, vpf);

      auto vpj_dc_t = _mm512_load2_m256d(&p[j_d_t].x, &p[j_c_t].x);
      vpj_dc_t = _mm512_fnmadd_pd(vdf_dc, vdq_dc_t, vpj_dc_t);
      _mm512_store2_m256d(&p[j_d_t].x, &p[j_c_t].x, vpj_dc_t);

      vqj_fe = _mm512_load2_m256d(&q[j_f_s].x, &q[j_e_s].x);
      vdq_fe_s = _mm512_sub_pd(vqj_fe, vqi);
      vpf = _mm512_fmadd_pd(vdf_fe, vdq_fe_t, vpf);

      auto vpj_fe_t = _mm512_load2_m256d(&p[j_f_t].x, &p[j_e_t].x);
      vpj_fe_t = _mm512_fnmadd_pd(vdf_fe, vdq_fe_t, vpj_fe_t);
      _mm512_store2_m256d(&p[j_f_t].x, &p[j_e_t].x, vpj_fe_t);

      vqj_hg = _mm512_load2_m256d(&q[j_h_s].x, &q[j_g_s].x);
      vdq_hg_s = _mm512_sub_pd(vqj_hg, vqi);
      vpf = _mm512_fmadd_pd(vdf_hg, vdq_hg_t, vpf);

      auto vpj_hg_t = _mm512_load2_m256d(&p[j_h_t].x, &p[j_g_t].x);
      vpj_hg_t = _mm512_fnmadd_pd(vdf_hg, vdq_hg_t, vpj_hg_t);
      _mm512_store2_m256d(&p[j_h_t].x, &p[j_g_t].x, vpj_hg_t);

      auto vr2 = _mm512_fmadd_pd(vdz,
                                 vdz,
                                 _mm512_fmadd_pd(vdy,
                                                 vdy,
                                                 _mm512_mul_pd(vdx, vdx)));
      auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                          _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      j_b_t = j_b; j_a_t = j_a;
      j_d_t = j_d; j_c_t = j_c;
      j_f_t = j_f; j_e_t = j_e;
      j_h_t = j_h; j_g_t = j_g;
      vdq_ba_t = vdq_ba;
      vdq_dc_t = vdq_dc;
      vdq_fe_t = vdq_fe;
      vdq_hg_t = vdq_hg;
    } // end of k loop
    auto vdf_ba = _mm512_permutex_pd(vdf, 0x00);
    auto vdf_dc = _mm512_permutex_pd(vdf, 0x55);
    auto vdf_fe = _mm512_permutex_pd(vdf, 0xaa);
    auto vdf_hg = _mm512_permutex_pd(vdf, 0xff);

    auto vpj_ba_t = _mm512_load2_m256d(&p[j_b_t].x, &p[j_a_t].x);
    vpj_ba_t = _mm512_fnmadd_pd(vdf_ba, vdq_ba_t, vpj_ba_t);
    _mm512_store2_m256d(&p[j_b_t].x, &p[j_a_t].x, vpj_ba_t);

    auto vpj_dc_t = _mm512_load2_m256d(&p[j_d_t].x, &p[j_c_t].x);
    vpj_dc_t = _mm512_fnmadd_pd(vdf_dc, vdq_dc_t, vpj_dc_t);
    _mm512_store2_m256d(&p[j_d_t].x, &p[j_c_t].x, vpj_dc_t);

    auto vpj_fe_t = _mm512_load2_m256d(&p[j_f_t].x, &p[j_e_t].x);
    vpj_fe_t = _mm512_fnmadd_pd(vdf_fe, vdq_fe_t, vpj_fe_t);
    _mm512_store2_m256d(&p[j_f_t].x, &p[j_e_t].x, vpj_fe_t);

    auto vpj_hg_t = _mm512_load2_m256d(&p[j_h_t].x, &p[j_g_t].x);
    vpj_hg_t = _mm512_fnmadd_pd(vdf_hg, vdq_hg_t, vpj_hg_t);
    _mm512_store2_m256d(&p[j_h_t].x, &p[j_g_t].x, vpj_hg_t);

    auto vpi = _mm512_castpd256_pd512(_mm256_load_pd(&p[i].x));

    vpf = _mm512_fmadd_pd(vdf_ba, vdq_ba_t, vpf);
    vpf = _mm512_fmadd_pd(vdf_dc, vdq_dc_t, vpf);
    vpf = _mm512_fmadd_pd(vdf_fe, vdq_fe_t, vpf);
    vpf = _mm512_fmadd_pd(vdf_hg, vdq_hg_t, vpf);
    vpf = _mm512_add_pd(vpf,
                        _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                              vpf));

    vpi = _mm512_add_pd(vpf, vpi);


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
// intrin gather and scatter, swp, 8 bytes, remaining loop opt
void
force_intrin_v6(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi64(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(z[i][X]);
    const auto vqyi = _mm512_set1_pd(z[i][Y]);
    const auto vqzi = _mm512_set1_pd(z[i][Z]);

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
                                      3);
    auto mask_a = _mm512_cmp_epi64_mask(vk_idx,
                                        vnp,
                                        _MM_CMPINT_LT);
    auto vqxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][X], 8);
    auto vqyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][Y], 8);
    auto vqzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][Z], 8);

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
      // load position
      auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                        3);
      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
      auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0][X], 8);
      vqyj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0][Y], 8);
      vqzj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0][Z], 8);

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

      // write back j particle momentum
      vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][PX], 8);
      auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][PY], 8);
      auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][PZ], 8);

      vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_pd(&z[0][PX], mask_a, vindex_a, vpxj, 8);
      _mm512_mask_i32scatter_pd(&z[0][PY], mask_a, vindex_a, vpyj, 8);
      _mm512_mask_i32scatter_pd(&z[0][PZ], mask_a, vindex_a, vpzj, 8);

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

    auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][PX], 8);
    auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][PY], 8);
    auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0][PZ], 8);

    vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
    vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
    vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

    _mm512_mask_i32scatter_pd(&z[0][PX], mask_a, vindex_a, vpxj, 8);
    _mm512_mask_i32scatter_pd(&z[0][PY], mask_a, vindex_a, vpyj, 8);
    _mm512_mask_i32scatter_pd(&z[0][PZ], mask_a, vindex_a, vpzj, 8);

    // write back i particle momentum
    z[i][PX] += _mm512_reduce_add_pd(vpxi);
    z[i][PY] += _mm512_reduce_add_pd(vpyi);
    z[i][PZ] += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin (with scatter & gather) + swp + remove remaining loop PF (momentum)
void
force_intrin_v7(void) {
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
    auto vqxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &q[0].x, 8);
    auto vqyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &q[0].y, 8);
    auto vqzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &q[0].z, 8);

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
      // load position
      auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                        2);
      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
      auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &q[0].x, 8);
      vqyj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &q[0].y, 8);
      vqzj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &q[0].z, 8);

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

      // write back j particle momentum
      vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &p[0].x, 8);
      auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &p[0].y, 8);
      auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &p[0].z, 8);

      vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
      _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
      _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);


      // calc force norm
      vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                          _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      // prefetch next line
      _mm512_mask_prefetch_i32gather_pd(vindex_b, mask_b, &p[0].x, 8, _MM_HINT_T0);

      // cutoff
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

    auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &p[0].x, 8);
    auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &p[0].y, 8);
    auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &p[0].z, 8);

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
// intrin (with scatter & gather) + swp + remove remaining loop PF (pos and momentum)
void
force_intrin_v8(void) {
  const auto vc24   = _mm512_set1_pd(24.0 * dt);
  const auto vc48   = _mm512_set1_pd(48.0 * dt);
  const auto vcl2   = _mm512_set1_pd(CL2);
  const auto vzero  = _mm512_setzero_pd();
  const auto pn     = particle_number;
  const auto vpitch = _mm512_set1_epi32(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(q[i].x);
    const auto vqyi = _mm512_set1_pd(q[i].y);
    const auto vqzi = _mm512_set1_pd(q[i].z);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi32(np);
    auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                   11, 10, 9,  8,
                                   7,  6,  5,  4,
                                   3,  2,  1,  0);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    // initial force calculation
    // load position
    auto vindex_a = _mm512_slli_epi32(_mm512_loadu_si512(&sorted_list[kp]),
                                      2);
    auto mask_a = _mm512_cmp_epi32_mask(vk_idx,
                                        vnp,
                                        _MM_CMPINT_LT);
    auto vqxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &q[0].x, 8);
    auto vqyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &q[0].y, 8);
    auto vqzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &q[0].z, 8);

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
      // load position
      auto vindex_b = _mm512_slli_epi32(_mm512_loadu_si512(&sorted_list[kp + k]),
                                        2);
      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
      auto mask_b = _mm512_cmp_epi32_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                      _mm512_castsi512_si256(vindex_b), &q[0].x, 8);
      vqyj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                      _mm512_castsi512_si256(vindex_b), &q[0].y, 8);
      vqzj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                      _mm512_castsi512_si256(vindex_b), &q[0].z, 8);

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

      // write back j particle momentum
      vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                           _mm512_castsi512_si256(vindex_a), &p[0].x, 8);
      auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                           _mm512_castsi512_si256(vindex_a), &p[0].y, 8);
      auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                           _mm512_castsi512_si256(vindex_a), &p[0].z, 8);

      vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_pd(&p[0].x, mask_a,
                                _mm512_castsi512_si256(vindex_a), vpxj, 8);
      _mm512_mask_i32scatter_pd(&p[0].y, mask_a,
                                _mm512_castsi512_si256(vindex_a), vpyj, 8);
      _mm512_mask_i32scatter_pd(&p[0].z, mask_a,
                                _mm512_castsi512_si256(vindex_a), vpzj, 8);

      // calc force norm
      vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                          _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      // prefetch next line
      _mm512_mask_prefetch_i32gather_pd(_mm512_castsi512_si256(vindex_b),
                                        mask_b, &p[0].x, 8, _MM_HINT_T0);
      __mmask8 mask_b_hi = mask_b >> 8;
      _mm512_mask_prefetch_i32gather_pd(_mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(vindex_b), 1)),
                                        mask_b_hi, &q[0].x, 8, _MM_HINT_T0);

      // cutoff
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

    auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &p[0].x, 8);
    auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &p[0].y, 8);
    auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &p[0].z, 8);

    vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
    vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
    vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

    _mm512_mask_i32scatter_pd(&p[0].x, mask_a,
                              _mm512_castsi512_si256(vindex_a), vpxj, 8);
    _mm512_mask_i32scatter_pd(&p[0].y, mask_a,
                              _mm512_castsi512_si256(vindex_a), vpyj, 8);
    _mm512_mask_i32scatter_pd(&p[0].z, mask_a,
                              _mm512_castsi512_si256(vindex_a), vpzj, 8);

    // write back i particle momentum
    p[i].x += _mm512_reduce_add_pd(vpxi);
    p[i].y += _mm512_reduce_add_pd(vpyi);
    p[i].z += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin gather and scatter, swp, 8 bytes, remaining loop opt, PF (pos and momentum)
void
force_intrin_v9(void) {
  const auto vc24   = _mm512_set1_pd(24.0 * dt);
  const auto vc48   = _mm512_set1_pd(48.0 * dt);
  const auto vcl2   = _mm512_set1_pd(CL2);
  const auto vzero  = _mm512_setzero_pd();
  const auto pn     = particle_number;
  const auto vpitch = _mm512_set1_epi32(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(z[i][X]);
    const auto vqyi = _mm512_set1_pd(z[i][Y]);
    const auto vqzi = _mm512_set1_pd(z[i][Z]);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi32(np);
    auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                   11, 10, 9,  8,
                                   7,  6,  5,  4,
                                   3,  2,  1,  0);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    // initial force calculation
    // load position
    auto vindex_a = _mm512_slli_epi32(_mm512_loadu_si512(&sorted_list[kp]),
                                      3);
    auto mask_a = _mm512_cmp_epi32_mask(vk_idx,
                                        vnp,
                                        _MM_CMPINT_LT);
    auto vqxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &z[0][X], 8);
    auto vqyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &z[0][Y], 8);
    auto vqzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &z[0][Z], 8);

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
      // load position
      auto vindex_b = _mm512_slli_epi32(_mm512_loadu_si512(&sorted_list[kp + k]),
                                        3);
      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
      auto mask_b = _mm512_cmp_epi32_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                      _mm512_castsi512_si256(vindex_b), &z[0][X], 8);
      vqyj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                      _mm512_castsi512_si256(vindex_b), &z[0][Y], 8);
      vqzj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                      _mm512_castsi512_si256(vindex_b), &z[0][Z], 8);

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

      // write back j particle momentum
      vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                           _mm512_castsi512_si256(vindex_a), &z[0][PX], 8);
      auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                           _mm512_castsi512_si256(vindex_a), &z[0][PY], 8);
      auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                           _mm512_castsi512_si256(vindex_a), &z[0][PZ], 8);

      vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_pd(&z[0][PX], mask_a,
                                _mm512_castsi512_si256(vindex_a), vpxj, 8);
      _mm512_mask_i32scatter_pd(&z[0][PY], mask_a,
                                _mm512_castsi512_si256(vindex_a), vpyj, 8);
      _mm512_mask_i32scatter_pd(&z[0][PZ], mask_a,
                                _mm512_castsi512_si256(vindex_a), vpzj, 8);

      // calc force norm
      vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                          _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      // prefetch next line
      __mmask8 mask_b_hi = mask_b >> 8;
      _mm512_mask_prefetch_i32gather_pd(_mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(vindex_b), 1)),
                                        mask_b_hi, &z[0][X], 8, _MM_HINT_T0);

      // cutoff
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

    auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &z[0][PX], 8);
    auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &z[0][PY], 8);
    auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                         _mm512_castsi512_si256(vindex_a), &z[0][PZ], 8);

    vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
    vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
    vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

    _mm512_mask_i32scatter_pd(&z[0][PX], mask_a,
                              _mm512_castsi512_si256(vindex_a), vpxj, 8);
    _mm512_mask_i32scatter_pd(&z[0][PY], mask_a,
                              _mm512_castsi512_si256(vindex_a), vpyj, 8);
    _mm512_mask_i32scatter_pd(&z[0][PZ], mask_a,
                              _mm512_castsi512_si256(vindex_a), vpzj, 8);

    // write back i particle momentum
    z[i][PX] += _mm512_reduce_add_pd(vpxi);
    z[i][PY] += _mm512_reduce_add_pd(vpyi);
    z[i][PZ] += _mm512_reduce_add_pd(vpzi);
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

      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
    } // end of k loop

    p[i].x += _mm512_reduce_add_pd(vpxi);
    p[i].y += _mm512_reduce_add_pd(vpyi);
    p[i].z += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_sorted_z_intrin(void) {
  const int pn = particle_number;
  const v8df vzero = _mm512_setzero_pd();
  const v8df vcl2 = _mm512_set1_pd(CL2);
  const v8df vc24 = _mm512_set1_pd(24.0*dt);
  const v8df vc48 = _mm512_set1_pd(48.0*dt);
  const __m512i idx = _mm512_set_epi64(3,2,1,0,7,6,5,4);
  const __m512i idx_q = _mm512_set_epi64(11,10,9,8,3,2,1,0);
  const __m512i idx_p = _mm512_set_epi64(15,14,13,12,7,6,5,4);
  const __m512i idx_f12 = _mm512_set_epi64(1,1,1,1,0,0,0,0);
  const __m512i idx_f34 = _mm512_set_epi64(3,3,3,3,2,2,2,2);
  const __m512i idx_f56 = _mm512_set_epi64(5,5,5,5,4,4,4,4);
  const __m512i idx_f78 = _mm512_set_epi64(7,7,7,7,6,6,6,6);
  const __m512i idx_0123 = _mm512_set_epi64(3,2,1,0,3,2,1,0);
  const __m512i idx2 = _mm512_set_epi64(13,9,12,8,5,1,4,0);
  const __m512i idx3 = _mm512_set_epi64(15,11,14,10,7,3,6,2);
  const __mmask8 khigh = 16+32+64+128;
  for (int i = 0; i < pn; i++) {
    const int np = number_of_partners[i];
    const int kp = pointer[i];
    v8df vzi = _mm512_loadu_pd((double*)(z+i));
    v8df vqi = _mm512_permutexvar_pd(idx_0123, vzi);;
    v8df vpi = _mm512_setzero_pd();
    for (int k = 0; k < (np/8*8); k+=8) {
      const int j_1 = sorted_list[kp + k];
      const int j_2 = sorted_list[kp + k + 1];
      const int j_3 = sorted_list[kp + k + 2];
      const int j_4 = sorted_list[kp + k + 3];
      const int j_5 = sorted_list[kp + k + 4];
      const int j_6 = sorted_list[kp + k + 5];
      const int j_7 = sorted_list[kp + k + 6];
      const int j_8 = sorted_list[kp + k + 7];
      v8df vzj_1 = _mm512_loadu_pd((double*)(z+j_1));
      v8df vzj_2 = _mm512_loadu_pd((double*)(z+j_2));
      v8df vqj_12= _mm512_permutex2var_pd(vzj_1, idx_q, vzj_2);
      v8df vpj_12= _mm512_permutex2var_pd(vzj_1, idx_p, vzj_2);
      v8df vdq_12 = vqj_12 - vqi;

      v8df vzj_3 = _mm512_loadu_pd((double*)(z+j_3));
      v8df vzj_4 = _mm512_loadu_pd((double*)(z+j_4));
      v8df vqj_34= _mm512_permutex2var_pd(vzj_3, idx_q, vzj_4);
      v8df vpj_34= _mm512_permutex2var_pd(vzj_3, idx_p, vzj_4);
      v8df vdq_34 = vqj_34 - vqi;

      v8df vzj_5 = _mm512_loadu_pd((double*)(z+j_5));
      v8df vzj_6 = _mm512_loadu_pd((double*)(z+j_6));
      v8df vqj_56= _mm512_permutex2var_pd(vzj_5, idx_q, vzj_6);
      v8df vpj_56= _mm512_permutex2var_pd(vzj_5, idx_p, vzj_6);
      v8df vdq_56 = vqj_56 - vqi;

      v8df vzj_7 = _mm512_loadu_pd((double*)(z+j_7));
      v8df vzj_8 = _mm512_loadu_pd((double*)(z+j_8));
      v8df vqj_78= _mm512_permutex2var_pd(vzj_7, idx_q, vzj_8);
      v8df vpj_78= _mm512_permutex2var_pd(vzj_7, idx_p, vzj_8);
      v8df vdq_78 = vqj_78 - vqi;

      v8df tmp0 = _mm512_unpacklo_pd(vdq_12, vdq_34);
      v8df tmp1 = _mm512_unpackhi_pd(vdq_12, vdq_34);
      v8df tmp2 = _mm512_unpacklo_pd(vdq_56, vdq_78);
      v8df tmp3 = _mm512_unpackhi_pd(vdq_56, vdq_78);

      v8df vdx = _mm512_permutex2var_pd(tmp0, idx2, tmp2);
      v8df vdy = _mm512_permutex2var_pd(tmp1, idx2, tmp3);
      v8df vdz = _mm512_permutex2var_pd(tmp0, idx3, tmp2);

      v8df vr2 = vdx*vdx + vdy*vdy + vdz*vdz;
      v8df vr6 = vr2 * vr2 * vr2;
      v8df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      __mmask8 kcmp = _mm512_cmp_pd_mask(vr2,vcl2, _CMP_GT_OS);
      vdf = _mm512_mask_blend_pd(kcmp, vdf, vzero);
      v8df vdf_12 = _mm512_permutexvar_pd(idx_f12, vdf);
      v8df vdf_34 = _mm512_permutexvar_pd(idx_f34, vdf);
      v8df vdf_56 = _mm512_permutexvar_pd(idx_f56, vdf);
      v8df vdf_78 = _mm512_permutexvar_pd(idx_f78, vdf);
      vpj_12 -= vdf_12 * vdq_12;
      vpj_34 -= vdf_34 * vdq_34;
      vpj_56 -= vdf_56 * vdq_56;
      vpj_78 -= vdf_78 * vdq_78;
      vpi += vdf_12 * vdq_12;
      vpi += vdf_34 * vdq_34;
      vpi += vdf_56 * vdq_56;
      vpi += vdf_78 * vdq_78;
      v8df vpj_11 = _mm512_permutexvar_pd(idx_0123, vpj_12);
      v8df vpj_33 = _mm512_permutexvar_pd(idx_0123, vpj_34);
      v8df vpj_55 = _mm512_permutexvar_pd(idx_0123, vpj_56);
      v8df vpj_77 = _mm512_permutexvar_pd(idx_0123, vpj_78);
      _mm512_mask_store_pd((double*)(z+j_1), khigh, vpj_11);
      _mm512_mask_store_pd((double*)(z+j_3), khigh, vpj_33);
      _mm512_mask_store_pd((double*)(z+j_5), khigh, vpj_55);
      _mm512_mask_store_pd((double*)(z+j_7), khigh, vpj_77);
      _mm512_mask_store_pd((double*)(z+j_2), khigh, vpj_12);
      _mm512_mask_store_pd((double*)(z+j_4), khigh, vpj_34);
      _mm512_mask_store_pd((double*)(z+j_6), khigh, vpj_56);
      _mm512_mask_store_pd((double*)(z+j_8), khigh, vpj_78);
    }
    v4df vpi_low = _mm512_extractf64x4_pd(vpi, 0);
    v4df vpi_high = _mm512_extractf64x4_pd(vpi, 1);
    v4df vdpi = vpi_low + vpi_high;
    v8df vzdpi = _mm512_insertf64x4(vzero, vdpi, 1); 
    vzi += vzdpi;
    _mm512_store_pd((double*)(z+i), vzi);
    const double qix = z[i][X];
    const double qiy = z[i][Y];
    const double qiz = z[i][Z];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    for (int k = (np/8*8); k < np; k++) {
      const int j = sorted_list[kp + k];
      double dx = z[j][X] - qix;
      double dy = z[j][Y] - qiy;
      double dz = z[j][Z] - qiz;
      double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df=0.0;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
      z[j][PX] -= df * dx;
      z[j][PY] -= df * dy;
      z[j][PZ] -= df * dz;
    }
    z[i][PX] += pfx;
    z[i][PY] += pfy;
    z[i][PZ] += pfz;
  }
}
//----------------------------------------------------------------------
void
force_sorted_z_intrin_swp(void) {
  const int pn = particle_number;
  const v8df vzero = _mm512_setzero_pd();
  const v8df vcl2 = _mm512_set1_pd(CL2);
  const v8df vc24 = _mm512_set1_pd(24.0*dt);
  const v8df vc48 = _mm512_set1_pd(48.0*dt);
  const __m512i idx = _mm512_set_epi64(3,2,1,0,7,6,5,4);
  const __m512i idx_q = _mm512_set_epi64(11,10,9,8,3,2,1,0);
  const __m512i idx_p = _mm512_set_epi64(15,14,13,12,7,6,5,4);
  const __m512i idx_f12 = _mm512_set_epi64(1,1,1,1,0,0,0,0);
  const __m512i idx_f34 = _mm512_set_epi64(3,3,3,3,2,2,2,2);
  const __m512i idx_f56 = _mm512_set_epi64(5,5,5,5,4,4,4,4);
  const __m512i idx_f78 = _mm512_set_epi64(7,7,7,7,6,6,6,6);
  const __m512i idx_0123 = _mm512_set_epi64(3,2,1,0,3,2,1,0);
  const __m512i idx2 = _mm512_set_epi64(13,9,12,8,5,1,4,0);
  const __m512i idx3 = _mm512_set_epi64(15,11,14,10,7,3,6,2);
  const __mmask8 khigh = 16+32+64+128;
  for (int i = 0; i < pn; i++) {
    const int np = number_of_partners[i];
    const int kp = pointer[i];
    v8df vzi = _mm512_loadu_pd((double*)(z+i));
    v8df vqi = _mm512_permutexvar_pd(idx_0123, vzi);;
    v8df vpi = _mm512_setzero_pd();
    int ja_1 = sorted_list[kp];
    int ja_2 = sorted_list[kp + 1];
    int ja_3 = sorted_list[kp + 2];
    int ja_4 = sorted_list[kp + 3];
    int ja_5 = sorted_list[kp + 4];
    int ja_6 = sorted_list[kp + 5];
    int ja_7 = sorted_list[kp + 6];
    int ja_8 = sorted_list[kp + 7];
    v8df vzja_1 = _mm512_loadu_pd((double*)(z+ja_1));
    v8df vzja_2 = _mm512_loadu_pd((double*)(z+ja_2));
    v8df vzja_3 = _mm512_loadu_pd((double*)(z+ja_3));
    v8df vzja_4 = _mm512_loadu_pd((double*)(z+ja_4));
    v8df vzja_5 = _mm512_loadu_pd((double*)(z+ja_5));
    v8df vzja_6 = _mm512_loadu_pd((double*)(z+ja_6));
    v8df vzja_7 = _mm512_loadu_pd((double*)(z+ja_7));
    v8df vzja_8 = _mm512_loadu_pd((double*)(z+ja_8));

    v8df vqj_12= _mm512_permutex2var_pd(vzja_1, idx_q, vzja_2);
    v8df vpja_12= _mm512_permutex2var_pd(vzja_1, idx_p, vzja_2);
    v8df vdqa_12 = vqj_12 - vqi;
    v8df vqj_34= _mm512_permutex2var_pd(vzja_3, idx_q, vzja_4);
    v8df vpja_34= _mm512_permutex2var_pd(vzja_3, idx_p, vzja_4);
    v8df vdqa_34 = vqj_34 - vqi;
    v8df vqj_56= _mm512_permutex2var_pd(vzja_5, idx_q, vzja_6);
    v8df vpja_56= _mm512_permutex2var_pd(vzja_5, idx_p, vzja_6);
    v8df vdqa_56 = vqj_56 - vqi;
    v8df vqj_78= _mm512_permutex2var_pd(vzja_7, idx_q, vzja_8);
    v8df vpja_78= _mm512_permutex2var_pd(vzja_7, idx_p, vzja_8);
    v8df vdqa_78 = vqj_78 - vqi;


    for (int k = 0; k < (np/8*8); k+=8) {
      const int j_1 = ja_1;
      const int j_2 = ja_2;
      const int j_3 = ja_3;
      const int j_4 = ja_4;
      const int j_5 = ja_5;
      const int j_6 = ja_6;
      const int j_7 = ja_7;
      const int j_8 = ja_8;
      v8df vzj_1 = vzja_1;
      v8df vzj_2 = vzja_2;
      v8df vzj_3 = vzja_3;
      v8df vzj_4 = vzja_4;
      v8df vzj_5 = vzja_5;
      v8df vzj_6 = vzja_6;
      v8df vzj_7 = vzja_7;
      v8df vzj_8 = vzja_8;
      v8df vpj_12 = vpja_12;
      v8df vpj_34 = vpja_34;
      v8df vpj_56 = vpja_56;
      v8df vpj_78 = vpja_78;
      v8df vdq_12 = vdqa_12;
      v8df vdq_34 = vdqa_34;
      v8df vdq_56 = vdqa_56;
      v8df vdq_78 = vdqa_78;

      ja_1 = sorted_list[kp + k + 8];
      ja_2 = sorted_list[kp + k + 1 + 8];
      ja_3 = sorted_list[kp + k + 2 + 8];
      ja_4 = sorted_list[kp + k + 3 + 8];
      ja_5 = sorted_list[kp + k + 4 + 8];
      ja_6 = sorted_list[kp + k + 5 + 8];
      ja_7 = sorted_list[kp + k + 6 + 8];
      ja_8 = sorted_list[kp + k + 7 + 8];


      v8df tmp0 = _mm512_unpacklo_pd(vdq_12, vdq_34);
      v8df tmp1 = _mm512_unpackhi_pd(vdq_12, vdq_34);
      v8df tmp2 = _mm512_unpacklo_pd(vdq_56, vdq_78);
      v8df tmp3 = _mm512_unpackhi_pd(vdq_56, vdq_78);

      v8df vdx = _mm512_permutex2var_pd(tmp0, idx2, tmp2);
      v8df vdy = _mm512_permutex2var_pd(tmp1, idx2, tmp3);
      v8df vdz = _mm512_permutex2var_pd(tmp0, idx3, tmp2);

      v8df vr2 = vdx*vdx + vdy*vdy + vdz*vdz;
      v8df vr6 = vr2 * vr2 * vr2;
      v8df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
      vzja_1 = _mm512_loadu_pd((double*)(z+ja_1));
      vzja_2 = _mm512_loadu_pd((double*)(z+ja_2));
      vzja_3 = _mm512_loadu_pd((double*)(z+ja_3));
      vzja_4 = _mm512_loadu_pd((double*)(z+ja_4));
      vzja_5 = _mm512_loadu_pd((double*)(z+ja_5));
      vzja_6 = _mm512_loadu_pd((double*)(z+ja_6));
      vzja_7 = _mm512_loadu_pd((double*)(z+ja_7));
      vzja_8 = _mm512_loadu_pd((double*)(z+ja_8));

      v8df vqja_12= _mm512_permutex2var_pd(vzja_1, idx_q, vzja_2);
      vpja_12= _mm512_permutex2var_pd(vzja_1, idx_p, vzja_2);
      vdqa_12 = vqja_12 - vqi;
      v8df vqj_34= _mm512_permutex2var_pd(vzja_3, idx_q, vzja_4);
      vpja_34= _mm512_permutex2var_pd(vzja_3, idx_p, vzja_4);
      vdqa_34 = vqj_34 - vqi;
      v8df vqj_56= _mm512_permutex2var_pd(vzja_5, idx_q, vzja_6);
      vpja_56= _mm512_permutex2var_pd(vzja_5, idx_p, vzja_6);
      vdqa_56 = vqj_56 - vqi;
      v8df vqj_78= _mm512_permutex2var_pd(vzja_7, idx_q, vzja_8);
      vpja_78= _mm512_permutex2var_pd(vzja_7, idx_p, vzja_8);
      vdqa_78 = vqj_78 - vqi;

      __mmask8 kcmp = _mm512_cmp_pd_mask(vr2,vcl2, _CMP_GT_OS);
      vdf = _mm512_mask_blend_pd(kcmp, vdf, vzero);
      v8df vdf_12 = _mm512_permutexvar_pd(idx_f12, vdf);
      v8df vdf_34 = _mm512_permutexvar_pd(idx_f34, vdf);
      v8df vdf_56 = _mm512_permutexvar_pd(idx_f56, vdf);
      v8df vdf_78 = _mm512_permutexvar_pd(idx_f78, vdf);
      vpj_12 -= vdf_12 * vdq_12;
      vpj_34 -= vdf_34 * vdq_34;
      vpj_56 -= vdf_56 * vdq_56;
      vpj_78 -= vdf_78 * vdq_78;
      vpi += vdf_12 * vdq_12;
      vpi += vdf_34 * vdq_34;
      vpi += vdf_56 * vdq_56;
      vpi += vdf_78 * vdq_78;
      v8df vpj_11 = _mm512_permutexvar_pd(idx_0123, vpj_12);
      v8df vpj_33 = _mm512_permutexvar_pd(idx_0123, vpj_34);
      v8df vpj_55 = _mm512_permutexvar_pd(idx_0123, vpj_56);
      v8df vpj_77 = _mm512_permutexvar_pd(idx_0123, vpj_78);
      _mm512_mask_store_pd((double*)(z+j_1), khigh, vpj_11);
      _mm512_mask_store_pd((double*)(z+j_3), khigh, vpj_33);
      _mm512_mask_store_pd((double*)(z+j_5), khigh, vpj_55);
      _mm512_mask_store_pd((double*)(z+j_7), khigh, vpj_77);
      _mm512_mask_store_pd((double*)(z+j_2), khigh, vpj_12);
      _mm512_mask_store_pd((double*)(z+j_4), khigh, vpj_34);
      _mm512_mask_store_pd((double*)(z+j_6), khigh, vpj_56);
      _mm512_mask_store_pd((double*)(z+j_8), khigh, vpj_78);
    }
    v4df vpi_low = _mm512_extractf64x4_pd(vpi, 0);
    v4df vpi_high = _mm512_extractf64x4_pd(vpi, 1);
    v4df vdpi = vpi_low + vpi_high;
    v8df vzdpi = _mm512_insertf64x4(vzero, vdpi, 1);
    vzi += vzdpi;
    _mm512_store_pd((double*)(z+i), vzi);
    const double qix = z[i][X];
    const double qiy = z[i][Y];
    const double qiz = z[i][Z];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    for (int k = (np/8*8); k < np; k++) {
      const int j = sorted_list[kp + k];
      double dx = z[j][X] - qix;
      double dy = z[j][Y] - qiy;
      double dz = z[j][Z] - qiz;
      double r2 = (dx * dx + dy * dy + dz * dz);
      double r6 = r2 * r2 * r2;
      double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      if (r2 > CL2) df=0.0;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
      z[j][PX] -= df * dx;
      z[j][PY] -= df * dy;
      z[j][PZ] -= df * dz;
    }
    z[i][PX] += pfx;
    z[i][PY] += pfy;
    z[i][PZ] += pfz;
  }
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
  measure(&force_intrin_v1_reactless, "with gather, reactless");
  print_result();
#elif defined INTRIN_v2 && defined REACTLESS
  measure(&force_intrin_v2_reactless, "without gather, reactless");
  print_result();
#elif defined INTRIN_v3 && defined REACTLESS
  measure(&force_intrin_v3_reactless, "with gather, reactless, remaining loop opt");
  print_result();
#elif INTRIN_v1
  measure(&force_intrin_v1, "with scatter & gather");
  print_result();
#elif INTRIN_v2
  measure(&force_intrin_v2, "with scatter & gather, remaining loop opt");
  print_result();
#elif INTRIN_v3
  measure(&force_intrin_v3, "with scatter & gather, remaining loop opt, swp");
  print_result();
#elif INTRIN_v4
  measure(&force_intrin_v4, "without scatter & gather");
  print_result();
#elif INTRIN_v5
  measure(&force_intrin_v5, "without scatter & gather, swp");
  print_result();
#elif INTRIN_v6
  copy_to_z();
  measure(&force_intrin_v6, "aos 8 bytes, gather and scatter, swp");
  copy_from_z();
  print_result();
#elif INTRIN_v7
  measure(&force_intrin_v7, "scatter & gather, remaining loop opt, swp, PF");
  print_result();
#elif INTRIN_v8
  measure(&force_intrin_v8, "scatter & gather, remaining loop opt, swp, PF2");
  print_result();
#elif INTRIN_v9
  copy_to_z();
  measure(&force_intrin_v9, "aos 8 bytes, gather and scatter, swp, PF");
  copy_from_z();
  print_result();
#elif AOS_8
  copy_to_z();
  measure(&force_sorted_z_intrin, "aos 8 bytes, intrin");
  copy_from_z();
  print_result();
#elif AOS_8_SWP
  copy_to_z();
  measure(&force_sorted_z_intrin_swp, "aos 8 bytes, intrin, swp");
  copy_from_z();
  print_result();
#else
  measure(&force_pair, "pair");
  measure(&force_sorted, "sorted");
  measure(&force_next, "next");
  measure(&force_intrin_v1, "with scatter & gather");
  measure(&force_intrin_v2, "with scatter & gather, remaining loop opt");
  measure(&force_intrin_v3, "with scatter & gather, remaining loop opt, swp");
  measure(&force_intrin_v4, "without scatter & gather");
  measure(&force_intrin_v5, "without scatter & gather, swp");
  copy_to_z();
  measure(&force_sorted_z_intrin, "aos 8 bytes, intrin");
  copy_from_z();
  copy_to_z();
  measure(&force_sorted_z_intrin_swp, "aos 8 bytes, intrin, swp");
  copy_from_z();
  copy_to_z();
  measure(&force_intrin_v6, "aos 8 bytes, gather and scatter, swp");
  copy_from_z();
  measure(&force_intrin_v7, "scatter & gather, remaining loop opt, swp, PF");
  measure(&force_intrin_v8, "scatter & gather, remaining loop opt, swp, PF2");
  copy_to_z();
  measure(&force_intrin_v9, "aos 8 bytes, gather and scatter, swp, PF");
  copy_from_z();
#endif
  deallocate();
}
//----------------------------------------------------------------------
