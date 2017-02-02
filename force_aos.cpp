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
#include <omp.h>
//----------------------------------------------------------------------
// const double density = 1.0;
const double density = 0.5;
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
first_touch(void) {
  int est_N = 0, est_pair_num = 0;

  if (density == 1.0) {
    est_N = 119164;
#ifdef REACTLESS
    est_pair_num = 2 * 7839886;
#else
    est_pair_num = 7839886;
#endif
  } else if (density == 0.5) {
    est_N = 62500;
#ifdef REACTLESS
    est_pair_num = 2 * 4536276;
#else
    est_pair_num = 4536276;
#endif
  } else {
    est_N = N;
    est_pair_num = MAX_PAIRS;
  }

#pragma omp parallel for
  for (int i = 0; i < est_N; i++) {
    q[i].x = q[i].y = q[i].z = q[i].w = 0.0;
    p[i].x = p[i].y = p[i].z = p[i].w = 0.0;
    number_of_partners[i] = pointer[i] = 0;
  }

#pragma omp parallel for
  for (int i = 0; i < est_pair_num; i++) {
    sorted_list[i] = 0;
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

#ifdef _OPENMP
  first_touch();
#endif

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
#pragma omp parallel
  {
    const auto nps = number_of_pairs;
#pragma omp for nowait
#pragma novector
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
#pragma omp atomic
      p[i].x += df * dx;
#pragma omp atomic
      p[i].y += df * dy;
#pragma omp atomic
      p[i].z += df * dz;
#pragma omp atomic
      p[j].x -= df * dx;
#pragma omp atomic
      p[j].y -= df * dy;
#pragma omp atomic
      p[j].z -= df * dz;
    }
  }
}
//----------------------------------------------------------------------
void
force_sorted(void) {
#pragma omp parallel
  {
    const auto pn = particle_number;
#pragma omp for nowait
    for (int i = 0; i < pn; i++) {
      const auto qx_key = q[i].x;
      const auto qy_key = q[i].y;
      const auto qz_key = q[i].z;
      const auto np = number_of_partners[i];
      double pfx = 0, pfy = 0, pfz = 0;
      const auto kp = pointer[i];
#pragma novector
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
#pragma omp atomic
        p[j].x -= df*dx;
#pragma omp atomic
        p[j].y -= df*dy;
#pragma omp atomic
        p[j].z -= df*dz;
      } // end of k loop
#pragma omp atomic
      p[i].x += pfx;
#pragma omp atomic
      p[i].y += pfy;
#pragma omp atomic
      p[i].z += pfz;
    } // end of i loop
  } // end of pragma omp for
}
//----------------------------------------------------------------------
void
force_sorted_reactless(void) {
#pragma omp parallel
  {
    const auto pn = particle_number;
#pragma omp for nowait
    for (int i = 0; i < pn; i++) {
      const auto qx_key = q[i].x;
      const auto qy_key = q[i].y;
      const auto qz_key = q[i].z;
      const auto np = number_of_partners[i];
      auto pfx = p[i].x, pfy = p[i].y, pfz = p[i].z;
      const auto kp = pointer[i];
#pragma novector
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
      } // end of k loop
      p[i].x = pfx;
      p[i].y = pfy;
      p[i].z = pfz;
    } // end of i loop
  } // end of pragma omp parallel
}
//----------------------------------------------------------------------
// enum {
//   NUM_SMT = 4,
// };
//----------------------------------------------------------------------
/*void
force_sorted_ht(void) {
#pragma omp parallel
  {
    const auto tid       = omp_get_thread_num();
    const auto n_threads = omp_get_num_threads();

    const auto i_ptcl_id = tid / NUM_SMT;
    const auto j_ptcl_id = tid % NUM_SMT;

    const auto n_ptcl_in_th = (particle_number - 1) / (n_threads / NUM_SMT) + 1;
    const auto beg_i = i_ptcl_id * n_ptcl_in_th;
    auto end_i = (i_ptcl_id + 1) * n_ptcl_in_th;
    if (end_i > particle_number) end_i = particle_number;

    for (int i = beg_i; i < end_i; i++) {
      const auto qx_key = q[i].x, qy_key = q[i].y, qz_key = q[i].z;
      const auto np = number_of_partners[i];
      double pfx = 0, pfy = 0, pfz = 0;
      const auto kp = pointer[i];
      int k = 0;
#pragma novector
      for (; k < (np / NUM_SMT) * NUM_SMT; k += NUM_SMT) {
        const auto j = sorted_list[kp + k + j_ptcl_id];
        const auto dx = q[j].x - qx_key;
        const auto dy = q[j].y - qy_key;
        const auto dz = q[j].z - qz_key;
        const auto r2 = (dx*dx + dy*dy + dz*dz);
        if (r2 <= CL2) {
          const auto r6 = r2*r2*r2;
          const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
          pfx += df*dx;
          pfy += df*dy;
          pfz += df*dz;
#pragma omp atomic
          p[j].x -= df*dx;
#pragma omp atomic
          p[j].y -= df*dy;
#pragma omp atomic
          p[j].z -= df*dz;
        }
      } // end of k loop

      // remaining loop
      if (j_ptcl_id < (np % NUM_SMT)) {
        const auto j = sorted_list[kp + k + j_ptcl_id];
        const auto dx = q[j].x - qx_key;
        const auto dy = q[j].y - qy_key;
        const auto dz = q[j].z - qz_key;
        const auto r2 = (dx*dx + dy*dy + dz*dz);
        if (r2 <= CL2) {
          const auto r6 = r2*r2*r2;
          const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
          pfx += df*dx;
          pfy += df*dy;
          pfz += df*dz;
#pragma omp atomic
          p[j].x -= df*dx;
#pragma omp atomic
          p[j].y -= df*dy;
#pragma omp atomic
          p[j].z -= df*dz;
        }
      } // end of remaining loop

#pragma omp atomic
      p[i].x += pfx;
#pragma omp atomic
      p[i].y += pfy;
#pragma omp atomic
      p[i].z += pfz;
    } // end of i loop
  } // end of pragma omp parallel
  }*/
//----------------------------------------------------------------------
void
force_next(void) {
#pragma omp parallel
  {
    const int pn = particle_number;
#pragma omp for nowait
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
#pragma omp atomic
        p[jb].x -= df * dxb;
#pragma omp atomic
        p[jb].y -= df * dyb;
#pragma omp atomic
        p[jb].z -= df * dzb;
        const double r6 = r2 * r2 * r2;
        df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
        jb = j;
        dxb = dx;
        dyb = dy;
        dzb = dz;
      }
#pragma omp atomic
      p[jb].x -= df * dxb;
#pragma omp atomic
      p[jb].y -= df * dyb;
#pragma omp atomic
      p[jb].z -= df * dzb;
#pragma omp atomic
      p[i].x += pfx + df * dxb;
#pragma omp atomic
      p[i].y += pfy + df * dyb;
#pragma omp atomic
      p[i].z += pfz + df * dzb;
    }
  }
}
//----------------------------------------------------------------------
static inline __m512d
_mm512_load2_m256d(const double* hiaddr,
                   const double* loaddr) {
  __m512d ret;
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(loaddr), 0x0);
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
void
force_intrin_v1(void) {
#pragma omp parallel
  {
    const auto vc24  = _mm512_set1_pd(24.0 * dt);
    const auto vc48  = _mm512_set1_pd(48.0 * dt);
    const auto vcl2  = _mm512_set1_pd(CL2);
    const auto vzero = _mm512_setzero_pd();
    const auto pn = particle_number;

#pragma omp for nowait
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
#if 0
      // horizontal sum
      auto vpwi = _mm512_setzero_pd();
      transpose_4x4x2(vpxi, vpyi, vpzi, vpwi);
      // vpxi = {vpia, vpie}
      // vpyi = {vpib, vpif}
      // vpzi = {vpic, vpig}
      // vpwi = {vpid, vpih}

      auto vpi_hilo = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vpxi,
                                                                vpyi),
                                                  vpzi),
                                    vpwi);
      auto vpi_lohi = _mm512_castsi512_pd(
                                          _mm512_alignr_epi64(_mm512_castpd_si512(vpi_hilo),
                                                              _mm512_castpd_si512(vpi_hilo),
                                                              4));
      auto vpi = _mm512_castpd512_pd256(_mm512_add_pd(vpi_hilo, vpi_lohi));

      // store
      vpi = _mm256_add_pd(vpi, _mm256_load_pd(&p[i].x));
      _mm256_storeu_pd(&p[i].x, vpi);

      auto pfx = p[i].x, pfy = p[i].y, pfz = p[i].z;
#else
      auto pfx = p[i].x + _mm512_reduce_add_pd(vpxi);
      auto pfy = p[i].y + _mm512_reduce_add_pd(vpyi);
      auto pfz = p[i].z + _mm512_reduce_add_pd(vpzi);
#endif

#pragma novector
      for (int k = (np / 8) * 8; k < np; k++) {
        const auto j = sorted_list[kp + k];
        const auto dx = q[j].x - q[i].x;
        const auto dy = q[j].y - q[i].y;
        const auto dz = q[j].z - q[i].z;
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
  } // end of pragma omp parallel
}
//----------------------------------------------------------------------
void
force_intrin_v2(void) {
#pragma omp parallel
  {
    const auto vc24  = _mm512_set1_pd(24.0 * dt);
    const auto vc48  = _mm512_set1_pd(48.0 * dt);
    const auto vcl2  = _mm512_set1_pd(CL2);
    const auto vzero = _mm512_setzero_pd();
    const auto pn     = particle_number;

#pragma omp for nowait
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
  } // end of pragma omp parallel
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
#elif defined SORTED && defined REACTLESS
  measure(&force_sorted_reactless, "sorted_reactless");
  print_result();
#elif SORTED
  measure(&force_sorted, "sorted");
  print_result();
#elif SORTED_HT
  // measure(&force_sorted_ht, "sorted_ht");
  // print_result();
#elif defined INTRIN_v1 && defined REACTLESS
  measure(&force_intrin_v1, "intrin_v1");
  print_result();
#elif defined INTRIN_v2 && defined REACTLESS
  measure(&force_intrin_v2, "intrin_v2");
  print_result();
#endif
  deallocate();
}
//----------------------------------------------------------------------
